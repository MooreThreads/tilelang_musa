/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \brief Replace copy from global to shared with async copy
 * \file inject_ptx_async_copy.cc
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"
#include "storage_access.h"
#include "tir/ir/buffer_common.h"
#include "tvm/tir/stmt.h"

namespace tvm {
namespace tl {

using namespace tir;

class PTXAsyncCopyInjector : public StmtMutator {
public:
  using StmtMutator::VisitStmt_;

  Stmt VisitStmt_(const AttrStmtNode *attr) {
    if (attr->attr_key == tl::attr::kSourceRobustDesc) {
      PrimExpr prev_desc = current_src_robust_desc_;
      current_src_robust_desc_ = attr->value;
      auto body = this->VisitStmt(attr->body);
      current_src_robust_desc_ = prev_desc;
      return AttrStmt(attr->node, attr->attr_key, attr->value, body);
    }
    if (attr->attr_key == tir::attr::async_scope) {
      bool in_async_prev = in_async;
      in_async = true;
      auto body = this->VisitStmt(attr->body);
      in_async = in_async_prev;
      return body;
    }
    if (attr->attr_key == tl::attr::kForceAsyncCopy) {
      bool in_async_prev = in_async;
      bool in_force_prev = in_force_async_copy_;
      in_async = true;
      in_force_async_copy_ = true;
      auto body = this->VisitStmt(attr->body);
      in_async = in_async_prev;
      in_force_async_copy_ = in_force_prev;
      if (!in_async_prev) {
        bool has_cp_async = false;
        PostOrderVisit(body, [&](const ObjectRef &obj) {
          if (const auto *call = obj.as<CallNode>()) {
            if (call->op.same_as(builtin::ptx_cp_async()) ||
                call->op.same_as(tl::musa_cp_async_robust())) {
              has_cp_async = true;
            }
          }
        });
        if (has_cp_async) {
          auto wait_cnt = IntImm(DataType::Int(32), 0);
          auto wait_stmt = Evaluate(
              Call(DataType::Void(), builtin::ptx_wait_group(), {wait_cnt}));
          body = SeqStmt(Array<Stmt>{body, wait_stmt});
        }
      }
      return AttrStmt(attr->node, attr->attr_key, attr->value, body);
    }
    return StmtMutator::VisitStmt_(attr);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    auto stmt = StmtMutator::VisitStmt_(op);
    const auto *loop = stmt.as<ForNode>();
    if (!loop || !is_zero(loop->min)) {
      return stmt;
    }

    const int64_t *extent_ptr = as_const_int(loop->extent);
    if (extent_ptr == nullptr || *extent_ptr <= 1) {
      return stmt;
    }

    LoopBodyInfo body_info;
    if (!MatchLoopBodyInfo(loop->body, &body_info)) {
      return stmt;
    }
    bool force_async_copy =
        in_force_async_copy_ || body_info.has_local_force_async;
    if (!force_async_copy) {
      return stmt;
    }
    const auto *store = body_info.store;
    if (!(store->buffer.scope() == "shared" ||
          store->buffer.scope() == "shared.dyn")) {
      return stmt;
    }
    const auto *load = body_info.load;
    bool predicated = body_info.predicated;
    PrimExpr predicate_value = body_info.predicate_value;
    if (load->buffer.scope() != "global") {
      return stmt;
    }
    if (store->indices.size() != 1 || load->indices.size() != 1) {
      return stmt;
    }
    if (store->indices[0].dtype().lanes() != 1 ||
        load->indices[0].dtype().lanes() != 1) {
      return stmt;
    }

    Optional<PrimExpr> dst_base =
        GetUnitStrideBase(store->indices[0], loop->loop_var);
    Optional<PrimExpr> src_base =
        GetUnitStrideBase(load->indices[0], loop->loop_var);
    if (!dst_base.defined() || !src_base.defined()) {
      return stmt;
    }

    int bytes = (*extent_ptr) * load->buffer->dtype.bytes();
    if (!(bytes == 4 || bytes == 8 || bytes == 16)) {
      return stmt;
    }

    auto dst_elem_type = GetPointerType(store->buffer->data->type_annotation);
    auto src_elem_type = GetPointerType(load->buffer->data->type_annotation);
    ICHECK(dst_elem_type.has_value() && src_elem_type.has_value())
        << "Both store and load buffer should have a pointer type annotation.";
    int index_factor = 1;
    if (dst_elem_type.value() != src_elem_type.value()) {
      ICHECK(dst_elem_type.value() == DataType::UInt(8));
      index_factor = src_elem_type->bytes();
    }

    auto cp_async = MakeCPAsync(load, store, dst_base.value(), src_base.value(),
                                bytes, predicated, predicate_value,
                                index_factor, body_info.local_src_robust_desc);
    if (!cp_async.defined()) {
      return stmt;
    }
    Stmt result = cp_async.value();
    if (body_info.has_local_force_async && !in_force_async_copy_) {
      result = SeqStmt(Array<Stmt>{result, MakeWaitGroupStmt()});
    }
    return result;
  }

  Stmt InjectPTX(const BufferLoadNode *load, const BufferStoreNode *store,
                 bool predicated = false,
                 const PrimExpr &predicate_value = PrimExpr()) {
    if (load->buffer.scope() == "global") {
      ICHECK(load->indices.size() == 1 && store->indices.size() == 1);
      ICHECK(load->indices[0]->dtype.lanes() ==
             store->indices[0]->dtype.lanes())
          << load->indices[0] << " vs. " << store->indices[0] << " with lanes "
          << load->indices[0]->dtype.lanes() << " vs. "
          << store->indices[0]->dtype.lanes();

      const int indices_lanes = load->indices[0]->dtype.lanes();
      const int bytes = indices_lanes * load->buffer->dtype.bytes();

      if (bytes == 4 || bytes == 8 || bytes == 16) {
        auto dst_elem_type =
            GetPointerType(store->buffer->data->type_annotation);
        auto src_elem_type =
            GetPointerType(load->buffer->data->type_annotation);
        ICHECK(dst_elem_type.has_value() && src_elem_type.has_value())
            << "Both store and load buffer should have a pointer type "
               "annotation.";

        int index_factor = 1;
        if (dst_elem_type.value() != src_elem_type.value()) {
          // The only case where src and dst have different dtypes is when the
          // dst shared memory is a byte buffer generated by merging dynamic
          // shared memory.
          ICHECK(store->buffer.scope() == "shared.dyn" ||
                 store->buffer.scope() == "shared");
          ICHECK(dst_elem_type.value() == DataType::UInt(8));
          // BufferStore/Load have the "pointer reinterpret" semantics according
          // to their "value" dtype. Their "indices" are supposed to be applied
          // after such pointer cast, for example:
          // ((*float16)(byte_buffer))[buffer->indices] = fp16_value; To replace
          // BufferStore/Load with cp.async, we need to multiply the store index
          // by the byte size of the "value" dtype, to get the correct offset
          // into the byte buffer.
          index_factor = src_elem_type->bytes();
        }

        if (indices_lanes == 1) {
          auto src_offset = load->indices[0];
          auto dst_offset = store->indices[0];
          auto call = MakeCPAsync(load, store, dst_offset, src_offset, bytes,
                                  predicated, predicate_value, index_factor);
          if (call.defined()) {
            return call.value();
          }
        }

        // Predicated load don't support vectorized indexing.
        if (!predicated) {
          // Only some vectorized indexing patterns are supported for now.
          auto src_offset = [=]() -> PrimExpr {
            if (load->indices[0]->IsInstance<RampNode>()) {
              return load->indices[0].as<RampNode>()->base;
            }
            return PrimExpr();
          }();

          auto dst_offset = [=]() -> PrimExpr {
            if (store->indices[0].as<RampNode>()) {
              return store->indices[0].as<RampNode>()->base;
            } else if (store->indices[0].as<AddNode>()) {
              // The case where the dst buffer is a byte buffer generated by
              // merging dynamic shared memory. A_shared.dyn[(ramp(...), 1, 8) +
              // x8(17408))] = A_global[ramp(...),1, 8)]
              auto *add = store->indices[0].as<AddNode>();
              if (!add->a->IsInstance<RampNode>())
                return PrimExpr();
              if (!add->b->IsInstance<BroadcastNode>())
                return PrimExpr();
              return tir::Add(add->a.as<RampNode>()->base,
                              add->b.as<BroadcastNode>()->value);
            }
            return PrimExpr();
          }();
          if (src_offset.defined() && dst_offset.defined()) {
            auto call = MakeCPAsync(load, store, dst_offset, src_offset, bytes,
                                    false, PrimExpr(), index_factor);
            if (call.defined()) {
              return call.value();
            }
          }
        } else {
          // Only some vectorized indexing patterns are supported for now.
          auto src_offset = [=]() -> PrimExpr {
            if (load->indices[0]->IsInstance<RampNode>()) {
              return load->indices[0].as<RampNode>()->base;
            }
            return PrimExpr();
          }();

          auto dst_offset = [=]() -> PrimExpr {
            if (store->indices[0].as<RampNode>()) {
              return store->indices[0].as<RampNode>()->base;
            } else if (store->indices[0].as<AddNode>()) {
              // The case where the dst buffer is a byte buffer generated by
              // merging dynamic shared memory. A_shared.dyn[(ramp(...), 1, 8) +
              // x8(17408))] = A_global[ramp(...),1, 8)]
              auto *add = store->indices[0].as<AddNode>();
              if (!add->a->IsInstance<RampNode>())
                return PrimExpr();
              if (!add->b->IsInstance<BroadcastNode>())
                return PrimExpr();
              return tir::Add(add->a.as<RampNode>()->base,
                              add->b.as<BroadcastNode>()->value);
            }
            return PrimExpr();
          }();

          if (src_offset.defined() && dst_offset.defined()) {
            auto call = MakeCPAsync(load, store, dst_offset, src_offset, bytes,
                                    true, predicate_value, index_factor);
            if (call.defined()) {
              return call.value();
            }
          }
        }
      }
    }
    return StmtMutator::VisitStmt_(store);
  }

  Stmt VisitStmt_(const BufferStoreNode *store) {
    bool is_shared = (store->buffer.scope() == "shared" ||
                      store->buffer.scope() == "shared.dyn");
    if (in_async && is_shared) {
      if (auto *load = store->value.as<BufferLoadNode>()) {
        return InjectPTX(load, store);
      }
      const BufferLoadNode *load = nullptr;
      PrimExpr predicate_value;
      if (MatchPredicatedLoad(store->value, &load, &predicate_value)) {
        return InjectPTX(load, store, true, predicate_value);
      }
    }
    return StmtMutator::VisitStmt_(store);
  }

private:
  struct LoopBodyInfo {
    const BufferStoreNode *store{nullptr};
    const BufferLoadNode *load{nullptr};
    bool predicated{false};
    PrimExpr predicate_value;
    bool has_local_force_async{false};
    PrimExpr local_src_robust_desc;
  };

  Stmt MakeWaitGroupStmt() const {
    auto wait_cnt = IntImm(DataType::Int(32), 0);
    return Evaluate(
        Call(DataType::Void(), builtin::ptx_wait_group(), {wait_cnt}));
  }

  bool ElseValueIsZero(const PrimExpr &value) const {
    if (auto *b = value.as<BroadcastNode>()) {
      if (auto *f = b->value.as<FloatImmNode>()) {
        return f->value == 0.0f;
      }
      if (auto *i = b->value.as<IntImmNode>()) {
        return i->value == 0;
      }
      return false;
    }
    if (auto *f = value.as<FloatImmNode>()) {
      return f->value == 0.0f;
    }
    if (auto *i = value.as<IntImmNode>()) {
      return i->value == 0;
    }
    return false;
  }

  bool MatchPredicatedLoad(const PrimExpr &value, const BufferLoadNode **load,
                           PrimExpr *predicate_value) const {
    const auto *call = value.as<CallNode>();
    if (!(call && call->op.same_as(builtin::if_then_else()) &&
          call->args.size() == 3)) {
      return false;
    }
    const auto *matched_load = call->args[1].as<BufferLoadNode>();
    if (matched_load == nullptr || !ElseValueIsZero(call->args[2])) {
      return false;
    }
    *load = matched_load;
    *predicate_value = call->args[0];
    return true;
  }

  bool MatchLoopBodyInfo(const Stmt &body, LoopBodyInfo *info) const {
    Stmt inner = body;
    while (const auto *attr = inner.as<AttrStmtNode>()) {
      if (attr->attr_key == tl::attr::kForceAsyncCopy) {
        info->has_local_force_async = true;
      } else if (attr->attr_key == tl::attr::kSourceRobustDesc) {
        info->local_src_robust_desc = attr->value;
      } else {
        return false;
      }
      inner = attr->body;
    }

    const auto *store = inner.as<BufferStoreNode>();
    if (store == nullptr) {
      return false;
    }
    info->store = store;
    info->load = store->value.as<BufferLoadNode>();
    if (info->load != nullptr) {
      return true;
    }
    if (!MatchPredicatedLoad(store->value, &info->load,
                             &info->predicate_value)) {
      return false;
    }
    info->predicated = true;
    return true;
  }

  Optional<Stmt>
  MakeCPAsync(const BufferLoadNode *load, const BufferStoreNode *store,
              const PrimExpr &dst_offset, const PrimExpr &src_offset, int bytes,
              bool predicated = false,
              const PrimExpr &predicate_value = PrimExpr(),
              int index_factor = 1, const PrimExpr &robust_desc = PrimExpr()) {
    Array<PrimExpr> args = {store->buffer->data,
                            mul(dst_offset, PrimExpr(index_factor)),
                            load->buffer->data, src_offset, PrimExpr(bytes)};
    Op op = tvm::tir::builtin::ptx_cp_async();
    PrimExpr effective_robust_desc =
        robust_desc.defined() ? robust_desc : current_src_robust_desc_;
    if (effective_robust_desc.defined()) {
      op = tl::musa_cp_async_robust();
      auto [robust_base, robust_size] =
          GetRobustDescArgs(effective_robust_desc);
      args.push_back(robust_base);
      args.push_back(robust_size);
    }
    if (predicated) {
      args.push_back(predicate_value);
    }
    return Evaluate(Call(store->buffer->dtype, op, args));
  }

  std::pair<PrimExpr, PrimExpr> GetRobustDescArgs(const PrimExpr &desc) const {
    const auto *call = desc.as<CallNode>();
    ICHECK(call && call->op.same_as(tl::make_robust_desc()))
        << "Expected tl.make_robust_desc call, but got " << desc;
    ICHECK_EQ(call->args.size(), 2);
    return {call->args[0], call->args[1]};
  }

  Optional<PrimExpr> GetUnitStrideBase(const PrimExpr &expr, const Var &var) {
    PrimExpr zero = make_zero(var.dtype());
    PrimExpr one = make_const(var.dtype(), 1);
    PrimExpr two = make_const(var.dtype(), 2);
    PrimExpr base0 = analyzer_.Simplify(Substitute(expr, {{var, zero}}));
    PrimExpr base1 = analyzer_.Simplify(Substitute(expr, {{var, one}}));
    PrimExpr base2 = analyzer_.Simplify(Substitute(expr, {{var, two}}));
    if (!analyzer_.CanProveEqual(base1 - base0, one)) {
      return Optional<PrimExpr>();
    }
    if (!analyzer_.CanProveEqual(base2 - base1, one)) {
      return Optional<PrimExpr>();
    }
    return base0;
  }

  arith::Analyzer analyzer_;
  bool in_async{false};
  bool in_force_async_copy_{false};
  PrimExpr current_src_robust_desc_;
};

using namespace tir::transform;

tvm::transform::Pass InjectPTXAsyncCopy() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    auto *n = f.CopyOnWrite();
    n->body = PTXAsyncCopyInjector()(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectPTXAsyncCopy", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InjectPTXAsyncCopy", InjectPTXAsyncCopy);
}

} // namespace tl
} // namespace tvm
