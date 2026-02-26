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
            if (call->op.same_as(builtin::ptx_cp_async())) {
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
    if (!loop || !in_async || !in_force_async_copy_ || !is_zero(loop->min)) {
      return stmt;
    }

    const int64_t *extent_ptr = as_const_int(loop->extent);
    if (extent_ptr == nullptr || *extent_ptr <= 1) {
      return stmt;
    }

    const auto *store = loop->body.as<BufferStoreNode>();
    if (store == nullptr) {
      return stmt;
    }
    if (!(store->buffer.scope() == "shared" ||
          store->buffer.scope() == "shared.dyn")) {
      return stmt;
    }
    const auto *load = store->value.as<BufferLoadNode>();
    if (load == nullptr || load->buffer.scope() != "global") {
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
                                bytes, false, PrimExpr(), index_factor);
    if (!cp_async.defined()) {
      return stmt;
    }
    return cp_async.value();
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
            return Evaluate(Call(
                store->buffer->dtype, tvm::tir::builtin::ptx_cp_async(),
                {store->buffer->data, mul(dst_offset, PrimExpr(index_factor)),
                 load->buffer->data, src_offset, PrimExpr(bytes)}));
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
            return Evaluate(Call(
                store->buffer->dtype, tvm::tir::builtin::ptx_cp_async(),
                {store->buffer->data, mul(dst_offset, PrimExpr(index_factor)),
                 load->buffer->data, src_offset, PrimExpr(bytes),
                 predicate_value}));
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
      } else if (auto *call = store->value.as<CallNode>()) {
        // tir.if_then_else is a call to tir::builtin::if_then_else()
        if (call->op.same_as(builtin::if_then_else()) &&
            call->args.size() == 3) {
          if (auto *load = call->args[1].as<BufferLoadNode>()) {
            // Only default value of 0 is supported since 0 is the default value
            // used by cp.async ptx. @see section 9.7.8.22.3. of
            // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-memory-operations
            bool else_value_is_zero = false;
            if (auto *b = call->args[2].as<BroadcastNode>()) {
              if (auto *f = b->value.as<FloatImmNode>()) {
                else_value_is_zero = f->value == 0.0f;
              } else if (auto *i = b->value.as<IntImmNode>()) {
                else_value_is_zero = i->value == 0;
              }
            }
            if (auto *f = call->args[2].as<FloatImmNode>()) {
              else_value_is_zero = f->value == 0.0f;
            } else if (auto *i = call->args[2].as<IntImmNode>()) {
              else_value_is_zero = i->value == 0;
            }
            if (else_value_is_zero) {
              return InjectPTX(load, store, true, call->args[0]);
            }
          }
        }
      }
    }
    return StmtMutator::VisitStmt_(store);
  }

private:
  Optional<Stmt> MakeCPAsync(const BufferLoadNode *load,
                             const BufferStoreNode *store,
                             const PrimExpr &dst_offset,
                             const PrimExpr &src_offset, int bytes,
                             bool predicated = false,
                             const PrimExpr &predicate_value = PrimExpr(),
                             int index_factor = 1) {
    Array<PrimExpr> args = {store->buffer->data,
                            mul(dst_offset, PrimExpr(index_factor)),
                            load->buffer->data, src_offset, PrimExpr(bytes)};
    if (predicated) {
      args.push_back(predicate_value);
    }
    return Evaluate(
        Call(store->buffer->dtype, tvm::tir::builtin::ptx_cp_async(), args));
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
