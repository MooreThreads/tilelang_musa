/*!
 *  \file lower_reduce_barrier.cc
 *  \brief Lower shared.reduce_bar buffers after warp specialization.
 */
#include "../op/builtin.h"
#include "tvm/ir/type.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/stmt.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace tvm {
namespace tl {

using namespace tir;

class ReduceBarrierRewriter : public StmtExprMutator {
public:
  static Stmt Rewrite(Stmt body, bool disable_shuffle_elect = false) {
    ReduceBarrierRewriter rewriter(disable_shuffle_elect);
    return rewriter(std::move(body));
  }

private:
  explicit ReduceBarrierRewriter(bool disable_shuffle_elect) {
    (void)disable_shuffle_elect;
  }

  // place init before body
  Stmt WrapBarrierVarsAndInits(Stmt body) {
    if (!init_calls_.empty()) {
      Array<Stmt> seq = init_calls_;
      seq.push_back(std::move(body));
      body = SeqStmt(seq);
    }
    for (int i = static_cast<int>(barrier_vars_.size()) - 1; i >= 0; --i) {
      PrimExpr placeholder =
          Call(DataType::Int(32), barrier_id_placeholder(), {});
      body = LetStmt(barrier_vars_[i], placeholder, body);
    }
    return body;
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    Block block = tvm::ffi::GetRef<Block>(op);
    Array<Buffer> alloc_buffers = op->alloc_buffers;

    // collect buffer
    std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_map;
    for (auto buffer : alloc_buffers) {
      buffer_map.insert({buffer->data, buffer});
    }
    for (auto match_buffer : op->match_buffers) {
      buffer_map.insert({match_buffer->buffer->data, match_buffer->buffer});
    }

    // find barrier buffer
    Array<Buffer> barrier_buffers;
    for (const auto &[data, buffer] : buffer_map) {
      const auto *ptr_type =
          buffer->data->type_annotation.as<PointerTypeNode>();
      ICHECK(ptr_type) << "Buffer Var's type annotation must be of PointerType";
      auto storage_scope = ptr_type->storage_scope;
      if (storage_scope == "shared.reduce_bar") {
        barrier_buffers.push_back(buffer);
      }
    }

    // return directly if no barrier buffer
    if (barrier_buffers.empty()) {
      return StmtExprMutator::VisitStmt_(op);
    }

    // create barrier_var and init_calls
    Map<Buffer, PrimExpr> local_expr_remap;
    for (auto buffer : barrier_buffers) {
      Var barrier_var = GetBarrierVar(buffer);
      local_expr_remap.Set(buffer, barrier_var);
      AddInitIfNeeded(buffer, barrier_var);
    }

    // delete barrier buffer from block's allocs
    Array<Buffer> filtered;
    filtered.reserve(alloc_buffers.size());
    for (auto buf : alloc_buffers) {
      if (!local_expr_remap.count(buf)) {
        filtered.push_back(buf);
      }
    }
    if (!filtered.same_as(op->alloc_buffers)) {
      block.CopyOnWrite()->alloc_buffers = filtered;
    }

    // update this block's BufferLoad
    buffer_expr_remap_stack_.push_back(local_expr_remap);
    Stmt updated = StmtExprMutator::VisitStmt_(block.get());
    buffer_expr_remap_stack_.pop_back();
    return updated;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        auto body = StmtExprMutator::VisitStmt(op->body);
        if (!wrapped_ && (!init_calls_.empty() || !barrier_vars_.empty())) {
          wrapped_ = true;
          body = WrapBarrierVarsAndInits(std::move(body));
        }
        return AttrStmt(op->node, op->attr_key, op->value, body);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  // find reduce_barrier[] and rewrite to reduce_barrier
  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto buffer = load->buffer;
    for (auto it = buffer_expr_remap_stack_.rbegin();
         it != buffer_expr_remap_stack_.rend(); ++it) {
      if (it->count(buffer)) {
        return it->at(buffer);
      }
    }
    return load;
  }

  // find reduce_barrier[] stats and assert
  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto buffer = store->buffer;
    for (auto it = buffer_expr_remap_stack_.rbegin();
         it != buffer_expr_remap_stack_.rend(); ++it) {
      if (it->count(buffer)) {
        ICHECK(false) << "Storing to a reduce barrier var is not supported.";
      }
    }
    return store;
  }

  Var GetBarrierVar(const Buffer &buffer) {
    auto it = barrier_var_map_.find(buffer.get());
    if (it != barrier_var_map_.end()) {
      return it->second;
    }
    Var barrier_var(buffer->name, DataType::Int(32));
    barrier_var_map_.emplace(buffer.get(), barrier_var);
    barrier_vars_.push_back(barrier_var);
    return barrier_var;
  }

  // create ptx_init_barrier_thread_count for buffer not inited
  void AddInitIfNeeded(const Buffer &buffer, const Var &barrier_var) {
    // if inited, return
    if (inited_buffers_.count(buffer.get())) {
      return;
    }
    // add to inited_buffers set
    inited_buffers_.insert(buffer.get());
    // create ptx_init_barrier_thread_count
    auto count = buffer->shape[0];
    auto call =
        Call(DataType::Handle(), builtin::ptx_init_barrier_thread_count(),
             {barrier_var, PrimExpr(count)});
    Stmt init = Evaluate(call);
    Stmt marked = AttrStmt(IntImm(DataType::Int(32), 0),
                           attr::kMusaReduceBarrierInit, 1, init);
    init_calls_.push_back(marked);
  }

  // Track which reduce barrier buffers already emitted init calls.
  std::unordered_set<const BufferNode *> inited_buffers_;

  // Unique barrier id vars shared across all blocks.
  Array<Var> barrier_vars_;

  // Init stmts marked for later hoisting/merging.
  Array<Stmt> init_calls_;

  // Scope-local remaps for BufferLoad/BufferStore rewriting.
  // Push/pop per block to avoid cross-block leakage.
  std::vector<Map<Buffer, PrimExpr>> buffer_expr_remap_stack_;

  // Global mapping: reduce barrier buffer -> unique barrier var.
  std::unordered_map<const BufferNode *, Var> barrier_var_map_;

  bool wrapped_{false};
};

PrimFunc LowerReduceBarrier(PrimFunc f, bool disable_shuffle_elect) {
  f.CopyOnWrite()->body =
      ReduceBarrierRewriter::Rewrite(std::move(f->body), disable_shuffle_elect);
  return f;
}

namespace transform {

using namespace tir::transform;

tvm::transform::Pass LowerReduceBarrier() {
  auto pass_func = [](PrimFunc f, IRModule m,
                      const tvm::transform::PassContext &ctx) {
    bool disable_shuffle_elect =
        ctx->GetConfig<Bool>(kDisableShuffleElect, Bool(false)).value();
    return tl::LowerReduceBarrier(std::move(f), disable_shuffle_elect);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerReduceBarrier", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerReduceBarrier", LowerReduceBarrier);
}

} // namespace transform
} // namespace tl
} // namespace tvm
