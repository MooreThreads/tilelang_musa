/*!
 * \file rewrite_partial_sync_to_barrier.cc
 * \brief Rewrite partial shared sync into mbarrier arrive/wait for MUSA.
 */
#include "../op/builtin.h"
#include "tvm/runtime/logging.h"
#include "tvm/tir/buffer.h"
#include "tvm/tir/builtin.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/op.h"
#include "tvm/tir/stmt_functor.h"
#include "tvm/tir/transform.h"
#include <iostream>
#include <tvm/ffi/reflection/registry.h>

#include <algorithm>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

// Collect all barrier in IR and rewrite partial sync to musa_sync placeholders.
class PartialSyncPrepass : public StmtExprMutator {
public:
  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.same_as(builtin::tvm_storage_sync())) {
        // rewrite partial thread sync IR from `tvm_storage_sync("shared.dyn",
        // barrier_id, count)` to `musa_sync(offset, count)`
        if (auto rewritten = RewriteStorageSync(call)) {
          return rewritten.value();
        }
      } else if (call->op.same_as(builtin::create_barriers())) {
        // collect barrier count from `T.create_barriers(barrier_count)`
        HandleCreateBarriers(call);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  const std::unordered_map<int, PrimExpr> &partial_syncs() const {
    return partial_syncs_;
  }
  int barrier_count() const { return barrier_count_; }
  int sync_count() const { return sync_count_; }
  int placeholder_count() const { return placeholder_count_; }

private:
  PrimExpr VisitExpr_(const CallNode *call) final {
    if (call->op.same_as(barrier_id_placeholder())) {
      placeholder_count_++;
    }
    return StmtExprMutator::VisitExpr_(call);
  }

  std::optional<Stmt> RewriteStorageSync(const CallNode *call) {
    if (call->args.size() != 3) {
      return std::nullopt;
    }
    const auto *scope = call->args[0].as<StringImmNode>();
    if (!scope) {
      return std::nullopt;
    }
    if (scope->value != "shared" && scope->value != "shared.dyn") {
      return std::nullopt;
    }
    Array<PrimExpr> args = {IntImm(DataType::Int(32), sync_count_),
                            VisitExpr(call->args[2])};
    partial_syncs_[sync_count_] = call->args[2];
    sync_count_++;
    auto new_call = Call(call->dtype, musa_sync(), args);
    return Evaluate(new_call);
  }

  void HandleCreateBarriers(const CallNode *call) {
    if (call->args.size() != 1)
      return;
    if (const auto *n = call->args[0].as<IntImmNode>()) {
      barrier_count_ += static_cast<int>(n->value);
    }
  }

  std::unordered_map<int, PrimExpr> partial_syncs_;
  int barrier_count_{0};
  int sync_count_{0};
  int placeholder_count_{0};
};

class MbarrierSyncRewriter : public StmtExprMutator {
public:
  MbarrierSyncRewriter(int base_count,
                       std::vector<std::pair<int, PrimExpr>> barrier_inits)
      : base_count_(base_count), barrier_inits_(std::move(barrier_inits)) {
    ICHECK(!barrier_inits_.empty());
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.same_as(tl::musa_sync())) {
        if (auto rewritten = RewriteMusaSync(call)) {
          return rewritten.value();
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  // insert T.ptx_init_barrier_thread_count
  Stmt VisitStmt_(const IfThenElseNode *op) final {
    // Find first tl_shuffle_elect
    if (!init_inserted_ && ShouldInsertInit(op->condition)) {
      auto then_case = StmtExprMutator::VisitStmt(op->then_case);
      Array<Stmt> stmts = MakeInitStmts();
      stmts.push_back(then_case);
      auto seq = stmts.size() == 1 ? stmts[0] : SeqStmt(stmts);
      init_inserted_ = true;
      Stmt else_case;
      if (op->else_case) {
        else_case = StmtExprMutator::VisitStmt(op->else_case.value());
      }
      return IfThenElse(op->condition, seq, else_case);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  bool init_inserted() const { return init_inserted_; }

  // Make statements for barrier inits
  Array<Stmt> MakeInitStmts() {
    Array<Stmt> stmts;
    for (const auto &[id, thread_count] : barrier_inits_) {
      auto barrier = Call(DataType::Handle(), get_mbarrier(),
                          {IntImm(DataType::Int(32), id)});
      auto count = VisitExpr(thread_count);
      auto init =
          Call(DataType::Handle(), builtin::ptx_init_barrier_thread_count(),
               {barrier, count});
      stmts.push_back(Evaluate(init));
    }
    return stmts;
  }

private:
  // rewrite musa_sync
  std::optional<Stmt> RewriteMusaSync(const CallNode *call) {
    ICHECK_EQ(call->args.size(), 2);
    auto offset = call->args[0].as<IntImmNode>()->value;
    int new_id = base_count_ + offset;
    Array<PrimExpr> args = {Call(DataType::Handle(), get_mbarrier(),
                                 {IntImm(DataType::Int(32), new_id)}),
                            VisitExpr(call->args[1])};
    auto new_call = Call(call->dtype, musa_sync(), args);
    return Evaluate(new_call);
  }

  bool ShouldInsertInit(const PrimExpr &cond) {
    if (const auto *call = cond.as<CallNode>()) {
      if (call->op.same_as(tl_shuffle_elect()) && !call->args.empty()) {
        if (const auto *imm = call->args[0].as<IntImmNode>()) {
          return imm->value == 0;
        }
      }
    } else if (const auto *eq = cond.as<EQNode>()) {
      if (const auto *rhs = eq->b.as<IntImmNode>()) {
        if (rhs->value == 0)
          return true;
      } else if (const auto *lhs = eq->a.as<IntImmNode>()) {
        if (lhs->value == 0)
          return true;
      }
    }
    return false;
  }

  std::vector<std::pair<int, PrimExpr>> barrier_inits_;
  int base_count_{0};
  bool init_inserted_{false};
};

class BarrierIdPlaceholderRewriter : public StmtExprMutator {
public:
  BarrierIdPlaceholderRewriter(int placeholder_start)
      : placeholder_next_id_(placeholder_start) {}

private:
  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(barrier_id_placeholder())) {
      int current_id = placeholder_next_id_ + 1;
      placeholder_next_id_++;
      return IntImm(DataType::Int(32), current_id);
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  int placeholder_next_id_{0};
};

class CreateBarrierRewriter : public StmtExprMutator {
public:
  static Stmt Rewrite(Stmt body, int new_barrier_count) {
    CreateBarrierRewriter rewriter;
    body = rewriter(std::move(body));

    // Insert a fresh create_barriers with the new count.
    Array<Stmt> stmts;
    auto create =
        Evaluate(Call(DataType::Handle(), builtin::create_barriers(),
                      {IntImm(DataType::Int(32), new_barrier_count)}));
    stmts.push_back(create);
    stmts.push_back(body);
    return SeqStmt(stmts);
  }

private:
  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.same_as(builtin::create_barriers())) {
        // Drop existing create_barriers; we'll emit a single unified one later.
        return Stmt();
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};

class MusaSyncPhaseRewriter : public StmtExprMutator {
public:
  Stmt Rewrite(Stmt body) {
    body = this->VisitStmt(std::move(body));
    for (auto it = phase_vars_.rbegin(); it != phase_vars_.rend(); ++it) {
      const Buffer &buf = *it;
      body = DeclBuffer(buf, body);
      Map<String, ffi::Any> annotations;
      annotations.Set(tl::attr::kLocalVarInit, IntImm(DataType::Int(32), 0));
      body = Allocate(buf->data, buf->dtype, buf->shape,
                      const_true(buf->dtype.lanes()), body, annotations);
    }
    return body;
  }

private:
  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.same_as(tl::musa_sync())) {
        return RewriteMusaSync(call);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt RewriteMusaSync(const CallNode *call) {
    ICHECK_EQ(call->args.size(), 2);
    Buffer phase_buf = MakePhaseVar();
    phase_vars_.push_back(phase_buf);
    Array<PrimExpr> indices = {IntImm(DataType::Int(32), 0)};
    PrimExpr phase = BufferLoad(phase_buf, indices);
    PrimExpr barrier = VisitExpr(call->args[0]);
    Stmt arrive = Evaluate(
        Call(DataType::Handle(), builtin::ptx_arrive_barrier(), {barrier}));
    Stmt wait = Evaluate(
        Call(DataType::Handle(), mbarrier_wait_parity(), {barrier, phase}));
    PrimExpr next_phase = bitwise_xor(phase, IntImm(DataType::Int(32), 1));
    Stmt store = BufferStore(phase_buf, next_phase, indices);
    return SeqStmt::Flatten(arrive, wait, store);
  }

  Buffer MakePhaseVar() {
    std::string name =
        "__musa_sync_phase_" + std::to_string(phase_vars_.size());
    Array<PrimExpr> shape = {IntImm(DataType::Int(32), 1)};
    return decl_buffer(shape, DataType::Int(32), name, "local.var");
  }

  std::vector<Buffer> phase_vars_;
};

PrimFunc RewritePartialSyncToBarrier(PrimFunc f) {
  auto *n = f.CopyOnWrite();

  PartialSyncPrepass prepass;
  // Run prepass on a copy of the body so we can still early-return the
  // untouched PrimFunc when there is no barrier to rewrite.
  Stmt body = prepass(n->body);
  int base_count = prepass.barrier_count();
  Array<Stmt> prefix;

  if (prepass.placeholder_count() != 0) {
    int placeholder_start = base_count;
    base_count += prepass.placeholder_count();
    BarrierIdPlaceholderRewriter rewriter(placeholder_start);
    body = rewriter(std::move(body));
  }

  if (prepass.sync_count() != 0) {
    std::vector<std::pair<int, PrimExpr>> syncs(prepass.partial_syncs().begin(),
                                                prepass.partial_syncs().end());
    std::sort(syncs.begin(), syncs.end(), [](const auto &lhs, const auto &rhs) {
      return lhs.first < rhs.first;
    });

    std::vector<std::pair<int, PrimExpr>> barrier_inits;
    barrier_inits.reserve(syncs.size());
    for (const auto &[offset, thread_count] : syncs) {
      barrier_inits.push_back({base_count + offset, thread_count});
    }

    MbarrierSyncRewriter rewriter(base_count, std::move(barrier_inits));
    body = rewriter(std::move(body));
    // If MbarrierSyncRewriter has not previously inserted barrier inits
    if (!rewriter.init_inserted()) {
      auto cond = Call(DataType::Bool(), tl_shuffle_elect(),
                       {IntImm(DataType::Int(32), 0)});
      auto init_stmts = rewriter.MakeInitStmts();
      auto seq = init_stmts.size() == 1 ? init_stmts[0] : SeqStmt(init_stmts);
      prefix.push_back(IfThenElse(cond, seq));
      if (!prefix.empty()) {
        prefix.push_back(body);
        body = SeqStmt(prefix);
      }
    }
  }

  body = CreateBarrierRewriter::Rewrite(std::move(body),
                                        base_count + prepass.sync_count());
  body = MusaSyncPhaseRewriter().Rewrite(std::move(body));

  n->body = std::move(body);
  return f;
}

namespace transform {

tvm::transform::Pass RewriteMUSAPartialSync() {
  auto pass_func = [](PrimFunc f, IRModule m,
                      const tvm::transform::PassContext &ctx) {
    return RewritePartialSyncToBarrier(std::move(f));
  };
  return tir::transform::CreatePrimFuncPass(pass_func, 0,
                                            "tl.RewriteMUSAPartialSync", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.RewriteMUSAPartialSync",
                        RewriteMUSAPartialSync);
}

} // namespace transform
} // namespace tl
} // namespace tvm
