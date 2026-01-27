/*!
 * \file late_vectorize_planner.cc
 * \brief Late vectorization planner for MUSA SIMD ops.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../target/utils.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "loop_vectorize.h"

namespace tvm {
namespace tl {

using namespace tir;
using arith::IRMutatorWithAnalyzer;

class LateVectorizePlanner : public IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    LateVectorizePlanner substituter(&analyzer);
    auto *fptr = f.CopyOnWrite();
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }

private:
  explicit LateVectorizePlanner(arith::Analyzer *analyzer)
      : IRMutatorWithAnalyzer(analyzer) {}

  bool ContainsMusaBurstOp(const Stmt &body) const {
    bool found = false;
    PostOrderVisit(body, [&](const ObjectRef &obj) {
      if (found) {
        return;
      }
      const auto *call = obj.as<CallNode>();
      if (!call) {
        return;
      }
      if (!call->op.same_as(Op::Get("tir.exp2"))) {
        return;
      }
      const DataType &t = call->dtype;
      if (t.is_float() && t.bits() == 32 && t.lanes() == 1) {
        found = true;
      }
    });
    return found;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    For for_node = Downcast<For>(IRMutatorWithAnalyzer::VisitStmt_(op));
    if (for_node->kind == ForKind::kVectorized) {
      return for_node;
    }
    if (!ContainsMusaBurstOp(for_node->body)) {
      return for_node;
    }
    return VectorizeLoop(for_node);
  }
};

tvm::transform::Pass LateVectorizePlanner() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return LateVectorizePlanner::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LateVectorizePlanner", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LateVectorizePlanner",
                        LateVectorizePlanner);
}

} // namespace tl
} // namespace tvm
