/*!
 * \file offset_mbarrier_id.cc
 * \brief Add 1 to all get_mbarrier ids.
 */
#include "../op/builtin.h"
#include "tvm/tir/op.h"
#include "tvm/tir/stmt_functor.h"
#include "tvm/tir/transform.h"
#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace tl {

using namespace tir;

class MbarrierIdOffsetRewriter : public StmtExprMutator {
public:
  static PrimFunc Rewrite(PrimFunc f) {
    auto *n = f.CopyOnWrite();
    n->body = MbarrierIdOffsetRewriter()(n->body);
    return f;
  }

private:
  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(get_mbarrier()) && op->args.size() == 1) {
      PrimExpr id = StmtExprMutator::VisitExpr(op->args[0]);
      PrimExpr new_id;
      if (const auto *imm = id.as<IntImmNode>()) {
        new_id = IntImm(DataType::Int(32), imm->value + 1);
      } else {
        new_id = Add(id, make_const(DataType::Int(32), 1));
      }
      return Call(op->dtype, op->op, {new_id}, {}, op->span);
    }
    return StmtExprMutator::VisitExpr_(op);
  }
};



namespace transform {

tvm::transform::Pass OffsetMbarrierId() {
  auto pass_func = [](PrimFunc f, IRModule m, const tvm::transform::PassContext &ctx) {
    return MbarrierIdOffsetRewriter::Rewrite(std::move(f));
  };
  return tir::transform::CreatePrimFuncPass(pass_func, 0, "tl.OffsetMbarrierId", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.OffsetMbarrierId", OffsetMbarrierId);
}

} // namespace transform
} // namespace tl
} // namespace tvm
