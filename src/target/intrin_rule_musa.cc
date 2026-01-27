/*!
 * \file intrin_rule_musa.cc
 * \brief MUSA intrinsic rules.
 */
#include <tvm/tir/builtin.h>
#include <tvm/tir/op_attr_types.h>

#include "../support/ffi_aliases.h"
#include "target/intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {
// Add float suffix to the intrinsics, MUSA fast math.
using tir::FLowerIntrinsic;

struct MUSAMath {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_float()) {
      switch (t.bits()) {
      case 64:
        return name;
      case 32:
        return name + 'f';
      case 16: {
        if (name == "fabs") {
          return "__habs";
        } else if (name == "round") {
          return "hrint";
        } else {
          return "h" + name;
        }
      }
      default:
        return "";
      }
    } else if (t.is_bfloat16()) {
      if (name == "fabs") {
        return "__habs";
      } else if (name == "round") {
        return "hrint";
      } else {
        return "h" + name;
      }
    } else if (t.is_int() || t.is_uint()) {
      switch (t.bits()) {
      case 32:
        return "__" + name;
      case 64:
        return "__" + name + "ll";
      default:
        return "";
      }
    }
    return "";
  }
};

struct MUSAFastMath : public MUSAMath {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_float() && t.bits() == 32) {
      return "__" + name + 'f';
    } else {
      return MUSAMath::operator()(t, name);
    }
    return "";
  }
};

struct MUSAFastMathTan : public MUSAMath {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_float()) {
      switch (t.bits()) {
      case 64:
        return name;
      // `__tanf` seems to produce some values too deviant from numpy tan
      // version. So, let's use just `tanf` instead.
      case 32:
        return name + 'f';
      case 16:
        return 'h' + name;
      default:
        return "";
      }
    }
    return "";
  }
};

struct MUSAPopcount {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_uint()) {
      switch (t.bits()) {
      case 32:
        return "__popc";
      case 64:
        return "__popcll";
      default:
        return "";
      }
    }
    return "";
  }
};

struct MUSAWarpIntrinsic {
  const Op operator()(DataType t, const Op &orig_op) const {
    if (orig_op.same_as(builtin::tvm_warp_shuffle())) {
      return Op::Get("tir.musa.__shfl_sync");
    } else if (orig_op.same_as(builtin::tvm_warp_shuffle_up())) {
      return Op::Get("tir.musa.__shfl_up_sync");
    } else {
      ICHECK(orig_op.same_as(builtin::tvm_warp_shuffle_down()));
      return Op::Get("tir.musa.__shfl_down_sync");
    }
  }
};

static PrimExpr DispatchMUSAWarpActiveMask(const PrimExpr &e) {
  const CallNode *call = e.as<CallNode>();
  return Call(call->dtype, Op::Get("tir.musa.__activemask"), call->args);
}

template <typename T> static PrimExpr DispatchMUSAShuffle(const PrimExpr &e) {
  const CallNode *call = e.as<CallNode>();
  ICHECK(call != nullptr);
  ICHECK_EQ(call->args.size(), 5); // mask, value, warp_id, width, warp_size
  Array<PrimExpr> musa_args{
      {call->args[0], call->args[1], call->args[2], call->args[3]}};
  return Call(call->dtype, T()(call->dtype, Downcast<Op>(call->op)), musa_args);
}

static PrimExpr DispatchMUSAExp2(const PrimExpr &e) {
  const CallNode *call = e.as<CallNode>();
  ICHECK(call != nullptr);
  ICHECK_EQ(call->args.size(), 1U);
  DataType t = call->dtype;
  if (t.is_float() && t.bits() == 32) {
    if (t.lanes() == 2) {
      return Call(t, builtin::call_pure_extern(),
                  {StringImm("tl::vec_exp2_f2"), call->args[0]});
    }
    if (t.lanes() == 4) {
      return Call(t, builtin::call_pure_extern(),
                  {StringImm("tl::vec_exp2_f4"), call->args[0]});
    }
  }
  return DispatchPureExtern<MUSAMath>(e);
}

TVM_REGISTER_OP("tir.rsqrt")
    .set_attr<FLowerIntrinsic>("musa.FLowerIntrinsic",
                               DispatchPureExtern<MUSAMath>);

TVM_REGISTER_OP("tir.exp2")
    .set_attr<FLowerIntrinsic>("musa.FLowerIntrinsic", DispatchMUSAExp2, 11);

} // namespace intrin
} // namespace codegen
} // namespace tvm
