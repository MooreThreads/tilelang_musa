/*!
 * \file tl/op/reduce.cc
 * \brief Implementation of reduction operators
 */

#include "reduce.h"

#include <tvm/arith/pattern.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt_functor.h>

#include "../layout/layout.h"
#include "../layout/utils.h"
#include "../op/parallel.h"
#include "../target/utils.h"
#include "../transform/loop_partition.h"
#include "builtin.h"
#include "tir/transforms/ir_utils.h"
#include "tvm/ir/expr.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/stmt.h"
#include <tvm/tir/transform.h>
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace {

const DataType kFloat32x4 = DataType::Float(32, 4);
const DataType kFloat32x2 = DataType::Float(32, 2);
const DataType kInt32 = DataType::Int(32);
const DataType kUInt32 = DataType::UInt(32);

struct MusaInThreadSimdPlan {
  bool enabled{false};
  Var rv;
  PrimExpr base;
  int64_t groups{0};
};

struct MusaInterThreadSimdPlan {
  bool enabled{false};
  Var dst_var;
  PrimExpr base;
  int64_t groups{0};
};

const char *GetMusaSimdExtern(const ReduceType &reduce_type, int lanes) {
  ICHECK(lanes == 2 || lanes == 4);
  if (reduce_type->isMax()) {
    return lanes == 4 ? "tl::vec_max_f4" : "tl::vec_max_f2";
  }
  if (reduce_type->isSum()) {
    return lanes == 4 ? "tl::vec_sum_f4" : "tl::vec_sum_f2";
  }
  LOG(FATAL) << "Unsupported reduce type for MUSA SIMD: " << reduce_type->type;
  return "";
}

bool CheckMusaSimdCommon(const Target &target, const ReduceType &reduce_type,
                         const Buffer &src_buffer, const Buffer &dst_buffer) {
  if (!TargetIsMusa(target)) {
    return false;
  }
  if (!reduce_type->isMax() && !reduce_type->isSum()) {
    return false;
  }
  if (!src_buffer->dtype.is_float() || src_buffer->dtype.bits() != 32 ||
      !src_buffer->dtype.is_scalar()) {
    return false;
  }
  if (!dst_buffer->dtype.is_float() || dst_buffer->dtype.bits() != 32 ||
      !dst_buffer->dtype.is_scalar()) {
    return false;
  }
  return true;
}

MusaInThreadSimdPlan
PlanMusaInThreadSimd(const Target &target, const ReduceType &reduce_type,
                     const Buffer &src_buffer, const Buffer &dst_buffer,
                     const Array<PrimExpr> &src_indice_compressed,
                     const Array<IterVar> &src_var_compressed) {
  MusaInThreadSimdPlan plan;
  if (!CheckMusaSimdCommon(target, reduce_type, src_buffer, dst_buffer)) {
    return plan;
  }
  if (src_indice_compressed.size() != 1 || src_var_compressed.size() != 1) {
    return plan;
  }

  Var rv = src_var_compressed[0]->var;
  auto uses_var = [](const PrimExpr &expr, const Var &var) {
    return UsesVar(expr, [&](const VarNode *v) { return v == var.get(); });
  };
  if (!uses_var(src_indice_compressed[0], rv)) {
    return plan;
  }

  auto coeffs = arith::DetectLinearEquation(src_indice_compressed[0], {rv});
  if (coeffs.size() != 2) {
    return plan;
  }
  auto coeff = as_const_int(coeffs[0]);
  if (coeff == nullptr || *coeff != 1 || uses_var(coeffs[1], rv)) {
    return plan;
  }

  auto extent = as_const_int(src_var_compressed[0]->dom->extent);
  if (extent == nullptr || *extent < 8 || (*extent % 8) != 0) {
    return plan;
  }

  plan.enabled = true;
  plan.rv = rv;
  plan.base = coeffs[1];
  plan.groups = *extent / 8;
  return plan;
}

MusaInterThreadSimdPlan
PlanMusaInterThreadSimd(const Target &target, const ReduceType &reduce_type,
                        const Buffer &src_buffer, const Buffer &dst_buffer,
                        const Array<PrimExpr> &dst_indices,
                        const Array<IterVar> &dst_vars,
                        const Fragment &dst_layout) {
  MusaInterThreadSimdPlan plan;
  if (!CheckMusaSimdCommon(target, reduce_type, src_buffer, dst_buffer)) {
    return plan;
  }
  if (dst_indices.size() != 1 || dst_vars.size() != 1) {
    return plan;
  }

  Var dv = dst_vars[0]->var;
  auto input_extent = as_const_int(dst_vars[0]->dom->extent);
  if (input_extent == nullptr) {
    return plan;
  }
  Array<PrimExpr> out_shape = dst_layout->OutputShape();
  if (out_shape.size() != 1) {
    return plan;
  }
  auto out_extent = as_const_int(out_shape[0]);
  if (out_extent == nullptr || *out_extent < 4 || (*out_extent % 4) != 0) {
    return plan;
  }

  arith::Analyzer local_analyzer;
  local_analyzer.Bind(
      dv, Range::FromMinExtent(make_zero(dv.dtype()),
                               make_const(dv.dtype(), *input_extent)));
  auto dst_set = local_analyzer.int_set(dst_indices[0]);
  auto dst_min = as_const_int(dst_set.min());
  auto dst_max = as_const_int(dst_set.max());
  if (!(dst_min && dst_max && *dst_min == 0 && *dst_max == (*out_extent - 1))) {
    return plan;
  }

  plan.enabled = true;
  plan.dst_var = dv;
  plan.base = make_zero(dv.dtype());
  plan.groups = *out_extent / 4;
  return plan;
}

} // namespace

// NormalizeToBufferRegion moved to src/op/utils.{h,cc}

// MakeAccessPtrFromRegion moved to src/op/utils.{h,cc}

ReduceOp::ReduceOp(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ObjectPtr<ReduceOpNode> node = tvm::ffi::make_object<ReduceOpNode>();
  // Accept BufferRegion/BufferLoad for src/dst
  node->srcRegion_ = NormalizeToBufferRegion(args[0]);
  node->dstRegion_ = NormalizeToBufferRegion(args[1]);
  node->src = node->srcRegion_->buffer;
  node->dst = node->dstRegion_->buffer;
  std::string reduce_type = args[2].as<StringImm>().value()->value;
  node->dim = args[3].as<IntImm>().value()->value;
  node->type = ReduceType(reduce_type);
  node->clear = args[4].as<Bool>().value();
  data_ = std::move(node);
}

TileOperator ReduceOpNode::Clone() const {
  auto op = tvm::ffi::make_object<ReduceOpNode>(*this);
  return ReduceOp(op);
}

TileOperator CumSumOpNode::Clone() const {
  auto op = tvm::ffi::make_object<CumSumOpNode>(*this);
  return CumSumOp(op);
}

PrimExpr ReduceOpNode::MakeInitValue() const {
  auto dst_dtype = dst->dtype;
  auto is_int = dst_dtype.is_int();
  bool is_uint = dst_dtype.is_uint();
  auto bits = dst_dtype.bits();

  if (type->isSum()) {
    return make_zero(dst->dtype);
  } else if (type->isAbsSum()) {
    return make_zero(dst->dtype);
  } else if (type->isMax()) {
    if (is_int) {
      return make_const(dst->dtype, -(1 << (bits - 1)));
    } else if (is_uint) {
      return make_const(dst->dtype, 0);
    } else {
      return make_const(dst->dtype, -INFINITY);
    }
  } else if (type->isMin()) {
    if (is_int) {
      return make_const(dst->dtype, (1 << (bits - 1)) - 1);
    } else if (is_uint) {
      return make_const(dst->dtype, (1 << bits) - 1);
    } else {
      return make_const(dst->dtype, INFINITY);
    }
  } else if (type->isAbsMax()) {
    return make_const(dst->dtype, 0);
  } else if (type->isBitAnd()) {
    if (is_int) {
      return make_const(dst->dtype, -1);
    } else if (is_uint) {
      return make_const(dst->dtype, (1 << bits) - 1);
    } else {
      // Should not arrive here
      return make_const(dst->dtype, -INFINITY);
    }
  } else if (type->isBitOr()) {
    return make_zero(dst->dtype);
  } else if (type->isBitXor()) {
    return make_zero(dst->dtype);
  } else {
    LOG(FATAL) << "Unsupported reduce type: " << type->type;
    return PrimExpr();
  }
}

PrimExpr ReduceOpNode::MakeReduce(const PrimExpr &acc,
                                  const PrimExpr &b) const {
  PrimExpr rhs = b;
  if (acc->dtype != rhs->dtype) {
    rhs = Cast(acc->dtype, rhs);
  }
  if (type->isSum()) {
    return acc + rhs;
  } else if (type->isAbsSum()) {
    return acc + Max(rhs, -rhs);
  } else if (type->isMax()) {
    return Max(acc, rhs);
  } else if (type->isMin()) {
    return Min(acc, rhs);
  } else if (type->isAbsMax()) {
    return Max(acc, tvm::abs(rhs));
  } else if (type->isBitAnd()) {
    return acc & rhs;
  } else if (type->isBitOr()) {
    return acc | rhs;
  } else if (type->isBitXor()) {
    return acc ^ rhs;
  } else {
    LOG(FATAL) << "Unsupported reduce type: " << type->type;
  }
}

std::string ReduceOpNode::MakeCodegenReducer() const {
  if (type->isSum()) {
    return "tl::SumOp";
  } else if (type->isAbsSum()) {
    return "tl::SumOp";
  } else if (type->isMax()) {
    return "tl::MaxOp";
  } else if (type->isMin()) {
    return "tl::MinOp";
  } else if (type->isAbsMax()) {
    return "tl::MaxOp";
  } else if (type->isBitAnd()) {
    return "tl::BitAndOp";
  } else if (type->isBitOr()) {
    return "tl::BitOrOp";
  } else if (type->isBitXor()) {
    return "tl::BitXorOp";
  } else {
    LOG(FATAL) << "Unsupported reduce type: " << type->type;
    return "";
  }
}

static Array<PrimExpr> InputPlaceholders(size_t n) {
  Array<PrimExpr> result;
  result.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    result.push_back(InputPlaceholder(i));
  }
  return result;
}

static Fragment ComputeReducerLayout(const Fragment &src_layout, int dim) {
  PrimExpr src_rep_extent = src_layout->ReplicateExtent();
  PrimExpr indice_rep_extent = src_layout->InputShape()[dim];
  PrimExpr reducer_rep_extent = indice_rep_extent * src_rep_extent;

  auto fwd = InputPlaceholders(src_layout->InputDim() - 1);
  fwd.insert(fwd.begin() + dim,
             FloorMod(ReplicationPlaceholder(), indice_rep_extent));

  auto thd = src_layout->ForwardThread(
      fwd, FloorDiv(ReplicationPlaceholder(), indice_rep_extent));

  auto reducer_shape = src_layout->InputShape();
  reducer_shape.erase(reducer_shape.begin() + dim);
  if (reducer_shape.empty()) {
    reducer_shape.push_back(1);
  }

  auto reducer_layout =
      Fragment(reducer_shape, {}, thd, reducer_rep_extent, std::nullopt)
          ->CondenseReplicateVar()
          ->BindThreadRange(src_layout->ThreadRange());
  return reducer_layout;
}

/**
 * @brief Lower the Reduce operator to a TIR statement.
 *
 * Lowers a ReduceOpNode operating on fragment-scoped buffers into a sequence of
 * TIR statements implementing: optional initialization, thread-local reduction
 * (unrolled inner loops), inter-thread reduction via a runtime AllReduce call
 * (Hopper targets use `NamedBarrier` instead of the default
 * `SyncThreadsBarrier`), and an optional accumulation or copy back to the
 * destination buffer when a temporary clear buffer is used.
 *
 * Behavior notes:
 * - Only supports src and dst in "local.fragment" scope; otherwise it checks
 *   and aborts with "Reduce for shared memory not implemented.".
 * - Supports both 1D reductions (scalar output) and reductions along a single
 *   extra dimension; validates layout dimensionality consistency.
 * - If `clear` is set (or for sum/abssum reductions), an initial value is
 *   written to the clear buffer; for non-clearing sum/abssum a duplicate
 *   temporary buffer is allocated and accumulated back into dst after
 * reduction.
 * - Performs iterator compression for local reduction loops using `analyzer`.
 * - Detects parallel thread splitting from the normalized iterator sum and
 *   emits a call to a templated `tl::AllReduce<...>::run`
 *   via `builtin::call_extern`. For sufficiently large reducing thread counts
 *   (> 32) a workspace is allocated via T.AddWorkspace and passed to the
 *   AllReduce call.
 * - The final body is wrapped in parallel loops over the destination spatial
 *   dimensions and partitioned by the lowering thread variable. If a temporary
 *   clear buffer is used, it is allocated for the body.
 *
 * @param T Lowering context providing buffer and layout maps, thread bounds,
 *          target information, thread variable, and workspace allocation
 * helper.
 * @param analyzer Analyzer used for iterator compression and arithmetic
 * normalization.
 * @return Stmt Lowered TIR statement implementing the reduction.
 */
Stmt ReduceOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  auto get_buffer = [&](const Buffer &buf) {
    if (T.buffer_remap.count(buf))
      return T.buffer_remap[buf];
    return buf;
  };

  auto src_scope = this->src.scope();
  auto dst_scope = this->dst.scope();

  if (src_scope == "local.fragment" && dst_scope == "local.fragment") {

    auto src_buffer = get_buffer(this->src);
    auto dst_buffer = get_buffer(this->dst);
    auto src_layout = T.layout_map[this->src].as<Fragment>().value();
    auto dst_layout = T.layout_map[this->dst].as<Fragment>().value();
    auto red_layout = ComputeReducerLayout(src_layout, dim);
    auto src_dim = src_layout->InputDim();
    auto dst_dim = dst_layout->InputDim();

    auto is_1d_reduce = src_dim == dst_dim && dst_dim == 1;

    if (is_1d_reduce) {
      ICHECK(is_one(dst_layout->OutputShape().back()))
          << "Reduce for scalar not implemented.";
    } else {
      ICHECK_EQ(src_dim, dst_dim + 1) << "Reduce dimension mismatch.";
    }

    Array<IterVar> dst_vars;
    for (size_t i = 0; i < dst_dim; ++i) {
      Var var = Var(std::string{char('i' + i)});
      dst_vars.push_back(IterVar(Range(0, dst_layout->InputShape()[i]), var,
                                 IterVarType::kDataPar));
    }

    Array<IterVar> src_vars;
    if (!is_1d_reduce) {
      src_vars = dst_vars;
    }
    Range reduce_dom(0, src_layout->InputShape()[this->dim]);
    IterVar reduce_iv(reduce_dom, Var("rv"), IterVarType::kDataPar);
    src_vars.insert(src_vars.begin() + this->dim, reduce_iv);

    auto src_indices = src_layout->Forward(
        src_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }));
    auto dst_indices = dst_layout->Forward(
        dst_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }));
    auto red_indices = red_layout->Forward(
        dst_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }));

    Array<Stmt> stmts;
    Array<Stmt> stmts_after_loop;

    auto require_init = this->clear;
    if (this->type->isSum() || this->type->isAbsSum() ||
        this->type->isBitAnd() || this->type->isBitOr() ||
        this->type->isBitXor()) {
      require_init = true;
    }

    auto clear_buffer = dst_buffer;
    auto need_duplicate = false;
    auto need_update = false;
    if ((this->type->isSum() || this->type->isAbsSum()) && !this->clear) {
      need_duplicate = true;
      need_update = true;
    } else if (this->type->isBitAnd() && !this->clear) {
      need_duplicate = true;
      need_update = true;
    } else if ((this->type->isBitOr() || this->type->isBitXor()) &&
               !this->clear) {
      need_duplicate = true;
      need_update = true;
    } else if ((this->type->isMax() || this->type->isMin() ||
                this->type->isAbsMax()) &&
               !this->clear) {
      need_duplicate = true;
      need_update = true;
    }

    // red_layout should always contain dst_layout
    // if we can prove they are the same, no need to duplicate buffer
    // otherwise, red_layout contains more replicated dimensions than dst_layout
    if (!analyzer->CanProve(dst_layout->ReplicateExtent() ==
                            red_layout->ReplicateExtent())) {
      need_duplicate = true;
    }
    ICHECK(!analyzer->CanProve(dst_layout->ReplicateExtent() >
                               red_layout->ReplicateExtent()))
        << "Inconsistent layouts between src and dst in ReduceOp: "
        << "dst_layout=" << dst_layout << "red_layout=" << red_layout;

    if (need_duplicate) {
      // Create a new buffer with same shape and dtype as dst_buffer
      clear_buffer = decl_buffer(red_layout->OutputShape(), dst_buffer->dtype,
                                 dst_buffer->name + "_clear",
                                 GetPtrStorageScope(dst_buffer->data));
    }
    // make reduce-init stmt
    // For max/min/absmax with clear=false and need_duplicate, we still need to
    // initialize the temporary buffer with identity values since the original
    // dst values will be combined later via need_update
    if (require_init ||
        (need_duplicate && (this->type->isMax() || this->type->isMin() ||
                            this->type->isAbsMax()))) {
      stmts.push_back(
          BufferStore(clear_buffer, this->MakeInitValue(), red_indices));
    }

    // make thread-local reduce
    Array<PrimExpr> src_indice_compressed;
    Array<IterVar> src_var_compressed;
    for (size_t i = 0; i < src_layout->OutputDim(); ++i) {
      auto [expr, var] = CompressIterator(src_indices[i], src_vars,
                                          src_vars[this->dim]->var, analyzer);
      src_indice_compressed.push_back(expr);
      src_var_compressed.push_back(var);
    }

    tvm::transform::PassContext pass_ctx = tvm::transform::PassContext::Current();
    bool enable_reduce_burst =
        pass_ctx->GetConfig<Bool>(kEnableReduceBurst, Bool(false)).value();

    // Use MUSA SIMD reduce when reduction is contiguous float32 with extent
    // % 8 and reduce burst is enabled.
    MusaInThreadSimdPlan simd_plan;
    if (enable_reduce_burst) {
      simd_plan =
          PlanMusaInThreadSimd(T.target, this->type, src_buffer, dst_buffer,
                               src_indice_compressed, src_var_compressed);
    }

    Stmt reduce_local;
    if (simd_plan.enabled) {
      Var rv_outer(simd_plan.rv->name_hint + "_vec", simd_plan.rv->dtype);
      PrimExpr rv_scale = make_const(rv_outer.dtype(), 8);
      PrimExpr idx_base =
          analyzer->Simplify(simd_plan.base + rv_outer * rv_scale);

      Var vec4_var("vec4", kFloat32x4);
      Var vec2_var("vec2", kFloat32x2);
      PrimExpr vec4_0 = BufferLoad(
          src_buffer, {Ramp(idx_base, make_const(idx_base.dtype(), 1), 4)});
      PrimExpr vec4_1 = BufferLoad(
          src_buffer,
          {Ramp(analyzer->Simplify(idx_base + make_const(idx_base.dtype(), 4)),
                make_const(idx_base.dtype(), 1), 4)});

      PrimExpr vec4_expr =
          Call(vec4_0.dtype(), builtin::call_pure_extern(),
               {StringImm(GetMusaSimdExtern(this->type, 4)), vec4_0, vec4_1});
      PrimExpr vec2_0 =
          Shuffle({vec4_var}, {make_const(kInt32, 0), make_const(kInt32, 1)});
      PrimExpr vec2_1 =
          Shuffle({vec4_var}, {make_const(kInt32, 2), make_const(kInt32, 3)});
      PrimExpr vec2_expr =
          Call(vec2_0.dtype(), builtin::call_pure_extern(),
               {StringImm(GetMusaSimdExtern(this->type, 2)), vec2_0, vec2_1});
      PrimExpr s0 = Shuffle({vec2_var}, {make_const(kInt32, 0)});
      PrimExpr s1 = Shuffle({vec2_var}, {make_const(kInt32, 1)});
      Var group_reduce_var("group_reduce", s0.dtype());

      bool has_init_stmt =
          require_init ||
          (need_duplicate && (this->type->isMax() || this->type->isMin() ||
                              this->type->isAbsMax()));
      if (has_init_stmt && simd_plan.groups == 1) {
        stmts.pop_back();
        reduce_local = BufferStore(clear_buffer, group_reduce_var, red_indices);
      } else {
        reduce_local =
            BufferStore(clear_buffer,
                        this->MakeReduce(BufferLoad(clear_buffer, red_indices),
                                         group_reduce_var),
                        red_indices);
      }

      reduce_local =
          LetStmt(group_reduce_var, this->MakeReduce(s0, s1), reduce_local);
      reduce_local = LetStmt(vec2_var, vec2_expr, reduce_local);
      reduce_local = LetStmt(vec4_var, vec4_expr, reduce_local);
      reduce_local =
          For(rv_outer, 0, make_const(rv_outer.dtype(), simd_plan.groups),
              ForKind::kUnrolled, reduce_local, std::nullopt,
              {{tir::attr::pragma_unroll_explicit, Bool(false)}});
    } else {
      reduce_local = BufferStore(
          clear_buffer,
          this->MakeReduce(BufferLoad(clear_buffer, red_indices),
                           BufferLoad(src_buffer, src_indice_compressed)),
          red_indices);

      for (int i = static_cast<int>(src_layout->OutputDim()) - 1; i >= 0; --i) {
        reduce_local = For(src_var_compressed[i]->var, 0,
                           src_var_compressed[i]->dom->extent,
                           ForKind::kUnrolled, reduce_local, std::nullopt,
                           {{tir::attr::pragma_unroll_explicit, Bool(false)}});
      }
    }
    stmts.push_back(reduce_local);

    auto src_thread = src_layout->ForwardThread(
        src_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }), {});
    auto iter_sum =
        arith::NormalizeToIterSum(src_thread, ToVMap(src_vars), analyzer);

    MusaInterThreadSimdPlan simd_allreduce_plan;
    if (enable_reduce_burst) {
      simd_allreduce_plan =
          PlanMusaInterThreadSimd(T.target, this->type, src_buffer, dst_buffer,
                                  red_indices, dst_vars, red_layout);
    }
    bool simd_allreduce_emitted = false;

    for (const auto &iter_split : iter_sum->args) {
      auto mark = iter_split->source->source.as<Var>();
      ICHECK(mark) << "Not a normalized iterator: " << iter_split->source;
      if (mark.value().same_as(src_vars[this->dim]->var)) {
        // `scale` is the stride of participating threads in the thread index
        // space.  When the thread-to-data mapping for the reduce dimension is
        // normalized as  threadIdx = source * scale + ...,
        //   * scale == 1  means threads are contiguous (0, 1, 2, ...),
        //   * scale  > 1  means threads are interleaved (0, scale, 2*scale,
        //     ...).
        // Both cases use the recursive XOR-butterfly reduce.
        // `extent` is the number of distinct thread positions along the reduce
        // dimension, so reducing_threads = extent * scale covers the full
        // thread index range that participates in the reduction.
        auto scale = as_const_int(iter_split->scale);
        auto extent = as_const_int(iter_split->extent);
        ICHECK(scale != nullptr && extent != nullptr);
        if (*extent == 1)
          continue;

        int reducing_threads = (*extent) * (*scale);

        // Fast path for contiguous float32 vectors using PH1 shuffle
        // intrinsics. Disable it when duplicate/predicate write-back is
        // required so ordering remains unchanged.
        if (simd_allreduce_plan.enabled && !simd_allreduce_emitted &&
            !need_duplicate && reducing_threads <= 32) {
          Optional<Var> dv_outer;
          PrimExpr dv_base;
          if (simd_allreduce_plan.groups > 1) {
            dv_outer = Var(simd_allreduce_plan.dst_var->name_hint + "_vec",
                           simd_allreduce_plan.dst_var->dtype);
            PrimExpr dv_scale = make_const(dv_outer.value().dtype(), 4);
            dv_base = analyzer->Simplify(simd_allreduce_plan.base +
                                         dv_outer.value() * dv_scale);
          } else {
            dv_base = simd_allreduce_plan.base;
          }

          int steps = 0;
          for (int t = reducing_threads; t > *scale; t >>= 1) {
            ++steps;
          }
          ICHECK_GT(steps, 0);

          Var offset_iter("offset_step", kInt32);
          PrimExpr offset_expr =
              right_shift(make_const(offset_iter.dtype(), reducing_threads),
                          offset_iter + make_const(offset_iter.dtype(), 1));
          Var offset_var("offset", kInt32);

          Var local_vec_var("local_vec", kFloat32x4);
          Var other_vec_var("other_vec", kFloat32x4);
          PrimExpr local_vec_expr = BufferLoad(
              clear_buffer, {Ramp(dv_base, make_const(dv_base.dtype(), 1), 4)});
          PrimExpr other_vec_expr = Call(
              kFloat32x4, builtin::call_pure_extern(),
              {StringImm("tl::shfl_xor_sync"), make_const(kUInt32, 0xffffffff),
               local_vec_var, offset_var});
          PrimExpr updated = Call(kFloat32x4, builtin::call_pure_extern(),
                                  {StringImm(GetMusaSimdExtern(this->type, 4)),
                                   local_vec_var, other_vec_var});
          Stmt update =
              BufferStore(clear_buffer, updated,
                          {Ramp(dv_base, make_const(dv_base.dtype(), 1), 4)});

          Stmt loop_body = LetStmt(other_vec_var, other_vec_expr, update);
          loop_body = LetStmt(local_vec_var, local_vec_expr, loop_body);
          loop_body = LetStmt(offset_var, offset_expr, loop_body);

          Stmt loop =
              For(offset_iter, 0, make_const(offset_iter.dtype(), steps),
                  ForKind::kUnrolled, loop_body, std::nullopt,
                  {{tir::attr::pragma_unroll_explicit, Bool(false)}});
          if (simd_allreduce_plan.groups > 1) {
            ICHECK(dv_outer.defined());
            Stmt group_loop =
                For(dv_outer.value(), 0,
                    make_const(dv_outer.value().dtype(),
                               simd_allreduce_plan.groups),
                    ForKind::kUnrolled, loop, std::nullopt,
                    {{tir::attr::pragma_unroll_explicit, Bool(false)}});
            stmts_after_loop.push_back(group_loop);
          } else {
            stmts_after_loop.push_back(loop);
          }
          simd_allreduce_emitted = true;
          continue;
        }

        std::stringstream ss;

        auto thread_offset = T.thread_bounds->min;
        bool use_musa_sync = TargetIsPH1(T.target);
        Buffer reduce_sync_barrier;
        if (TargetIsHopper(T.target) || TargetIsSm100(T.target)) {
          auto all_threads = T.thread_bounds->extent;
          ss << "tl::AllReduce<" << this->MakeCodegenReducer() << ", "
             << reducing_threads << ", " << (*scale) << ", " << thread_offset
             << ", " << all_threads << ">::run_hopper";
        } else if (use_musa_sync) {
          auto all_threads = T.thread_bounds->extent;
          reduce_sync_barrier = T.AddBarrier(*as_const_int(all_threads));
          ss << "tl::AllReduceWS<" << this->MakeCodegenReducer() << ", "
             << reducing_threads << ", " << (*scale) << ", " << thread_offset
             << ", " << all_threads << ">::run";
        } else {
          ss << "tl::AllReduce<" << this->MakeCodegenReducer() << ", "
             << reducing_threads << ", " << (*scale) << ", " << thread_offset
             << ">::run";
        }
        Array<PrimExpr> thread_reduce_args = {
            StringImm(ss.str()), BufferLoad(clear_buffer, red_indices)};
        if (use_musa_sync) {
          PrimExpr barrier_id =
              BufferLoad(reduce_sync_barrier, {IntImm(DataType::Int(32), 0)});
          thread_reduce_args.push_back(barrier_id);
        }
        // The butterfly reduce path needs one shared-memory slot per
        // thread in the block.
        if (reducing_threads >= 32) {
          int workspace_size =
              static_cast<int>(*as_const_int(T.thread_bounds->extent));
          PrimExpr workspace =
              T.AddWorkspace(workspace_size, clear_buffer->dtype);
          thread_reduce_args.push_back(workspace);
        }
        auto call = Call(clear_buffer->dtype, builtin::call_extern(),
                         thread_reduce_args);
        stmts.push_back(BufferStore(clear_buffer, call, red_indices));
      }
    }

    // Layout status in the loop:
    //     clear_buffer: red_layout
    //     dst_buffer:   dst_layout
    //     loop_layout:  red_layout
    // At each step of the loop, we do reduction on
    // `clear_buffer[red_layout(loop_idx)]`
    //   and then transfer it to `dst_buffer[dst_layout(loop_idx)]`
    // However, since the red_layout is larger than dst_layout, not all write
    // operations are valid We need to add predicate to guard the write
    // operations
    PrimExpr predicate = Bool(true);
    {
      // dst_indices is the same as loop_indices
      auto dst_th_indices = dst_indices;
      dst_th_indices.push_back(T.thread_var);
      // 1. compute loop_idx based on thread: [dst_indices, T.thread_var] =>
      // [loop_indices]
      auto inv = dst_layout->Inverse()->Forward(dst_th_indices);
      inv.pop_back(); // remove replicate var
      // 2. ensure computed loop_idx maps back to the same [loop_indices]
      for (int i = 0; i < static_cast<int>(dst_layout->InputDim()); i++) {
        predicate = predicate && (inv[i] == dst_vars[i]->var);
      }
      // 3. simplify predicate
      predicate = analyzer->Simplify(predicate);
    }
    if (need_duplicate) {
      PrimExpr update;
      if (need_update) {
        auto src_val = BufferLoad(clear_buffer, red_indices);
        auto dst_val = BufferLoad(dst_buffer, dst_indices);
        if (this->type->isSum() || this->type->isAbsSum()) {
          update = dst_val + src_val;
        } else if (this->type->isBitAnd()) {
          update = this->clear ? src_val : bitwise_and(dst_val, src_val);
        } else if (this->type->isBitOr()) {
          update = bitwise_or(dst_val, src_val);
        } else if (this->type->isBitXor()) {
          update = bitwise_xor(dst_val, src_val);
        } else if (this->type->isMax() || this->type->isAbsMax()) {
          update = Max(dst_val, src_val);
        } else if (this->type->isMin()) {
          update = Min(dst_val, src_val);
        } else {
          LOG(FATAL) << "Unsupported reduce type: " << this->type->type;
        }
      } else {
        update = BufferLoad(clear_buffer, red_indices);
      }
      auto store = BufferStore(dst_buffer, update, dst_indices);
      if (analyzer->CanProve(predicate)) {
        stmts.push_back(store);
      } else {
        stmts.push_back(IfThenElse(predicate, store));
      }
    }

    auto body = stmts.size() > 1 ? SeqStmt(stmts) : stmts[0];
    for (int i = static_cast<int>(dst_layout->InputDim()) - 1; i >= 0; --i) {
      body = For(dst_vars[i]->var, 0, dst_vars[i]->dom->extent,
                 ForKind::kParallel, body);
    }

    if (dst_layout->InputDim() > 0) {
      body = PartitionLoop(Downcast<For>(body), T.thread_var, analyzer,
                           red_layout);
      body = PragmaUnrollLoop(Downcast<For>(body));
    } else {
      auto guard = (T.thread_var == T.thread_bounds->min);
      body = IfThenElse(guard, body);
    }

    if (stmts_after_loop.size() > 0) {
      Stmt after = stmts_after_loop.size() > 1 ? SeqStmt(stmts_after_loop)
                                               : stmts_after_loop[0];
      body = SeqStmt({body, after});
    }

    if (need_duplicate) {
      body = Allocate(clear_buffer->data, clear_buffer->dtype,
                      clear_buffer->shape, const_true(), body);
    }
    return body;
  }

  LOG(FATAL) << "Reduce for buffers in scope (" << src_scope << ", "
             << dst_scope << ") is not implemented.";
  return Stmt();
}

LayoutMap ReduceOpNode::InferLayout(const LayoutInferArgs &T,
                                    InferLevel level) const {
  if (level >= InferLevel::kStrict)
    return {};

  if (IsFragmentBuffer(src) && IsFragmentBuffer(dst) &&
      T.layout_map.count(src)) {
    auto src_layout = T.layout_map[src].as<Fragment>().value();
    auto reducer_layout = ComputeReducerLayout(src_layout, this->dim);

    if (!T.layout_map.count(dst)) {
      return {{dst, reducer_layout}};
    }

    auto orig_dst_layout = T.layout_map.Get(dst).value().as<Fragment>().value();
    ICHECK(reducer_layout->InputDim() == orig_dst_layout->InputDim());

    auto indices = InputPlaceholders(reducer_layout->InputDim());
    arith::Analyzer analyzer;
    for (size_t i = 0; i < indices.size(); i++) {
      analyzer.Bind(Downcast<Var>(indices[i]),
                    Range(0, reducer_layout->InputShape()[i]));
    }
    if (!ProveFragmentContains(orig_dst_layout, reducer_layout, indices,
                               indices, analyzer)) {
      std::ostringstream oss;
      oss << "Layout may conflict with ReduceOp for buffer " << dst << " vs. "
          << src << "\n"
          << "src_layout = " << src_layout << "\n"
          << "reducer_layout = " << reducer_layout << "\n"
          << "orig_dst_layout = " << orig_dst_layout << "\n"
          << "You may need to use a shared memory to transform the "
             "layout";
      throw LayoutConflictException(oss.str());
    }
  }
  return {};
}

TIR_REGISTER_TL_TILE_OP(ReduceOp, reduce)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// Normalize "Buffer" to BufferRegion. Use the shape of the buffer as the
// ranges.
static BufferRegion ConvertBufferToBufferRegion(const Buffer &buf) {
  Array<Range> ranges;
  for (PrimExpr extent : buf->shape) {
    ranges.push_back(Range(IntImm(extent->dtype, 0), extent));
  }
  return BufferRegion(buf, ranges);
}

CumSumOp::CumSumOp(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  /// CumSum constructor arguments:
  /// - src: input buffer
  /// - dst: output buffer
  /// - dim: dimension to cumsum
  /// - reverse: whether to cumsum in reverse order
  CHECK_EQ(args.size(), 4);
  ObjectPtr<CumSumOpNode> node = tvm::ffi::make_object<CumSumOpNode>();
  // node->src = vmap[GetVarFromAccessPtr(args[0])];
  // node->dst = vmap[GetVarFromAccessPtr(args[1])];
  node->srcRegion_ = NormalizeToBufferRegion(args[0]);
  node->dstRegion_ = NormalizeToBufferRegion(args[1]);
  node->src = node->srcRegion_->buffer;
  node->dst = node->dstRegion_->buffer;
  node->dim = args[2].as<IntImm>().value()->value;
  node->reverse = args[3].as<Bool>().value();
  CHECK_LT(node->dim, static_cast<int>(node->src->shape.size()))
      << "The dim of cumsum should be less than the number of dimensions. Got "
         "dim="
      << node->dim << ", but src has " << node->src->shape.size() << " dims.";

  data_ = std::move(node);
}

Stmt CumSumOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  if (IsFragmentBuffer(this->src) && IsFragmentBuffer(this->dst)) {
    LOG(FATAL) << "CumSum for fragment not implemented, please raise an issue "
                  "if you need this feature.";
  } else if (this->src.scope() == "shared.dyn" ||
             this->src.scope() == "shared") {
    ICHECK(this->dst.scope() == "shared.dyn" || this->dst.scope() == "shared");
    std::stringstream ss;
    auto threads = T.thread_bounds->extent;
    Array<PrimExpr> args;

    // Build access pointers from regions locally
    PrimExpr srcPtr = MakeAccessPtrFromRegion(srcRegion_, 1);
    PrimExpr dstPtr = MakeAccessPtrFromRegion(dstRegion_, 2);

    // Use region extents instead of buffer shape for correct slice handling
    Array<PrimExpr> src_extents;
    for (const auto &range : srcRegion_->region) {
      src_extents.push_back(range->extent);
    }
    int ndim = static_cast<int>(src_extents.size());

    if (ndim == 1) {
      ICHECK_EQ(dim, 0) << "Cumulative sum over a 1D buffer only supports dim "
                           "= 0.";
      ss << "tl::CumSum1D<" << threads << ", " << (reverse ? "true" : "false")
         << ">::run";
      args = {StringImm(ss.str()), srcPtr, dstPtr, src_extents[0]};
    } else if (ndim == 2) {
      ss << "tl::CumSum2D<" << threads << ", " << dim << ", "
         << (reverse ? "true" : "false") << ">::run";
      args = {StringImm(ss.str()), srcPtr, dstPtr, src_extents[0],
              src_extents[1]};
    } else {
      LOG(FATAL) << "CumSum currently supports only 1D or 2D buffers, got "
                 << ndim << "D.";
    }
    return Evaluate(Call(dst->dtype, builtin::call_extern(), args));
  } else {
    ICHECK(false) << "Cannot lower cumsum for " << this->src.scope() << " and "
                  << this->dst.scope();
  }

  return Stmt();
}

LayoutMap CumSumOpNode::InferLayout(const LayoutInferArgs &T,
                                    InferLevel level) const {
  // Only infer layout in strict mode
  if (level != InferLevel::kStrict) {
    return {};
  }

  LayoutMap result_map;

  auto make_linear_layout = [](const Buffer &buf) -> Layout {
    return makeLinearLayout(buf->shape);
  };

  auto check_or_set_linear_layout = [&](const Buffer &buf) {
    if (!IsSharedBuffer(buf))
      return;

    Layout linear_layout = make_linear_layout(buf);
    if (T.layout_map.count(buf)) {
      // Check if existing layout is linear
      Layout existing = T.layout_map.Get(buf).value().as<Layout>().value();
      ICHECK(StructuralEqual()(existing, linear_layout))
          << "CumSum requires linear layout for shared buffer " << buf->name
          << ", but got non-linear layout.";
    } else {
      result_map.Set(buf, linear_layout);
    }
  };

  check_or_set_linear_layout(src);
  check_or_set_linear_layout(dst);

  return result_map;
}

TIR_REGISTER_TL_TILE_OP(CumSumOp, cumsum)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() {
  ReduceOpNode::RegisterReflection();
  CumSumOpNode::RegisterReflection();
  ReduceTypeNode::RegisterReflection();
}

} // namespace tl
} // namespace tvm
