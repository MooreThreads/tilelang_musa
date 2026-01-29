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

#include "../layout/utils.h"
#include "../op/parallel.h"
#include "../target/utils.h"
#include "../transform/loop_partition.h"
#include "builtin.h"
#include "tir/transforms/ir_utils.h"
#include <tvm/tir/transform.h>

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
  auto uses_var = [](const PrimExpr &expr, const Var &var) {
    return UsesVar(expr, [&](const VarNode *v) { return v == var.get(); });
  };
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

ReduceOp::ReduceOp(Array<PrimExpr> args, BufferMap vmap) {
  ObjectPtr<ReduceOpNode> node = tvm::ffi::make_object<ReduceOpNode>();
  node->src = vmap[GetVarFromAccessPtr(args[0])];
  node->dst = vmap[GetVarFromAccessPtr(args[1])];
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

PrimExpr ReduceOpNode::MakeReduce(const PrimExpr &lhs,
                                  const PrimExpr &b) const {
  PrimExpr rhs = b;
  if (lhs->dtype != rhs->dtype) {
    rhs = Cast(lhs->dtype, rhs);
  }
  if (type->isSum()) {
    return lhs + rhs;
  } else if (type->isAbsSum()) {
    return lhs + Max(rhs, -rhs);
  } else if (type->isMax()) {
    return Max(lhs, rhs);
  } else if (type->isMin()) {
    return Min(lhs, rhs);
  } else if (type->isAbsMax()) {
    return Max(tvm::abs(lhs), tvm::abs(rhs));
  } else if (type->isBitAnd()) {
    return lhs & rhs;
  } else if (type->isBitOr()) {
    return lhs | rhs;
  } else if (type->isBitXor()) {
    return lhs ^ rhs;
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

/**
 * @brief Lower the Reduce operator to a TIR statement.
 *
 * Lowers a ReduceOpNode operating on fragment-scoped buffers into a sequence of
 * TIR statements implementing: optional initialization, thread-local reduction
 * (unrolled inner loops), inter-thread reduction via a runtime AllReduce call
 * (Hopper-specific `run_hopper` variant when TargetIsHopper(T.target) is true),
 * and an optional accumulation or copy back to the destination buffer when a
 * temporary clear buffer is used.
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
 *   emits a call to a templated `tl::AllReduce<...>::run` (or `run_hopper`)
 *   via `builtin::call_extern`. For sufficiently large reducing thread counts
 *   (>= 32) a workspace is allocated via T.AddWorkspace and passed to the
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
    Buffer src_buffer = get_buffer(this->src);
    Buffer dst_buffer = get_buffer(this->dst);
    Fragment src_layout = T.layout_map[this->src].as<Fragment>().value();
    Fragment dst_layout = T.layout_map[this->dst].as<Fragment>().value();
    size_t src_dim = src_layout->InputDim();
    size_t dst_dim = dst_layout->InputDim();

    bool is_1d_reduce = src_dim == dst_dim && dst_dim == 1;

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

    Array<PrimExpr> src_indices = src_layout->Forward(
        src_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }));
    Array<PrimExpr> dst_indices = dst_layout->Forward(
        dst_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }));

    Array<Stmt> stmts;
    Array<Stmt> stmts_after_loop;

    bool require_init = this->clear;
    if (this->type->isSum() || this->type->isAbsSum() ||
        this->type->isBitAnd() || this->type->isBitOr() ||
        this->type->isBitXor()) {
      require_init = true;
    }

    Buffer clear_buffer = dst_buffer;
    bool need_duplicate = false;
    if ((this->type->isSum() || this->type->isAbsSum()) && !this->clear) {
      need_duplicate = true;
    } else if (this->type->isBitAnd() && !this->clear) {
      need_duplicate = true;
    } else if ((this->type->isBitOr() || this->type->isBitXor()) &&
               !this->clear) {
      need_duplicate = true;
    }

    if (need_duplicate) {
      // Create a new buffer with same shape and dtype as dst_buffer
      clear_buffer = decl_buffer(dst_buffer->shape, dst_buffer->dtype,
                                 dst_buffer->name + "_clear",
                                 GetPtrStorageScope(dst_buffer->data));
    }
    // make reduce-init stmt
    if (require_init) {
      stmts.push_back(
          BufferStore(clear_buffer, this->MakeInitValue(), dst_indices));
    }

    // make thread-local reduce
    Array<PrimExpr> src_indice_compressed;
    Array<IterVar> src_var_compressed;
    for (size_t i = 0; i < src_layout->OutputDim(); ++i) {
      PrimExpr expr;
      IterVar var;
      std::tie(expr, var) = CompressIterator(
          src_indices[i], src_vars, src_vars[this->dim]->var, analyzer);
      src_indice_compressed.push_back(expr);
      src_var_compressed.push_back(var);
    }

    tvm::transform::PassContext pass_ctx =
        tvm::transform::PassContext::Current();
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

      if (require_init && simd_plan.groups == 1) {
        stmts.pop_back();
        reduce_local = BufferStore(clear_buffer, group_reduce_var, dst_indices);
      } else {
        reduce_local =
            BufferStore(clear_buffer,
                        this->MakeReduce(BufferLoad(clear_buffer, dst_indices),
                                         group_reduce_var),
                        dst_indices);
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
          this->MakeReduce(BufferLoad(clear_buffer, dst_indices),
                           BufferLoad(src_buffer, src_indice_compressed)),
          dst_indices);
      for (int i = static_cast<int>(src_layout->OutputDim()) - 1; i >= 0; --i) {
        reduce_local = For(src_var_compressed[i]->var, 0,
                           src_var_compressed[i]->dom->extent,
                           ForKind::kUnrolled, reduce_local, std::nullopt,
                           {{tir::attr::pragma_unroll_explicit, Bool(false)}});
      }
    }
    stmts.push_back(reduce_local);

    PrimExpr src_thread = src_layout->ForwardThread(
        src_vars.Map([](const auto &iv) { return PrimExpr(iv->var); }), {});
    auto iter_sum =
        arith::NormalizeToIterSum(src_thread, ToVMap(src_vars), analyzer);

    MusaInterThreadSimdPlan simd_allreduce_plan;
    if (enable_reduce_burst) {
      simd_allreduce_plan =
          PlanMusaInterThreadSimd(T.target, this->type, src_buffer, dst_buffer,
                                  dst_indices, dst_vars, dst_layout);
    }

    bool simd_allreduce_emitted = false;
    for (const auto &iter_split : iter_sum->args) {
      auto mark = iter_split->source->source.as<Var>();
      ICHECK(mark) << "Not a normalized iterator: " << iter_split->source;
      if (mark.value().same_as(src_vars[this->dim]->var)) { // reduce axis
        auto scale = as_const_int(iter_split->scale);
        auto extent = as_const_int(iter_split->extent);
        ICHECK(scale != nullptr && extent != nullptr);
        if (*extent == 1)
          continue;
        int reducing_threads = (*extent) * (*scale);

        // fast path (use simd op)
        if (simd_allreduce_plan.enabled && !simd_allreduce_emitted &&
            reducing_threads <= 32) {
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
          PrimExpr vec = BufferLoad(
              clear_buffer, {Ramp(dv_base, make_const(dv_base.dtype(), 1), 4)});

          int steps = 0;
          for (int t = reducing_threads; t > *scale; t >>= 1) {
            ++steps;
          }
          ICHECK(steps > 0);

          // iter var [0, steps)
          Var offset_iter("offset_step", kInt32);
          // thread offset expr
          PrimExpr offset_expr =
              right_shift(make_const(offset_iter.dtype(), reducing_threads),
                          offset_iter + make_const(offset_iter.dtype(), 1));
          // thread offset var
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
            StringImm(ss.str()), BufferLoad(clear_buffer, dst_indices)};
        if (use_musa_sync) {
          PrimExpr barrier_id =
              BufferLoad(reduce_sync_barrier, {IntImm(DataType::Int(32), 0)});
          thread_reduce_args.push_back(barrier_id);
        }
        if (reducing_threads >= 32) {
          PrimExpr workspace = T.AddWorkspace(
              *as_const_int(T.thread_bounds->extent), clear_buffer->dtype);
          thread_reduce_args.push_back(workspace);
        }
        auto call = Call(clear_buffer->dtype, builtin::call_extern(),
                         thread_reduce_args);
        stmts.push_back(BufferStore(clear_buffer, call, dst_indices));
      }
    }

    if (need_duplicate) {
      PrimExpr src_val = BufferLoad(clear_buffer, dst_indices);
      PrimExpr dst_val = BufferLoad(dst_buffer, dst_indices);
      PrimExpr update;
      if (this->type->isSum() || this->type->isAbsSum()) {
        update = dst_val + src_val;
      } else if (this->type->isBitAnd()) {
        update = this->clear ? src_val : bitwise_and(dst_val, src_val);
      } else if (this->type->isBitOr()) {
        update = bitwise_or(dst_val, src_val);
      } else if (this->type->isBitXor()) {
        update = bitwise_xor(dst_val, src_val);
      } else {
        LOG(FATAL) << "Unsupported reduce type: " << this->type->type;
      }
      stmts.push_back(BufferStore(dst_buffer, update, dst_indices));
    }

    Stmt body = stmts.size() > 1 ? SeqStmt(stmts) : stmts[0];
    for (int i = static_cast<int>(dst_layout->InputDim()) - 1; i >= 0; --i) {
      body = For(dst_vars[i]->var, 0, dst_vars[i]->dom->extent,
                 ForKind::kParallel, body);
    }

    if (dst_layout->InputDim() > 0) {
      body = PartitionLoop(Downcast<For>(body), T.thread_var, analyzer,
                           dst_layout);
    } else {
      PrimExpr guard = (T.thread_var == T.thread_bounds->min);
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
  if (src.scope() == "local.fragment" && dst.scope() == "local.fragment" &&
      T.layout_map.count(src)) {
    auto src_layout = T.layout_map[src].as<Fragment>().value();

    PrimExpr indice_rep_extent = src->shape[dim];
    PrimExpr src_rep_extent = src_layout->ReplicateExtent();
    PrimExpr dest_buffer_rep_extent = indice_rep_extent * src_rep_extent;

    Array<PrimExpr> fwd;
    for (int i = 0; i < static_cast<int>(src->shape.size()); i++) {
      if (i == dim) {
        fwd.push_back(FloorMod(ReplicationPlaceholder(), indice_rep_extent));
      } else if (i < dim) {
        fwd.push_back(InputPlaceholder(i));
      } else if (i > dim) {
        fwd.push_back(InputPlaceholder(i - 1));
      }
    }
    auto thd = src_layout->ForwardThread(
        fwd, FloorDiv(ReplicationPlaceholder(), indice_rep_extent));
    Fragment dst_layout =
        Fragment(dst->shape, {}, thd, dest_buffer_rep_extent, std::nullopt)
            ->CondenseReplicateVar()
            ->BindThreadRange(T.thread_bounds);
    if (!T.layout_map.count(dst))
      return {{dst, dst_layout}};
    else {
      // Check if computed layout is compatible with existing: the existing one
      // must strictly contains the computed layout
      auto orig_dst_layout =
          T.layout_map.Get(dst).value().as<Fragment>().value();
      ICHECK(dst_layout->InputDim() == orig_dst_layout->InputDim());
      Array<PrimExpr> indices;
      indices.reserve(dst_layout->InputDim());
      arith::Analyzer inner_analyzer;
      for (int i = 0; i < dst_layout->InputDim(); ++i) {
        auto x = InputPlaceholder(i);
        indices.push_back(x);
        // should be literal - literal = 0, any analyzer will work
        ICHECK(is_zero(inner_analyzer.Simplify(
            dst_layout->InputShape()[i] - orig_dst_layout->InputShape()[i])));
        inner_analyzer.Bind(x, Range(0, dst_layout->InputShape()[i]));
      }

      ICHECK(as_const_int(dst_layout->ReplicateExtent()));
      ICHECK(as_const_int(src_layout->ReplicateExtent()));
      auto dst_rep = *as_const_int(dst_layout->ReplicateExtent());
      auto src_rep = *as_const_int(src_layout->ReplicateExtent());
      if (dst_rep < src_rep ||
          !ProveFragmentContains(orig_dst_layout, dst_layout, indices, indices,
                                 inner_analyzer)) {
        std::ostringstream oss;
        oss << "Layout may conflict with ReduceOp for buffer " << dst << " vs. "
            << src << "\nLHS = " << src_layout->DebugOutput()
            << "\nRHS = " << orig_dst_layout->DebugOutput()
            << "\nYou may need to use a shared memory to transform the "
               "layout";
        throw LayoutConflictException(oss.str());
      }

      if (dst_rep > src_rep) {
        return {{dst, dst_layout}};
      }
    }
  }
  return {};
}

TIR_REGISTER_TL_OP(ReduceOp, reduce)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

CumSumOp::CumSumOp(Array<PrimExpr> args, BufferMap vmap) {
  /// CumSum constructor arguments:
  /// - src: input buffer
  /// - dst: output buffer
  /// - dim: dimension to cumsum
  /// - reverse: whether to cumsum in reverse order
  CHECK_EQ(args.size(), 4);
  ObjectPtr<CumSumOpNode> node = tvm::ffi::make_object<CumSumOpNode>();
  node->src = vmap[GetVarFromAccessPtr(args[0])];
  node->dst = vmap[GetVarFromAccessPtr(args[1])];
  node->dim = args[2].as<IntImm>().value()->value;
  node->reverse = args[3].as<Bool>().value();
  CHECK_LT(node->dim, static_cast<int>(node->src->shape.size()));
  data_ = std::move(node);
}

Stmt CumSumOpNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  if (this->src.scope() == "local.fragment" &&
      this->dst.scope() == "local.fragment") {
    LOG(FATAL) << "CumSum for fragment not implemented, please raise an issue "
                  "if you need this feature.";
  } else if (this->src.scope() == "shared.dyn" ||
             this->src.scope() == "shared") {
    ICHECK(this->dst.scope() == "shared.dyn" || this->dst.scope() == "shared");
    std::stringstream ss;
    auto threads = T.thread_bounds->extent;
    Array<PrimExpr> args;
    int ndim = static_cast<int>(src->shape.size());
    if (ndim == 1) {
      ICHECK_EQ(dim, 0) << "Cumulative sum over a 1D buffer only supports dim "
                           "= 0.";
      ss << "tl::CumSum1D<" << threads << ", " << (reverse ? "true" : "false")
         << ">::run";
      args = {StringImm(ss.str()), src.access_ptr(1), dst.access_ptr(3),
              src->shape[0]};
    } else if (ndim == 2) {
      ss << "tl::CumSum2D<" << threads << ", " << dim << ", "
         << (reverse ? "true" : "false") << ">::run";
      args = {StringImm(ss.str()), src.access_ptr(1), dst.access_ptr(3),
              src->shape[0], src->shape[1]};
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
  return {};
}

TIR_REGISTER_TL_OP(CumSumOp, cumsum)
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
