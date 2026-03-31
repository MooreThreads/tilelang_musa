#pragma once

#include <mute/algorithm/clear.hpp>
#include <mute/arch/mma_mp22.hpp>
#include <mute/atom/mma_atom.hpp>
#include <mute/underscore.hpp>

#include "common.h"
#include "intrin.h"

namespace mute::tl_mma {

template <typename A_type, typename B_type, typename C_type, int num_warp_m,
          int num_warp_n, int N>
struct DispatchInstruction;

using _X = Underscore;

} // namespace mute::tl_mma

#define TL_DISPATCH_MMA(A_type, B_type, C_type, MMA_instr)                     \
  namespace mute::tl_mma {                                                     \
  template <int num_warp_m, int num_warp_n, int N>                             \
  struct DispatchInstruction<A_type, B_type, C_type, num_warp_m, num_warp_n,   \
                             N> {                                              \
    using MMA = MMA_Atom<MMA_instr>;                                           \
    using MMA_Group = Tile<_X, Int<std::min(num_warp_n * 32, N)>, _X>;         \
  };                                                                           \
  }

#ifdef __MUSA_ARCH_LIST__
#if __MUSA_ARCH_LIST__ == 220
#include <mute/arch/mma_mp22.hpp>
TL_DISPATCH_MMA(half_t, half_t, float, MP22_32x32x16_F32F16F16F32_TN)
TL_DISPATCH_MMA(bfloat16_t, bfloat16_t, float, MP22_32x32x16_F32BF16BF16F32_TN)
TL_DISPATCH_MMA(tfloat32_t, tfloat32_t, float, MP22_32x32x8_F32TF32TF32F32_TN)
TL_DISPATCH_MMA(float, float, float, MP22_32x32x8_F32TF32TF32F32_TN)
TL_DISPATCH_MMA(int8_t, int8_t, int, MP22_32x32x32_S32S8S8S32_TN)
#endif
#endif // __MUSA_ARCH_LIST__
#undef TL_DISPATCH_MMA

namespace mute::tl_mma {

template <int N, int num_warp_n, bool transpose> struct SelectCopy {
  using type = DefaultCopy;
};

template <int Bits, int N, int K, bool K_inner, int num_warp_n, int leading_dim>
struct OperandTraits {
  using Layout = typename std::conditional<
      K_inner,
      Layout<Shape<Int<N>, Int<leading_dim>>, Stride<Int<leading_dim>, _1>>,
      Layout<Shape<Int<leading_dim>, Int<K>>,
             Stride<_1, Int<leading_dim>>>>::type;
  using Copy = DefaultCopy;
};

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, int lda, int ldb, int offset_a,
          int offset_b, typename A_type_raw, typename B_type_raw,
          typename C_type_raw>
class GemmTensorOp {
public:
  using A_type_mute = typename tl::to_mute_type<A_type_raw>::type;
  using B_type_mute = typename tl::to_mute_type<B_type_raw>::type;
  using A_type =
      typename std::conditional<std::is_same<A_type_mute, float>::value,
                                tfloat32_t, A_type_mute>::type;
  using B_type =
      typename std::conditional<std::is_same<B_type_mute, float>::value,
                                tfloat32_t, B_type_mute>::type;
  using C_type = C_type_raw;

  using Instruction = DispatchInstruction<A_type_raw, B_type_raw, C_type_raw,
                                          num_warp_m, num_warp_n, N>;

  using OperandATraits = OperandTraits<sizeof_bits<A_type>::value, M, K,
                                       !trans_A, num_warp_m, lda>;

  using OperandBTraits =
      OperandTraits<sizeof_bits<B_type>::value, N, K, trans_B, num_warp_n, ldb>;

  using SmemLayoutA = typename OperandATraits::Layout;
  using SmemLayoutB = typename OperandBTraits::Layout;
  using SmemCopyA = Copy_Atom<typename OperandATraits::Copy, A_type>;
  using SmemCopyB = Copy_Atom<typename OperandBTraits::Copy, B_type>;

  using TileMma = TiledMMA<typename Instruction::MMA,
                           Layout<Shape<Int<num_warp_m>, Int<num_warp_n>, _1>>,
                           typename Instruction::MMA_Group>;

  template <class... Args>
  static MUTE_DEVICE auto remove_swizzle(Layout<Args...> const &layout) {
    return layout;
  }
  // In fp16, when layout is KxN and n_warp is 1 and N % 64 == 0
  // the original layout fail to compile, currently using this as a workaround
  template <class... Args>
  static MUTE_DEVICE auto
  remove_swizzle(ComposedLayout<Args...> const &layout) {
    if constexpr (sizeof(A_type) == 2)
      return layout.layout_b();
    else
      return layout;
  }

  template <int offset, int NN, int KK, bool trans, int lddim, typename Engine0,
            typename Layout0>
  static MUTE_DEVICE auto get_region_tensor(Tensor<Engine0, Layout0> &sa) {
    if constexpr (offset == 0) {
      using NewLayout =
          Layout<Shape<Int<NN>, Int<KK>>,
                 Stride<_1, typename std::conditional<trans, Int<NN>,
                                                      Int<lddim>>::type>>;
      auto combined_layout = composition(sa.layout(), NewLayout{});
      return make_tensor(sa.data(), combined_layout);
    } else {
      if constexpr (trans) {
        static_assert(offset % KK == 0, "Offset must be a multiple of K");
        constexpr int offset_n = offset / KK;
        return flat_divide(sa, Shape<Int<NN>, Int<KK>>{})(_, _, _0{},
                                                          Int<offset_n>{});
      } else {
        static_assert(offset % NN == 0, "Offset must be a multiple of N");
        constexpr int offset_n = offset / NN;
        return flat_divide(sa, Shape<Int<NN>, Int<KK>>{})(_, _, Int<offset_n>{},
                                                          _0{});
      }
    }
  }

  static MUTE_DEVICE void body(A_type_raw *pA, B_type_raw *pB, C_type_raw *pC) {
    const int tid = threadIdx.x;
    Tensor sA_all = make_tensor(make_smem_ptr(reinterpret_cast<A_type *>(pA)),
                                SmemLayoutA{});
    Tensor sB_all = make_tensor(make_smem_ptr(reinterpret_cast<B_type *>(pB)),
                                SmemLayoutB{});
    Tensor sA = get_region_tensor<offset_a, M, K, !trans_A, lda>(sA_all);
    Tensor sB = get_region_tensor<offset_b, N, K, trans_B, ldb>(sB_all);
    TileMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    auto tiled_copy_A = make_tiled_copy_A(SmemCopyA{}, tiled_mma);
    auto tiled_copy_B = make_tiled_copy_B(SmemCopyB{}, tiled_mma);
    auto thr_copy_A = tiled_copy_A.get_thread_slice(tid);
    auto thr_copy_B = tiled_copy_B.get_thread_slice(tid);

    Tensor tCrA = thr_mma.partition_fragment_A(sA);
    Tensor tCrB = thr_mma.partition_fragment_B(sB);
    Tensor tCsA = thr_copy_A.partition_S(sA);
    Tensor tCsB = thr_copy_B.partition_S(sB);

    Tensor tCrA_copy_view = thr_copy_A.retile_D(tCrA);
    Tensor tCrB_copy_view = thr_copy_B.retile_D(tCrB);

    Tensor acc =
        make_tensor(make_rmem_ptr(reinterpret_cast<C_type *>(pC)),
                    partition_shape_C(tiled_mma, Shape<Int<M>, Int<N>>{}));

    // when layout is KxN and n_warp is 1, there seem to be a bug, use this as a
    // workaround
    auto tCrA_view = make_tensor(tCrA.data(), remove_swizzle(tCrA.layout()));
    auto tCrB_view = make_tensor(tCrB.data(), remove_swizzle(tCrB.layout()));
    if constexpr (clear_accum) {
      clear(acc);
    }
    MUTE_UNROLL
    for (int k = 0; k < size<2>(tCrA); ++k) {
      copy(tiled_copy_A, tCsA(_, _, k), tCrA_copy_view(_, _, k));
      copy(tiled_copy_B, tCsB(_, _, k), tCrB_copy_view(_, _, k));
      gemm(tiled_mma, tCrA_view(_, _, k), tCrB_view(_, _, k), acc);
    }
  }

  static MUTE_DEVICE void body_rs(A_type_raw *pA, B_type_raw *pB,
                                  C_type_raw *pC) {
    const int tid = threadIdx.x;
    Tensor sB_all = make_tensor(make_smem_ptr(reinterpret_cast<B_type *>(pB)),
                                SmemLayoutB{});
    Tensor sB = get_region_tensor<offset_b, N, K, trans_B, ldb>(sB_all);
    TileMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    auto tiled_copy_B = make_tiled_copy_B(SmemCopyB{}, tiled_mma);
    auto thr_copy_B = tiled_copy_B.get_thread_slice(tid);

    Tensor tCrB = thr_mma.partition_fragment_B(sB);
    Tensor tCsB = thr_copy_B.partition_S(sB);

    Tensor tCrB_copy_view = thr_copy_B.retile_D(tCrB);

    Tensor acc =
        make_tensor(make_rmem_ptr(reinterpret_cast<C_type *>(pC)),
                    partition_shape_C(tiled_mma, Shape<Int<M>, Int<N>>{}));
    Tensor tCrA =
        make_tensor(make_rmem_ptr(reinterpret_cast<A_type *>(pA)),
                    partition_shape_A(tiled_mma, Shape<Int<M>, Int<K>>{}));
    auto tCrB_view = make_tensor(tCrB.data(), remove_swizzle(tCrB.layout()));
    if constexpr (clear_accum) {
      clear(acc);
    }
    copy(tiled_copy_B, tCsB(_, _, 0), tCrB_copy_view(_, _, 0));
    MUTE_UNROLL
    for (int k = 0; k < size<2>(tCrA); ++k) {
      if (k < size<2>(tCrA) - 1) {
        copy(tiled_copy_B, tCsB(_, _, k + 1), tCrB_copy_view(_, _, k + 1));
      }
      gemm(tiled_mma, tCrA(_, _, k), tCrB_view(_, _, k), acc);
    }
  }

  static MUTE_DEVICE void body_sr(A_type_raw *pA, B_type_raw *pB,
                                  C_type_raw *pC) {
    const int tid = threadIdx.x;
    Tensor sA_all = make_tensor(make_smem_ptr(reinterpret_cast<A_type *>(pA)),
                                SmemLayoutA{});
    Tensor sA = get_region_tensor<offset_a, M, K, !trans_A, lda>(sA_all);
    TileMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    auto tiled_copy_A = make_tiled_copy_A(SmemCopyA{}, tiled_mma);
    auto thr_copy_A = tiled_copy_A.get_thread_slice(tid);

    Tensor tCrA = thr_mma.partition_fragment_A(sA);
    Tensor tCsA = thr_copy_A.partition_S(sA);

    Tensor tCrA_copy_view = thr_copy_A.retile_D(tCrA);

    Tensor acc =
        make_tensor(make_rmem_ptr(reinterpret_cast<C_type *>(pC)),
                    partition_shape_C(tiled_mma, Shape<Int<M>, Int<N>>{}));
    Tensor tCrB =
        make_tensor(make_rmem_ptr(reinterpret_cast<B_type *>(pB)),
                    partition_shape_B(tiled_mma, Shape<Int<N>, Int<K>>{}));
    auto tCrA_view = make_tensor(tCrA.data(), remove_swizzle(tCrA.layout()));
    if constexpr (clear_accum) {
      clear(acc);
    }
    copy(tiled_copy_A, tCsA(_, _, 0), tCrA_copy_view(_, _, 0));
    MUTE_UNROLL
    for (int k = 0; k < size<2>(tCrA); ++k) {
      if (k < size<2>(tCrA) - 1) {
        copy(tiled_copy_A, tCsA(_, _, k + 1), tCrA_copy_view(_, _, k + 1));
      }
      gemm(tiled_mma, tCrA_view(_, _, k), tCrB(_, _, k), acc);
    }
  }
};

} // namespace mute::tl_mma

namespace tl::tl_mma {

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, int lda, int ldb, int offset_a,
          int offset_b, typename A_type, typename B_type, typename C_type>
MUTLASS_DEVICE void gemm_ss(A_type *pA, B_type *pB, C_type *accum) {
  using MMA =
      mute::tl_mma::GemmTensorOp<M, N, K, num_warp_m, num_warp_n, trans_A,
                                 trans_B, clear_accum, lda, ldb, offset_a,
                                 offset_b, A_type, B_type, C_type>;
  MMA::body(pA, pB, accum);
}

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, int lda, int ldb, int offset_a,
          int offset_b, typename A_type, typename B_type, typename C_type>
MUTLASS_DEVICE void gemm_rs(A_type *pA, B_type *pB, C_type *accum) {
  using MMA =
      mute::tl_mma::GemmTensorOp<M, N, K, num_warp_m, num_warp_n, trans_A,
                                 trans_B, clear_accum, lda, ldb, offset_a,
                                 offset_b, A_type, B_type, C_type>;
  MMA::body_rs(pA, pB, accum);
}

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, int lda, int ldb, int offset_a,
          int offset_b, typename A_type, typename B_type, typename C_type>
MUTLASS_DEVICE void gemm_sr(A_type *pA, B_type *pB, C_type *accum) {
  using MMA =
      mute::tl_mma::GemmTensorOp<M, N, K, num_warp_m, num_warp_n, trans_A,
                                 trans_B, clear_accum, lda, ldb, offset_a,
                                 offset_b, A_type, B_type, C_type>;
  MMA::body_sr(pA, pB, accum);
}

} // namespace tl::tl_mma