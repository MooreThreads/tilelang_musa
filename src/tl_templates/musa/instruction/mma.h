#pragma once

#include "../common.h"
#include <cstdint>
#include <mute/arch/mma_mp31.hpp>
#include <type_traits>

namespace tl {

#ifndef TL_ALWAYS_FALSE_V_DEFINED
#define TL_ALWAYS_FALSE_V_DEFINED
template <class> inline constexpr bool always_false_v = false;
#endif

namespace detail {

template <class To, class From>
TL_DEVICE To bit_cast(From value) {
  static_assert(sizeof(To) == sizeof(From), "bit_cast size mismatch");
  union {
    From from;
    To to;
  } u{value};
  return u.to;
}

TL_DEVICE float unpack_half_from_u16(uint16_t bits) {
  union {
    uint16_t u;
    __half h;
  } raw{bits};
  return __half2float(raw.h);
}

TL_DEVICE uint16_t pack_half_to_u16(float value) {
  union {
    __half h;
    uint16_t u;
  } raw{__float2half(value)};
  return raw.u;
}

template <class Impl> struct MmaImplTraits {
  using DReg = std::remove_extent_t<typename Impl::DRegisters>;
  using AReg = std::remove_extent_t<typename Impl::ARegisters>;
  using BReg = std::remove_extent_t<typename Impl::BRegisters>;
  using CReg = std::remove_extent_t<typename Impl::CRegisters>;
};

template <DataType AType, DataType BType, DataType CType, int M, int N, int K,
          bool TransA, bool TransB, bool Saturate>
struct MmaDispatcher {
  using CRegType = void;
  using ARegType = void;
  using BRegType = void;

  static TL_DEVICE void exec(CRegType *, const ARegType *, const BRegType *,
                             const CRegType *) {
    static_assert(always_false_v<std::integral_constant<int, M>>,
                  "tl::mma_sync: unsupported configuration on MUSA");
  }
};

#define TL_MUSA_MAJOR_FROM_TRANS(TransValue)                                   \
  ((TransValue) ? mute::TCE::Major::MN : mute::TCE::Major::K)

#define TL_MUSA_MP31(ImplName, TransAValue, TransBValue)                       \
  mute::ImplName<TL_MUSA_MAJOR_FROM_TRANS(TransAValue),                        \
                 TL_MUSA_MAJOR_FROM_TRANS(TransBValue)>

#define TL_MUSA_DEFINE_MMA_DISPATCHER(ATypeEnum, BTypeEnum, CTypeEnum, MValue, \
                                      NValue, KValue, TransAValue,             \
                                      TransBValue, SaturateValue, ImplType)    \
  template <>                                                                   \
  struct MmaDispatcher<DataType::ATypeEnum, DataType::BTypeEnum,               \
                       DataType::CTypeEnum, MValue, NValue, KValue,            \
                       TransAValue, TransBValue, SaturateValue> {              \
    using Impl = ImplType;                                                      \
    using Traits = MmaImplTraits<Impl>;                                         \
    using CRegType = typename Traits::DReg;                                     \
    using ARegType = typename Traits::AReg;                                     \
    using BRegType = typename Traits::BReg;                                     \
    static_assert(                                                              \
        std::is_same_v<typename Traits::DReg, typename Traits::CReg>,           \
        "tl::mma_sync requires matching accumulator/output regs");              \
    static TL_DEVICE void exec(CRegType *d, const ARegType *a,                 \
                               const BRegType *b, const CRegType *c) {         \
      Impl::fma(reinterpret_cast<int32_t *>(d),                                \
                reinterpret_cast<const int32_t *>(a),                          \
                reinterpret_cast<const int32_t *>(b),                          \
                reinterpret_cast<const int32_t *>(c));                         \
    }                                                                           \
  };

#define TL_MUSA_DEFINE_MMA_LAYOUT4(ATypeEnum, BTypeEnum, CTypeEnum, MValue,    \
                                   NValue, KValue, ImplName)                   \
  TL_MUSA_DEFINE_MMA_DISPATCHER(                                                \
      ATypeEnum, BTypeEnum, CTypeEnum, MValue, NValue, KValue, false, false,   \
      false, TL_MUSA_MP31(ImplName, false, false))                              \
  TL_MUSA_DEFINE_MMA_DISPATCHER(                                                \
      ATypeEnum, BTypeEnum, CTypeEnum, MValue, NValue, KValue, false, true,    \
      false, TL_MUSA_MP31(ImplName, false, true))                               \
  TL_MUSA_DEFINE_MMA_DISPATCHER(                                                \
      ATypeEnum, BTypeEnum, CTypeEnum, MValue, NValue, KValue, true, false,    \
      false, TL_MUSA_MP31(ImplName, true, false))                               \
  TL_MUSA_DEFINE_MMA_DISPATCHER(                                                \
      ATypeEnum, BTypeEnum, CTypeEnum, MValue, NValue, KValue, true, true,     \
      false, TL_MUSA_MP31(ImplName, true, true))

// Native m16n8k{4,8,16} MP31 coverage.
TL_MUSA_DEFINE_MMA_LAYOUT4(kFloat16, kFloat16, kFloat32, 16, 8, 8,
                           MP31_16x8x8_F32F16F16F32)
TL_MUSA_DEFINE_MMA_LAYOUT4(kFloat16, kFloat16, kFloat32, 16, 8, 16,
                           MP31_16x8x16_F32F16F16F32)
TL_MUSA_DEFINE_MMA_LAYOUT4(kBFloat16, kBFloat16, kFloat32, 16, 8, 8,
                           MP31_16x8x8_F32BF16BF16F32)
TL_MUSA_DEFINE_MMA_LAYOUT4(kBFloat16, kBFloat16, kFloat32, 16, 8, 16,
                           MP31_16x8x16_F32BF16BF16F32)
TL_MUSA_DEFINE_MMA_LAYOUT4(kInt8, kInt8, kInt32, 16, 8, 16,
                           MP31_16x8x16_S32S8S8S32)
TL_MUSA_DEFINE_MMA_LAYOUT4(kFloat8_e4m3, kFloat8_e4m3, kFloat32, 16, 8, 16,
                           MP31_16x8x16_F32E4M3E4M3F32)
TL_MUSA_DEFINE_MMA_LAYOUT4(kFloat8_e5m2, kFloat8_e5m2, kFloat32, 16, 8, 16,
                           MP31_16x8x16_F32E5M2E5M2F32)
TL_MUSA_DEFINE_MMA_LAYOUT4(kFloat8_e4m3, kFloat8_e5m2, kFloat32, 16, 8, 16,
                           MP31_16x8x16_F32E4M3E5M2F32)
TL_MUSA_DEFINE_MMA_LAYOUT4(kFloat8_e5m2, kFloat8_e4m3, kFloat32, 16, 8, 16,
                           MP31_16x8x16_F32E5M2E4M3F32)
TL_MUSA_DEFINE_MMA_LAYOUT4(kTensorFloat32, kTensorFloat32, kFloat32, 16, 8, 4,
                           MP31_16x8x4_F32TF32TF32F32)
TL_MUSA_DEFINE_MMA_LAYOUT4(kTensorFloat32, kTensorFloat32, kFloat32, 16, 8, 8,
                           MP31_16x8x8_F32TF32TF32F32)

// MUSA lacks m16n8k32 forms for several dtypes; compose from two k16 ops.
template <bool TransA, bool TransB>
struct MmaDispatcher<DataType::kInt8, DataType::kInt8, DataType::kInt32, 16, 8,
                     32, TransA, TransB, false> {
  using CRegType = uint32_t;
  using ARegType = uint32_t;
  using BRegType = uint32_t;

  static TL_DEVICE void exec(CRegType *d, const ARegType *a, const BRegType *b,
                             const CRegType *c) {
    using ImplK16 = TL_MUSA_MP31(MP31_16x8x16_S32S8S8S32, TransA, TransB);
    int32_t accum[4];
    for (int i = 0; i < 4; ++i) {
      accum[i] = static_cast<int32_t>(c[i]);
    }
    auto a_i32 = reinterpret_cast<const int32_t *>(a);
    auto b_i32 = reinterpret_cast<const int32_t *>(b);
    ImplK16::fma(accum, a_i32, b_i32, accum);
    ImplK16::fma(accum, a_i32 + 2, b_i32 + 1, accum);
    for (int i = 0; i < 4; ++i) {
      d[i] = static_cast<uint32_t>(accum[i]);
    }
  }
};

template <DataType AType, DataType BType, bool TransA, bool TransB>
struct MmaDispatcherFp8K32;

#define TL_MUSA_DEFINE_MMA_FP8_K32_DISPATCHER(ATypeEnum, BTypeEnum, ImplName)  \
  template <bool TransA, bool TransB>                                           \
  struct MmaDispatcherFp8K32<DataType::ATypeEnum, DataType::BTypeEnum, TransA, \
                             TransB> {                                           \
    static TL_DEVICE void exec(float *d, const uint32_t *a, const uint32_t *b, \
                               const float *c) {                                 \
      using ImplK16 = TL_MUSA_MP31(ImplName, TransA, TransB);                   \
      int32_t accum[4];                                                          \
      for (int i = 0; i < 4; ++i) {                                              \
        accum[i] = bit_cast<int32_t>(c[i]);                                      \
      }                                                                          \
      auto a_i32 = reinterpret_cast<const int32_t *>(a);                         \
      auto b_i32 = reinterpret_cast<const int32_t *>(b);                         \
      ImplK16::fma(accum, a_i32, b_i32, accum);                                  \
      ImplK16::fma(accum, a_i32 + 2, b_i32 + 1, accum);                          \
      for (int i = 0; i < 4; ++i) {                                              \
        d[i] = bit_cast<float>(accum[i]);                                        \
      }                                                                          \
    }                                                                            \
  };

TL_MUSA_DEFINE_MMA_FP8_K32_DISPATCHER(kFloat8_e4m3, kFloat8_e4m3,
                                      MP31_16x8x16_F32E4M3E4M3F32)
TL_MUSA_DEFINE_MMA_FP8_K32_DISPATCHER(kFloat8_e5m2, kFloat8_e5m2,
                                      MP31_16x8x16_F32E5M2E5M2F32)
TL_MUSA_DEFINE_MMA_FP8_K32_DISPATCHER(kFloat8_e4m3, kFloat8_e5m2,
                                      MP31_16x8x16_F32E4M3E5M2F32)
TL_MUSA_DEFINE_MMA_FP8_K32_DISPATCHER(kFloat8_e5m2, kFloat8_e4m3,
                                      MP31_16x8x16_F32E5M2E4M3F32)

template <DataType AType, DataType BType, bool TransA, bool TransB>
struct MmaDispatcher<AType, BType, DataType::kFloat32, 16, 8, 32, TransA,
                     TransB, false> {
  using CRegType = float;
  using ARegType = uint32_t;
  using BRegType = uint32_t;

  static TL_DEVICE void exec(CRegType *d, const ARegType *a, const BRegType *b,
                             const CRegType *c) {
    MmaDispatcherFp8K32<AType, BType, TransA, TransB>::exec(d, a, b, c);
  }
};

// Emulate fp16 accumulation via f32 accumulator and round-to-half pack.
template <bool TransA, bool TransB>
struct MmaDispatcher<DataType::kFloat16, DataType::kFloat16, DataType::kFloat16,
                     16, 8, 16, TransA, TransB, false> {
  using CRegType = uint32_t;
  using ARegType = uint32_t;
  using BRegType = uint32_t;

  static TL_DEVICE void exec(CRegType *d, const ARegType *a, const BRegType *b,
                             const CRegType *c) {
    using Impl = TL_MUSA_MP31(MP31_16x8x16_F32F16F16F32, TransA, TransB);
    int32_t accum[4];
    accum[0] = bit_cast<int32_t>(
        unpack_half_from_u16(static_cast<uint16_t>(c[0] & 0xffffu)));
    accum[1] = bit_cast<int32_t>(
        unpack_half_from_u16(static_cast<uint16_t>((c[0] >> 16) & 0xffffu)));
    accum[2] = bit_cast<int32_t>(
        unpack_half_from_u16(static_cast<uint16_t>(c[1] & 0xffffu)));
    accum[3] = bit_cast<int32_t>(
        unpack_half_from_u16(static_cast<uint16_t>((c[1] >> 16) & 0xffffu)));

    Impl::fma(accum, reinterpret_cast<const int32_t *>(a),
              reinterpret_cast<const int32_t *>(b), accum);

    uint16_t h0 = pack_half_to_u16(bit_cast<float>(accum[0]));
    uint16_t h1 = pack_half_to_u16(bit_cast<float>(accum[1]));
    uint16_t h2 = pack_half_to_u16(bit_cast<float>(accum[2]));
    uint16_t h3 = pack_half_to_u16(bit_cast<float>(accum[3]));
    d[0] = static_cast<uint32_t>(h0) | (static_cast<uint32_t>(h1) << 16);
    d[1] = static_cast<uint32_t>(h2) | (static_cast<uint32_t>(h3) << 16);
  }
};

#undef TL_MUSA_DEFINE_MMA_FP8_K32_DISPATCHER
#undef TL_MUSA_DEFINE_MMA_LAYOUT4
#undef TL_MUSA_DEFINE_MMA_DISPATCHER
#undef TL_MUSA_MP31
#undef TL_MUSA_MAJOR_FROM_TRANS

} // namespace detail

template <DataType AType, DataType BType, DataType CType, int M, int N, int K,
          bool TransA, bool TransB, bool Saturate = false>
TL_DEVICE void mma_sync(
    typename detail::MmaDispatcher<AType, BType, CType, M, N, K, TransA, TransB,
                                   Saturate>::CRegType *c,
    const typename detail::MmaDispatcher<AType, BType, CType, M, N, K, TransA,
                                         TransB, Saturate>::ARegType *a,
    const typename detail::MmaDispatcher<AType, BType, CType, M, N, K, TransA,
                                         TransB, Saturate>::BRegType *b) {
  using Dispatcher = detail::MmaDispatcher<AType, BType, CType, M, N, K, TransA,
                                           TransB, Saturate>;
  static_assert(!std::is_void_v<typename Dispatcher::CRegType>,
                "tl::mma_sync: unsupported configuration on MUSA");
  Dispatcher::exec(c, a, b, c);
}

} // namespace tl
