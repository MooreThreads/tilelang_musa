#pragma once

#include "atomic.h"
#include <cstdio> // snprintf
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <mute/arch/simd_mp31.hpp>
#include <mutlass/fast_math.h>
#include <mutlass/numeric_types.h>
#include <sstream> // std::stringstream

using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

using mutlass::bfloat16_t;
using mutlass::half_t;
using mutlass::tfloat32_t;
using int4_t = int4;

using v2i32_t = int32_t __attribute__((vector_size(8)));
using v3i32_t = int32_t __attribute__((vector_size(12)));
using v4i32_t = int32_t __attribute__((vector_size(16)));
using v5i32_t = int32_t __attribute__((vector_size(20)));
using v8i32_t = int32_t __attribute__((vector_size(32)));
using v16i32_t = int32_t __attribute__((vector_size(64)));
using v32i32_t = int32_t __attribute__((vector_size(128)));

#define __log2 log2
#define __log2f log2f

#define hexp mutlass::fast_exp
#define hlog mutlass::fast_log
#define hsqrt mutlass::fast_sqrt
#define hsin mutlass::fast_sin
#define hcos mutlass::fast_cos
#define htanh mutlass::fast_tanh
#define hpow powf

#define TL_HOST_DEVICE __forceinline__ __host__ __device__
#define TL_DEVICE __forceinline__ __device__
#define TL_DEVICE_NOINLINE __noinline__ __device__
#define TL_PATCH

#define _AS1 __attribute__((address_space(1)))
#define _AS3 __attribute__((address_space(3)))

#define TILELANG_CHECK(stmt)                                                   \
  do {                                                                         \
    musaError_t __err = (stmt);                                                \
    if (__err != musaSuccess) {                                                \
      snprintf(error_buf, ERROR_BUF_SIZE, "%s:%d: %s - %s", __FILE__,          \
               __LINE__, musaGetErrorName(__err), musaGetErrorString(__err));  \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define TILELANG_CHECK_LAST_ERROR(kernel_name)                                 \
  do {                                                                         \
    musaError_t __err = musaGetLastError();                                    \
    if (__err != musaSuccess) {                                                \
      snprintf(error_buf, ERROR_BUF_SIZE, kernel_name ": %s - %s",             \
               musaGetErrorName(__err), musaGetErrorString(__err));            \
      return -1;                                                               \
    }                                                                          \
  } while (0)

static const int NumThreadsPerWarpBeforeMP31 = 128;
static const int NumThreadsPerWarp = 32;
static const int NumThreadsPerWarpSquad = 128;
static const int NumWarpsPerWarpSquad =
    NumThreadsPerWarpSquad / NumThreadsPerWarp;
static const int NumThreadsPerHalfWarp = NumThreadsPerWarp / 2;

enum class SmemSwizzleGranularity : uint8_t {
  NONE = 0,
  B16 = 1,
  B32 = 2,
  B64 = 3,
};

enum class SmemSwizzleStride : uint8_t {
  B32 = 0,
  B64 = 1,
  B128 = 2,
  B256 = 3,
};

enum class SmemSwizzleLine : uint8_t {
  B128 = 0,
  B256 = 1,
};

enum class CacheHint : uint8_t {
  CACHE_NONE = 0,
  CACHE_ONCE = 1,
  CACHE_NORMAL = 2,
  CACHE_PERSIST = 3,
};

enum class PrefetchSize : uint8_t {
  NONE = 0,
  B64 = 64,
  B128 = 128,
};

enum class AddressSpace {
  Generic = 0,
  Global = 1,
  Shared = 3,
};

template <AddressSpace AS>
    TL_HOST_DEVICE constexpr void
    __attribute__((address_space(static_cast<int>(AS)))) *
    make_ptr_with_address_space(uint64_t ptr) {
  return reinterpret_cast<void __attribute__((
      address_space(static_cast<int>(AS)))) *>(ptr);
}

/// MUTE helper to cast SMEM pointer to unsigned
TL_HOST_DEVICE
uint32_t cast_smem_ptr_to_uint(void const *const ptr) {
  /// MUTE helper to get SMEM pointer
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// using mutlass abs function for half_t
TL_PATCH TL_DEVICE half_t __habs(const half_t x) { return abs(x); }

// using mutlass abs function for bfloat_t
TL_PATCH TL_DEVICE bfloat16_t __habs(const bfloat16_t x) { return abs(x); }

// hrsqrt function for half_t
TL_PATCH TL_DEVICE half_t hrsqrt(const half_t x) {
  return half_t(hrsqrt(x.to_half()));
}

// Pack two half values.
TL_DEVICE unsigned __pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Pack two half_t values.
TL_DEVICE unsigned __pack_half2(const half_t x, const half_t y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Pack two bfloat16_t values.
TL_DEVICE unsigned __pack_half2(const bfloat16_t x, const bfloat16_t y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Pack two bfloat16_t values.
TL_DEVICE unsigned __pack_mt_bfloat162(const bfloat16_t x, const bfloat16_t y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Pack four char values.
TL_DEVICE int make_int(signed char x0, signed char x1, signed char x2,
                       signed char x3) {
  return (x3 << 24) | (x2 << 16) | (x1 << 8) | x0;
}

// Pack eight char values.
TL_DEVICE int2 make_int2(signed char x0, signed char x1, signed char x2,
                         signed char x3, signed char y0, signed char y1,
                         signed char y2, signed char y3) {
  int2 result;
  result.x = make_int(x0, x1, x2, x3);
  result.y = make_int(y0, y1, y2, y3);
  return result;
}

// Pack sixteen char values.
TL_DEVICE int4_t make_int4(signed char x0, signed char x1, signed char x2,
                           signed char x3, signed char y0, signed char y1,
                           signed char y2, signed char y3, signed char z0,
                           signed char z1, signed char z2, signed char z3,
                           signed char w0, signed char w1, signed char w2,
                           signed char w3) {
  int4_t result;
  result.x = make_int(x0, x1, x2, x3);
  result.y = make_int(y0, y1, y2, y3);
  result.z = make_int(z0, z1, z2, z3);
  result.w = make_int(w0, w1, w2, w3);
  return result;
}

// Pack eight int values.
TL_DEVICE longlong4 make_longlong4(int x0, int x1, int y0, int y1, int z0,
                                   int z1, int w0, int w1) {
  longlong4 result;
  *((int2 *)&result.x) = make_int2(x0, x1);
  *((int2 *)&result.y) = make_int2(y0, y1);
  *((int2 *)&result.z) = make_int2(z0, z1);
  *((int2 *)&result.w) = make_int2(w0, w1);
  return result;
}

namespace tl {

// Any
template <typename T> TL_DEVICE bool Any(T *a, int size) {
  for (int i = 0; i < size; i++) {
    if (a[i]) {
      return true;
    }
  }
  return false;
}

// All
template <typename T> TL_DEVICE bool All(T *a, int size) {
  for (int i = 0; i < size; i++) {
    if (!a[i]) {
      return false;
    }
  }
  return true;
}

// Pow of int
template <int y = 1, typename T> TL_DEVICE T pow_of_int(T x) {
  T result = x;
  for (int i = 1; i < y; i++) {
    result *= x;
  }
  return result;
}

struct float_e4m3_t : public mutlass::float_e4m3_t {
  using mutlass::float_e4m3_t::float_e4m3_t;

  TL_HOST_DEVICE
  float_e4m3_t() = default;

  TL_HOST_DEVICE
  explicit float_e4m3_t(__mt_bfloat16 x)
      : float_e4m3_t(static_cast<float>(x)) {}
};

struct float_e5m2_t : public mutlass::float_e5m2_t {
  using mutlass::float_e5m2_t::float_e5m2_t;

  TL_HOST_DEVICE
  float_e5m2_t() = default;

  TL_HOST_DEVICE
  explicit float_e5m2_t(__mt_bfloat16 x)
      : float_e5m2_t(static_cast<float>(x)) {}
};

template <typename T> struct to_mute_type {
  using type = T;
};

template <> struct to_mute_type<tl::float_e4m3_t> {
  using type = mutlass::float_e4m3_t;
};
template <> struct to_mute_type<tl::float_e5m2_t> {
  using type = mutlass::float_e5m2_t;
};

// Generic passthroughs
template <typename T>
TL_DEVICE T shfl_xor_sync(unsigned mask, T val, int laneMask) {
  return __shfl_xor_sync(mask, val, laneMask);
}

TL_DEVICE float2 shfl_xor_sync(unsigned mask, float2 val, int laneMask) {
  float2 out;
  out.x = __shfl_xor_sync(mask, val.x, laneMask);
  out.y = __shfl_xor_sync(mask, val.y, laneMask);
  return out;
}

TL_DEVICE float4 shfl_xor_sync(unsigned mask, float4 val, int laneMask) {
  float4 out;
  out.x = __shfl_xor_sync(mask, val.x, laneMask);
  out.y = __shfl_xor_sync(mask, val.y, laneMask);
  out.z = __shfl_xor_sync(mask, val.z, laneMask);
  out.w = __shfl_xor_sync(mask, val.w, laneMask);
  return out;
}

template <typename T>
TL_DEVICE T shfl_down_sync(unsigned mask, T val, int delta) {
  return __shfl_down_sync(mask, val, delta);
}

template <typename T>
TL_DEVICE T shfl_up_sync(unsigned mask, T val, int delta) {
  return __shfl_up_sync(mask, val, delta);
}

template <typename T> TL_DEVICE T shfl_sync(unsigned mask, T val, int srcLane) {
  return __shfl_sync(mask, val, srcLane);
}

// Specializations for mutlass::half_t
template <>
TL_DEVICE half_t shfl_xor_sync(unsigned mask, half_t val, int laneMask) {
  float f = static_cast<float>(val);
  float r = __shfl_xor_sync(mask, f, laneMask);
  return half_t(r);
}

template <>
TL_DEVICE half_t shfl_down_sync(unsigned mask, half_t val, int delta) {
  float f = static_cast<float>(val);
  float r = __shfl_down_sync(mask, f, delta);
  return half_t(r);
}

template <>
TL_DEVICE half_t shfl_up_sync(unsigned mask, half_t val, int delta) {
  float f = static_cast<float>(val);
  float r = __shfl_up_sync(mask, f, delta);
  return half_t(r);
}

template <> TL_DEVICE half_t shfl_sync(unsigned mask, half_t val, int srcLane) {
  float f = static_cast<float>(val);
  float r = __shfl_sync(mask, f, srcLane);
  return half_t(r);
}

// Specializations for mutlass::bfloat16_t
template <>
TL_DEVICE bfloat16_t shfl_xor_sync(unsigned mask, bfloat16_t val,
                                   int laneMask) {
  float f = static_cast<float>(val);
  float r = __shfl_xor_sync(mask, f, laneMask);
  return bfloat16_t(r);
}

template <>
TL_DEVICE bfloat16_t shfl_down_sync(unsigned mask, bfloat16_t val, int delta) {
  float f = static_cast<float>(val);
  float r = __shfl_down_sync(mask, f, delta);
  return bfloat16_t(r);
}

template <>
TL_DEVICE bfloat16_t shfl_up_sync(unsigned mask, bfloat16_t val, int delta) {
  float f = static_cast<float>(val);
  float r = __shfl_up_sync(mask, f, delta);
  return bfloat16_t(r);
}

template <>
TL_DEVICE bfloat16_t shfl_sync(unsigned mask, bfloat16_t val, int srcLane) {
  float f = static_cast<float>(val);
  float r = __shfl_sync(mask, f, srcLane);
  return bfloat16_t(r);
}

TL_DEVICE float2 vec_max_f2(float2 a, float2 b) {
  float2 out;
  mute::max(out, a, b);
  return out;
}

TL_DEVICE float4 vec_max_f4(float4 a, float4 b) {
  float4 out;
  mute::max(out, a, b);
  return out;
}

TL_DEVICE float2 vec_sum_f2(float2 a, float2 b) {
  float2 out;
  mute::add(out, a, b);
  return out;
}

TL_DEVICE float4 vec_sum_f4(float4 a, float4 b) {
  float4 out;
  out.x = a.x + b.x;
  out.y = a.y + b.y;
  out.z = a.z + b.z;
  out.w = a.w + b.w;
  // todo: check bug
  // mute::add(out, a, b);
  return out;
}

TL_DEVICE float2 vec_exp2_f2(float2 a) {
  float2 out;
  mute::fast_exp2(out, a);
  return out;
}

TL_DEVICE float4 vec_exp2_f4(float4 a) {
  float4 out;
  mute::fast_exp2(out, a);
  return out;
}

} // namespace tl
