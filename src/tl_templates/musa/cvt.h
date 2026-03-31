#pragma once

#include "common.h"
#include "musa_fp8.h"

namespace tl {

struct __MUSA_ALIGN__(8) half4_t {
  half2 lo;
  half2 hi;
};

struct __MUSA_ALIGN__(8) bfloat164_t {
  __mt_bfloat162 lo;
  __mt_bfloat162 hi;
};

TL_DEVICE half4_t make_half4_t(half2 lo, half2 hi) {
  half4_t out;
  out.lo = lo;
  out.hi = hi;
  return out;
}

TL_DEVICE bfloat164_t make_bfloat164_t(__mt_bfloat162 lo, __mt_bfloat162 hi) {
  bfloat164_t out;
  out.lo = lo;
  out.hi = hi;
  return out;
}

TL_DEVICE float4 make_float4_from_float2(float2 lo, float2 hi) {
  float4 out = {lo.x, lo.y, hi.x, hi.y};
  return out;
}

// fp16 <-> fp32
TL_DEVICE float2 cvt_half_to_float_x2(half2 in) { return __half22float2(in); }

TL_DEVICE float4 cvt_half_to_float_x4(half4_t in) {
  return make_float4_from_float2(__half22float2(in.lo), __half22float2(in.hi));
}

TL_DEVICE half2 cvt_float_to_half_x2(float2 in) {
  return __float22half2_rn(in);
}

TL_DEVICE half4_t cvt_float_to_half_x4(float4 in) {
  float2 lo = {in.x, in.y};
  float2 hi = {in.z, in.w};
  return make_half4_t(__float22half2_rn(lo), __float22half2_rn(hi));
}

// bf16 <-> fp32
TL_DEVICE float2 cvt_bfloat16_to_float_x2(__mt_bfloat162 in) {
  return __bfloat1622float2(in);
}

TL_DEVICE float4 cvt_bfloat16_to_float_x4(bfloat164_t in) {
  return make_float4_from_float2(__bfloat1622float2(in.lo),
                                 __bfloat1622float2(in.hi));
}

TL_DEVICE __mt_bfloat162 cvt_float_to_bfloat16_x2(float2 in) {
  return __float22bfloat162_rn(in);
}

TL_DEVICE bfloat164_t cvt_float_to_bfloat16_x4(float4 in) {
  float2 lo = {in.x, in.y};
  float2 hi = {in.z, in.w};
  return make_bfloat164_t(__float22bfloat162_rn(lo), __float22bfloat162_rn(hi));
}

// fp8 -> fp16 / fp32
TL_DEVICE float4 cvt_fp8e4m3_to_float_x4(fp8_e4_4_t in);
TL_DEVICE float4 cvt_fp8e5m2_to_float_x4(fp8_e5_4_t in);

TL_DEVICE half2 cvt_fp8e4m3_to_half_x2(fp8_e4_2_t in) {
  __mt_fp8x2_e4m3 packed;
  packed.__x = *reinterpret_cast<const __mt_fp8x2_storage_t *>(&in);
  return static_cast<half2>(packed);
}

TL_DEVICE half4_t cvt_fp8e4m3_to_half_x4(fp8_e4_4_t in) {
  return cvt_float_to_half_x4(cvt_fp8e4m3_to_float_x4(in));
}

TL_DEVICE half2 cvt_fp8e5m2_to_half_x2(fp8_e5_2_t in) {
  __mt_fp8x2_e5m2 packed;
  packed.__x = *reinterpret_cast<const __mt_fp8x2_storage_t *>(&in);
  return static_cast<half2>(packed);
}

TL_DEVICE half4_t cvt_fp8e5m2_to_half_x4(fp8_e5_4_t in) {
  return cvt_float_to_half_x4(cvt_fp8e5m2_to_float_x4(in));
}

TL_DEVICE float2 cvt_fp8e4m3_to_float_x2(fp8_e4_2_t in) {
  __mt_fp8x2_e4m3 packed;
  packed.__x = *reinterpret_cast<const __mt_fp8x2_storage_t *>(&in);
  return static_cast<float2>(packed);
}

TL_DEVICE float4 cvt_fp8e4m3_to_float_x4(fp8_e4_4_t in) {
  __mt_fp8x4_e4m3 packed;
  packed.__x = *reinterpret_cast<const __mt_fp8x4_storage_t *>(&in);
  return static_cast<float4>(packed);
}

TL_DEVICE float2 cvt_fp8e5m2_to_float_x2(fp8_e5_2_t in) {
  __mt_fp8x2_e5m2 packed;
  packed.__x = *reinterpret_cast<const __mt_fp8x2_storage_t *>(&in);
  return static_cast<float2>(packed);
}

TL_DEVICE float4 cvt_fp8e5m2_to_float_x4(fp8_e5_4_t in) {
  __mt_fp8x4_e5m2 packed;
  packed.__x = *reinterpret_cast<const __mt_fp8x4_storage_t *>(&in);
  return static_cast<float4>(packed);
}

// fp16 / fp32 -> fp8
TL_DEVICE fp8_e4_2_t cvt_half_to_fp8e4m3_x2(half2 in) {
  fp8_e4_2_t out;
  *reinterpret_cast<__mt_fp8x2_storage_t *>(&out) = __mt_fp8x2_e4m3(in).__x;
  return out;
}

TL_DEVICE fp8_e4_4_t cvt_half_to_fp8e4m3_x4(half4_t in) {
  fp8_e4_4_t out;
  *reinterpret_cast<__mt_fp8x4_storage_t *>(&out) =
      __mt_fp8x4_e4m3(in.lo, in.hi).__x;
  return out;
}

TL_DEVICE fp8_e5_2_t cvt_half_to_fp8e5m2_x2(half2 in) {
  fp8_e5_2_t out;
  *reinterpret_cast<__mt_fp8x2_storage_t *>(&out) = __mt_fp8x2_e5m2(in).__x;
  return out;
}

TL_DEVICE fp8_e5_4_t cvt_half_to_fp8e5m2_x4(half4_t in) {
  fp8_e5_4_t out;
  *reinterpret_cast<__mt_fp8x4_storage_t *>(&out) =
      __mt_fp8x4_e5m2(in.lo, in.hi).__x;
  return out;
}

TL_DEVICE fp8_e4_2_t cvt_float_to_fp8e4m3_x2(float2 in) {
  fp8_e4_2_t out;
  *reinterpret_cast<__mt_fp8x2_storage_t *>(&out) = __mt_fp8x2_e4m3(in).__x;
  return out;
}

TL_DEVICE fp8_e4_4_t cvt_float_to_fp8e4m3_x4(float4 in) {
  fp8_e4_4_t out;
  *reinterpret_cast<__mt_fp8x4_storage_t *>(&out) = __mt_fp8x4_e4m3(in).__x;
  return out;
}

TL_DEVICE fp8_e5_2_t cvt_float_to_fp8e5m2_x2(float2 in) {
  fp8_e5_2_t out;
  *reinterpret_cast<__mt_fp8x2_storage_t *>(&out) = __mt_fp8x2_e5m2(in).__x;
  return out;
}

TL_DEVICE fp8_e5_4_t cvt_float_to_fp8e5m2_x4(float4 in) {
  fp8_e5_4_t out;
  *reinterpret_cast<__mt_fp8x4_storage_t *>(&out) = __mt_fp8x4_e5m2(in).__x;
  return out;
}

} // namespace tl
