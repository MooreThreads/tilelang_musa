#pragma once

#include "common.h"
#include "musa_fp8.h"

namespace tl {

// fp8 -> fp16 / fp32
TL_DEVICE half2 cvt_fp8e4m3_to_half_x2(fp8_e4_2_t in) {
  return make_half2_from_halves(static_cast<half>(in.x),
                                static_cast<half>(in.y));
}

TL_DEVICE __half4 cvt_fp8e4m3_to_half_x4(fp8_e4_4_t in) {
  return make_half4_from_halves(
      static_cast<half>(in.x), static_cast<half>(in.y), static_cast<half>(in.z),
      static_cast<half>(in.w));
}

TL_DEVICE half2 cvt_fp8e5m2_to_half_x2(fp8_e5_2_t in) {
  return make_half2_from_halves(static_cast<half>(in.x),
                                static_cast<half>(in.y));
}

TL_DEVICE __half4 cvt_fp8e5m2_to_half_x4(fp8_e5_4_t in) {
  return make_half4_from_halves(
      static_cast<half>(in.x), static_cast<half>(in.y), static_cast<half>(in.z),
      static_cast<half>(in.w));
}

TL_DEVICE float2 cvt_fp8e4m3_to_float_x2(fp8_e4_2_t in) {
  return __half22float2(cvt_fp8e4m3_to_half_x2(in));
}

TL_DEVICE float4 cvt_fp8e4m3_to_float_x4(fp8_e4_4_t in) {
  return __half42float4(cvt_fp8e4m3_to_half_x4(in));
}

TL_DEVICE float2 cvt_fp8e5m2_to_float_x2(fp8_e5_2_t in) {
  return __half22float2(cvt_fp8e5m2_to_half_x2(in));
}

TL_DEVICE float4 cvt_fp8e5m2_to_float_x4(fp8_e5_4_t in) {
  return __half42float4(cvt_fp8e5m2_to_half_x4(in));
}

// fp16 -> fp8 / fp32
TL_DEVICE fp8_e4_2_t cvt_half_to_fp8e4m3_x2(half2 in) {
  const half *lanes = reinterpret_cast<const half *>(&in);
  return make_fp8_e4_2_t(fp8_e4_t(lanes[0]), fp8_e4_t(lanes[1]));
}

TL_DEVICE fp8_e4_4_t cvt_half_to_fp8e4m3_x4(__half4 in) {
  const half *lanes = reinterpret_cast<const half *>(&in);
  return make_fp8_e4_4_t(fp8_e4_t(lanes[0]), fp8_e4_t(lanes[1]),
                         fp8_e4_t(lanes[2]), fp8_e4_t(lanes[3]));
}

TL_DEVICE fp8_e5_2_t cvt_half_to_fp8e5m2_x2(half2 in) {
  const half *lanes = reinterpret_cast<const half *>(&in);
  return make_fp8_e5_2_t(fp8_e5_t(lanes[0]), fp8_e5_t(lanes[1]));
}

TL_DEVICE fp8_e5_4_t cvt_half_to_fp8e5m2_x4(__half4 in) {
  const half *lanes = reinterpret_cast<const half *>(&in);
  return make_fp8_e5_4_t(fp8_e5_t(lanes[0]), fp8_e5_t(lanes[1]),
                         fp8_e5_t(lanes[2]), fp8_e5_t(lanes[3]));
}

TL_DEVICE float2 cvt_half_to_float_x2(half2 in) { return __half22float2(in); }

TL_DEVICE float4 cvt_half_to_float_x4(__half4 in) { return __half42float4(in); }

// bf16 -> fp32
TL_DEVICE float2 cvt_bfloat16_to_float_x2(__mt_bfloat162 in) {
  return __bfloat1622float2(in);
}

TL_DEVICE float4 cvt_bfloat16_to_float_x4(__mt_bfloat164 in) {
  return __bfloat1642float4(in);
}

// fp32 -> fp8 / fp16 / bf16
TL_DEVICE fp8_e4_2_t cvt_float_to_fp8e4m3_x2(float2 in) {
  fp8_e4_2_t out;
  *reinterpret_cast<__mt_fp8x2_storage_t *>(&out) =
      __musa_cvt_float2_to_fp8x2(in, __MT_SATFINITE, __MT_E4M3);
  return out;
}

TL_DEVICE fp8_e4_4_t cvt_float_to_fp8e4m3_x4(float4 in) {
  fp8_e4_4_t out;
  *reinterpret_cast<__mt_fp8x4_storage_t *>(&out) =
      __musa_cvt_float4_to_fp8x4(in, __MT_SATFINITE, __MT_E4M3);
  return out;
}

TL_DEVICE fp8_e5_2_t cvt_float_to_fp8e5m2_x2(float2 in) {
  fp8_e5_2_t out;
  *reinterpret_cast<__mt_fp8x2_storage_t *>(&out) =
      __musa_cvt_float2_to_fp8x2(in, __MT_SATFINITE, __MT_E5M2);
  return out;
}

TL_DEVICE fp8_e5_4_t cvt_float_to_fp8e5m2_x4(float4 in) {
  fp8_e5_4_t out;
  *reinterpret_cast<__mt_fp8x4_storage_t *>(&out) =
      __musa_cvt_float4_to_fp8x4(in, __MT_SATFINITE, __MT_E5M2);
  return out;
}

TL_DEVICE half2 cvt_float_to_half_x2(float2 in) {
  return __float22half2_rn(in);
}

TL_DEVICE __half4 cvt_float_to_half_x4(float4 in) {
  return __float42half4_rn(in);
}

TL_DEVICE __mt_bfloat162 cvt_float_to_bfloat16_x2(float2 in) {
  return __float22bfloat162_rn(in);
}

TL_DEVICE __mt_bfloat164 cvt_float_to_bfloat16_x4(float4 in) {
  return __float42bfloat164_rn(in);
}

} // namespace tl
