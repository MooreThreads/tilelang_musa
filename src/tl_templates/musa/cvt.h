#pragma once

#include "common.h"
#include "musa_fp8.h"

namespace tl {

using tl_char_v2 = char __attribute__((ext_vector_type(2)));
using tl_char_v4 = char __attribute__((ext_vector_type(4)));
using tl_float_v2 = float __attribute__((ext_vector_type(2)));
using tl_float_v4 = float __attribute__((ext_vector_type(4)));

// fp8 -> fp16 / fp32
TL_DEVICE half2 cvt_fp8e4m3_to_half_x2(fp8_e4_2_t in) {
  return __musa_e4m32f16_rn_bst2(*reinterpret_cast<const tl_char_v2 *>(&in));
}

TL_DEVICE __half4 cvt_fp8e4m3_to_half_x4(fp8_e4_4_t in) {
  return __musa_e4m32f16_rn_bst4(*reinterpret_cast<const tl_char_v4 *>(&in));
}

TL_DEVICE half2 cvt_fp8e5m2_to_half_x2(fp8_e5_2_t in) {
  return __musa_e5m22f16_rn_bst2(*reinterpret_cast<const tl_char_v2 *>(&in));
}

TL_DEVICE __half4 cvt_fp8e5m2_to_half_x4(fp8_e5_4_t in) {
  return __musa_e5m22f16_rn_bst4(*reinterpret_cast<const tl_char_v4 *>(&in));
}

TL_DEVICE float2 cvt_fp8e4m3_to_float_x2(fp8_e4_2_t in) {
  const tl_float_v2 out = __musa_f162f32_rn_bst2(
      __musa_e4m32f16_rn_bst2(*reinterpret_cast<const tl_char_v2 *>(&in)));
  return *reinterpret_cast<const float2 *>(&out);
}

TL_DEVICE float4 cvt_fp8e4m3_to_float_x4(fp8_e4_4_t in) {
  const tl_float_v4 out = __musa_f162f32_rn_bst4(
      __musa_e4m32f16_rn_bst4(*reinterpret_cast<const tl_char_v4 *>(&in)));
  return *reinterpret_cast<const float4 *>(&out);
}

TL_DEVICE float2 cvt_fp8e5m2_to_float_x2(fp8_e5_2_t in) {
  const tl_float_v2 out = __musa_f162f32_rn_bst2(
      __musa_e5m22f16_rn_bst2(*reinterpret_cast<const tl_char_v2 *>(&in)));
  return *reinterpret_cast<const float2 *>(&out);
}

TL_DEVICE float4 cvt_fp8e5m2_to_float_x4(fp8_e5_4_t in) {
  const tl_float_v4 out = __musa_f162f32_rn_bst4(
      __musa_e5m22f16_rn_bst4(*reinterpret_cast<const tl_char_v4 *>(&in)));
  return *reinterpret_cast<const float4 *>(&out);
}

// fp16 -> fp8 / fp32
TL_DEVICE fp8_e4_2_t cvt_half_to_fp8e4m3_x2(half2 in) {
  fp8_e4_2_t out;
  *reinterpret_cast<__mt_fp8x2_storage_t *>(&out) =
      (__mt_fp8x2_storage_t)__musa_f162e4m3_rn_bst2(in);
  return out;
}

TL_DEVICE fp8_e4_4_t cvt_half_to_fp8e4m3_x4(__half4 in) {
  fp8_e4_4_t out;
  *reinterpret_cast<__mt_fp8x4_storage_t *>(&out) =
      (__mt_fp8x4_storage_t)__musa_f162e4m3_rn_bst4(in);
  return out;
}

TL_DEVICE fp8_e5_2_t cvt_half_to_fp8e5m2_x2(half2 in) {
  fp8_e5_2_t out;
  *reinterpret_cast<__mt_fp8x2_storage_t *>(&out) =
      (__mt_fp8x2_storage_t)__musa_f162e5m2_rn_bst2(in);
  return out;
}

TL_DEVICE fp8_e5_4_t cvt_half_to_fp8e5m2_x4(__half4 in) {
  fp8_e5_4_t out;
  *reinterpret_cast<__mt_fp8x4_storage_t *>(&out) =
      (__mt_fp8x4_storage_t)__musa_f162e5m2_rn_bst4(in);
  return out;
}

TL_DEVICE float2 cvt_half_to_float_x2(half2 in) {
  const tl_float_v2 out = __musa_f162f32_rn_bst2(in);
  return *reinterpret_cast<const float2 *>(&out);
}

TL_DEVICE float4 cvt_half_to_float_x4(__half4 in) {
  const tl_float_v4 out = __musa_f162f32_rn_bst4(in);
  return *reinterpret_cast<const float4 *>(&out);
}

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
      (__mt_fp8x2_storage_t)__musa_f2e4m3_rn_bst2(
          *reinterpret_cast<const tl_float_v2 *>(&in));
  return out;
}

TL_DEVICE fp8_e4_4_t cvt_float_to_fp8e4m3_x4(float4 in) {
  fp8_e4_4_t out;
  *reinterpret_cast<__mt_fp8x4_storage_t *>(&out) =
      (__mt_fp8x4_storage_t)__musa_f2e4m3_rn_bst4(
          *reinterpret_cast<const tl_float_v4 *>(&in));
  return out;
}

TL_DEVICE fp8_e5_2_t cvt_float_to_fp8e5m2_x2(float2 in) {
  fp8_e5_2_t out;
  *reinterpret_cast<__mt_fp8x2_storage_t *>(&out) =
      (__mt_fp8x2_storage_t)__musa_f2e5m2_rn_bst2(
          *reinterpret_cast<const tl_float_v2 *>(&in));
  return out;
}

TL_DEVICE fp8_e5_4_t cvt_float_to_fp8e5m2_x4(float4 in) {
  fp8_e5_4_t out;
  *reinterpret_cast<__mt_fp8x4_storage_t *>(&out) =
      (__mt_fp8x4_storage_t)__musa_f2e5m2_rn_bst4(
          *reinterpret_cast<const tl_float_v4 *>(&in));
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
