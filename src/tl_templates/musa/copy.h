#pragma once

#include <cstdint>

#include "common.h"

#if defined(__MUSA_ARCH_LIST__) && (__MUSA_ARCH_LIST__) >= 310
#include "copy_mp31.h"
#include <musa_robust.h>
#endif

namespace tl {

TL_DEVICE void cp_async_commit() {}

template <int N> TL_DEVICE void cp_async_wait() { __musa_memcpy_g2s_wait(); }

template <int N>
TL_DEVICE void cp_async_gs(void const *const smem_addr,
                           void const *global_ptr) {
  __musa_memcpy_g2s((void _AS3 *)smem_addr, (void const _AS1 *)global_ptr,
                    N /* total_bytes */, 0 /* prefetch_size */);
}

template <int N>
TL_DEVICE void cp_async_gs_conditional(void const *const smem_addr,
                                       void const *global_ptr, bool cond) {
  if (cond) {
    cp_async_gs<N>(smem_addr, global_ptr);
  } else {
    auto *smem_ptr = (uint8_t _AS3 *)smem_addr;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      smem_ptr[i] = 0;
    }
  }
}

#if defined(__MUSA_ARCH_LIST__) && (__MUSA_ARCH_LIST__) >= 310

template <typename To, typename From>
TL_DEVICE To robust_bit_cast(const From &from) {
  static_assert(sizeof(To) == sizeof(From));
  union {
    From from;
    To to;
  } storage{from};
  return storage.to;
}

TL_DEVICE __musa::robust_v4_addr_t make_robust_desc(void const *robust_base_ptr,
                                                    uint64_t robust_size) {
  auto *typed_base =
      reinterpret_cast<int8_t *>(const_cast<void *>(robust_base_ptr));
  __musa::robust_ptr<int8_t> robust_ptr(typed_base, robust_size);
  return robust_ptr.get_v4_robust_addr();
}

template <typename T>
TL_DEVICE T robust_load(void const *global_ptr, void const *robust_base_ptr,
                        uint64_t robust_size) {
  auto robust_desc = make_robust_desc(robust_base_ptr, robust_size);
  void const *addr = global_ptr;
  if constexpr (sizeof(T) == 1) {
    auto raw = __musa_ld_v4_robust_i8(addr, robust_desc, 0);
    return robust_bit_cast<T>(raw);
  } else if constexpr (sizeof(T) == 2) {
    auto raw = __musa_ld_v4_robust_i16(addr, robust_desc, 0);
    return robust_bit_cast<T>(raw);
  } else if constexpr (sizeof(T) == 4) {
    auto raw = __musa_ld_v4_robust_i32(addr, robust_desc, 0);
    return robust_bit_cast<T>(raw);
  } else if constexpr (sizeof(T) == 8) {
    auto raw = __musa_ld_v4_robust_i64(addr, robust_desc, 0);
    return robust_bit_cast<T>(raw);
  } else if constexpr (sizeof(T) == 12) {
    auto raw = __musa_ld_v4_robust_v3i32(addr, robust_desc, 0);
    return robust_bit_cast<T>(raw);
  } else if constexpr (sizeof(T) == 16) {
    auto raw = __musa_ld_v4_robust_v4i32(addr, robust_desc, 0);
    return robust_bit_cast<T>(raw);
  } else if constexpr (sizeof(T) == 32) {
    auto raw = __musa_ld_v4_robust_v8i32(addr, robust_desc, 0);
    return robust_bit_cast<T>(raw);
  } else if constexpr (sizeof(T) == 64) {
    auto raw = __musa_ld_v4_robust_v16i32(addr, robust_desc, 0);
    return robust_bit_cast<T>(raw);
  } else if constexpr (sizeof(T) == 128) {
    auto raw = __musa_ld_v4_robust_v32i32(addr, robust_desc, 0);
    return robust_bit_cast<T>(raw);
  } else {
    static_assert(sizeof(T) <= 128, "Unsupported robust load width");
    static_assert(sizeof(T) > 128, "Unsupported robust load width");
  }
}

template <int N>
TL_DEVICE void cp_async_gs_robust(void const *const smem_addr,
                                  void const *global_ptr,
                                  void const *robust_base_ptr,
                                  uint64_t robust_size) {
  auto robust_desc = make_robust_desc(robust_base_ptr, robust_size);
  __musa_memcpy_g2s_robust_v4(
      (void _AS3 *)smem_addr, (void const _AS1 *)global_ptr,
      N /* total_bytes */, robust_desc, 0 /* prefetch_size */);
}

template <int N>
TL_DEVICE void cp_async_gs_robust_conditional(void const *const smem_addr,
                                              void const *global_ptr,
                                              void const *robust_base_ptr,
                                              uint64_t robust_size, bool cond) {
  uint64_t selected_size = cond ? robust_size : 0;
  cp_async_gs_robust<N>(smem_addr, global_ptr, robust_base_ptr, selected_size);
}

#endif

} // namespace tl
