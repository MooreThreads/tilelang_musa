#pragma once

#include "common.h"

#if defined(__MUSA_ARCH_LIST__) && (__MUSA_ARCH_LIST__) >= 310
#include "copy_mp31.h"
#endif

namespace tl {

TL_DEVICE void cp_async_commit() {}

template <int N> TL_DEVICE void cp_async_wait() { __musa_memcpy_g2s_wait(); }

template <int N>
TL_DEVICE void cp_async_gs(void const *const smem_addr, void *global_ptr) {
  __musa_memcpy_g2s((void _AS3 *)smem_addr, (void _AS1 *)global_ptr,
                    N /* total_bytes */, 0 /* prefetch_size */);
}

template <int N>
TL_DEVICE void cp_async_gs_conditional(void const *const smem_addr,
                                       void *global_ptr, bool cond) {
  if (cond) {
    __musa_memcpy_g2s((void _AS3 *)smem_addr, (void _AS1 *)global_ptr,
                      N /* total_bytes */, 0 /* prefetch_size */);
  }
}

} // namespace tl
