#pragma once

#include "common.h"

namespace tl {
TL_DEVICE void
mbarrier_init(uint32_t barrier_id, // 32 bits user-managed barrier's id
              uint32_t warp_count = 1, // Warp count expected to arrive/wait on this barrier
              uint32_t init_phase = 0) { // Init phase on this barrier
  // #if __MUSA_ARCH__ >= 310
  __musa_async_init_arrival(barrier_id, warp_count, init_phase);
  // #endif
}

TL_DEVICE void mbarrier_wait(uint32_t barrier_id, int phase_bit) {
  // #if __MUSA_ARCH__ >= 310
  __musa_async_wait(barrier_id, phase_bit);
  // #endif
}

TL_DEVICE uint32_t mbarrier_arrive(uint32_t barrier_id) {
  // #if __MUSA_ARCH__ >= 310
  return __musa_async_arrive(barrier_id);
  // #endif
}

TL_DEVICE void mbarrier_arrive_expect_tx(uint32_t barrier_id,
                                         uint32_t transaction_bytes) {
  // #if __MUSA_ARCH__ >= 310
  __musa_async_add_trans(barrier_id, transaction_bytes);
  // #endif
}

template <int Count> TL_DEVICE void tma_store_wait() {
  // #if __MUSA_ARCH__ >= 310
  __musa_tme_idf_l2();
  // #endif
}

} // namespace tl
