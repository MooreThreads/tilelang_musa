#pragma once

#include "common.h"
#include "mutlass/mutlass.h"

namespace tl {

namespace detail {

// Provide architecture-specific defaults so callers may omit arguments.
TL_DEVICE constexpr int default_warp_size() {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_DEVICE_COMPILE__)
  return 64;
#elif defined(__MUSA_ARCH_LIST__) && (__MUSA_ARCH_LIST__ <= 220)
  return 128;
#else
  return 32;
#endif
}

TL_DEVICE constexpr int default_warps_per_group() { return 4; }

TL_DEVICE int linear_thread_idx_in_block() {
#if defined(__CUDA_ARCH__) || defined(__MUSA_ARCH__) ||                        \
    defined(__HIP_DEVICE_COMPILE__)
  return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
#else
  return 0;
#endif
}

} // namespace detail

TL_DEVICE int get_lane_idx(int warp_size = detail::default_warp_size()) {
  warp_size = warp_size > 0 ? warp_size : detail::default_warp_size();
  return detail::linear_thread_idx_in_block() % warp_size;
}

TL_DEVICE int get_warp_idx_sync(int warp_size = detail::default_warp_size()) {
  warp_size = warp_size > 0 ? warp_size : detail::default_warp_size();
  return detail::linear_thread_idx_in_block() / warp_size;
}

TL_DEVICE int get_warp_idx(int warp_size = detail::default_warp_size()) {
  warp_size = warp_size > 0 ? warp_size : detail::default_warp_size();
  return detail::linear_thread_idx_in_block() / warp_size;
}

TL_DEVICE int
get_warp_group_idx(int warp_size = detail::default_warp_size(),
                   int warps_per_group = detail::default_warps_per_group()) {
  warp_size = warp_size > 0 ? warp_size : detail::default_warp_size();
  warps_per_group =
      warps_per_group > 0 ? warps_per_group : detail::default_warps_per_group();
  int threads_per_group = warp_size * warps_per_group;
  threads_per_group = threads_per_group > 0 ? threads_per_group : warp_size;
  return detail::linear_thread_idx_in_block() / threads_per_group;
}

// Elect one thread in the warp. The elected thread gets its predicate set to
// true, all others obtain false.
#if defined(__MUSA_ARCH_LIST__) && (__MUSA_ARCH_LIST__ >= 310)
TL_DEVICE uint32_t elect_one_sync() { return (threadIdx.x % 32) == 0; }
#else
TL_DEVICE uint32_t elect_one_sync() { return (threadIdx.x % 128) == 0; }
#endif

/// Returns a warp-uniform value indicating the canonical warp index of the
/// calling threads. Threads within the warp must be converged.
TL_DEVICE
int canonical_warp_idx_sync() {
#if defined(__MUSA_ARCH__)
#if (__MUSA_ARCH__ >= 310)
  return __shfl_sync(0xffffffff, threadIdx.x / NumThreadsPerWarp, 0);
#else
  return threadIdx.x / NumThreadsPerWarpBeforeMP31;
#endif // #if (__MUSA_ARCH__ >= 310)
#else
  return 0;
#endif // #if defined(__MUSA_ARCH__)
}

// Template parameter:
//   thread_extent: the logical size (in number of threads) of each "group"
//                  within which we want to elect exactly ONE representative
//                  thread.
template <int thread_extent> TL_DEVICE bool tl_shuffle_elect() {
  if constexpr (thread_extent == 0) {
    // Elect exactly one thread in the whole block (warp-0 leader).
    return canonical_warp_idx_sync() == 0 && elect_one_sync();
  } else {
    // Elect one representative per logical thread group.
#if defined(__MUSA_ARCH_LIST__) && (__MUSA_ARCH_LIST__ >= 310)
    constexpr int kWarpsPerGroup = (thread_extent + 31) / 32;
    return __shfl_sync(0xffffffff,
                       (threadIdx.x / 32) % kWarpsPerGroup,
                       0) == 0 &&
           elect_one_sync();
#else
    constexpr int kWarpsPerGroup = (thread_extent + 127) / 128;
    return __shfl_sync(0xffffffff,
                       (threadIdx.x / 128) % kWarpsPerGroup,
                       0) == 0 &&
           elect_one_sync();
#endif
  }
}

} // namespace tl
