#pragma once

#include "common.h"
#include "mutlass/mutlass.h"

namespace tl {

namespace detail {

// Provide architecture-specific defaults so callers may omit arguments.
TL_DEVICE constexpr int default_warp_size() {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_DEVICE_COMPILE__)
  return 64;
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

  // Special case: thread_extent == 0 means "elect exactly one thread
  // in the entire thread block", i.e., the leader of the first warp of the
  // block.
  if constexpr (thread_extent == 0) {
    // cutlass::canonical_warp_idx_sync():
    //   Returns the warp ID within the thread block in a "canonical" way
    //   (0 for the first warp, 1 for the second, ...).
    // cute::elect_one_sync():
    //   Elect exactly one lane in the warp to return true (typically lane 0),
    //   other lanes return false.
    // The condition ensures that:
    //   (1) We are in warp 0 of the block.
    //   (2) We are the elected lane in this warp.
    return canonical_warp_idx_sync() == 0 && elect_one_sync();
  }

  // General case: thread_extent != 0
  // (threadIdx.x / 32) is the warp index in the block.
  // (thread_extent / 32) is the number of warps in one group of size
  // thread_extent. We take warp_id % num_warps_in_group to get the warp's index
  // within the group.
  // __shfl_sync(mask, value, srcLane): broadcast 'value' from srcLane to all
  // lanes in the warp. Here it broadcasts the group-local warp index from lane
  // 0. Comparing to 0 selects only the group's warp 0.
#if defined(__MUSA_ARCH_LIST__) && (__MUSA_ARCH_LIST__ >= 310)
  return __shfl_sync(0xffffffff, // full warp mask
                     (threadIdx.x / 32) %
                         (thread_extent / 32), // warp index within group
                     0                         // take the value from lane 0
                     ) == 0 &&
         // Within that group leader warp, elect exactly one lane (typically
         // lane 0) to be the single representative for the group.
         elect_one_sync();
#else
  return __shfl_sync(0xffffffff, // full warp mask
                     (threadIdx.x / 128) %
                         (thread_extent / 128), // warp index within group
                     0                          // take the value from lane 0
                     ) == 0 &&
         // Within that group leader warp, elect exactly one lane (typically
         // lane 0) to be the single representative for the group.
         elect_one_sync();
#endif // #if defined(__MUSA_ARCH_LIST__) && (__MUSA_ARCH_LIST__ >= 310)
}

} // namespace tl
