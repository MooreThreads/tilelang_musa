/*!
 * \file tl/runtime/runtime.h
 * \brief Runtime functions.
 *
 */

#ifndef TVM_TL_RUNTIME_RUNTIME_H_
#define TVM_TL_RUNTIME_RUNTIME_H_

namespace tvm {
namespace tl {

#if (defined(CUDA_MAJOR_VERSION) && (CUDA_MAJOR_VERSION >= 12)) || defined(MUSA_MAJOR_VERSION)
constexpr const char *tvm_tensormap_create_tiled =
    "__tvm_tensormap_create_tiled";
constexpr const char *tvm_tensormap_create_im2col =
    "__tvm_tensormap_create_im2col";
#endif // (defined(CUDA_MAJOR_VERSION) && (CUDA_MAJOR_VERSION >= 12)) || defined(MUSA_MAJOR_VERSION)

// CUDA stream access policy window helpers
constexpr const char *tvm_cuda_stream_set_access_policy_window =
    "__tvm_cuda_stream_set_access_policy_window";
constexpr const char *tvm_cuda_stream_reset_access_policy_window =
    "__tvm_cuda_stream_reset_access_policy_window";

// MUSA stream access policy window helpers
constexpr const char *tvm_musa_stream_set_access_policy_window =
    "__tvm_musa_stream_set_access_policy_window";
constexpr const char *tvm_musa_stream_reset_access_policy_window =
    "__tvm_musa_stream_reset_access_policy_window";
} // namespace tl
} // namespace tvm

#endif //  TVM_TL_RUNTIME_RUNTIME_H_
