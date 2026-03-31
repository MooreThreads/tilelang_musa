/*!
 * \file tl/runtime/runtime.h
 * \brief Runtime functions.
 *
 */

#include "runtime.h"

#if defined(MUSA_MAJOR_VERSION)
#include <musa.h>
#else
#include "../target/stubs/cuda.h"
#endif

#include <tvm/ffi/function.h>
#include <tvm/node/node.h>

namespace tvm {
namespace tl {

#if 1
// Thread-local storage for restoring the L2 persisting cache limit
static thread_local size_t __tl_prev_persisting_l2_cache_size = 0;
static thread_local bool __tl_prev_persisting_l2_cache_saved = false;
#endif

#if defined(CUDA_MAJOR_VERSION) && (CUDA_MAJOR_VERSION >= 12)
template <typename T> static std::string ArrayToStr(const T *ptr, size_t n) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < n; i++) {
    if (i > 0)
      ss << ", ";
    ss << ptr[i]; // NOLINT(clang-analyzer-security.ArrayBound)
  }
  ss << "]";
  return ss.str();
}

struct TensorMapArgs {
  CUtensorMap *map;
  CUtensorMapDataType type;
  cuuint32_t tensorRank;
  void *globalAddress;
  cuuint64_t globalDim[5], globalStride[5];
  cuuint32_t boxDim[5], elementStrides[5];
  CUtensorMapInterleave interleave;
  CUtensorMapSwizzle swizzle;
  CUtensorMapL2promotion l2Promotion;
  CUtensorMapFloatOOBfill oobFill;

  static TensorMapArgs Extract(PackedArgs args) {
    TensorMapArgs T;
    int idx = 0;
    ICHECK(args.size() >= 8);
    T.map = reinterpret_cast<CUtensorMap *>(args[idx++].cast<void *>());
    T.type = static_cast<CUtensorMapDataType>(args[idx++].cast<int64_t>());
    T.tensorRank = static_cast<cuuint32_t>(args[idx++].cast<int64_t>());
    T.globalAddress = args[idx++].cast<void *>();
    ICHECK(T.tensorRank >= 1 && T.tensorRank <= 5);
    ICHECK(args.size() == static_cast<int>(8 + T.tensorRank * 4));
    for (size_t i = 0; i < T.tensorRank; i++) {
      T.globalDim[i] = args[idx++].cast<cuuint64_t>();
    }
    for (size_t i = 0; i < T.tensorRank; i++) {
      T.globalStride[i] = args[idx++].cast<cuuint64_t>();
    }
    for (size_t i = 0; i < T.tensorRank; i++) {
      T.boxDim[i] = args[idx++].cast<cuuint64_t>();
    }
    for (size_t i = 0; i < T.tensorRank; i++) {
      T.elementStrides[i] = args[idx++].cast<cuuint64_t>();
    }
    T.interleave =
        static_cast<CUtensorMapInterleave>(args[idx++].cast<int64_t>());
    T.swizzle = static_cast<CUtensorMapSwizzle>(args[idx++].cast<int64_t>());
    T.l2Promotion =
        static_cast<CUtensorMapL2promotion>(args[idx++].cast<int64_t>());
    T.oobFill =
        static_cast<CUtensorMapFloatOOBfill>(args[idx++].cast<int64_t>());
    return T;
  }

  std::string ToDebugString() {
    std::stringstream ss;
    ss << "TMA Desc Addr:   " << map << '\n'
       << "format         " << type << '\n'
       << "dim            " << tensorRank << '\n'
       << "gmem_address   " << globalAddress << '\n'
       << "globalDim      " << ArrayToStr(globalDim, tensorRank) << '\n'
       << "globalStrides  " << ArrayToStr(globalStride, tensorRank) << '\n'
       << "boxDim         " << ArrayToStr(boxDim, tensorRank) << '\n'
       << "elementStrides " << ArrayToStr(elementStrides, tensorRank) << '\n'
       << "interleave     " << interleave << '\n'
       << "swizzle        " << swizzle << '\n'
       << "l2Promotion    " << l2Promotion << '\n'
       << "oobFill        " << oobFill << '\n';
    return ss.str();
  }
};

// set device api
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // Register using the canonical names defined in runtime.h
  refl::GlobalDef().def_packed(
      tl::tvm_tensormap_create_tiled, [](PackedArgs args, Any *ret) {
        TensorMapArgs T = TensorMapArgs::Extract(args);
        CUresult result = cuTensorMapEncodeTiled(
            T.map, T.type, T.tensorRank, T.globalAddress, T.globalDim,
            T.globalStride + 1, T.boxDim, T.elementStrides, T.interleave,
            T.swizzle, T.l2Promotion, T.oobFill);
        if (result != CUDA_SUCCESS) {
          LOG_FATAL << "Failed to initialize the TMA descriptor " << result
                    << '\n'
                    << T.ToDebugString();
        }
        *ret = static_cast<int>(result);
      });
}

struct TensorMapIm2ColArgs {
  CUtensorMap *map;
  CUtensorMapDataType type;
  cuuint32_t tensorRank;
  void *globalAddress;
  cuuint64_t globalDim[5], globalStride[5];
  cuuint32_t elementStrides[5];
  int pixelBoxLowerCorner[3], pixelBoxUpperCorner[3];
  cuuint32_t smem_box_channel, smem_box_pixel;
  CUtensorMapInterleave interleave;
  CUtensorMapSwizzle swizzle;
  CUtensorMapL2promotion l2Promotion;
  CUtensorMapFloatOOBfill oobFill;

  static TensorMapIm2ColArgs Extract(PackedArgs args) {
    TensorMapIm2ColArgs T;
    int idx = 0;
    ICHECK(args.size() >= 8);
    T.map = reinterpret_cast<CUtensorMap *>(args[idx++].cast<void *>());
    T.type = static_cast<CUtensorMapDataType>(args[idx++].cast<int64_t>());
    T.tensorRank = static_cast<cuuint32_t>(args[idx++].cast<int64_t>());
    T.globalAddress = args[idx++].cast<void *>();
    ICHECK(T.tensorRank >= 3 && T.tensorRank <= 5);
    ICHECK(args.size() == static_cast<int>(6 + T.tensorRank * 5));
    for (size_t i = 0; i < T.tensorRank; i++) {
      T.globalDim[i] = args[idx++].cast<cuuint64_t>();
    }
    for (size_t i = 0; i < T.tensorRank; i++) {
      T.globalStride[i] = args[idx++].cast<cuuint64_t>();
    }
    for (size_t i = 0; i < T.tensorRank; i++) {
      T.elementStrides[i] = args[idx++].cast<cuuint64_t>();
    }
    for (size_t i = 0; i < T.tensorRank - 2; i++) {
      T.pixelBoxLowerCorner[i] = args[idx++].cast<int>();
    }
    for (size_t i = 0; i < T.tensorRank - 2; i++) {
      T.pixelBoxUpperCorner[i] = args[idx++].cast<int>();
    }
    T.smem_box_pixel = args[idx++].cast<cuuint64_t>();
    T.smem_box_channel = args[idx++].cast<cuuint64_t>();
    T.interleave =
        static_cast<CUtensorMapInterleave>(args[idx++].cast<int64_t>());
    T.swizzle = static_cast<CUtensorMapSwizzle>(args[idx++].cast<int64_t>());
    T.l2Promotion =
        static_cast<CUtensorMapL2promotion>(args[idx++].cast<int64_t>());
    T.oobFill =
        static_cast<CUtensorMapFloatOOBfill>(args[idx++].cast<int64_t>());
    return T;
  }

  std::string ToDebugString() {
    std::stringstream ss;
    ss << "TMA Desc Addr:   " << map << '\n'
       << "format         " << type << '\n'
       << "dim            " << tensorRank << '\n'
       << "gmem_address   " << globalAddress << '\n'
       << "globalDim      " << ArrayToStr(globalDim, tensorRank) << '\n'
       << "globalStrides  " << ArrayToStr(globalStride, tensorRank) << '\n'
       << "smem_box_pixel " << smem_box_pixel << '\n'
       << "smem_box_channel " << smem_box_channel << '\n'
       << "pixelBoxLowerCorner  "
       << ArrayToStr(pixelBoxLowerCorner, tensorRank - 2) << '\n'
       << "pixelBoxUpperCorner  "
       << ArrayToStr(pixelBoxUpperCorner, tensorRank - 2) << '\n'
       << "elementStrides " << ArrayToStr(elementStrides, tensorRank) << '\n'
       << "interleave     " << interleave << '\n'
       << "swizzle        " << swizzle << '\n'
       << "l2Promotion    " << l2Promotion << '\n'
       << "oobFill        " << oobFill << '\n';
    return ss.str();
  }
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed(
      tl::tvm_tensormap_create_im2col, [](PackedArgs args, Any *ret) {
        TensorMapIm2ColArgs T = TensorMapIm2ColArgs::Extract(args);
        CUresult result = cuTensorMapEncodeIm2col(
            T.map, T.type, T.tensorRank, T.globalAddress, T.globalDim,
            T.globalStride + 1, T.pixelBoxLowerCorner, T.pixelBoxUpperCorner,
            T.smem_box_channel, T.smem_box_pixel, T.elementStrides,
            T.interleave, T.swizzle, T.l2Promotion, T.oobFill);
        if (result != CUDA_SUCCESS) {
          LOG_FATAL << "Failed to initialize the TMA descriptor " << result
                    << '\n'
                    << T.ToDebugString();
        }
        *ret = static_cast<int>(result);
      });
}

#endif // defined(CUDA_MAJOR_VERSION) && (CUDA_MAJOR_VERSION >= 12)

#if defined(MUSA_MAJOR_VERSION)
struct MusaTensorDescriptorArgs {
  MUtensorDescriptor *desc;
  MUtensorDescriptorDataType type;
  muuint32_t tensor_rank;
  void *global_address;
  muuint64_t global_dim[5];
  muuint64_t global_stride[5];
  MUtensorDescriptorInterleave interleave;
  muuint64_t oob_fill;

  static MusaTensorDescriptorArgs Extract(PackedArgs args) {
    MusaTensorDescriptorArgs t{};
    int idx = 0;

    ICHECK(args.size() >= 8)
        << "Invalid __tvm_tensormap_create_tiled args size: " << args.size();

    t.desc = reinterpret_cast<MUtensorDescriptor *>(args[idx++].cast<void *>());
    t.type = static_cast<MUtensorDescriptorDataType>(args[idx++].cast<int64_t>());
    t.tensor_rank = static_cast<muuint32_t>(args[idx++].cast<int64_t>());
    t.global_address = args[idx++].cast<void *>();

    ICHECK(t.tensor_rank >= 1 && t.tensor_rank <= 5)
        << "Invalid tensor rank for MUSA TMA descriptor: " << t.tensor_rank;

    // Expected packed args:
    // desc, dtype, rank, global_addr,
    // global_dim[rank], global_stride[rank], smem_box[rank], elem_stride[rank],
    // interleave, swizzle, l2_promotion, oob_fill
    int expected = static_cast<int>(8 + t.tensor_rank * 4);
    ICHECK_EQ(args.size(), expected)
        << "Unexpected __tvm_tensormap_create_tiled args size: got "
        << args.size() << ", expected " << expected;

    for (size_t i = 0; i < t.tensor_rank; ++i) {
      t.global_dim[i] = args[idx++].cast<muuint64_t>();
    }
    for (size_t i = 0; i < t.tensor_rank; ++i) {
      t.global_stride[i] = args[idx++].cast<muuint64_t>();
    }

    // Skip smem_box[rank] and element_stride[rank]. They are not required by
    // muTensorDescriptorEncode.
    idx += static_cast<int>(t.tensor_rank * 2);

    t.interleave =
        static_cast<MUtensorDescriptorInterleave>(args[idx++].cast<int64_t>());

    // Skip swizzle and l2_promotion for descriptor encode.
    idx += 2;
    t.oob_fill = static_cast<muuint64_t>(args[idx++].cast<int64_t>());

    return t;
  }
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed(tl::tvm_tensormap_create_tiled,
                               [](PackedArgs args, Any *ret) {
                                 MusaTensorDescriptorArgs t =
                                     MusaTensorDescriptorArgs::Extract(args);
                                 MUresult result = muTensorDescriptorEncode(
                                     t.desc, t.type, t.tensor_rank,
                                     t.global_address, t.global_dim,
                                     t.global_stride + 1, t.interleave,
                                     t.oob_fill);
                                 ICHECK(result == MUSA_SUCCESS)
                                     << "muTensorDescriptorEncode failed with "
                                     << result;
                                 *ret = static_cast<int>(result);
                               });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed(tl::tvm_tensormap_create_im2col,
                               [](PackedArgs args, Any *ret) {
                                 LOG(FATAL)
                                     << "__tvm_tensormap_create_im2col is not "
                                        "supported in MUSA TVM FFI runtime yet.";
                                 *ret = -1;
                               });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed(
      tl::tvm_musa_stream_set_access_policy_window,
      [](PackedArgs args, Any *ret) {
        ICHECK(args.size() >= 2) << "Expected at least base_ptr and num_bytes";

        void *base_ptr = args[0].cast<void *>();
        size_t num_bytes = static_cast<size_t>(args[1].cast<int64_t>());
        float hit_ratio = 0.8f;
        if (args.size() >= 3) {
          hit_ratio = static_cast<float>(args[2].cast<double>());
        }
        MUstream stream = nullptr;
        if (args.size() >= 4) {
          stream = reinterpret_cast<MUstream>(args[3].cast<void *>());
        }
        size_t l2_limit_bytes = num_bytes;
        if (args.size() >= 5) {
          l2_limit_bytes = static_cast<size_t>(args[4].cast<int64_t>());
        }

        MUdevice device;
        MUresult result = muCtxGetDevice(&device);
        if (result != MUSA_SUCCESS) {
          LOG_FATAL << "Failed to get current MUSA device: " << result;
        }

        int max_persisting = 0;
        result = muDeviceGetAttribute(
            &max_persisting, MU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE,
            device);
        if (result != MUSA_SUCCESS) {
          LOG_FATAL << "Failed to query MAX_PERSISTING_L2_CACHE_SIZE: "
                    << result;
        }
        if (max_persisting > 0 &&
            l2_limit_bytes > static_cast<size_t>(max_persisting)) {
          l2_limit_bytes = static_cast<size_t>(max_persisting);
        }

        size_t init_persisting_l2_cache_size = 0;
        result = muCtxGetLimit(&init_persisting_l2_cache_size,
                               MU_LIMIT_PERSISTING_L2_CACHE_SIZE);
        if (result != MUSA_SUCCESS) {
          LOG_FATAL << "Failed to get current persisting L2 cache size limit: "
                    << result;
        }
        __tl_prev_persisting_l2_cache_size = init_persisting_l2_cache_size;
        __tl_prev_persisting_l2_cache_saved = true;

        result =
            muCtxSetLimit(MU_LIMIT_PERSISTING_L2_CACHE_SIZE, l2_limit_bytes);
        if (result != MUSA_SUCCESS) {
          LOG_FATAL << "Failed to set persisting L2 cache size limit: "
                    << result;
        }

        MUstreamAttrValue stream_attribute{};
        stream_attribute.accessPolicyWindow.base_ptr = base_ptr;
        stream_attribute.accessPolicyWindow.num_bytes = l2_limit_bytes;
        stream_attribute.accessPolicyWindow.hitRatio = hit_ratio;
        stream_attribute.accessPolicyWindow.hitProp =
            MU_ACCESS_PROPERTY_PERSISTING;
        stream_attribute.accessPolicyWindow.missProp =
            MU_ACCESS_PROPERTY_STREAMING;

        result = muStreamSetAttribute(
            stream, MU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW, &stream_attribute);
        if (result != MUSA_SUCCESS) {
          LOG_FATAL << "Failed to set stream access policy window: " << result;
        }

        *ret = static_cast<int>(result);
      });

  refl::GlobalDef().def_packed(
      tl::tvm_musa_stream_reset_access_policy_window,
      [](PackedArgs args, Any *ret) {
        MUstream stream = nullptr;
        if (args.size() >= 1) {
          stream = reinterpret_cast<MUstream>(args[0].cast<void *>());
        }

        MUstreamAttrValue stream_attribute{};
        stream_attribute.accessPolicyWindow.num_bytes = 0;

        MUresult result = muStreamSetAttribute(
            stream, MU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW, &stream_attribute);
        if (result != MUSA_SUCCESS) {
          LOG_FATAL << "Failed to reset stream access policy window: "
                    << result;
        }

        result = muCtxResetPersistingL2Cache();
        if (result != MUSA_SUCCESS) {
          LOG_FATAL << "Failed to reset persisting L2 cache lines: " << result;
        }

        if (__tl_prev_persisting_l2_cache_saved) {
          result = muCtxSetLimit(MU_LIMIT_PERSISTING_L2_CACHE_SIZE,
                                 __tl_prev_persisting_l2_cache_size);
          if (result != MUSA_SUCCESS) {
            LOG_FATAL << "Failed to restore persisting L2 cache size limit: "
                      << result;
          }
          __tl_prev_persisting_l2_cache_saved = false;
        }

        *ret = static_cast<int>(result);
      });
}
#endif // defined(MUSA_MAJOR_VERSION)

#if defined(CUDA_MAJOR_VERSION)
//
// CUDA L2 Persisting Cache Access Policy Window helpers.
// Exposed as TVM FFI packed functions similar to TMA initialization.
//
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // Set stream access policy window and adjust persisting L2 cache size
  // Args:
  //  [0]: void* base_ptr (required)
  //  [1]: int64 num_bytes (required)
  //  [2]: float hit_ratio (optional, default 0.8)
  //  [3]: void* stream (optional, default 0 => default stream)
  //  [4]: int64 l2_limit_bytes (optional, default = num_bytes)
  refl::GlobalDef().def_packed(
      tl::tvm_cuda_stream_set_access_policy_window,
      [](PackedArgs args, Any *ret) {
        ICHECK(args.size() >= 2) << "Expected at least base_ptr and num_bytes";

        void *base_ptr = args[0].cast<void *>();
        size_t num_bytes = static_cast<size_t>(args[1].cast<int64_t>());
        float hit_ratio = 0.8f;
        if (args.size() >= 3) {
          // Accept double/float
          hit_ratio = static_cast<float>(args[2].cast<double>());
        }
        CUstream stream = nullptr;
        if (args.size() >= 4) {
          stream = reinterpret_cast<CUstream>(args[3].cast<void *>());
        }
        size_t l2_limit_bytes = num_bytes;
        if (args.size() >= 5) {
          l2_limit_bytes = static_cast<size_t>(args[4].cast<int64_t>());
        }

        // Clamp requested limit to device capability
        CUdevice device;
        CUresult result = cuCtxGetDevice(&device);
        if (result != CUDA_SUCCESS) {
          LOG_FATAL << "Failed to get current CUDA device: " << result;
        }
        int max_persisting = 0;
        result = cuDeviceGetAttribute(
            &max_persisting, CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE,
            device);
        if (result != CUDA_SUCCESS) {
          LOG_FATAL << "Failed to query MAX_PERSISTING_L2_CACHE_SIZE: "
                    << result;
        }
        if (max_persisting > 0 &&
            l2_limit_bytes > static_cast<size_t>(max_persisting)) {
          l2_limit_bytes = static_cast<size_t>(max_persisting);
        }

        // Save current limit to restore later
        size_t init_persisting_l2_cache_size = 0;
        result = cuCtxGetLimit(&init_persisting_l2_cache_size,
                               CU_LIMIT_PERSISTING_L2_CACHE_SIZE);
        if (result != CUDA_SUCCESS) {
          LOG_FATAL << "Failed to get current persisting L2 cache size limit: "
                    << result;
        }
        __tl_prev_persisting_l2_cache_size = init_persisting_l2_cache_size;
        __tl_prev_persisting_l2_cache_saved = true;

        // Set new limit
        result =
            cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE, l2_limit_bytes);
        if (result != CUDA_SUCCESS) {
          LOG_FATAL << "Failed to set persisting L2 cache size limit: "
                    << result;
        }

        // Apply access policy window to stream
        CUstreamAttrValue stream_attribute;
        memset(&stream_attribute, 0, sizeof(stream_attribute));
        stream_attribute.accessPolicyWindow.base_ptr = base_ptr;
        stream_attribute.accessPolicyWindow.num_bytes = l2_limit_bytes;
        stream_attribute.accessPolicyWindow.hitRatio = hit_ratio;
        stream_attribute.accessPolicyWindow.hitProp =
            CU_ACCESS_PROPERTY_PERSISTING;
        stream_attribute.accessPolicyWindow.missProp =
            CU_ACCESS_PROPERTY_STREAMING;

        result = cuStreamSetAttribute(stream,
                                      CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,
                                      &stream_attribute);
        if (result != CUDA_SUCCESS) {
          LOG_FATAL << "Failed to set stream access policy window: " << result;
        }

        *ret = static_cast<int>(result);
      });

  // Reset stream access policy window and restore the previous L2 cache size
  // Args:
  //  [0]: void* stream (optional, default 0)
  refl::GlobalDef().def_packed(
      tl::tvm_cuda_stream_reset_access_policy_window,
      [](PackedArgs args, Any *ret) {
        CUstream stream = nullptr;
        if (args.size() >= 1) {
          stream = reinterpret_cast<CUstream>(args[0].cast<void *>());
        }

        CUstreamAttrValue stream_attribute;
        memset(&stream_attribute, 0, sizeof(stream_attribute));
        // num_bytes = 0 disables the access policy window on the stream
        stream_attribute.accessPolicyWindow.num_bytes = 0;

        CUresult result = cuStreamSetAttribute(
            stream, CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,
            &stream_attribute);
        if (result != CUDA_SUCCESS) {
          LOG_FATAL << "Failed to reset stream access policy window: "
                    << result;
        }

        result = cuCtxResetPersistingL2Cache();
        if (result != CUDA_SUCCESS) {
          LOG_FATAL << "Failed to reset persisting L2 cache lines: " << result;
        }

        if (__tl_prev_persisting_l2_cache_saved) {
          result = cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE,
                                 __tl_prev_persisting_l2_cache_size);
          if (result != CUDA_SUCCESS) {
            LOG_FATAL << "Failed to restore persisting L2 cache size limit: "
                      << result;
          }
          __tl_prev_persisting_l2_cache_saved = false;
        }

        *ret = static_cast<int>(result);
      });
}
#endif // defined(CUDA_MAJOR_VERSION)

} // namespace tl
} // namespace tvm
