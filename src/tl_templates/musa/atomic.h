#pragma once

#include <cstdint>
#include <musa_fp16.h>
#include <type_traits>
#include <mutlass/numeric_types.h>

#define TL_DEVICE __forceinline__ __device__

using mutlass::bfloat16_t;
using mutlass::half_t;

template <typename T> struct normalize_atomic_type {
  using type = T;
};

template <> struct normalize_atomic_type<half_t> {
  using type = half;
};

template <> struct normalize_atomic_type<bfloat16_t> {
  using type = __mt_bfloat16;
};

template <> struct normalize_atomic_type<int64_t> {
  using type = unsigned long long;
};

template <typename T> TL_DEVICE unsigned short BitCastToU16(T value) {
  static_assert(sizeof(T) == sizeof(unsigned short));
  union {
    T value;
    unsigned short bits;
  } caster;
  caster.value = value;
  return caster.bits;
}

template <typename T> TL_DEVICE T BitCastFromU16(unsigned short bits) {
  static_assert(sizeof(T) == sizeof(unsigned short));
  union {
    unsigned short bits;
    T value;
  } caster;
  caster.bits = bits;
  return caster.value;
}

template <typename T> TL_DEVICE int BitCastToI32(T value) {
  static_assert(sizeof(T) == sizeof(int));
  union {
    T value;
    int bits;
  } caster;
  caster.value = value;
  return caster.bits;
}

template <typename T> TL_DEVICE T BitCastFromI32(int bits) {
  static_assert(sizeof(T) == sizeof(int));
  union {
    int bits;
    T value;
  } caster;
  caster.bits = bits;
  return caster.value;
}

template <typename T1, typename T2>
TL_DEVICE void AtomicMax(T1 &ref, T2 val, int memory_order = 0) {
  (void)memory_order;
  using RawT = std::remove_cv_t<T1>;
  RawT *address = reinterpret_cast<RawT *>(&ref);
  RawT casted = static_cast<RawT>(val);
  if constexpr (std::is_same_v<RawT, float>) {
    int *address_as_i32 = reinterpret_cast<int *>(address);
    int old = *address_as_i32;
    while (casted > BitCastFromI32<float>(old)) {
      int assumed = old;
      old = atomicCAS(address_as_i32, assumed, BitCastToI32(casted));
      if (assumed == old) {
        break;
      }
    }
  } else if constexpr (std::is_same_v<RawT, half_t> ||
                       std::is_same_v<RawT, bfloat16_t>) {
    unsigned short *address_as_u16 = reinterpret_cast<unsigned short *>(address);
    unsigned short val_as_u16 = BitCastToU16(casted);
    unsigned short old = *address_as_u16;
    while (static_cast<float>(casted) >
           static_cast<float>(BitCastFromU16<RawT>(old))) {
      unsigned short assumed = old;
      old = atomicCAS(address_as_u16, assumed, val_as_u16);
      if (assumed == old) {
        break;
      }
    }
  } else {
    using NT1 = typename normalize_atomic_type<RawT>::type;
    atomicMax(reinterpret_cast<NT1 *>(address), static_cast<NT1>(casted));
  }
}

template <typename T1, typename T2>
TL_DEVICE T1 AtomicMaxRet(T1 &ref, T2 val, int memory_order = 0) {
  (void)memory_order;
  using RawT = std::remove_cv_t<T1>;
  RawT *address = reinterpret_cast<RawT *>(&ref);
  RawT casted = static_cast<RawT>(val);
  if constexpr (std::is_same_v<RawT, float>) {
    int *address_as_i32 = reinterpret_cast<int *>(address);
    int old = *address_as_i32;
    while (casted > BitCastFromI32<float>(old)) {
      int assumed = old;
      old = atomicCAS(address_as_i32, assumed, BitCastToI32(casted));
      if (assumed == old) {
        break;
      }
    }
    return static_cast<T1>(BitCastFromI32<float>(old));
  } else if constexpr (std::is_same_v<RawT, half_t> ||
                       std::is_same_v<RawT, bfloat16_t>) {
    unsigned short *address_as_u16 = reinterpret_cast<unsigned short *>(address);
    unsigned short val_as_u16 = BitCastToU16(casted);
    unsigned short old = *address_as_u16;
    while (static_cast<float>(casted) >
           static_cast<float>(BitCastFromU16<RawT>(old))) {
      unsigned short assumed = old;
      old = atomicCAS(address_as_u16, assumed, val_as_u16);
      if (assumed == old) {
        break;
      }
    }
    return static_cast<T1>(BitCastFromU16<RawT>(old));
  } else {
    using NT1 = typename normalize_atomic_type<RawT>::type;
    return static_cast<T1>(
        atomicMax(reinterpret_cast<NT1 *>(address), static_cast<NT1>(casted)));
  }
}

template <typename T1, typename T2>
TL_DEVICE void AtomicMin(T1 &ref, T2 val, int memory_order = 0) {
  (void)memory_order;
  using RawT = std::remove_cv_t<T1>;
  RawT *address = reinterpret_cast<RawT *>(&ref);
  RawT casted = static_cast<RawT>(val);
  if constexpr (std::is_same_v<RawT, float>) {
    int *address_as_i32 = reinterpret_cast<int *>(address);
    int old = *address_as_i32;
    while (casted < BitCastFromI32<float>(old)) {
      int assumed = old;
      old = atomicCAS(address_as_i32, assumed, BitCastToI32(casted));
      if (assumed == old) {
        break;
      }
    }
  } else if constexpr (std::is_same_v<RawT, half_t> ||
                       std::is_same_v<RawT, bfloat16_t>) {
    unsigned short *address_as_u16 = reinterpret_cast<unsigned short *>(address);
    unsigned short val_as_u16 = BitCastToU16(casted);
    unsigned short old = *address_as_u16;
    while (static_cast<float>(casted) <
           static_cast<float>(BitCastFromU16<RawT>(old))) {
      unsigned short assumed = old;
      old = atomicCAS(address_as_u16, assumed, val_as_u16);
      if (assumed == old) {
        break;
      }
    }
  } else {
    using NT1 = typename normalize_atomic_type<RawT>::type;
    atomicMin(reinterpret_cast<NT1 *>(address), static_cast<NT1>(casted));
  }
}

template <typename T1, typename T2>
TL_DEVICE T1 AtomicMinRet(T1 &ref, T2 val, int memory_order = 0) {
  (void)memory_order;
  using RawT = std::remove_cv_t<T1>;
  RawT *address = reinterpret_cast<RawT *>(&ref);
  RawT casted = static_cast<RawT>(val);
  if constexpr (std::is_same_v<RawT, float>) {
    int *address_as_i32 = reinterpret_cast<int *>(address);
    int old = *address_as_i32;
    while (casted < BitCastFromI32<float>(old)) {
      int assumed = old;
      old = atomicCAS(address_as_i32, assumed, BitCastToI32(casted));
      if (assumed == old) {
        break;
      }
    }
    return static_cast<T1>(BitCastFromI32<float>(old));
  } else if constexpr (std::is_same_v<RawT, half_t> ||
                       std::is_same_v<RawT, bfloat16_t>) {
    unsigned short *address_as_u16 = reinterpret_cast<unsigned short *>(address);
    unsigned short val_as_u16 = BitCastToU16(casted);
    unsigned short old = *address_as_u16;
    while (static_cast<float>(casted) <
           static_cast<float>(BitCastFromU16<RawT>(old))) {
      unsigned short assumed = old;
      old = atomicCAS(address_as_u16, assumed, val_as_u16);
      if (assumed == old) {
        break;
      }
    }
    return static_cast<T1>(BitCastFromU16<RawT>(old));
  } else {
    using NT1 = typename normalize_atomic_type<RawT>::type;
    return static_cast<T1>(
        atomicMin(reinterpret_cast<NT1 *>(address), static_cast<NT1>(casted)));
  }
}

template <typename T1, typename T2>
TL_DEVICE void AtomicAdd(T1 &ref, T2 val, int memory_order = 0) {
  (void)memory_order;
  using RawT = std::remove_cv_t<T1>;
  using NT1 = typename normalize_atomic_type<RawT>::type;
  RawT *address = reinterpret_cast<RawT *>(&ref);
  atomicAdd(reinterpret_cast<NT1 *>(address), static_cast<NT1>(val));
}

template <typename T1, typename T2>
TL_DEVICE T1 AtomicAddRet(T1 &ref, T2 val, int memory_order = 0) {
  (void)memory_order;
  using RawT = std::remove_cv_t<T1>;
  using NT1 = typename normalize_atomic_type<RawT>::type;
  RawT *address = reinterpret_cast<RawT *>(&ref);
  return static_cast<T1>(
      atomicAdd(reinterpret_cast<NT1 *>(address), static_cast<NT1>(val)));
}

// Helper to get integer type of same size for atomic operations
template <typename T> struct atomic_int_type;
template <> struct atomic_int_type<float> {
  using type = int;
};
template <> struct atomic_int_type<double> {
  using type = long long;
};
template <> struct atomic_int_type<int> {
  using type = int;
};
template <> struct atomic_int_type<long long> {
  using type = long long;
};
template <> struct atomic_int_type<unsigned int> {
  using type = unsigned int;
};
template <> struct atomic_int_type<unsigned long long> {
  using type = unsigned long long;
};

template <typename T>
TL_DEVICE std::remove_cv_t<T> AtomicLoad(const T &ref, int memory_order) {
  using ValueType = std::remove_cv_t<T>;
  using IntType = typename atomic_int_type<ValueType>::type;
  const IntType *address = reinterpret_cast<const IntType *>(&ref);
  IntType loaded = __atomic_load_n(address, memory_order);
  union {
    IntType bits;
    ValueType value;
  } caster;
  caster.bits = loaded;
  return caster.value;
}

template <typename T1, typename T2>
TL_DEVICE void AtomicStore(T1 &ref, T2 value, int memory_order) {
  using ValueType = std::remove_cv_t<T1>;
  using IntType = typename atomic_int_type<ValueType>::type;
  ValueType val = static_cast<ValueType>(value);
  IntType *address = reinterpret_cast<IntType *>(&ref);
  union {
    ValueType value;
    IntType bits;
  } caster;
  caster.value = val;
  __atomic_store_n(address, caster.bits, memory_order);
}

// AtomicAddx2 for half_t
TL_DEVICE void AtomicAddx2(half_t *ref, const half_t *val,
                           int memory_order = 0) {
  AtomicAdd(ref[0], val[0], memory_order);
  AtomicAdd(ref[1], val[1], memory_order);
}

template <typename ValType,
          typename std::enable_if<!std::is_pointer<ValType>::value,
                                  int>::type = 0>
TL_DEVICE void AtomicAddx2(half_t *ref, ValType val, int memory_order = 0) {
  const half_t *val_ptr = reinterpret_cast<const half_t *>(&val);
  AtomicAddx2(ref, val_ptr, memory_order);
}

// AtomicAddx2 for bfloat16_t
TL_DEVICE void AtomicAddx2(bfloat16_t *ref, const bfloat16_t *val,
                           int memory_order = 0) {
  AtomicAdd(ref[0], val[0], memory_order);
  AtomicAdd(ref[1], val[1], memory_order);
}

template <typename ValType,
          typename std::enable_if<!std::is_pointer<ValType>::value,
                                  int>::type = 0>
TL_DEVICE void AtomicAddx2(bfloat16_t *ref, ValType val,
                           int memory_order = 0) {
  const bfloat16_t *val_ptr = reinterpret_cast<const bfloat16_t *>(&val);
  AtomicAddx2(ref, val_ptr, memory_order);
}

// AtomicAddx2 for float
TL_DEVICE void AtomicAddx2(float *ref, const float *val, int memory_order = 0) {
  AtomicAdd(ref[0], val[0], memory_order);
  AtomicAdd(ref[1], val[1], memory_order);
}

template <typename ValType,
          typename std::enable_if<!std::is_pointer<ValType>::value,
                                  int>::type = 0>
TL_DEVICE void AtomicAddx2(float *ref, ValType val, int memory_order = 0) {
  const float *val_ptr = reinterpret_cast<const float *>(&val);
  AtomicAddx2(ref, val_ptr, memory_order);
}

// AtomicAddx4 for float
TL_DEVICE void AtomicAddx4(float *ref, const float *val, int memory_order = 0) {
  AtomicAdd(ref[0], val[0], memory_order);
  AtomicAdd(ref[1], val[1], memory_order);
  AtomicAdd(ref[2], val[2], memory_order);
  AtomicAdd(ref[3], val[3], memory_order);
}

template <typename ValType,
          typename std::enable_if<!std::is_pointer<ValType>::value,
                                  int>::type = 0>
TL_DEVICE void AtomicAddx4(float *ref, ValType val, int memory_order = 0) {
  const float *val_ptr = reinterpret_cast<const float *>(&val);
  AtomicAddx4(ref, val_ptr, memory_order);
}

// AtomicAddx2Ret for half_t
TL_DEVICE half2 AtomicAddx2Ret(half_t *ref, const half_t *val,
                               int memory_order = 0) {
  half2 ret;
  half_t *ret_ptr = reinterpret_cast<half_t *>(&ret);
  ret_ptr[0] = AtomicAddRet(ref[0], val[0], memory_order);
  ret_ptr[1] = AtomicAddRet(ref[1], val[1], memory_order);
  return ret;
}

template <typename ValType,
          typename std::enable_if<!std::is_pointer<ValType>::value,
                                  int>::type = 0>
TL_DEVICE half2 AtomicAddx2Ret(half_t *ref, ValType val,
                               int memory_order = 0) {
  const half_t *val_ptr = reinterpret_cast<const half_t *>(&val);
  return AtomicAddx2Ret(ref, val_ptr, memory_order);
}

// AtomicAddx2Ret for bfloat16_t
TL_DEVICE __mt_bfloat162 AtomicAddx2Ret(bfloat16_t *ref,
                                        const bfloat16_t *val,
                                        int memory_order = 0) {
  __mt_bfloat162 ret;
  bfloat16_t *ret_ptr = reinterpret_cast<bfloat16_t *>(&ret);
  ret_ptr[0] = AtomicAddRet(ref[0], val[0], memory_order);
  ret_ptr[1] = AtomicAddRet(ref[1], val[1], memory_order);
  return ret;
}

template <typename ValType,
          typename std::enable_if<!std::is_pointer<ValType>::value,
                                  int>::type = 0>
TL_DEVICE __mt_bfloat162 AtomicAddx2Ret(bfloat16_t *ref, ValType val,
                                        int memory_order = 0) {
  const bfloat16_t *val_ptr = reinterpret_cast<const bfloat16_t *>(&val);
  return AtomicAddx2Ret(ref, val_ptr, memory_order);
}

// AtomicAddx2Ret for float
TL_DEVICE float2 AtomicAddx2Ret(float *ref, const float *val,
                                int memory_order = 0) {
  float2 ret;
  ret.x = AtomicAddRet(ref[0], val[0], memory_order);
  ret.y = AtomicAddRet(ref[1], val[1], memory_order);
  return ret;
}

template <typename ValType,
          typename std::enable_if<!std::is_pointer<ValType>::value,
                                  int>::type = 0>
TL_DEVICE float2 AtomicAddx2Ret(float *ref, ValType val, int memory_order = 0) {
  const float *val_ptr = reinterpret_cast<const float *>(&val);
  return AtomicAddx2Ret(ref, val_ptr, memory_order);
}

// AtomicAddx4Ret for float
TL_DEVICE float4 AtomicAddx4Ret(float *ref, const float *val,
                                int memory_order = 0) {
  float4 ret;
  ret.x = AtomicAddRet(ref[0], val[0], memory_order);
  ret.y = AtomicAddRet(ref[1], val[1], memory_order);
  ret.z = AtomicAddRet(ref[2], val[2], memory_order);
  ret.w = AtomicAddRet(ref[3], val[3], memory_order);
  return ret;
}

template <typename ValType,
          typename std::enable_if<!std::is_pointer<ValType>::value,
                                  int>::type = 0>
TL_DEVICE float4 AtomicAddx4Ret(float *ref, ValType val, int memory_order = 0) {
  const float *val_ptr = reinterpret_cast<const float *>(&val);
  return AtomicAddx4Ret(ref, val_ptr, memory_order);
}
