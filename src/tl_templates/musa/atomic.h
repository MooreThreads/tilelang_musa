#pragma once

#include <musa_fp16.h>
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

template <typename T1, typename T2>
TL_DEVICE void AtomicMax(T1 &ref, T2 val, int memory_order = 0) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  T1 *address = &ref;
  atomicMax(reinterpret_cast<NT1 *>(address), static_cast<NT1>(val));
}

template <typename T1, typename T2>
TL_DEVICE T1 AtomicMaxRet(T1 &ref, T2 val, int memory_order = 0) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  T1 *address = &ref;
  return static_cast<T1>(
      atomicMax(reinterpret_cast<NT1 *>(address), static_cast<NT1>(val)));
}

template <typename T1, typename T2>
TL_DEVICE void AtomicMin(T1 &ref, T2 val, int memory_order = 0) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  T1 *address = &ref;
  atomicMin(reinterpret_cast<NT1 *>(address), static_cast<NT1>(val));
}

template <typename T1, typename T2>
TL_DEVICE T1 AtomicMinRet(T1 &ref, T2 val, int memory_order = 0) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  T1 *address = &ref;
  return static_cast<T1>(
      atomicMin(reinterpret_cast<NT1 *>(address), static_cast<NT1>(val)));
}

template <typename T1, typename T2>
TL_DEVICE void AtomicAdd(T1 &ref, T2 val, int memory_order = 0) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  T1 *address = &ref;
  atomicAdd(reinterpret_cast<NT1 *>(address), static_cast<NT1>(val));
}

template <typename T1, typename T2>
TL_DEVICE T1 AtomicAddRet(T1 &ref, T2 val, int memory_order = 0) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  T1 *address = &ref;
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

template <typename T> TL_DEVICE T AtomicLoad(T &ref, int memory_order) {
  using IntType = typename atomic_int_type<T>::type;
  IntType *address = reinterpret_cast<IntType *>(&ref);
  IntType loaded = __atomic_load_n(address, memory_order);
  return *reinterpret_cast<T *>(&loaded);
}

template <typename T1, typename T2>
TL_DEVICE void AtomicStore(T1 &ref, T2 value, int memory_order) {
  using IntType = typename atomic_int_type<T1>::type;
  T1 val = static_cast<T1>(value);
  IntType *address = reinterpret_cast<IntType *>(&ref);
  IntType int_val = *reinterpret_cast<IntType *>(&val);
  __atomic_store_n(address, int_val, memory_order);
}

// AtomicAddx2 for half_t
TL_DEVICE void AtomicAddx2(half_t *ref, half_t *val, int memory_order = 0) {
  atomicAdd(reinterpret_cast<half2 *>(ref),
            static_cast<half2>(*reinterpret_cast<half2 *>(val)));
}

// AtomicAddx2 for bfloat16_t
TL_DEVICE void AtomicAddx2(bfloat16_t *ref, bfloat16_t *val,
                           int memory_order = 0) {
  atomicAdd(
      reinterpret_cast<__mt_bfloat162 *>(ref),
      static_cast<__mt_bfloat162>(*reinterpret_cast<__mt_bfloat162 *>(val)));
}

// AtomicAddx2 for float
TL_DEVICE void AtomicAddx2(float *ref, float *val, int memory_order = 0) {
  atomicAdd(reinterpret_cast<float2 *>(ref),
            static_cast<float2>(*reinterpret_cast<float2 *>(val)));
}

// AtomicAddx4 for float
TL_DEVICE void AtomicAddx4(float *ref, float *val, int memory_order = 0) {
  atomicAdd(reinterpret_cast<float4 *>(ref),
            static_cast<float4>(*reinterpret_cast<float4 *>(val)));
}

// AtomicAddx2Ret for half_t
TL_DEVICE half2 AtomicAddx2Ret(half_t *ref, half_t *val, int memory_order = 0) {
  return atomicAdd(reinterpret_cast<half2 *>(ref),
                   static_cast<half2>(*reinterpret_cast<half2 *>(val)));
}

// AtomicAddx2Ret for bfloat16_t
TL_DEVICE __mt_bfloat162 AtomicAddx2Ret(bfloat16_t *ref, bfloat16_t *val,
                                        int memory_order = 0) {
  return atomicAdd(
      reinterpret_cast<__mt_bfloat162 *>(ref),
      static_cast<__mt_bfloat162>(*reinterpret_cast<__mt_bfloat162 *>(val)));
}

// AtomicAddx2Ret for float
TL_DEVICE float2 AtomicAddx2Ret(float *ref, float *val, int memory_order = 0) {
  return atomicAdd(reinterpret_cast<float2 *>(ref),
                   static_cast<float2>(*reinterpret_cast<float2 *>(val)));
}

// AtomicAddx4Ret for float
TL_DEVICE float4 AtomicAddx4Ret(float *ref, float *val, int memory_order = 0) {
  return atomicAdd(reinterpret_cast<float4 *>(ref),
                   static_cast<float4>(*reinterpret_cast<float4 *>(val)));
}
