#pragma once

#if defined(__MUSA_ARCH_LIST__) && (__MUSA_ARCH_LIST__ >= 310)
#include "gemm_mp31.h"
#elif defined(__MUSA_ARCH_LIST__) && (__MUSA_ARCH_LIST__ == 220)
#include "gemm_mp22.h"
#else
// No matching architecture found
#endif