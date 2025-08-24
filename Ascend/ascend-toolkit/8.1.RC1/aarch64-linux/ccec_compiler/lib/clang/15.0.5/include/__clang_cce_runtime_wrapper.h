//===----------------------------------------------------------------------===//
//
// Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
//
//===----------------------------------------------------------------------===//

/*
 * WARNING: This header is intended to be directly -include'd by
 * the compiler and is not supposed to be included by users.
 */

#ifndef __CCE_RUNTIME_WRAPPER_H__
#define __CCE_RUNTIME_WRAPPER_H__

#if defined(__CCE__) && defined(__clang__)

#include <stddef.h>
#include <stdint.h>

#ifndef __CCE_ARCH__
#define __CCE_ARCH__ 100
#endif

#ifndef __CCE_AICPU_NO_FIRMWARE__
#include "__clang_cce_types.h"

#include "__clang_cce_aicore_builtin_vars.h"

#include "__clang_cce_runtime.h"

#include "__clang_cce_aicore_intrinsics.h"

#include "__clang_cce_aicore_functions.h"

#include "__clang_cce_aicpu_neon.h"

// May need vector type for host compilation if the inline vector functions
// have vector type arguments
#include "__clang_cce_vector_types.h"

#elif defined(__CCE_NON_AICPU_CODES_NO_FIRMWARE__)
#include "__clang_cce_types.h"

#include "__clang_cce_aicore_builtin_vars.h"

#include "__clang_cce_runtime.h"

#include "__clang_cce_aicore_intrinsics.h"

#include "__clang_cce_aicore_functions.h"

#else // CCE AICPU CODES in no firmware mode
#include "__clang_cce_types.h"

#include "__clang_cce_aicore_builtin_vars.h"

#include "__clang_cce_runtime.h"

#include "__clang_cce_aicpu_neon.h"

#endif // __CCE_AICPU_NO_FIRMWARE__

#if (defined __DAV_L210__) || (defined __DAV_T210__) ||                        \
    (defined __DAV_M210_VEC__) || (defined __DAV_M300__) ||                    \
    (defined __DAV_L300__) || (defined __DAV_L300_VEC__) ||                    \
    (defined __DAV_T300__) || (defined __DAV_C310__) ||                        \
    (defined __DAV_M310__) || (defined __DAV_L310__)
#include "__clang_cce_vector_intrinsics.h"
#endif

#ifdef __CCE_AICORE_EN_TL__
namespace TL {};
#endif
// Tensor related types can only be used in AICore.
#if (defined __CCE_AICORE_EN_TL__) && (defined __CCE_IS_AICORE__)
#include "tl_lib/__clang_tl_ops.h"
#endif

// For SIMT VF
#ifdef __CCE_AICORE_SUPPORT_SIMT__
#include "__clang_cce_simt.h"
#endif // __CCE_AICORE_SUPPORT_SIMT__

#ifdef __cplusplus
extern "C" {
#endif
inline __attribute__((alway_inline)) int __cce_getOrSetBlockNum(int value,
                                                                int type) {
  static thread_local int local = 0;
  if (type == 0)
    local = value;
  return local;
}
#ifdef __cplusplus
}
#endif
inline __attribute__((alway_inline)) unsigned int
__cce_rtConfigureCall(unsigned int numBlocks, void *smDesc = nullptr,
                      void *stream = nullptr) {
  __cce_getOrSetBlockNum(numBlocks, 0);
  return rtConfigureCall(numBlocks, smDesc, stream);
}

#if (defined __CCE_ENABLE_ASAN__)
#include "asan/__clang_cce_sanitizer.h"
#endif
#if (defined __CCE_ENABLE_MSTRACE__)
#include "mstrace/__clang_cce_mstrace.h"
#endif
#if (defined __CCE_ENABLE_PRINT__)
#include "ccelib/__ccelib.h"
#endif

#if (defined __CCE_ENABLE_PRINT_AICORE__)
#include "ccelib/__ccelib_aicore.h"
#endif

#endif // __CCE__
#endif // __CCE_RUNTIME_WRAPPER_H__
