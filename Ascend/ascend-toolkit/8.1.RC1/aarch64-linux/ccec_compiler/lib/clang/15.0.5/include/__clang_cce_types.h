//===----------------------------------------------------------------------===//
//
// Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
//
//===----------------------------------------------------------------------===//

#if !defined(__CCE_TYPES_H__)
#define __CCE_TYPES_H__

#include "__clang_cce_defines.h"

#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_C220_CUBE__) ||                 \
    defined(__DAV_C220_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__) || (defined __DAV_L310__)
typedef __bf16 bfloat16_t;
#endif

#if defined(__DAV_M300__) || defined(__DAV_C310__)
typedef __hif8 hifloat8_t;
#endif

struct __device_builtin__ dim3 {
  unsigned int x, y, z;
#if defined(__cplusplus)
  dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
      : x(vx), y(vy), z(vz) {}
#endif /* __cplusplus */
};

typedef __device_builtin__ struct dim3 dim3;
#define BISHENG_BUILTIN_TYPE_PROXY(NAME) __BISHENG_BUILTIN_TYPE_PROXY_##NAME

#define DEFINE_BISHENG_BUILTIN_TYPE_PROXY(NAME, BYTE_SIZE)                     \
  struct [[bisheng::mangling_hint(NAME)]] BISHENG_BUILTIN_TYPE_PROXY(NAME) {   \
    BISHENG_BUILTIN_TYPE_PROXY(NAME)() = delete;                               \
    BISHENG_BUILTIN_TYPE_PROXY(NAME)                                           \
    (const BISHENG_BUILTIN_TYPE_PROXY(NAME) &) = delete;                       \
    BISHENG_BUILTIN_TYPE_PROXY(NAME)                                           \
    (BISHENG_BUILTIN_TYPE_PROXY(NAME) &&) = delete;                            \
                                                                               \
  private:                                                                     \
    char Storage[BYTE_SIZE];                                                   \
  }

namespace bisheng {
DEFINE_BISHENG_BUILTIN_TYPE_PROXY(BFloat16, 2);
DEFINE_BISHENG_BUILTIN_TYPE_PROXY(HFloat8, 1);
DEFINE_BISHENG_BUILTIN_TYPE_PROXY(HFloat4x2, 1);
DEFINE_BISHENG_BUILTIN_TYPE_PROXY(Float8_E4M3, 1);
DEFINE_BISHENG_BUILTIN_TYPE_PROXY(Float8_E5M2, 1);
DEFINE_BISHENG_BUILTIN_TYPE_PROXY(Float4_E2M1, 1);
DEFINE_BISHENG_BUILTIN_TYPE_PROXY(Float4_E1M2, 1);
} // namespace bisheng

#ifndef __CCE_AICORE__
// #ifdef __BISHENG_SUPPORT_BFLOAT16__
// typedef __bf16 bfloat16_t;
// #else
typedef bisheng::BISHENG_BUILTIN_TYPE_PROXY(BFloat16) bfloat16_t;
// #endif

typedef bisheng::BISHENG_BUILTIN_TYPE_PROXY(HFloat8) hifloat8_t;
typedef bisheng::BISHENG_BUILTIN_TYPE_PROXY(HFloat4x2) hifloat4x2_t;
typedef bisheng::BISHENG_BUILTIN_TYPE_PROXY(Float8_E4M3) float8_e4m3_t;
typedef bisheng::BISHENG_BUILTIN_TYPE_PROXY(Float8_E5M2) float8_e5m2_t;
typedef bisheng::BISHENG_BUILTIN_TYPE_PROXY(Float4_E2M1) float4_e2m1_t;
typedef bisheng::BISHENG_BUILTIN_TYPE_PROXY(Float4_E1M2) float4_e1m2_t;
#endif

#define PIPE_LSU1 _Pragma("GCC warning \"'PIPE_LSU1' is deprecated\"") PIPE_MTE1
#define PIPE_LSU2 _Pragma("GCC warning \"'PIPE_LSU2' is deprecated\"") PIPE_MTE2
#define PIPE_LSU3 _Pragma("GCC warning \"'PIPE_LSU3' is deprecated\"") PIPE_MTE3

#if __CCE_AICORE__ == 100
#define PIPE_MTE4                                                              \
  _Pragma("GCC error \"'PIPE_MTE4' is only available in v200 targets\"")       \
      PIPE_MTE2
#define PIPE_MTE5                                                              \
  _Pragma("GCC error \"'PIPE_MTE5' is only available in v200 targets\"")       \
      PIPE_MTE3
#define PIPE_V2                                                                \
  _Pragma("GCC error \"'PIPE_V2' is only available in v200 targets\"") PIPE_V
#define EVENT_ID4                                                              \
  _Pragma("GCC error \"'EVENT_ID4' is only available in v200 targets\"")       \
      EVENT_ID0
#define EVENT_ID5                                                              \
  _Pragma("GCC error \"'EVENT_ID5' is only available in v200 targets\"")       \
      EVENT_ID1
#define EVENT_ID6                                                              \
  _Pragma("GCC error \"'EVENT_ID6' is only available in v200 targets\"")       \
      EVENT_ID2
#define EVENT_ID7                                                              \
  _Pragma("GCC error \"'EVENT_ID7' is only available in v200 targets\"")       \
      EVENT_ID3
#endif

#if (defined __DAV_L210__) || (defined __DAV_T210__)
#define PIPE_V2                                                                \
  _Pragma("GCC error \"'PIPE_V2' is not available in dav-t210 and dav-l210\"") \
      PIPE_V
#endif

#include "cce_aicore_intrinsics.h"

#endif /* !__CCE_TYPES_H__ */
