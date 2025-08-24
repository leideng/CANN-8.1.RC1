//===----------------------------------------------------------------------===//
//
// Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
//
//===----------------------------------------------------------------------===//

#if !defined(__CCE_DEFINES_H__)
#define __CCE_DEFINES_H__

#define __no_return__ __attribute__((noreturn))

#define __noinline__ __attribute__((noinline))

#define __sync_noalias__ __attribute__((sync_noalias))

#define __sync_alias__ __attribute__((sync_alias))

#define __check_sync_alias__ __attribute__((check_sync_alias))

#define __sync_in__ __attribute__((sync_in))

#define __sync_out__ __attribute__((sync_out))

#define __in_pipe__(...) __attribute__((in_pipe(#__VA_ARGS__)))

#define __out_pipe__(...) __attribute__((out_pipe(#__VA_ARGS__)))

#define __inout_pipe__(...) __attribute__((inout_pipe(#__VA_ARGS__)))

#define __forceinline__                                                        \
  __inline__ __attribute__((cce_builtin_api, always_inline))
#define __align__(n) __attribute__((aligned(n)))

#define __global__ __attribute__((cce_kernel))

#define __gm__ __attribute__((cce_global))
#define __ca__ __attribute__((cce_cube_a))
#define __cb__ __attribute__((cce_cube_b))
#define __cc__ __attribute__((cce_cube_c))
#define __ubuf__ __attribute__((cce_unif_buff))
#define __cbuf__ __attribute__((cce_cube_buff))
#define __fbuf__ __attribute__((cce_fixpipe_buff))

#define __copyval__ __attribute__((copy_value))
#define __no_specialization__ __attribute__((no_specialization))

/// CCE Read Only
/// This is to mark flowtable structs as read only to help guide alias analysis
#define __device_immutable__ __attribute__((device_immutable))

#define __device_builtin__

#endif /* !__CCE_DEFINES_H__ */
