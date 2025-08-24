//===----------------------------------------------------------------------===//
//
// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef __CLANG_CCE_VECTOR_TYPES_H
#define __CLANG_CCE_VECTOR_TYPES_H

#ifndef __VECTOR_ADDRESS__
#define __VECTOR_ADDRESS__ // Add macro for vector_address to check whether the
                           // compiler needs users use vector_address for the
                           // return value type of vag or not.

#include <stdint.h>

// The vector register size in bytes.
#if defined(__CCE_VF_VEC_LEN__)
#warning The vector size is manually reset.
constexpr uint16_t CCE_VL = __CCE_VF_VEC_LEN__;
#elif defined(__DAV_L210__) || (defined __DAV_L310__)
constexpr uint16_t CCE_VL = 128;
#elif defined(__DAV_T210__)
constexpr uint16_t CCE_VL = 64;
#elif defined(__DAV_M210_VEC__) || defined(__DAV_M300__) ||                    \
    defined(__DAV_L300__) || defined(__DAV_L300_VEC__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_M310__)
constexpr uint16_t CCE_VL = 256;
#elif defined(__DAV_T300__)
constexpr uint16_t CCE_VL = 32;
#endif // __CCE_VF_VEC_LEN__

#if defined(__DAV_L210__) || defined(__DAV_T210__) ||                          \
    defined(__DAV_M210_VEC__) || defined(__DAV_M300__) ||                      \
    defined(__DAV_L300__) || defined(__DAV_L300_VEC__) ||                      \
    defined(__DAV_C310__) || defined(__DAV_T300__) || defined(__DAV_M310__) || \
    (defined __DAV_L310__)
static_assert(CCE_VL > 0 && CCE_VL % 4 == 0 && "CCE_VL must be a power of 4");
enum ELE_CNT : uint16_t {
  ELE_CNT_B8 = CCE_VL,
  ELE_CNT_B16 = CCE_VL / 2,
  ELE_CNT_B32 = CCE_VL / 4,
};
#endif

#ifdef __CCE_VF_VEC_LEN__
#define __CCE_VF_VEC_LEN_ATTR__ cce_vf_vl(__CCE_VF_VEC_LEN__)
#define __CCE_VF_PRE_LEN_ATTR__ cce_vf_vl(__CCE_VF_VEC_LEN__ / 8)
// The alignment of vector must be power of 2.
// The width of "Wide Vector Register" is 3 times the "Vector Registers";
#define __CCE_VF_WVEC_LEN_ATTR__ cce_vf_vl(__CCE_VF_VEC_LEN__ * 4)
#else // __CCE_VF_VEC_LEN__
#define __CCE_VF_VEC_LEN_ATTR__
#define __CCE_VF_PRE_LEN_ATTR__
#define __CCE_VF_WVEC_LEN_ATTR__
#endif // __CCE_VF_VEC_LEN__

typedef long long vector_s64
    __attribute__((ext_vector_type(32), __CCE_VF_VEC_LEN_ATTR__)); // 32 x s64
typedef int vector_s32
    __attribute__((ext_vector_type(64), __CCE_VF_VEC_LEN_ATTR__)); // 64 x s32
typedef short vector_s16
    __attribute__((ext_vector_type(128), __CCE_VF_VEC_LEN_ATTR__)); // 128 x s16
typedef char vector_s8
    __attribute__((ext_vector_type(256), __CCE_VF_VEC_LEN_ATTR__)); // 256 x s8
typedef vector_s8 vector_s4x2;

typedef unsigned long long vector_u64
    __attribute__((ext_vector_type(32), __CCE_VF_VEC_LEN_ATTR__)); // 32 x u64
typedef unsigned int vector_u32
    __attribute__((ext_vector_type(64), __CCE_VF_VEC_LEN_ATTR__)); // 64 x u32
typedef unsigned short vector_u16
    __attribute__((ext_vector_type(128), __CCE_VF_VEC_LEN_ATTR__)); // 128 x u16
typedef unsigned char vector_u8
    __attribute__((ext_vector_type(256), __CCE_VF_VEC_LEN_ATTR__)); // 256 x u8
typedef float vector_f32
    __attribute__((ext_vector_type(64), __CCE_VF_VEC_LEN_ATTR__)); // 64 x f32
typedef half vector_f16
    __attribute__((ext_vector_type(128), __CCE_VF_VEC_LEN_ATTR__)); // 128 x f16
#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_C220_CUBE__) ||                 \
    defined(__DAV_C220_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
typedef bfloat16_t vector_bf16 __attribute__((
    ext_vector_type(128), __CCE_VF_VEC_LEN_ATTR__)); // 128 x bf16
#endif

// V4LLi bitcast to 256 x i1
typedef long long vector_bool
    __attribute__((ext_vector_type(4), __CCE_VF_PRE_LEN_ATTR__));

// Alignment Register is 32 Bytes, so to define it as v32i8.
typedef char vector_align_data __attribute__((ext_vector_type(32))); // 32 x i8

// Define a structure as Alignment Register for user.
// It will call default init builtin-func when user define a vector_align
// variable. so that we don't worry about Alignment Register undef result in
// compiler crash problem.
typedef struct vector_align {
  [aicore] __attribute__((cce_builtin_api, always_inline)) vector_align() {
    Data = __builtin_cce_init_vector_align_data();
  }

  [aicore] __attribute__((cce_builtin_api, always_inline))
  operator vector_align_data &() {
    return Data;
  }

  [aicore] __attribute__((cce_builtin_api, always_inline)) vector_align &
  operator=(vector_align_data alignData) {
    Data = alignData;
    return *this;
  }
  vector_align_data Data;
} vector_align;

typedef uint32_t vector_address __attribute__((ext_vector_type(1)));

// wide register type
// Compiler uses 256 x i32 for wvector_s24, 128 x i64 for wvector_s48, and
// 64 x i128 for wvector_s64
typedef int wvector_s24
    __attribute__((ext_vector_type(256), __CCE_VF_WVEC_LEN_ATTR__));
typedef long long wvector_s48
    __attribute__((ext_vector_type(128), __CCE_VF_WVEC_LEN_ATTR__));
typedef signed __int128 wvector_s64
    __attribute__((ext_vector_type(64), __CCE_VF_WVEC_LEN_ATTR__));

#undef __CCE_VF_VEC_LEN_ATTR__
#undef __CCE_VF_PRE_LEN_ATTR__
#undef __CCE_VF_WVEC_LEN_ATTR__

// struct vector
typedef struct vector_s64x2_t {
  vector_s64 val[2];
} vector_s64x2_t;

typedef struct vector_u64x2_t {
  vector_u64 val[2];
} vector_u64x2_t;

typedef struct vector_s32x2_t {
  vector_s32 val[2];
} vector_s32x2_t;

typedef struct vector_u32x2_t {
  vector_u32 val[2];
} vector_u32x2_t;

typedef struct vector_s16x2_t {
  vector_s16 val[2];
} vector_s16x2_t;

typedef struct vector_u16x2_t {
  vector_u16 val[2];
} vector_u16x2_t;

typedef struct vector_s8x2_t {
  vector_s8 val[2];
} vector_s8x2_t;

typedef struct vector_u8x2_t {
  vector_u8 val[2];
} vector_u8x2_t;

typedef struct vector_f32x2_t {
  vector_f32 val[2];
} vector_f32x2_t;

typedef struct vector_f16x2_t {
  vector_f16 val[2];
} vector_f16x2_t;

typedef struct vector_boolx2_t {
  vector_bool val[2];
} vector_boolx2_t;

#if defined(__DAV_C310__)

#define CCE_INTRINSIC                                                          \
  [aicore] inline __attribute__((cce_builtin_api, always_inline))

struct vector_2xvl_u64;

typedef struct vector_2xvl_s64 {
  vector_s32 val[2];

  CCE_INTRINSIC vector_2xvl_s64() = default;

  CCE_INTRINSIC explicit vector_2xvl_s64(vector_2xvl_u64 &v64u64);

  CCE_INTRINSIC ~vector_2xvl_s64() = default;
} vector_2xvl_s64;

typedef struct vector_2xvl_u64 {
  vector_u32 val[2];

  CCE_INTRINSIC vector_2xvl_u64() = default;

  CCE_INTRINSIC explicit vector_2xvl_u64(vector_2xvl_s64 &v64s64) {
    this->val[0] = (vector_u32)v64s64.val[0];
    this->val[1] = (vector_u32)v64s64.val[1];
  }

  CCE_INTRINSIC ~vector_2xvl_u64() = default;
} vector_2xvl_u64;

CCE_INTRINSIC vector_2xvl_s64::vector_2xvl_s64(vector_2xvl_u64 &v64u64) {
  this->val[0] = (vector_s32)v64u64.val[0];
  this->val[1] = (vector_s32)v64u64.val[1];
}

#undef CCE_INTRINSIC
#endif

#if defined(__DAV_M300__) || defined(__DAV_L300__) ||                          \
    defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||                      \
    defined(__DAV_C310__)
typedef struct vector_bf16x2_t {
  vector_bf16 val[2];
} vector_bf16x2_t;
#endif // defined(__DAV_M300__) || defined(__DAV_L300__) ||
       // defined(__DAV_L300_VEC__) || defined(__DAV_T300__) ||
       // defined(__DAV_C310__)

#endif //__VECTOR_ADDRESS__
#endif //__CLANG_CCE_VECTOR_TYPES_H
