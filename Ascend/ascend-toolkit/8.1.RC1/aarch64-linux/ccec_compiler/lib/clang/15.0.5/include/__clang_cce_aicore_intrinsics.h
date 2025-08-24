//===----------------------------------------------------------------------===//
//
// Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef __CCE_INTRINSICS_H__
#define __CCE_INTRINSICS_H__

#define CCE_INTRINSIC                                                          \
  static __attribute__((overloadable, cce_builtin_api, always_inline))

// Just used for get_ub_virtual_address()
#define CCE_GET_VA_INTRINSIC                                                   \
  static __attribute__((overloadable, always_inline, cce_empty_func_body))

#ifdef __CCE_AICORE__

#define CCE_TRUE 1
#define CCE_FALSE 0

#ifdef __CCE_DEBUG_MODE__
#if defined(__DAV_C100__) || defined(__DAV_M100__)
#define debug_assert(expr)                                                     \
  if (!(expr))                                                                 \
    __builtin_cce_debug_assert_fail(#expr, __FILE__, __LINE__);
#include <type_traits>
enum class MType { NA, UB, CBUF, GM, CA, CB, CC, FBUF };
enum class DType { NA, U8, S8, U16, S16, U32, S32, FP16, FP32 };
#define AS_NA std::integral_constant<MType, MType::NA>()
#define AS_UB std::integral_constant<MType, MType::UB>()
#define AS_CBUF std::integral_constant<MType, MType::CBUF>()
#define AS_GM std::integral_constant<MType, MType::GM>()
#define AS_CA std::integral_constant<MType, MType::CA>()
#define AS_CB std::integral_constant<MType, MType::CB>()
#define AS_CC std::integral_constant<MType, MType::CC>()
#define AS_FBUF std::integral_constant<MType, MType::FBUF>()

#define DT_NA std::integral_constant<DType, DType::NA>()
#define DT_U8 std::integral_constant<DType, DType::U8>()
#define DT_S8 std::integral_constant<DType, DType::S8>()
#define DT_U16 std::integral_constant<DType, DType::U16>()
#define DT_S16 std::integral_constant<DType, DType::S16>()
#define DT_U32 std::integral_constant<DType, DType::U32>()
#define DT_S32 std::integral_constant<DType, DType::S32>()
#define DT_FP16 std::integral_constant<DType, DType::FP16>()
#define DT_FP32 std::integral_constant<DType, DType::FP32>()

#define INVALID_VALUE "The addrspace type is not valid"
#define ERROR_VALUE_AS "The addrspace argument can not be: NA, FBUF"
#define ERROR_VALUE_DT "The datatype argument can not be: NA"

#define READ_US(BIT_WIDTH)                                                     \
  template <typename T>                                                        \
  CCE_INTRINSIC[aicore] uint##BIT_WIDTH##_t debug_read_u##BIT_WIDTH(           \
      T addrspace, uint64_t addr) {                                            \
    static_assert(std::is_class<T>::value, INVALID_VALUE);                     \
    static_assert(addrspace.value != AS_NA.value &&                            \
                      addrspace.value != AS_FBUF.value,                        \
                  ERROR_VALUE_AS);                                             \
    return debug_read((uint8_t)addrspace.value, (uint8_t)DType::U##BIT_WIDTH,  \
                      addr);                                                   \
  }                                                                            \
  template <typename T>                                                        \
  CCE_INTRINSIC[aicore] int##BIT_WIDTH##_t debug_read_s##BIT_WIDTH(            \
      T addrspace, uint64_t addr) {                                            \
    static_assert(std::is_class<T>::value, INVALID_VALUE);                     \
    static_assert(addrspace.value != AS_NA.value &&                            \
                      addrspace.value != AS_FBUF.value,                        \
                  ERROR_VALUE_AS);                                             \
    union {                                                                    \
      int##BIT_WIDTH##_t s;                                                    \
      uint##BIT_WIDTH##_t u;                                                   \
    } res;                                                                     \
    res.u = debug_read((uint8_t)addrspace.value, (uint8_t)DType::U##BIT_WIDTH, \
                       addr);                                                  \
                                                                               \
    return res.s;                                                              \
  }

READ_US(8)
READ_US(16)
READ_US(32)
#undef READ_US

#define READ_FP(BIT_WIDTH, RET_TYPE)                                           \
  template <typename T>                                                        \
  CCE_INTRINSIC[aicore] RET_TYPE debug_read_f##BIT_WIDTH(T addrspace,          \
                                                         uint64_t addr) {      \
    static_assert(std::is_class<T>::value, INVALID_VALUE);                     \
    static_assert(addrspace.value != AS_NA.value &&                            \
                      addrspace.value != AS_FBUF.value,                        \
                  ERROR_VALUE_AS);                                             \
    union {                                                                    \
      RET_TYPE f;                                                              \
      uint##BIT_WIDTH##_t u;                                                   \
    } res;                                                                     \
    res.u = debug_read((uint8_t)addrspace.value, (uint8_t)DType::U##BIT_WIDTH, \
                       addr);                                                  \
                                                                               \
    return res.f;                                                              \
  }

READ_FP(16, half)
READ_FP(32, float)
#undef READ_FP

template <typename T1, typename T2>
CCE_INTRINSIC[aicore] void debug_dump(__gm__ const char *__format, T1 addrspace,
                                      uint64_t addr, T2 dtype, uint64_t num)
    __attribute__((format(debug_dump, 1, 0))) {
  static_assert(std::is_class<T1>::value, INVALID_VALUE);
  static_assert(std::is_class<T2>::value, INVALID_VALUE);
  static_assert(addrspace.value != AS_NA.value &&
                    addrspace.value != AS_FBUF.value,
                ERROR_VALUE_AS);
  static_assert(dtype.value != DT_NA.value, ERROR_VALUE_DT);
  __builtin_cce_debug_dump(__format, (uint8_t)addrspace.value, addr,
                           (uint8_t)dtype.value, num);
}

#endif // __DAV_C100__, __DAV_M100__
#endif // __CCE_DEBUG_MODE__

#endif // ifdef __CCE_AICORE__

#if defined(__DAV_C100__) || defined(__DAV_M200__) ||                          \
    defined(__DAV_M200_VEC__) || defined(__DAV_M201___) ||                     \
    defined(__DAV_M210_VEC__) || defined(__DAV_C220_CUBE__) ||                 \
    defined(__DAV_C220_VEC__)
CCE_INTRINSIC[aicore] void trap(uint64_t err_code) {
  __builtin_cce_trap_mov(err_code);
}
#endif // __DAV_C100__, __DAV_M200__, __DAV_M200_VEC__, __DAV_M201___,
       // __DAV_M210_VEC__, __DAV_C220_CUBE__, __DAV_C220_VEC__

#undef CCE_INTRINSIC
#endif // defined(__CLANG_CCE_INTRINSICS_H__)
