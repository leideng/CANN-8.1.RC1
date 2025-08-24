
//===----------------------------------------------------------------------===//
//
// This file, different from __clang_cce_aicore_intrinsics.h, do not use macros
// in HiIPU.cpp to define intrinsics, which is not recommended any more.
// Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef __CLANG_CCE_AICORE_FUNCTIONS_H__
#define __CLANG_CCE_AICORE_FUNCTIONS_H__

#define CCE_INTRINSIC                                                          \
  static __attribute__((overloadable, cce_builtin_api, always_inline))

#define PIPE_ID(v) __attribute__((__pipe_type__(v)))

#if defined(__DAV_N350__) || defined(__DAV_C310__) || defined(__DAV_M310__)
#include <type_traits>
#endif

/*-------------------------------get_imm------------------------------------*/
#ifdef __CCE_GET_IMM
#define __get_imm1(imm0) (uint64_t)(imm0)
#define __get_imm2(imm0, imm1) __get_imm1(imm0) + (imm1 * (1 << 16))
#define __get_imm3(imm0, imm1, imm2)                                           \
  __get_imm2(imm0, imm1) + (imm2 * ((uint64_t)1 << 32))
#define __get_imm4(imm0, imm1, imm2, imm3)                                     \
  __get_imm3(imm0, imm1, imm2) + (imm3 * ((uint64_t)1 << 48))
#define __get_imm_va_args(_1, _2, _3, _4, __get_imm_select, ...)               \
  __get_imm_select

#define get_imm(...)                                                           \
  __get_imm_va_args(__VA_ARGS__, __get_imm4, __get_imm3, __get_imm2,           \
                    __get_imm1)(__VA_ARGS__)
#endif
/* Para sid is added in newest isa.
 * To be compatible with previous api in M310.
 * */
#if (defined __DAV_M310__)
#define LOAD_OUT_TO_DST(DST, ATYPE)                                            \
  CCE_INTRINSIC[aicore] void load_gm_to_##DST##_2dv2(                          \
      __##DST##__ ATYPE *dst, __gm__ ATYPE *src, uint32_t mStartPosition,      \
      uint32_t kStartPosition, int32_t srcStride, uint8_t dstStride,           \
      uint8_t mStep, uint8_t kStep) {                                          \
    load_gm_to_##DST##_2dv2(dst, src, mStartPosition, kStartPosition,          \
                            srcStride, dstStride, mStep, kStep, 0);            \
  }

LOAD_OUT_TO_DST(ca, int16_t);
LOAD_OUT_TO_DST(ca, int8_t);
LOAD_OUT_TO_DST(ca, int32_t);
LOAD_OUT_TO_DST(ca, int64_t);
LOAD_OUT_TO_DST(ca, uint8_t);
LOAD_OUT_TO_DST(ca, half);
LOAD_OUT_TO_DST(ca, float);
LOAD_OUT_TO_DST(ca, uint32_t);
LOAD_OUT_TO_DST(ca, void);

LOAD_OUT_TO_DST(cbuf, int16_t);
LOAD_OUT_TO_DST(cbuf, int8_t);
LOAD_OUT_TO_DST(cbuf, int32_t);
LOAD_OUT_TO_DST(cbuf, int64_t);
LOAD_OUT_TO_DST(cbuf, uint8_t);
LOAD_OUT_TO_DST(cbuf, half);
LOAD_OUT_TO_DST(cbuf, float);
LOAD_OUT_TO_DST(cbuf, uint32_t);
LOAD_OUT_TO_DST(cbuf, void);

LOAD_OUT_TO_DST(cb, int16_t);
LOAD_OUT_TO_DST(cb, int8_t);
LOAD_OUT_TO_DST(cb, int32_t);
LOAD_OUT_TO_DST(cb, int64_t);
LOAD_OUT_TO_DST(cb, uint8_t);
LOAD_OUT_TO_DST(cb, half);
LOAD_OUT_TO_DST(cb, float);
LOAD_OUT_TO_DST(cb, uint32_t);
LOAD_OUT_TO_DST(cb, void);
#undef LOAD_OUT_TO_DST
#endif

#if (defined __DAV_M310__) || (defined __DAV_L310__)

#define LOAD_L1_TO_L0A_WINOGRAD_V2(ATYPE)                                      \
  CCE_INTRINSIC[aicore] void load_cbuf_to_ca_winograd_v2(                      \
      __ca__ ATYPE *dst_addr, __cbuf__ ATYPE *scr_addr, uint64_t config0,      \
      uint64_t config1) {                                                      \
    uint64_t add = (uint64_t)scr_addr;                                         \
    load_cbuf_to_ca_winograd_v2(dst_addr, add, config0, config1);              \
  }                                                                            \
                                                                               \
  CCE_INTRINSIC[aicore] void load_cbuf_to_ca_winograd_v2(                      \
      __ca__ ATYPE *dst_addr, __cbuf__ ATYPE *bias_addr,                       \
      __cbuf__ ATYPE *padding_value_addr, uint16_t FMWidth, uint16_t FMHeight, \
      uint16_t FMChannel, uint8_t dstStride, uint8_t padModeH,                 \
      uint8_t padModeV, uint16_t stepK, uint16_t posK, uint16_t stepM,         \
      uint16_t posM, bool FMenable) {                                          \
    uint64_t add = (((uint64_t)padding_value_addr & 0xffffffffULL) << 32) |    \
                   ((uint64_t)bias_addr & 0xffffffffULL);                      \
    load_cbuf_to_ca_winograd_v2(dst_addr, add, FMWidth, FMHeight, FMChannel,   \
                                dstStride, padModeH, padModeV, stepK, posK,    \
                                stepM, posM, FMenable);                        \
  }

LOAD_L1_TO_L0A_WINOGRAD_V2(int8_t);
LOAD_L1_TO_L0A_WINOGRAD_V2(int16_t);

#define MOV_OUT_TO_UB_ALIGN(ATYPE)                                             \
  CCE_INTRINSIC[aicore] void copy_gm_to_ubuf(                                  \
      __ubuf__ ATYPE *dst_addr, __gm__ ATYPE *scr_addr, uint64_t config) {     \
    uint64_t config0 = config & 0xffffffffULL;                                 \
    uint64_t config1 = ((config & 0xffff00000000ULL) >> 32) |                  \
                       ((config & 0xffff000000000000ULL) >> 16);               \
    copy_gm_to_ubuf_align_no_padding(dst_addr, scr_addr, config0, config1);    \
  }                                                                            \
                                                                               \
  CCE_INTRINSIC[aicore] void copy_gm_to_ubuf(                                  \
      __ubuf__ ATYPE *dst_addr, __gm__ ATYPE *scr_addr, uint8_t sid,           \
      uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride,                  \
      uint16_t dstStride) {                                                    \
    uint32_t srcGap = (uint32_t)srcStride;                                     \
    uint32_t dstGap = (uint32_t)dstStride;                                     \
    uint32_t lBurst = (uint32_t)lenBurst;                                      \
    uint8_t leftPaddingNum = 0;                                                \
    uint8_t rightPaddingNum = 0;                                               \
    copy_gm_to_ubuf_align_no_padding(dst_addr, scr_addr, sid, nBurst, lBurst,  \
                                     leftPaddingNum, rightPaddingNum, srcGap,  \
                                     dstGap);                                  \
  }

MOV_OUT_TO_UB_ALIGN(void);
#undef MOV_OUT_TO_UB_ALIGN

#define MOV_L1_TO_OUT_ALIGN(ATYPE)                                             \
  CCE_INTRINSIC[aicore] void copy_cbuf_to_gm(                                  \
      __gm__ ATYPE *dst_addr, __cbuf__ ATYPE *scr_addr, uint64_t config) {     \
    uint64_t config0 = config & 0xffffffffULL;                                 \
    uint64_t config1 = ((config & 0xffff00000000ULL) >> 32) |                  \
                       ((config & 0xffff000000000000ULL) >> 16);               \
    copy_cbuf_to_gm_align_no_padding(dst_addr, scr_addr, config0, config1);    \
  }                                                                            \
                                                                               \
  CCE_INTRINSIC[aicore] void copy_cbuf_to_gm(                                  \
      __gm__ ATYPE *dst_addr, __cbuf__ ATYPE *scr_addr, uint8_t sid,           \
      uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride,                  \
      uint16_t dstStride) {                                                    \
    uint32_t srcGap = (uint32_t)srcStride;                                     \
    uint32_t dstGap = (uint32_t)dstStride;                                     \
    uint32_t lBurst = (uint32_t)lenBurst;                                      \
    copy_cbuf_to_gm_align_no_padding(dst_addr, scr_addr, sid, nBurst, lBurst,  \
                                     srcGap, dstGap);                          \
  }

MOV_L1_TO_OUT_ALIGN(void);
#undef MOV_L1_TO_OUT_ALIGN

#define MOV_UB_TO_OUT_ALIGN(ATYPE)                                             \
  CCE_INTRINSIC[aicore] void copy_ubuf_to_gm(                                  \
      __gm__ ATYPE *dst_addr, __ubuf__ ATYPE *scr_addr, uint64_t config) {     \
    uint64_t config0 = config & 0xffffffffULL;                                 \
    uint64_t config1 = ((config & 0xffff00000000ULL) >> 32) |                  \
                       ((config & 0xffff000000000000ULL) >> 16);               \
    copy_ubuf_to_gm_align_no_padding(dst_addr, scr_addr, config0, config1);    \
  }                                                                            \
                                                                               \
  CCE_INTRINSIC[aicore] void copy_ubuf_to_gm(                                  \
      __gm__ ATYPE *dst_addr, __ubuf__ ATYPE *scr_addr, uint8_t sid,           \
      uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride,                  \
      uint16_t dstStride) {                                                    \
    uint32_t srcGap = (uint32_t)srcStride;                                     \
    uint32_t dstGap = (uint32_t)dstStride;                                     \
    uint32_t lBurst = (uint32_t)lenBurst;                                      \
    copy_ubuf_to_gm_align_no_padding(dst_addr, scr_addr, sid, nBurst, lBurst,  \
                                     srcGap, dstGap);                          \
  }

MOV_UB_TO_OUT_ALIGN(void);
#undef MOV_UB_TO_OUT_ALIGN

#endif // #if (defined __DAV_M310__) || (defined __DAV_L310__)

#if (defined __DAV_N350__)

CCE_INTRINSIC[aicore] half get_max_min_cnt_value() {
  int64_t value = get_max_min_cnt();
  uint64_t tmp = 0xffffffffULL;
  value &= tmp;
  return *((half *)(&value));
}

CCE_INTRINSIC[aicore] uint32_t get_max_min_cnt_index() {
  int64_t value = get_max_min_cnt();
  uint64_t tmp = 0xffffffffULL;
  value = (value >> 32) & tmp;
  return (uint32_t)value;
}

//----------------------------------------------------------------------------//
//  MOVEVA + write va reg
//----------------------------------------------------------------------------//
// For N350, only VA0-3 are used
typedef std::integral_constant<ub_addr8_t, ub_addr8_t::VAReg0> VA0_Type;
typedef std::integral_constant<ub_addr8_t, ub_addr8_t::VAReg1> VA1_Type;
typedef std::integral_constant<ub_addr8_t, ub_addr8_t::VAReg2> VA2_Type;
typedef std::integral_constant<ub_addr8_t, ub_addr8_t::VAReg3> VA3_Type;
#define VA0 VA0_Type()
#define VA1 VA1_Type()
#define VA2 VA2_Type()
#define VA3 VA3_Type()

#define INVALID_VALUE_VA "VA reg can only be VA0, VA1, VA2, VA3"

#define __CCE_SET_VA_REG_4VA(regid, vaid, xn, xm)                              \
  MOVEVA_4VA((regid), vaid, (xn), (xm))

#define SET_VA_REG(ArrayType, FunctionName)                                    \
  template <class T>                                                           \
  CCE_INTRINSIC[aicore] void FunctionName(T addr, ArrayType *array) {          \
    static_assert(std::is_class<T>::value, INVALID_VALUE_VA);                  \
    static_assert(std::is_same<T, VA0_Type>::value ||                          \
                      std::is_same<T, VA1_Type>::value ||                      \
                      std::is_same<T, VA2_Type>::value ||                      \
                      std::is_same<T, VA3_Type>::value,                        \
                  INVALID_VALUE_VA);                                           \
    if (std::is_same<T, VA0_Type>::value) {                                    \
      __CCE_SET_VA_REG_4VA(VAReg0, 0, array[0], array[1]);                     \
      __CCE_SET_VA_REG_4VA(VAReg0, 2, array[2], array[3]);                     \
      __CCE_SET_VA_REG_4VA(VAReg0, 4, array[4], array[5]);                     \
      __CCE_SET_VA_REG_4VA(VAReg0, 6, array[6], array[7]);                     \
    } else if (std::is_same<T, VA1_Type>::value) {                             \
      __CCE_SET_VA_REG_4VA(VAReg1, 0, array[0], array[1]);                     \
      __CCE_SET_VA_REG_4VA(VAReg1, 2, array[2], array[3]);                     \
      __CCE_SET_VA_REG_4VA(VAReg1, 4, array[4], array[5]);                     \
      __CCE_SET_VA_REG_4VA(VAReg1, 6, array[6], array[7]);                     \
    } else if (std::is_same<T, VA2_Type>::value) {                             \
      __CCE_SET_VA_REG_4VA(VAReg2, 0, array[0], array[1]);                     \
      __CCE_SET_VA_REG_4VA(VAReg2, 2, array[2], array[3]);                     \
      __CCE_SET_VA_REG_4VA(VAReg2, 4, array[4], array[5]);                     \
      __CCE_SET_VA_REG_4VA(VAReg2, 6, array[6], array[7]);                     \
    } else if (std::is_same<T, VA3_Type>::value) {                             \
      __CCE_SET_VA_REG_4VA(VAReg3, 0, array[0], array[1]);                     \
      __CCE_SET_VA_REG_4VA(VAReg3, 2, array[2], array[3]);                     \
      __CCE_SET_VA_REG_4VA(VAReg3, 4, array[4], array[5]);                     \
      __CCE_SET_VA_REG_4VA(VAReg3, 6, array[6], array[7]);                     \
    }                                                                          \
  }

SET_VA_REG(uint64_t, set_va_reg_sb)
SET_VA_REG(__ubuf__ uint64_t, set_va_reg)
#undef __CCE_SET_VA_REG_4VA
#undef SET_VA_REG

#define VNCHWCONV(type)                                                        \
  template <class T1, class T2>                                                \
  CCE_INTRINSIC[aicore] void scatter_vnchwconv_##type(T1 dst, T2 src,          \
                                                      uint64_t config) {       \
    static_assert(std::is_class<T1>::value, INVALID_VALUE_VA);                 \
    static_assert(std::is_class<T2>::value, INVALID_VALUE_VA);                 \
    static_assert(std::is_same<T1, VA0_Type>::value ||                         \
                      std::is_same<T1, VA1_Type>::value ||                     \
                      std::is_same<T1, VA2_Type>::value ||                     \
                      std::is_same<T1, VA3_Type>::value,                       \
                  INVALID_VALUE_VA);                                           \
    static_assert(std::is_same<T2, VA0_Type>::value ||                         \
                      std::is_same<T2, VA1_Type>::value ||                     \
                      std::is_same<T2, VA2_Type>::value ||                     \
                      std::is_same<T2, VA3_Type>::value,                       \
                  INVALID_VALUE_VA);                                           \
    if (std::is_same<T1, VA0_Type>::value) {                                   \
      if (std::is_same<T2, VA0_Type>::value) {                                 \
        __cce_scatter_vnchwconv_##type(VAReg0, VAReg0, config);                \
      } else if (std::is_same<T2, VA1_Type>::value) {                          \
        __cce_scatter_vnchwconv_##type(VAReg0, VAReg1, config);                \
      } else if (std::is_same<T2, VA2_Type>::value) {                          \
        __cce_scatter_vnchwconv_##type(VAReg0, VAReg2, config);                \
      } else if (std::is_same<T2, VA3_Type>::value) {                          \
        __cce_scatter_vnchwconv_##type(VAReg0, VAReg3, config);                \
      }                                                                        \
    } else if (std::is_same<T1, VA1_Type>::value) {                            \
      if (std::is_same<T2, VA0_Type>::value) {                                 \
        __cce_scatter_vnchwconv_##type(VAReg1, VAReg0, config);                \
      } else if (std::is_same<T2, VA1_Type>::value) {                          \
        __cce_scatter_vnchwconv_##type(VAReg1, VAReg1, config);                \
      } else if (std::is_same<T2, VA2_Type>::value) {                          \
        __cce_scatter_vnchwconv_##type(VAReg1, VAReg2, config);                \
      } else if (std::is_same<T2, VA3_Type>::value) {                          \
        __cce_scatter_vnchwconv_##type(VAReg1, VAReg3, config);                \
      }                                                                        \
    } else if (std::is_same<T1, VA2_Type>::value) {                            \
      if (std::is_same<T2, VA0_Type>::value) {                                 \
        __cce_scatter_vnchwconv_##type(VAReg2, VAReg0, config);                \
      } else if (std::is_same<T2, VA1_Type>::value) {                          \
        __cce_scatter_vnchwconv_##type(VAReg2, VAReg1, config);                \
      } else if (std::is_same<T2, VA2_Type>::value) {                          \
        __cce_scatter_vnchwconv_##type(VAReg2, VAReg2, config);                \
      } else if (std::is_same<T2, VA3_Type>::value) {                          \
        __cce_scatter_vnchwconv_##type(VAReg2, VAReg3, config);                \
      }                                                                        \
    } else if (std::is_same<T1, VA3_Type>::value) {                            \
      if (std::is_same<T2, VA0_Type>::value) {                                 \
        __cce_scatter_vnchwconv_##type(VAReg3, VAReg0, config);                \
      } else if (std::is_same<T2, VA1_Type>::value) {                          \
        __cce_scatter_vnchwconv_##type(VAReg3, VAReg1, config);                \
      } else if (std::is_same<T2, VA2_Type>::value) {                          \
        __cce_scatter_vnchwconv_##type(VAReg3, VAReg2, config);                \
      } else if (std::is_same<T2, VA3_Type>::value) {                          \
        __cce_scatter_vnchwconv_##type(VAReg3, VAReg3, config);                \
      }                                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
  template <class T1, class T2>                                                \
  CCE_INTRINSIC[aicore] void scatter_vnchwconv_##type(                         \
      T1 dst, T2 src, uint8_t repeat, uint16_t dstStride,                      \
      uint16_t srcStride) {                                                    \
    static_assert(std::is_class<T1>::value, INVALID_VALUE_VA);                 \
    static_assert(std::is_class<T2>::value, INVALID_VALUE_VA);                 \
    static_assert(std::is_same<T1, VA0_Type>::value ||                         \
                      std::is_same<T1, VA1_Type>::value ||                     \
                      std::is_same<T1, VA2_Type>::value ||                     \
                      std::is_same<T1, VA3_Type>::value,                       \
                  INVALID_VALUE_VA);                                           \
    static_assert(std::is_same<T2, VA0_Type>::value ||                         \
                      std::is_same<T2, VA1_Type>::value ||                     \
                      std::is_same<T2, VA2_Type>::value ||                     \
                      std::is_same<T2, VA3_Type>::value,                       \
                  INVALID_VALUE_VA);                                           \
    if (std::is_same<T1, VA0_Type>::value) {                                   \
      if (std::is_same<T2, VA0_Type>::value) {                                 \
        __cce_scatter_vnchwconv_##type(VAReg0, VAReg0, repeat, dstStride,      \
                                       srcStride);                             \
      } else if (std::is_same<T2, VA1_Type>::value) {                          \
        __cce_scatter_vnchwconv_##type(VAReg0, VAReg1, repeat, dstStride,      \
                                       srcStride);                             \
      } else if (std::is_same<T2, VA2_Type>::value) {                          \
        __cce_scatter_vnchwconv_##type(VAReg0, VAReg2, repeat, dstStride,      \
                                       srcStride);                             \
      } else if (std::is_same<T2, VA3_Type>::value) {                          \
        __cce_scatter_vnchwconv_##type(VAReg0, VAReg3, repeat, dstStride,      \
                                       srcStride);                             \
      }                                                                        \
    } else if (std::is_same<T1, VA1_Type>::value) {                            \
      if (std::is_same<T2, VA0_Type>::value) {                                 \
        __cce_scatter_vnchwconv_##type(VAReg1, VAReg0, repeat, dstStride,      \
                                       srcStride);                             \
      } else if (std::is_same<T2, VA1_Type>::value) {                          \
        __cce_scatter_vnchwconv_##type(VAReg1, VAReg1, repeat, dstStride,      \
                                       srcStride);                             \
      } else if (std::is_same<T2, VA2_Type>::value) {                          \
        __cce_scatter_vnchwconv_##type(VAReg1, VAReg2, repeat, dstStride,      \
                                       srcStride);                             \
      } else if (std::is_same<T2, VA3_Type>::value) {                          \
        __cce_scatter_vnchwconv_##type(VAReg1, VAReg3, repeat, dstStride,      \
                                       srcStride);                             \
      }                                                                        \
    } else if (std::is_same<T1, VA2_Type>::value) {                            \
      if (std::is_same<T2, VA0_Type>::value) {                                 \
        __cce_scatter_vnchwconv_##type(VAReg2, VAReg0, repeat, dstStride,      \
                                       srcStride);                             \
      } else if (std::is_same<T2, VA1_Type>::value) {                          \
        __cce_scatter_vnchwconv_##type(VAReg2, VAReg1, repeat, dstStride,      \
                                       srcStride);                             \
      } else if (std::is_same<T2, VA2_Type>::value) {                          \
        __cce_scatter_vnchwconv_##type(VAReg2, VAReg2, repeat, dstStride,      \
                                       srcStride);                             \
      } else if (std::is_same<T2, VA3_Type>::value) {                          \
        __cce_scatter_vnchwconv_##type(VAReg2, VAReg3, repeat, dstStride,      \
                                       srcStride);                             \
      }                                                                        \
    } else if (std::is_same<T1, VA3_Type>::value) {                            \
      if (std::is_same<T2, VA0_Type>::value) {                                 \
        __cce_scatter_vnchwconv_##type(VAReg3, VAReg0, repeat, dstStride,      \
                                       srcStride);                             \
      } else if (std::is_same<T2, VA1_Type>::value) {                          \
        __cce_scatter_vnchwconv_##type(VAReg3, VAReg1, repeat, dstStride,      \
                                       srcStride);                             \
      } else if (std::is_same<T2, VA2_Type>::value) {                          \
        __cce_scatter_vnchwconv_##type(VAReg3, VAReg2, repeat, dstStride,      \
                                       srcStride);                             \
      } else if (std::is_same<T2, VA3_Type>::value) {                          \
        __cce_scatter_vnchwconv_##type(VAReg3, VAReg3, repeat, dstStride,      \
                                       srcStride);                             \
      }                                                                        \
    }                                                                          \
  }

VNCHWCONV(b8)
VNCHWCONV(b16)
#undef VNCHWCONV

template <class T>
CCE_INTRINSIC[aicore] void ldva(T dst, uint64_t src) {
  static_assert(std::is_class<T>::value, INVALID_VALUE_VA);
  static_assert(
      std::is_same<T, VA0_Type>::value || std::is_same<T, VA1_Type>::value ||
          std::is_same<T, VA2_Type>::value || std::is_same<T, VA3_Type>::value,
      INVALID_VALUE_VA);
  if (std::is_same<T, VA0_Type>::value) {
    __cce_ldva(VAReg0, src);
  } else if (std::is_same<T, VA1_Type>::value) {
    __cce_ldva(VAReg1, src);
  } else if (std::is_same<T, VA2_Type>::value) {
    __cce_ldva(VAReg2, src);
  } else if (std::is_same<T, VA3_Type>::value) {
    __cce_ldva(VAReg3, src);
  }
}

#define CONV_UB_TO_UB(FMTYPE, WTTYPE, TYPE)                                    \
  CCE_INTRINSIC[aicore] void conv_ub_to_ub(                                    \
      __ubuf__ void *dst_addr, __ubuf__ FMTYPE *fm_addr,                       \
      __ubuf__ WTTYPE *wt_addr, __ubuf__ int32_t *bias_addr,                   \
      uint64_t config) {                                                       \
    uint64_t add = (((uint64_t)bias_addr & 0xffffffffULL) << 32) |             \
                   ((uint64_t)fm_addr & 0xffffffffULL);                        \
    conv_ub_to_ub_##TYPE(dst_addr, add, wt_addr, config);                      \
  }                                                                            \
                                                                               \
  CCE_INTRINSIC[aicore] void conv_ub_to_ub(                                    \
      __ubuf__ void *dst_addr, __ubuf__ FMTYPE *fm_addr,                       \
      __ubuf__ WTTYPE *wt_addr, __ubuf__ int32_t *bias_addr,                   \
      bool bias_init_ctrl, bool bias_broadcast_en, QuantMode_t quant_type,     \
      ReluMode_t relu_type, uint16_t fm_w, uint16_t fm_h, uint16_t fm_co,      \
      bool clip_relu_en, bool depthwise_mode_en) {                             \
    uint64_t add;                                                              \
    if (bias_init_ctrl == 1)                                                   \
      add = (uint64_t)fm_addr & 0xffffffffULL;                                 \
    else                                                                       \
      add = (((uint64_t)bias_addr & 0xffffffffULL) << 32) |                    \
            ((uint64_t)fm_addr & 0xffffffffULL);                               \
    conv_ub_to_ub_##TYPE(dst_addr, add, wt_addr, bias_init_ctrl,               \
                         bias_broadcast_en, quant_type, relu_type, fm_w, fm_h, \
                         fm_co, clip_relu_en, depthwise_mode_en);              \
  }

CONV_UB_TO_UB(int8_t, int8_t, s8s8);
CONV_UB_TO_UB(int16_t, int8_t, s16s8);
CONV_UB_TO_UB(int8_t, void, s8s4);
#undef CONV_UB_TO_UB

//----------------------------------------------------------------------------//
//  MATMUL
//----------------------------------------------------------------------------//
#define MATMUL_UB_TO_UB(LTYPE, RTYPE, TYPESTR)                                 \
  CCE_INTRINSIC[aicore] void matmul_ub_to_ub(                                  \
      __ubuf__ void *dst_addr, __ubuf__ LTYPE *left_addr,                      \
      __ubuf__ RTYPE *right_addr, __ubuf__ int32_t *bias_addr,                 \
      uint64_t config) {                                                       \
    uint64_t addr = (((uint64_t)bias_addr & 0xffffffffULL) << 32) |            \
                    ((uint64_t)left_addr & 0xffffffffULL);                     \
    matmul_ub_to_ub_##TYPESTR(dst_addr, addr, right_addr, config);             \
  }                                                                            \
                                                                               \
  CCE_INTRINSIC[aicore] void matmul_ub_to_ub(                                  \
      __ubuf__ void *dst_addr, __ubuf__ LTYPE *left_addr,                      \
      __ubuf__ RTYPE *right_addr, __ubuf__ int32_t *bias_addr,                 \
      bool bias_init_ctrl, bool bias_broadcast_en, QuantMode_t quant_type,     \
      ReluMode_t relu_type, uint16_t M, uint16_t K, uint16_t N,                \
      bool clip_relu_en) {                                                     \
    uint64_t addr;                                                             \
    if (bias_init_ctrl == 1)                                                   \
      addr = (uint64_t)left_addr & 0xffffffffULL;                              \
    else                                                                       \
      addr = (((uint64_t)bias_addr & 0xffffffffULL) << 32) |                   \
             ((uint64_t)left_addr & 0xffffffffULL);                            \
    matmul_ub_to_ub_##TYPESTR(dst_addr, addr, right_addr, bias_init_ctrl,      \
                              bias_broadcast_en, quant_type, relu_type, M, K,  \
                              N, clip_relu_en);                                \
  }

MATMUL_UB_TO_UB(int8_t, int8_t, s8s8);
MATMUL_UB_TO_UB(int16_t, int8_t, s16s8);
MATMUL_UB_TO_UB(int8_t, void, s8s4);
#undef MATMUL_UB_TO_UB

#endif // #if (defined __DAV_N350__)

#if (defined __DAV_C310__)

struct nddma_desc {
  struct loop_desc {
    uint32_t size : 20;       // Xt[loop4, loop3] Xm[loop2, loop1, loop0]
    uint32_t dst_stride : 20; // LOOPx_STRIDE_NDDMA[20:0]
    uint64_t src_stride : 40; // LOOPx_STRIDE_NDDMA[59:20]
    uint8_t lp_size;          // PAD_CNT_NDDMA[loop4 ~ loop1] Xt[loop0]
    uint8_t rp_size;          // PAD_CNT_NDDMA[loop4 ~ loop1] Xt[loop0]
    [aicore] __attribute__((cce_builtin_api, always_inline))
    loop_desc(uint32_t size = 1, uint32_t dst_stride = 0,
              uint32_t src_stride = 0, uint8_t lp_size = 0, uint8_t rp_size = 0)
        : size(size), dst_stride(dst_stride), src_stride(src_stride),
          lp_size(lp_size), rp_size(rp_size){};
  };

  loop_desc loop4;
  loop_desc loop3;
  loop_desc loop2;
  loop_desc loop1;
  loop_desc loop0;
  enum pad_mode { NEAREST_PADDING_VALUE = 0, CONSTANT_PADDING_VALUE = 1 };
  [aicore] __attribute__((cce_builtin_api, always_inline)) nddma_desc(){};
  [aicore] __attribute__((cce_builtin_api, always_inline)) nddma_desc(loop_desc loop0)
      : loop0(loop0){};
  [aicore] __attribute__((cce_builtin_api, always_inline))
  nddma_desc(loop_desc loop1, loop_desc loop0)
      : loop1(loop1), loop0(loop0){};
  [aicore] __attribute__((cce_builtin_api, always_inline))
  nddma_desc(loop_desc loop2, loop_desc loop1, loop_desc loop0)
      : loop2(loop2), loop1(loop1), loop0(loop0){};
  [aicore] __attribute__((cce_builtin_api, always_inline))
  nddma_desc(loop_desc loop3, loop_desc loop2, loop_desc loop1, loop_desc loop0)
      : loop3(loop3), loop2(loop2), loop1(loop1), loop0(loop0){};
  [aicore] __attribute__((cce_builtin_api, always_inline))
  nddma_desc(loop_desc loop4, loop_desc loop3, loop_desc loop2, loop_desc loop1,
             loop_desc loop0)
      : loop4(loop4), loop3(loop3), loop2(loop2), loop1(loop1), loop0(loop0){};
};

typedef std::integral_constant<nddma_desc::pad_mode,
                               nddma_desc::pad_mode::NEAREST_PADDING_VALUE>
    NEAREST_PADDING_Type;
typedef std::integral_constant<nddma_desc::pad_mode,
                               nddma_desc::pad_mode::CONSTANT_PADDING_VALUE>
    CONSTANT_PADDING_Type;
#define NEAREST_PADDING NEAREST_PADDING_Type()
#define CONSTANT_PADDING CONSTANT_PADDING_Type()
#define INVALID_VALUE_PADDING_OUTTOUB                                          \
  "The 6th argument can only be NEAREST_PADDING, CONSTANT_PADDING"
#define INVALID_VALUE_PADDING_UBTOUB                                           \
  "The 5th argument can only be NEAREST_PADDING, CONSTANT_PADDING"

#define __CCE__NDDMA_OUT_TO_UB(FTYPE, SUFFIX)                                  \
  template <class T>                                                           \
  CCE_INTRINSIC[aicore] void nddma_out_to_ub(                                  \
      __ubuf__ FTYPE *ub_dst, __gm__ FTYPE *gm_src, uint8_t sid, nddma_desc n, \
      uint32_t pad_val, T mode) {                                              \
    static_assert(std::is_same<T, NEAREST_PADDING_Type>::value ||              \
                      std::is_same<T, CONSTANT_PADDING_Type>::value,           \
                  INVALID_VALUE_PADDING_OUTTOUB);                              \
    set_pad_cnt_nddma(((uint64_t)n.loop1.lp_size & 0xFFULL) |                  \
                      ((uint64_t)n.loop1.rp_size & 0xFFULL) << 8 |             \
                      ((uint64_t)n.loop2.lp_size & 0xFFULL) << 16 |            \
                      ((uint64_t)n.loop2.rp_size & 0xFFULL) << 24 |            \
                      ((uint64_t)n.loop3.lp_size & 0xFFULL) << 32 |            \
                      ((uint64_t)n.loop3.rp_size & 0xFFULL) << 40 |            \
                      ((uint64_t)n.loop4.lp_size & 0xFFULL) << 48 |            \
                      ((uint64_t)n.loop4.rp_size & 0xFFULL) << 56);            \
    set_pad_val_nddma(pad_val);                                                \
    set_loop0_stride_nddma(((uint64_t)n.loop0.dst_stride & 0xFFFFFULL) |       \
                           ((uint64_t)n.loop0.src_stride & 0xFFFFFFFFFFULL)    \
                               << 20);                                         \
    set_loop1_stride_nddma(((uint64_t)n.loop1.dst_stride & 0xFFFFFULL) |       \
                           ((uint64_t)n.loop1.src_stride & 0xFFFFFFFFFFULL)    \
                               << 20);                                         \
    set_loop2_stride_nddma(((uint64_t)n.loop2.dst_stride & 0xFFFFFULL) |       \
                           ((uint64_t)n.loop2.src_stride & 0xFFFFFFFFFFULL)    \
                               << 20);                                         \
    set_loop3_stride_nddma(((uint64_t)n.loop3.dst_stride & 0xFFFFFULL) |       \
                           ((uint64_t)n.loop3.src_stride & 0xFFFFFFFFFFULL)    \
                               << 20);                                         \
    set_loop4_stride_nddma(((uint64_t)n.loop4.dst_stride & 0xFFFFFULL) |       \
                           ((uint64_t)n.loop4.src_stride & 0xFFFFFFFFFFULL)    \
                               << 20);                                         \
    nddma_out_to_ub_##SUFFIX(ub_dst, gm_src,                                   \
                             ((uint64_t)sid & 0xFULL) |                        \
                                 ((uint64_t)n.loop0.size & 0xFFFFFULL) << 4 |  \
                                 ((uint64_t)n.loop1.size & 0xFFFFFULL) << 24 | \
                                 ((uint64_t)n.loop2.size & 0xFFFFFULL) << 44,  \
                             ((uint64_t)n.loop3.size & 0xFFFFFULL) |           \
                                 ((uint64_t)n.loop4.size & 0xFFFFFULL) << 20 | \
                                 ((uint64_t)n.loop0.lp_size & 0xFFULL) << 40 | \
                                 ((uint64_t)n.loop0.rp_size & 0xFFULL) << 48 | \
                                 ((uint64_t)mode.value & 0x1ULL) << 63);       \
  }
__CCE__NDDMA_OUT_TO_UB(int8_t, b8)
__CCE__NDDMA_OUT_TO_UB(uint8_t, b8)
__CCE__NDDMA_OUT_TO_UB(int16_t, b16)
__CCE__NDDMA_OUT_TO_UB(uint16_t, b16)
__CCE__NDDMA_OUT_TO_UB(half, b16)
__CCE__NDDMA_OUT_TO_UB(bfloat16_t, b16)
__CCE__NDDMA_OUT_TO_UB(int32_t, b32)
__CCE__NDDMA_OUT_TO_UB(uint32_t, b32)
__CCE__NDDMA_OUT_TO_UB(float, b32)
#undef __CCE__NDDMA_OUT_TO_UB // __CCE__NDDMA_OUT_TO_UB

#define __CCE__NDDMA_UB_TO_UB(FTYPE, SUFFIX)                                   \
  template <class T>                                                           \
  CCE_INTRINSIC[aicore] void nddma_ub_to_ub(                                   \
      __ubuf__ FTYPE *ub_dst, __ubuf__ FTYPE *gm_src, nddma_desc n,            \
      uint32_t pad_val, T mode) {                                              \
    static_assert(std::is_same<T, NEAREST_PADDING_Type>::value ||              \
                      std::is_same<T, CONSTANT_PADDING_Type>::value,           \
                  INVALID_VALUE_PADDING_UBTOUB);                               \
    set_pad_cnt_nddma(((uint64_t)n.loop1.lp_size & 0xFFULL) |                  \
                      ((uint64_t)n.loop1.rp_size & 0xFFULL) << 8 |             \
                      ((uint64_t)n.loop2.lp_size & 0xFFULL) << 16 |            \
                      ((uint64_t)n.loop2.rp_size & 0xFFULL) << 24 |            \
                      ((uint64_t)n.loop3.lp_size & 0xFFULL) << 32 |            \
                      ((uint64_t)n.loop3.rp_size & 0xFFULL) << 40 |            \
                      ((uint64_t)n.loop4.lp_size & 0xFFULL) << 48 |            \
                      ((uint64_t)n.loop4.rp_size & 0xFFULL) << 56);            \
    set_pad_val_nddma(pad_val);                                                \
    set_loop0_stride_nddma(((uint64_t)n.loop0.dst_stride & 0xFFFFFULL) |       \
                           ((uint64_t)n.loop0.src_stride & 0xFFFFFFFFFFULL)    \
                               << 20);                                         \
    set_loop1_stride_nddma(((uint64_t)n.loop1.dst_stride & 0xFFFFFULL) |       \
                           ((uint64_t)n.loop1.src_stride & 0xFFFFFFFFFFULL)    \
                               << 20);                                         \
    set_loop2_stride_nddma(((uint64_t)n.loop2.dst_stride & 0xFFFFFULL) |       \
                           ((uint64_t)n.loop2.src_stride & 0xFFFFFFFFFFULL)    \
                               << 20);                                         \
    set_loop3_stride_nddma(((uint64_t)n.loop3.dst_stride & 0xFFFFFULL) |       \
                           ((uint64_t)n.loop3.src_stride & 0xFFFFFFFFFFULL)    \
                               << 20);                                         \
    set_loop4_stride_nddma(((uint64_t)n.loop4.dst_stride & 0xFFFFFULL) |       \
                           ((uint64_t)n.loop4.src_stride & 0xFFFFFFFFFFULL)    \
                               << 20);                                         \
    nddma_ub_to_ub_##SUFFIX(ub_dst, gm_src,                                    \
                            ((uint64_t)n.loop0.size & 0xFFFFFULL) << 4 |       \
                                ((uint64_t)n.loop1.size & 0xFFFFFULL) << 24 |  \
                                ((uint64_t)n.loop2.size & 0xFFFFFULL) << 44,   \
                            ((uint64_t)n.loop3.size & 0xFFFFFULL) |            \
                                ((uint64_t)n.loop4.size & 0xFFFFFULL) << 20 |  \
                                ((uint64_t)n.loop0.lp_size & 0xFFULL) << 40 |  \
                                ((uint64_t)n.loop0.rp_size & 0xFFULL) << 48 |  \
                                ((uint64_t)mode.value & 0x1ULL) << 63);        \
  }
__CCE__NDDMA_UB_TO_UB(int8_t, b8)
__CCE__NDDMA_UB_TO_UB(uint8_t, b8)
__CCE__NDDMA_UB_TO_UB(int16_t, b16)
__CCE__NDDMA_UB_TO_UB(uint16_t, b16)
__CCE__NDDMA_UB_TO_UB(half, b16)
__CCE__NDDMA_UB_TO_UB(bfloat16_t, b16)
__CCE__NDDMA_UB_TO_UB(int32_t, b32)
__CCE__NDDMA_UB_TO_UB(uint32_t, b32)
__CCE__NDDMA_UB_TO_UB(float, b32)
#undef __CCE__NDDMA_UB_TO_UB // __CCE__NDDMA_UB_TO_UB
#endif                       // __DAV_C310__

#if (defined __DAV_M200__) || (defined __DAV_M200_VEC__) ||                    \
    (defined __DAV_C220_VEC__) || (defined __DAV_M300__) ||                    \
    (defined __DAV_M310__)
CCE_INTRINSIC[aicore] void
copy_data_align64(uint8_t *dst, __ubuf__ uint8_t *src, uint64_t size) {
  __builtin_cce_copy_from_ub_align64(dst, src, size);
}
#endif // if (defined __DAV_M200__) || (defined
       // __DAV_M200_VEC__) || (defined __DAV_C220_VEC__) || (defined
       // __DAV_M300__) || (defined __DAV_M310__)

#if (defined __DAV_M200__) || (defined __DAV_M200_VEC__) ||                    \
    (defined __DAV_C220_CUBE__) || (defined __DAV_C220_VEC__) ||               \
    (defined __DAV_M300__) || (defined __DAV_M310__)
CCE_INTRINSIC[aicore] void copy_data_align64(uint8_t *dst, __gm__ uint8_t *src,
                                             uint64_t size) {
  __builtin_cce_copy_from_gm_align64(dst, src, size);
}
#endif // if (defined __DAV_M200__) || (defined __DAV_M200_VEC__) || (defined
       // __DAV_C220_CUBE__) || (defined __DAV_C220_VEC__) || (defined
       // __DAV_M300__) || (defined __DAV_M310__)

#if (defined __DAV_M200__) || (defined __DAV_M200_VEC__) ||                    \
    (defined __DAV_C220_CUBE__) || (defined __DAV_C220_VEC__) ||               \
    (defined __DAV_M300__) || (defined __DAV_M310__)
CCE_INTRINSIC[aicore] void copy_data_align64(uint8_t *dst, uint8_t *src,
                                             uint64_t size) {
  __builtin_cce_copy_from_stack_align64(dst, src, size);
}
#endif // if (defined __DAV_M200__) || (defined __DAV_M200_VEC__) || (defined
       // __DAV_C220_CUBE__) || (defined __DAV_C220_VEC__) || (defined
       // __DAV_M300__) || (defined __DAV_M310__)

#if (defined __DAV_M200__) || (defined __DAV_M200_VEC__) ||                    \
    (defined __DAV_C220_VEC__) || (defined __DAV_M300__) ||                    \
    (defined __DAV_M310__)
CCE_INTRINSIC[aicore] void copy_data(uint8_t *dst, __ubuf__ uint8_t *src,
                                     uint64_t size) {
  __builtin_cce_copy_from_ub(dst, src, size);
}
#endif // if (defined __DAV_M200__) || (defined
       // __DAV_M200_VEC__) || (defined __DAV_C220_VEC__) || (defined
       // __DAV_M300__) || (defined __DAV_M310__)

#if (defined __DAV_M200__) || (defined __DAV_M200_VEC__) ||                    \
    (defined __DAV_C220_CUBE__) || (defined __DAV_C220_VEC__) ||               \
    (defined __DAV_M300__) || (defined __DAV_M310__)
CCE_INTRINSIC[aicore] void copy_data(uint8_t *dst, __gm__ uint8_t *src,
                                     uint64_t size) {
  __builtin_cce_copy_from_gm(dst, src, size);
}
#endif // if (defined __DAV_M200__) || (defined __DAV_M200_VEC__) || (defined
       // __DAV_C220_CUBE__) || (defined __DAV_C220_VEC__) || (defined
       // __DAV_M300__) || (defined __DAV_M310__)

#if (defined __DAV_M200__) || (defined __DAV_M200_VEC__) ||                    \
    (defined __DAV_C220_CUBE__) || (defined __DAV_C220_VEC__) ||               \
    (defined __DAV_M300__) || (defined __DAV_M310__)
CCE_INTRINSIC[aicore] void copy_data(uint8_t *dst, uint8_t *src,
                                     uint64_t size) {
  __builtin_cce_copy_from_stack(dst, src, size);
}
#endif // if (defined __DAV_M200__) || (defined __DAV_M200_VEC__) || (defined
       // __DAV_C220_CUBE__) || (defined __DAV_C220_VEC__) || (defined
       // __DAV_M300__) || (defined __DAV_M310__)

#if (defined __DAV_M200__) || (defined __DAV_M200_VEC__)
CCE_INTRINSIC[aicore] void vscatter(__ubuf__ uint32_t *dst,
                                    __ubuf__ uint16_t *src, uint32_t offset,
                                    bool strideSizeMode) {
  vscatter(dst, src, offset, strideSizeMode, false, 0, 0);
}

CCE_INTRINSIC[aicore] void vscatter(__ubuf__ uint32_t *dst,
                                    __ubuf__ uint32_t *src, uint32_t offset,
                                    bool strideSizeMode) {
  vscatter(dst, src, offset, strideSizeMode, false, 0, 0);
}
#endif // if (defined __DAV_M200__) || (defined __DAV_M200_VEC__)

#ifdef __CCE_AICORE__
/*------------------------------CCE_SET_FLAG----------------------------------*/
CCE_INTRINSIC[aicore] void __cce_set_flag(pipe_t p, pipe_t tp, event_t n) {
  set_flag(PIPE_M, PIPE_V, n);
  (void)p;
  (void)tp;
}

/*------------------------------CCE_WAIT_FLAG------------------------------------*/
CCE_INTRINSIC[aicore] void __cce_wait_flag(pipe_t p, pipe_t tp, event_t n) {
  wait_flag(PIPE_M, PIPE_V, n);
  (void)p;
  (void)tp;
}

/*-------------------------------CCE_PIPE_BARRIER------------------------------------*/
CCE_INTRINSIC[aicore] void __cce_pipe_barrier(pipe_t p) {
  pipe_barrier(PIPE_V);
  (void)p;
}

/*-------------------------------ABS------------------------------------*/
// abs,fabs
#define ABS(TYPE, SUFFIX)                                                      \
  CCE_INTRINSIC[aicore] TYPE abs(TYPE in) { return __builtin_cce_##SUFFIX(in); }

ABS(long long, llabs);
ABS(long, labs);
ABS(int, abs);
ABS(short, abs);
ABS(char, abs);
ABS(float, fabsf);
#if (defined __DAV_C310__)
ABS(half, fabsf16);
ABS(bfloat16_t, fabsbf16);
ABS(hifloat8_t, fabshif8);
#endif
#undef ABS

/*-------------------------------SQRT------------------------------------*/
CCE_INTRINSIC[aicore] int64_t sqrt(long long in) { return SQRT_s64(in); }

CCE_INTRINSIC[aicore] int64_t sqrt(long in) { return SQRT_s64(in); }

CCE_INTRINSIC[aicore] int64_t sqrt(int in) { return SQRT_s64(in); }

CCE_INTRINSIC[aicore] int64_t sqrt(short in) { return SQRT_s64(in); }

CCE_INTRINSIC[aicore] int64_t sqrt(char in) { return SQRT_s64(in); }

CCE_INTRINSIC[aicore] float sqrt(float in) { return __builtin_cce_sqrtf(in); }

CCE_INTRINSIC[aicore] half sqrt(half in) { return __builtin_cce_sqrtf16(in); }

/*-------------------------------DUMMY_PIPE------------------------------------*/
CCE_INTRINSIC PIPE_ID(PIPE_S)[aicore] void __dummy_pipe_s() {}

CCE_INTRINSIC PIPE_ID(PIPE_M)[aicore] void __dummy_pipe_m() {}

CCE_INTRINSIC PIPE_ID(PIPE_V)[aicore] void __dummy_pipe_v() {}

CCE_INTRINSIC PIPE_ID(PIPE_MTE1)[aicore] void __dummy_pipe_mte1() {}

CCE_INTRINSIC PIPE_ID(PIPE_MTE2)[aicore] void __dummy_pipe_mte2() {}

CCE_INTRINSIC PIPE_ID(PIPE_MTE3)[aicore] void __dummy_pipe_mte3() {}

/*-------------------------------MAX------------------------------------*/
CCE_INTRINSIC[aicore] long long max(long long in1, long long in2) {
  return __builtin_cce_llsmax(in1, in2);
}

CCE_INTRINSIC[aicore] long max(long in1, long in2) {
  return __builtin_cce_lsmax(in1, in2);
}

CCE_INTRINSIC[aicore] int max(int in1, int in2) {
  return __builtin_cce_smax(in1, in2);
}

CCE_INTRINSIC[aicore] short max(short in1, short in2) {
  return __builtin_cce_smax(in1, in2);
}

CCE_INTRINSIC[aicore] char max(char in1, char in2) {
  return __builtin_cce_smax(in1, in2);
}

CCE_INTRINSIC[aicore] unsigned long long max(unsigned long long in1,
                                             unsigned long long in2) {
  return __builtin_cce_llumax(in1, in2);
}

CCE_INTRINSIC[aicore] unsigned long max(unsigned long in1, unsigned long in2) {
  return __builtin_cce_lumax(in1, in2);
}

CCE_INTRINSIC[aicore] unsigned int max(unsigned int in1, unsigned int in2) {
  return __builtin_cce_umax(in1, in2);
}

CCE_INTRINSIC[aicore] unsigned short max(unsigned short in1,
                                         unsigned short in2) {
  return __builtin_cce_umax(in1, in2);
}

CCE_INTRINSIC[aicore] unsigned char max(unsigned char in1, unsigned char in2) {
  return __builtin_cce_umax(in1, in2);
}

CCE_INTRINSIC[aicore] float max(float in1, float in2) {
  return __builtin_cce_fmaxf(in1, in2);
}

CCE_INTRINSIC[aicore] half max(half in1, half in2) {
  return __builtin_cce_fmaxf16(in1, in2);
}

/*-------------------------------MIN------------------------------------*/
CCE_INTRINSIC[aicore] long long min(long long in1, long long in2) {
  return __builtin_cce_llsmin(in1, in2);
}

CCE_INTRINSIC[aicore] long min(long in1, long in2) {
  return __builtin_cce_lsmin(in1, in2);
}

CCE_INTRINSIC[aicore] int min(int in1, int in2) {
  return __builtin_cce_smin(in1, in2);
}

CCE_INTRINSIC[aicore] short min(short in1, short in2) {
  return __builtin_cce_smin(in1, in2);
}

CCE_INTRINSIC[aicore] char min(char in1, char in2) {
  return __builtin_cce_smin(in1, in2);
}

CCE_INTRINSIC[aicore] unsigned long long min(unsigned long long in1,
                                             unsigned long long in2) {
  return __builtin_cce_llumin(in1, in2);
}

CCE_INTRINSIC[aicore] unsigned long min(unsigned long in1, unsigned long in2) {
  return __builtin_cce_lumin(in1, in2);
}

CCE_INTRINSIC[aicore] unsigned int min(unsigned int in1, unsigned int in2) {
  return __builtin_cce_umin(in1, in2);
}

CCE_INTRINSIC[aicore] unsigned short min(unsigned short in1,
                                         unsigned short in2) {
  return __builtin_cce_umin(in1, in2);
}

CCE_INTRINSIC[aicore] unsigned char min(unsigned char in1, unsigned char in2) {
  return __builtin_cce_umin(in1, in2);
}

CCE_INTRINSIC[aicore] float min(float in1, float in2) {
  return __builtin_cce_fminf(in1, in2);
}

CCE_INTRINSIC[aicore] half min(half in1, half in2) {
  return __builtin_cce_fminf16(in1, in2);
}

/*-------------------------------COPY_UBUF_TO_SBUF------------------------------------*/
// MOV_UB_TO_SB [Xd],[Xn], Xt, Xm : for checking addrspace for arg1
CCE_INTRINSIC[aicore] void copy_ubuf_to_sbuf(void *dst, __ubuf__ void *src,
                                             uint64_t size, int64_t inc) {
  MOV_UB_TO_SB(dst, src, size, inc);
}

/*-------------------------------COPY_SBUF_TO_UBUF------------------------------------*/
// MOV_SB_TO_UB [Xd],[Xn], Xt, Xm : for checking addrspace for arg1
CCE_INTRINSIC[aicore] void copy_sbuf_to_ubuf(__ubuf__ void *dst, void *src,
                                             uint64_t size, int64_t inc) {
  MOV_SB_TO_UB(dst, src, size, inc);
}

/*-------------------------------VBS16------------------------------------*/
CCE_INTRINSIC[aicore] void vbitsort(__ubuf__ half *dst, __ubuf__ half *src,
                                    uint8_t repeat) {
  uint64_t config = static_cast<uint64_t>(repeat) << 56;
  VBS16_f16(dst, src, config);
}

CCE_INTRINSIC[aicore] void vbitsort(__ubuf__ float *dst, __ubuf__ float *src,
                                    uint8_t repeat) {
  uint64_t config = static_cast<uint64_t>(repeat) << 56;
  VBS16_f32(dst, src, config);
}

/*-------------------------------VBS32------------------------------------*/
CCE_INTRINSIC[aicore] void vbitsort(__ubuf__ half *dst, __ubuf__ half *src0,
                                    __ubuf__ unsigned int *src1,
                                    uint8_t repeat) {
  uint64_t config = static_cast<uint64_t>(repeat) << 56;
  VBS32_f16(dst, src0, src1, config);
}

CCE_INTRINSIC[aicore] void vbitsort(__ubuf__ float *dst, __ubuf__ float *src0,
                                    __ubuf__ unsigned int *src1,
                                    uint8_t repeat) {
  uint64_t config = static_cast<uint64_t>(repeat) << 56;
  VBS32_f32(dst, src0, src1, config);
}

// For a series of OBI instruction, add wrappers due to integer reload problem.
/*-------------------------------VAADD------------------------------------*/
CCE_INTRINSIC[aicore] void vaadd(__ubuf__ half *dst, __ubuf__ half *src0,
                                 __ubuf__ half *src1, uint8_t repeat) {
  uint64_t config = static_cast<uint64_t>(repeat) << 56;
  VAADD_f16(dst, src0, src1, config);
}
CCE_INTRINSIC[aicore] void vaadd(__ubuf__ float *dst, __ubuf__ float *src0,
                                 __ubuf__ float *src1, uint8_t repeat) {
  uint64_t config = static_cast<uint64_t>(repeat) << 56;
  VAADD_f32(dst, src0, src1, config);
}

/*-------------------------------VMERGECH------------------------------------*/
CCE_INTRINSIC[aicore] void vmergech(__ubuf__ uint8_t *dst,
                                    __ubuf__ uint8_t *src, uint8_t repeat) {
  uint64_t config = static_cast<uint64_t>(repeat) << 56;
  VMERGECH_b8(dst, src, config);
}

CCE_INTRINSIC[aicore] void vmergech(__ubuf__ half *dst, __ubuf__ half *src,
                                    uint8_t repeat) {
  uint64_t config = static_cast<uint64_t>(repeat) << 56;
  VMERGECH_f16(dst, src, config);
}

/*-------------------------------VRPAC------------------------------------*/
CCE_INTRINSIC[aicore] void vrpac(__ubuf__ half *dst, __ubuf__ half *src,
                                 uint8_t repeat) {
  uint64_t config = static_cast<uint64_t>(repeat) << 56;
  VRPAC_f16(dst, src, config);
}

CCE_INTRINSIC[aicore] void vrpac(__ubuf__ float *dst, __ubuf__ float *src,
                                 uint8_t repeat) {
  uint64_t config = static_cast<uint64_t>(repeat) << 56;
  VRPAC_f32(dst, src, config);
}

/*-------------------------------VIOU------------------------------------*/
CCE_INTRINSIC[aicore] void viou(__ubuf__ half *dst, __ubuf__ half *src0,
                                __ubuf__ half *src1, uint8_t repeat) {
  uint64_t config = static_cast<uint64_t>(repeat) << 56;
  VIOU_f16(dst, src0, src1, config);
}

CCE_INTRINSIC[aicore] void viou(__ubuf__ float *dst, __ubuf__ float *src0,
                                __ubuf__ float *src1, uint8_t repeat) {
  uint64_t config = static_cast<uint64_t>(repeat) << 56;
  VIOU_f32(dst, src0, src1, config);
}

/*-------------------------------PRELOAD------------------------------------*/
#if __CCE_AICORE__ >= 220
CCE_INTRINSIC[aicore] inline void icache_preload(int64_t n) {
  preload((const void *)get_pc(), n);
}
#elif __CCE_AICORE__ >= 200
CCE_INTRINSIC[aicore] inline void icache_preload() {
  preload((const void *)get_pc());
}
#endif

/*-------------------------------VMS4------------------------------------*/
// Expose2User : Line with this means that this intrinsic is
//               Officially expose to our user/programmer
//               - we had a tool to extract the intrinsis
//                 and publish to user:
//        $CLANG_ROOT/tools/cce-tools/publish_builtin_tool.py
// Expose2User : ISA5.6.2 : VMS4.f16 [Xd],[Xn],Xt : src4 is 4 byte addr of UB
#if (__CCE_AICORE__ == 100) || (__CCE_AICORE__ == 200) ||                      \
    (__CCE_AICORE__ == 210)
static __attribute__((cce_builtin_api, always_inline))[aicore] uint64_t
get_vmrgsort_real_addr(__ubuf__ uint64_t *src, unsigned shift) {
  uint64_t realAddr = 0;
  realAddr = ((uint64_t)src[0] >> shift) & 0xFFFFULL;
  realAddr |= (((uint64_t)src[1] >> shift) & 0xFFFFULL) << 16;
  realAddr |= (((uint64_t)src[2] >> shift) & 0xFFFFULL) << 32;
  realAddr |= (((uint64_t)src[3] >> shift) & 0xFFFFULL) << 48;
  return realAddr;
}

static __attribute__((cce_builtin_api, always_inline))[aicore] uint64_t
get_vmrgsort_real_addr(__ubuf__ half *src[4], unsigned shift) {
  uint64_t realAddr = 0;
  realAddr = ((uint64_t)src[0] >> shift) & 0xFFFFULL;
  realAddr |= (((uint64_t)src[1] >> shift) & 0xFFFFULL) << 16;
  realAddr |= (((uint64_t)src[2] >> shift) & 0xFFFFULL) << 32;
  realAddr |= (((uint64_t)src[3] >> shift) & 0xFFFFULL) << 48;
  return realAddr;
}

static __attribute__((cce_builtin_api, always_inline))[aicore] uint64_t
get_vmrgsort_real_addr(__ubuf__ float *src[4], unsigned shift) {
  uint64_t realAddr = 0;
  realAddr = ((uint64_t)src[0] >> shift) & 0xFFFFULL;
  realAddr |= (((uint64_t)src[1] >> shift) & 0xFFFFULL) << 16;
  realAddr |= (((uint64_t)src[2] >> shift) & 0xFFFFULL) << 32;
  realAddr |= (((uint64_t)src[3] >> shift) & 0xFFFFULL) << 48;
  return realAddr;
}

CCE_INTRINSIC PIPE_ID(PIPE_V)[aicore] void vmrgsort4(__ubuf__ half *dst,
                                                     __ubuf__ uint64_t *src,
                                                     uint64_t config) {
  uint64_t realAddr = get_vmrgsort_real_addr(src, 4);
  vmrgsort4(dst, (__ubuf__ half *)realAddr, config);
}

CCE_INTRINSIC PIPE_ID(PIPE_V)[aicore] void vmrgsort4(
    __ubuf__ half *dst, __ubuf__ uint64_t *src, uint8_t repeat, uint16_t list0,
    uint16_t list1, uint16_t list2, uint16_t list3, bool enable_exh_sus,
    uint8_t mask) __attribute__((cce_range_check("VMRGSORT_f16_cfg"))) {
  uint64_t realAddr = get_vmrgsort_real_addr(src, 4);
  vmrgsort4(dst, (__ubuf__ half *)realAddr, repeat, list0, list1, list2, list3,
            enable_exh_sus, mask);
}

CCE_INTRINSIC PIPE_ID(PIPE_V)[aicore] void vmrgsort4(__ubuf__ half *dst,
                                                     __ubuf__ half *src[4],
                                                     uint64_t config) {
  uint64_t realAddr = get_vmrgsort_real_addr(src, 4);
  vmrgsort4(dst, (__ubuf__ half *)realAddr, config);
}

CCE_INTRINSIC PIPE_ID(PIPE_V)[aicore] void vmrgsort4(
    __ubuf__ half *dst, __ubuf__ half *src[4], uint8_t repeat, uint16_t list0,
    uint16_t list1, uint16_t list2, uint16_t list3, bool enable_exh_sus,
    uint8_t mask) __attribute__((cce_range_check("VMRGSORT_f16_cfg"))) {
  uint64_t realAddr = get_vmrgsort_real_addr(src, 4);
  vmrgsort4(dst, (__ubuf__ half *)realAddr, repeat, list0, list1, list2, list3,
            enable_exh_sus, mask);
}

CCE_INTRINSIC PIPE_ID(PIPE_V)[aicore] void vmrgsort4(__ubuf__ float *dst,
                                                     __ubuf__ uint64_t *src,
                                                     uint64_t config) {
  uint64_t realAddr = get_vmrgsort_real_addr(src, 5);
  vmrgsort4(dst, (__ubuf__ float *)realAddr, config);
}

CCE_INTRINSIC PIPE_ID(PIPE_V)[aicore] void vmrgsort4(
    __ubuf__ float *dst, __ubuf__ uint64_t *src, uint8_t repeat, uint16_t list0,
    uint16_t list1, uint16_t list2, uint16_t list3, bool enable_exh_sus,
    uint8_t mask) __attribute__((cce_range_check("VMRGSORT_f32_cfg"))) {
  uint64_t realAddr = get_vmrgsort_real_addr(src, 5);
  vmrgsort4(dst, (__ubuf__ float *)realAddr, repeat, list0, list1, list2, list3,
            enable_exh_sus, mask);
}

CCE_INTRINSIC PIPE_ID(PIPE_V)[aicore] void vmrgsort4(__ubuf__ float *dst,
                                                     __ubuf__ float *src[4],
                                                     uint64_t config) {
  uint64_t realAddr = get_vmrgsort_real_addr(src, 5);
  vmrgsort4(dst, (__ubuf__ float *)realAddr, config);
}

CCE_INTRINSIC PIPE_ID(PIPE_V)[aicore] void vmrgsort4(
    __ubuf__ float *dst, __ubuf__ float *src[4], uint8_t repeat, uint16_t list0,
    uint16_t list1, uint16_t list2, uint16_t list3, bool enable_exh_sus,
    uint8_t mask) __attribute__((cce_range_check("VMRGSORT_f32_cfg"))) {
  uint64_t realAddr = get_vmrgsort_real_addr(src, 5);
  vmrgsort4(dst, (__ubuf__ float *)realAddr, repeat, list0, list1, list2, list3,
            enable_exh_sus, mask);
}
// For VMS4, V300 reuse the interface of V220.
#else
static __attribute__((cce_builtin_api, always_inline))[aicore] uint64_t
get_vmrgsort_real_addr(__ubuf__ uint64_t *src, unsigned shift,
                       uint64_t config = 0x0F00ULL) {
  uint64_t realAddr = 0;
  realAddr =
      (config & 0x0100ULL) ? (((uint64_t)src[0] >> shift) & 0xFFFFULL) : 0;
  realAddr |=
      ((config & 0x0200ULL) ? (((uint64_t)src[1] >> shift) & 0xFFFFULL) : 0)
      << 16;
  realAddr |=
      ((config & 0x0400ULL) ? (((uint64_t)src[2] >> shift) & 0xFFFFULL) : 0)
      << 32;
  realAddr |=
      ((config & 0x0800ULL) ? (((uint64_t)src[3] >> shift) & 0xFFFFULL) : 0)
      << 48;
  return realAddr;
}

static __attribute__((cce_builtin_api, always_inline))[aicore] uint64_t
get_vmrgsort_real_addr(__ubuf__ half *src[4], unsigned shift,
                       uint64_t config = 0x0F00ULL) {
  uint64_t realAddr = 0;
  realAddr =
      (config & 0x0100ULL) ? (((uint64_t)src[0] >> shift) & 0xFFFFULL) : 0;
  realAddr |=
      ((config & 0x0200ULL) ? (((uint64_t)src[1] >> shift) & 0xFFFFULL) : 0)
      << 16;
  realAddr |=
      ((config & 0x0400ULL) ? (((uint64_t)src[2] >> shift) & 0xFFFFULL) : 0)
      << 32;
  realAddr |=
      ((config & 0x0800ULL) ? (((uint64_t)src[3] >> shift) & 0xFFFFULL) : 0)
      << 48;
  return realAddr;
}

static __attribute__((cce_builtin_api, always_inline))[aicore] uint64_t
get_vmrgsort_real_addr(__ubuf__ float *src[4], unsigned shift,
                       uint64_t config = 0x0F00ULL) {
  uint64_t realAddr = 0;
  realAddr =
      (config & 0x0100ULL) ? (((uint64_t)src[0] >> shift) & 0xFFFFULL) : 0;
  realAddr |=
      ((config & 0x0200ULL) ? (((uint64_t)src[1] >> shift) & 0xFFFFULL) : 0)
      << 16;
  realAddr |=
      ((config & 0x0400ULL) ? (((uint64_t)src[2] >> shift) & 0xFFFFULL) : 0)
      << 32;
  realAddr |=
      ((config & 0x0800ULL) ? (((uint64_t)src[3] >> shift) & 0xFFFFULL) : 0)
      << 48;
  return realAddr;
}

CCE_INTRINSIC PIPE_ID(PIPE_V)[aicore] void vmrgsort4(__ubuf__ half *dst,
                                                     __ubuf__ uint64_t *src0,
                                                     uint64_t src1,
                                                     uint64_t config) {
  uint64_t realAddr = get_vmrgsort_real_addr(src0, 3, config);
  vmrgsort4(dst, (__ubuf__ half *)realAddr, src1, config);
}

CCE_INTRINSIC PIPE_ID(PIPE_V)[aicore] void vmrgsort4(
    __ubuf__ half *dst, __ubuf__ uint64_t *src, uint8_t repeat, uint16_t list0,
    uint16_t list1, uint16_t list2, uint16_t list3, bool enable_exh_sus,
    uint8_t mask) __attribute__((cce_range_check("VMRGSORT_f16_V220_cfg"))) {
  uint64_t realAddr = get_vmrgsort_real_addr(src, 3);
  vmrgsort4(dst, (__ubuf__ half *)realAddr, repeat, list0, list1, list2, list3,
            enable_exh_sus, mask);
}

CCE_INTRINSIC PIPE_ID(PIPE_V)[aicore] void vmrgsort4(__ubuf__ half *dst,
                                                     __ubuf__ half *src0[4],
                                                     uint64_t src1,
                                                     uint64_t config) {
  uint64_t realAddr = get_vmrgsort_real_addr(src0, 3, config);
  vmrgsort4(dst, (__ubuf__ half *)realAddr, src1, config);
}

CCE_INTRINSIC PIPE_ID(PIPE_V)[aicore] void vmrgsort4(
    __ubuf__ half *dst, __ubuf__ half *src[4], uint8_t repeat, uint16_t list0,
    uint16_t list1, uint16_t list2, uint16_t list3, bool enable_exh_sus,
    uint8_t mask) __attribute__((cce_range_check("VMRGSORT_f16_V220_cfg"))) {
  uint64_t realAddr = get_vmrgsort_real_addr(src, 3);
  vmrgsort4(dst, (__ubuf__ half *)realAddr, repeat, list0, list1, list2, list3,
            enable_exh_sus, mask);
}

CCE_INTRINSIC PIPE_ID(PIPE_V)[aicore] void vmrgsort4(__ubuf__ float *dst,
                                                     __ubuf__ uint64_t *src0,
                                                     uint64_t src1,
                                                     uint64_t config) {
  uint64_t realAddr = get_vmrgsort_real_addr(src0, 3, config);
  vmrgsort4(dst, (__ubuf__ float *)realAddr, src1, config);
}

CCE_INTRINSIC PIPE_ID(PIPE_V)[aicore] void vmrgsort4(
    __ubuf__ float *dst, __ubuf__ uint64_t *src, uint8_t repeat, uint16_t list0,
    uint16_t list1, uint16_t list2, uint16_t list3, bool enable_exh_sus,
    uint8_t mask) __attribute__((cce_range_check("VMRGSORT_f32_V220_cfg"))) {
  uint64_t realAddr = get_vmrgsort_real_addr(src, 3);
  vmrgsort4(dst, (__ubuf__ float *)realAddr, repeat, list0, list1, list2, list3,
            enable_exh_sus, mask);
}

CCE_INTRINSIC PIPE_ID(PIPE_V)[aicore] void vmrgsort4(__ubuf__ float *dst,
                                                     __ubuf__ float *src0[4],
                                                     uint64_t src1,
                                                     uint64_t config) {
  uint64_t realAddr = get_vmrgsort_real_addr(src0, 3, config);
  vmrgsort4(dst, (__ubuf__ float *)realAddr, src1, config);
}

CCE_INTRINSIC PIPE_ID(PIPE_V)[aicore] void vmrgsort4(
    __ubuf__ float *dst, __ubuf__ float *src[4], uint8_t repeat, uint16_t list0,
    uint16_t list1, uint16_t list2, uint16_t list3, bool enable_exh_sus,
    uint8_t mask) __attribute__((cce_range_check("VMRGSORT_f32_V220_cfg"))) {
  uint64_t realAddr = get_vmrgsort_real_addr(src, 3);
  vmrgsort4(dst, (__ubuf__ float *)realAddr, repeat, list0, list1, list2, list3,
            enable_exh_sus, mask);
}
#endif

/*-------------------------------L0_SET_VALUE------------------------------------*/
#define CCE_INTRINSIC_L0_SET_VALUE(ValueType, TypeStr)                         \
  CCE_INTRINSIC[aicore] void set_l0_set_value(ValueType value) {               \
    set_l0_set_value_##TypeStr(value);                                         \
  }

CCE_INTRINSIC_L0_SET_VALUE(uint32_t, ui);
CCE_INTRINSIC_L0_SET_VALUE(half, h);
#if (__CCE_AICORE__ == 220) || (__CCE_AICORE__ == 300) || (defined __DAV_C310__)
CCE_INTRINSIC_L0_SET_VALUE(bfloat16_t, bf16);
#endif
#undef CCE_INTRINSIC_L0_SET_VALUE

/*-------------------------------CREATE_MATRIX------------------------------------*/
#define CCE_INTRINSIC_CREATE_MATRIX(DataType, ValueType, TypeStr, DataStr)     \
  CCE_INTRINSIC[aicore] void create_##DataStr##_matrix(                        \
      __##DataStr##__ DataType *dst, int64_t repeat, ValueType value) {        \
    create_##DataStr##_matrix_##TypeStr(dst, repeat, value);                   \
  }

#if (__CCE_AICORE__ == 220) || (__CCE_AICORE__ == 300) || (defined __DAV_C310__)
CCE_INTRINSIC_CREATE_MATRIX(bfloat16_t, half, h, ca);
CCE_INTRINSIC_CREATE_MATRIX(bfloat16_t, uint32_t, ui, ca);
CCE_INTRINSIC_CREATE_MATRIX(bfloat16_t, bfloat16_t, bf16, ca);
CCE_INTRINSIC_CREATE_MATRIX(bfloat16_t, half, h, cb);
CCE_INTRINSIC_CREATE_MATRIX(bfloat16_t, uint32_t, ui, cb);
CCE_INTRINSIC_CREATE_MATRIX(bfloat16_t, bfloat16_t, bf16, cb);
CCE_INTRINSIC_CREATE_MATRIX(bfloat16_t, half, h, cbuf);
CCE_INTRINSIC_CREATE_MATRIX(bfloat16_t, uint32_t, ui, cbuf);
CCE_INTRINSIC_CREATE_MATRIX(bfloat16_t, bfloat16_t, bf16, cbuf);
#endif
#undef CCE_INTRINSIC_CREATE_MATRIX

/*-------------------------------SET_VA_REG------------------------------------*/
#define SET_VA_REG_E0(regid, xn, xm) MOVEVA((regid), 0, (xn), (xm))
#define SET_VA_REG_E2(regid, xn, xm) MOVEVA((regid), 2, (xn), (xm))
#define SET_VA_REG_E4(regid, xn, xm) MOVEVA((regid), 4, (xn), (xm))
#define SET_VA_REG_E6(regid, xn, xm) MOVEVA((regid), 6, (xn), (xm))
// For V210, this instruction was pushq instruction, different from V200.
#define SET_VA_REG_E0_V210(regid, xn, xm) MOVEVA_V210((regid), 0, (xn), (xm))
#define SET_VA_REG_E2_V210(regid, xn, xm) MOVEVA_V210((regid), 2, (xn), (xm))
#define SET_VA_REG_E4_V210(regid, xn, xm) MOVEVA_V210((regid), 4, (xn), (xm))
#define SET_VA_REG_E6_V210(regid, xn, xm) MOVEVA_V210((regid), 6, (xn), (xm))
// For V300, this instruction was pushq instruction, different from V200.
#define SET_VA_REG_E0_V300(regid, xn, xm) MOVEVA_V300((regid), 0, (xn), (xm))
#define SET_VA_REG_E2_V300(regid, xn, xm) MOVEVA_V300((regid), 2, (xn), (xm))
#define SET_VA_REG_E4_V300(regid, xn, xm) MOVEVA_V300((regid), 4, (xn), (xm))
#define SET_VA_REG_E6_V300(regid, xn, xm) MOVEVA_V300((regid), 6, (xn), (xm))

// For compatibility of V200 and V210(pushq instructions)
#if __CCE_AICORE__ == 210
// stack version : MOVEVA VAd[n], Xn, Xm
CCE_INTRINSIC[aicore] void set_va_reg_sb(ub_addr8_t addr, uint64_t *array) {
  switch (addr) {
  case VA0:
    SET_VA_REG_E0_V210(VA0, array[0], array[1]);
    SET_VA_REG_E2_V210(VA0, array[2], array[3]);
    SET_VA_REG_E4_V210(VA0, array[4], array[5]);
    SET_VA_REG_E6_V210(VA0, array[6], array[7]);
    break;
  case VA1:
    SET_VA_REG_E0_V210(VA1, array[0], array[1]);
    SET_VA_REG_E2_V210(VA1, array[2], array[3]);
    SET_VA_REG_E4_V210(VA1, array[4], array[5]);
    SET_VA_REG_E6_V210(VA1, array[6], array[7]);
    break;
  case VA2:
    SET_VA_REG_E0_V210(VA2, array[0], array[1]);
    SET_VA_REG_E2_V210(VA2, array[2], array[3]);
    SET_VA_REG_E4_V210(VA2, array[4], array[5]);
    SET_VA_REG_E6_V210(VA2, array[6], array[7]);
    break;
  case VA3:
    SET_VA_REG_E0_V210(VA3, array[0], array[1]);
    SET_VA_REG_E2_V210(VA3, array[2], array[3]);
    SET_VA_REG_E4_V210(VA3, array[4], array[5]);
    SET_VA_REG_E6_V210(VA3, array[6], array[7]);
    break;
  case VA4:
    SET_VA_REG_E0_V210(VA4, array[0], array[1]);
    SET_VA_REG_E2_V210(VA4, array[2], array[3]);
    SET_VA_REG_E4_V210(VA4, array[4], array[5]);
    SET_VA_REG_E6_V210(VA4, array[6], array[7]);
    break;
  case VA5:
    SET_VA_REG_E0_V210(VA5, array[0], array[1]);
    SET_VA_REG_E2_V210(VA5, array[2], array[3]);
    SET_VA_REG_E4_V210(VA5, array[4], array[5]);
    SET_VA_REG_E6_V210(VA5, array[6], array[7]);
    break;
  case VA6:
    SET_VA_REG_E0_V210(VA6, array[0], array[1]);
    SET_VA_REG_E2_V210(VA6, array[2], array[3]);
    SET_VA_REG_E4_V210(VA6, array[4], array[5]);
    SET_VA_REG_E6_V210(VA6, array[6], array[7]);
    break;
  case VA7:
    SET_VA_REG_E0_V210(VA7, array[0], array[1]);
    SET_VA_REG_E2_V210(VA7, array[2], array[3]);
    SET_VA_REG_E4_V210(VA7, array[4], array[5]);
    SET_VA_REG_E6_V210(VA7, array[6], array[7]);
    break;
  default:
    break;
  }
}
// UBuf version
CCE_INTRINSIC[aicore] void set_va_reg(ub_addr8_t addr,
                                      __ubuf__ uint64_t *array) {
  switch (addr) {
  case VA0:
    SET_VA_REG_E0_V210(VA0, array[0], array[1]);
    SET_VA_REG_E2_V210(VA0, array[2], array[3]);
    SET_VA_REG_E4_V210(VA0, array[4], array[5]);
    SET_VA_REG_E6_V210(VA0, array[6], array[7]);
    break;
  case VA1:
    SET_VA_REG_E0_V210(VA1, array[0], array[1]);
    SET_VA_REG_E2_V210(VA1, array[2], array[3]);
    SET_VA_REG_E4_V210(VA1, array[4], array[5]);
    SET_VA_REG_E6_V210(VA1, array[6], array[7]);
    break;
  case VA2:
    SET_VA_REG_E0_V210(VA2, array[0], array[1]);
    SET_VA_REG_E2_V210(VA2, array[2], array[3]);
    SET_VA_REG_E4_V210(VA2, array[4], array[5]);
    SET_VA_REG_E6_V210(VA2, array[6], array[7]);
    break;
  case VA3:
    SET_VA_REG_E0_V210(VA3, array[0], array[1]);
    SET_VA_REG_E2_V210(VA3, array[2], array[3]);
    SET_VA_REG_E4_V210(VA3, array[4], array[5]);
    SET_VA_REG_E6_V210(VA3, array[6], array[7]);
    break;
  case VA4:
    SET_VA_REG_E0_V210(VA4, array[0], array[1]);
    SET_VA_REG_E2_V210(VA4, array[2], array[3]);
    SET_VA_REG_E4_V210(VA4, array[4], array[5]);
    SET_VA_REG_E6_V210(VA4, array[6], array[7]);
    break;
  case VA5:
    SET_VA_REG_E0_V210(VA5, array[0], array[1]);
    SET_VA_REG_E2_V210(VA5, array[2], array[3]);
    SET_VA_REG_E4_V210(VA5, array[4], array[5]);
    SET_VA_REG_E6_V210(VA5, array[6], array[7]);
    break;
  case VA6:
    SET_VA_REG_E0_V210(VA6, array[0], array[1]);
    SET_VA_REG_E2_V210(VA6, array[2], array[3]);
    SET_VA_REG_E4_V210(VA6, array[4], array[5]);
    SET_VA_REG_E6_V210(VA6, array[6], array[7]);
    break;
  case VA7:
    SET_VA_REG_E0_V210(VA7, array[0], array[1]);
    SET_VA_REG_E2_V210(VA7, array[2], array[3]);
    SET_VA_REG_E4_V210(VA7, array[4], array[5]);
    SET_VA_REG_E6_V210(VA7, array[6], array[7]);
    break;
  default:
    break;
  }
}

#elif (__CCE_AICORE__ == 300) || (__CCE_AICORE__ == 310)
// stack version : MOVEVA VAd[n], Xn, Xm
CCE_INTRINSIC[aicore] void set_va_reg_sb(ub_addr8_t addr, uint64_t *array) {
  switch (addr) {
  case VA0:
    SET_VA_REG_E0_V300(VA0, array[0], array[1]);
    SET_VA_REG_E2_V300(VA0, array[2], array[3]);
    SET_VA_REG_E4_V300(VA0, array[4], array[5]);
    SET_VA_REG_E6_V300(VA0, array[6], array[7]);
    break;
  case VA1:
    SET_VA_REG_E0_V300(VA1, array[0], array[1]);
    SET_VA_REG_E2_V300(VA1, array[2], array[3]);
    SET_VA_REG_E4_V300(VA1, array[4], array[5]);
    SET_VA_REG_E6_V300(VA1, array[6], array[7]);
    break;
  case VA2:
    SET_VA_REG_E0_V300(VA2, array[0], array[1]);
    SET_VA_REG_E2_V300(VA2, array[2], array[3]);
    SET_VA_REG_E4_V300(VA2, array[4], array[5]);
    SET_VA_REG_E6_V300(VA2, array[6], array[7]);
    break;
  case VA3:
    SET_VA_REG_E0_V300(VA3, array[0], array[1]);
    SET_VA_REG_E2_V300(VA3, array[2], array[3]);
    SET_VA_REG_E4_V300(VA3, array[4], array[5]);
    SET_VA_REG_E6_V300(VA3, array[6], array[7]);
    break;
  case VA4:
    SET_VA_REG_E0_V300(VA4, array[0], array[1]);
    SET_VA_REG_E2_V300(VA4, array[2], array[3]);
    SET_VA_REG_E4_V300(VA4, array[4], array[5]);
    SET_VA_REG_E6_V300(VA4, array[6], array[7]);
    break;
  case VA5:
    SET_VA_REG_E0_V300(VA5, array[0], array[1]);
    SET_VA_REG_E2_V300(VA5, array[2], array[3]);
    SET_VA_REG_E4_V300(VA5, array[4], array[5]);
    SET_VA_REG_E6_V300(VA5, array[6], array[7]);
    break;
  case VA6:
    SET_VA_REG_E0_V300(VA6, array[0], array[1]);
    SET_VA_REG_E2_V300(VA6, array[2], array[3]);
    SET_VA_REG_E4_V300(VA6, array[4], array[5]);
    SET_VA_REG_E6_V300(VA6, array[6], array[7]);
    break;
  case VA7:
    SET_VA_REG_E0_V300(VA7, array[0], array[1]);
    SET_VA_REG_E2_V300(VA7, array[2], array[3]);
    SET_VA_REG_E4_V300(VA7, array[4], array[5]);
    SET_VA_REG_E6_V300(VA7, array[6], array[7]);
    break;
  default:
    break;
  }
}
// UBuf version
CCE_INTRINSIC[aicore] void set_va_reg(ub_addr8_t addr,
                                      __ubuf__ uint64_t *array) {
  switch (addr) {
  case VA0:
    SET_VA_REG_E0_V300(VA0, array[0], array[1]);
    SET_VA_REG_E2_V300(VA0, array[2], array[3]);
    SET_VA_REG_E4_V300(VA0, array[4], array[5]);
    SET_VA_REG_E6_V300(VA0, array[6], array[7]);
    break;
  case VA1:
    SET_VA_REG_E0_V300(VA1, array[0], array[1]);
    SET_VA_REG_E2_V300(VA1, array[2], array[3]);
    SET_VA_REG_E4_V300(VA1, array[4], array[5]);
    SET_VA_REG_E6_V300(VA1, array[6], array[7]);
    break;
  case VA2:
    SET_VA_REG_E0_V300(VA2, array[0], array[1]);
    SET_VA_REG_E2_V300(VA2, array[2], array[3]);
    SET_VA_REG_E4_V300(VA2, array[4], array[5]);
    SET_VA_REG_E6_V300(VA2, array[6], array[7]);
    break;
  case VA3:
    SET_VA_REG_E0_V300(VA3, array[0], array[1]);
    SET_VA_REG_E2_V300(VA3, array[2], array[3]);
    SET_VA_REG_E4_V300(VA3, array[4], array[5]);
    SET_VA_REG_E6_V300(VA3, array[6], array[7]);
    break;
  case VA4:
    SET_VA_REG_E0_V300(VA4, array[0], array[1]);
    SET_VA_REG_E2_V300(VA4, array[2], array[3]);
    SET_VA_REG_E4_V300(VA4, array[4], array[5]);
    SET_VA_REG_E6_V300(VA4, array[6], array[7]);
    break;
  case VA5:
    SET_VA_REG_E0_V300(VA5, array[0], array[1]);
    SET_VA_REG_E2_V300(VA5, array[2], array[3]);
    SET_VA_REG_E4_V300(VA5, array[4], array[5]);
    SET_VA_REG_E6_V300(VA5, array[6], array[7]);
    break;
  case VA6:
    SET_VA_REG_E0_V300(VA6, array[0], array[1]);
    SET_VA_REG_E2_V300(VA6, array[2], array[3]);
    SET_VA_REG_E4_V300(VA6, array[4], array[5]);
    SET_VA_REG_E6_V300(VA6, array[6], array[7]);
    break;
  case VA7:
    SET_VA_REG_E0_V300(VA7, array[0], array[1]);
    SET_VA_REG_E2_V300(VA7, array[2], array[3]);
    SET_VA_REG_E4_V300(VA7, array[4], array[5]);
    SET_VA_REG_E6_V300(VA7, array[6], array[7]);
    break;
  default:
    break;
  }
}

#elif __CCE_AICORE__ != 350
// stack version : MOVEVA VAd[n], Xn, Xm
CCE_INTRINSIC[aicore] void set_va_reg_sb(ub_addr8_t addr, uint64_t *array) {
  switch (addr) {
  case VA0:
    SET_VA_REG_E0(VA0, array[0], array[1]);
    SET_VA_REG_E2(VA0, array[2], array[3]);
    SET_VA_REG_E4(VA0, array[4], array[5]);
    SET_VA_REG_E6(VA0, array[6], array[7]);
    break;
  case VA1:
    SET_VA_REG_E0(VA1, array[0], array[1]);
    SET_VA_REG_E2(VA1, array[2], array[3]);
    SET_VA_REG_E4(VA1, array[4], array[5]);
    SET_VA_REG_E6(VA1, array[6], array[7]);
    break;
  case VA2:
    SET_VA_REG_E0(VA2, array[0], array[1]);
    SET_VA_REG_E2(VA2, array[2], array[3]);
    SET_VA_REG_E4(VA2, array[4], array[5]);
    SET_VA_REG_E6(VA2, array[6], array[7]);
    break;
  case VA3:
    SET_VA_REG_E0(VA3, array[0], array[1]);
    SET_VA_REG_E2(VA3, array[2], array[3]);
    SET_VA_REG_E4(VA3, array[4], array[5]);
    SET_VA_REG_E6(VA3, array[6], array[7]);
    break;
  case VA4:
    SET_VA_REG_E0(VA4, array[0], array[1]);
    SET_VA_REG_E2(VA4, array[2], array[3]);
    SET_VA_REG_E4(VA4, array[4], array[5]);
    SET_VA_REG_E6(VA4, array[6], array[7]);
    break;
  case VA5:
    SET_VA_REG_E0(VA5, array[0], array[1]);
    SET_VA_REG_E2(VA5, array[2], array[3]);
    SET_VA_REG_E4(VA5, array[4], array[5]);
    SET_VA_REG_E6(VA5, array[6], array[7]);
    break;
  case VA6:
    SET_VA_REG_E0(VA6, array[0], array[1]);
    SET_VA_REG_E2(VA6, array[2], array[3]);
    SET_VA_REG_E4(VA6, array[4], array[5]);
    SET_VA_REG_E6(VA6, array[6], array[7]);
    break;
  case VA7:
    SET_VA_REG_E0(VA7, array[0], array[1]);
    SET_VA_REG_E2(VA7, array[2], array[3]);
    SET_VA_REG_E4(VA7, array[4], array[5]);
    SET_VA_REG_E6(VA7, array[6], array[7]);
    break;
  default:
    break;
  }
}
// UBuf version
CCE_INTRINSIC[aicore] void set_va_reg(ub_addr8_t addr,
                                      __ubuf__ uint64_t *array) {
  switch (addr) {
  case VA0:
    SET_VA_REG_E0(VA0, array[0], array[1]);
    SET_VA_REG_E2(VA0, array[2], array[3]);
    SET_VA_REG_E4(VA0, array[4], array[5]);
    SET_VA_REG_E6(VA0, array[6], array[7]);
    break;
  case VA1:
    SET_VA_REG_E0(VA1, array[0], array[1]);
    SET_VA_REG_E2(VA1, array[2], array[3]);
    SET_VA_REG_E4(VA1, array[4], array[5]);
    SET_VA_REG_E6(VA1, array[6], array[7]);
    break;
  case VA2:
    SET_VA_REG_E0(VA2, array[0], array[1]);
    SET_VA_REG_E2(VA2, array[2], array[3]);
    SET_VA_REG_E4(VA2, array[4], array[5]);
    SET_VA_REG_E6(VA2, array[6], array[7]);
    break;
  case VA3:
    SET_VA_REG_E0(VA3, array[0], array[1]);
    SET_VA_REG_E2(VA3, array[2], array[3]);
    SET_VA_REG_E4(VA3, array[4], array[5]);
    SET_VA_REG_E6(VA3, array[6], array[7]);
    break;
  case VA4:
    SET_VA_REG_E0(VA4, array[0], array[1]);
    SET_VA_REG_E2(VA4, array[2], array[3]);
    SET_VA_REG_E4(VA4, array[4], array[5]);
    SET_VA_REG_E6(VA4, array[6], array[7]);
    break;
  case VA5:
    SET_VA_REG_E0(VA5, array[0], array[1]);
    SET_VA_REG_E2(VA5, array[2], array[3]);
    SET_VA_REG_E4(VA5, array[4], array[5]);
    SET_VA_REG_E6(VA5, array[6], array[7]);
    break;
  case VA6:
    SET_VA_REG_E0(VA6, array[0], array[1]);
    SET_VA_REG_E2(VA6, array[2], array[3]);
    SET_VA_REG_E4(VA6, array[4], array[5]);
    SET_VA_REG_E6(VA6, array[6], array[7]);
    break;
  case VA7:
    SET_VA_REG_E0(VA7, array[0], array[1]);
    SET_VA_REG_E2(VA7, array[2], array[3]);
    SET_VA_REG_E4(VA7, array[4], array[5]);
    SET_VA_REG_E6(VA7, array[6], array[7]);
    break;
  default:
    break;
  }
}
#endif

#undef SET_VA_REG_E0_V210
#undef SET_VA_REG_E2_V210
#undef SET_VA_REG_E4_V210
#undef SET_VA_REG_E6_V210
#undef SET_VA_REG_E0_V300
#undef SET_VA_REG_E2_V300
#undef SET_VA_REG_E4_V300
#undef SET_VA_REG_E6_V300
#undef SET_VA_REG_E0
#undef SET_VA_REG_E2
#undef SET_VA_REG_E4
#undef SET_VA_REG_E6

// V210 mov fixpipe dual-output interface.
/*--------------------FIXPIPE_DUAL_OUTPUT-------------------*/
// DUAL_MODE = 0b0100, dst0 after req, dst1 before req.
#if __CCE_AICORE__ == 210
CCE_INTRINSIC[aicore] void
fix_matrix_cc_to_cbuf_dualout(__cbuf__ uint8_t *dst0, __cbuf__ half *dst1,
                              __cc__ half *src, uint64_t xm, uint64_t xt) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  FIX_L0C16_TO_L1_f162u8((__cbuf__ uint8_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void
fix_matrix_cc_to_cbuf_dualout(__cbuf__ int8_t *dst0, __cbuf__ half *dst1,
                              __cc__ half *src, uint64_t xm, uint64_t xt) {
  int64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  FIX_L0C16_TO_L1_f162s8((__cbuf__ int8_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void
fix_matrix_cc_to_cbuf_dualout(__cbuf__ int8_t *dst0, __cbuf__ half *dst1,
                              __cc__ int32_t *src, uint64_t xm, uint64_t xt) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  FIX_L0C32_TO_L1_s322s8((__cbuf__ int8_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void
fix_matrix_cc_to_cbuf_dualout(__cbuf__ int8_t *dst0, __cbuf__ int16_t *dst1,
                              __cc__ int32_t *src, uint64_t xm, uint64_t xt) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  FIX_L0C32_TO_L1_s322s8((__cbuf__ int8_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void
fix_matrix_cc_to_cbuf_dualout(__cbuf__ uint8_t *dst0, __cbuf__ half *dst1,
                              __cc__ int32_t *src, uint64_t xm, uint64_t xt) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  FIX_L0C32_TO_L1_s322u8((__cbuf__ uint8_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void
fix_matrix_cc_to_cbuf_dualout(__cbuf__ uint8_t *dst0, __cbuf__ int16_t *dst1,
                              __cc__ int32_t *src, uint64_t xm, uint64_t xt) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  FIX_L0C32_TO_L1_s322u8((__cbuf__ uint8_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void
fix_winograd_cc_to_cbuf_dualout(__cbuf__ int8_t *dst0, __cbuf__ half *dst1,
                                __cc__ int32_t *src, uint64_t xm, uint64_t xt) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  FIX_L0CWINO32_TO_L1_s322s8((__cbuf__ int8_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void
fix_winograd_cc_to_cbuf_dualout(__cbuf__ int8_t *dst0, __cbuf__ int16_t *dst1,
                                __cc__ int32_t *src, uint64_t xm, uint64_t xt) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  FIX_L0CWINO32_TO_L1_s322s8((__cbuf__ int8_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void
fix_winograd_cc_to_cbuf_dualout(__cbuf__ uint8_t *dst0, __cbuf__ half *dst1,
                                __cc__ int32_t *src, uint64_t xm, uint64_t xt) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  FIX_L0CWINO32_TO_L1_s322u8((__cbuf__ uint8_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void
fix_winograd_cc_to_cbuf_dualout(__cbuf__ uint8_t *dst0, __cbuf__ int16_t *dst1,
                                __cc__ int32_t *src, uint64_t xm, uint64_t xt) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  FIX_L0CWINO32_TO_L1_s322u8((__cbuf__ uint8_t *)dst, src, xm, xt);
}

// DUAL_MODE=0b0011, dst0 final output, dst1 before up-sampling.
CCE_INTRINSIC[aicore] void
fix_matrix_cc_to_cbuf_dualout(__cbuf__ half *dst0, __cbuf__ half *dst1,
                              __cc__ half *src, uint64_t xm, uint64_t xt) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  FIX_L0C16_TO_L1_f16((__cbuf__ half *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void
fix_matrix_cc_to_cbuf_dualout(__cbuf__ half *dst0, __cbuf__ half *dst1,
                              __cc__ int32_t *src, uint64_t xm, uint64_t xt) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  FIX_L0C32_TO_L1_s322f16((__cbuf__ half *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void
fix_matrix_cc_to_cbuf_dualout(__cbuf__ int8_t *dst0, __cbuf__ int8_t *dst1,
                              __cc__ int32_t *src, uint64_t xm, uint64_t xt) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  FIX_L0C32_TO_L1_s322s8((__cbuf__ int8_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void
fix_winograd_cc_to_cbuf_dualout(__cbuf__ half *dst0, __cbuf__ half *dst1,
                                __cc__ int32_t *src, uint64_t xm, uint64_t xt) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  FIX_L0CWINO32_TO_L1_s322f16((__cbuf__ half *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void
fix_winograd_cc_to_cbuf_dualout(__cbuf__ int8_t *dst0, __cbuf__ int8_t *dst1,
                                __cc__ int32_t *src, uint64_t xm, uint64_t xt) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  FIX_L0CWINO32_TO_L1_s322s8((__cbuf__ int8_t *)dst, src, xm, xt);
}

// DUAL_MODE=0b0010, dst0 final output, dst1 before pooling.
CCE_INTRINSIC[aicore] void
fix_winograd_cc_to_cbuf_dualout(__cbuf__ int32_t *dst0, __cbuf__ int32_t *dst1,
                                __cc__ int32_t *src, uint64_t xm, uint64_t xt) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  FIX_L0CWINO32_TO_L1_s32((__cbuf__ int32_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void
fix_winograd_cc_to_cbuf_dualout(__cbuf__ int16_t *dst0, __cbuf__ int16_t *dst1,
                                __cc__ int32_t *src, uint64_t xm, uint64_t xt) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  FIX_L0CWINO32_TO_L1_s322s16((__cbuf__ int16_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void
fix_winograd_cc_to_cbuf_dualout(__cbuf__ uint8_t *dst0, __cbuf__ uint8_t *dst1,
                                __cc__ int32_t *src, uint64_t xm, uint64_t xt) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  FIX_L0CWINO32_TO_L1_s322u8((__cbuf__ uint8_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void
fix_matrix_cc_to_cbufubuf_dualout(__ubuf__ half *dst0, __cbuf__ half *dst1,
                                  __cc__ half *src, uint64_t xm, uint64_t xt) {
  uint64_t dst = 0;
  dst = ((uint64_t)dst0 << 32) | (uint64_t)dst1;
  FIX_L0C16_TO_L1UB_f16((__ubuf__ half *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void
fix_matrix_cc_to_cbufubuf_dualout(__ubuf__ int32_t *dst0,
                                  __cbuf__ int32_t *dst1, __cc__ int32_t *src,
                                  uint64_t xm, uint64_t xt) {
  uint64_t dst = 0;
  dst = ((uint64_t)dst0 << 32) | (uint64_t)dst1;
  FIX_L0C32_TO_L1UB_s32((__ubuf__ int32_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void
fix_matrix_cc_to_cbufubuf_dualout(__ubuf__ half *dst0, __cbuf__ half *dst1,
                                  __cc__ int32_t *src, uint64_t xm,
                                  uint64_t xt) {
  uint64_t dst = 0;
  dst = ((uint64_t)dst0 << 32) | (uint64_t)dst1;
  FIX_L0C32_TO_L1UB_s322f16((__ubuf__ half *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void
fix_matrix_cc_to_cbufubuf_dualout(__ubuf__ int16_t *dst0,
                                  __cbuf__ int16_t *dst1, __cc__ int32_t *src,
                                  uint64_t xm, uint64_t xt) {
  uint64_t dst = 0;
  dst = ((uint64_t)dst0 << 32) | (uint64_t)dst1;
  FIX_L0C32_TO_L1UB_s322s16((__ubuf__ int16_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void
fix_matrix_cc_to_cbufubuf_dualout(__ubuf__ int8_t *dst0, __cbuf__ int8_t *dst1,
                                  __cc__ int32_t *src, uint64_t xm,
                                  uint64_t xt) {
  uint64_t dst = 0;
  dst = ((uint64_t)dst0 << 32) | (uint64_t)dst1;
  FIX_L0C32_TO_L1UB_s322s8((__ubuf__ int8_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void
fix_matrix_cc_to_cbufubuf_dualout(__ubuf__ uint8_t *dst0,
                                  __cbuf__ uint8_t *dst1, __cc__ int32_t *src,
                                  uint64_t xm, uint64_t xt) {
  uint64_t dst = 0;
  dst = ((uint64_t)dst0 << 32) | (uint64_t)dst1;
  FIX_L0C32_TO_L1UB_s322u8((__ubuf__ uint8_t *)dst, src, xm, xt);
}

#endif

/*-------------------------------FIXPIPE_CFG_DUAL_OUTPUT------------------------------------*/
// CFG interfaces
#if __CCE_AICORE__ == 210
static __attribute__((cce_builtin_api, always_inline))[aicore] uint64_t
__set_fixpipe_xm(unit_flag_t unitFlagMode, uint8_t elewiseOp, uint16_t NSize,
                 uint16_t MSize, uint16_t srcBurstGap, uint16_t dstBurstGap) {
  uint64_t config;
  config = ((uint64_t)(unitFlagMode & 0x3ULL)) |
           ((uint64_t)(elewiseOp & 0x3ULL) << 2) |
           ((uint64_t)(NSize & 0xFFFULL) << 4) |
           ((uint64_t)(MSize & 0xFFFFULL) << 16) |
           ((uint64_t)(srcBurstGap & 0xFFFFULL) << 32) |
           ((uint64_t)(dstBurstGap & 0xFFFFULL) << 48);
  return config;
}

static __attribute__((cce_builtin_api, always_inline))[aicore] uint64_t
__set_fixpipe_xt(ConvReluFix_t crMode, bool biasEn, Relu_t relub,
                 bool elewiseEn, Relu_t relua, Pool_t pool, Req_t req,
                 DualMode_t dualMode, uint16_t Ws, uint16_t WSize) {
  uint64_t config;
  config =
      ((uint64_t)(crMode & 0xFULL)) | ((uint64_t)(biasEn & 0x1ULL) << 4) |
      ((uint64_t)(relub & 0x3ULL) << 5) |
      ((uint64_t)(elewiseEn & 0x1ULL) << 7) |
      ((uint64_t)(relua & 0x3ULL) << 8) | ((uint64_t)(pool & 0x3ULL) << 10) |
      ((uint64_t)(req & 0x3ULL) << 12) | ((uint64_t)(dualMode & 0xFULL) << 16) |
      ((uint64_t)(Ws & 0xFFFFULL) << 32) |
      ((uint64_t)(WSize & 0xFFFFULL) << 48);
  return config;
}

// DUAL_MODE = 0b0100, dst0 after req, dst1 before req.
CCE_INTRINSIC[aicore] void fix_matrix_cc_to_cbuf_dualout(
    __cbuf__ uint8_t *dst0, __cbuf__ half *dst1, __cc__ half *src,
    unit_flag_t unitFlagMode, uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
    uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
    bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
    Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  uint64_t xm = __set_fixpipe_xm(unitFlagMode, elewiseOp, NSize, MSize,
                                 srcBurstGap, dstBurstGap);
  uint64_t xt = __set_fixpipe_xt(crMode, biasEn, relub, elewiseEn, relua, pool,
                                 req, dualMode, Ws, WSize);
  FIX_L0C16_TO_L1_f162u8((__cbuf__ uint8_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void fix_matrix_cc_to_cbuf_dualout(
    __cbuf__ int8_t *dst0, __cbuf__ half *dst1, __cc__ half *src,
    unit_flag_t unitFlagMode, uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
    uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
    bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
    Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  uint64_t xm = __set_fixpipe_xm(unitFlagMode, elewiseOp, NSize, MSize,
                                 srcBurstGap, dstBurstGap);
  uint64_t xt = __set_fixpipe_xt(crMode, biasEn, relub, elewiseEn, relua, pool,
                                 req, dualMode, Ws, WSize);
  FIX_L0C16_TO_L1_f162s8((__cbuf__ int8_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void fix_matrix_cc_to_cbuf_dualout(
    __cbuf__ int8_t *dst0, __cbuf__ half *dst1, __cc__ int32_t *src,
    unit_flag_t unitFlagMode, uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
    uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
    bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
    Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  uint64_t xm = __set_fixpipe_xm(unitFlagMode, elewiseOp, NSize, MSize,
                                 srcBurstGap, dstBurstGap);
  uint64_t xt = __set_fixpipe_xt(crMode, biasEn, relub, elewiseEn, relua, pool,
                                 req, dualMode, Ws, WSize);
  FIX_L0C32_TO_L1_s322s8((__cbuf__ int8_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void fix_matrix_cc_to_cbuf_dualout(
    __cbuf__ int8_t *dst0, __cbuf__ int16_t *dst1, __cc__ int32_t *src,
    unit_flag_t unitFlagMode, uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
    uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
    bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
    Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  uint64_t xm = __set_fixpipe_xm(unitFlagMode, elewiseOp, NSize, MSize,
                                 srcBurstGap, dstBurstGap);
  uint64_t xt = __set_fixpipe_xt(crMode, biasEn, relub, elewiseEn, relua, pool,
                                 req, dualMode, Ws, WSize);
  FIX_L0C32_TO_L1_s322s8((__cbuf__ int8_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void fix_matrix_cc_to_cbuf_dualout(
    __cbuf__ uint8_t *dst0, __cbuf__ half *dst1, __cc__ int32_t *src,
    unit_flag_t unitFlagMode, uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
    uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
    bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
    Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  uint64_t xm = __set_fixpipe_xm(unitFlagMode, elewiseOp, NSize, MSize,
                                 srcBurstGap, dstBurstGap);
  uint64_t xt = __set_fixpipe_xt(crMode, biasEn, relub, elewiseEn, relua, pool,
                                 req, dualMode, Ws, WSize);
  FIX_L0C32_TO_L1_s322u8((__cbuf__ uint8_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void fix_matrix_cc_to_cbuf_dualout(
    __cbuf__ uint8_t *dst0, __cbuf__ int16_t *dst1, __cc__ int32_t *src,
    unit_flag_t unitFlagMode, uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
    uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
    bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
    Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  uint64_t xm = __set_fixpipe_xm(unitFlagMode, elewiseOp, NSize, MSize,
                                 srcBurstGap, dstBurstGap);
  uint64_t xt = __set_fixpipe_xt(crMode, biasEn, relub, elewiseEn, relua, pool,
                                 req, dualMode, Ws, WSize);
  FIX_L0C32_TO_L1_s322u8((__cbuf__ uint8_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void fix_winograd_cc_to_cbuf_dualout(
    __cbuf__ int8_t *dst0, __cbuf__ half *dst1, __cc__ int32_t *src,
    unit_flag_t unitFlagMode, uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
    uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
    bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
    Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  uint64_t xm = __set_fixpipe_xm(unitFlagMode, elewiseOp, NSize, MSize,
                                 srcBurstGap, dstBurstGap);
  uint64_t xt = __set_fixpipe_xt(crMode, biasEn, relub, elewiseEn, relua, pool,
                                 req, dualMode, Ws, WSize);
  FIX_L0CWINO32_TO_L1_s322s8((__cbuf__ int8_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void fix_winograd_cc_to_cbuf_dualout(
    __cbuf__ int8_t *dst0, __cbuf__ int16_t *dst1, __cc__ int32_t *src,
    unit_flag_t unitFlagMode, uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
    uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
    bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
    Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  uint64_t xm = __set_fixpipe_xm(unitFlagMode, elewiseOp, NSize, MSize,
                                 srcBurstGap, dstBurstGap);
  uint64_t xt = __set_fixpipe_xt(crMode, biasEn, relub, elewiseEn, relua, pool,
                                 req, dualMode, Ws, WSize);
  FIX_L0CWINO32_TO_L1_s322s8((__cbuf__ int8_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void fix_winograd_cc_to_cbuf_dualout(
    __cbuf__ uint8_t *dst0, __cbuf__ half *dst1, __cc__ int32_t *src,
    unit_flag_t unitFlagMode, uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
    uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
    bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
    Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  uint64_t xm = __set_fixpipe_xm(unitFlagMode, elewiseOp, NSize, MSize,
                                 srcBurstGap, dstBurstGap);
  uint64_t xt = __set_fixpipe_xt(crMode, biasEn, relub, elewiseEn, relua, pool,
                                 req, dualMode, Ws, WSize);
  FIX_L0CWINO32_TO_L1_s322u8((__cbuf__ uint8_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void fix_winograd_cc_to_cbuf_dualout(
    __cbuf__ uint8_t *dst0, __cbuf__ int16_t *dst1, __cc__ int32_t *src,
    unit_flag_t unitFlagMode, uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
    uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
    bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
    Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  uint64_t xm = __set_fixpipe_xm(unitFlagMode, elewiseOp, NSize, MSize,
                                 srcBurstGap, dstBurstGap);
  uint64_t xt = __set_fixpipe_xt(crMode, biasEn, relub, elewiseEn, relua, pool,
                                 req, dualMode, Ws, WSize);
  FIX_L0CWINO32_TO_L1_s322u8((__cbuf__ uint8_t *)dst, src, xm, xt);
}

// DUAL_MODE=0b0011, dst0 final output, dst1 before up-sampling.
CCE_INTRINSIC[aicore] void fix_matrix_cc_to_cbuf_dualout(
    __cbuf__ half *dst0, __cbuf__ half *dst1, __cc__ half *src,
    unit_flag_t unitFlagMode, uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
    uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
    bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
    Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  uint64_t xm = __set_fixpipe_xm(unitFlagMode, elewiseOp, NSize, MSize,
                                 srcBurstGap, dstBurstGap);
  uint64_t xt = __set_fixpipe_xt(crMode, biasEn, relub, elewiseEn, relua, pool,
                                 req, dualMode, Ws, WSize);
  FIX_L0C16_TO_L1_f16((__cbuf__ half *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void fix_matrix_cc_to_cbuf_dualout(
    __cbuf__ half *dst0, __cbuf__ half *dst1, __cc__ int32_t *src,
    unit_flag_t unitFlagMode, uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
    uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
    bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
    Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  uint64_t xm = __set_fixpipe_xm(unitFlagMode, elewiseOp, NSize, MSize,
                                 srcBurstGap, dstBurstGap);
  uint64_t xt = __set_fixpipe_xt(crMode, biasEn, relub, elewiseEn, relua, pool,
                                 req, dualMode, Ws, WSize);
  FIX_L0C32_TO_L1_s322f16((__cbuf__ half *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void fix_matrix_cc_to_cbuf_dualout(
    __cbuf__ int8_t *dst0, __cbuf__ int8_t *dst1, __cc__ int32_t *src,
    unit_flag_t unitFlagMode, uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
    uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
    bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
    Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  uint64_t xm = __set_fixpipe_xm(unitFlagMode, elewiseOp, NSize, MSize,
                                 srcBurstGap, dstBurstGap);
  uint64_t xt = __set_fixpipe_xt(crMode, biasEn, relub, elewiseEn, relua, pool,
                                 req, dualMode, Ws, WSize);
  FIX_L0C32_TO_L1_s322s8((__cbuf__ int8_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void fix_winograd_cc_to_cbuf_dualout(
    __cbuf__ half *dst0, __cbuf__ half *dst1, __cc__ int32_t *src,
    unit_flag_t unitFlagMode, uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
    uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
    bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
    Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  uint64_t xm = __set_fixpipe_xm(unitFlagMode, elewiseOp, NSize, MSize,
                                 srcBurstGap, dstBurstGap);
  uint64_t xt = __set_fixpipe_xt(crMode, biasEn, relub, elewiseEn, relua, pool,
                                 req, dualMode, Ws, WSize);
  FIX_L0CWINO32_TO_L1_s322f16((__cbuf__ half *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void fix_winograd_cc_to_cbuf_dualout(
    __cbuf__ int8_t *dst0, __cbuf__ int8_t *dst1, __cc__ int32_t *src,
    unit_flag_t unitFlagMode, uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
    uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
    bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
    Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  uint64_t xm = __set_fixpipe_xm(unitFlagMode, elewiseOp, NSize, MSize,
                                 srcBurstGap, dstBurstGap);
  uint64_t xt = __set_fixpipe_xt(crMode, biasEn, relub, elewiseEn, relua, pool,
                                 req, dualMode, Ws, WSize);
  FIX_L0CWINO32_TO_L1_s322s8((__cbuf__ int8_t *)dst, src, xm, xt);
}

// DUAL_MODE=0b0010, dst0 final output, dst1 before pooling.
CCE_INTRINSIC[aicore] void fix_winograd_cc_to_cbuf_dualout(
    __cbuf__ int32_t *dst0, __cbuf__ int32_t *dst1, __cc__ int32_t *src,
    unit_flag_t unitFlagMode, uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
    uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
    bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
    Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  uint64_t xm = __set_fixpipe_xm(unitFlagMode, elewiseOp, NSize, MSize,
                                 srcBurstGap, dstBurstGap);
  uint64_t xt = __set_fixpipe_xt(crMode, biasEn, relub, elewiseEn, relua, pool,
                                 req, dualMode, Ws, WSize);
  FIX_L0CWINO32_TO_L1_s32((__cbuf__ int32_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void fix_winograd_cc_to_cbuf_dualout(
    __cbuf__ int16_t *dst0, __cbuf__ int16_t *dst1, __cc__ int32_t *src,
    unit_flag_t unitFlagMode, uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
    uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
    bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
    Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  uint64_t xm = __set_fixpipe_xm(unitFlagMode, elewiseOp, NSize, MSize,
                                 srcBurstGap, dstBurstGap);
  uint64_t xt = __set_fixpipe_xt(crMode, biasEn, relub, elewiseEn, relua, pool,
                                 req, dualMode, Ws, WSize);
  FIX_L0CWINO32_TO_L1_s322s16((__cbuf__ int16_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void fix_winograd_cc_to_cbuf_dualout(
    __cbuf__ uint8_t *dst0, __cbuf__ uint8_t *dst1, __cc__ int32_t *src,
    unit_flag_t unitFlagMode, uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
    uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
    bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
    Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  uint64_t xm = __set_fixpipe_xm(unitFlagMode, elewiseOp, NSize, MSize,
                                 srcBurstGap, dstBurstGap);
  uint64_t xt = __set_fixpipe_xt(crMode, biasEn, relub, elewiseEn, relua, pool,
                                 req, dualMode, Ws, WSize);
  FIX_L0CWINO32_TO_L1_s322u8((__cbuf__ uint8_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void fix_matrix_cc_to_cbufubuf_dualout(
    __ubuf__ half *dst0, __cbuf__ half *dst1, __cc__ half *src,
    unit_flag_t unitFlagMode, uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
    uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
    bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
    Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  uint64_t xm = __set_fixpipe_xm(unitFlagMode, elewiseOp, NSize, MSize,
                                 srcBurstGap, dstBurstGap);
  uint64_t xt = __set_fixpipe_xt(crMode, biasEn, relub, elewiseEn, relua, pool,
                                 req, dualMode, Ws, WSize);
  FIX_L0C16_TO_L1UB_f16((__ubuf__ half *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void fix_matrix_cc_to_cbufubuf_dualout(
    __ubuf__ int32_t *dst0, __cbuf__ int32_t *dst1, __cc__ int32_t *src,
    unit_flag_t unitFlagMode, uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
    uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
    bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
    Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  uint64_t xm = __set_fixpipe_xm(unitFlagMode, elewiseOp, NSize, MSize,
                                 srcBurstGap, dstBurstGap);
  uint64_t xt = __set_fixpipe_xt(crMode, biasEn, relub, elewiseEn, relua, pool,
                                 req, dualMode, Ws, WSize);
  FIX_L0C32_TO_L1UB_s32((__ubuf__ int32_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void fix_matrix_cc_to_cbufubuf_dualout(
    __ubuf__ half *dst0, __cbuf__ half *dst1, __cc__ int32_t *src,
    unit_flag_t unitFlagMode, uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
    uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
    bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
    Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  uint64_t xm = __set_fixpipe_xm(unitFlagMode, elewiseOp, NSize, MSize,
                                 srcBurstGap, dstBurstGap);
  uint64_t xt = __set_fixpipe_xt(crMode, biasEn, relub, elewiseEn, relua, pool,
                                 req, dualMode, Ws, WSize);
  FIX_L0C32_TO_L1UB_s322f16((__ubuf__ half *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void fix_matrix_cc_to_cbufubuf_dualout(
    __ubuf__ int16_t *dst0, __cbuf__ int16_t *dst1, __cc__ int32_t *src,
    unit_flag_t unitFlagMode, uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
    uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
    bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
    Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  uint64_t xm = __set_fixpipe_xm(unitFlagMode, elewiseOp, NSize, MSize,
                                 srcBurstGap, dstBurstGap);
  uint64_t xt = __set_fixpipe_xt(crMode, biasEn, relub, elewiseEn, relua, pool,
                                 req, dualMode, Ws, WSize);
  FIX_L0C32_TO_L1UB_s322s16((__ubuf__ int16_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void fix_matrix_cc_to_cbufubuf_dualout(
    __ubuf__ int8_t *dst0, __cbuf__ int8_t *dst1, __cc__ int32_t *src,
    unit_flag_t unitFlagMode, uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
    uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
    bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
    Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  uint64_t xm = __set_fixpipe_xm(unitFlagMode, elewiseOp, NSize, MSize,
                                 srcBurstGap, dstBurstGap);
  uint64_t xt = __set_fixpipe_xt(crMode, biasEn, relub, elewiseEn, relua, pool,
                                 req, dualMode, Ws, WSize);
  FIX_L0C32_TO_L1UB_s322s8((__ubuf__ int8_t *)dst, src, xm, xt);
}

CCE_INTRINSIC[aicore] void fix_matrix_cc_to_cbufubuf_dualout(
    __ubuf__ uint8_t *dst0, __cbuf__ uint8_t *dst1, __cc__ int32_t *src,
    unit_flag_t unitFlagMode, uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
    uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
    bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
    Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize) {
  uint64_t dst = 0;
  dst = (uint64_t)dst0 | ((uint64_t)dst1 << 32);
  uint64_t xm = __set_fixpipe_xm(unitFlagMode, elewiseOp, NSize, MSize,
                                 srcBurstGap, dstBurstGap);
  uint64_t xt = __set_fixpipe_xt(crMode, biasEn, relub, elewiseEn, relua, pool,
                                 req, dualMode, Ws, WSize);
  FIX_L0C32_TO_L1UB_s322u8((__ubuf__ uint8_t *)dst, src, xm, xt);
}

#endif

/*------------------V300 V310 VBS32------------------*/
#if (__CCE_AICORE__ == 300) || (__CCE_AICORE__ == 310)
CCE_INTRINSIC PIPE_ID(PIPE_V)[aicore] void vbs(__ubuf__ half *dst,
                                               __ubuf__ half *src0,
                                               __ubuf__ uint32_t *src1,
                                               uint64_t config) {
  VBS32_V300_f16(dst, src0, src1, config);
}

CCE_INTRINSIC PIPE_ID(PIPE_V)[aicore] void vbs(__ubuf__ float *dst,
                                               __ubuf__ float *src0,
                                               __ubuf__ uint32_t *src1,
                                               uint64_t config) {
  VBS32_V300_f32(dst, src0, src1, config);
}

CCE_INTRINSIC PIPE_ID(PIPE_V)[aicore] void vbs(
    __ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ uint32_t *src1,
    uint8_t repeat, uint8_t dstBlockStride, uint8_t src0BlockStride,
    uint8_t src1BlockStride, uint8_t dstRepeatStride, uint8_t src0RepeatStride,
    uint8_t src1RepeatStride, bool repeatStrideMode, bool strideSizeMode) {
  uint64_t config = 0;
  config = (((uint64_t)repeat & 0xff) << 56 |
            ((uint64_t)dstBlockStride & 0xff) << 0 |
            ((uint64_t)src0BlockStride & 0xff) << 8 |
            ((uint64_t)src1BlockStride & 0xff) << 16 |
            ((uint64_t)dstRepeatStride & 0xff) << 24 |
            ((uint64_t)src0RepeatStride & 0xff) << 32 |
            ((uint64_t)src1RepeatStride & 0xff) << 40 |
            ((uint64_t)repeatStrideMode & 0x1) << 54 |
            ((uint64_t)strideSizeMode & 0x1) << 55);
  VBS32_V300_f16(dst, src0, src1, config);
}

CCE_INTRINSIC PIPE_ID(PIPE_V)[aicore] void vbs(
    __ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ uint32_t *src1,
    uint8_t repeat, uint8_t dstBlockStride, uint8_t src0BlockStride,
    uint8_t src1BlockStride, uint8_t dstRepeatStride, uint8_t src0RepeatStride,
    uint8_t src1RepeatStride, bool repeatStrideMode, bool strideSizeMode) {
  uint64_t config = 0;
  config = (((uint64_t)repeat & 0xff) << 56 |
            ((uint64_t)dstBlockStride & 0xff) << 0 |
            ((uint64_t)src0BlockStride & 0xff) << 8 |
            ((uint64_t)src1BlockStride & 0xff) << 16 |
            ((uint64_t)dstRepeatStride & 0xff) << 24 |
            ((uint64_t)src0RepeatStride & 0xff) << 32 |
            ((uint64_t)src1RepeatStride & 0xff) << 40 |
            ((uint64_t)repeatStrideMode & 0x1) << 54 |
            ((uint64_t)strideSizeMode & 0x1) << 55);
  VBS32_V300_f32(dst, src0, src1, config);
}
#endif

/*-------------------------------CLEAR_OVERFLOW_STATUS------------------------------------*/
CCE_INTRINSIC[aicore] void clear_overflow_status() {
  uint64_t a = fake_overflow_status_1();
  uint64_t b = fake_overflow_status_2();
  CLEAR_OVERFLOW_STATUS(a, b);
}

/*-------------------------------MMAD------------------------------------*/
// This interface was used to add fbuf addr to Xd[63:32].
// In v220 for mmad instruction.
// If Xt[62]=1, the C matrix is in bias table(BT) buffer,
// whose address in the BT buffer is configured in Xd[63:32].
#define CCE_INTRINSIC_MMAD_FBUF_ADDR(CCTYPE, CATYPE, CBTYPE)                   \
  CCE_INTRINSIC[aicore] void mad(__cc__ CCTYPE *c, __ca__ CATYPE *a,           \
                                 __cb__ CBTYPE *b, uint64_t addr,              \
                                 uint64_t config) {                            \
    mad((__cc__ CCTYPE *)(((uint64_t)c) & 0xffffffffULL |                      \
                          (((uint64_t)addr & 0xffffffffULL) << 32)),           \
        a, b, config);                                                         \
  }

#define CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG(CCTYPE, CATYPE, CBTYPE)               \
  CCE_INTRINSIC[aicore] void mad(                                              \
      __cc__ CCTYPE *c, __ca__ CATYPE *a, __cb__ CBTYPE *b, uint64_t addr,     \
      uint16_t m, uint16_t k, uint16_t n, uint8_t unitFlag,                    \
      bool kDirectionAlign, bool cmatrixSource, bool cmatrixInitVal) {         \
    if (cmatrixSource == 1) {                                                  \
      mad((__cc__ CCTYPE *)(((uint64_t)c) & 0xffffffffULL |                    \
                            (((uint64_t)addr & 0xffffffffULL) << 32)),         \
          a, b, m, k, n, unitFlag, kDirectionAlign, cmatrixSource,             \
          cmatrixInitVal);                                                     \
    } else {                                                                   \
      mad(c, a, b, m, k, n, unitFlag, kDirectionAlign, cmatrixSource,          \
          cmatrixInitVal);                                                     \
    }                                                                          \
  }
#define CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_LONG_PARAS(CCTYPE, CATYPE, CBTYPE)    \
  CCE_INTRINSIC[aicore] void mad(                                              \
      __cc__ CCTYPE *c, __ca__ CATYPE *a, __cb__ CBTYPE *b, uint64_t addr,     \
      uint16_t m, uint16_t k, uint16_t n, uint8_t featOffset,                  \
      uint8_t smaskOffset, uint8_t unitFlag, bool kDirectionAlign,             \
      bool isWeightOffset, bool cmatrixSource, bool cmatrixInitVal) {          \
    if (cmatrixSource == 1) {                                                  \
      mad((__cc__ CCTYPE *)(((uint64_t)c) & 0xffffffffULL |                    \
                            (((uint64_t)addr & 0xffffffffULL) << 32)),         \
          a, b, m, k, n, featOffset, smaskOffset, unitFlag, kDirectionAlign,   \
          isWeightOffset, cmatrixSource, cmatrixInitVal);                      \
    } else {                                                                   \
      mad(c, a, b, m, k, n, featOffset, smaskOffset, unitFlag,                 \
          kDirectionAlign, isWeightOffset, cmatrixSource, cmatrixInitVal);     \
    }                                                                          \
  }
#define CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_L300(CCTYPE, CATYPE, CBTYPE)          \
  CCE_INTRINSIC[aicore] void mad(                                              \
      __cc__ CCTYPE *c, __ca__ CATYPE *a, __cb__ CBTYPE *b, uint64_t addr,     \
      uint16_t m, uint16_t k, uint16_t n, uint8_t unitFlag,                    \
      bool isWeightOffset, bool cmatrixSource, bool cmatrixInitVal) {          \
    if (cmatrixSource == 1) {                                                  \
      mad((__cc__ CCTYPE *)(((uint64_t)c) & 0xffffffffULL |                    \
                            (((uint64_t)addr & 0xffffffffULL) << 32)),         \
          a, b, m, k, n, unitFlag, isWeightOffset, cmatrixSource,              \
          cmatrixInitVal);                                                     \
    } else {                                                                   \
      mad(c, a, b, m, k, n, unitFlag, isWeightOffset, cmatrixSource,           \
          cmatrixInitVal);                                                     \
    }                                                                          \
  }
#define CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_L300_OTHER(TYPE, CCTYPE, CATYPE,      \
                                                    CBTYPE)                    \
  CCE_INTRINSIC[aicore] void mad_##TYPE(                                       \
      __cc__ CCTYPE *c, __ca__ CATYPE *a, __cb__ CBTYPE *b, uint64_t addr,     \
      uint16_t m, uint16_t k, uint16_t n, uint8_t unitFlag,                    \
      bool isWeightOffset, bool cmatrixSource, bool cmatrixInitVal) {          \
    if (cmatrixSource == 1) {                                                  \
      mad_##TYPE((__cc__ CCTYPE *)(((uint64_t)c) & 0xffffffffULL |             \
                                   (((uint64_t)addr & 0xffffffffULL) << 32)),  \
                 a, b, m, k, n, unitFlag, isWeightOffset, cmatrixSource,       \
                 cmatrixInitVal);                                              \
    } else {                                                                   \
      mad_##TYPE(c, a, b, m, k, n, unitFlag, isWeightOffset, cmatrixSource,    \
                 cmatrixInitVal);                                              \
    }                                                                          \
  }
#define CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_I8_L300(CCTYPE, CATYPE, CBTYPE)       \
  CCE_INTRINSIC[aicore] void mad(                                              \
      __cc__ CCTYPE *c, __ca__ CATYPE *a, __cb__ CBTYPE *b, uint64_t addr,     \
      uint16_t m, uint16_t k, uint16_t n, uint8_t featOffset,                  \
      uint8_t smaskOffset, uint8_t unitFlag, bool isWeightOffset,              \
      bool cmatrixSource, bool cmatrixInitVal) {                               \
    if (cmatrixSource == 1) {                                                  \
      mad((__cc__ CCTYPE *)(((uint64_t)c) & 0xffffffffULL |                    \
                            (((uint64_t)addr & 0xffffffffULL) << 32)),         \
          a, b, m, k, n, featOffset, smaskOffset, unitFlag, isWeightOffset,    \
          cmatrixSource, cmatrixInitVal);                                      \
    } else {                                                                   \
      mad(c, a, b, m, k, n, featOffset, smaskOffset, unitFlag, isWeightOffset, \
          cmatrixSource, cmatrixInitVal);                                      \
    }                                                                          \
  }
#define CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_S8S16_L300(CCTYPE, CATYPE, CBTYPE)    \
  CCE_INTRINSIC[aicore] void mad(                                              \
      __cc__ CCTYPE *c, __ca__ CATYPE *a, __cb__ CBTYPE *b, uint64_t addr,     \
      uint16_t m, uint16_t k, uint16_t n, uint8_t unitFlag, bool rightShift,   \
      bool bmatrixSource, bool isWeightOffset, bool cmatrixSource,             \
      bool cmatrixInitVal) {                                                   \
    if (cmatrixSource == 1) {                                                  \
      mad((__cc__ CCTYPE *)(((uint64_t)c) & 0xffffffffULL |                    \
                            (((uint64_t)addr & 0xffffffffULL) << 32)),         \
          a, b, m, k, n, unitFlag, rightShift, bmatrixSource, isWeightOffset,  \
          cmatrixSource, cmatrixInitVal);                                      \
    } else {                                                                   \
      mad(c, a, b, m, k, n, unitFlag, rightShift, bmatrixSource,               \
          isWeightOffset, cmatrixSource, cmatrixInitVal);                      \
    }                                                                          \
  }
// For type s4 and tf322f32
#define CCE_INTRINSIC_MMAD_FBUF_ADDR_OTHER(TYPE, CCTYPE, CATYPE, CBTYPE)       \
  CCE_INTRINSIC[aicore] void mad_##TYPE(__cc__ CCTYPE *c, __ca__ CATYPE *a,    \
                                        __cb__ CBTYPE *b, uint64_t addr,       \
                                        uint64_t config) {                     \
    mad_##TYPE((__cc__ CCTYPE *)(((uint64_t)c) & 0xffffffffULL |               \
                                 (((uint64_t)addr & 0xffffffffULL) << 32)),    \
               a, b, config);                                                  \
  }
#define CCE_INTRINSIC_MMAD_FBUF_ADDR_OTHER_CFG(TYPE, CCTYPE, CATYPE, CBTYPE)   \
  CCE_INTRINSIC[aicore] void mad_##TYPE(                                       \
      __cc__ CCTYPE *c, __ca__ CATYPE *a, __cb__ CBTYPE *b, uint64_t addr,     \
      uint16_t m, uint16_t k, uint16_t n, uint8_t unitFlag,                    \
      bool kDirectionAlign, bool cmatrixSource, bool cmatrixInitVal) {         \
    if (cmatrixSource == 1) {                                                  \
      mad_##TYPE((__cc__ CCTYPE *)(((uint64_t)c) & 0xffffffffULL |             \
                                   (((uint64_t)addr & 0xffffffffULL) << 32)),  \
                 a, b, m, k, n, unitFlag, kDirectionAlign, cmatrixSource,      \
                 cmatrixInitVal);                                              \
    } else {                                                                   \
      mad_##TYPE(c, a, b, m, k, n, unitFlag, kDirectionAlign, cmatrixSource,   \
                 cmatrixInitVal);                                              \
    }                                                                          \
  }
#define CCE_INTRINSIC_MMAD_FBUF_ADDR_OTHER_CFG_LONG_PARAS(TYPE, CCTYPE,        \
                                                          CATYPE, CBTYPE)      \
  CCE_INTRINSIC[aicore] void mad_##TYPE(                                       \
      __cc__ CCTYPE *c, __ca__ CATYPE *a, __cb__ CBTYPE *b, uint64_t addr,     \
      uint16_t m, uint16_t k, uint16_t n, uint8_t featOffset,                  \
      uint8_t smaskOffset, uint8_t unitFlag, bool kDirectionAlign,             \
      bool isWeightOffset, bool cmatrixSource, bool cmatrixInitVal) {          \
    if (cmatrixSource == 1) {                                                  \
      mad_##TYPE((__cc__ CCTYPE *)(((uint64_t)c) & 0xffffffffULL |             \
                                   (((uint64_t)addr & 0xffffffffULL) << 32)),  \
                 a, b, m, k, n, featOffset, smaskOffset, unitFlag,             \
                 kDirectionAlign, isWeightOffset, cmatrixSource,               \
                 cmatrixInitVal);                                              \
    } else {                                                                   \
      mad_##TYPE(c, a, b, m, k, n, featOffset, smaskOffset, unitFlag,          \
                 kDirectionAlign, isWeightOffset, cmatrixSource,               \
                 cmatrixInitVal);                                              \
    }                                                                          \
  }
#define CCE_INTRINSIC_MMAD_SP_FBUF_ADDR(CCTYPE, CATYPE, CBTYPE)                \
  CCE_INTRINSIC[aicore] void mad_sp(__cc__ CCTYPE *c, __ca__ CATYPE *a,        \
                                    __cb__ CBTYPE *b, uint64_t addr,           \
                                    uint64_t config) {                         \
    mad_sp((__cc__ CCTYPE *)(((uint64_t)c) & 0xffffffffULL |                   \
                             (((uint64_t)addr & 0xffffffffULL) << 32)),        \
           a, b, config);                                                      \
  }
#define CCE_INTRINSIC_MMAD_SP_FBUF_ADDR_CFG(CCTYPE, CATYPE, CBTYPE)            \
  CCE_INTRINSIC[aicore] void mad_sp(                                           \
      __cc__ CCTYPE *c, __ca__ CATYPE *a, __cb__ CBTYPE *b, uint64_t addr,     \
      uint16_t m, uint16_t k, uint16_t n, uint8_t unitFlag,                    \
      bool cmatrixSource, bool cmatrixInitVal) {                               \
    if (cmatrixSource == 1) {                                                  \
      mad_sp((__cc__ CCTYPE *)(((uint64_t)c) & 0xffffffffULL |                 \
                               (((uint64_t)addr & 0xffffffffULL) << 32)),      \
             a, b, m, k, n, unitFlag, cmatrixSource, cmatrixInitVal);          \
    } else {                                                                   \
      mad_sp(c, a, b, m, k, n, unitFlag, cmatrixSource, cmatrixInitVal);       \
    }                                                                          \
  }

#define CCE_INTRINSIC_MMAD_PARA_GEMV(FUNCNAME, CCTYPE, CATYPE, CBTYPE)         \
  CCE_INTRINSIC[aicore] void FUNCNAME(                                         \
      __cc__ CCTYPE *c, __ca__ CATYPE *a, __cb__ CBTYPE *b, uint16_t m,        \
      uint16_t k, uint16_t n, uint8_t unitFlag, bool gemvDisable,              \
      bool cmatrixSource, bool cmatrixInitVal) {                               \
    FUNCNAME##_inner(c, a, b, m, k, n, unitFlag, gemvDisable, cmatrixSource,   \
                     cmatrixInitVal);                                          \
  }

#define CCE_INTRINSIC_MMAD_PARA_GEMV_2(FUNCNAME, CCTYPE, CATYPE, CBTYPE)       \
  CCE_INTRINSIC[aicore] void FUNCNAME(                                         \
      __cc__ CCTYPE *c, __ca__ CATYPE *a, __cb__ CBTYPE *b, uint16_t m,        \
      uint16_t k, uint16_t n, uint8_t smaskOffset, uint8_t unitFlag,           \
      bool gemvDisable, bool cmatrixSource, bool cmatrixInitVal) {             \
    FUNCNAME##_inner(c, a, b, m, k, n, smaskOffset, unitFlag, gemvDisable,     \
                     cmatrixSource, cmatrixInitVal);                           \
  }

#define CCE_INTRINSIC_MMAD_FBUF_ADDR_PARA_GEMV(FUNCNAME, CCTYPE, CATYPE,       \
                                               CBTYPE)                         \
  CCE_INTRINSIC[aicore] void FUNCNAME(                                         \
      __cc__ CCTYPE *c, __ca__ CATYPE *a, __cb__ CBTYPE *b, uint64_t addr,     \
      uint16_t m, uint16_t k, uint16_t n, uint8_t unitFlag, bool gemvDisable,  \
      bool cmatrixSource, bool cmatrixInitVal) {                               \
    if (cmatrixSource == 1) {                                                  \
      FUNCNAME##_inner(                                                        \
          (__cc__ CCTYPE *)(((uint64_t)c) & 0xffffffffULL |                    \
                            (((uint64_t)addr & 0xffffffffULL) << 32)),         \
          a, b, m, k, n, unitFlag, gemvDisable, cmatrixSource,                 \
          cmatrixInitVal);                                                     \
    } else {                                                                   \
      FUNCNAME##_inner(c, a, b, m, k, n, unitFlag, gemvDisable, cmatrixSource, \
                       cmatrixInitVal);                                        \
    }                                                                          \
  }

#define CCE_INTRINSIC_MMAD_FBUF_ADDR_PARA_GEMV_2(FUNCNAME, CCTYPE, CATYPE,     \
                                                 CBTYPE)                       \
  CCE_INTRINSIC[aicore] void FUNCNAME(                                         \
      __cc__ CCTYPE *c, __ca__ CATYPE *a, __cb__ CBTYPE *b, uint64_t addr,     \
      uint16_t m, uint16_t k, uint16_t n, uint8_t smaskOffset,                 \
      uint8_t unitFlag, bool gemvDisable, bool cmatrixSource,                  \
      bool cmatrixInitVal) {                                                   \
    if (cmatrixSource == 1) {                                                  \
      FUNCNAME##_inner(                                                        \
          (__cc__ CCTYPE *)(((uint64_t)c) & 0xffffffffULL |                    \
                            (((uint64_t)addr & 0xffffffffULL) << 32)),         \
          a, b, m, k, n, smaskOffset, unitFlag, gemvDisable, cmatrixSource,    \
          cmatrixInitVal);                                                     \
    } else {                                                                   \
      FUNCNAME##_inner(c, a, b, m, k, n, smaskOffset, unitFlag, gemvDisable,   \
                       cmatrixSource, cmatrixInitVal);                         \
    }                                                                          \
  }

#if (defined __DAV_L300__) || (defined __DAV_L300_VEC__) ||                    \
    (defined __DAV_L310__)
CCE_INTRINSIC_MMAD_FBUF_ADDR(uint32_t, uint8_t, uint8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_I8_L300(uint32_t, uint8_t, uint8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR(int32_t, uint8_t, uint8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_I8_L300(int32_t, uint8_t, uint8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR(int32_t, int8_t, int8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_I8_L300(int32_t, int8_t, int8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR(int32_t, uint8_t, int8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_I8_L300(int32_t, uint8_t, int8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR(int32_t, int16_t, int8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_S8S16_L300(int32_t, int16_t, int8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR(half, half, half);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_L300(half, half, half);
CCE_INTRINSIC_MMAD_FBUF_ADDR_OTHER(s4, int32_t, void, void);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_L300_OTHER(s4, int32_t, void, void);
CCE_INTRINSIC_MMAD_FBUF_ADDR_OTHER(s8s4, int32_t, int8_t, void);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_L300_OTHER(s8s4, int32_t, int8_t, void);
CCE_INTRINSIC_MMAD_SP_FBUF_ADDR(int32_t, int8_t, int8_t);
CCE_INTRINSIC_MMAD_SP_FBUF_ADDR_CFG(int32_t, int8_t, int8_t);
#elif (defined __DAV_M300__) || (defined __DAV_C310__)
CCE_INTRINSIC_MMAD_FBUF_ADDR(float, bfloat16_t, bfloat16_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG(float, bfloat16_t, bfloat16_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_LONG_PARAS(float, bfloat16_t, bfloat16_t);
#elif (__CCE_AICORE__ == 220)
CCE_INTRINSIC_MMAD_FBUF_ADDR(float, bfloat16_t, bfloat16_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG(float, bfloat16_t, bfloat16_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_LONG_PARAS(float, bfloat16_t, bfloat16_t);
#endif

#if (defined __DAV_C310__)
CCE_INTRINSIC_MMAD_FBUF_ADDR(float, hifloat8_t, hifloat8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG(float, hifloat8_t, hifloat8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_LONG_PARAS(float, hifloat8_t, hifloat8_t);
#endif

#if (defined __DAV_M300__) || (__CCE_AICORE__ == 220) || (defined __DAV_C310__)
CCE_INTRINSIC_MMAD_FBUF_ADDR(half, half, half);
CCE_INTRINSIC_MMAD_FBUF_ADDR(float, half, half);
CCE_INTRINSIC_MMAD_FBUF_ADDR(float, float, float);
CCE_INTRINSIC_MMAD_FBUF_ADDR(int32_t, int8_t, int8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR(uint32_t, uint8_t, uint8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR(int32_t, uint8_t, uint8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR(int32_t, uint8_t, int8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG(half, half, half);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG(float, half, half);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG(float, float, float);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG(int32_t, int8_t, int8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG(uint32_t, uint8_t, uint8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG(int32_t, uint8_t, uint8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG(int32_t, uint8_t, int8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_LONG_PARAS(half, half, half);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_LONG_PARAS(float, half, half);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_LONG_PARAS(float, float, float);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_LONG_PARAS(int32_t, int8_t, int8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_LONG_PARAS(uint32_t, uint8_t, uint8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_LONG_PARAS(int32_t, uint8_t, uint8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_LONG_PARAS(int32_t, uint8_t, int8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR_OTHER(s4, int32_t, void, void);
CCE_INTRINSIC_MMAD_FBUF_ADDR_OTHER(tf322f32, float, float, float);
CCE_INTRINSIC_MMAD_FBUF_ADDR_OTHER_CFG(s4, int32_t, void, void);
CCE_INTRINSIC_MMAD_FBUF_ADDR_OTHER_CFG(tf322f32, float, float, float);
CCE_INTRINSIC_MMAD_FBUF_ADDR_OTHER_CFG_LONG_PARAS(s4, int32_t, void, void);
CCE_INTRINSIC_MMAD_FBUF_ADDR_OTHER_CFG_LONG_PARAS(tf322f32, float, float,
                                                  float);
CCE_INTRINSIC_MMAD_SP_FBUF_ADDR(int32_t, int8_t, int8_t);
CCE_INTRINSIC_MMAD_SP_FBUF_ADDR_CFG(int32_t, int8_t, int8_t);
#endif

#if (defined __DAV_M310__)
CCE_INTRINSIC_MMAD_FBUF_ADDR(float, half, half);
CCE_INTRINSIC_MMAD_FBUF_ADDR(int32_t, int8_t, int8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR_OTHER(s4, int32_t, void, void);

CCE_INTRINSIC_MMAD_FBUF_ADDR_PARA_GEMV(mad, float, half, half);
CCE_INTRINSIC_MMAD_FBUF_ADDR_PARA_GEMV(mad, int32_t, int8_t, int8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR_PARA_GEMV(mad_s4, int32_t, void, void);

CCE_INTRINSIC_MMAD_FBUF_ADDR_PARA_GEMV_2(mad, float, half, half);
CCE_INTRINSIC_MMAD_FBUF_ADDR_PARA_GEMV_2(mad, int32_t, int8_t, int8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR_PARA_GEMV_2(mad_s4, int32_t, void, void);

CCE_INTRINSIC_MMAD_PARA_GEMV(mad, float, half, half);
CCE_INTRINSIC_MMAD_PARA_GEMV(mad, int32_t, int8_t, int8_t);
CCE_INTRINSIC_MMAD_PARA_GEMV(mad_s4, int32_t, void, void);

CCE_INTRINSIC_MMAD_PARA_GEMV_2(mad, float, half, half);
CCE_INTRINSIC_MMAD_PARA_GEMV_2(mad, int32_t, int8_t, int8_t);
CCE_INTRINSIC_MMAD_PARA_GEMV_2(mad_s4, int32_t, void, void);

CCE_INTRINSIC_MMAD_SP_FBUF_ADDR(int32_t, int8_t, int8_t);
CCE_INTRINSIC_MMAD_SP_FBUF_ADDR_CFG(int32_t, int8_t, int8_t);
#endif // __DAV_M310__

#if (defined __DAV_T300__)
CCE_INTRINSIC_MMAD_FBUF_ADDR(half, half, half);
CCE_INTRINSIC_MMAD_FBUF_ADDR(float, half, half);
CCE_INTRINSIC_MMAD_FBUF_ADDR(int32_t, int8_t, int8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR(int32_t, uint8_t, uint8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR(int32_t, uint8_t, int8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG(half, half, half);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG(float, half, half);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG(int32_t, int8_t, int8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG(int32_t, uint8_t, uint8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG(int32_t, uint8_t, int8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_LONG_PARAS(half, half, half);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_LONG_PARAS(float, half, half);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_LONG_PARAS(int32_t, int8_t, int8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_LONG_PARAS(int32_t, uint8_t, uint8_t);
CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_LONG_PARAS(int32_t, uint8_t, int8_t);
#endif

#undef CCE_INTRINSIC_MMAD_FBUF_ADDR
#undef CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG
#undef CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_LONG_PARAS
#undef CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_L300
#undef CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_L300_OTHER
#undef CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_I8_L300
#undef CCE_INTRINSIC_MMAD_FBUF_ADDR_CFG_S8S16_L300
#undef CCE_INTRINSIC_MMAD_FBUF_ADDR_OTHER
#undef CCE_INTRINSIC_MMAD_FBUF_ADDR_OTHER_CFG
#undef CCE_INTRINSIC_MMAD_FBUF_ADDR_OTHER_CFG_LONG_PARAS
#undef CCE_INTRINSIC_MMAD_SP_FBUF_ADDR
#undef CCE_INTRINSIC_MMAD_SP_FBUF_ADDR_CFG
#undef CCE_INTRINSIC_MMAD_PARA_GEMV
#undef CCE_INTRINSIC_MMAD_PARA_GEMV_2
#undef CCE_INTRINSIC_MMAD_FBUF_ADDR_PARA_GEMV
#undef CCE_INTRINSIC_MMAD_FBUF_ADDR_PARA_GEMV_2

/*-----------------------GET_UB_VIRTUAL_ADDR---------------------*/
#ifdef __CCE_NEED_ADDR_TRANS
CCE_GET_VA_INTRINSIC[aicore] __ubuf__ int8_t *
get_ub_virtual_address(__ubuf__ int8_t *addr) {
  return addr;
}
CCE_GET_VA_INTRINSIC[aicore] __ubuf__ uint8_t *
get_ub_virtual_address(__ubuf__ uint8_t *addr) {
  return addr;
}
CCE_GET_VA_INTRINSIC[aicore] __ubuf__ int16_t *
get_ub_virtual_address(__ubuf__ int16_t *addr) {
  return addr;
}
CCE_GET_VA_INTRINSIC[aicore] __ubuf__ uint16_t *
get_ub_virtual_address(__ubuf__ uint16_t *addr) {
  return addr;
}
CCE_GET_VA_INTRINSIC[aicore] __ubuf__ half *
get_ub_virtual_address(__ubuf__ half *addr) {
  return addr;
}
CCE_GET_VA_INTRINSIC[aicore] __ubuf__ int32_t *
get_ub_virtual_address(__ubuf__ int32_t *addr) {
  return addr;
}
CCE_GET_VA_INTRINSIC[aicore] __ubuf__ uint32_t *
get_ub_virtual_address(__ubuf__ uint32_t *addr) {
  return addr;
}
CCE_GET_VA_INTRINSIC[aicore] __ubuf__ float *
get_ub_virtual_address(__ubuf__ float *addr) {
  return addr;
}
CCE_GET_VA_INTRINSIC[aicore] uint64_t get_ub_virtual_address(uint64_t addr) {
  return addr;
}
#else
CCE_INTRINSIC[aicore] __ubuf__ int8_t *
get_ub_virtual_address(__ubuf__ int8_t *addr) {
  return (__ubuf__ int8_t *)(get_sys_va_base() + 0x80000 + (uint64_t)addr);
}
CCE_INTRINSIC[aicore] __ubuf__ uint8_t *
get_ub_virtual_address(__ubuf__ uint8_t *addr) {
  return (__ubuf__ uint8_t *)(get_sys_va_base() + 0x80000 + (uint64_t)addr);
}
CCE_INTRINSIC[aicore] __ubuf__ int16_t *
get_ub_virtual_address(__ubuf__ int16_t *addr) {
  return (__ubuf__ int16_t *)(get_sys_va_base() + 0x80000 + (uint64_t)addr);
}
CCE_INTRINSIC[aicore] __ubuf__ uint16_t *
get_ub_virtual_address(__ubuf__ uint16_t *addr) {
  return (__ubuf__ uint16_t *)(get_sys_va_base() + 0x80000 + (uint64_t)addr);
}
CCE_INTRINSIC[aicore] __ubuf__ half *
get_ub_virtual_address(__ubuf__ half *addr) {
  return (__ubuf__ half *)(get_sys_va_base() + 0x80000 + (uint64_t)addr);
}
CCE_INTRINSIC[aicore] __ubuf__ int32_t *
get_ub_virtual_address(__ubuf__ int32_t *addr) {
  return (__ubuf__ int32_t *)(get_sys_va_base() + 0x80000 + (uint64_t)addr);
}
CCE_INTRINSIC[aicore] __ubuf__ uint32_t *
get_ub_virtual_address(__ubuf__ uint32_t *addr) {
  return (__ubuf__ uint32_t *)(get_sys_va_base() + 0x80000 + (uint64_t)addr);
}
CCE_INTRINSIC[aicore] __ubuf__ float *
get_ub_virtual_address(__ubuf__ float *addr) {
  return (__ubuf__ float *)(get_sys_va_base() + 0x80000 + (uint64_t)addr);
}
CCE_INTRINSIC[aicore] uint64_t get_ub_virtual_address(uint64_t addr) {
  return get_sys_va_base() + 0x80000 + addr;
}
#endif

/*-------------------------------ATOMIC------------------------------------*/
// For atomic store intrinsics
#if __CCE_AICORE__ == 220
template <typename T>
CCE_INTRINSIC[aicore] void st_atomic(int64_t value, __gm__ T *addr) {
  __atomic_store_n(addr, value, __ATOMIC_RELAXED);
}
CCE_INTRINSIC[aicore] void st_atomic(float value, __gm__ float *addr) {
  typedef union {
    int32_t i;
    float f;
  } val;
  val tmp;
  tmp.f = value;
  __atomic_store_n(reinterpret_cast<__gm__ int32_t *>(addr), tmp.i,
                   __ATOMIC_RELAXED);
}
CCE_INTRINSIC[aicore] void st_atomic(half value, __gm__ half *addr) {
  typedef union {
    int16_t i;
    half f;
  } val;
  val tmp;
  tmp.f = value;
  __atomic_store_n(reinterpret_cast<__gm__ int16_t *>(addr), tmp.i,
                   __ATOMIC_RELAXED);
}
#endif

namespace bisheng {
namespace cce {
#if (defined __DAV_C220_VEC__) || (defined __DAV_C220_CUBE__) ||               \
    (defined __DAV_C310__)
CCE_INTRINSIC[aicore] void metrics_prof_start() {
  pipe_barrier(PIPE_ALL);
  set_ctrl(sbitset1(get_ctrl(), 0));
  pipe_barrier(PIPE_ALL);
}

CCE_INTRINSIC[aicore] void metrics_prof_stop() {
  pipe_barrier(PIPE_ALL);
  set_ctrl(sbitset0(get_ctrl(), 0));
  pipe_barrier(PIPE_ALL);
}
#endif
} // namespace cce
} // namespace bisheng

#endif // ifdef __CCE_AICORE__

#undef CCE_AICORE_INLINE
#endif // #ifndef __CLANG_CCE_AICORE_FUNCTIONS_H__
