/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file kernel_operator_vec_vconv_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_VCONV_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_VCONV_IMPL_H
#include "kernel_utils.h"
#include "kernel_operator.h"
#include "kernel_struct_binary.h"
#include "kernel_struct_unary.h"
#include "kernel_struct_vdeq.h"

namespace AscendC {
#define VCVT_U8_TO_F16(round_mode) vcvt(vreg1, vreg0, preg, PART_EVEN)
#define VCVT_S8_TO_F16(round_mode) vcvt(vreg1, vreg0, preg, PART_EVEN)
#define VCVT_F16_TO_F32(round_mode) vcvt(vreg1, vreg0, preg, PART_EVEN)
#define VCVT_F16_TO_S32(round_mode) vcvt(vreg1, vreg0, preg, round_mode, PART_EVEN)
#define VCVT_S16_TO_F32(round_mode) vcvt(vreg1, vreg0, preg, PART_EVEN)

#define VCVT_F16_TO_U8(round_mode) vcvt(vreg1, vreg0, preg, round_mode, RS_ENABLE, PART_EVEN)
#define VCVT_F16_TO_S8(round_mode) vcvt(vreg1, vreg0, preg, round_mode, RS_ENABLE, PART_EVEN)
#define VCVT_F32_TO_F16(round_mode) vcvt(vreg1, vreg0, preg, round_mode, RS_ENABLE, PART_EVEN)
#define VCVT_F32_TO_S16(round_mode) vcvt(vreg1, vreg0, preg, round_mode, RS_ENABLE, PART_EVEN)
#define VCVT_S32_TO_F16(round_mode)                                     \
    vector_f32 vregTmpF32;                                              \
    vcvt(vregTmpF32, vreg0, preg, round_mode);                          \
    vcvt(vreg1, vregTmpF32, preg, round_mode, RS_ENABLE, PART_EVEN)
#define VCVT_S32_TO_S16(round_mode) vcvt(vreg1, vreg0, preg, RS_ENABLE, PART_EVEN)

#define VCVT_F16_TO_S16(round_mode) vcvt(vreg1, vreg0, preg, round_mode, RS_ENABLE)
#define VCVT_S16_TO_F16(round_mode) vcvt(vreg1, vreg0, preg, round_mode)
#define VCVT_F32_TO_F32(round_mode) vtrc(vreg1, vreg0, round_mode, preg)
#define VCVT_F32_TO_S32(round_mode) vcvt(vreg1, vreg0, preg, round_mode, RS_ENABLE)
#define VCVT_S32_TO_F32(round_mode) vcvt(vreg1, vreg0, preg, round_mode)

#define GEN_PLT_INSTR_B8(preg, sreg) preg = plt_b8(sreg, POST_UPDATE)
#define GEN_PLT_INSTR_B16(preg, sreg) preg = plt_b16(sreg, POST_UPDATE)
#define GEN_PLT_INSTR_B32(preg, sreg) preg = plt_b32(sreg, POST_UPDATE)

#define GEN_VSTS_INSTR_B8(vreg, base, offset, dist, preg) vsts(vreg, base, offset, dist, preg)
#define GEN_VSTS_INSTR_B16(vreg, base, offset, dist, preg) vsts(vreg, base, offset, dist, preg)
#define GEN_VSTS_INSTR_B32(vreg, base, offset, dist, preg) vsts(vreg, base, offset, dist, preg)

// deal 128 elements each repeat. (b8->b16 / b16->b8 / b16->b16, depends on the larger data type.)
#define LV2_INIT_128(repeat_size)                                                                           \
    uint32_t sregLower = (uint32_t)repeat_size;                                                             \
    uint32_t sregUpper = (uint32_t)repeat_size

// deal 64 elements each repeat. (b16->b32 / b32->b16 / b32->b32, depends on the larger data type.)
#define LV2_INIT_64(repeat_size)                                                                            \
    uint32_t sregLower = (uint32_t)repeat_size;                                                             \
    uint32_t sregUpper = (uint32_t)repeat_size

#define LV2_LOAD_UPPER(src_bits, dst_bits)                                                                  \
    GEN_PLT_INSTR_B##dst_bits(preg, sregPlt);                                                               \
    vlds(vreg0, src, i * sregLower, UNPK_B##src_bits)

#define LV2_STORE_UPPER(src_bits, dst_bits)                                                                 \
    GEN_VSTS_INSTR_B##dst_bits(vreg1, dst, i * sregUpper, NORM_B##dst_bits, preg)

#define LV2_LOAD_LOWER(src_bits, dst_bits)                                                                  \
    GEN_PLT_INSTR_B##src_bits(preg, sregPlt);                                                               \
    vlds(vreg0, src, i * sregLower, NORM)

#define LV2_STORE_LOWER(src_bits, dst_bits)                                                                 \
    GEN_VSTS_INSTR_B##dst_bits(vreg1, dst, i * sregLower, PK_B##src_bits, preg)

#define LV2_LOAD_EQUAL(src_bits, dst_bits)                                                                  \
    GEN_PLT_INSTR_B##dst_bits(preg, sregPlt);                                                               \
    vlds(vreg0, src, i * sregLower, NORM)

#define LV2_STORE_EQUAL(src_bits, dst_bits)                                                                 \
    GEN_VSTS_INSTR_B##dst_bits(vreg1, dst, i * sregLower, NORM_B##dst_bits, preg)

// Cast::Level 2
#define REGISTER_CAST_LV2(src_type, dst_type, src_type_short, dst_type_short, src_bits, dst_bits,           \
    repeat_size, round_str, round_mode, load_func, cast_func, store_func)                                   \
    __aicore__ inline void CastIntrinsicsImpl##round_str(__ubuf__ dst_type* dst, __ubuf__ src_type* src,    \
        const uint32_t calCount)                                                                            \
    {                                                                                                       \
        __VEC_SCOPE__                                                                                       \
        {                                                                                                   \
            vector_##src_type_short vreg0;                                                                  \
            vector_##dst_type_short vreg1;                                                                  \
            uint32_t sregPlt = (uint32_t)calCount;                                                          \
            LV2_INIT_##repeat_size(repeat_size);                                                            \
            vector_bool preg;                                                                               \
            uint16_t repeatTimes = CeilDivision(calCount, repeat_size);                                     \
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {                                          \
                load_func(src_bits, dst_bits);                                                              \
                cast_func(round_mode);                                                                      \
                store_func(src_bits, dst_bits);                                                             \
            }                                                                                               \
        }                                                                                                   \
    }

#define GEN_VSSTB_INSTR_B8(vreg, base, offset, config, preg) vsstb(vreg, base + offset, config, preg)
#define GEN_VSSTB_INSTR_B16(vreg, base, offset, config, preg) vsstb(vreg, base + offset, config, preg)
#define GEN_VSSTB_INSTR_B32(vreg, base, offset, config, preg) vsstb(vreg, base + offset, config, preg)

// deal 128 elements each repeat. (b8->b16 / b16->b8 / b16->b16, depends on the larger data type.)
#define BIT_INIT_128(preg_lower, preg_upper)                                                               \
    plds(preg_upper, ((__ubuf__ uint32_t*)maskBuf), 0, NORM);                                              \
    punpack(preg_lower, preg_upper, LOWER)

// deal 64 elements each repeat. (b16->b32 / b32->b16 / b32->b32, depends on the larger data type.)
#define BIT_INIT_64(preg_lower, preg_upper)                                                                 \
    plds(preg_upper, ((__ubuf__ uint32_t*)maskBuf), 0, US);                                                 \
    punpack(preg_lower, preg_upper, LOWER)

// deal 128 elements each repeat. (b8->b16 / b16->b8 / b16->b16, depends on the larger data type.)
#define COUNT_INIT_128(preg_lower, preg_upper, bits_lower, bits_upper)                                      \
    uint32_t sregLower = (uint32_t)mask;                                                                    \
    uint32_t sregUpper = (uint32_t)mask;                                                                    \
    GEN_PLT_INSTR_B##bits_lower(preg_lower, sregLower);                                                     \
    GEN_PLT_INSTR_B##bits_upper(preg_upper, sregUpper)

// deal 64 elements each repeat. (b16->b32 / b32->b16 / b32->b32, depends on the larger data type.)
#define COUNT_INIT_64(preg_lower, preg_upper, bits_lower, bits_upper)                                       \
    uint32_t sregLower = (uint32_t)mask;                                                                    \
    uint32_t sregUpper = (uint32_t)mask;                                                                    \
    GEN_PLT_INSTR_B##bits_lower(preg_lower, sregLower);                                                     \
    GEN_PLT_INSTR_B##bits_upper(preg_upper, sregUpper)

#define BIT_INIT_UPPER(src_dtype_short, dst_dtype_short, src_bits, dst_bits, repeat_size)                   \
    vector_bool pregLower;                                                                                  \
    BIT_INIT_##repeat_size(preg, pregLower);                                                                \
    vector_s##src_bits vregTmp

#define COUNTER_INIT_UPPER(src_dtype_short, dst_dtype_short, src_bits, dst_bits, repeat_size)               \
    vector_bool pregLower;                                                                                  \
    COUNT_INIT_##repeat_size(pregLower, preg, src_bits, dst_bits);                                          \
    vector_s##src_bits vregTmp

#define LV0_LOAD_UPPER(src_bits)                                                                            \
    vsldb(vreg0, src + i * strideOffset0, strideConfig0, pregLower);                                        \
    vintlv((vector_s##src_bits&)vreg0, vregTmp, (vector_s##src_bits&)vreg0, vregTmp)

#define LV0_STORE_UPPER(dst_bits)                                                                           \
    GEN_VSSTB_INSTR_B##dst_bits(vreg1, dst, i * strideOffset1, strideConfig1, preg)

#define BIT_INIT_LOWER(src_dtype_short, dst_dtype_short, src_bits, dst_bits, repeat_size)                   \
    vector_bool pregLower;                                                                                  \
    BIT_INIT_##repeat_size(preg, pregLower);                                                                \
    vector_s##dst_bits vregTmp

#define COUNTER_INIT_LOWER(src_dtype_short, dst_dtype_short, src_bits, dst_bits, repeat_size)               \
    vector_bool pregLower;                                                                                  \
    COUNT_INIT_##repeat_size(pregLower, preg, dst_bits, src_bits);                                          \
    vector_s##dst_bits vregTmp

#define LV0_LOAD_LOWER(src_bits)                                                                            \
    vsldb(vreg0, src + i * strideOffset0, strideConfig0, preg)

#define LV0_STORE_LOWER(dst_bits)                                                                           \
    vdintlv((vector_s##dst_bits&)vreg1, vregTmp, (vector_s##dst_bits&)vreg1, vregTmp);                      \
    GEN_VSSTB_INSTR_B##dst_bits(vreg1, dst, i * strideOffset1, strideConfig1, pregLower)

#define BIT_INIT_EQUAL(src_dtype_short, dst_dtype_short, src_bits, dst_bits, repeat_size)                   \
    vector_bool pregLower;                                                                                  \
    BIT_INIT_##repeat_size(preg, pregLower)

#define COUNTER_INIT_EQUAL(src_dtype_short, dst_dtype_short, src_bits, dst_bits, repeat_size)               \
    vector_bool preg1;                                                                                      \
    COUNT_INIT_##repeat_size(preg1, preg, dst_bits, src_bits)

#define LV0_LOAD_EQUAL(src_bits)                                                                            \
    vsldb(vreg0, src + i * strideOffset0, strideConfig0, preg)

#define LV0_STORE_EQUAL(dst_bits)                                                                           \
    GEN_VSSTB_INSTR_B##dst_bits(vreg1, dst, i * strideOffset1, strideConfig1, preg)

// common vf function of Cast::Level 0
#define CAST_LV0_VF(src_type_short, dst_type_short, src_bits, dst_bits,                                     \
    repeat_size, round_mode, init_func, load_func, cast_func, store_func)                                   \
    __VEC_SCOPE__                                                                                           \
    {                                                                                                       \
        vector_##src_type_short vreg0;                                                                      \
        vector_##dst_type_short vreg1;                                                                      \
        vector_bool preg;                                                                                   \
        init_func(src_type_short, dst_type_short, src_bits, dst_bits, repeat_size);                         \
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);                             \
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);                             \
        uint32_t strideOffset0 = (uint32_t)repeatParams.srcRepStride * 256 / src_bits;                      \
        uint32_t strideOffset1 = (uint32_t)repeatParams.dstRepStride * 256 / dst_bits;                      \
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {                                              \
            load_func(src_bits);                                                                            \
            cast_func(round_mode);                                                                          \
            store_func(dst_bits);                                                                           \
        }                                                                                                   \
    }

// Cast::Level 0 - mask bit mode
#define REGISTER_CAST_BIT(src_type, dst_type, src_type_short, dst_type_short, src_bits, dst_bits,           \
    repeat_size, round_str, round_mode, init_func, load_func, cast_func, store_func)                        \
    __aicore__ inline void CastIntrinsicsImpl##round_str(__ubuf__ dst_type* dst, __ubuf__ src_type* src,    \
        const uint64_t mask[], uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)                 \
    {                                                                                                       \
        __ubuf__ uint64_t* maskBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(TMP_UB_OFFSET, 4);      \
        maskBuf[0] = mask[0];                                                                               \
        maskBuf[1] = mask[1];                                                                               \
        maskBuf[2] = 0;                                                                                     \
        maskBuf[3] = 0;                                                                                     \
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));            \
        SetFlag<HardEvent::S_V>(eventIdSToV);                                                               \
        WaitFlag<HardEvent::S_V>(eventIdSToV);                                                              \
        CAST_LV0_VF(src_type_short, dst_type_short, src_bits, dst_bits, repeat_size, round_mode,            \
            init_func, load_func, cast_func, store_func);                                                   \
        AscendCUtils::FreeTemporaryBuffer<uint64_t>(maskBuf);                                               \
    }

// Cast::Level 0 - mask counter mode
#define REGISTER_CAST_COUNTER(src_type, dst_type, src_type_short, dst_type_short, src_bits, dst_bits,       \
    repeat_size, round_str, round_mode, init_func, load_func, cast_func, store_func)                        \
    __aicore__ inline void CastIntrinsicsImpl##round_str(__ubuf__ dst_type* dst, __ubuf__ src_type* src,    \
        const uint64_t mask, uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)                    \
    {                                                                                                       \
        CAST_LV0_VF(src_type_short, dst_type_short, src_bits, dst_bits, repeat_size, round_mode,            \
            init_func, load_func, cast_func, store_func);                                                   \
    }

// for data_type size: src < dst
// repeat_size: the num of elements processed in each repeat depends on the larger data type.
#define REGISTER_CAST_UPPER(cast_func, round_str, round_mode,                                               \
    src_type, dst_type, src_type_short, dst_type_short, src_bits, dst_bits, repeat_size)                    \
    REGISTER_CAST_LV2(src_type, dst_type, src_type_short, dst_type_short, src_bits, dst_bits,               \
        repeat_size, round_str, round_mode, LV2_LOAD_UPPER, cast_func, LV2_STORE_UPPER);                    \
    REGISTER_CAST_BIT(src_type, dst_type, src_type_short, dst_type_short, src_bits, dst_bits,               \
        repeat_size, round_str, round_mode, BIT_INIT_UPPER, LV0_LOAD_UPPER, cast_func, LV0_STORE_UPPER);    \
    REGISTER_CAST_COUNTER(src_type, dst_type, src_type_short, dst_type_short, src_bits, dst_bits,           \
        repeat_size, round_str, round_mode, COUNTER_INIT_UPPER, LV0_LOAD_UPPER, cast_func, LV0_STORE_UPPER)

// for data_type size: src > dst
// repeat_size: the num of elements processed in each repeat depends on the larger data type.
#define REGISTER_CAST_LOWER(cast_func, round_str, round_mode,                                               \
    src_type, dst_type, src_type_short, dst_type_short, src_bits, dst_bits, repeat_size)                    \
    REGISTER_CAST_LV2(src_type, dst_type, src_type_short, dst_type_short, src_bits, dst_bits,               \
        repeat_size, round_str, round_mode, LV2_LOAD_LOWER, cast_func, LV2_STORE_LOWER);                    \
    REGISTER_CAST_BIT(src_type, dst_type, src_type_short, dst_type_short, src_bits, dst_bits,               \
        repeat_size, round_str, round_mode, BIT_INIT_LOWER, LV0_LOAD_LOWER, cast_func, LV0_STORE_LOWER);    \
    REGISTER_CAST_COUNTER(src_type, dst_type, src_type_short, dst_type_short, src_bits, dst_bits,           \
        repeat_size, round_str, round_mode, COUNTER_INIT_LOWER, LV0_LOAD_LOWER, cast_func, LV0_STORE_LOWER)

// for data_type size: src == dst
// repeat_size: the num of elements processed in each repeat depends on the larger data type.
#define REGISTER_CAST_EQUAL(cast_func, round_str, round_mode,                                               \
    src_type, dst_type, src_type_short, dst_type_short, src_bits, dst_bits, repeat_size)                    \
    REGISTER_CAST_LV2(src_type, dst_type, src_type_short, dst_type_short, src_bits, dst_bits,               \
        repeat_size, round_str, round_mode, LV2_LOAD_EQUAL, cast_func, LV2_STORE_EQUAL);                    \
    REGISTER_CAST_BIT(src_type, dst_type, src_type_short, dst_type_short, src_bits, dst_bits,               \
        repeat_size, round_str, round_mode, BIT_INIT_EQUAL, LV0_LOAD_EQUAL, cast_func, LV0_STORE_EQUAL);    \
    REGISTER_CAST_COUNTER(src_type, dst_type, src_type_short, dst_type_short, src_bits, dst_bits,           \
        repeat_size, round_str, round_mode, COUNTER_INIT_EQUAL, LV0_LOAD_EQUAL, cast_func, LV0_STORE_EQUAL)

// Cast::Level 2
#define REGISTER_CAST_LV2_NOT_SUPPORTED(round_str, src_type, dst_type)                                      \
    __aicore__ inline void CastIntrinsicsImpl##round_str(__ubuf__ dst_type* dst, __ubuf__ src_type* src,    \
        const uint32_t calCount)                                                                            \
        {                                                                                                   \
            ASCENDC_ASSERT((false), {                                                                       \
                KERNEL_LOG(KERNEL_ERROR,                                                                    \
                    "round_str from src_type to dst_type not supported on current device");                    \
            });                                                                                             \
        }

// Cast::Level 0 - mask counter mode
#define REGISTER_CAST_COUNTER_NOT_SUPPORTED(round_str, src_type, dst_type)                                  \
    __aicore__ inline void CastIntrinsicsImpl##round_str(__ubuf__ dst_type* dst, __ubuf__ src_type* src,    \
        const uint64_t mask, uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)                    \
        {                                                                                                   \
            ASCENDC_ASSERT((false), {                                                                       \
                KERNEL_LOG(KERNEL_ERROR,                                                                    \
                    "round_str from src_type to dst_type not supported on current device");                    \
            });                                                                                             \
        }

// Cast::Level 0 - mask bit mode
#define REGISTER_CAST_BIT_NOT_SUPPORTED(round_str, src_type, dst_type)                                      \
    __aicore__ inline void CastIntrinsicsImpl##round_str(__ubuf__ dst_type* dst, __ubuf__ src_type* src,    \
        const uint64_t mask[], uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)                 \
        {                                                                                                   \
            ASCENDC_ASSERT((false), {                                                                       \
                KERNEL_LOG(KERNEL_ERROR,                                                                    \
                    "round_str from src_type to dst_type not supported on current device");                    \
            });                                                                                             \
        }

#define REGISTER_CAST_NOT_SUPPORTED(round_str, src_type, dst_type)                                          \
    REGISTER_CAST_LV2_NOT_SUPPORTED(round_str, src_type, dst_type);                                         \
    REGISTER_CAST_COUNTER_NOT_SUPPORTED(round_str, src_type, dst_type);                                     \
    REGISTER_CAST_BIT_NOT_SUPPORTED(round_str, src_type, dst_type)

#define REGISTER_CAST_ROUND_MODE_NOT_SUPPORTED(src_type, dst_type)                                          \
    REGISTER_CAST_NOT_SUPPORTED(CastRint, src_type, dst_type);                                              \
    REGISTER_CAST_NOT_SUPPORTED(CastRound, src_type, dst_type);                                             \
    REGISTER_CAST_NOT_SUPPORTED(CastFloor, src_type, dst_type);                                             \
    REGISTER_CAST_NOT_SUPPORTED(CastCeil, src_type, dst_type);                                              \
    REGISTER_CAST_NOT_SUPPORTED(CastTrunc, src_type, dst_type);                                             \
    REGISTER_CAST_NOT_SUPPORTED(CastOdd, src_type, dst_type)

#define REGISTER_CAST_NONE_MODE_NOT_SUPPORTED(src_type, dst_type)                                           \
    REGISTER_CAST_NOT_SUPPORTED(CastNone, src_type, dst_type)

#define REGISTER_CAST_ODD_MODE_NOT_SUPPORTED(src_type, dst_type)                                            \
    REGISTER_CAST_NOT_SUPPORTED(CastOdd, src_type, dst_type)

// support CAST_RINT, CAST_FLOOR, CAST_CEIL, CAST_ROUND, CAST_TRUNC, CAST_NONE
#define REGISTER_CAST_ROUND_NONE(size_mode, cast_func, src_type, dst_type,                                  \
    src_type_short, dst_type_short, src_bits, dst_bits, repeat_size)                                        \
    REGISTER_CAST_##size_mode(cast_func, CastRound, ROUND_A, src_type, dst_type,                            \
        src_type_short, dst_type_short, src_bits, dst_bits, repeat_size);                                   \
    REGISTER_CAST_##size_mode(cast_func, CastRint, ROUND_R, src_type, dst_type,                             \
        src_type_short, dst_type_short, src_bits, dst_bits, repeat_size);                                   \
    REGISTER_CAST_##size_mode(cast_func, CastFloor, ROUND_F, src_type, dst_type,                            \
        src_type_short, dst_type_short, src_bits, dst_bits, repeat_size);                                   \
    REGISTER_CAST_##size_mode(cast_func, CastTrunc, ROUND_Z, src_type, dst_type,                            \
        src_type_short, dst_type_short, src_bits, dst_bits, repeat_size);                                   \
    REGISTER_CAST_##size_mode(cast_func, CastCeil, ROUND_C, src_type, dst_type,                             \
        src_type_short, dst_type_short, src_bits, dst_bits, repeat_size);                                   \
    REGISTER_CAST_##size_mode(cast_func, CastNone, ROUND_R, src_type, dst_type,                             \
        src_type_short, dst_type_short, src_bits, dst_bits, repeat_size);                                   \
    REGISTER_CAST_ODD_MODE_NOT_SUPPORTED(src_type, dst_type)

// support CAST_RINT, CAST_FLOOR, CAST_CEIL, CAST_ROUND, CAST_TRUNC, CAST_NONE, CAST_ODD
#define REGISTER_CAST_ALL(size_mode, cast_func, src_type, dst_type,                                         \
    src_type_short, dst_type_short, src_bits, dst_bits, repeat_size)                                        \
    REGISTER_CAST_##size_mode(cast_func, CastCeil, ROUND_C, src_type, dst_type,                             \
        src_type_short, dst_type_short, src_bits, dst_bits, repeat_size);                                   \
    REGISTER_CAST_##size_mode(cast_func, CastRound, ROUND_A, src_type, dst_type,                            \
        src_type_short, dst_type_short, src_bits, dst_bits, repeat_size);                                   \
    REGISTER_CAST_##size_mode(cast_func, CastOdd, ROUND_O, src_type, dst_type,                              \
        src_type_short, dst_type_short, src_bits, dst_bits, repeat_size);                                   \
    REGISTER_CAST_##size_mode(cast_func, CastFloor, ROUND_F, src_type, dst_type,                            \
        src_type_short, dst_type_short, src_bits, dst_bits, repeat_size);                                   \
    REGISTER_CAST_##size_mode(cast_func, CastRint, ROUND_R, src_type, dst_type,                             \
        src_type_short, dst_type_short, src_bits, dst_bits, repeat_size);                                   \
    REGISTER_CAST_##size_mode(cast_func, CastTrunc, ROUND_Z, src_type, dst_type,                            \
        src_type_short, dst_type_short, src_bits, dst_bits, repeat_size);                                   \
    REGISTER_CAST_##size_mode(cast_func, CastNone, ROUND_R, src_type, dst_type,                             \
        src_type_short, dst_type_short, src_bits, dst_bits, repeat_size)                                    \

// support CAST_RINT, CAST_FLOOR, CAST_CEIL, CAST_ROUND, CAST_TRUNC
// not support CAST_NONE
#define REGISTER_CAST_ONLY_ROUND(size_mode, cast_func, src_type, dst_type,                                  \
    src_type_short, dst_type_short, src_bits, dst_bits, repeat_size)                                        \
    REGISTER_CAST_##size_mode(cast_func, CastRint, ROUND_R, src_type, dst_type,                             \
        src_type_short, dst_type_short, src_bits, dst_bits, repeat_size);                                   \
    REGISTER_CAST_##size_mode(cast_func, CastRound, ROUND_A, src_type, dst_type,                            \
        src_type_short, dst_type_short, src_bits, dst_bits, repeat_size);                                   \
    REGISTER_CAST_##size_mode(cast_func, CastFloor, ROUND_F, src_type, dst_type,                            \
        src_type_short, dst_type_short, src_bits, dst_bits, repeat_size);                                   \
    REGISTER_CAST_##size_mode(cast_func, CastCeil, ROUND_C, src_type, dst_type,                             \
        src_type_short, dst_type_short, src_bits, dst_bits, repeat_size);                                   \
    REGISTER_CAST_##size_mode(cast_func, CastTrunc, ROUND_Z, src_type, dst_type,                            \
        src_type_short, dst_type_short, src_bits, dst_bits, repeat_size);                                   \
    REGISTER_CAST_NONE_MODE_NOT_SUPPORTED(src_type, dst_type);                                              \
    REGISTER_CAST_ODD_MODE_NOT_SUPPORTED(src_type, dst_type)

// support CAST_NONE
// not support CAST_RINT, CAST_FLOOR, CAST_CEIL, CAST_ROUND, CAST_TRUNC
#define REGISTER_CAST_ONLY_NONE(size_mode, cast_func, src_type, dst_type,                                   \
    src_type_short, dst_type_short, src_bits, dst_bits, repeat_size)                                        \
    REGISTER_CAST_ROUND_MODE_NOT_SUPPORTED(src_type, dst_type);                                             \
    REGISTER_CAST_##size_mode(cast_func, CastNone, ROUND_R, src_type, dst_type,                             \
        src_type_short, dst_type_short, src_bits, dst_bits, repeat_size)

REGISTER_CAST_ONLY_NONE(UPPER, VCVT_U8_TO_F16, uint8_t, half, u8, f16, 8, 16, 128);
REGISTER_CAST_ONLY_NONE(UPPER, VCVT_S8_TO_F16, int8_t, half, s8, f16, 8, 16, 128);
REGISTER_CAST_ONLY_NONE(UPPER, VCVT_F16_TO_F32, half, float, f16, f32, 16, 32, 64);
REGISTER_CAST_ONLY_ROUND(UPPER, VCVT_F16_TO_S32, half, int32_t, f16, s32, 16, 32, 64);
REGISTER_CAST_ONLY_NONE(UPPER, VCVT_S16_TO_F32, int16_t, float, s16, f32, 16, 32, 64);

REGISTER_CAST_ROUND_NONE(LOWER, VCVT_F16_TO_U8, half, uint8_t, f16, u8, 16, 8, 128);
REGISTER_CAST_ROUND_NONE(LOWER, VCVT_F16_TO_S8, half, int8_t, f16, s8, 16, 8, 128);
REGISTER_CAST_ALL(LOWER, VCVT_F32_TO_F16, float, half, f32, f16, 32, 16, 64);
REGISTER_CAST_ONLY_ROUND(LOWER, VCVT_F32_TO_S16, float, int16_t, f32, s16, 32, 16, 64);
REGISTER_CAST_ONLY_ROUND(LOWER, VCVT_S32_TO_F16, int32_t, half, s32, f16, 32, 16, 64);
REGISTER_CAST_ONLY_NONE(LOWER, VCVT_S32_TO_S16, int32_t, int16_t, s32, s16, 32, 16, 64);

REGISTER_CAST_ONLY_ROUND(EQUAL, VCVT_F16_TO_S16, half, int16_t, f16, s16, 16, 16, 128);
REGISTER_CAST_ROUND_NONE(EQUAL, VCVT_S16_TO_F16, int16_t, half, s16, f16, 16, 16, 128);
REGISTER_CAST_ONLY_ROUND(EQUAL, VCVT_F32_TO_F32, float, float, f32, f32, 32, 32, 64);
REGISTER_CAST_ONLY_ROUND(EQUAL, VCVT_F32_TO_S32, float, int32_t, f32, s32, 32, 32, 64);
REGISTER_CAST_ROUND_NONE(EQUAL, VCVT_S32_TO_F32, int32_t, float, s32, f32, 32, 32, 64);

// Cast::Level 2
template <typename U, typename T>
__aicore__ inline void CastImpl(__ubuf__ U* dst, __ubuf__ T* src, const RoundMode& roundMode,
    const uint32_t calCount)
{
    switch (roundMode) {
        case RoundMode::CAST_RINT:
            CastIntrinsicsImplCastRint(dst, src, calCount);
            break;
        case RoundMode::CAST_FLOOR:
            CastIntrinsicsImplCastFloor(dst, src, calCount);
            break;
        case RoundMode::CAST_CEIL:
            CastIntrinsicsImplCastCeil(dst, src, calCount);
            break;
        case RoundMode::CAST_ROUND:
            CastIntrinsicsImplCastRound(dst, src, calCount);
            break;
        case RoundMode::CAST_TRUNC:
            CastIntrinsicsImplCastTrunc(dst, src, calCount);
            break;
        case RoundMode::CAST_ODD:
            CastIntrinsicsImplCastOdd(dst, src, calCount);
            break;
        case RoundMode::CAST_NONE:
            CastIntrinsicsImplCastNone(dst, src, calCount);
            break;
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            break;
    }
}

// Cast::Level 0 - mask bit mode
template <typename U, typename T, bool isSetMask = true>
__aicore__ inline void CastImpl(__ubuf__ U* dst, __ubuf__ T* src, const RoundMode& roundMode,
    const uint64_t mask[], uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    switch (roundMode) {
        case RoundMode::CAST_NONE:
            CastIntrinsicsImplCastNone(dst, src, mask, repeatTimes, repeatParams);
            break;
        case RoundMode::CAST_ROUND:
            CastIntrinsicsImplCastRound(dst, src, mask, repeatTimes, repeatParams);
            break;
        case RoundMode::CAST_RINT:
            CastIntrinsicsImplCastRint(dst, src, mask, repeatTimes, repeatParams);
            break;
        case RoundMode::CAST_FLOOR:
            CastIntrinsicsImplCastFloor(dst, src, mask, repeatTimes, repeatParams);
            break;
        case RoundMode::CAST_TRUNC:
            CastIntrinsicsImplCastTrunc(dst, src, mask, repeatTimes, repeatParams);
            break;
        case RoundMode::CAST_ODD:
            CastIntrinsicsImplCastOdd(dst, src, mask, repeatTimes, repeatParams);
            break;
        case RoundMode::CAST_CEIL:
            CastIntrinsicsImplCastCeil(dst, src, mask, repeatTimes, repeatParams);
            break;
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            break;
    }
}

// Cast::Level 0 - mask count mode
template <typename U, typename T, bool isSetMask = true>
__aicore__ inline void CastImpl(__ubuf__ U* dst, __ubuf__ T* src, const RoundMode& roundMode,
    const uint64_t mask, uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    switch (roundMode) {
        case RoundMode::CAST_RINT:
            CastIntrinsicsImplCastRint(dst, src, mask, repeatTimes, repeatParams);
            break;
        case RoundMode::CAST_FLOOR:
            CastIntrinsicsImplCastFloor(dst, src, mask, repeatTimes, repeatParams);
            break;
        case RoundMode::CAST_CEIL:
            CastIntrinsicsImplCastCeil(dst, src, mask, repeatTimes, repeatParams);
            break;
        case RoundMode::CAST_ROUND:
            CastIntrinsicsImplCastRound(dst, src, mask, repeatTimes, repeatParams);
            break;
        case RoundMode::CAST_TRUNC:
            CastIntrinsicsImplCastTrunc(dst, src, mask, repeatTimes, repeatParams);
            break;
        case RoundMode::CAST_ODD:
            CastIntrinsicsImplCastOdd(dst, src, mask, repeatTimes, repeatParams);
            break;
        case RoundMode::CAST_NONE:
            CastIntrinsicsImplCastNone(dst, src, mask, repeatTimes, repeatParams);
            break;
        default:
            ASCENDC_ASSERT((false),
                { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            break;
    }
}

template <typename U, typename T, bool isVecDeq, bool halfBlock>
__aicore__ inline void CastDeqImpl(__ubuf__ U* dst, __ubuf__ T* src,
    const uint32_t calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "CastDeq is not supported on current device"); });
}

template <typename U, typename T, bool isSetMask = true, bool isVecDeq, bool halfBlock>
__aicore__ inline void CastDeqImpl(__ubuf__ U* dst, __ubuf__ T* src,
    const uint64_t mask[], uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "CastDeq is not supported on current device"); });
}

template <typename U, typename T, bool isSetMask = true, bool isVecDeq, bool halfBlock>
__aicore__ inline void CastDeqImpl(__ubuf__ U* dst, __ubuf__ T* src,
    const int32_t mask, uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "CastDeq is not supported on current device"); });
}

// AddReluCast::Level 0 - mask count mode
template <typename DST_TYPE, typename SRC_TYPE, bool isSetMask = true>
__aicore__ inline void AddReluCastImpl(__ubuf__ DST_TYPE* dst, __ubuf__ SRC_TYPE* src0, __ubuf__ SRC_TYPE* src1,
    const uint64_t mask, uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "AddReluCast is not supported on current device"); });
}

// AddReluCast::Level 0 - mask bit mode
template <typename DST_TYPE, typename SRC_TYPE, bool isSetMask = true>
__aicore__ inline void AddReluCastImpl(__ubuf__ DST_TYPE* dst, __ubuf__ SRC_TYPE* src0, __ubuf__ SRC_TYPE* src1,
    const uint64_t mask[], uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "AddReluCast is not supported on current device"); });
}

// AddReluCast::Level 2
template <typename DST_TYPE, typename SRC_TYPE>
__aicore__ inline void AddReluCastImpl(__ubuf__ DST_TYPE* dst, __ubuf__ SRC_TYPE* src0, __ubuf__ SRC_TYPE* src1,
    const uint32_t calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "AddReluCast is not supported on current device"); });
}

// SubReluCast::Level 0 - mask count mode
template <typename DST_TYPE, typename SRC_TYPE, bool isSetMask = true>
__aicore__ inline void SubReluCastImpl(__ubuf__ DST_TYPE* dst, __ubuf__ SRC_TYPE* src0, __ubuf__ SRC_TYPE* src1,
    const uint64_t mask, uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SubReluCast is not supported on current device"); });
}

// SubReluCast::Level 0 - mask bit mode
template <typename DST_TYPE, typename SRC_TYPE, bool isSetMask = true>
__aicore__ inline void SubReluCastImpl(__ubuf__ DST_TYPE* dst, __ubuf__ SRC_TYPE* src0, __ubuf__ SRC_TYPE* src1,
    const uint64_t mask[], uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SubReluCast is not supported on current device"); });
}

// SubReluCast::Level 2
template <typename DST_TYPE, typename SRC_TYPE>
__aicore__ inline void SubReluCastImpl(__ubuf__ DST_TYPE* dst, __ubuf__ SRC_TYPE* src0, __ubuf__ SRC_TYPE* src1,
    const uint32_t calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SubReluCast is not supported on current device"); });
}

__aicore__ inline void SetDeqScaleImpl(float scale, int16_t offset, bool signMode)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetDeqScale is not supported on current device"); });
}

template <typename T>
__aicore__ inline void SetDeqScaleImpl(const LocalTensor<T>& vdeqTensor, const VdeqInfo& vdeqInfo)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetDeqScale is not supported on current device"); });
}

template<typename T>
__aicore__ inline void SetDeqScaleImpl(T config)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SetDeqScale is not supported on current device"); });
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_VCONV_IMPL_H
