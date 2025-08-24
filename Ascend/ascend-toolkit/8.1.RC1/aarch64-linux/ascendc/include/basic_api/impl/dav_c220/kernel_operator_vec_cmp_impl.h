/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
 * \file kernel_operator_vec_cmp_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_CMP_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_CMP_IMPL_H
#include "kernel_common.h"
#include "kernel_utils.h"
#include "kernel_struct_binary.h"
#include "kernel_struct_unary.h"

namespace AscendC {
/* ***************************************************************************************
 * ************************************** Compare ****************************************
 * ************************************************************************************** */
template <typename T>
__aicore__ inline void VcmpvIntrinsicsImpl(__ubuf__ uint8_t* dst, __ubuf__ T* src0, __ubuf__ T* src1, CMPMODE cmpMode,
    uint8_t repeat, const BinaryRepeatParams& repeatParams)
{
    switch (cmpMode) {
        case CMPMODE::LT: {
            vcmpv_lt(dst, src0, src1, repeat, repeatParams.dstBlkStride,
                repeatParams.src0BlkStride, repeatParams.src1BlkStride, repeatParams.dstRepStride,
                repeatParams.src0RepStride, repeatParams.src1RepStride);
            break;
        }
        case CMPMODE::GT: {
            vcmpv_gt(dst, src0, src1, repeat, repeatParams.dstBlkStride,
                repeatParams.src0BlkStride, repeatParams.src1BlkStride, repeatParams.dstRepStride,
                repeatParams.src0RepStride, repeatParams.src1RepStride);
            break;
        }
        case CMPMODE::EQ: {
            vcmpv_eq(dst, src0, src1, repeat, repeatParams.dstBlkStride,
                repeatParams.src0BlkStride, repeatParams.src1BlkStride, repeatParams.dstRepStride,
                repeatParams.src0RepStride, repeatParams.src1RepStride);
            break;
        }
        case CMPMODE::LE: {
            vcmpv_le(dst, src0, src1, repeat, repeatParams.dstBlkStride,
                repeatParams.src0BlkStride, repeatParams.src1BlkStride, repeatParams.dstRepStride,
                repeatParams.src0RepStride, repeatParams.src1RepStride);
            break;
        }
        case CMPMODE::GE: {
            vcmpv_ge(dst, src0, src1, repeat, repeatParams.dstBlkStride,
                repeatParams.src0BlkStride, repeatParams.src1BlkStride, repeatParams.dstRepStride,
                repeatParams.src0RepStride, repeatParams.src1RepStride);
            break;
        }
        case CMPMODE::NE: {
            vcmpv_ne(dst, src0, src1, repeat, repeatParams.dstBlkStride,
                repeatParams.src0BlkStride, repeatParams.src1BlkStride, repeatParams.dstRepStride,
                repeatParams.src0RepStride, repeatParams.src1RepStride);
            break;
        }
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cmp mode %d", static_cast<int32_t>(cmpMode)); });
            break;
    }
}

template <typename T>
__aicore__ inline void VcmpIntrinsicsImpl(__ubuf__ T* src0, __ubuf__ T* src1, CMPMODE cmpMode,
    const BinaryRepeatParams& repeatParams)
{
    switch (cmpMode) {
        case CMPMODE::LT: {
            vcmp_lt(src0, src1, 1,
                repeatParams.dstBlkStride, repeatParams.src0BlkStride, repeatParams.src1BlkStride,
                repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
            break;
        }
        case CMPMODE::GT: {
            vcmp_gt(src0, src1, 1,
                repeatParams.dstBlkStride, repeatParams.src0BlkStride, repeatParams.src1BlkStride,
                repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
            break;
        }
        case CMPMODE::EQ: {
            vcmp_eq(src0, src1, 1,
                repeatParams.dstBlkStride, repeatParams.src0BlkStride, repeatParams.src1BlkStride,
                repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
            break;
        }
        case CMPMODE::LE: {
            vcmp_le(src0, src1, 1,
                repeatParams.dstBlkStride, repeatParams.src0BlkStride, repeatParams.src1BlkStride,
                repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
            break;
        }
        case CMPMODE::GE: {
            vcmp_ge(src0, src1, 1,
                repeatParams.dstBlkStride, repeatParams.src0BlkStride, repeatParams.src1BlkStride,
                repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
            break;
        }
        case CMPMODE::NE: {
            vcmp_ne(src0, src1, 1,
                repeatParams.dstBlkStride, repeatParams.src0BlkStride, repeatParams.src1BlkStride,
                repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
            break;
        }
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cmp mode %d", static_cast<int32_t>(cmpMode)); });
            break;
    }
}

__aicore__ inline void VcmpvIntrinsicsImpl(__ubuf__ uint8_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1,
    CMPMODE cmpMode, uint8_t repeat, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(
        (cmpMode == CMPMODE::EQ),
        { KERNEL_LOG(KERNEL_ERROR, "Compare: Only CMPMODE::EQ is valid for int32_t input!"); });
    vcmpv_eq(dst, src0, src1, repeat, repeatParams.dstBlkStride,
        repeatParams.src0BlkStride, repeatParams.src1BlkStride, repeatParams.dstRepStride,
        repeatParams.src0RepStride, repeatParams.src1RepStride);
}

template <typename T, typename U>
__aicore__ inline void CompareCompute(__ubuf__ U* dst, __ubuf__ T* src0, __ubuf__ T* src1, CMPMODE cmpMode,
    const uint32_t calCount)
{
    if ASCEND_IS_AIV {
        struct BinaryRepeatParams repeatParams;
        uint32_t sumRepeat = calCount * sizeof(T) / ONE_REPEAT_BYTE_SIZE;
        constexpr uint32_t repeatNormal = 252;
        uint32_t repeatRound = sumRepeat / repeatNormal;
        uint32_t repeatTail = sumRepeat % repeatNormal;
        uint32_t srcOffset = repeatNormal * ONE_REPEAT_BYTE_SIZE / sizeof(T);
        uint32_t dstOffset = srcOffset / ONE_BYTE_BIT_SIZE;

        for (uint32_t i = 0; i < repeatRound; ++i) {
            VcmpvImpl(reinterpret_cast<__ubuf__ uint8_t*>(dst) + i * dstOffset,
                src0 + i * srcOffset,
                src1 + i * srcOffset, cmpMode, MASK_PLACEHOLDER, repeatNormal,
                repeatParams);
        }
        VcmpvImpl(reinterpret_cast<__ubuf__ uint8_t*>(dst) + repeatRound * dstOffset,
            src0 + repeatRound * srcOffset,
            src1 + repeatRound * srcOffset, cmpMode, MASK_PLACEHOLDER, repeatTail,
            repeatParams);
    }
}

// Compare::Level 0 - mask bit mode
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvImpl(__ubuf__ U* dst, __ubuf__ T* src0, __ubuf__ T* src1, CMPMODE cmpMode,
    const uint64_t mask[], uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    (void)(mask);
    if ASCEND_IS_AIV {
        VcmpvIntrinsicsImpl(reinterpret_cast<__ubuf__ uint8_t*>(dst), src0, src1, cmpMode,
            repeatTimes, repeatParams);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void VcmpImpl(__ubuf__ T* src0, __ubuf__ T* src1, CMPMODE cmpMode,
    const uint64_t mask[], const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
        VcmpIntrinsicsImpl(src0, src1, cmpMode, repeatParams);
    }
}

// Compare::Level 0 - mask count mode
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvImpl(__ubuf__ U* dst, __ubuf__ T* src0, __ubuf__ T* src1, CMPMODE cmpMode,
    const uint64_t mask, uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    (void)(mask);
    if ASCEND_IS_AIV {
        VcmpvIntrinsicsImpl(reinterpret_cast<__ubuf__ uint8_t*>(dst), src0, src1, cmpMode,
            repeatTimes, repeatParams);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void VcmpImpl(__ubuf__ T* src0, __ubuf__ T* src1, CMPMODE cmpMode,
    const uint64_t mask, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask);
        VcmpIntrinsicsImpl(src0, src1, cmpMode, repeatParams);
    }
}

// Compare::Level 2
template <typename T, typename U>
__aicore__ inline void VcmpvImpl(__ubuf__ U* dst, __ubuf__ T* src0, __ubuf__ T* src1, CMPMODE cmpMode,
    const uint32_t calCount)
{
    if ASCEND_IS_AIV {
        if constexpr (IsSameType<T, int32_t>::value) {
            ASCENDC_ASSERT((cmpMode == CMPMODE::EQ),
                { KERNEL_LOG(KERNEL_ERROR, "Failed to check CMPMODE in Compare, current api only support "
                    "CMPMODE::EQ, when src dtype is int32_t."); });
        }
        CompareCompute(reinterpret_cast<__ubuf__ uint8_t*>(dst), src0, src1, cmpMode, calCount);
    }
}

/* ***************************************************************************************
 * *********************************** CompareScalar *************************************
 * ************************************************************************************** */
template <typename T>
__aicore__ inline void VcmpvsIntrinsicsImpl(__ubuf__ uint8_t* dst, __ubuf__ T* src0, T src1, CMPMODE cmpMode,
    uint8_t repeat, const UnaryRepeatParams& repeatParams)
{
    switch (cmpMode) {
        case CMPMODE::LT: {
            vcmpvs_lt(dst, src0, src1, repeat,
                static_cast<uint16_t>(repeatParams.dstBlkStride), static_cast<uint16_t>(repeatParams.srcBlkStride),
                static_cast<uint16_t>(repeatParams.dstRepStride), static_cast<uint16_t>(repeatParams.srcRepStride));
            break;
        }
        case CMPMODE::GT: {
            vcmpvs_gt(dst, src0, src1, repeat,
                static_cast<uint16_t>(repeatParams.dstBlkStride), static_cast<uint16_t>(repeatParams.srcBlkStride),
                static_cast<uint16_t>(repeatParams.dstRepStride), static_cast<uint16_t>(repeatParams.srcRepStride));
            break;
        }
        case CMPMODE::EQ: {
            vcmpvs_eq(dst, src0, src1, repeat,
                static_cast<uint16_t>(repeatParams.dstBlkStride), static_cast<uint16_t>(repeatParams.srcBlkStride),
                static_cast<uint16_t>(repeatParams.dstRepStride), static_cast<uint16_t>(repeatParams.srcRepStride));
            break;
        }
        case CMPMODE::LE: {
            vcmpvs_le(dst, src0, src1, repeat,
                static_cast<uint16_t>(repeatParams.dstBlkStride), static_cast<uint16_t>(repeatParams.srcBlkStride),
                static_cast<uint16_t>(repeatParams.dstRepStride), static_cast<uint16_t>(repeatParams.srcRepStride));
            break;
        }
        case CMPMODE::GE: {
            vcmpvs_ge(dst, src0, src1, repeat,
                static_cast<uint16_t>(repeatParams.dstBlkStride), static_cast<uint16_t>(repeatParams.srcBlkStride),
                static_cast<uint16_t>(repeatParams.dstRepStride), static_cast<uint16_t>(repeatParams.srcRepStride));
            break;
        }
        case CMPMODE::NE: {
            vcmpvs_ne(dst, src0, src1, repeat,
                static_cast<uint16_t>(repeatParams.dstBlkStride), static_cast<uint16_t>(repeatParams.srcBlkStride),
                static_cast<uint16_t>(repeatParams.dstRepStride), static_cast<uint16_t>(repeatParams.srcRepStride));
            break;
        }
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cmp mode %d", static_cast<int32_t>(cmpMode)); });
            break;
    }
}

__aicore__ inline void VcmpvsIntrinsicsImpl(__ubuf__ uint8_t* dst, __ubuf__ int32_t* src0, int32_t src1,
    CMPMODE cmpMode, uint8_t repeat, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(
        (cmpMode == CMPMODE::EQ),
        { KERNEL_LOG(KERNEL_ERROR, "CompareScalar: Only CMPMODE::EQ is valid for int32_t input!"); });
    vcmpvs_eq(dst, src0, src1, repeat,
        static_cast<uint16_t>(repeatParams.dstBlkStride), static_cast<uint16_t>(repeatParams.srcBlkStride),
        static_cast<uint16_t>(repeatParams.dstRepStride), static_cast<uint16_t>(repeatParams.srcRepStride));
}

template <typename T, typename U>
__aicore__ inline void CompareScalarCompute(__ubuf__ U* dst, __ubuf__ T* src0, T src1, CMPMODE cmpMode,
    const uint32_t calCount)
{
    if ASCEND_IS_AIV {
        struct UnaryRepeatParams repeatParams;
        uint32_t sumRepeat = calCount * sizeof(T) / ONE_REPEAT_BYTE_SIZE;
        constexpr uint32_t repeatNormal = 252;
        uint32_t repeatRound = sumRepeat / repeatNormal;
        uint32_t repeatTail = sumRepeat % repeatNormal;
        uint32_t srcOffset = repeatNormal * ONE_REPEAT_BYTE_SIZE / sizeof(T);
        uint32_t dstOffset = srcOffset / ONE_BYTE_BIT_SIZE;
        for (uint32_t i = 0; i < repeatRound; ++i) {
            VcmpvsImpl(reinterpret_cast<__ubuf__ uint8_t*>(dst) + i * dstOffset,
                src0 + i * srcOffset,
                src1, cmpMode, MASK_PLACEHOLDER, repeatNormal,
                repeatParams);
        }
        VcmpvsImpl(reinterpret_cast<__ubuf__ uint8_t*>(dst) + repeatRound * dstOffset,
            src0 + repeatRound * srcOffset,
            src1, cmpMode, MASK_PLACEHOLDER, repeatTail,
            repeatParams);
    }
}

// CompareScalar::Level 0 - mask bit mode
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvsImpl(__ubuf__ U* dst, __ubuf__ T* src0, T src1, CMPMODE cmpMode,
    const uint64_t mask[], uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    (void)(mask);
    if ASCEND_IS_AIV {
        VcmpvsIntrinsicsImpl(reinterpret_cast<__ubuf__ uint8_t*>(dst), src0, src1, cmpMode,
            repeatTimes, repeatParams);
    }
}

// CompareScalar::Level 0 - mask count mode
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvsImpl(__ubuf__ U* dst, __ubuf__ T* src0, T src1, CMPMODE cmpMode,
    const uint64_t mask, uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    (void)(mask);
    if ASCEND_IS_AIV {
        VcmpvsIntrinsicsImpl(reinterpret_cast<__ubuf__ uint8_t*>(dst), src0, src1, cmpMode,
            repeatTimes, repeatParams);
    }
}

// CompareScalar::Level 2
template <typename T, typename U>
__aicore__ inline void VcmpvsImpl(__ubuf__ U* dst, __ubuf__ T* src0, T src1, CMPMODE cmpMode,
    const uint32_t calCount)
{
    if ASCEND_IS_AIV {
        if constexpr (IsSameType<T, int32_t>::value) {
            ASCENDC_ASSERT((cmpMode == CMPMODE::EQ),
                { KERNEL_LOG(KERNEL_ERROR, "Failed to check CMPMODE in CompareScalar, current api only support "
                    "CMPMODE::EQ, when src dtype is int32_t."); });
        }
        CompareScalarCompute(reinterpret_cast<__ubuf__ uint8_t*>(dst), src0, src1, cmpMode, calCount);
    }
}

template <typename T>
__aicore__ inline void GetCmpMaskImpl(__ubuf__ T* dst)
{
    get_cmpmask(dst);
}

template <typename T>
__aicore__ inline void SetCmpMaskImpl(__ubuf__ T* src)
{
    set_cmpmask(src);
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_CMP_IMPL_H
