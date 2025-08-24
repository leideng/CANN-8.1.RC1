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
 * \file kernel_operator_vec_cmpsel_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_CMPSEL_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_CMPSEL_IMPL_H
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

template <typename T, typename U>
__aicore__ inline void CompareCompute(__ubuf__ U* dst, __ubuf__ T* src0, __ubuf__ T* src1, CMPMODE cmpMode,
    const uint32_t calCount)
{
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

// Compare::Level 0 - mask bit mode
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvImpl(__ubuf__ U* dst, __ubuf__ T* src0, __ubuf__ T* src1, CMPMODE cmpMode,
    const uint64_t mask[], uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    (void)(mask);
    VcmpvIntrinsicsImpl(reinterpret_cast<__ubuf__ uint8_t*>(dst), src0, src1, cmpMode, repeatTimes, repeatParams);
}

// Compare::Level 0 - mask count mode
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvImpl(__ubuf__ U* dst, __ubuf__ T* src0, __ubuf__ T* src1, CMPMODE cmpMode,
    const uint64_t mask, uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    (void)(mask);
    VcmpvIntrinsicsImpl(reinterpret_cast<__ubuf__ uint8_t*>(dst), src0, src1, cmpMode, repeatTimes, repeatParams);
}

// Compare::Level 2
template <typename T, typename U>
__aicore__ inline void VcmpvImpl(__ubuf__ U* dst, __ubuf__ T* src0, __ubuf__ T* src1, CMPMODE cmpMode,
    const uint32_t calCount)
{
    CompareCompute(dst, src0, src1, cmpMode, calCount);
}

// Compare written to CMPMASK
template <typename T, bool isSetMask = true>
__aicore__ inline void VcmpImpl(__ubuf__ T* src0, __ubuf__ T* src1, CMPMODE cmpMode,
    const uint64_t mask[], const BinaryRepeatParams& repeatParams)
{
    static_assert((__CCE_AICORE__ == 100), "Compare written to CMPMASK is not supported on current device");
}

template <typename T, bool isSetMask = true>
__aicore__ inline void VcmpImpl(__ubuf__ T* src0, __ubuf__ T* src1, CMPMODE cmpMode,
    const uint64_t mask, const BinaryRepeatParams& repeatParams)
{
    static_assert((__CCE_AICORE__ == 100), "Compare written to CMPMASK is not supported on current device");
}

/* ***************************************************************************************
 * *********************************** CompareScalar *************************************
 * ************************************************************************************** */
// CompareScalar::Level 0 - mask bit mode
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvsImpl(__ubuf__ U* dst, __ubuf__ T* src0, T src1, CMPMODE cmpMode,
    const uint64_t mask[], uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "CompareScalar");
}

// CompareScalar::Level 0 - mask count mode
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvsImpl(__ubuf__ U* dst, __ubuf__ T* src0, T src1, CMPMODE cmpMode,
    const uint64_t mask, uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "CompareScalar");
}

// CompareScalar::Level 2
template <typename T, typename U>
__aicore__ inline void VcmpvsImpl(__ubuf__ U* dst, __ubuf__ T* src0, T src1, CMPMODE cmpMode,
    const uint32_t calCount)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "CompareScalar");
}

/* ***************************************************************************************
 * *************************************** Select ****************************************
 * ************************************************************************************** */
// ============ select mode: 0/2 ============
template <typename T, typename U>
__aicore__ inline void VselIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, __ubuf__ T* src1,
    SELMODE selMode, int32_t repeat, const BinaryRepeatParams& repeatParams)
{
    if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
        set_cmpmask(sel);
        PipeBarrier<PIPE_V>();
        vsel(dst, src0, src1, repeat, repeatParams.dstBlkStride, repeatParams.src0BlkStride, repeatParams.src1BlkStride,
            repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
    } else {
        ASCENDC_ASSERT(false,
                       { KERNEL_LOG(KERNEL_ERROR, "current device only support select mode 0 (VSEL_CMPMASK_SPR) !"); });
    }
}

template <typename T, SELMODE selMode = SELMODE::VSEL_CMPMASK_SPR>
__aicore__ inline void SelectCal(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, int32_t repeat,
    const BinaryRepeatParams& repeatParams)
{
    if constexpr (selMode == SELMODE::VSEL_CMPMASK_SPR) {
        vsel(dst, src0, src1, repeat, repeatParams.dstBlkStride, repeatParams.src0BlkStride, repeatParams.src1BlkStride,
            repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
    } else {
        ASCENDC_ASSERT(false,
                       { KERNEL_LOG(KERNEL_ERROR, "current device only support select mode 0 (VSEL_CMPMASK_SPR) !"); });
    }
}

template <typename T, typename U>
__aicore__ inline void SelectCal(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, int32_t repeat,
    const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current device only support select mode 0 (VSEL_CMPMASK_SPR) !"); });
}

/* ***************************************************************************************
 * *************************************** Select ****************************************
 * ************************************************************************************** */
// ============ select mode: 0/2 ============
// only for sel mode 0, cuz no selmask offset
template <typename T, typename U>
__aicore__ inline void VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, __ubuf__ T* src1, SELMODE selMode,
    uint32_t calCount)
{
    BinaryRepeatParams repeatParams;
    IntriInfo intriInfo = AscendCUtils::CalIntriInfo(sizeof(T), calCount);

    uint32_t dstOffset = 0;
    uint32_t src0Offset = 0;
    uint32_t src1Offset = 0;
    const auto dstOffsetCount = MAX_REPEAT_TIMES * repeatParams.dstRepStride * intriInfo.c0Count;
    const auto src0OffsetCount = MAX_REPEAT_TIMES * repeatParams.src0RepStride * intriInfo.c0Count;
    const auto src1OffsetCount = MAX_REPEAT_TIMES * repeatParams.src1RepStride * intriInfo.c0Count;

    const int32_t fullMask = intriInfo.c0Count * DEFAULT_BLK_NUM;
    AscendCUtils::SetMask<T>(fullMask);

    for (int32_t i = 0; i < intriInfo.repeatRounding; i++) {
        VselIntrinsicsImpl(reinterpret_cast<__ubuf__ T*>(dst + dstOffset), reinterpret_cast<__ubuf__ U*>(sel),
            reinterpret_cast<__ubuf__ T*>(src0 + src0Offset), reinterpret_cast<__ubuf__ T*>(src1 + src1Offset), selMode,
            MAX_REPEAT_TIMES, repeatParams);
        dstOffset += dstOffsetCount;
        src0Offset += src0OffsetCount;
        src1Offset += src1OffsetCount;
    }

    dstOffset = (intriInfo.repeatRounding * MAX_REPEAT_TIMES) * repeatParams.dstRepStride * intriInfo.c0Count;
    src0Offset = (intriInfo.repeatRounding * MAX_REPEAT_TIMES) * repeatParams.src0RepStride * intriInfo.c0Count;
    src1Offset = (intriInfo.repeatRounding * MAX_REPEAT_TIMES) * repeatParams.src1RepStride * intriInfo.c0Count;

    if (intriInfo.repeatRemaining != 0) {
        VselIntrinsicsImpl(reinterpret_cast<__ubuf__ T*>(dst + dstOffset), reinterpret_cast<__ubuf__ U*>(sel),
            reinterpret_cast<__ubuf__ T*>(src0 + src0Offset), reinterpret_cast<__ubuf__ T*>(src1 + src1Offset), selMode,
            intriInfo.repeatRemaining, repeatParams);
    }

    if (intriInfo.tail != 0) {
        AscendCUtils::SetMask<T>(intriInfo.tail);
        // cal sel mask offset, only for sel mode 0
        int32_t selMaskOriginOffset = intriInfo.repeat * DEFAULT_REPEAT_STRIDE * intriInfo.c0Count;
        int32_t selMaskOffset = selMaskOriginOffset / AscendCUtils::GetBitSize(sizeof(U));

        dstOffset = intriInfo.repeat * repeatParams.dstRepStride * intriInfo.c0Count;
        src0Offset = intriInfo.repeat * repeatParams.src0RepStride * intriInfo.c0Count;
        src1Offset = intriInfo.repeat * repeatParams.src1RepStride * intriInfo.c0Count;
        VselIntrinsicsImpl(reinterpret_cast<__ubuf__ T*>(dst + dstOffset), reinterpret_cast<__ubuf__ U*>(sel),
            reinterpret_cast<__ubuf__ T*>(src0 + src0Offset), reinterpret_cast<__ubuf__ T*>(src1 + src1Offset), selMode,
            1, repeatParams);
        ResetMask();
    }
}

template <typename T, typename U>
__aicore__ inline void VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, T src1, SELMODE selMode,
    uint32_t calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current device only support select mode 0 (VSEL_CMPMASK_SPR) !"); });
}

// select mode: 0/1/2
template <typename T, typename U>
__aicore__ inline void VselImpl(__ubuf__ T* dstLocal, __ubuf__ U* selMask, __ubuf__ T* src0Local, __ubuf__ T* src1Local,
    SELMODE selMode, const uint64_t mask[], const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    AscendCUtils::SetMask<T>(mask[1], mask[0]);
    VselIntrinsicsImpl(dstLocal, selMask, src0Local, src1Local, selMode, repeatTimes, repeatParams);
}

// select mode: 0/1/2
template <typename T, typename U>
__aicore__ inline void VselImpl(__ubuf__ T* dstLocal, __ubuf__ U* selMask, __ubuf__ T* src0Local, __ubuf__ T* src1Local,
    SELMODE selMode, const uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    AscendCUtils::SetMask<T>(mask);
    VselIntrinsicsImpl(dstLocal, selMask, src0Local, src1Local, selMode, repeatTimes, repeatParams);
}

// select mode: 1
template <typename T, typename U>
__aicore__ inline void VselImpl(__ubuf__ T* dstLocal, __ubuf__ U* selMask, __ubuf__ T* src0Local, T src1Local,
    SELMODE selMode, const uint64_t mask[], const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current device only support select mode 0 (VSEL_CMPMASK_SPR) !"); });
}

// select mode: 1
template <typename T, typename U>
__aicore__ inline void VselImpl(__ubuf__ T* dstLocal, __ubuf__ U* selMask, __ubuf__ T* src0Local, T src1Local,
    SELMODE selMode, const uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current device only support select mode 0 (VSEL_CMPMASK_SPR) !"); });
}

template <typename T>
__aicore__ inline void GetCmpMaskImpl(__ubuf__ T* dst)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "GetCmpMask");
}

template <typename T>
__aicore__ inline void SetCmpMaskImpl(__ubuf__ T* src)
{
    set_cmpmask(src);
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_CMPSEL_IMPL_H
