/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file clamp_common_impl.h
 * \brief
 */
#ifndef IMPL_MATH_CLAMP_CLAMP_COMMON_IMPL_H
#define IMPL_MATH_CLAMP_CLAMP_COMMON_IMPL_H
#include "kernel_tensor.h"
#include "../../common/check.h"

namespace AscendC {
constexpr uint32_t CLAMP_FLOAT_MASK = 64;
constexpr uint32_t CLAMP_HALF_MASK = 128;
constexpr uint32_t CLAMP_BYTE_PER_REPEAT = 512;

struct ClampParams {
    __aicore__ ClampParams(){};
    uint32_t vcmpvsRepeat = 0;
    uint64_t ClampMask = 0;
    uint64_t selectTailElement = 0;
    uint32_t selectTailOffset = 0;
    uint32_t clampSplitCount = 0;
    uint32_t selectTailRepeatLoop = 0;
    uint32_t selectTailRepeatTail = 0;
    uint32_t selectTailPreRepeatOffset = 0;
    uint32_t selectTailMainRepeatOffset = 0;
    uint32_t selectTailTailRepeatOffset = 0;
    uint32_t loopCount = 0;
    uint32_t calcTail = 0;
    uint32_t vcmpvsRepeatLoop = 0;
    uint32_t vcmpvsRepeatTail = 0;
    uint32_t vcmpvsPreRepeatOffset = 0;
    uint32_t vcmpvsMainRepeatOffset = 0;
};

template <typename T>
__aicore__ inline void ClampComputeCount(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const T scalar, const uint32_t repeat, const uint64_t mask,
    CLAMPMODE selMode, const BinaryRepeatParams& repeatParams)
{
    if (selMode == CLAMPMODE::CLAMP_MAX) {
        CompareScalar(sharedTmpBuffer, srcTensor, static_cast<T>(scalar), CMPMODE::LT, mask, (uint8_t)repeat,
            { DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    } else if (selMode == CLAMPMODE::CLAMP_MIN) {
        CompareScalar(sharedTmpBuffer, srcTensor, static_cast<T>(scalar), CMPMODE::GT, mask, (uint8_t)repeat,
            { DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    } else {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "selMode is not supported"); });
    }
    PipeBarrier<PIPE_V>();
    // 2、selsct 1
    Select(dstTensor, sharedTmpBuffer, srcTensor, static_cast<T>(scalar), SELMODE::VSEL_TENSOR_SCALAR_MODE, mask,
        (uint8_t)repeat, repeatParams);
    PipeBarrier<PIPE_V>();
}

template <typename T> __aicore__ inline void GetMainParams(const uint32_t calCount, ClampParams& params,
    const uint32_t sharedTmpBufferSize)
{
    if constexpr (sizeof(T) == sizeof(uint16_t)) {
        params.clampSplitCount = params.clampSplitCount / CLAMP_HALF_MASK * CLAMP_HALF_MASK;
        params.vcmpvsRepeat = params.clampSplitCount / CLAMP_HALF_MASK;
        params.ClampMask = CLAMP_HALF_MASK;
    } else {
        params.clampSplitCount = params.clampSplitCount / CLAMP_FLOAT_MASK * CLAMP_FLOAT_MASK;
        params.vcmpvsRepeat = params.clampSplitCount / CLAMP_FLOAT_MASK;
        params.ClampMask = CLAMP_FLOAT_MASK;
    }
    CheckTmpBufferSize(params.clampSplitCount, 0, sharedTmpBufferSize);
    params.loopCount = calCount / params.clampSplitCount;
    params.calcTail = calCount % params.clampSplitCount;
    params.vcmpvsRepeatLoop = params.vcmpvsRepeat / MAX_REPEAT_TIMES;
    params.vcmpvsRepeatTail = params.vcmpvsRepeat % MAX_REPEAT_TIMES;
    params.vcmpvsPreRepeatOffset = MAX_REPEAT_TIMES * params.ClampMask;
    params.vcmpvsMainRepeatOffset = params.vcmpvsRepeatLoop * MAX_REPEAT_TIMES * params.ClampMask;
}

template <typename T> __aicore__ inline void GetTailParams(const uint32_t calcTail, ClampParams& params)
{
    if constexpr (sizeof(T) == sizeof(uint16_t)) {
        params.vcmpvsRepeat = calcTail / CLAMP_HALF_MASK;
        params.ClampMask = (calcTail < CLAMP_HALF_MASK) ? calcTail : CLAMP_HALF_MASK;
        params.selectTailElement = calcTail % CLAMP_HALF_MASK;
        params.selectTailOffset = params.vcmpvsRepeat * CLAMP_HALF_MASK;
    } else {
        params.vcmpvsRepeat = calcTail / CLAMP_FLOAT_MASK;
        params.ClampMask = (calcTail < CLAMP_FLOAT_MASK) ? calcTail : CLAMP_FLOAT_MASK;
        params.selectTailElement = calcTail % CLAMP_FLOAT_MASK;
        params.selectTailOffset = params.vcmpvsRepeat * CLAMP_FLOAT_MASK;
    }
    params.selectTailRepeatLoop = params.vcmpvsRepeat / MAX_REPEAT_TIMES;
    params.selectTailRepeatTail = params.vcmpvsRepeat % MAX_REPEAT_TIMES;
    params.selectTailPreRepeatOffset = MAX_REPEAT_TIMES * params.ClampMask;
    params.selectTailMainRepeatOffset = params.selectTailRepeatLoop * MAX_REPEAT_TIMES * params.ClampMask;
    params.selectTailTailRepeatOffset = params.selectTailRepeatLoop * MAX_REPEAT_TIMES * params.ClampMask +
        params.selectTailRepeatTail * params.ClampMask;
}

template <typename T>
__aicore__ inline void ClampCompute(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const T scalar, const uint32_t calCount, CLAMPMODE selMode,
    ClampParams& params)
{
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");

    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Clamp");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Clamp");

    ASCENDC_ASSERT((std::is_same<T, float>::value || std::is_same<T, half>::value), {
        KERNEL_LOG( KERNEL_ERROR, "Failed to check the data types, current api support data types are half/float."); });
    uint32_t sharedTmpBufferSize = sharedTmpBuffer.GetSize();
    params.clampSplitCount = sharedTmpBufferSize * sizeof(uint8_t) / sizeof(uint8_t);
    // split the input based on the stack buffer.
    GetMainParams<T>(calCount, params, sharedTmpBufferSize);
    BinaryRepeatParams vselRepeatParams;
    // clampSplitCount is full mask aligned. The loop in loopCount does not have a tail block.
    for (uint32_t i = 0; i < params.loopCount; i++) {
        for (uint32_t j = 0; j < params.vcmpvsRepeatLoop; j++) {
            ClampComputeCount<T>(dstTensor[i * params.clampSplitCount + j * params.vcmpvsPreRepeatOffset],
                srcTensor[i * params.clampSplitCount + j * params.vcmpvsPreRepeatOffset], sharedTmpBuffer, scalar,
                MAX_REPEAT_TIMES, params.ClampMask, selMode, vselRepeatParams);
        }
        if (params.vcmpvsRepeatTail) {
            ClampComputeCount<T>(dstTensor[i * params.clampSplitCount + params.vcmpvsMainRepeatOffset],
                srcTensor[i * params.clampSplitCount + params.vcmpvsMainRepeatOffset], sharedTmpBuffer, scalar,
                params.vcmpvsRepeatTail, params.ClampMask, selMode, vselRepeatParams);
        }
    }
    // calcTail is a tail block smaller than clampSplitCount：
    // 1. calcTail < 128 , vcmpvs_lt calculate 128 element, select mask=calcTail；
    // 2. 128 <= calcTail < clampSplitCount：
    //   1) main, vcmpvs_lt calculate 128 element, select mask=128；
    //   2) tail, vcmpvs_lt calculate 128 element, select mask=tailElement；
    uint32_t mainCount = params.loopCount * params.clampSplitCount;
    if (params.calcTail > 0) {
        GetTailParams<T>(params.calcTail, params);
        for (uint32_t j = 0; j < params.selectTailRepeatLoop; j++) {
            ClampComputeCount<T>(dstTensor[mainCount + j * params.selectTailPreRepeatOffset],
                srcTensor[mainCount + j * params.selectTailPreRepeatOffset], sharedTmpBuffer, scalar, MAX_REPEAT_TIMES,
                params.ClampMask, selMode, vselRepeatParams);
        }
        if (params.selectTailRepeatTail) {
            ClampComputeCount<T>(dstTensor[mainCount + params.selectTailMainRepeatOffset],
                srcTensor[mainCount + params.selectTailMainRepeatOffset], sharedTmpBuffer, scalar,
                params.selectTailRepeatTail, params.ClampMask, selMode, vselRepeatParams);
        }
        if (params.selectTailElement) {
            ClampComputeCount<T>(dstTensor[mainCount + params.selectTailTailRepeatOffset],
                srcTensor[mainCount + params.selectTailTailRepeatOffset], sharedTmpBuffer, scalar, 1,
                params.selectTailElement, selMode, vselRepeatParams);
        }
    }
}
/* **************************************************************************************************
 * ClampMax                                           *
 * ************************************************************************************************* */
#pragma begin_pipe(V)
template <typename T, bool isReuseSource = false>
__aicore__ inline void ClampMaxImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const T scalar, const uint32_t calCount)
{
    ClampParams params;
    ClampCompute<T>(dstTensor, srcTensor, sharedTmpBuffer, scalar, calCount, CLAMPMODE::CLAMP_MAX, params);
}

/* **************************************************************************************************
 * ClampMin                                           *
 * ************************************************************************************************* */

template <typename T, bool isReuseSource = false>
__aicore__ inline void ClampMinImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const T scalar, const uint32_t calCount)
{
    ClampParams params;
    ClampCompute<T>(dstTensor, srcTensor, sharedTmpBuffer, scalar, calCount, CLAMPMODE::CLAMP_MIN, params);
}
} // namespace AscendC
#endif // IMPL_MATH_CLAMP_CLAMP_COMMON_IMPL_H