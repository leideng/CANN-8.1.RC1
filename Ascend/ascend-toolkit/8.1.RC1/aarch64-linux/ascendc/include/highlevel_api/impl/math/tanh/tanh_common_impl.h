/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tanh_common_impl.h
 * \brief
 */
#ifndef IMPL_MATH_TANH_TANH_COMMON_IMPL_H
#define IMPL_MATH_TANH_TANH_COMMON_IMPL_H

#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "../../common/check.h"

#include "../math_common_impl.h"

namespace AscendC {
constexpr float FP32_MIN_V2 = -8.8;
constexpr float FP32_MAX_V2 = 8.8;
constexpr float DOUBLE_X = 2;
const uint8_t TANH_HALF_CALC_PROCEDURE = 2;
const uint8_t TANH_FLOAT_CALC_PROCEDURE = 1;
/*
 * Formula is y= (e^(2x)-1)/(e^(2x)+1)
 */
__aicore__ inline void TanhFormulaImpl(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
    const TanhParams<float>& params)
{
    const LocalTensor<float>& tmpClip = params.tmpClip;
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;

    // Clip x to [FP32_MIN_V2, FP32_MAX_V2] in float
    Mins<float, false>(tmpClip, srcTensor, FP32_MAX_V2, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Maxs<float, false>(tmpClip, tmpClip, FP32_MIN_V2, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    // 2 * x
    Muls<float, false>(tmpClip, tmpClip, DOUBLE_X, MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    PipeBarrier<PIPE_V>();

    // e^(2 * x)
    Exp<float, false>(tmpClip, tmpClip, MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    PipeBarrier<PIPE_V>();

    // e^(2 * x) - 1
    Adds<float, false>(dstTensor, tmpClip, -1.0, MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    PipeBarrier<PIPE_V>();

    // e^(2 * x) + 1
    Adds<float, false>(tmpClip, tmpClip, 1.0, MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    PipeBarrier<PIPE_V>();

    Div<float, false>(dstTensor, dstTensor, tmpClip, MASK_PLACEHOLDER, params.repeatTimes, binaryParams);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void TanhCompute(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const TanhParams<float>& params)
{
    TanhFormulaImpl(dstTensor, srcTensor, params);
}

template <>
__aicore__ inline void TanhCompute(const LocalTensor<half>& dstTensor, const LocalTensor<half>& srcTensor,
    const TanhParams<float>& params)
{
    const LocalTensor<float>& tempTensorConv = params.tempTensorConv;
    Cast<float, half, false>(tempTensorConv, srcTensor,
        RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();

    TanhFormulaImpl(tempTensorConv, tempTensorConv, params);

    Cast<half, float, false>(dstTensor, tempTensorConv,
        RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, { 1, 1, HALF_DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void TanhFormulasTmpCalc(TanhParams<float>& params, uint32_t tmpBufferSize)
{
    uint32_t tmpUbIndex = 0;
    if constexpr (sizeof(T) == sizeof(half)) {
        params.stackSize = params.tmpBufferSize / TANH_HALF_CALC_PROCEDURE / ONE_BLK_SIZE * ONE_BLK_SIZE;
    } else {
        params.stackSize = params.tmpBufferSize / TANH_FLOAT_CALC_PROCEDURE / ONE_BLK_SIZE * ONE_BLK_SIZE;
    }
    CheckTmpBufferSize(params.stackSize, 0, tmpBufferSize);
    if constexpr (sizeof(T) == sizeof(half)) {
        params.tempTensorConv = params.sharedTmpBuffer[params.stackSize * (tmpUbIndex++)];
    }
    params.tmpClip = params.sharedTmpBuffer[params.stackSize * (tmpUbIndex++)];
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void TanhImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");

    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Tanh");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Tanh");

    ASCENDC_ASSERT((std::is_same<T, float>::value || std::is_same<T, half>::value), {
        KERNEL_LOG( KERNEL_ERROR, "Failed to check the data types, current api support data types are half/float."); });

    TanhParams<float> params;
    params.calCount = calCount;
    uint32_t tmpBufferSize = sharedTmpBuffer.GetSize();
    params.tmpBufferSize = tmpBufferSize / sizeof(float);
    CheckTmpBufferSize(params.tmpBufferSize, 0, tmpBufferSize);

    // all temporary variables are float type.
    params.sharedTmpBuffer = sharedTmpBuffer.ReinterpretCast<float>();

    TanhFormulasTmpCalc<T>(params, tmpBufferSize);

    const uint32_t round = params.calCount / params.stackSize;
    const uint32_t tail = params.calCount % params.stackSize;

    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(0, params.stackSize);

    uint32_t offset = 0;
    for (uint32_t i = 0; i < round; i++) {
        TanhCompute(dstTensor[offset], srcTensor[offset], params);
        offset = offset + params.stackSize;
    }

    if (tail != 0) {
        SetVectorMask<T, MaskMode::COUNTER>(0, tail);
        TanhCompute(dstTensor[offset], srcTensor[offset], params);
    }

    SetMaskNorm();
    ResetMask();
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void TanhImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    // Using the Stack Space to Allocate tmpBuffer
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    TanhImpl(dstTensor, srcTensor, sharedTmpBuffer, calCount);
}
} // namespace AscendC

#endif // IMPL_MATH_TANH_TANH_COMMON_IMPL_H