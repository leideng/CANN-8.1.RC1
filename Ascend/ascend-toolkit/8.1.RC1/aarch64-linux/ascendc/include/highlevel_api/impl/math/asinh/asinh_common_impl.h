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
 * \file asinh_common_impl.h
 * \brief
 */

#ifndef IMPL_MATH_ASINH_ASINH_COMMON_IMPL_H
#define IMPL_MATH_ASINH_ASINH_COMMON_IMPL_H
#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../common/check.h"

namespace AscendC {
constexpr uint32_t ASINH_HALF_CALC_PROC = 3;
constexpr uint32_t ASINH_FLOAT_CALC_PROC = 3;
constexpr float ASINH_ONE = 1;
constexpr float ASINH_FP16_MAX = 32768;                 // 2^15
constexpr float ASINH_FP16_MIN = 3.0517578125e-05;      // 2^-15
constexpr float ASINH_FP32_MAX = 4611686018427387904;   // 2^62
constexpr float ASINH_FP32_MIN = 2.168404344971009e-19; // 2^-62
constexpr uint32_t ASINH_STRIDE_DIGITS = 2;

template <typename T, bool isReuseSource = false>
__aicore__ inline void AsinhImpl(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    // tmpbuf
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ret = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ret), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    AsinhImpl<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, srcTensor.GetSize());
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void AsinhImpl(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
    const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    // tmpbuf
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ret = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ret), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    AsinhImpl<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, calCount);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void AsinhImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");
    
    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Asinh");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Asinh");

    ASCENDC_ASSERT((std::is_same<T, float>::value || std::is_same<T, half>::value), {
        KERNEL_LOG( KERNEL_ERROR, "Failed to check the data types, current api support data types are half/float."); });
    uint32_t tmpBufferSize = sharedTmpBuffer.GetSize();
    uint32_t splitCount = tmpBufferSize / sizeof(float);
    LocalTensor<float> tmpBuffer = sharedTmpBuffer.ReinterpretCast<float>();
    tmpBuffer.SetSize(sharedTmpBuffer.GetSize() / sizeof(float));

    if constexpr (sizeof(T) == sizeof(half)) {
        splitCount = splitCount / ASINH_HALF_CALC_PROC / ONE_BLK_SIZE * ONE_BLK_SIZE;;
    } else {
        splitCount = splitCount / ASINH_FLOAT_CALC_PROC / ONE_BLK_SIZE * ONE_BLK_SIZE;;
    }
    CheckTmpBufferSize(splitCount, 0, tmpBufferSize);

    uint32_t loopCount = calCount / splitCount;
    uint32_t calcTail = calCount % splitCount;
    SetMaskCount();
    SetVectorMask<T>(0, splitCount);
    for (uint32_t i = 0; i < loopCount; ++i) {
        AsinhCompute(dstTensor[i * splitCount], srcTensor[i * splitCount], tmpBuffer, splitCount);
    }
    if (calcTail > 0) {
        uint32_t tailCount = calcTail / ONE_BLK_SIZE * ONE_BLK_SIZE;
        tailCount = (calcTail % ONE_BLK_SIZE == 0) ? tailCount : (tailCount + ONE_BLK_SIZE);
        SetVectorMask<T>(0, calcTail);
        AsinhCompute(dstTensor[loopCount * splitCount], srcTensor[loopCount * splitCount], tmpBuffer, tailCount);
    }
    SetMaskNorm();
    ResetMask();
}

template <typename T> __aicore__ inline void AsinhGetSign(const LocalTensor<T> &dst, const LocalTensor<T> &src,
    const LocalTensor<T> &denominator)
{
    UnaryRepeatParams unaryParams;
    BinaryRepeatParams binaryParams;
    constexpr float kFpMax = sizeof(T) == sizeof(float) ? ASINH_FP32_MAX : ASINH_FP16_MAX;
    constexpr float kFpMin = sizeof(T) == sizeof(float) ? ASINH_FP32_MIN : ASINH_FP16_MIN;
    Muls<T, false>(dst, src, static_cast<T>(kFpMax), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Abs<T, false>(denominator, dst, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Adds<T, false>(denominator, denominator, static_cast<T>(kFpMin), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Div<T, false>(dst, dst, denominator, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void AsinhCompute(const LocalTensor<T> &dst, const LocalTensor<T> &src,
    const LocalTensor<float> &tmpBuffer, uint32_t calCount)
{
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;
    // asinh(x) = ln(x + sqrt(x^2 + 1))
    // asinh(x) = -asinh(-x)
    // Calculate the amount of data that can be stored in the temporary space and split the data into the entire block
    // and tail block.

    LocalTensor<float> tmpFloatBuffer1 = tmpBuffer;
    LocalTensor<float> tmpFloatBuffer3 = tmpFloatBuffer1[calCount];
    LocalTensor<float> tmpFloatBuffer2 = tmpFloatBuffer1[calCount * 2];

    // abs(x)
    Abs<T, false>(tmpFloatBuffer1, src, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // x^2
    Mul<T, false>(tmpFloatBuffer2, tmpFloatBuffer1, tmpFloatBuffer1, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    // x^2 + 1
    Adds<T, false>(tmpFloatBuffer2, tmpFloatBuffer2, static_cast<T>(ASINH_ONE), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // sqrt(x^2 + 1)
    Sqrt<T, false>(tmpFloatBuffer2, tmpFloatBuffer2, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // x + sqrt(x^2 + 1)
    Add<T, false>(tmpFloatBuffer2, tmpFloatBuffer2, tmpFloatBuffer1, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    // ln(x + sqrt(x^2 + 1))
    Ln<T, false>(tmpFloatBuffer2, tmpFloatBuffer2, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // asinh(x) = -asinh(-x)
    AsinhGetSign(tmpFloatBuffer1, src, tmpFloatBuffer3);
    PipeBarrier<PIPE_V>();

    Mul<T, false>(dst, tmpFloatBuffer2, tmpFloatBuffer1, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
}

template <>
__aicore__ inline void AsinhCompute(const LocalTensor<half> &dst, const LocalTensor<half> &src,
    const LocalTensor<float> &tmpBuffer, uint32_t calCount)
{
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;

    LocalTensor<float> tmpFloatBuffer1 = tmpBuffer;
    LocalTensor<float> tmpFloatBuffer3 = tmpFloatBuffer1[calCount];
    LocalTensor<float> tmpFloatBuffer2 = tmpFloatBuffer1[calCount * 2];

    // asinh(x) = ln(x + sqrt(x^2 + 1))
    // asinh(x) = -asinh(-x)
    // Calculate the amount of data that can be stored in the temporary space and split the data into the entire block
    // and tail block
    Cast<float, half, false>(tmpFloatBuffer1, src, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();

    // abs(x)
    Abs<float, false>(tmpFloatBuffer1, tmpFloatBuffer1, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // x^2
    Mul<float, false>(tmpFloatBuffer2, tmpFloatBuffer1, tmpFloatBuffer1, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    // x^2 + 1
    Adds<float, false>(tmpFloatBuffer2, tmpFloatBuffer2, static_cast<half>(ASINH_ONE), MASK_PLACEHOLDER,
        1, unaryParams);
    PipeBarrier<PIPE_V>();

    // sqrt(x^2 + 1)
    Sqrt<float, false>(tmpFloatBuffer2, tmpFloatBuffer2, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // x + sqrt(x^2 + 1)
    Add<float, false>(tmpFloatBuffer2, tmpFloatBuffer2, tmpFloatBuffer1, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    // ln(x + sqrt(x^2 + 1))
    Ln<float, false>(tmpFloatBuffer2, tmpFloatBuffer2, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Cast<float, half, false>(tmpFloatBuffer1, src, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();

    // asinh(x) = -asinh(-x)
    AsinhGetSign(tmpFloatBuffer1, tmpFloatBuffer1, tmpFloatBuffer3);
    PipeBarrier<PIPE_V>();

    // Revert Symbol
    Mul<float, false>(tmpFloatBuffer2, tmpFloatBuffer2, tmpFloatBuffer1, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Cast<half, float, false>(dst, tmpFloatBuffer2, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE / ASINH_STRIDE_DIGITS, DEFAULT_REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();
}
}
#endif