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
 * \file atan_common_impl.h
 * \brief
 */
#ifndef IMPL_MATH_ATAN_ATAN_COMMON_IMPL_H
#define IMPL_MATH_ATAN_ATAN_COMMON_IMPL_H
#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "../../common/check.h"

#include "../math_common_impl.h"

namespace AscendC {
constexpr uint8_t ATAN_HALF_CALC_PROCEDURE = 6;
constexpr uint8_t ATAN_FLOAT_CALC_PROCEDURE = 5;
constexpr float ATAN_FP16_MAX = 32768;                 // 2^15
constexpr float ATAN_FP16_MIN = 3.0517578125e-05;      // 2^-15
constexpr float ATAN_FP32_MAX = 4611686018427387904;   // 2^62
constexpr float ATAN_FP32_MIN = 2.168404344971009e-19; // 2^-62
constexpr uint8_t TAYLOR_COUNT_FOUR = 4; // x belongs to (0, tan(pi/8))
constexpr uint8_t TAYLOR_COUNT_SIX = 6; // x belongs to (tan(pi/8), tan(pi/4))
constexpr float MIN_INPUT_VALUE = -10000;
constexpr float MAX_INPUT_VALUE = 10000;

// Calculates the Sign of given values.
// Algorithm:
//         FP16: sign(x) = 2**(15) * x /(2**(-15) + 2**(15) *|x|)
//         FP32: sign(x) = 2**(62) * x /(2**(-62) + 2**(62) *|x|)
template <typename T>
__aicore__ inline void Sign(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<T>& denominator)
{
    UnaryRepeatParams unaryParams;
    BinaryRepeatParams binaryParams;
    constexpr float kFpMax = sizeof(T) == sizeof(float) ? ATAN_FP32_MAX : ATAN_FP16_MAX;
    constexpr float kFpMin = sizeof(T) == sizeof(float) ? ATAN_FP32_MIN : ATAN_FP16_MIN;
    Muls<T, false>(dst, src, static_cast<T>(kFpMax), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Abs<T, false>(denominator, dst, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Adds<T, false>(denominator, denominator, static_cast<T>(kFpMin), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Div<T, false>(dst, dst, denominator, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void TaylorExpand(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
    const LocalTensor<float>& squareTensor, int32_t expandLevel)
{
    // arctan(x) = x - x^3/3 + x^5/5 + ... + (-1)^k*x^(k*2+1)/( k*2+1)
    // 1/(k*2+1)
    constexpr float factorList[7] = {1, -0.3333333333333333, 0.2, -0.14285714285714285,
        0.1111111111111111, - 0.09090909090909091, 0.07692307692307693};
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;
    // The initial value of dstTensor is assigned as the coefficient of the last item of expansion.
    Mul<float, false>(squareTensor, srcTensor, srcTensor, MASK_PLACEHOLDER, 1, binaryParams);
    Mul<float, false>(dstTensor, srcTensor, srcTensor, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
    Muls<float, false>(dstTensor, dstTensor, factorList[expandLevel], MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    for (uint32_t i = expandLevel - 1; i > 0; --i) {
        // dst*x^2+ the previois expand factor
        Adds<float, false>(dstTensor, dstTensor, factorList[i], MASK_PLACEHOLDER, 1, unaryParams);
        PipeBarrier<PIPE_V>();
        Mul<float, false>(dstTensor, dstTensor, squareTensor, MASK_PLACEHOLDER, 1, binaryParams);
        PipeBarrier<PIPE_V>();
    }
    Adds<float, false>(dstTensor, dstTensor, factorList[0], MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Mul<float, false>(dstTensor, dstTensor, srcTensor, MASK_PLACEHOLDER, 1, binaryParams);
}

__aicore__ inline void AtanTransform(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
    const LocalTensor<float>& tmpTensor, const float transFactor)
{
    // (x-y)/(1+xy)
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binParams;
    const float transFactorNeg = 0 - transFactor;

    // x*y
    Muls<float, false>(dstTensor, srcTensor, transFactor, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    // x*y + 1
    Adds<float, false>(dstTensor, dstTensor, static_cast<float>(1.0), MASK_PLACEHOLDER, 1, unaryParams);
    // x=x-y
    Adds<float, false>(tmpTensor, srcTensor, transFactorNeg, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    // (x-y)/(1+xy)
    Div<float, false>(dstTensor, tmpTensor, dstTensor, MASK_PLACEHOLDER, 1, binParams);
    PipeBarrier<PIPE_V>();
    Abs<float, false>(dstTensor, dstTensor, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
}

//  when x < 0, Atan(x) = atan(-x)
//  when x belongs to (0, tan(pi/8)), Atan(x) = atan(x)
//  when x belongs to (tan(pi/8), tan(pi/4)), Atan(x) = pi/8 + atan((x- tan(pi/8)) / (1+ x*tan(pi/8)))
//  when x belongs to (tan(pi/4), +∞), Atan(x) = pi/4 + atan((x-1)/(x+1))
__aicore__ inline void AtanFormulaImpl(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
    const LocalTensor<float>& tmpTensor, const uint32_t splitSize)
{
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binParams;
    const float piByFour = 0.78539816339744830961566084581988;
    const float piByEight = 0.39269908169872415480783042290994;
    const float tanPiByEight = 0.4142135623730950;
    LocalTensor<float> clipTensor = tmpTensor[splitSize];
    LocalTensor<float> absTensor = clipTensor[splitSize];
    LocalTensor<float> tmpTensor2 = absTensor[splitSize];
    LocalTensor<float> squareTensor = tmpTensor2[splitSize];

    // when x's value is too large the first caculator of TaylorExpand will be overflow. when epsilon is 0.0001,
    // the approximate value of `tan(pi/2 - 0.0001)` is 10000
    // Clip x to [MIN_INPUT_VALUE, MAX_INPUT_VALUE] in float
    Mins<float, false>(clipTensor, srcTensor, MAX_INPUT_VALUE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Maxs<float, false>(clipTensor, clipTensor, MIN_INPUT_VALUE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Abs<float, false>(absTensor, clipTensor, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // 1. x in (0, tan(pi/8))
    TaylorExpand(dstTensor, absTensor, squareTensor, TAYLOR_COUNT_FOUR);

    // 2. x in (tan(pi/8), tan(pi/4)), atan(x) = pi/8 + atan((x-tan(pi/8)) / (1 + x*tan(pi/8)))
    // normalize x to (0, tan(pi/8))
    AtanTransform(tmpTensor, absTensor, tmpTensor2, tanPiByEight); // tan(pi/8)
    TaylorExpand(tmpTensor2, tmpTensor, squareTensor, TAYLOR_COUNT_FOUR);
    PipeBarrier<PIPE_V>();
    Adds<float, false>(tmpTensor2, tmpTensor2, piByEight, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Min<float, false>(dstTensor, dstTensor, tmpTensor2, MASK_PLACEHOLDER, 1, binParams);

    // x in (tan(pi/4), +∞), atan(x) = pi/4 + atan((x-1)/(x+1))
    // calculate |(x-1)/(x+1)|, normalize x to (0, tan(pi/4))
    // find the minimum value between atan(|(x-1)/(x+1)|) calculate in (0, tan(pi/8)) and (tan(pi/8), tan(pi/4))
    Adds<float, false>(tmpTensor2, absTensor, static_cast<float>(1.0), MASK_PLACEHOLDER, 1, unaryParams);
    Adds<float, false>(tmpTensor, absTensor, -static_cast<float>(1.0), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Div<float, false>(tmpTensor, tmpTensor, tmpTensor2, MASK_PLACEHOLDER, 1, binParams);
    PipeBarrier<PIPE_V>();
    Abs<float, false>(tmpTensor, tmpTensor, MASK_PLACEHOLDER, 1, unaryParams); // take the absolute value
    PipeBarrier<PIPE_V>();

    // 3. atan(|(x-1)/(x+1)|)
    TaylorExpand(tmpTensor2, tmpTensor, squareTensor, TAYLOR_COUNT_FOUR);
    PipeBarrier<PIPE_V>();
    // pi/4 + atan(|(x-1)/(x+1)|)
    Adds<float, false>(tmpTensor2, tmpTensor2, piByFour, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Min<float, false>(dstTensor, dstTensor, tmpTensor2, MASK_PLACEHOLDER, 1, binParams);

    // 4.reuse the transform result in step 3, and calculate (x-tan(pi/8)) / (1 + x*tan(pi/8))
    AtanTransform(tmpTensor2, tmpTensor, squareTensor, tanPiByEight);
    TaylorExpand(tmpTensor, tmpTensor2, squareTensor, TAYLOR_COUNT_SIX);
    PipeBarrier<PIPE_V>();
    // pi/8 + pi/4 + atan((x-tan(pi/8)) / (1 + x*tan(pi/8)))
    Adds<float, false>(tmpTensor, tmpTensor, piByEight, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Adds<float, false>(tmpTensor, tmpTensor, piByFour, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Min<float, false>(dstTensor, dstTensor, tmpTensor, MASK_PLACEHOLDER, 1, binParams);
    PipeBarrier<PIPE_V>();

    Sign(tmpTensor, clipTensor, tmpTensor2);

    // dst = sign(x) * dst.
    Mul<float, false>(dstTensor, dstTensor, tmpTensor, MASK_PLACEHOLDER, 1, binParams);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void AtanCompute(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<float>& tmpTensor, const uint32_t splitSize)
{
    AtanFormulaImpl(dstTensor, srcTensor, tmpTensor, splitSize);
}

template <>
__aicore__ inline void AtanCompute(const LocalTensor<half>& dstTensor, const LocalTensor<half>& srcTensor,
    const LocalTensor<float>& tmpTensor, const uint32_t splitSize)
{
    const LocalTensor<float>& tempTensorConv = tmpTensor[splitSize * 5];
    Cast<float, half, false>(tempTensorConv, srcTensor,
        RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
    AtanFormulaImpl(tempTensorConv, tempTensorConv, tmpTensor, splitSize);
    Cast<half, float, false>(dstTensor, tempTensorConv,
        RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, { 1, 1, HALF_DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void AtanImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");

    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Atan");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Atan");

    ASCENDC_ASSERT((std::is_same<T, float>::value || std::is_same<T, half>::value), {
        KERNEL_LOG( KERNEL_ERROR, "Failed to check the data types, current api support data types are half/float."); });
    const uint32_t bufferSize = sharedTmpBuffer.GetSize();
    const uint32_t tmpBufferSize = bufferSize / sizeof(float);
    CheckTmpBufferSize(tmpBufferSize, 0, bufferSize);
    // all temporary variables are float type.
    LocalTensor<float> tmpBuffer = sharedTmpBuffer.ReinterpretCast<float>();

    uint32_t calSize = 0;
    if constexpr (sizeof(T) == sizeof(half)) {
        calSize = tmpBufferSize / ATAN_HALF_CALC_PROCEDURE / ONE_BLK_SIZE * ONE_BLK_SIZE;
    } else {
        calSize = tmpBufferSize / ATAN_FLOAT_CALC_PROCEDURE / ONE_BLK_SIZE * ONE_BLK_SIZE;
    }

    CheckTmpBufferSize(calSize, 0, bufferSize);

    const uint32_t round = calCount / calSize;
    const uint32_t tail = calCount % calSize;

    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(0, calSize);

    uint32_t offset = 0;
    for (uint32_t i = 0; i < round; i++) {
        AtanCompute(dstTensor[offset], srcTensor[offset], tmpBuffer, calSize);
        offset = offset + calSize;
    }

    if (tail != 0) {
        SetVectorMask<T, MaskMode::COUNTER>(0, tail);
        AtanCompute(dstTensor[offset], srcTensor[offset], tmpBuffer, calSize);
    }

    SetMaskNorm();
    ResetMask();
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void AtanImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
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
    AtanImpl(dstTensor, srcTensor, sharedTmpBuffer, calCount);
}
}  // namespace AscendC

#endif  // IMPL_MATH_ATAN_ATAN_COMMON_IMPL_H