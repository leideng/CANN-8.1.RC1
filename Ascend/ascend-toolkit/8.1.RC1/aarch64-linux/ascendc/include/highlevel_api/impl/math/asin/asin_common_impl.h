/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef IMPL_MATH_ASIN_ASIN_COMMON_IMPL_H
#define IMPL_MATH_ASIN_ASIN_COMMON_IMPL_H
#include "kernel_tensor.h"
#include "../math_constant_util.h"
#include "../../common/check.h"

namespace AscendC {
constexpr uint8_t ASIN_HALF_CALC_PROCEDURE = 6;
constexpr uint8_t ASIN_FLOAT_CALC_PROCEDURE = 4;
constexpr uint32_t ASIN_TAYLOR_EXPAND_COUNT = 7;
// Coefficient values of taylor expansion of asin.
constexpr float kCOEF[] = {
    1.0,
    0.16666666666666666666666666666667,
    0.075,
    0.04464285714285714285714285714286,
    0.03038194444444444444444444444444,
    0.02237215909090909090909090909091,
    0.01735276442307692307692307692308,
    0.01396484375,
};

// Calculates the Sign of given values.
// Algorithm:
// FP16: sign(x) = 2^(15) * x /(2^(-15) + 2^(15) *|x|)
// FP32: sign(x) = 2^(62) * x /(2^(-62) + 2^(62) *|x|)
template <typename T>
__aicore__ inline void GetSign(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<T>& denominator)
{
    UnaryRepeatParams unaryParams;
    BinaryRepeatParams binaryParams;
    constexpr float FP16_MAX = 32768;                 // 2^15
    constexpr float FP16_MIN = 3.0517578125e-05;      // 2^-15
    constexpr float FP32_MAX = 4611686018427387904;   // 2^62
    constexpr float FP32_MIN = 2.168404344971009e-19; // 2^-62
    constexpr float kFpMax = sizeof(T) == sizeof(float) ? FP32_MAX : FP16_MAX;
    constexpr float kFpMin = sizeof(T) == sizeof(float) ? FP32_MIN : FP16_MIN;
    Muls<T, false>(dst, src, static_cast<T>(kFpMax), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Abs<T, false>(denominator, dst, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Adds<T, false>(denominator, denominator, static_cast<T>(kFpMin), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Div<T, false>(dst, dst, denominator, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
}

// Calculate Taylor Expansion of Asin by using extra Buffer and not modifying source.
template <typename T>
__aicore__ inline void AsinTaylorCompute(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<T>& localTemp)
{
    BinaryRepeatParams binaryParams;
    UnaryRepeatParams unaryParams;
    // Calculates Taylor Expansion according to (((k_nx^2 + k_n) * x^2 + k_(n-1)) * x^2 +k_(n-2) ……)*x^2 +k_0)*x.
    Mul<T, false>(dst, src, src, MASK_PLACEHOLDER, 1, binaryParams);
    Mul<T, false>(localTemp, src, src, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
    Muls<T, false>(dst, dst, static_cast<T>(kCOEF[ASIN_TAYLOR_EXPAND_COUNT]), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    for (uint32_t i = ASIN_TAYLOR_EXPAND_COUNT - 1; i > 0; i--) {
        // Accumlates nth Taylor Expansion item.
        Adds<T, false>(dst, dst, static_cast<T>(kCOEF[i]), MASK_PLACEHOLDER, 1, unaryParams);
        PipeBarrier<PIPE_V>();
        Mul<T, false>(dst, dst, localTemp, MASK_PLACEHOLDER, 1, binaryParams);
        PipeBarrier<PIPE_V>();
    }
    Adds<T, false>(dst, dst, static_cast<T>(kCOEF[0]), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Mul<T, false>(dst, dst, src, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
}

// Calculate Taylor Expansion of Asin based on its square value, and finally setting the source to be sqrt(x).
template <typename T>
__aicore__ inline void AsinTaylorComputeBySquareValue(const LocalTensor<T>& dst, const LocalTensor<T>& src)
{
    BinaryRepeatParams binaryParams;
    UnaryRepeatParams unaryParams;
    // Calculates Taylor Expansion according to (((k_nx^2 + k_n) * x^2 + k_(n-1)) * x^2 +k_(n-2) ……)*x^2 +k_0)*x.
    Muls<T, false>(dst, src, static_cast<T>(NUM_ONE), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Muls<T, false>(dst, dst, static_cast<T>(kCOEF[ASIN_TAYLOR_EXPAND_COUNT]), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    for (uint32_t i = ASIN_TAYLOR_EXPAND_COUNT - 1; i > 0; i--) {
        // Accumlates nth Taylor Expansion item.
        Adds<T, false>(dst, dst, static_cast<T>(kCOEF[i]), MASK_PLACEHOLDER, 1, unaryParams);
        PipeBarrier<PIPE_V>();
        Mul<T, false>(dst, dst, src, MASK_PLACEHOLDER, 1, binaryParams);
        PipeBarrier<PIPE_V>();
    }
    Adds<T, false>(dst, dst, static_cast<T>(kCOEF[0]), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    // Updates src to be sqrt(x).
    Sqrt<T, false>(src, src, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Mul<T, false>(dst, dst, src, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
}

// Computes high precision asin values for half type inputs by converting inputs to float types and save float asin
// result at tmp Buffer. Requires 6 times extra buffer of input sizes.
__aicore__ inline void AsinFp16Compute(const LocalTensor<half>& dst, const LocalTensor<half>& src,
    const LocalTensor<half>& stackBuffer, uint32_t calSize)
{
    UnaryRepeatParams unaryParams;
    BinaryRepeatParams binaryParams;

    // Temp space will be Splited into 6 parts, 4 for float calculation, 2 for half calculation
    const LocalTensor<float>& tmpFloatBuffer1 = stackBuffer.ReinterpretCast<float>();
    const LocalTensor<float>& tmpFloatBuffer2 = tmpFloatBuffer1[calSize];
    const uint32_t tmpHalfBuffer1Offset = calSize * 4;
    const uint32_t tmpHalfBuffer2Offset = calSize * 5;
    const LocalTensor<half>& tmpHalfBuffer1 = stackBuffer[tmpHalfBuffer1Offset];
    const LocalTensor<half>& tmpHalfBuffer2 = stackBuffer[tmpHalfBuffer2Offset];

    // Cast src from half to float type for getting more precise results, but only computes by finishing
    // taylor expansion computation as it's the majority reason of precision loss.
    Cast<float, half, false>(tmpFloatBuffer2, src, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
    // Calculates res2 = PI*0.5 - taylor_compute(sqrt(1 - x^2)) -> tmpFloatBuffer1.
    Mul<float, false>(tmpFloatBuffer2, tmpFloatBuffer2, tmpFloatBuffer2, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
    Muls<float, false>(tmpFloatBuffer2, tmpFloatBuffer2, NEG_ONE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Adds<float, false>(tmpFloatBuffer2, tmpFloatBuffer2, NUM_ONE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    AsinTaylorComputeBySquareValue(tmpFloatBuffer1, tmpFloatBuffer2);
    PipeBarrier<PIPE_V>();
    Muls<float, false>(tmpFloatBuffer1, tmpFloatBuffer1, NEG_ONE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Adds<float, false>(tmpFloatBuffer1, tmpFloatBuffer1, HALF_PI, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // Calculates res1 = taylor_compute(abs(x)) -> dst.
    Abs<half, false>(tmpHalfBuffer2, src, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    AsinTaylorCompute(dst, tmpHalfBuffer2, tmpHalfBuffer1);
    PipeBarrier<PIPE_V>();

    // As NPU are not good at scalar process like CPU for if-else statement, the solution here used for handling above
    // 3 scenarios is to calculates 0/1 choices combining the results on both options.
    // e.g.
    // Step1: Calculates both option results of x, no matter which range it's at.
    // result1(x), result2(x)
    // Step2: Calculates 0/1 choices of both option results of x, no matter which range it's at.
    // choice1(x), choice2(x)
    // Step3: Combines choice result and options results, since at least one choice should be zero.
    // Result = choice1(x) * result1(x) + choice2(x) * result2(x)
    // choice_1 = -Floor(min(abs(x), BOUNDARY) - BOUNDARY).
    // choice_2 = 1 - choice_1
    // res = res_1 * choice_1 + res_2 * choice_2
    Mins<half, false>(tmpHalfBuffer2, tmpHalfBuffer2, static_cast<half>(BOUNDARY), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Adds<half, false>(tmpHalfBuffer2, tmpHalfBuffer2, static_cast<half>(-BOUNDARY), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    const LocalTensor<int8_t>& tmpS8Buffer = tmpHalfBuffer1.ReinterpretCast<int8_t>();
    Cast<int8_t, half, false>(tmpS8Buffer, tmpHalfBuffer2, RoundMode::CAST_FLOOR, MASK_PLACEHOLDER, 1,
        { 1, 1, HALF_DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
    Cast<half, int8_t, false>(tmpHalfBuffer2, tmpS8Buffer, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();

    Muls<half, false>(tmpHalfBuffer2, tmpHalfBuffer2, static_cast<half>(NEG_ONE), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Mul<half, false>(dst, dst, tmpHalfBuffer2, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Muls<half, false>(tmpHalfBuffer2, tmpHalfBuffer2, static_cast<half>(NEG_ONE), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Adds<half, false>(tmpHalfBuffer2, tmpHalfBuffer2, static_cast<half>(NUM_ONE), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Cast<float, half, false>(tmpFloatBuffer2, tmpHalfBuffer2, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
    Mul<float, false>(tmpFloatBuffer1, tmpFloatBuffer1, tmpFloatBuffer2, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Cast<float, half, false>(tmpFloatBuffer2, dst, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();

    // Calculates res1 = res1 + res2.
    Add<float, false>(tmpFloatBuffer1, tmpFloatBuffer2, tmpFloatBuffer1, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
    // Restores sign of asin.
    // res1 = sign(x) * res1.
    GetSign(tmpHalfBuffer1, src, tmpHalfBuffer2);
    PipeBarrier<PIPE_V>();
    Cast<float, half, false>(tmpFloatBuffer2, tmpHalfBuffer1, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
    Mul<float, false>(tmpFloatBuffer1, tmpFloatBuffer1, tmpFloatBuffer2, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
    Cast<half, float, false>(dst, tmpFloatBuffer1, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        { 1, 1, HALF_DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
}

// Computes asin values based on input types.
// Requires 2 times extra buffer of input sizes.
// asin(x) = arcsin(sqrt(1-x^2)) - PI*0.5, x belongs to (-1, -2^(-0.5))
// asin(x) = the 15th order taylor expansion, x belongs to (-2^(-0.5), 2^(-0.5))
// asin(x) = PI*0.5 - arcsin(sqrt(1-x^2)), x belongs to (2^(-0.5), 1)
template <typename T>
__aicore__ inline void AsinCompute(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<T>& tmpBuffer, uint32_t calSize)
{
    UnaryRepeatParams unaryParams;
    BinaryRepeatParams binaryParams;

    const LocalTensor<T>& tmpBuffer2 = tmpBuffer[calSize];
    const LocalTensor<T>& dupBuffer = tmpBuffer[calSize * 2];
    // Calculates res2 = PI*0.5 - taylor_compute(sqrt(1 - x^2)) -> tmpBuffer2.
    Mul<T, false>(tmpBuffer2, src, src, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
    Muls<T, false>(tmpBuffer2, tmpBuffer2, NEG_ONE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Adds<T, false>(tmpBuffer2, tmpBuffer2, NUM_ONE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Sqrt<T, false>(dst, tmpBuffer2, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    AsinTaylorCompute(tmpBuffer2, dst, tmpBuffer);
    PipeBarrier<PIPE_V>();
    Muls<T, false>(tmpBuffer2, tmpBuffer2, NEG_ONE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Adds<T, false>(tmpBuffer2, tmpBuffer2, HALF_PI, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // Calculates res1 = taylor_compute(abs(x)) -> dst, abs(x) -> tmpBuffer.
    Mul<T, false>(tmpBuffer, src, src, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
    AsinTaylorComputeBySquareValue(dst, tmpBuffer);
    PipeBarrier<PIPE_V>();

    // As NPU are not good at scalar process like CPU for if-else statement, the solution here used for handling above
    // 3 scenarios is to calculates 0/1 choices combining the results on both options.
    // e.g.
    // Step1: Calculates both option results of x, no matter which range it's at.
    // result1(x), result2(x)
    // Step2: Calculates 0/1 choices of both option results of x, no matter which range it's at.
    // choice1(x), choice2(x)
    // Step3: Combines choice result and options results, since at least one choice should be zero.
    // Result = choice1(x) * result1(x) + choice2(x) * result2(x)
    // choice_1 = -Floor(min(abs(x), BOUNDARY) - BOUNDARY).
    // choice_2 = 1 - choice_1
    // res = res_1 * choice_1 + res_2 * choice_2
    Mins<T, false>(tmpBuffer, tmpBuffer, BOUNDARY, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Adds<T, false>(tmpBuffer, tmpBuffer, -BOUNDARY, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    LocalTensor<int32_t> tmpS32Buffer = tmpBuffer.template ReinterpretCast<int32_t>();
    Cast<int32_t, T, false>(tmpS32Buffer, tmpBuffer, RoundMode::CAST_FLOOR, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Cast<T, int32_t, false>(tmpBuffer, tmpS32Buffer, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    PipeBarrier<PIPE_V>();
    Muls<T, false>(tmpBuffer, tmpBuffer, NEG_ONE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Mul<T, false>(dst, dst, tmpBuffer, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Muls<T, false>(tmpBuffer, tmpBuffer, NEG_ONE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Adds<T, false>(tmpBuffer, tmpBuffer, NUM_ONE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Mul<T, false>(tmpBuffer2, tmpBuffer2, tmpBuffer, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Add<T, false>(dst, dst, tmpBuffer2, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    // Restores sign of asin.
    // res1 = sign(x) * res1.
    GetSign(tmpBuffer2, src, tmpBuffer);
    PipeBarrier<PIPE_V>();
    Mul<T, false>(dst, dst, tmpBuffer2, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
}

template <>
__aicore__ inline void AsinCompute<half>(const LocalTensor<half>& dst, const LocalTensor<half>& src,
    const LocalTensor<half>& tmpBuffer, uint32_t calSize)
{
    // Due to using half type to computing the above formula, it doesn't statisify the precision standard,
    // upcast to float for getting more precise results.
    AsinFp16Compute(dst, src, tmpBuffer, calSize);
}

template <typename T>
__aicore__ inline void AsinImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");

    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Asin");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Asin");

    ASCENDC_ASSERT((std::is_same<T, float>::value || std::is_same<T, half>::value), {
        KERNEL_LOG( KERNEL_ERROR, "Failed to check the data types, current api support data types are half/float."); });
    uint32_t bufferSize = sharedTmpBuffer.GetSize();
    uint32_t tmpBufferSize = bufferSize / sizeof(T);
    CheckTmpBufferSize(tmpBufferSize, 0, bufferSize);

    LocalTensor<T> tmpBuffer = sharedTmpBuffer.ReinterpretCast<T>();

    // For float input, the temporary variable is 2 floats.
    // For half input, the temporary variable is 2 floats + 2 halfs. The memory is equivalent to 6 halfs.
    uint32_t calSize = 0;
    if constexpr (sizeof(T) == sizeof(float)) {
        calSize = tmpBufferSize / ASIN_FLOAT_CALC_PROCEDURE / ONE_BLK_SIZE * ONE_BLK_SIZE;
    } else {
        calSize = tmpBufferSize / ASIN_HALF_CALC_PROCEDURE / ONE_BLK_SIZE * ONE_BLK_SIZE;
    }
    CheckTmpBufferSize(calSize, 0, bufferSize);

    const uint32_t round = calCount / calSize;
    const uint32_t tail = calCount % calSize;

    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(0, calSize);

    uint32_t offset = 0;
    for (uint32_t i = 0; i < round; i++) {
        AsinCompute(dstTensor[offset], srcTensor[offset], tmpBuffer, calSize);
        offset = offset + calSize;
    }

    if (tail != 0) {
        SetVectorMask<T, MaskMode::COUNTER>(0, tail);
        AsinCompute(dstTensor[offset], srcTensor[offset], tmpBuffer, calSize);
    }

    SetMaskNorm();
    ResetMask();
}

template <typename T>
__aicore__ inline void AsinImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
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
    AsinImpl(dstTensor, srcTensor, sharedTmpBuffer, calCount);
}
} // namespace AscendC

#endif // IMPL_MATH_ASIN_ASIN_COMMON_IMPL_H
