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
 * \file tan_common_impl.h
 * \brief
 */
#ifndef IMPL_MATH_TAN_TAN_COMMON_IMPL_H
#define IMPL_MATH_TAN_TAN_COMMON_IMPL_H

#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "../../common/check.h"
#if __CCE_AICORE__ == 220
#include "tan_v220_impl.h"
#elif __CCE_AICORE__ == 200
#include "tan_v200_impl.h"
#endif

namespace AscendC {
constexpr uint32_t TAN_HALF_CALC_PROCEDURE = 10;
constexpr uint32_t TAN_FLOAT_CALC_PROCEDURE = 4;
// define the number of x div pi
constexpr float PI_FOR_X_TODIV = 0.3183098733425140380859375;
constexpr float KPI_FIRS_PI_MULS = 0.0009670257568359375;
// define the PI for compute
constexpr float PI_V2 = 3.140625;
// define the number of down of pi_div
constexpr float PI_DOWN = 1.57079637050628662109375;
constexpr float PI_DOWN_NEG = -1.57079637050628662109375;
// kpi_2
constexpr float KPI_TWI_PI_MULS = 6.2771141529083251953125e-7;
constexpr float PI_RESDOWN_ADDS = 0.00000004371139000189375;
constexpr float PI_RESDOWN_ADDS_NEG = -0.00000004371139000189375;
// kpi_3
constexpr float KPI_THIR_PI_MULS = 1.21644916362129151821136474609375e-10;
// kpi_4
constexpr float KPI_FOR_PI_MULS = -1.0291767438275201129727065563201904296875e-13;
// define the number of tan_compute
constexpr float TAN_RES_MULIT_SCA = 0.0698520831551998762793;
constexpr float TAN_RES_ADDICT_UP = -6.8711573651634203789;
constexpr float TAN_2ADDS = 61.20362572811089435388;
constexpr float TAN_3ADDS = -24.8048928861126769186219;

__aicore__ inline void KPI_0(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
    const LocalTensor<float>& roundTensor)
{
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binParams;
    // kpi_0 = round_pi_div*3.140625
    Muls<float, false>(dstTensor, roundTensor, PI_V2, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    // input_x = (x-kpi_0)
    Sub<float, false>(srcTensor, srcTensor, dstTensor, MASK_PLACEHOLDER, 1, binParams);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void KPI_1(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
    const LocalTensor<float>& roundTensor, const LocalTensor<float>& resTensor1, const LocalTensor<float>& resTensor2)
{
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binParams;
    // kpi_1 = muls(round_pi_div, 0.0009670257568359375)
    Muls<float, false>(dstTensor, roundTensor, KPI_FIRS_PI_MULS, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    // input_x = sub(input_x, kpi_1)
    Sub<float, false>(srcTensor, srcTensor, dstTensor, MASK_PLACEHOLDER, 1, binParams);
    PipeBarrier<PIPE_V>();
    // res_down1 =adds(input_x, 1.57079637050628662109375)
    Adds<float, false>(resTensor1, srcTensor, PI_DOWN, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    // res_down2 = adds(input_x, -1.57079637050628662109375)
    Adds<float, false>(resTensor2, srcTensor, PI_DOWN_NEG, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void KPI_2(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
    const LocalTensor<float>& roundTensor, const LocalTensor<float>& resTensor1, const LocalTensor<float>& resTensor2)
{
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binParams;
    // kpi_2 = muls(round_pi_div, 6.2771141529083251953125e-7)
    Muls<float, false>(dstTensor, roundTensor, KPI_TWI_PI_MULS, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    // input_x = sub(input_x, kpi_2)
    Sub<float, false>(srcTensor, srcTensor, dstTensor, MASK_PLACEHOLDER, 1, binParams);
    PipeBarrier<PIPE_V>();
    // res_down1 = sub(res_down1, kpi_2)
    Sub<float, false>(resTensor1, resTensor1, dstTensor, MASK_PLACEHOLDER, 1, binParams);
    PipeBarrier<PIPE_V>();
    // res_down2 = sub(res_down2, kpi_2)
    Sub<float, false>(resTensor2, resTensor2, dstTensor, MASK_PLACEHOLDER, 1, binParams);
    PipeBarrier<PIPE_V>();
    // res_down1 = adds(res_down1, -0.00000004371139000189375)
    Adds<float, false>(resTensor1, resTensor1, PI_RESDOWN_ADDS_NEG, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    // res_down2 =adds(res_down2, 0.00000004371139000189375)
    Adds<float, false>(resTensor2, resTensor2, PI_RESDOWN_ADDS, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void KPI_3(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
    const LocalTensor<float>& roundTensor, const LocalTensor<float>& resTensor1, const LocalTensor<float>& resTensor2)
{
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binParams;
    // kpi_3 = muls(round_pi_div, 1.21644916362129151821136474609375e-10)
    Muls<float, false>(dstTensor, roundTensor, KPI_THIR_PI_MULS, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    // input_x =sub(input_x, kpi_3)
    Sub<float, false>(srcTensor, srcTensor, dstTensor, MASK_PLACEHOLDER, 1, binParams);
    PipeBarrier<PIPE_V>();
    // res_down1 = sub(res_down1, kpi_3)
    Sub<float, false>(resTensor1, resTensor1, dstTensor, MASK_PLACEHOLDER, 1, binParams);
    PipeBarrier<PIPE_V>();
    // res_down2 = sub(res_down2, kpi_3)
    Sub<float, false>(resTensor2, resTensor2, dstTensor, MASK_PLACEHOLDER, 1, binParams);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void KPI_4(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
    const LocalTensor<float>& roundTensor, const LocalTensor<float>& resTensor1, const LocalTensor<float>& resTensor2)
{
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binParams;
    // kpi_4 = muls(round_pi_div, -1.0291767438275201129727065563201904296875e-13)
    Muls<float, false>(dstTensor, roundTensor, KPI_FOR_PI_MULS, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    // input_x =sub(input_x, kpi_4)
    Sub<float, false>(srcTensor, srcTensor, dstTensor, MASK_PLACEHOLDER, 1, binParams);
    PipeBarrier<PIPE_V>();
    // res_down1 = sub(res_down1, kpi_4)
    Sub<float, false>(resTensor1, resTensor1, dstTensor, MASK_PLACEHOLDER, 1, binParams);
    PipeBarrier<PIPE_V>();
    // res_down2 = sub(res_down2, kpi_4)
    Sub<float, false>(resTensor2, resTensor2, dstTensor, MASK_PLACEHOLDER, 1, binParams);
    PipeBarrier<PIPE_V>();
}

// normalized x to (-pi/2,pi/2) using x = x-round(x/π)*π
__aicore__ inline void TanRound(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
    const LocalTensor<float>& roundTensor, const LocalTensor<float>& resTensor1, const LocalTensor<float>& resTensor2)
{
    /*
    k=round(x/π), x0=x-kπ, x0∈(-π/2, π/2)
    π=π_0+π_1+π_2+π_3+π_4 achieve final precision compensation.
    Final solution：
    k = round(x * invpi)
    x -= k * pi_0
    x -= k * pi_1
    down1 = x + pio2_high // pi/2 + x
    down2 = x - pio2_high // x - pi/2
    x -= k * pi_2
    down1 -= k * pi_2
    down2 -= k * pi_2
    x -= k * pi_3
    down1 -= k * pi_3
    down2 -= k * pi_3
    x -= k * pi_4
    down1 -= k * pi_4
    down2 -= k * pi_4
    */
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binParams;
    // round_pi_div= round(x*0.3183098733425140380859375)
    Muls<float, false>(roundTensor, srcTensor, PI_FOR_X_TODIV, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    // tie to even
    TanCast(roundTensor, roundTensor, RoundMode::CAST_RINT);

    KPI_0(dstTensor, srcTensor, roundTensor);
    KPI_1(dstTensor, srcTensor, roundTensor, resTensor1, resTensor2);
    KPI_2(dstTensor, srcTensor, roundTensor, resTensor1, resTensor2);
    KPI_3(dstTensor, srcTensor, roundTensor, resTensor1, resTensor2);
    KPI_4(dstTensor, srcTensor, roundTensor, resTensor1, resTensor2);
}

__aicore__ inline void TanPolynomialApproximation(const LocalTensor<float>& dstTensor,
    const LocalTensor<float>& srcTensor, const LocalTensor<float>& roundTensor,
    const LocalTensor<float>& resTensor1, const LocalTensor<float>& resTensor2)
{
    /*
    tan(x) = xP(x) / ((π/2 - x)(π/2 + x)Q(x))
    P(x) = (x^2 * R0 + R1) * x^2 + R2
    Q(x) = x^2 * R3
    R0 = 0.0698520831551998762793
    R1 = -6.8711573651634203789
    R2 = 61.20362572811089435388
    R3 = -24.8048928861126769186219
    */
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binParams;

    // x^2 = mul(input_x, input_x)
    Mul<float, false>(roundTensor, srcTensor, srcTensor, MASK_PLACEHOLDER, 1, binParams);
    PipeBarrier<PIPE_V>();
    // res_up = muls(x^2, 0.0698520831551998762793)
    Muls<float, false>(dstTensor, roundTensor, TAN_RES_MULIT_SCA, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    // res_up = adds(res_up, -6.8711573651634203789)
    Adds<float, false>(dstTensor, dstTensor, TAN_RES_ADDICT_UP, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    // res_up =mul(res_up, x^2)
    Mul<float, false>(dstTensor, dstTensor, roundTensor, MASK_PLACEHOLDER, 1, binParams);
    PipeBarrier<PIPE_V>();
    // res_up = adds(res_up, 61.20362572811089435388)
    Adds<float, false>(dstTensor, dstTensor, TAN_2ADDS, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    // res_up = mul(res_up, input_x)
    Mul<float, false>(dstTensor, dstTensor,  srcTensor, MASK_PLACEHOLDER, 1, binParams);
    PipeBarrier<PIPE_V>();
    // res_down = adds(x^2, -24.8048928861126769186219)
    Adds<float, false>(roundTensor, roundTensor, TAN_3ADDS, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    // res_down = mul(res_down, res_down1)
    Mul<float, false>(roundTensor, roundTensor, resTensor1, MASK_PLACEHOLDER, 1, binParams);
    PipeBarrier<PIPE_V>();
    // res_down = mul(res_down, res_down2)
    Mul<float, false>(roundTensor, roundTensor, resTensor2, MASK_PLACEHOLDER, 1, binParams);
    PipeBarrier<PIPE_V>();
    // res = div(res_up, res_down)
    Div<float, false>(dstTensor, dstTensor, roundTensor, MASK_PLACEHOLDER, 1, binParams);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void TanCompute(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& tmpBuffer, uint32_t calSize)
{
    const UnaryRepeatParams unaryParams;
    const LocalTensor<T>& tmpTensor1 = tmpBuffer.ReinterpretCast<float>();
    const LocalTensor<T>& tmpTensor2 = tmpTensor1[calSize];
    const LocalTensor<T>& tmpTensor3 = tmpTensor2[calSize];
    const LocalTensor<T>& tmpTensor4 = tmpTensor3[calSize];

    Adds<T, false>(tmpTensor4, srcTensor, static_cast<float>(0.0), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    // the input is normalized to (-pi/2,pi/2)
    TanRound(dstTensor, tmpTensor4, tmpTensor1, tmpTensor2, tmpTensor3);
    TanPolynomialApproximation(dstTensor, tmpTensor4, tmpTensor1, tmpTensor2, tmpTensor3);
}

template <>
__aicore__ inline void TanCompute(const LocalTensor<half>& dstTensor, const LocalTensor<half>& srcTensor,
    const LocalTensor<uint8_t>& tmpBuffer, uint32_t calSize)
{
    const LocalTensor<float>& tempTensorConv = tmpBuffer.ReinterpretCast<float>();
    const LocalTensor<float>& tmpTensor1 = tempTensorConv[calSize];
    const LocalTensor<float>& tmpTensor2 = tmpTensor1[calSize];
    const LocalTensor<float>& tmpTensor3 = tmpTensor2[calSize];
    const LocalTensor<float>& tmpTensor4 = tmpTensor3[calSize];

    Cast<float, half, false>(tmpTensor1, srcTensor,
        RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();

    // the input is normalized to (-pi/2,pi/2)
    TanRound(tempTensorConv, tmpTensor1, tmpTensor2, tmpTensor3, tmpTensor4);
    TanPolynomialApproximation(tempTensorConv, tmpTensor1, tmpTensor2, tmpTensor3, tmpTensor4);

    Cast<half, float, false>(dstTensor, tempTensorConv,
        RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, { 1, 1, HALF_DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void TanImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");

    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Tan");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Tan");

    ASCENDC_ASSERT((std::is_same<T, float>::value || std::is_same<T, half>::value), {
        KERNEL_LOG( KERNEL_ERROR, "Failed to check the data types, current api support data types are half/float."); });
    uint32_t tmpBufferSize = sharedTmpBuffer.GetSize();
    uint32_t splitCount = tmpBufferSize / sizeof(T);
    if constexpr (sizeof(T) == sizeof(half)) {
        splitCount = splitCount / TAN_HALF_CALC_PROCEDURE / ONE_BLK_SIZE * ONE_BLK_SIZE;
    } else {
        splitCount = splitCount / TAN_FLOAT_CALC_PROCEDURE / ONE_BLK_SIZE * ONE_BLK_SIZE;
    }
    CheckTmpBufferSize(splitCount, 0, tmpBufferSize);

    const uint32_t loopCount = calCount / splitCount;
    const uint32_t calcTail = calCount % splitCount;

    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(0, splitCount);

    uint32_t offset = 0;
    for (uint32_t i = 0; i < loopCount; ++i) {
        TanCompute(dstTensor[i * splitCount], srcTensor[i * splitCount], sharedTmpBuffer, splitCount);
    }

    if (calcTail > 0) {
        uint32_t tailCount = calcTail / ONE_BLK_SIZE * ONE_BLK_SIZE;
        tailCount = (calcTail % ONE_BLK_SIZE == 0) ? tailCount : (tailCount + ONE_BLK_SIZE);
        SetVectorMask<T, MaskMode::COUNTER>(0, calcTail);
        TanCompute(dstTensor[loopCount * splitCount], srcTensor[loopCount * splitCount], sharedTmpBuffer, tailCount);
    }

    SetMaskNorm();
    ResetMask();
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void TanImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
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
    TanImpl(dstTensor, srcTensor, sharedTmpBuffer, calCount);
}
} // namespace AscendC

#endif // IMPL_MATH_TAN_TAN_COMMON_IMPL_H