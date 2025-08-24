/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef IMPL_MATH_ERFC_ERFC_COMMON_IMPL_H
#define IMPL_MATH_ERFC_ERFC_COMMON_IMPL_H
#include "kernel_tensor.h"
#include "../math_constant_util.h"
#include "../../common/check.h"

namespace AscendC {
constexpr float ERFC_BOUNDARY_MAX = 10;
constexpr uint8_t TMPBUF_IDX_3 = 2;
constexpr uint8_t TMPBUF_IDX_5 = 4;
constexpr uint8_t TMPBUF_IDX_6 = 5;

__aicore__ inline constexpr RoundMode GetErfcCastType()
{
#if __CCE_AICORE__ == 220
    return RoundMode::CAST_ROUND;
#else
    return RoundMode::CAST_NONE;
#endif
}

// compute Erfc with in [-1, 1], x / (|x| + fp32_min)
__aicore__ inline void ErfcPreCompute(const LocalTensor<float>& dstBuf1, const LocalTensor<float>& srcBuf1,
    const LocalTensor<float>& tmpCompBuf1, uint32_t calSize)
{
    BinaryRepeatParams binaryParams;
    UnaryRepeatParams unaryParams;

    Abs<float, false>(tmpCompBuf1, srcBuf1, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    constexpr float SCALAR_ERFC_FP32_MIN = 2.168404344971009e-19; // 2^-62
    Adds<float, false>(tmpCompBuf1, tmpCompBuf1, static_cast<float>(SCALAR_ERFC_FP32_MIN), MASK_PLACEHOLDER, 1,
        unaryParams);
    PipeBarrier<PIPE_V>();

    Div<float, false>(dstBuf1, srcBuf1, tmpCompBuf1, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
}

// compute Erfc R(Z) within [-inf, -1) and (1, inf]
// R(z) = ((((((((z*r0 + r1)*z + r2)*z + r3)*z + r4)*z + r5)*z + r6)*z + r7)*z + r8)
__aicore__ inline void ErfcComputeR(const LocalTensor<float>& tmpCompBuf1, uint32_t calSize)
{
    BinaryRepeatParams binaryParams;
    UnaryRepeatParams unaryParams;

    LocalTensor<float> tmpCompBuf3 = tmpCompBuf1[TMPBUF_IDX_3 * calSize];
    LocalTensor<float> tmpCompBuf4 = tmpCompBuf3[calSize];

    // Specific value used for approximate calculation.
    constexpr float R0 = 0.1735313680e-7;
    constexpr float R1 = -0.9856738394e-6;
    constexpr float R2 = 0.2517003236e-4;
    constexpr float R3 = -0.3848015171e-3;
    constexpr float R4 = 0.5681528564e0;
    constexpr float R5 = 0.5245623129e1;
    constexpr float R6 = 0.2107740710e2;
    constexpr float R7 = 0.4212761755e2;
    constexpr float R8 = 0.4380524149e2;

    // R(z) = ((((((((z*r0 + r1)*z + r2)*z + r3)*z + r4)*z + r5)*z + r6)*z + r7)*z + r8)
    Muls<float, false>(tmpCompBuf3, tmpCompBuf1, static_cast<float>(R0), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Adds<float, false>(tmpCompBuf4, tmpCompBuf3, static_cast<float>(R1), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Mul<float, false>(tmpCompBuf3, tmpCompBuf1, tmpCompBuf4, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Adds<float, false>(tmpCompBuf4, tmpCompBuf3, static_cast<float>(R2), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Mul<float, false>(tmpCompBuf3, tmpCompBuf1, tmpCompBuf4, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Adds<float, false>(tmpCompBuf4, tmpCompBuf3, static_cast<float>(R3), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Mul<float, false>(tmpCompBuf3, tmpCompBuf1, tmpCompBuf4, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Adds<float, false>(tmpCompBuf4, tmpCompBuf3, static_cast<float>(R4), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Mul<float, false>(tmpCompBuf3, tmpCompBuf1, tmpCompBuf4, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Adds<float, false>(tmpCompBuf4, tmpCompBuf3, static_cast<float>(R5), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Mul<float, false>(tmpCompBuf3, tmpCompBuf1, tmpCompBuf4, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Adds<float, false>(tmpCompBuf4, tmpCompBuf3, static_cast<float>(R6), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Mul<float, false>(tmpCompBuf3, tmpCompBuf1, tmpCompBuf4, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Adds<float, false>(tmpCompBuf4, tmpCompBuf3, static_cast<float>(R7), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Mul<float, false>(tmpCompBuf3, tmpCompBuf1, tmpCompBuf4, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Adds<float, false>(tmpCompBuf4, tmpCompBuf3, static_cast<float>(R8), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
}

// compute Erfc S(Z) within [-inf, -1) and (1, inf]
// S(z) = (((((z + s1)*z + s2)*z + s3)*z + s4)*z + s5)
__aicore__ inline void ErfcComputeS(const LocalTensor<float>& tmpCompBuf1, uint32_t calSize)
{
    BinaryRepeatParams binaryParams;
    UnaryRepeatParams unaryParams;

    LocalTensor<float> tmpCompBuf3 = tmpCompBuf1[TMPBUF_IDX_3 * calSize];
    LocalTensor<float> tmpCompBuf5 = tmpCompBuf1[TMPBUF_IDX_5 * calSize];

    // Specific value used for approximate calculation.
    constexpr float S1 = 0.9349684299e1;
    constexpr float S2 = 0.3756930664e2;
    constexpr float S3 = 0.8058268949e2;
    constexpr float S4 = 0.9155653738e2;
    constexpr float S5 = 0.4380524152e2;

    // S(z) = (((((z + s1)*z + s2)*z + s3)*z + s4)*z + s5)
    Adds<float, false>(tmpCompBuf3, tmpCompBuf1, static_cast<float>(S1), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Mul<float, false>(tmpCompBuf5, tmpCompBuf1, tmpCompBuf3, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Adds<float, false>(tmpCompBuf3, tmpCompBuf5, static_cast<float>(S2), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Mul<float, false>(tmpCompBuf5, tmpCompBuf1, tmpCompBuf3, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Adds<float, false>(tmpCompBuf3, tmpCompBuf5, static_cast<float>(S3), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Mul<float, false>(tmpCompBuf5, tmpCompBuf1, tmpCompBuf3, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Adds<float, false>(tmpCompBuf3, tmpCompBuf5, static_cast<float>(S4), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Mul<float, false>(tmpCompBuf5, tmpCompBuf1, tmpCompBuf3, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Adds<float, false>(tmpCompBuf3, tmpCompBuf5, static_cast<float>(S5), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
}

// compute Erfc Part2 within [-inf, -1) and (1, inf]
__aicore__ inline void ErfcPostCompute(const LocalTensor<float>& dstBuf1, const LocalTensor<float>& srcBuf1,
    const LocalTensor<float>& tmpCompBuf1, uint32_t calSize)
{
    BinaryRepeatParams binaryParams;
    UnaryRepeatParams unaryParams;

    LocalTensor<float> tmpCompBuf2 = tmpCompBuf1[calSize];
    LocalTensor<float> tmpCompBuf3 = tmpCompBuf2[calSize];

    Muls<float, false>(tmpCompBuf2, srcBuf1, static_cast<float>(NEG_ONE), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Adds<float, false>(tmpCompBuf2, tmpCompBuf2, static_cast<float>(NUM_ONE), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Mul<float, false>(tmpCompBuf3, tmpCompBuf3, srcBuf1, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Add<float, false>(dstBuf1, tmpCompBuf3, tmpCompBuf2, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
}

// merge all public steps in half mode and float mode
__aicore__ inline void ErfcPublicSteps(const LocalTensor<float>& tmpCompBuf1, uint32_t calSize)
{
    BinaryRepeatParams binaryParams;
    UnaryRepeatParams unaryParams;

    LocalTensor<float> tmpCompBuf2 = tmpCompBuf1[calSize];
    LocalTensor<float> tmpCompBuf3 = tmpCompBuf2[calSize];
    LocalTensor<float> tmpCompBuf4 = tmpCompBuf3[calSize];
    LocalTensor<float> tmpCompBuf5 = tmpCompBuf4[calSize];

    Mins<float, false>(tmpCompBuf2, tmpCompBuf1, static_cast<float>(ERFC_BOUNDARY_MAX), MASK_PLACEHOLDER, 1,
        unaryParams);
    PipeBarrier<PIPE_V>();
    ErfcComputeR(tmpCompBuf1, calSize);
    ErfcComputeS(tmpCompBuf1, calSize);

    // exp(-x^2) * (R(z) / S(z))
    Div<float, false>(tmpCompBuf3, tmpCompBuf4, tmpCompBuf3, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Mul<float, false>(tmpCompBuf1, tmpCompBuf1, tmpCompBuf1, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
    Muls<float, false>(tmpCompBuf1, tmpCompBuf1, static_cast<float>(NEG_ONE), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Exp<float, false>(tmpCompBuf1, tmpCompBuf1, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Mul<float, false>(tmpCompBuf3, tmpCompBuf1, tmpCompBuf3, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
}

// Clip x to [-ERFC_BOUNDARY_MAX, ERFC_BOUNDARY_MAX]
__aicore__ inline void ErfcClip(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const LocalTensor<float>& tmpBuffer)
{
    UnaryRepeatParams unaryParams;

    Mins<float, false>(dst, src, static_cast<float>(ERFC_BOUNDARY_MAX), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Maxs<float, false>(tmpBuffer, dst, static_cast<float>(-ERFC_BOUNDARY_MAX), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
}

// Computes erfc values based on input types.
template <typename T>
__aicore__ inline void ErfcCompute(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<T>& tmpBuffer, uint32_t calSize)
{
    // used to save ErfcCompute result
    LocalTensor<T> tmpCompBuf1 = tmpBuffer;

    // clip x to [-ERFC_BOUNDARY_MAX, ERFC_BOUNDARY_MAX] stored on dst.
    ErfcClip(dst, src, dst);
    ErfcPreCompute(dst, dst, tmpCompBuf1, calSize);

    // merge the public parts of half and float to one function
    ErfcPublicSteps(tmpCompBuf1, calSize);

    ErfcPostCompute(dst, dst, tmpCompBuf1, calSize);
}

template <>
__aicore__ inline void ErfcCompute<half>(const LocalTensor<half>& dst, const LocalTensor<half>& src,
    const LocalTensor<half>& tmpBuffer, uint32_t calSize)
{
    // used to save ErfcCompute result
    LocalTensor<float> tmpCompBuf1 = tmpBuffer.ReinterpretCast<float>();
    LocalTensor<float> tmpCompBuf6 = tmpCompBuf1[TMPBUF_IDX_6 * calSize];

    // Cast src from half to float type for getting more precise results.
    Cast<float, half, false>(tmpCompBuf6, src, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE / 2 });
    PipeBarrier<PIPE_V>();

    // clip x to [-10, 10], saved in tmpCompBuf6
    ErfcClip(tmpCompBuf1, tmpCompBuf6, tmpCompBuf6);

    ErfcPreCompute(tmpCompBuf6, tmpCompBuf6, tmpCompBuf1, calSize);

    ErfcPublicSteps(tmpCompBuf1, calSize);

    ErfcPostCompute(tmpCompBuf1, tmpCompBuf6, tmpCompBuf1, calSize);

    constexpr RoundMode castType = GetErfcCastType();

    Cast<half, float, false>(dst, tmpCompBuf1, castType, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE / 2, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void ErfcImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");

    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Erfc");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Erfc");

    ASCENDC_ASSERT((std::is_same<T, float>::value || std::is_same<T, half>::value), {
        KERNEL_LOG( KERNEL_ERROR, "Failed to check the data types, current api support data types are half/float."); });

    LocalTensor<T> tmpBuffer = sharedTmpBuffer.ReinterpretCast<T>();
    uint32_t bufferSize = sharedTmpBuffer.GetSize();
    uint32_t tmpBufferSize = bufferSize / sizeof(T); // all temporary variables are float type.
    CheckTmpBufferSize(tmpBufferSize, 0, bufferSize);

    constexpr uint8_t ERFC_HALF_CALC_PROCEDURE = 12;
    constexpr uint8_t ERFC_FLOAT_CALC_PROCEDURE = 5;
    uint32_t calSize = 0;
    if constexpr (sizeof(T) == sizeof(half)) {
        calSize = tmpBufferSize / ERFC_HALF_CALC_PROCEDURE / ONE_BLK_SIZE * ONE_BLK_SIZE;
    } else {
        calSize = tmpBufferSize / ERFC_FLOAT_CALC_PROCEDURE / ONE_BLK_SIZE * ONE_BLK_SIZE;
    }
    CheckTmpBufferSize(calSize, 0, bufferSize);

    const uint32_t round = calCount / calSize;
    const uint32_t tail = calCount % calSize;

    SetMaskCount();
    SetVectorMask<half, MaskMode::COUNTER>(0, calSize);

    uint32_t offset = 0;

    for (uint32_t i = 0; i < round; i++) {
        ErfcCompute(dstTensor[offset], srcTensor[offset], tmpBuffer, calSize);
        offset = offset + calSize;
    }

    if (tail != 0) {
        SetVectorMask<half, MaskMode::COUNTER>(0, tail);
        ErfcCompute(dstTensor[offset], srcTensor[offset], tmpBuffer, calSize);
    }

    SetMaskNorm();
    ResetMask();
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void ErfcImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    ErfcImpl(dstTensor, srcTensor, sharedTmpBuffer, calCount);
}
} // namespace AscendC
#endif // IMPL_MATH_ERFC_ERFC_COMMON_IMPL_H
