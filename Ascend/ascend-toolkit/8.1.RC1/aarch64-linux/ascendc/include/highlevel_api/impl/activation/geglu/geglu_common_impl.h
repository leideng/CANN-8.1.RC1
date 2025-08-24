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
 * \file geglu_common_impl.h
 * \brief
 */

#ifndef IMPL_ACTIVATION_GEGLU_GEGLU_COMMON_IMPL_H
#define IMPL_ACTIVATION_GEGLU_GEGLU_COMMON_IMPL_H
#include "kernel_tensor.h"


namespace AscendC {
constexpr float COEFF0 = -0.0713548162726;
constexpr float COEFF1 = 2.2363860002236e1;
constexpr uint32_t GEGLU_HALF_BUFFER_SIZE = 8; // 1 half data needs 2 float tmpbuffer which is 8 uint8 size
constexpr uint32_t GEGLU_FLOAT_BUFFER_SIZE = 0;
constexpr uint32_t GEGLU_STRIDE_DIGITS = 2;
constexpr uint32_t GEGLU_ALGINED = 31;

template <typename T, bool isReuseSource = false>
__aicore__ inline void GeGLUImpl(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor0,
    const LocalTensor<T> &srcTensor1, uint32_t calCount)
{
    // Only for AI Vector Core.
    if (g_coreType == AIC) {
        return;
    }
    LocalTensor<uint8_t> tmpBuffer;
    PopStackBuffer<uint8_t, TPosition::LCM>(tmpBuffer);
    GeGLUImpl<T, isReuseSource>(dstTensor, srcTensor0, srcTensor1, tmpBuffer, calCount);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void GeGLUImpl(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor0,
    const LocalTensor<T> &srcTensor1, const LocalTensor<uint8_t> &sharedTmpBuffer, uint32_t calCount)
{
    // Only for AI Vector Core
    if (g_coreType == AIC) {
        return;
    }
#ifdef ASCENDC_CPU_DEBUG
    bool ret = (srcTensor0.GetSize() == srcTensor1.GetSize());
    ASCENDC_ASSERT(ret, { KERNEL_LOG(KERNEL_ERROR, "Size of src0: %u is not equal to size of src1: %u",
        srcTensor0.GetSize(), srcTensor1.GetSize()); });
    ret = (calCount <= srcTensor0.GetSize()) && (calCount <= dstTensor.GetSize());
    ASCENDC_ASSERT(ret, { KERNEL_LOG(KERNEL_ERROR, "calCount must be less than or equal to src/dst tensor"); });

    ret = (std::is_same<T, half>::value) || (std::is_same<T, float>::value);
    ASCENDC_ASSERT(ret, { KERNEL_LOG(KERNEL_ERROR, "type must be half or float"); });
#endif
    LocalTensor<float> tmpBuffer = sharedTmpBuffer.ReinterpretCast<float>();
    tmpBuffer.SetSize(sharedTmpBuffer.GetSize() / sizeof(float));
    SetMaskCount();

    if (sizeof(T) == sizeof(half)) {
        auto tmpBufCount = sharedTmpBuffer.GetSize() / GEGLU_HALF_BUFFER_SIZE;
        tmpBufCount = tmpBufCount * sizeof(T) / ONE_BLK_SIZE * ONE_BLK_SIZE / sizeof(T);
        for (uint32_t offset = 0; offset < calCount; offset += tmpBufCount) {
            auto splitSize = (calCount - offset) > tmpBufCount ? tmpBufCount : (calCount - offset);
            SetVectorMask<T>(0, splitSize);
            splitSize = (splitSize * sizeof(T) + GEGLU_ALGINED) / ONE_BLK_SIZE * ONE_BLK_SIZE / sizeof(T);
            GeGLUCompute(dstTensor[offset], srcTensor0[offset], srcTensor1[offset], tmpBuffer, splitSize);
        }
    } else {
        SetVectorMask<T>(0, calCount);
        GeGLUCompute(dstTensor, srcTensor0, srcTensor1, tmpBuffer, calCount);
    }

    SetMaskNorm();
    ResetMask();
}

template <typename T>
__aicore__ inline void GeGLUCompute(const LocalTensor<T> &dst, const LocalTensor<T> &src0, const LocalTensor<T> &src1,
    const LocalTensor<float> &tmpBuffer, uint32_t calSize)
{
    UnaryRepeatParams unaryParams;
    BinaryRepeatParams binaryParams;

    // Calculate x^2
    Mul<T, false>(dst, src1, src1, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    // Calculate x^2 + COEFF1
    Adds<T, false>(dst, dst, static_cast<T>(COEFF1), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // Calculate (x^2 + COEFF1) * x
    Mul<T, false>(dst, dst, src1, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    // Calculate (x^2 + COEFF1) * x * COEFF0
    Muls<T, false>(dst, dst, static_cast<T>(COEFF0), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // Calculate e^((x^2 + COEFF1) * x * COEFF0)
    Exp<T, false>(dst, dst, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // Calculate e^((x^2 + COEFF1) * x * COEFF0) + 1
    Adds<T, false>(dst, dst, static_cast<T>(1.0), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Div<T, false>(dst, src1, dst, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    // Calculate x1 * Gelu(x2)
    Mul<T, false>(dst, src0, dst, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
}

// Compute high precision GeGLU values for half type inputs by converting inputs to float types and save float GeGLU
// result in tmpBuffer. Requires 4 times extra buffer for input data.
template <>
__aicore__ inline void GeGLUCompute(const LocalTensor<half> &dst, const LocalTensor<half> &src0,
    const LocalTensor<half> &src1, const LocalTensor<float> &tmpBuffer, uint32_t calSize)
{
    UnaryRepeatParams unaryParams;
    BinaryRepeatParams binaryParams;

    LocalTensor<float> tmpFloatBuffer1 = tmpBuffer;
    LocalTensor<float> tmpFloatBuffer2 = tmpBuffer[calSize];

    Cast<float, half, false>(tmpFloatBuffer1, src1, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        {1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE / GEGLU_STRIDE_DIGITS});
    PipeBarrier<PIPE_V>();

    // Calculate x^2
    Mul<float, false>(tmpFloatBuffer2, tmpFloatBuffer1, tmpFloatBuffer1, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    // Calculate x^2 + COEFF1
    Adds<float, false>(tmpFloatBuffer2, tmpFloatBuffer2, COEFF1, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // Calculate (x^2 + COEFF1) * x
    Mul<float, false>(tmpFloatBuffer2, tmpFloatBuffer2, tmpFloatBuffer1, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    // Calculate (x^2 + COEFF1) * x * COEFF0
    Muls<float, false>(tmpFloatBuffer2, tmpFloatBuffer2, COEFF0, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // Calculate e^((x^2 + COEFF1) * x * COEFF0)
    Exp<float, false>(tmpFloatBuffer2, tmpFloatBuffer2, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // Calculate e^((x^2 + COEFF1) * x * COEFF0) + 1
    Adds<float, false>(tmpFloatBuffer2, tmpFloatBuffer2, static_cast<float>(1.0), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // Calculate Gelu(x2) = x / (e^((x^2 + COEFF1) * x * COEFF0) + 1)
    Div<float, false>(tmpFloatBuffer2, tmpFloatBuffer1, tmpFloatBuffer2, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Cast<float, half, false>(tmpFloatBuffer1, src0, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        {1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE / GEGLU_STRIDE_DIGITS});
    PipeBarrier<PIPE_V>();

    // Calculate x1 * Gelu(x2)
    Mul<float, false>(tmpFloatBuffer2, tmpFloatBuffer1, tmpFloatBuffer2, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Cast<half, float, false>(dst, tmpFloatBuffer2, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE / GEGLU_STRIDE_DIGITS, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
}
}
#endif