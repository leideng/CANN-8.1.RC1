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
 * \file cosh_common_impl.h
 * \brief
 */
#ifndef IMPL_MATH_COSH_COSH_COMMON_IMPL_H
#define IMPL_MATH_COSH_COSH_COMMON_IMPL_H

#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "../../common/check.h"
#if __CCE_AICORE__ == 220
#include "cosh_v220_impl.h"
#elif __CCE_AICORE__ == 200
#include "cosh_v200_impl.h"
#endif

namespace AscendC {
constexpr float SCALAR_LN2 = -0.69314718055994530941723212145818;
constexpr float SCALAR_BROAD_CAST = 0.25;
const uint8_t COSH_HALF_CALC_PROCEDURE = 6;
const uint8_t COSH_FLOAT_CALC_PROCEDURE = 2;

// Computes cosh values based on input types.
// According formula: cosh(x) = (e^x+e^(-x))/2 = e^(x-ln2) + 0.25/(e^(x-ln2)).
template <typename T>
__aicore__ inline void CoshCompute(
    const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<T>& tmpBuffer, uint32_t calSize)
{
    UnaryRepeatParams unaryParams;
    BinaryRepeatParams binaryParams;

    LocalTensor<T> tmpBuffer2 = tmpBuffer[calSize];
    // Calculates y1=e^(x-ln2).
    Adds<T, false>(tmpBuffer2, src, static_cast<T>(SCALAR_LN2), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Exp<T, false>(tmpBuffer, tmpBuffer2, MASK_PLACEHOLDER, 1, unaryParams);

    // Calculate y2=0.25/(e^(x-ln2))
    PipeBarrier<PIPE_V>();
    Duplicate<T, false>(dst, static_cast<T>(SCALAR_BROAD_CAST), MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
    Div<T, false>(tmpBuffer2, dst, tmpBuffer, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    // Calculates y = y1 + y2.
    Add<T, false>(dst, tmpBuffer, tmpBuffer2, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
}

template <>
__aicore__ inline void CoshCompute<half>(const LocalTensor<half>& dst, const LocalTensor<half>& src,
    const LocalTensor<half>& tmpBuffer, uint32_t calSize)
{
    UnaryRepeatParams unaryParams;
    BinaryRepeatParams binaryParams;

    const LocalTensor<float>& tmpFloatBuffer1 = tmpBuffer.ReinterpretCast<float>();
    const LocalTensor<float>& tmpFloatBuffer2 = tmpFloatBuffer1[calSize];
    const LocalTensor<float>& tmpFloatBuffer3 = tmpFloatBuffer2[calSize];

    Cast<float, half, false>(tmpFloatBuffer3, src, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
    // Calculates y1=e^(x-ln2).
    Adds<float, false>(tmpFloatBuffer2, tmpFloatBuffer3, SCALAR_LN2, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Exp<float, false>(tmpFloatBuffer1, tmpFloatBuffer2, MASK_PLACEHOLDER, 1, unaryParams);

    // Calculate y2=0.25/(e^(x-ln2))
    PipeBarrier<PIPE_V>();
    Duplicate<float, false>(tmpFloatBuffer3, SCALAR_BROAD_CAST, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
    Div<float, false>(tmpFloatBuffer2, tmpFloatBuffer3, tmpFloatBuffer1, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    // Calculates y = y1 + y2.
    Add<float, false>(tmpFloatBuffer3, tmpFloatBuffer1, tmpFloatBuffer2, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
    CoshCast(dst, tmpFloatBuffer3);
}


template <typename T, bool isReuseSource = false>
__aicore__ inline void CoshImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");

    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Cosh");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Cosh");

    ASCENDC_ASSERT((std::is_same<T, float>::value || std::is_same<T, half>::value), {
        KERNEL_LOG( KERNEL_ERROR, "Failed to check the data types, current api support data types are half/float."); });

    const uint32_t bufferSize = sharedTmpBuffer.GetSize();
    const uint32_t tmpBufferSize = bufferSize / sizeof(T);
    CheckTmpBufferSize(tmpBufferSize, 0, bufferSize);

    LocalTensor<T> tmpBuffer = sharedTmpBuffer.ReinterpretCast<T>();

    uint32_t calSize = 0;
    if constexpr (sizeof(T) == sizeof(half)) {
        calSize = tmpBufferSize / COSH_HALF_CALC_PROCEDURE / ONE_BLK_SIZE * ONE_BLK_SIZE;
    } else {
        calSize = tmpBufferSize / COSH_FLOAT_CALC_PROCEDURE / ONE_BLK_SIZE * ONE_BLK_SIZE;
    }
    CheckTmpBufferSize(calSize, 0, bufferSize);


    const uint32_t round = calCount / calSize;
    const uint32_t tail = calCount % calSize;

    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(0, calSize);

    uint32_t offset = 0;
    for (uint32_t i = 0; i < round; i++) {
        CoshCompute(dstTensor[offset], srcTensor[offset], tmpBuffer, calSize);
        offset = offset + calSize;
    }

    if (tail != 0) {
        SetVectorMask<T, MaskMode::COUNTER>(0, tail);
        CoshCompute(dstTensor[offset], srcTensor[offset], tmpBuffer, calSize);
    }

    SetMaskNorm();
    ResetMask();
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void CoshImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
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
    CoshImpl(dstTensor, srcTensor, sharedTmpBuffer, calCount);
}
} // namespace AscendC

#endif // IMPL_MATH_COSH_COSH_COMMON_IMPL_H
