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
 * \file acosh_common_impl.h
 * \brief
 */

#ifndef IMPL_MATH_ACOSH_ACOSH_COMMON_IMPL_H
#define IMPL_MATH_ACOSH_ACOSH_COMMON_IMPL_H

#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../common/check.h"

namespace AscendC {
constexpr uint32_t ACOSH_HALF_CALC_PROC = 2;
constexpr uint32_t ACOSH_FLOAT_CALC_PROC = 1;
constexpr float ACOSH_NEG_ONE = -1;
constexpr uint32_t ACOSH_STRIDE_DIGITS = 2;

template <typename T>
__aicore__ inline void AcoshCompute(const LocalTensor<T> &dst, const LocalTensor<T> &src,
    const LocalTensor<float> &tmpBuffer, uint32_t calCount)
{
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;
    LocalTensor<float> tmpFloatBuffer1 = tmpBuffer;
    // Calculate the amount of data that can be stored in the temporary space and split the data into the entire block
    // and tail block.
    // acosh(x) = ln(x + sqrt(x^2 - 1))

    // x^2
    Mul<T, false>(tmpFloatBuffer1, src, src, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
    // x^2 - 1
    Adds<T, false>(tmpFloatBuffer1, tmpFloatBuffer1, static_cast<T>(ACOSH_NEG_ONE), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    // sqrt(x^2 - 1)
    Sqrt<T, false>(tmpFloatBuffer1, tmpFloatBuffer1, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    // x + sqrt(x^2 - 1)
    Add<T, false>(tmpFloatBuffer1, src, tmpFloatBuffer1, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Div<T, false>(dst, tmpFloatBuffer1, tmpFloatBuffer1, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Mul<T, false>(tmpFloatBuffer1, tmpFloatBuffer1, dst, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    // ln(x + sqrt(x^2 - 1))
    Ln<T, false>(dst, tmpFloatBuffer1, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
}

template <>
__aicore__ inline void AcoshCompute(const LocalTensor<half> &dst, const LocalTensor<half> &src,
    const LocalTensor<float> &tmpBuffer, uint32_t calCount)
{
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;
    LocalTensor<float> tmpFloatBuffer1 = tmpBuffer;
    LocalTensor<float> tmpFloatBuffer2 = tmpFloatBuffer1[calCount];

    // Calculate the amount of data that can be stored in the temporary space and split the data into the entire block
    // and tail block.
    // In the case of the half data type, there is no direct instruction for the round operation. Therefore, multiple
    // conversions are required.
    // acosh(x) = ln(x + sqrt(x^2 - 1))
    Cast<float, half, false>(tmpFloatBuffer1, src, RoundMode::CAST_NONE,
        MASK_PLACEHOLDER, 1, { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE / ACOSH_STRIDE_DIGITS });
    PipeBarrier<PIPE_V>();

    // x^2
    Mul<float, false>(tmpFloatBuffer2, tmpFloatBuffer1, tmpFloatBuffer1, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    // x^2 - 1
    Adds<float, false>(tmpFloatBuffer2, tmpFloatBuffer2, static_cast<half>(ACOSH_NEG_ONE),
        MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // sqrt(x^2 - 1)
    Sqrt<float, false>(tmpFloatBuffer2, tmpFloatBuffer2, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // x + sqrt(x^2 - 1)
    Add<float, false>(tmpFloatBuffer2, tmpFloatBuffer2, tmpFloatBuffer1, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Div<float, false>(tmpFloatBuffer1, tmpFloatBuffer2, tmpFloatBuffer2, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Mul<float, false>(tmpFloatBuffer2, tmpFloatBuffer2, tmpFloatBuffer1, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    // ln(x + sqrt(x^2 - 1))
    Ln<float, false>(tmpFloatBuffer2, tmpFloatBuffer2, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Cast<half, float, false>(dst, tmpFloatBuffer2, RoundMode::CAST_NONE,
        MASK_PLACEHOLDER, 1, { 1, 1, DEFAULT_REPEAT_STRIDE / ACOSH_STRIDE_DIGITS, DEFAULT_REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void AcoshImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }


    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");
    
    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Acosh");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Acosh");

    ASCENDC_ASSERT((std::is_same<T, float>::value || std::is_same<T, half>::value), {
        KERNEL_LOG( KERNEL_ERROR, "Failed to check the data types, current api support data types are half/float."); });

    uint32_t tmpBufferSize = sharedTmpBuffer.GetSize();
    uint32_t splitCount = tmpBufferSize / sizeof(float);

    if constexpr (sizeof(T) == sizeof(half)) {
        splitCount = splitCount / ACOSH_HALF_CALC_PROC / ONE_BLK_SIZE * ONE_BLK_SIZE;
    } else {
        splitCount = splitCount / ACOSH_FLOAT_CALC_PROC / ONE_BLK_SIZE * ONE_BLK_SIZE;
    }
    CheckTmpBufferSize(splitCount, 0, tmpBufferSize);

    uint32_t loopCount = calCount / splitCount;
    uint32_t calcTail = calCount % splitCount;
    LocalTensor<float> tmpBuffer = sharedTmpBuffer.ReinterpretCast<float>();
    tmpBuffer.SetSize(sharedTmpBuffer.GetSize() / sizeof(float));

    SetMaskCount();
    SetVectorMask<T>(0, splitCount);
    for (uint32_t i = 0; i < loopCount; ++i) {
        AcoshCompute(dstTensor[i * splitCount], srcTensor[i * splitCount], tmpBuffer, splitCount);
    }
    if (calcTail > 0) {
        uint32_t tailCount = calcTail / ONE_BLK_SIZE * ONE_BLK_SIZE;
        tailCount = (calcTail % ONE_BLK_SIZE == 0) ? tailCount : (tailCount + ONE_BLK_SIZE);
        SetVectorMask<T>(0, calcTail);
        AcoshCompute(dstTensor[loopCount * splitCount], srcTensor[loopCount * splitCount], tmpBuffer, tailCount);
    }
    SetMaskNorm();
    ResetMask();
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void AcoshImpl(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
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
    AcoshImpl<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, calCount);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void AcoshImpl(const LocalTensor<T> &dstTensor,
	const LocalTensor<T> &srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    AcoshImpl<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, srcTensor.GetSize());
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void AcoshImpl(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor)
{
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ret = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ret), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    AcoshImpl<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, srcTensor.GetSize());
}
}
#endif