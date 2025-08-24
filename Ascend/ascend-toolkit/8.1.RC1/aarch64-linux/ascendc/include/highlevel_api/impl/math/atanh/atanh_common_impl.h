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
 * \file atanh_common_impl.h
 * \brief
 */

#ifndef IMPL_MATH_ATANH_ATANH_COMMON_IMPL_H
#define IMPL_MATH_ATANH_ATANH_COMMON_IMPL_H

#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../common/check.h"

namespace AscendC {
constexpr uint32_t ATANH_FLOAT_CALC_PROC = 1;
constexpr uint32_t ATANH_HALF_CALC_PROC = 4;

// Computes atanh values based on input types.
// Require 1 times extra buffer of input sizes.
// atanh(x) = 0.5 * ln((1 + x) / (1 - x)) x:(-1, 1)
template <typename T>
__aicore__ inline void AtanhCompute(const LocalTensor<T>& dstTensor,
    const LocalTensor<T>& srcTensor,
    const LocalTensor<float> &tmpBuffer,
    uint32_t calSize)
{
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;

    // Caculates 1 + x
    Adds<float, false>(tmpBuffer, srcTensor, static_cast<T>(1), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // Caculates 1 - x
    Muls<float, false>(dstTensor, srcTensor, static_cast<T>(-1), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Adds<float, false>(dstTensor, dstTensor, static_cast<T>(1), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // Caulates 0.5 * ln((1 + x) / (1 - x))
    Div<float, false>(dstTensor, tmpBuffer, dstTensor, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Ln<float, false>(dstTensor, dstTensor, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Muls<float, false>(dstTensor, dstTensor, static_cast<T>(0.5), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
}

// Computes high precision atanh values for half type inputs by converting inputs to float types and save float atanh
// result at tmp Buffer. Requires 4 times extra buffer of input sizes. x:(-1, 1)
template <>
__aicore__ inline void AtanhCompute(const LocalTensor<half>& dstTensor,
    const LocalTensor<half>& srcTensor,
    const LocalTensor<float> &tmpBuffer,
    uint32_t calSize)
{
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;

    LocalTensor<float> tmpFloatBuffer1 = tmpBuffer;
    LocalTensor<float> tmpFloatBuffer2 = tmpBuffer[calSize];

    // half -> float
    Cast<float, half, false>(tmpFloatBuffer1, srcTensor, RoundMode::CAST_NONE, MASK_PLACEHOLDER,
        1, { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE / 2 });
    PipeBarrier<PIPE_V>();

    // Caculates 1 + x
    Adds<float, false>(tmpFloatBuffer2, tmpFloatBuffer1, static_cast<float>(1), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // Caculates 1 - x
    Muls<float, false>(tmpFloatBuffer1, tmpFloatBuffer1, static_cast<float>(-1), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Adds<float, false>(tmpFloatBuffer1, tmpFloatBuffer1, static_cast<float>(1), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // Caulates 0.5 * ln((1 + x) / (1 - x))
    Div<float, false>(tmpFloatBuffer1, tmpFloatBuffer2, tmpFloatBuffer1, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Ln<float, false>(tmpFloatBuffer1, tmpFloatBuffer1, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Muls<float, false>(tmpFloatBuffer1, tmpFloatBuffer1, static_cast<float>(0.5), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // float -> half
    Cast<half, float, false>(dstTensor, tmpFloatBuffer1, RoundMode::CAST_NONE, MASK_PLACEHOLDER,
        1, { 1, 1, DEFAULT_REPEAT_STRIDE / 2, DEFAULT_REPEAT_STRIDE });
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void AtanhImpl(const LocalTensor<T> &dstTensor,
    const LocalTensor<T> &srcTensor,
    const LocalTensor<uint8_t> &sharedTmpBuffer,
    const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");
    
    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Atanh");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Atanh");

    ASCENDC_ASSERT((std::is_same<T, float>::value || std::is_same<T, half>::value), {
        KERNEL_LOG( KERNEL_ERROR, "Failed to check the data types, current api support data types are half/float."); });
    uint32_t tmpBufferSize = sharedTmpBuffer.GetSize();
    uint32_t splitCount = tmpBufferSize / sizeof(T);

    if constexpr (sizeof(T) == sizeof(half)) {
        splitCount = splitCount / ATANH_HALF_CALC_PROC / ONE_BLK_SIZE * ONE_BLK_SIZE;
    } else {
        splitCount = splitCount / ATANH_FLOAT_CALC_PROC / ONE_BLK_SIZE * ONE_BLK_SIZE;
    }
    CheckTmpBufferSize(splitCount, 0, tmpBufferSize);

    uint32_t loopCount = calCount / splitCount;
    uint32_t calcTail = calCount % splitCount;

    SetMaskCount();
    SetVectorMask<T>(0, splitCount);

    LocalTensor<float> tmpBuffer = sharedTmpBuffer.ReinterpretCast<float>();

    for (uint32_t i = 0; i < loopCount; i++) {
        AtanhCompute(dstTensor[i * splitCount], srcTensor[i * splitCount], tmpBuffer, splitCount);
    }

    if (calcTail > 0) {
        uint32_t tailCount = calcTail / ONE_BLK_SIZE * ONE_BLK_SIZE;
        tailCount = (calcTail % ONE_BLK_SIZE == 0) ? tailCount : (tailCount + ONE_BLK_SIZE);
        SetVectorMask<T>(0, calcTail);
        AtanhCompute(dstTensor[loopCount * splitCount], srcTensor[loopCount * splitCount], tmpBuffer, tailCount);
    }

    SetMaskNorm();
    ResetMask();
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void AtanhImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor)
{
    // Using the stack space to allocate tmpbuf
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ret = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ret), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });

    AtanhImpl<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, srcTensor.GetSize());
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void AtanhImpl(const LocalTensor<T> &dstTensor,
    const LocalTensor<T> &srcTensor,
    const LocalTensor<uint8_t> &sharedTmpBuffer)
{
    AtanhImpl<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, srcTensor.GetSize());
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void AtanhImpl(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
    const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ret = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ret), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });

    AtanhImpl<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, calCount);
}
} // namespace AscendC

#endif // IMPL_MATH_ATANH_ATANH_COMMON_IMPL_H