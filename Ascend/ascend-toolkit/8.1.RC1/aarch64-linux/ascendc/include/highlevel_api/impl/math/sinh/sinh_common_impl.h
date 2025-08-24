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
 * \file sinh_common_impl.h
 * \brief
 */
#ifndef IMPL_MATH_SINH_SINH_COMMON_IMPL_H
#define IMPL_MATH_SINH_SINH_COMMON_IMPL_H

#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "../../common/check.h"

#include "../math_common_impl.h"

namespace AscendC {
constexpr float SINH_NEG_LN_TWO = -0.69314718055994530941723212145818;
constexpr float SINH_NEG_ONE = -1.0;
constexpr float SINH_POINT_FIVE = 0.5;
constexpr uint32_t SINH_HALF_CALC_PROC = 4;
constexpr uint32_t SINH_FLOAT_CALC_PROC = 1;

// Computes sinh values based on input types.
template <typename T>
__aicore__ inline void SinhCompute(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<uint8_t>& tmpBuffer, uint32_t offset)
{
    UnaryRepeatParams unaryParams;
    BinaryRepeatParams binaryParams;

    const LocalTensor<T>& tmpFloatBuffer1 = tmpBuffer.ReinterpretCast<T>();

    // Calculates e^(-x-ln2)
    Muls<T, false>(tmpFloatBuffer1, src, static_cast<T>(SINH_NEG_ONE), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Adds<T, false>(tmpFloatBuffer1, tmpFloatBuffer1, static_cast<T>(SINH_NEG_LN_TWO), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Exp<T, false>(tmpFloatBuffer1, tmpFloatBuffer1, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // Calculates e^(x-ln2)
    Adds<T, false>(dst, src, static_cast<T>(SINH_NEG_LN_TWO), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Exp<T, false>(dst, dst, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // sinh(x) = e^(x-ln2) - e^(-x-ln2)
    Sub<T, false>(dst, dst, tmpFloatBuffer1, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
}

// Computes high precision sinh values for half type inputs by converting inputs to float types and save float sinh
// result at tmp Buffer.
// sinh(x) = e^(x-ln2) - e^(-x-ln2)
template <>
__aicore__ inline void SinhCompute(const LocalTensor<half>& dst, const LocalTensor<half>& src,
    const LocalTensor<uint8_t>& tmpBuffer, uint32_t offset)
{
    UnaryRepeatParams unaryParams;
    BinaryRepeatParams binaryParams;

    const LocalTensor<float>& tmpFloatBuffer1 = tmpBuffer.ReinterpretCast<float>();
    const LocalTensor<float>& tmpFloatBuffer2 = tmpFloatBuffer1[offset];

    Cast<float, half, false>(tmpFloatBuffer1, src, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();

    // Calculates e^(x-ln2)
    Adds<float, false>(tmpFloatBuffer2, tmpFloatBuffer1, SINH_NEG_LN_TWO, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Exp<float, false>(tmpFloatBuffer2, tmpFloatBuffer2, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // Calculates e^(-x-ln2)
    Muls<float, false>(tmpFloatBuffer1, tmpFloatBuffer1, SINH_NEG_ONE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Adds<float, false>(tmpFloatBuffer1, tmpFloatBuffer1, SINH_NEG_LN_TWO, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Exp<float, false>(tmpFloatBuffer1, tmpFloatBuffer1, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // sinh(x) = e^(x-ln2) - e^(-x-ln2)
    Sub<float, false>(tmpFloatBuffer2, tmpFloatBuffer2, tmpFloatBuffer1, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Cast<half, float, false>(dst, tmpFloatBuffer2, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        { 1, 1, HALF_DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void SinhImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");

    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Sinh");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Sinh");

    ASCENDC_ASSERT((std::is_same<T, float>::value || std::is_same<T, half>::value), {
        KERNEL_LOG( KERNEL_ERROR, "Failed to check the data types, current api support data types are half/float."); });
    uint32_t tmpBufferSize = sharedTmpBuffer.GetSize();
    uint32_t splitCount = tmpBufferSize / sizeof(T);

    if constexpr (sizeof(T) == sizeof(half)) {
        splitCount = splitCount / SINH_HALF_CALC_PROC / ONE_BLK_SIZE * ONE_BLK_SIZE;
    } else {
        splitCount = splitCount / SINH_FLOAT_CALC_PROC / ONE_BLK_SIZE * ONE_BLK_SIZE;
    }
    CheckTmpBufferSize(splitCount, 0, tmpBufferSize);

    const uint32_t loopCount = calCount / splitCount;
    const uint32_t calcTail = calCount % splitCount;

    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(0, splitCount);
    for (uint32_t i = 0; i < loopCount; ++i) {
        SinhCompute(dstTensor[i * splitCount], srcTensor[i * splitCount], sharedTmpBuffer, splitCount);
    }
    if (calcTail > 0) {
        SetVectorMask<T, MaskMode::COUNTER>(0, calcTail);
        SinhCompute(dstTensor[loopCount * splitCount], srcTensor[loopCount * splitCount], sharedTmpBuffer, splitCount);
    }
    SetMaskNorm();
    ResetMask();
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void SinhImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
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
    SinhImpl(dstTensor, srcTensor, sharedTmpBuffer, calCount);
}
} // namespace AscendC

#endif // IMPL_MATH_SINH_SINH_COMMON_IMPL_H
