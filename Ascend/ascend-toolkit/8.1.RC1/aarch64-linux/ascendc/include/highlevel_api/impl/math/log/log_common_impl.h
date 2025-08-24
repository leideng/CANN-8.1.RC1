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
 * \file log_common_impl.h
 * \brief
 */
#ifndef IMPL_MATH_LOG_LOG_COMMON_IMPL_H
#define IMPL_MATH_LOG_LOG_COMMON_IMPL_H
#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../common/check.h"

#if __CCE_AICORE__ >= 200

namespace AscendC {
template <typename T>
__aicore__ inline void Log2Compute(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor)
{
    // Log2x = Lnx/Ln2
    const T Ln2Reciprocal = 1.4426950408889634; // 1.0/Ln2;
    const UnaryRepeatParams unaryParams;
    Ln<float, false>(dstTensor, srcTensor, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Muls<float, false>(dstTensor, dstTensor, Ln2Reciprocal, MASK_PLACEHOLDER, 1, unaryParams);
}

template <typename T>
__aicore__ inline void Log2Compute(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& tmpTensor)
{
    // Log2x = Lnx/Ln2
    const float Ln2Reciprocal = 1.4426950408889634; // 1.0/Ln2;
    const UnaryRepeatParams unaryParams;

    // src->tmp
    Cast<float, T, false>(tmpTensor.ReinterpretCast<float>(), srcTensor,
        RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();

    // tmp->tmp
    Ln<float, false>(tmpTensor.ReinterpretCast<float>(),
        tmpTensor.ReinterpretCast<float>(),
        MASK_PLACEHOLDER,
        1,
        unaryParams);
    PipeBarrier<PIPE_V>();
    Muls<float, false>(tmpTensor.ReinterpretCast<float>(),
        tmpTensor.ReinterpretCast<float>(),
        static_cast<float>(Ln2Reciprocal),
        MASK_PLACEHOLDER,
        1,
        unaryParams);
    PipeBarrier<PIPE_V>();

    // tmp->dst
    Cast<T, float, false>(dstTensor, tmpTensor.ReinterpretCast<float>(),
        RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, { 1, 1, HALF_DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void LogImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    uint32_t calCount)
{
    // Logx = Lnx
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");

    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Log");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Log");

    ASCENDC_ASSERT((std::is_same<T, float>::value || std::is_same<T, half>::value), {
        KERNEL_LOG( KERNEL_ERROR, "Failed to check the data types, current api support data types are half/float."); });

    const UnaryRepeatParams unaryParams;
    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(0, calCount);
    Ln<T, false>(dstTensor, srcTensor, MASK_PLACEHOLDER, 1, unaryParams);
    SetMaskNorm();
    SetVectorMask<half, MaskMode::NORMAL>(FULL_MASK, FULL_MASK);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void Log2Impl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)
{
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");

    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Log");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Log");

    ASCENDC_ASSERT((std::is_same<T, float>::value || std::is_same<T, half>::value), {
        KERNEL_LOG( KERNEL_ERROR, "Failed to check the data types, current api support data types are half/float."); });

    SetMaskCount();
    if constexpr (sizeof(T) == sizeof(float)) {
        SetVectorMask<T, MaskMode::COUNTER>(0, calCount);
        Log2Compute(dstTensor, srcTensor);
    } else {
        CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");

        uint32_t tmpBufferSize = sharedTmpBuffer.GetSize();
        uint32_t splitSize = tmpBufferSize / sizeof(float) / ONE_BLK_SIZE * ONE_BLK_SIZE;
        CheckTmpBufferSize(splitSize, 0, tmpBufferSize);
        uint32_t loopCount = calCount / splitSize;
        uint32_t calcTail = calCount % splitSize;
        SetVectorMask<T, MaskMode::COUNTER>(0, splitSize);
        for (uint32_t i = 0; i < loopCount; ++i) {
            Log2Compute(dstTensor[i * splitSize], srcTensor[i * splitSize], sharedTmpBuffer);
        }
        if (calcTail > 0) {
            SetVectorMask<T, MaskMode::COUNTER>(0, calcTail);
            Log2Compute(dstTensor[loopCount * splitSize], srcTensor[loopCount * splitSize], sharedTmpBuffer);
        }
    }
    SetMaskNorm();
    SetVectorMask<half, MaskMode::NORMAL>(FULL_MASK, FULL_MASK);
}


template <typename T, bool isReuseSource = false>
__aicore__ inline void Log10Impl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    uint32_t calCount)
{
    // Log10x = Lnx/Ln10
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");

    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Log");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Log");

    ASCENDC_ASSERT((std::is_same<T, float>::value || std::is_same<T, half>::value), {
        KERNEL_LOG( KERNEL_ERROR, "Failed to check the data types, current api support data types are half/float."); });
    const T Ln10Reciprocal = 0.43429448190325176; // 1.0/Ln10;
    const UnaryRepeatParams unaryParams;

    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(0, calCount);
    Ln<T, false>(dstTensor, srcTensor, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Muls<T, false>(dstTensor, dstTensor, Ln10Reciprocal, MASK_PLACEHOLDER, 1, unaryParams);
    SetMaskNorm();
    SetVectorMask<half, MaskMode::NORMAL>(FULL_MASK, FULL_MASK);
}
}  // namespace AscendC
#endif
#endif  // IMPL_MATH_LOG_LOG_COMMON_IMPL_H
