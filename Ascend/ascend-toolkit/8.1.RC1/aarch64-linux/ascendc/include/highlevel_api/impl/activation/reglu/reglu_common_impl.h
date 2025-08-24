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
 * \file reglu_common_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_REGLU_REGLU_COMMON_IMPL_H
#define IMPL_ACTIVATION_REGLU_REGLU_COMMON_IMPL_H

#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tiling/kernel_tiling.h"

#if __CCE_AICORE__ == 220
#include "reglu_v220_impl.h"
#elif __CCE_AICORE__ == 200
#include "reglu_v200_impl.h"
#endif

namespace AscendC {
const uint8_t REGLU_HALF_CALC_PROCEDURE = 3;
const uint32_t REGLU_TEMP_BUFFER_OFFSET = 2U;

__aicore__ inline void Compute(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor0,
    const LocalTensor<float>& srcTensor1)
{
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;
    // max(0, x)
    Maxs<float, false>(dstTensor, srcTensor1, static_cast<float>(0), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Mul<float, false>(dstTensor, srcTensor0, dstTensor, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void ReGluCompute(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0,
    const LocalTensor<T>& srcTensor1, const LocalTensor<float>& tmpTensor, const uint32_t splitSize)
{
    const LocalTensor<float>& x0CastBuffer = tmpTensor;
    const LocalTensor<float>& x1CastBuffer = tmpTensor[splitSize];
    const LocalTensor<float>& yCastBuffer = tmpTensor[splitSize * REGLU_TEMP_BUFFER_OFFSET];

    Cast<float, T, false>(x0CastBuffer, srcTensor0, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE });
    Cast<float, T, false>(x1CastBuffer, srcTensor1, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();

    Compute(yCastBuffer, x0CastBuffer, x1CastBuffer);
    ReGluCast(dstTensor, yCastBuffer);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void ReGluImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0,
    const LocalTensor<T>& srcTensor1, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    ASCENDC_ASSERT((isReuseSource == false),
        { KERNEL_LOG(KERNEL_ERROR, "isReuseSource value only support false"); });
    ASCENDC_ASSERT((calCount <= srcTensor0.GetSize()), { KERNEL_LOG(KERNEL_ERROR, "calCount must <= srcSize!"); });

    ASCENDC_ASSERT(((TPosition)dstTensor.GetPosition() == TPosition::VECIN ||
                    (TPosition)dstTensor.GetPosition() == TPosition::VECOUT ||
                    (TPosition)dstTensor.GetPosition() == TPosition::VECCALC), { KERNEL_LOG(KERNEL_ERROR,
        "dst position not support, just support position is VECIN, VECOUT, VECCALC.");});

    SetMaskCount();
    if constexpr (IsSameType<T, float>::value) {
        SetVectorMask<T, MaskMode::COUNTER>(0, calCount);
        Compute(dstTensor, srcTensor0, srcTensor1);
    } else {
        uint32_t tmpBufferSize = sharedTmpBuffer.GetSize() / sizeof(float);
        ASCENDC_ASSERT((tmpBufferSize > 0), { KERNEL_LOG(KERNEL_ERROR, "tmpBufferSize must > 0!"); });
        LocalTensor<float> tmpBuffer;
        tmpBuffer = sharedTmpBuffer.ReinterpretCast<float>();
        uint32_t stackSize = 0;

        stackSize = tmpBufferSize / REGLU_HALF_CALC_PROCEDURE / ONE_BLK_SIZE * ONE_BLK_SIZE;
        ASCENDC_ASSERT((stackSize > 0), { KERNEL_LOG(KERNEL_ERROR, "stackSize must > 0!"); });

        const uint32_t round = calCount / stackSize;
        const uint32_t tail = calCount %  stackSize;
        SetVectorMask<T, MaskMode::COUNTER>(0, stackSize);
        uint32_t offset = 0;

        for (uint32_t i = 0; i < round; i++) {
            ReGluCompute(dstTensor[offset], srcTensor0[offset], srcTensor1[offset], tmpBuffer, stackSize);
            offset = offset + stackSize;
        }

        if (tail != 0) {
            SetVectorMask<T, MaskMode::COUNTER>(0, tail);
            ReGluCompute(dstTensor[offset], srcTensor0[offset], srcTensor1[offset], tmpBuffer, stackSize);
        }
    }
    SetMaskNorm();
    ResetMask();
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void ReGluImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0,
    const LocalTensor<T>& srcTensor1, const uint32_t calCount)
{
    // Using the stack space to allocate tmpbuf
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool hasStackBuffer = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((hasStackBuffer), {KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!");});
    ReGluImpl<T, isReuseSource>(dstTensor, srcTensor0, srcTensor1, sharedTmpBuffer, calCount);
}
}  // namespace AscendC
#endif  // IMPL_ACTIVATION_REGLU_REGLU_COMMON_IMPL_H