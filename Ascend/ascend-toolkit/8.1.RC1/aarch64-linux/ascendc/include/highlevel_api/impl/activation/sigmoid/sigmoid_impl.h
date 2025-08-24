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
 * \file sigmoid_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SIGMOID_SIGMOID_IMPL_H
#define IMPL_ACTIVATION_SIGMOID_SIGMOID_IMPL_H
#include "kernel_tensor.h"
#include "kernel_tiling/kernel_tiling.h"

namespace AscendC {
template <typename T>
__aicore__ inline void SigmoidIntrinsicsImpl(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<T>& stackBuffer)
{
    struct UnaryRepeatParams repeatParams;
    struct BinaryRepeatParams binaryRepeatParams;
    PipeBarrier<PIPE_V>();
    Muls<T, false>(dst, src, static_cast<T>(-1.0), MASK_PLACEHOLDER, 1, repeatParams);
    PipeBarrier<PIPE_V>();
    Exp<T, false>(dst, dst, MASK_PLACEHOLDER, 1, repeatParams);
    PipeBarrier<PIPE_V>();
    Adds<T, false>(dst, dst, static_cast<T>(1), MASK_PLACEHOLDER, 1, repeatParams);
    Duplicate<T, false>(stackBuffer, static_cast<T>(1.0), MASK_PLACEHOLDER, 1,
        DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
    Div<T, false>(dst, stackBuffer, dst, MASK_PLACEHOLDER, 1, binaryRepeatParams);
    PipeBarrier<PIPE_V>();
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void SigmoidCompute(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<T>& sharedTmpBuffer, const uint32_t splitSize, const uint32_t loopCount, const uint32_t calcTail)
{
    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(0, splitSize);
    for (uint32_t i = 0; i < loopCount; ++i) {
        SigmoidIntrinsicsImpl(dstTensor[i * splitSize], srcTensor[i * splitSize], sharedTmpBuffer);
    }
    if (calcTail > 0) {
        SetVectorMask<T, MaskMode::COUNTER>(0, calcTail);
        SigmoidIntrinsicsImpl(dstTensor[loopCount * splitSize], srcTensor[loopCount * splitSize], sharedTmpBuffer);
    }

    SetMaskNorm();
    ResetMask();
}
} //  namespace AscendC
#endif // IMPL_ACTIVATION_SIGMOID_SIGMOID_IMPL_H
