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
 * \file sigmoid_common_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SIGMOID_SIGMOID_COMMON_IMPL_H
#define IMPL_ACTIVATION_SIGMOID_SIGMOID_COMMON_IMPL_H

#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tiling/kernel_tiling.h"

// V100 does not support counter mode. Thus different implementation is needed.
#if __CCE_AICORE__ == 220 || __CCE_AICORE__ == 200
#include "sigmoid_impl.h"
#elif __CCE_AICORE__ == 100
#include "sigmoid_v100_impl.h"
#endif

namespace AscendC {

template <typename T, bool isReuseSource = false>
__aicore__ inline void SigmoidImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    ASSERT(((TPosition)dstTensor.GetPosition() == TPosition::VECIN ||
            (TPosition)dstTensor.GetPosition() == TPosition::VECOUT ||
            (TPosition)dstTensor.GetPosition() == TPosition::VECCALC));
    ASCENDC_ASSERT((calCount <= srcTensor.GetSize()), {KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should not "
        "larger than srcTensor size %u", calCount, srcTensor.GetSize());});

    uint32_t splitSize = sharedTmpBuffer.GetSize() / sizeof(T);
    ASCENDC_ASSERT((splitSize != 0), {KERNEL_LOG(KERNEL_ERROR, "splitSize should not be 0!");});

    uint32_t loopCount = calCount / splitSize;
    uint32_t calcTail = calCount % splitSize;
    LocalTensor<T> tmpBuffer = sharedTmpBuffer.ReinterpretCast<T>();
    SigmoidCompute<T, isReuseSource>(dstTensor, srcTensor, tmpBuffer, splitSize, loopCount, calcTail);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void SigmoidImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    LocalTensor<uint8_t> stackBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(stackBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    SigmoidImpl<T, isReuseSource>(dstTensor, srcTensor, stackBuffer, calCount);
}
}  // namespace AscendC
#endif  // IMPL_ACTIVATION_SIGMOID_SIGMOID_COMMON_IMPL_H