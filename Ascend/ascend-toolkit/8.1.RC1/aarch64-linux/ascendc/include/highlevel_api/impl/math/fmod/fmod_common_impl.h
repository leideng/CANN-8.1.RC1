/* *
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/* !
 * \file fmod_common_impl.h
 * \brief
 */

#ifndef IMPL_MATH_FMOD_FMOD_COMMON_IMPL_H
#define IMPL_MATH_FMOD_FMOD_COMMON_IMPL_H
#include "kernel_tensor.h"
#include "../../common/check.h"

namespace AscendC {
namespace {
constexpr uint32_t SRC0_IDX = 1;
constexpr uint32_t SRC1_IDX = 2;
constexpr uint32_t TRUNC_IDX = 3;
}

__aicore__ inline void FmodCompute(const LocalTensor<float> &dstTensor, const LocalTensor<float> &src0Tensor,
    const LocalTensor<float> &src1Tensor, const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t stackSize,
    const uint32_t calCount)
{
    PipeBarrier<PIPE_V>();

    Div(dstTensor, src0Tensor, src1Tensor, calCount);
    PipeBarrier<PIPE_V>();

    Trunc(dstTensor, dstTensor, sharedTmpBuffer, calCount);
    PipeBarrier<PIPE_V>();

    Mul(dstTensor, dstTensor, src1Tensor, calCount);
    PipeBarrier<PIPE_V>();

    Sub(dstTensor, src0Tensor, dstTensor, calCount);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void FmodCompute(const LocalTensor<half> &dstTensor, const LocalTensor<half> &src0Tensor,
    const LocalTensor<half> &src1Tensor, const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t stackSize,
    const uint32_t calCount)
{
    // floatTmpTensor<float>    = | dst | src0 | src1 |
    // sharedTmpBuffer<uint8_t> = | dst | src0 | src1 | trunc |
    LocalTensor<float> floatTmpTensor = sharedTmpBuffer.ReinterpretCast<float>();
    LocalTensor<float> tmpSrc0 = floatTmpTensor[SRC0_IDX * stackSize]; // Allocate stackSize space
    LocalTensor<float> tmpSrc1 = floatTmpTensor[SRC1_IDX * stackSize];

    PipeBarrier<PIPE_V>();

    Cast<float, half>(tmpSrc0, src0Tensor, RoundMode::CAST_NONE, calCount);

    Cast<float, half>(tmpSrc1, src1Tensor, RoundMode::CAST_NONE, calCount);
    PipeBarrier<PIPE_V>();

    FmodCompute(floatTmpTensor, tmpSrc0, tmpSrc1, sharedTmpBuffer[TRUNC_IDX * stackSize * sizeof(float)], stackSize, calCount);

    Cast<half, float>(dstTensor, floatTmpTensor, RoundMode::CAST_NONE, calCount);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void FmodImpl(const LocalTensor<T> &dstTensor, const LocalTensor<T> &src0Tensor,
    const LocalTensor<T> &src1Tensor, const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }

    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(src0Tensor, "src0Tensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(src1Tensor, "src1Tensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");

    CheckCalCount(calCount, "calCount", src0Tensor, "src0Tensor", "Fmod");
    CheckCalCount(calCount, "calCount", src1Tensor, "src1Tensor", "Fmod");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Fmod");

    ASCENDC_ASSERT((std::is_same<T, float>::value || std::is_same<T, half>::value), {
        KERNEL_LOG( KERNEL_ERROR, "Failed to check the data types, current api support data types are half/float."); });

    ASCENDC_ASSERT((src0Tensor.GetSize() == src1Tensor.GetSize()),
                   { KERNEL_LOG(KERNEL_ERROR, "Input params.GetSize must be equal with each other!"); });

    if constexpr (sizeof(T) == sizeof(float)) {
        FmodCompute(dstTensor, src0Tensor, src1Tensor, sharedTmpBuffer, src0Tensor.GetSize(), calCount);
        return;
    }

    constexpr uint32_t maxLiveNodeCnt = 8; // The corresponding maxLiveNodeCnt for half is 8, extra is 3 * 2 + trunc 2.
    uint32_t bufferSize = sharedTmpBuffer.GetSize();
    uint32_t stackSize =
        bufferSize / sizeof(T) / maxLiveNodeCnt / ONE_BLK_SIZE * ONE_BLK_SIZE; // divided by how many counts
    CheckTmpBufferSize(stackSize, 0, bufferSize);
    ASCENDC_ASSERT((src0Tensor.GetSize() > 0), { KERNEL_LOG(KERNEL_ERROR, "src0Tensor size must > 0!"); });
    stackSize = stackSize > src0Tensor.GetSize() ? src0Tensor.GetSize() : stackSize; // No more than localTensor

    const uint32_t round = calCount / stackSize;
    const uint32_t tail = calCount % stackSize;

    for (uint32_t i = 0; i < round; ++i) {
        FmodCompute(dstTensor[i * stackSize], src0Tensor[i * stackSize], src1Tensor[i * stackSize], sharedTmpBuffer,
            stackSize, stackSize);
    }
    if (tail > 0) {
        FmodCompute(dstTensor[round * stackSize], src0Tensor[round * stackSize], src1Tensor[round * stackSize],
            sharedTmpBuffer, stackSize, tail);
    }
}
} // namespace AscendC
#endif // IMPL_MATH_FMOD_FMOD_COMMON_IMPL_H