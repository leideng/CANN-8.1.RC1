/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef IMPL_MATH_FRAC_FRAC_COMMON_IMPL_H
#define IMPL_MATH_FRAC_FRAC_COMMON_IMPL_H
#include "kernel_tensor.h"
#include "../../common/check.h"
#if __CCE_AICORE__ == 200
#include "frac_v200_impl.h"
#elif __CCE_AICORE__ == 220
#include "frac_v220_impl.h"
#endif

namespace AscendC {
template <typename T>
__aicore__ inline void FracCompute(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<float>& tmpTensor, const uint32_t splitSize)
{
    // Get interger part of x = [x]
    TruncCastForFrac(dstTensor, srcTensor, dstTensor);

    Sub<T, false>(dstTensor, srcTensor, dstTensor, MASK_PLACEHOLDER, 1,
        { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
}

template <>
__aicore__ inline void FracCompute(const LocalTensor<half>& dstTensor, const LocalTensor<half>& srcTensor,
    const LocalTensor<float>& tmpTensor, const uint32_t splitSize)
{
    LocalTensor<float> srcTmpTensor = tmpTensor;
    LocalTensor<float> dstTmpTensor = tmpTensor[splitSize];

    // Cast src from half to float type for getting more precise results.
    Cast<float, half, false>(srcTmpTensor, srcTensor, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE / 2 });
    PipeBarrier<PIPE_V>();
    // Get interger part of x = [x]
    TruncCastForFrac(dstTmpTensor, srcTmpTensor, dstTmpTensor);

    Sub<float, false>(dstTmpTensor, srcTmpTensor, dstTmpTensor, MASK_PLACEHOLDER, 1,
        { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();

    Cast<half, float, false>(dstTensor, dstTmpTensor, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE / 2, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void FracImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }

    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");

    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Frac");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Frac");

    ASCENDC_ASSERT((std::is_same<T, float>::value || std::is_same<T, half>::value), {
        KERNEL_LOG( KERNEL_ERROR, "Failed to check the data types, current api support data types are half/float."); });

    uint32_t bufferSize = sharedTmpBuffer.GetSize();
    uint32_t tmpBufferSize = bufferSize / sizeof(float);
    CheckTmpBufferSize(tmpBufferSize, 0, bufferSize);
    LocalTensor<float> tmpBuffer = sharedTmpBuffer.ReinterpretCast<float>();
    uint32_t stackSize = 0;
    uint32_t round = 1;
    uint32_t tail = 0;
    constexpr uint8_t FRAC_HALF_CALC_PROCEDURE = 2;
    if constexpr (sizeof(T) == sizeof(half)) {
        stackSize = tmpBufferSize / FRAC_HALF_CALC_PROCEDURE / ONE_BLK_SIZE * ONE_BLK_SIZE; // 32-byte aligned
        CheckTmpBufferSize(stackSize, 0, bufferSize);
        round = calCount / stackSize;
        tail = calCount % stackSize;
        SetMaskCount();
        SetVectorMask<half, MaskMode::COUNTER>(0, stackSize);
    } else {
        SetMaskCount();
        SetVectorMask<half, MaskMode::COUNTER>(0, calCount);
    }

    uint32_t offset = 0;
    for (uint32_t i = 0; i < round; i++) {
        FracCompute(dstTensor[offset], srcTensor[offset], tmpBuffer, stackSize);
        offset = offset + stackSize;
    }
    if (tail != 0) {
        SetVectorMask<half, MaskMode::COUNTER>(0, tail);
        FracCompute(dstTensor[offset], srcTensor[offset], tmpBuffer, stackSize);
    }

    SetMaskNorm();
    ResetMask();
}

template <typename T>
__aicore__ inline void FracImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }

    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    FracImpl(dstTensor, srcTensor, sharedTmpBuffer, calCount);
}
} // namespace AscendC

#endif // IMPL_MATH_FRAC_FRAC_COMMON_IMPL_H