/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef IMPL_MATH_ACOS_ACOS_COMMON_IMPL_H
#define IMPL_MATH_ACOS_ACOS_COMMON_IMPL_H
#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"

#include "../math_common_impl.h"
#include "../asin/asin_common_impl.h"
#include "../../common/check.h"

namespace AscendC {
// Computes acos values based on input types.
// According formula: arccos(x) = PI*0.5 - arcsin(x).
template <typename T>
__aicore__ inline void AcosCompute(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<T>& tmpBuffer, uint32_t calSize)
{
    UnaryRepeatParams unaryParams;
    AsinCompute(dst, src, tmpBuffer, calSize);
    PipeBarrier<PIPE_V>();
    Adds<T, false>(dst, dst, static_cast<T>(-HALF_PI), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Muls<T, false>(dst, dst, static_cast<T>(NEG_ONE), MASK_PLACEHOLDER, 1, unaryParams);
}

template <>
__aicore__ inline void AcosCompute<half>(const LocalTensor<half>& dst, const LocalTensor<half>& src,
    const LocalTensor<half>& tmpBuffer, uint32_t calSize)
{
    UnaryRepeatParams unaryParams;
    // res = PI*0.5 - float(asin(x)) -> dst.
    AsinFp16Compute(dst, src, tmpBuffer, calSize);
    PipeBarrier<PIPE_V>();

    // Get float result of asin, which is implicitly stored in tmpBuffer, because using the half result of Asin is
    // unable to statisify Acos precision requirement.
    LocalTensor<float> tmpFloatBuffer1 = tmpBuffer.ReinterpretCast<float>();
    Adds<float, false>(tmpFloatBuffer1, tmpFloatBuffer1, -HALF_PI, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Muls<float, false>(tmpFloatBuffer1, tmpFloatBuffer1, NEG_ONE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Cast<half, float, false>(dst, tmpFloatBuffer1, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        { 1, 1, HALF_DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
}

template <typename T>
__aicore__ inline void AcosImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");
    
    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Acos");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Acos");

    ASCENDC_ASSERT((std::is_same<T, float>::value || std::is_same<T, half>::value), {
        KERNEL_LOG( KERNEL_ERROR, "Failed to check the data types, current api support data types are half/float."); });

    const uint32_t bufferSize = sharedTmpBuffer.GetSize();
    const uint32_t tmpBufferSize = bufferSize / sizeof(T);
    CheckTmpBufferSize(tmpBufferSize, 0, bufferSize);

    LocalTensor<T> tmpBuffer = sharedTmpBuffer.ReinterpretCast<T>();

    // For float input, the temporary variable is 2 floats.
    // For half input, the temporary variable is 2 floats + 2 halfs. The memory is equivalent to 6 halfs.
    uint32_t calSize = 0;
    if constexpr (sizeof(T) == sizeof(half)) {
        calSize = tmpBufferSize / ASIN_HALF_CALC_PROCEDURE / ONE_BLK_SIZE * ONE_BLK_SIZE;
    } else {
        calSize = tmpBufferSize / ASIN_FLOAT_CALC_PROCEDURE / ONE_BLK_SIZE * ONE_BLK_SIZE;
    }
    CheckTmpBufferSize(calSize, 0, bufferSize);

    const uint32_t round = calCount / calSize;
    const uint32_t tail = calCount % calSize;

    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(0, calSize);

    uint32_t offset = 0;
    for (uint32_t i = 0; i < round; i++) {
        AcosCompute(dstTensor[offset], srcTensor[offset], tmpBuffer, calSize);
        offset = offset + calSize;
    }

    if (tail != 0) {
        SetVectorMask<T, MaskMode::COUNTER>(0, tail);
        AcosCompute(dstTensor[offset], srcTensor[offset], tmpBuffer, calSize);
    }

    SetMaskNorm();
    ResetMask();
}

template <typename T>
__aicore__ inline void AcosImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
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
    AcosImpl(dstTensor, srcTensor, sharedTmpBuffer, calCount);
}
} // namespace AscendC
#endif // IMPL_MATH_ACOS_ACOS_COMMON_IMPL_H
