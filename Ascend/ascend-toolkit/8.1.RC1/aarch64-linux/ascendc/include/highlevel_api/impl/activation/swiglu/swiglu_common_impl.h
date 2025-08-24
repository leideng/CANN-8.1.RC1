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
 * \file swiglu_common_impl.h
 * \brief
 */

#ifndef IMPL_ACTIVATION_SWIGLU_SWIGLU_COMMON_IMPL_H
#define IMPL_ACTIVATION_SWIGLU_SWIGLU_COMMON_IMPL_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"


namespace AscendC {
constexpr float NUMBER_ONE = 1.0;
constexpr uint32_t REPEAT_TIME_SWIGLU = 1;
constexpr uint32_t SWIGLU_HALF_BUFFER_SIZE = 3; // need 3 times of size of calCout<T> as tmpBuffer
constexpr uint32_t SWIGLU_FLOAT_TMP_BUFFER_SIZE = 0;
constexpr uint32_t SWIGLU_STRIDE_DIGITS = 2;

template <typename T, bool isReuseSource = false>
__aicore__ inline void SwiGLUImpl(LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor0,
    const LocalTensor<T> &srcTensor1, const float &scalarValue, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    // tmpBuffer is only used when dtype of srcTensor is half
    // and the result of the calculation does not match with the golden.
    LocalTensor<uint8_t> sharedTmpBuffer;
    PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    SwiGLUImpl<T, isReuseSource>(dstTensor, srcTensor0, srcTensor1, scalarValue, sharedTmpBuffer, calCount);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void SwiGLUImpl(LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor0,
    const LocalTensor<T> &srcTensor1, const float &scalarValue)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    // tmpBuffer is only used when dtype of srcTensor is half
    // and the result of the calculation does not match with the golden.
    LocalTensor<uint8_t> sharedTmpBuffer;
    PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    SwiGLUImpl<T, isReuseSource>(dstTensor, srcTensor0, srcTensor1, scalarValue, sharedTmpBuffer, srcTensor0.GetSize());
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void SwiGLUImpl(LocalTensor<T> &dstTensor, LocalTensor<T> &srcTensor0, LocalTensor<T> &srcTensor1,
                              const float &scalarValue)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    // tmpBuffer is only used when dtype of srcTensor is half
    // and the result of the calculation does not match with the golden.
    LocalTensor<uint8_t> sharedTmpBuffer;
    PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    SwiGLUImpl<T, isReuseSource>(dstTensor, srcTensor0, srcTensor1, scalarValue, sharedTmpBuffer, srcTensor0.GetSize());
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void SwiGLUImpl(LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor0,
                              const LocalTensor<T> &srcTensor1, const float &scalarValue,
                              const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    ASCENDC_ASSERT(((TPosition)dstTensor.GetPosition() == TPosition::VECIN ||
                    (TPosition)dstTensor.GetPosition() == TPosition::VECOUT ||
                    (TPosition)dstTensor.GetPosition() == TPosition::VECCALC),
                   { KERNEL_LOG(KERNEL_ERROR, "dst position not support");});
    ASCENDC_ASSERT((srcTensor0.GetSize() == srcTensor1.GetSize()),
        { KERNEL_LOG(KERNEL_ERROR, "Input params.GetSize must be equal with each other!"); });
    ASCENDC_ASSERT((srcTensor0.GetSize() >= calCount && dstTensor.GetSize() >= calCount),
        { KERNEL_LOG(KERNEL_ERROR, "Size of calCount should be less than or equal to size of Input and Output!"); });
    ASCENDC_ASSERT((calCount != 0 && (calCount * sizeof(T)) % ONE_BLK_SIZE == 0),
        { KERNEL_LOG(KERNEL_ERROR, "Size of calCount should be larger than 0 && should be 32B aligned!"); });
    ASCENDC_ASSERT(((sharedTmpBuffer.GetSize() * sizeof(uint8_t)) % ONE_BLK_SIZE == 0),
        { KERNEL_LOG(KERNEL_ERROR, "Size of sharedTmpBuffer should be 32B aligned!"); });
    ASCENDC_ASSERT((std::is_same<T, half>::value || std::is_same<T, float>::value),
        { KERNEL_LOG(KERNEL_ERROR, "type must be half or float!"); });

    LocalTensor<float> tmpBuffer = sharedTmpBuffer.ReinterpretCast<float>();
    tmpBuffer.SetSize(sharedTmpBuffer.GetSize() / sizeof(float));
    uint32_t stackSize = calCount;
    // only half needs shardTmpBuffer
    if (sizeof(T) == sizeof(half)) {
        // type of sharedTmpBuffer is uint8, the size of sharedTmpBuffer
        // is sizeof(float)/sizeof(uint8) times of stackSize
        stackSize = sharedTmpBuffer.GetSize() / sizeof(float) / SWIGLU_HALF_BUFFER_SIZE;
    }
    // buffer size of number of stackSize's data must be 32B aligned.
    stackSize = ((stackSize * sizeof(T)) / ONE_BLK_SIZE * ONE_BLK_SIZE) / sizeof(T);
    // in case stackSize is 0
    if (stackSize <= 0) {
        stackSize = ONE_BLK_SIZE / sizeof(T);
    }
    // calCount>tmpBuffer
    const uint32_t round = calCount / stackSize;
    const uint32_t tail = calCount % stackSize;

    SetMaskCount();
    SetVectorMask<T>(0, stackSize);

    uint32_t offset = 0;
    for (uint32_t i = 0; i < round; i++) {
        SwiGLUImpl(dstTensor[offset], srcTensor0[offset], srcTensor1[offset], scalarValue, tmpBuffer, stackSize);
        offset = offset + stackSize;
    }
    if (tail != 0) {
        // if size of tail is not 32B aligned
        bool isTail32BAligned = (tail * sizeof(T) % ONE_BLK_SIZE == 0);
        auto tail32BAligned = (tail * sizeof(T) / ONE_BLK_SIZE + (isTail32BAligned ? 0 : 1)) *
                              ONE_BLK_SIZE / sizeof(T);
        SetVectorMask<T>(0, tail);
        SwiGLUImpl(dstTensor[offset], srcTensor0[offset], srcTensor1[offset],
                   scalarValue, tmpBuffer, tail32BAligned);
    }

    SetMaskNorm();
    ResetMask();
}

template <typename T>
__aicore__ inline void SwishCalcSimplified(
   const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const float &scalarValue)
{
    // swish(x) = x / (1 + e^(-βx))
    // x1 = 1 + e^(-βx)
    const UnaryRepeatParams unaryParams;

    Muls<float, false>(dstTensor, srcTensor, scalarValue, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Exp<float, false>(dstTensor, dstTensor, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Adds<float, false>(dstTensor, dstTensor, static_cast<T>(1), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    const BinaryRepeatParams binaryParams;
    Div<float, false>(dstTensor, srcTensor, dstTensor, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void SwiGLUImpl(const LocalTensor<T> &dst, const LocalTensor<T> &src0, const LocalTensor<T> &src1,
                                  const float &beta, const LocalTensor<float> &sharedTmpBuffer, uint32_t calCount)
{
    // Calculate dstTensor = Swish(srcTensor1)
    float scalar = static_cast<float>(static_cast<float>(-1.0) * static_cast<float>(beta));
    SwishCalcSimplified(dst, src1, scalar);

    const BinaryRepeatParams binaryParams;
    // Calculate srcTensor0 Mul dstTensor
    Mul<float, false>(dst, src0, dst, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
}

template <>
__aicore__ inline void SwiGLUImpl<half>(const LocalTensor<half> &dst, const LocalTensor<half> &src0,
                                        const LocalTensor<half> &src1, const float &beta,
                                        const LocalTensor<float> &sharedTmpBuffer, uint32_t calCount)
{
    LocalTensor<float> tmpSrc1FloatBuffer1 = sharedTmpBuffer;
    LocalTensor<float> tmpSrc1FloatBuffer2 = sharedTmpBuffer[calCount];
    LocalTensor<float> tmpSrc0FloatBuffer = sharedTmpBuffer[2 * calCount];

    // src1(half) -> tmpSrc1FloatBuffer1(float)
    Cast<float, half, false>(tmpSrc1FloatBuffer1, src1, RoundMode::CAST_NONE, MASK_PLACEHOLDER,
        1, { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE / SWIGLU_STRIDE_DIGITS });
    PipeBarrier<PIPE_V>();

    // Calculate tmpSrc1FloatBuffer2 = Swish(tmpSrc1FloatBuffer1)
    float scalar = static_cast<float>(static_cast<float>(-1.0) * static_cast<float>(beta));
    SwishCalcSimplified(tmpSrc1FloatBuffer2, tmpSrc1FloatBuffer1, scalar);

    // src0(half) -> tmpSrc0FloatBuffer(float)
    Cast<float, half, false>(tmpSrc0FloatBuffer, src0, RoundMode::CAST_NONE, MASK_PLACEHOLDER,
        1, { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE / SWIGLU_STRIDE_DIGITS });
    PipeBarrier<PIPE_V>();

    const BinaryRepeatParams binaryParams;
    // Calculate tmpSrc1FloatBuffer2 Mul tmpSrc0FloatBuffer
    Mul<float, false>(tmpSrc1FloatBuffer2, tmpSrc0FloatBuffer, tmpSrc1FloatBuffer2, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    // tmpSrc1FloatBuffer2(float) -> dst(half)
    Cast<half, float, false>(dst, tmpSrc1FloatBuffer2, RoundMode::CAST_NONE, MASK_PLACEHOLDER,
        1, { 1, 1, DEFAULT_REPEAT_STRIDE / SWIGLU_STRIDE_DIGITS, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
}
}
#endif