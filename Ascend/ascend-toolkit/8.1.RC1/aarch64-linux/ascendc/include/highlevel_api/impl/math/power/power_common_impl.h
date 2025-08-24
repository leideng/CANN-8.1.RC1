/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file power_common_impl.h
 * \brief
 */
#ifndef IMPL_MATH_POWER_POWER_COMMON_IMPL_H
#define IMPL_MATH_POWER_POWER_COMMON_IMPL_H
#include "kernel_pop_stack_buffer.h"
#include "power_int_impl.h"
#include "power_float_impl.h"
#include "../../common/check.h"

namespace AscendC {
// PowerImpl(tensor, tensor) half input
__aicore__ inline void PowerImpl(const LocalTensor<half>& dstTensor, const LocalTensor<half>& srcTensor0,
    const LocalTensor<half>& srcTensor1, const LocalTensor<uint8_t>& stackTensor, uint32_t calCount)
{
    constexpr uint32_t tripleFactor = 3;
    uint32_t tmpBufferSize = stackTensor.GetSize();
    CheckTmpBufferSize(tmpBufferSize, ONE_REPEAT_BYTE_SIZE, tmpBufferSize);

    uint32_t splitSize = (tmpBufferSize - ONE_REPEAT_BYTE_SIZE) / sizeof(float) /
        TENSOR_TENSOR_HALF / ONE_BLK_SIZE * ONE_BLK_SIZE;
    CheckTmpBufferSize(splitSize, 0, tmpBufferSize);

    LocalTensor<float> tmpScalar = stackTensor.ReinterpretCast<float>();
    tmpScalar.SetSize(ONE_REPEAT_BYTE_SIZE / sizeof(float));
    LocalTensor<float> stackSrc0 = stackTensor[ONE_REPEAT_BYTE_SIZE].ReinterpretCast<float>();
    LocalTensor<float> stackSrc1 = stackSrc0[splitSize];
    LocalTensor<float> stackDst = stackSrc1[splitSize];
    stackSrc1.SetSize(splitSize);
    stackSrc0.SetSize(splitSize);
    stackDst.SetSize(splitSize);
    AscPowerFParams powerParam;
    PowerFParamsCalc(
        stackTensor[ONE_REPEAT_BYTE_SIZE + tripleFactor * splitSize * sizeof(float)].ReinterpretCast<float>(),
        powerParam, splitSize);
    InitTmpScalar(tmpScalar);
    SetVectorMask<half>(0, splitSize);
    uint32_t loopCount = calCount / splitSize;
    uint32_t calcTail = calCount % splitSize;
    struct UnaryRepeatParams fp162fp32Param(1, 1, DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE);
    struct UnaryRepeatParams fp322fp16Param(1, 1, HALF_DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
    for (uint32_t i = 0; i < loopCount; ++i) {
        Cast<float, half, false>(
            stackSrc0, srcTensor0[i * splitSize], RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, fp162fp32Param);
        Cast<float, half, false>(
            stackSrc1, srcTensor1[i * splitSize], RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, fp162fp32Param);
        PipeBarrier<PIPE_V>();
        CommonPowerF(stackDst, stackSrc0, stackSrc1, tmpScalar, powerParam, splitSize);
        Cast<half, float, false>(
            dstTensor[i * splitSize], stackDst, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, fp322fp16Param);
        PipeBarrier<PIPE_V>();
    }
    if (calcTail > 0) {
        SetVectorMask<half>(0, calcTail);
        Cast<float, half, false>(
            stackSrc0, srcTensor0[loopCount * splitSize], RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, fp162fp32Param);
        Cast<float, half, false>(
            stackSrc1, srcTensor1[loopCount * splitSize], RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, fp162fp32Param);
        PipeBarrier<PIPE_V>();
        CommonPowerF(stackDst, stackSrc0, stackSrc1, tmpScalar, powerParam, calcTail);
        Cast<half, float, false>(
            dstTensor[loopCount * splitSize], stackDst, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, fp322fp16Param);
        PipeBarrier<PIPE_V>();
    }
}

// PowerImpl(tensor, tensor) float input
__aicore__ inline void PowerImpl(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor0,
    const LocalTensor<float>& srcTensor1, const LocalTensor<uint8_t>& stackTensor, uint32_t calCount)
{
    uint32_t tmpBufferSize = stackTensor.GetSize();
    CheckTmpBufferSize(tmpBufferSize, ONE_REPEAT_BYTE_SIZE, tmpBufferSize);

    uint32_t splitSize = (tmpBufferSize - ONE_REPEAT_BYTE_SIZE) / sizeof(float) /
        TENSOR_TENSOR_FLOAT / ONE_BLK_SIZE * ONE_BLK_SIZE;
    CheckTmpBufferSize(splitSize, 0, tmpBufferSize);

    LocalTensor<float> tmpScalar = stackTensor.ReinterpretCast<float>();
    tmpScalar.SetSize(ONE_REPEAT_BYTE_SIZE / sizeof(float));
    AscPowerFParams powerParam;
    PowerFParamsCalc(stackTensor[ONE_REPEAT_BYTE_SIZE].ReinterpretCast<float>(), powerParam, splitSize);
    InitTmpScalar(tmpScalar);
    SetVectorMask<float>(0, splitSize);
    uint32_t loopCount = calCount / splitSize;
    uint32_t calcTail = calCount % splitSize;
    for (uint32_t i = 0; i < loopCount; ++i) {
        PipeBarrier<PIPE_V>();
        CommonPowerF(dstTensor[i * splitSize], srcTensor0[i * splitSize],
            srcTensor1[i * splitSize], tmpScalar, powerParam, splitSize);
    }
    if (calcTail > 0) {
        SetVectorMask<float>(0, calcTail);
        CommonPowerF(dstTensor[loopCount * splitSize], srcTensor0[loopCount * splitSize],
            srcTensor1[loopCount * splitSize], tmpScalar, powerParam, calcTail);
    }
}

// PowerImpl(tensor, tensor) int32_t input
__aicore__ inline void PowerImpl(const LocalTensor<int32_t>& dstTensor, const LocalTensor<int32_t>& srcTensor0,
    const LocalTensor<int32_t>& srcTensor1, const LocalTensor<uint8_t>& stackTensor, uint32_t calCount)
{
    uint32_t tmpBufferSize = stackTensor.GetSize();
    uint32_t splitSize = tmpBufferSize / sizeof(int32_t) /
        TENSOR_TENSOR_INT / ONE_BLK_SIZE * ONE_BLK_SIZE;
    CheckTmpBufferSize(splitSize, 0, tmpBufferSize);

    AscPowerIParams powerParam;
    PowerIParamsCalc(stackTensor, powerParam, splitSize);
    SetVectorMask<int32_t>(0, splitSize);
    uint32_t loopCount = calCount / splitSize;
    uint32_t calcTail = calCount % splitSize;
    for (uint32_t i = 0; i < loopCount; ++i) {
        PipeBarrier<PIPE_V>();
        CommonPowerI(dstTensor[i * splitSize], srcTensor0[i * splitSize], srcTensor1[i * splitSize],
            powerParam, splitSize);
    }
    if (calcTail > 0) {
        SetVectorMask<int32_t>(0, calcTail);
        PipeBarrier<PIPE_V>();
        CommonPowerI(dstTensor[loopCount * splitSize], srcTensor0[loopCount * splitSize],
            srcTensor1[loopCount * splitSize], powerParam, calcTail);
        PipeBarrier<PIPE_V>();
    }
}

// PowerImpl(tensor, scalar) half input
__aicore__ inline void PowerImpl(const LocalTensor<half>& dstTensor, const LocalTensor<half>& srcTensor0,
    const half& scalarValue, const LocalTensor<uint8_t>& stackTensor, uint32_t calCount)
{
    constexpr uint32_t tripleFactor = 3;
    uint32_t tmpBufferSize = stackTensor.GetSize();
    CheckTmpBufferSize(tmpBufferSize, ONE_REPEAT_BYTE_SIZE, tmpBufferSize);

    uint32_t splitSize = (tmpBufferSize - ONE_REPEAT_BYTE_SIZE) / sizeof(float) / TENSOR_SCALAR_HALF /
        ONE_BLK_SIZE * ONE_BLK_SIZE;
    CheckTmpBufferSize(splitSize, 0, tmpBufferSize);

    LocalTensor<float> tmpScalar = stackTensor.ReinterpretCast<float>();
    tmpScalar.SetSize(ONE_REPEAT_BYTE_SIZE / sizeof(float));
    LocalTensor<float> stackSrc0 = stackTensor[ONE_REPEAT_BYTE_SIZE].ReinterpretCast<float>();
    LocalTensor<float> stackSrc1 = stackSrc0[splitSize];
    LocalTensor<float> stackDst = stackSrc1[splitSize];
    stackDst.SetSize(splitSize);
    stackSrc0.SetSize(splitSize);
    stackSrc1.SetSize(splitSize);
    AscPowerFParams powerParam;
    PowerFParamsCalc(
        stackTensor[ONE_REPEAT_BYTE_SIZE + tripleFactor * splitSize * sizeof(float)].ReinterpretCast<float>(),
        powerParam, splitSize);
    InitTmpScalar(tmpScalar);
    SetVectorMask<half>(0, splitSize);
    Duplicate<float, false>(stackSrc1, static_cast<float>(scalarValue), MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
    uint32_t loopCount = calCount / splitSize;
    uint32_t calcTail = calCount % splitSize;
    struct UnaryRepeatParams fp162fp32Param(1, 1, DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE);
    struct UnaryRepeatParams fp322fp16Param(1, 1, HALF_DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
    for (uint32_t i = 0; i < loopCount; ++i) {
        Cast<float, half, false>(
            stackSrc0, srcTensor0[i * splitSize], RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, fp162fp32Param);
        PipeBarrier<PIPE_V>();
        CommonPowerF(stackDst, stackSrc0, stackSrc1, tmpScalar, powerParam, splitSize);
        Cast<half, float, false>(
            dstTensor[i * splitSize], stackDst, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, fp322fp16Param);
        PipeBarrier<PIPE_V>();
    }
    if (calcTail > 0) {
        SetVectorMask<half>(0, calcTail);
        Cast<float, half, false>(
            stackSrc0, srcTensor0[loopCount * splitSize], RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, fp162fp32Param);
        PipeBarrier<PIPE_V>();
        CommonPowerF(stackDst, stackSrc0, stackSrc1, tmpScalar, powerParam, calcTail);
        Cast<half, float, false>(
            dstTensor[loopCount * splitSize], stackDst, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, fp322fp16Param);
        PipeBarrier<PIPE_V>();
    }
}

// PowerImpl(tensor, scalar) float input
__aicore__ inline void PowerImpl(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor0,
    const float& scalarValue, const LocalTensor<uint8_t>& stackTensor, uint32_t calCount)
{
    uint32_t tmpBufferSize = stackTensor.GetSize();
    CheckTmpBufferSize(tmpBufferSize, ONE_REPEAT_BYTE_SIZE, tmpBufferSize);
    uint32_t splitSize = (tmpBufferSize - ONE_REPEAT_BYTE_SIZE) / sizeof(float) /
        TENSOR_SCALAR_FLOAT / ONE_BLK_SIZE * ONE_BLK_SIZE;
    CheckTmpBufferSize(splitSize, 0, tmpBufferSize);

    LocalTensor<float> tmpScalar = stackTensor.ReinterpretCast<float>();
    tmpScalar.SetSize(ONE_REPEAT_BYTE_SIZE / sizeof(float));
    LocalTensor<float> stackSrc1 = stackTensor[ONE_REPEAT_BYTE_SIZE].ReinterpretCast<float>();
    stackSrc1.SetSize(splitSize);
    AscPowerFParams powerParam;
    PowerFParamsCalc(stackTensor[ONE_REPEAT_BYTE_SIZE + splitSize * sizeof(float)].ReinterpretCast<float>(),
        powerParam, splitSize);
    InitTmpScalar(tmpScalar);
    SetVectorMask<float>(0, splitSize);
    PipeBarrier<PIPE_V>();
    Duplicate<float, false>(stackSrc1, scalarValue, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
    uint32_t loopCount = calCount / splitSize;
    uint32_t calcTail = calCount % splitSize;
    for (uint32_t i = 0; i < loopCount; ++i) {
        CommonPowerF(dstTensor[i * splitSize], srcTensor0[i * splitSize],
            stackSrc1, tmpScalar, powerParam, splitSize);
    }
    if (calcTail > 0) {
        SetVectorMask<float>(0, calcTail);
        CommonPowerF(dstTensor[loopCount * splitSize], srcTensor0[loopCount * splitSize],
            stackSrc1, tmpScalar, powerParam, calcTail);
    }
}

// PowerImpl(tensor, scalar) int32_t input
__aicore__ inline void PowerImpl(const LocalTensor<int32_t>& dstTensor, const LocalTensor<int32_t>& srcTensor0,
    const int32_t& scalarValue, const LocalTensor<uint8_t>& stackTensor, uint32_t calCount)
{
    uint32_t tmpBufferSize = stackTensor.GetSize();
    uint32_t splitSize = tmpBufferSize / sizeof(int32_t) /
        TENSOR_SCALAR_INT / ONE_BLK_SIZE * ONE_BLK_SIZE;
    CheckTmpBufferSize(splitSize, 0, tmpBufferSize);

    LocalTensor<int32_t> stackSrc1 = stackTensor.ReinterpretCast<int32_t>();
    stackSrc1.SetSize(splitSize);
    AscPowerIParams powerParam;
    PowerIParamsCalc(stackTensor[splitSize * sizeof(int32_t)], powerParam, splitSize);
    SetVectorMask<int32_t>(0, splitSize);
    uint32_t loopCount = calCount / splitSize;
    uint32_t calcTail = calCount % splitSize;
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t, false>(stackSrc1, scalarValue, MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < loopCount; ++i) {
        CommonPowerI(dstTensor[i * splitSize], srcTensor0[i * splitSize], stackSrc1,
            powerParam, splitSize);
        PipeBarrier<PIPE_V>();
    }
    if (calcTail > 0) {
        SetVectorMask<int32_t>(0, calcTail);
        CommonPowerI(dstTensor[loopCount * splitSize], srcTensor0[loopCount * splitSize], stackSrc1,
            powerParam, calcTail);
        PipeBarrier<PIPE_V>();
    }
}

// PowerImpl(scalar, tensor) half input
__aicore__ inline void PowerImpl(const LocalTensor<half>& dstTensor, const half& scalarValue,
    const LocalTensor<half>& srcTensor1, const LocalTensor<uint8_t>& stackTensor, uint32_t calCount)
{
    constexpr uint32_t tripleFactor = 3;
    uint32_t tmpBufferSize = stackTensor.GetSize();
    CheckTmpBufferSize(tmpBufferSize, ONE_REPEAT_BYTE_SIZE, tmpBufferSize);

    uint32_t splitSize = (tmpBufferSize - ONE_REPEAT_BYTE_SIZE) / sizeof(float) /
        TENSOR_SCALAR_HALF / ONE_BLK_SIZE * ONE_BLK_SIZE;
    CheckTmpBufferSize(splitSize, 0, tmpBufferSize);

    LocalTensor<float> tmpScalar = stackTensor.ReinterpretCast<float>();
    tmpScalar.SetSize(ONE_REPEAT_BYTE_SIZE / sizeof(float));
    LocalTensor<float> stackSrc0 = stackTensor[ONE_REPEAT_BYTE_SIZE].ReinterpretCast<float>();
    LocalTensor<float> stackSrc1 = stackSrc0[splitSize];
    LocalTensor<float> stackDst = stackSrc1[splitSize];
    stackSrc0.SetSize(splitSize);
    stackSrc1.SetSize(splitSize);
    stackDst.SetSize(splitSize);
    AscPowerFParams powerParam;
    PowerFParamsCalc(
        stackTensor[ONE_REPEAT_BYTE_SIZE + tripleFactor * splitSize * sizeof(float)].ReinterpretCast<float>(),
        powerParam, splitSize);
    InitTmpScalar(tmpScalar);
    SetVectorMask<half>(0, splitSize);
    PipeBarrier<PIPE_V>();
    Duplicate<float, false>(stackSrc0, static_cast<float>(scalarValue), MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
    uint32_t loopCount = calCount / splitSize;
    uint32_t calcTail = calCount % splitSize;
    struct UnaryRepeatParams fp162fp32Param(1, 1, DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE);
    struct UnaryRepeatParams fp322fp16Param(1, 1, HALF_DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
    for (uint32_t i = 0; i < loopCount; ++i) {
        Cast<float, half, false>(
            stackSrc1, srcTensor1[i * splitSize], RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, fp162fp32Param);
        PipeBarrier<PIPE_V>();
        CommonPowerF(stackDst, stackSrc0, stackSrc1, tmpScalar, powerParam, splitSize);
        Cast<half, float, false>(
            dstTensor[i * splitSize], stackDst, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, fp322fp16Param);
        PipeBarrier<PIPE_V>();
    }
    if (calcTail > 0) {
        SetVectorMask<half>(0, calcTail);
        Cast<float, half, false>(
            stackSrc1, srcTensor1[loopCount * splitSize], RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, fp162fp32Param);
        PipeBarrier<PIPE_V>();
        CommonPowerF(stackDst, stackSrc0, stackSrc1, tmpScalar, powerParam, calcTail);
        Cast<half, float, false>(
            dstTensor[loopCount * splitSize], stackDst, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, fp322fp16Param);
        PipeBarrier<PIPE_V>();
    }
}

// PowerImpl(scalar, tensor) float input
__aicore__ inline void PowerImpl(const LocalTensor<float>& dstTensor, const float& scalarValue,
    const LocalTensor<float>& srcTensor1, const LocalTensor<uint8_t>& stackTensor, uint32_t calCount)
{
    uint32_t tmpBufferSize = stackTensor.GetSize();
    CheckTmpBufferSize(tmpBufferSize, ONE_REPEAT_BYTE_SIZE, tmpBufferSize);

    uint32_t splitSize = (tmpBufferSize - ONE_REPEAT_BYTE_SIZE) / sizeof(float) /
        TENSOR_SCALAR_FLOAT / ONE_BLK_SIZE * ONE_BLK_SIZE;
    CheckTmpBufferSize(splitSize, 0, tmpBufferSize);

    LocalTensor<float> tmpScalar = stackTensor.ReinterpretCast<float>();
    tmpScalar.SetSize(ONE_REPEAT_BYTE_SIZE / sizeof(float));
    LocalTensor<float> stackSrc0 = stackTensor[ONE_REPEAT_BYTE_SIZE].ReinterpretCast<float>();
    stackSrc0.SetSize(splitSize);
    AscPowerFParams powerParam;
    PowerFParamsCalc(stackTensor[ONE_REPEAT_BYTE_SIZE + splitSize * sizeof(float)].ReinterpretCast<float>(),
        powerParam, splitSize);
    InitTmpScalar(tmpScalar);
    SetVectorMask<float>(0, splitSize);
    PipeBarrier<PIPE_V>();
    Duplicate<float, false>(stackSrc0, scalarValue, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
    uint32_t loopCount = calCount / splitSize;
    uint32_t calcTail = calCount % splitSize;
    for (uint32_t i = 0; i < loopCount; ++i) {
        CommonPowerF(dstTensor[i * splitSize], stackSrc0, srcTensor1[i * splitSize],
            tmpScalar, powerParam, splitSize);
    }
    if (calcTail > 0) {
        SetVectorMask<float>(0, calcTail);
        CommonPowerF(dstTensor[loopCount * splitSize], stackSrc0, srcTensor1[loopCount * splitSize],
            tmpScalar, powerParam, calcTail);
    }
}

// PowerImpl(scalar, tensor) int32_t input
__aicore__ inline void PowerImpl(const LocalTensor<int32_t>& dstTensor, const int32_t& scalarValue,
    const LocalTensor<int32_t>& srcTensor1, const LocalTensor<uint8_t>& stackTensor, uint32_t calCount)
{
    uint32_t tmpBufferSize = stackTensor.GetSize();
    uint32_t splitSize = tmpBufferSize / sizeof(int32_t) /
        TENSOR_SCALAR_INT / ONE_BLK_SIZE * ONE_BLK_SIZE;
    CheckTmpBufferSize(splitSize, 0, tmpBufferSize);

    LocalTensor<int32_t> stackSrc0 = stackTensor.ReinterpretCast<int32_t>();
    stackSrc0.SetSize(splitSize);
    AscPowerIParams powerParam;
    PowerIParamsCalc(stackTensor[splitSize * sizeof(int32_t)], powerParam, splitSize);
    SetVectorMask<int32_t>(0, splitSize);
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t, false>(stackSrc0, scalarValue, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
    uint32_t loopCount = calCount / splitSize;
    uint32_t calcTail = calCount % splitSize;
    for (uint32_t i = 0; i < loopCount; ++i) {
        CommonPowerI(dstTensor[i * splitSize], stackSrc0, srcTensor1[i * splitSize],
            powerParam, splitSize);
        PipeBarrier<PIPE_V>();
    }
    if (calcTail > 0) {
        SetVectorMask<int32_t>(0, calcTail);
        CommonPowerI(dstTensor[loopCount * splitSize], stackSrc0, srcTensor1[loopCount * splitSize],
            powerParam, calcTail);
        PipeBarrier<PIPE_V>();
    }
}

/*
Power(scalar, tensor) Converts scalar to tensor for processing.
The interface input parameter contains tmpTensor.
*/
template <typename T, bool isReuseSource = false>
__aicore__ inline void PowerCommonImpl(const LocalTensor<T>& dstTensor, const T& scalarValue,
    const LocalTensor<T>& srcTensor1, const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }

    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor1, "srcTensor1", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");

    CheckCalCount(calCount, "calCount", srcTensor1, "srcTensor1", "Power");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Power");

    ASCENDC_ASSERT((std::is_same<T, float>::value || std::is_same<T, half>::value || std::is_same<T, int32_t>::value), {
        KERNEL_LOG( KERNEL_ERROR,
            "Failed to check the data types, current api support data types are half/float/int32_t."); });

    SetMaskCount();
    PowerImpl(dstTensor, scalarValue, srcTensor1, sharedTmpBuffer, calCount);
    SetMaskNorm();
    ResetMask();
}

/*
Power(tensor, scalar) Converts scalar to tensor for processing.
The interface input parameter contains tmpTensor.
*/
template <typename T, bool isReuseSource = false>
__aicore__ inline void PowerCommonImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0,
    const T& scalarValue, const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor0, "srcTensor0", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");

    CheckCalCount(calCount, "calCount", srcTensor0, "srcTensor0", "Power");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Power");

    ASCENDC_ASSERT((std::is_same<T, float>::value || std::is_same<T, half>::value || std::is_same<T, int32_t>::value), {
        KERNEL_LOG( KERNEL_ERROR,
            "Failed to check the data types, current api support data types are half/float/int32_t."); });

    SetMaskCount();
    PowerImpl(dstTensor, srcTensor0, scalarValue, sharedTmpBuffer, calCount);
    SetMaskNorm();
    ResetMask();
}

/*
Power(tensor, tensor)
The interface input parameter contains tmpTensor.
*/
template <typename T, bool isReuseSource = false>
__aicore__ inline void PowerCommonImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0,
    const LocalTensor<T>& srcTensor1, const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor0, "srcTensor0", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor1, "srcTensor1", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");

    CheckCalCount(calCount, "calCount", srcTensor0, "srcTensor0", "Power");
    CheckCalCount(calCount, "calCount", srcTensor1, "srcTensor1", "Power");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Power");

    ASCENDC_ASSERT((std::is_same<T, float>::value || std::is_same<T, half>::value || std::is_same<T, int32_t>::value), {
        KERNEL_LOG( KERNEL_ERROR,
            "Failed to check the data types, current api support data types are half/float/int32_t."); });

    SetMaskCount();
    PowerImpl(dstTensor, srcTensor0, srcTensor1, sharedTmpBuffer, calCount);
    SetMaskNorm();
    ResetMask();
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void PowerCommonImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor,
    const LocalTensor<T>& src1Tensor, uint32_t calCount)
{
    LocalTensor<uint8_t> stackTensor;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(stackTensor);
    ASCENDC_ASSERT((ans),
                   { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });

    PowerCommonImpl<T, isReuseSource>(dstTensor, src0Tensor, src1Tensor, stackTensor, calCount);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void PowerCommonImpl(const LocalTensor<T>& dstTensor, const T& src0Scalar,
    const LocalTensor<T>& src1Tensor, uint32_t calCount)
{
    LocalTensor<uint8_t> stackTensor;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(stackTensor);
    ASCENDC_ASSERT((ans),
                   { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });

    PowerCommonImpl<T, isReuseSource>(dstTensor, src0Scalar, src1Tensor, stackTensor, calCount);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void PowerCommonImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor,
    const T& src1Scalar, uint32_t calCount)
{
    LocalTensor<uint8_t> stackTensor;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(stackTensor);
    ASCENDC_ASSERT((ans),
                   { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });

    PowerCommonImpl<T, isReuseSource>(dstTensor, src0Tensor, src1Scalar, stackTensor, calCount);
}
}  // namespace AscendC
#endif // IMPL_MATH_POWER_POWER_COMMON_IMPL_H