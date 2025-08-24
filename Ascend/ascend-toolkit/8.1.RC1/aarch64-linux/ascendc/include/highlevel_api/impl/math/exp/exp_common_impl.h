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
 * \file exp_common_impl.h
 * \brief
 */
#ifndef IMPL_MATH_EXP_EXP_COMMON_IMPL_H
#define IMPL_MATH_EXP_EXP_COMMON_IMPL_H
#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "../../common/check.h"

namespace AscendC {
namespace ExpAPI {
constexpr uint8_t HALF_REPEAT_STRIDE = 4;       // DEFAULT_REPEAT_STRIDE / sizeof(half)
constexpr uint32_t EXP_TWO = 2;                 // when FP32 + reusesrc, only need 2 tmpBuffer
constexpr uint32_t EXP_THREE = 3;               // when FP32 + not reusesrc, need 3 tmpBuffer
constexpr uint32_t EXP_FOUR = 4;                // when FP16, need 4 tmpBuffer

// Assign tmpBuffer for exp and calculate relevant parameter
template <typename T, bool isReuseSource = false, uint8_t expandLevel = 10>
__aicore__ inline void UpdataExpParams(const LocalTensor<T>& src, const uint32_t calCount,
    const LocalTensor<float>& stackBuffer, ExpParams<float>& params)
{
    uint32_t alignNum = ONE_BLK_SIZE / sizeof(T);
    // FP16:                   FloorX + FloorXPow + Res + IntPart
    // FP32 + not reusesrc:    FloorX + FloorXPow +     + IntPart
    // FP32 + reusesrc:               + FloorXPow +     + IntPart
    bool isFloat = IsSameType<T, float>::value;
    uint32_t numberOfTmpBuf = EXP_FOUR;
    if (isFloat) {
        numberOfTmpBuf = isReuseSource ? EXP_TWO : EXP_THREE;
    }

    uint32_t inputSize = calCount;                           // calCount decides how many number do Exp calculation
    uint32_t stackBufferSize = stackBuffer.GetSize();        // how many FP32 can store
    uint32_t oneTmpSize = stackBufferSize / numberOfTmpBuf;  // FP32 num for each tmpTensor after alignment
    oneTmpSize = oneTmpSize / alignNum * alignNum;
    uint32_t secondOffset = (isFloat && isReuseSource)? 0 : oneTmpSize;   // FP32 + reusesrc do not need FloorX
    uint32_t fourthOffset = isFloat ? 0 : oneTmpSize;                     // FP32 do not need tmpTensorRes

    CheckTmpBufferSize(oneTmpSize, 0, stackBufferSize);

    params.inputSize = inputSize;
    params.oneTmpSize = oneTmpSize;
    params.firstTmpStartPos = 0;                                         // tempTensorFloorX
    params.secondTmpStartPos = secondOffset;                             // tempTensorFloorXPow
    params.thirdTmpStartPos = params.secondTmpStartPos + oneTmpSize;     // tempTensorRes    mandatory for FP16
    params.fourthTmpStartPos = params.thirdTmpStartPos + fourthOffset;   // tmpTensorIntPart
    params.loopNum = inputSize / oneTmpSize;                             // how many times needed for main block
    params.tailSize = inputSize % oneTmpSize;                            // tail block size
    params.tailPos = inputSize - params.tailSize;                        // tail block start position
    params.curDataLength = oneTmpSize;                                   // current data num for calculation
    params.expandLevel = expandLevel;                                    // taylor param expand level
}

template <bool isReuseSource = false, uint8_t expandLevel = 10>
__aicore__ inline void GetExpTensorInfo(const LocalTensor<half>& src, const LocalTensor<half>& dst,
    const uint32_t calCount, const LocalTensor<float>& stackBuffer, ExpParams<float>& params)
{
    UpdataExpParams<half, isReuseSource, expandLevel>(src, calCount, stackBuffer, params);
    params.tempTensorFloorX = stackBuffer[params.firstTmpStartPos];
    params.tempTensorFloorXPow = stackBuffer[params.secondTmpStartPos];
    params.tempTensorRes = stackBuffer[params.thirdTmpStartPos];
    params.tempTensorIntPart = stackBuffer[params.fourthTmpStartPos];
}

template <bool isReuseSource = false, uint8_t expandLevel = 10>
__aicore__ inline void GetExpTensorInfo(const LocalTensor<float>& src, const LocalTensor<float>& dst,
    const uint32_t calCount, const LocalTensor<float>& stackBuffer, ExpParams<float>& params)
{
    UpdataExpParams<float, isReuseSource, expandLevel>(src, calCount, stackBuffer, params);
    if constexpr(isReuseSource) {
        params.tempTensorFloorX = src;
    } else {
        params.tempTensorFloorX = stackBuffer[params.firstTmpStartPos];
    }
    params.tempTensorFloorXPow = stackBuffer[params.secondTmpStartPos];
    params.tempTensorRes = dst;
    params.tempTensorIntPart = stackBuffer[params.fourthTmpStartPos];
}

// Put input into params.tempTensorFloorX    Fp16: cast to FP32     Fp32: direct copy
template <typename T>
__aicore__ inline void GetExpInputInTmp(const LocalTensor<T>& src, const ExpParams<float>& params, uint32_t maskLength)
{
    UnaryRepeatParams unaryParams;
    // FP16 -> cast -> FP32
    SetVectorMask<float, MaskMode::COUNTER>(0, maskLength);
    if constexpr (IsSameType<T, half>::value) {
        unaryParams.srcRepStride = HALF_REPEAT_STRIDE;
        Cast<float, half, false>(params.tempTensorFloorX, src, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams);
    } else {  // FP32: copy
        Adds<float, false>(params.tempTensorFloorX, src, 0.0, MASK_PLACEHOLDER, 1, unaryParams);
    }
    PipeBarrier<PIPE_V>();
}

// IntX: floor(x) to extract int part of x                           store in tempTensorIntPart
// DecimalX: x - floor(x) to extract decimal part of x               store in tempTensorFloorX
// ExpIntX:  exp(IntX), because res = exp(IntX) * exp(DecimalX)      store in tempTensorIntPart
__aicore__ inline void GetExpFloorInput(const ExpParams<float>& params, uint32_t maskLength)
{
    UnaryRepeatParams unaryParams;
    BinaryRepeatParams binaryParams;

    SetVectorMask<float, MaskMode::COUNTER>(0, maskLength);

    // tmpTensorIntPart:   floor(x)       ->   IntX
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
    Cast<float, float, false>(params.tempTensorIntPart, params.tempTensorFloorX, RoundMode::CAST_FLOOR,
        MASK_PLACEHOLDER, 1, unaryParams);
#elif defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
    Cast<int32_t, float, false>(params.tempTensorIntPart.ReinterpretCast<int32_t>(), params.tempTensorFloorX,
        RoundMode::CAST_FLOOR, MASK_PLACEHOLDER, 1, { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
    Cast<float, int32_t, false>(params.tempTensorIntPart, params.tempTensorIntPart.ReinterpretCast<int32_t>(),
        RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
#endif
    PipeBarrier<PIPE_V>();

    // tmpTensorFloorX:   x - floor(x)    ->   DecimalX
    Sub<float, false>(params.tempTensorFloorX, params.tempTensorFloorX, params.tempTensorIntPart, MASK_PLACEHOLDER, 1,
        binaryParams);
    PipeBarrier<PIPE_V>();

    // tmpTensorIntPart:  exp(floor(x))   ->   ExpIntX
    Exp<float, false>(params.tempTensorIntPart, params.tempTensorIntPart, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
}

// use Taylor's Formula to calculate exp DecimalX with higher precision    maskLength decides how many to calculate
// exp(x) = exp(IntX) * exp(DecimalX)     exp(DecimalX) = sum ((DecimalX ^ n) / n!)  n-> expandLevel
// tmpTensorRes: FP16 always use tmpBuffer, thus offset 0 is ok.  FP32 use dst tensor, thus need offset to shift
__aicore__ inline void ExpHighPrecisionExec(const ExpParams<float>& params, uint32_t maskLength, uint32_t offset)
{
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;
    SetVectorMask<float, MaskMode::COUNTER>(0, maskLength);

    // copy FP32 FloorX to tempTensorFloorXPow:        FloorXPow = DecimalX
    Adds<float, false>(params.tempTensorFloorXPow, params.tempTensorFloorX, 0.0, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // + x      Res = DecimalX
    Adds<float, false>(params.tempTensorRes[offset], params.tempTensorFloorX, 0.0, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // + 1      Res = 1 + DecimalX
    Adds<float, false>(params.tempTensorRes[offset], params.tempTensorRes[offset], 1, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // (x^2 / 2!) + (x^3 / 3!) + .... + (x^expandLevel / expandLevel!)
    for (int32_t i = 2; i < params.expandLevel + 1; i++) {
        // FloorXPow: (x ^ n) / n! * x  ->   (x ^ (n+1)) / n!
        Mul<float, false>(params.tempTensorFloorXPow, params.tempTensorFloorX, params.tempTensorFloorXPow,
            MASK_PLACEHOLDER, 1, binaryParams);
        PipeBarrier<PIPE_V>();

        // FloorXPow: (x ^ (n+1)) / n! * [1 / (n+1)]  -> (x ^ (n+1)) / (n+1)!
        Muls<float, false>(params.tempTensorFloorXPow, params.tempTensorFloorXPow, float(1.0) / float(i),
            MASK_PLACEHOLDER, 1, unaryParams);
        PipeBarrier<PIPE_V>();

        // Res = Res + FloorXPow
        Add<float, false>(params.tempTensorRes[offset], params.tempTensorRes[offset], params.tempTensorFloorXPow,
            MASK_PLACEHOLDER, 1, binaryParams);
        PipeBarrier<PIPE_V>();
    }

    // Res = exp(Int(x)) * exp(Decimal(x))
    Mul<float, false>(params.tempTensorRes[offset], params.tempTensorRes[offset], params.tempTensorIntPart,
        MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
}

// cast FP32 tempTensorRes to FP16
__aicore__ inline void GetExpCastedResult(const LocalTensor<half>& dst, const ExpParams<float>& params,
    uint32_t maskLength)
{
    UnaryRepeatParams unaryParams;
    unaryParams.dstRepStride = HALF_REPEAT_STRIDE;
    SetVectorMask<float, MaskMode::COUNTER>(0, maskLength);
    Cast<half, float, false>(dst, params.tempTensorRes, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void ExpHighPrecisionND(const LocalTensor<T>& src, const LocalTensor<T>& dst,
    const ExpParams<float>& params, uint32_t offset, uint32_t maskLength)
{
    GetExpInputInTmp(src[offset], params, maskLength);
    GetExpFloorInput(params, maskLength);

    // FP16 needs to cast FP32 output.
    if constexpr(IsSameType<T, half>::value) {
        ExpHighPrecisionExec(params, maskLength, 0);
        GetExpCastedResult(dst[offset], params, maskLength);
    } else {
        ExpHighPrecisionExec(params, maskLength, offset);
    }
}

// high precision: calculate by FP32 with Taylor's formula
template <typename T>
__aicore__ inline void ExpND(const LocalTensor<T>& src, const LocalTensor<T>& dst, const ExpParams<float>& params)
{
    SetMaskCount();

    uint32_t offset = 0;
    for (uint32_t index = 0; index < params.loopNum; index++) {
        ExpHighPrecisionND(src, dst, params, offset, params.curDataLength);
        offset += params.oneTmpSize;
    }

    if (params.tailSize > 0) {
        ExpHighPrecisionND(src, dst, params, offset, params.tailSize);
    }
}

template <typename T, uint8_t taylorExpandLevel, bool isReuseSource>
__aicore__ inline void ExpImpl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    if (taylorExpandLevel == 0) {
        Exp<T>(dstLocal, srcLocal, calCount);
        return;
    }

    CheckTensorPosition(dstLocal, "dstLocal", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcLocal, "srcLocal", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");

    CheckCalCount(calCount, "calCount", srcLocal, "srcLocal", "Exp");
    CheckCalCount(calCount, "calCount", dstLocal, "dstLocal", "Exp");

    uint32_t bufferSize = sharedTmpBuffer.GetSize();
    CheckTmpBufferSize(bufferSize, 0, bufferSize);

    LocalTensor<float> stackBuffer = sharedTmpBuffer.ReinterpretCast<float>();
    ExpParams<float> expParams;

    ExpAPI::GetExpTensorInfo<isReuseSource, taylorExpandLevel>(srcLocal, dstLocal, calCount, stackBuffer, expParams);
    ExpAPI::ExpND<T>(srcLocal, dstLocal, expParams);

    SetMaskNorm();
    ResetMask();
}

template <typename T, uint8_t taylorExpandLevel, bool isReuseSource>
__aicore__ inline void ExpImpl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const uint32_t calCount)
{
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    ExpImpl<T, taylorExpandLevel, isReuseSource>(dstLocal, srcLocal, sharedTmpBuffer, calCount);
}

} // namespace ExpAPI
}  // namespace AscendC
#endif  // IMPL_MATH_EXP_EXP_COMMON_IMPL_H
