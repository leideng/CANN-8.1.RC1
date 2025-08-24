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
 * \file power_v200_impl.h
 * \brief
 */
#ifndef IMPL_MATH_POWER_POWER_V200_IMPL_H
#define IMPL_MATH_POWER_POWER_V200_IMPL_H
#include "power_common_utils.h"

namespace AscendC {
// Power api tmpTensor calc Factor For V200
constexpr uint32_t TENSOR_TENSOR_FLOAT = 5;
constexpr uint32_t TENSOR_TENSOR_INT = 7;
constexpr uint32_t TENSOR_TENSOR_HALF = 8;
constexpr uint32_t TENSOR_SCALAR_FLOAT = 6;
constexpr uint32_t TENSOR_SCALAR_INT = 8;
constexpr uint32_t TENSOR_SCALAR_HALF = 8;

// allocate tmpBuffer for PowerI on V200
__aicore__ inline void PowerIParamsCalc(const LocalTensor<uint8_t>& tmpTensor, AscPowerIParams& param,
    uint32_t splitSize)
{
    param.expUBIterate = tmpTensor.ReinterpretCast<int32_t>();
    param.oriAbsExp = param.expUBIterate[splitSize];
    param.recordExpNode = param.oriAbsExp[splitSize];
    param.tmpTensor1 = param.recordExpNode[splitSize];
    param.tmpTensor2 = param.tmpTensor1[splitSize];
    param.tmpTensor3 = param.tmpTensor2[splitSize];
    param.negMask = param.tmpTensor3[splitSize].ReinterpretCast<uint8_t>();
    param.mask = param.negMask[splitSize];
    param.tmpScalar = param.mask[splitSize];
    param.expUBIterate.SetSize(splitSize);
    param.oriAbsExp.SetSize(splitSize);
    param.recordExpNode.SetSize(splitSize);
    param.tmpTensor1.SetSize(splitSize);
    param.tmpTensor2.SetSize(splitSize);
    param.tmpTensor3.SetSize(splitSize);
    param.negMask.SetSize(splitSize);
    param.mask.SetSize(splitSize);
    param.tmpScalar.SetSize(ONE_BLK_SIZE);
}

// allocate tmpBuffer for PowerF on V200
__aicore__ inline void PowerFParamsCalc(const LocalTensor<float>& tmpTensor,
    AscPowerFParams& param, uint32_t splitSize)
{
    param.tmpTensor1 = tmpTensor;
    param.tmpTensor2 = tmpTensor[splitSize];
    param.tmpTensor3 = param.tmpTensor2[splitSize];
    param.tmpTensor4 = param.tmpTensor3[splitSize];
    param.tmpMask1 = param.tmpTensor4[splitSize].ReinterpretCast<uint8_t>();
    param.tmpMask2 = param.tmpMask1[splitSize];
    param.tmpMask3 = param.tmpMask2[splitSize];
    param.finiteIntegerYMask = param.tmpMask3[splitSize];
    param.tmpTensor1.SetSize(splitSize);
    param.tmpTensor2.SetSize(splitSize);
    param.tmpTensor3.SetSize(splitSize);
    param.tmpTensor4.SetSize(splitSize);
    param.tmpMask1.SetSize(splitSize);
    param.tmpMask2.SetSize(splitSize);
    param.tmpMask3.SetSize(splitSize);
    param.finiteIntegerYMask.SetSize(splitSize);
}

// Compare input with a int(0), write result to mask in bit
__aicore__ inline void CompareIntZero(const LocalTensor<uint8_t>& mask, const LocalTensor<int32_t>& intInput,
    const LocalTensor<int32_t>& tmpTensor, const UnaryRepeatParams& unaryParam, const uint8_t repeat)
{
    constexpr int32_t floatOne = 0x3f800000;
    Adds<int32_t, false>(tmpTensor, intInput, floatOne, MASK_PLACEHOLDER, 1, unaryParam);
    PipeBarrier<PIPE_V>();
    CompareScalar<float, uint8_t, false>(mask, tmpTensor.ReinterpretCast<float>(), 1.0f, CMPMODE::EQ, MASK_PLACEHOLDER,
        repeat, unaryParam);
}

// Cast float->float with different RoundMode
__aicore__ inline void CastFloat2Float(const LocalTensor<float>& dst, const LocalTensor<float>& src, RoundMode mode,
    const UnaryRepeatParams& unaryParam)
{
    Cast<int32_t, float, false>(
        dst.ReinterpretCast<int32_t>(), src, mode, MASK_PLACEHOLDER, 1, unaryParam);
    PipeBarrier<PIPE_V>();
    Cast<float, int32_t, false>(
        dst, dst.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParam);
}

// copy sign bit to dst
__aicore__ inline void GrepSignBit(const LocalTensor<uint8_t>& dst, const LocalTensor<float>& src,
    const LocalTensor<float>& tmpTensor1, const LocalTensor<float>& tmpTensor2,
    const UnaryRepeatParams& unaryParam, const BinaryRepeatParams& binaryParam, const uint32_t calCount)
{
    constexpr uint32_t highestBitOne = 0x80000000u;
    constexpr uint32_t floatOne = 0x3f800000u;
    constexpr uint32_t doubleFactor = 2;
    const uint8_t repeat = DivCeil(calCount * sizeof(float), ONE_REPEAT_BYTE_SIZE);

    Duplicate<uint32_t, false>(tmpTensor1.ReinterpretCast<uint32_t>(), highestBitOne, MASK_PLACEHOLDER, 1,
        DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    Duplicate<uint32_t, false>(tmpTensor2.ReinterpretCast<uint32_t>(), floatOne, MASK_PLACEHOLDER, 1,
        DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
    SetVectorMask<uint16_t>(0, calCount * doubleFactor);
    And<uint16_t, false>(tmpTensor1.ReinterpretCast<uint16_t>(), src.ReinterpretCast<uint16_t>(),
        tmpTensor1.ReinterpretCast<uint16_t>(), MASK_PLACEHOLDER, 1, binaryParam);
    PipeBarrier<PIPE_V>();
    Or<uint16_t, false>(tmpTensor1.ReinterpretCast<uint16_t>(), tmpTensor2.ReinterpretCast<uint16_t>(),
        tmpTensor1.ReinterpretCast<uint16_t>(), MASK_PLACEHOLDER, 1, binaryParam);
    PipeBarrier<PIPE_V>();
    SetVectorMask<uint16_t>(0, calCount);
    CompareScalar<float, uint8_t, false>(dst, tmpTensor1, 0.0f, CMPMODE::LT, MASK_PLACEHOLDER, repeat, unaryParam);
}

// s32 tensor with positive elements shift right 1 bit
__aicore__ inline void ShiftRightOneBit(const LocalTensor<int32_t>& srcDst, const LocalTensor<int32_t>& tmpTensor,
    const UnaryRepeatParams& unaryParam, const BinaryRepeatParams& binaryParam, const uint32_t calCount)
{
    constexpr uint32_t lastZero = 0xfffffffeu;
    constexpr uint32_t doubleFactor = 2;

    Duplicate<uint32_t, false>(tmpTensor.ReinterpretCast<uint32_t>(),
        lastZero, MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
    SetVectorMask<uint16_t>(0, calCount * doubleFactor);
    And<uint16_t, false>(tmpTensor.ReinterpretCast<uint16_t>(), srcDst.ReinterpretCast<uint16_t>(),
        tmpTensor.ReinterpretCast<uint16_t>(), MASK_PLACEHOLDER, 1, binaryParam);
    PipeBarrier<PIPE_V>();
    SetVectorMask<uint32_t>(0, calCount);
    Cast<float, int32_t, false>(tmpTensor.ReinterpretCast<float>(),
        tmpTensor.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParam);
    PipeBarrier<PIPE_V>();
    Muls<float, false>(tmpTensor.ReinterpretCast<float>(),
        tmpTensor.ReinterpretCast<float>(), 0.5f, MASK_PLACEHOLDER, 1, unaryParam);
    PipeBarrier<PIPE_V>();
    Cast<int32_t, float, false>(srcDst, tmpTensor.ReinterpretCast<float>(),
        RoundMode::CAST_RINT, MASK_PLACEHOLDER, 1, unaryParam);
}

// positive element compare with 0
__aicore__ inline void CompareZeroPositive(const LocalTensor<uint8_t>& dst, const LocalTensor<int32_t>& src,
    const UnaryRepeatParams& unaryParam, const uint8_t repeat)
{
    CompareScalar<float, uint8_t, false>(dst, src.ReinterpretCast<float>(), static_cast<float>(0), CMPMODE::EQ,
        MASK_PLACEHOLDER, repeat, unaryParam);
}

__aicore__ inline void ReduceSumCount(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const LocalTensor<float>& tmpTensor, const uint32_t calCount)
{
    SetMaskNorm();
    ReduceSum<float, false>(dst, src, tmpTensor, calCount);
    SetMaskCount();
    SetVectorMask<int32_t>(0, calCount);
}
} // namespace AscendC
#endif // IMPL_MATH_POWER_POWER_V200_IMPL_H
