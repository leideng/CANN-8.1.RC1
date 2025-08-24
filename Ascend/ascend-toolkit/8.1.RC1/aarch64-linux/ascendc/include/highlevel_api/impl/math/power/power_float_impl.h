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
 * \file power_float_impl.h
 * \brief
 */
#ifndef IMPL_MATH_POWER_POWER_FLOAT_IMPL_H
#define IMPL_MATH_POWER_POWER_FLOAT_IMPL_H

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
#include "power_v220_impl.h"
#elif defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
#include "power_v200_impl.h"
#endif

namespace AscendC {
/*!
 * \ingroup PowerF
 * \brief Intrinsics of Power(Float Input).
 */
// init float input tmpScalar
__aicore__ inline void InitTmpScalar(const LocalTensor<float>& tmpScalar)
{
    NotNumUnion notNum;
    notNum.i = F32_NAN;
    SetVectorMask<float>(0, ONE_BLK_SIZE / sizeof(float));
    Duplicate<float, false>(tmpScalar, 1.0f, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
    Duplicate<float, false>(
        tmpScalar[ONE_BLK_SIZE / sizeof(float)], notNum.f, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
}

// init Power result(dstTensor) = exp(src1 * ln(|src0|))
__aicore__ inline void InitDst(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
    const LocalTensor<float>& src1,
    const UnaryRepeatParams& unaryParam, const BinaryRepeatParams& binaryParam)
{
    Abs<float, false>(dst, src0, MASK_PLACEHOLDER, 1, unaryParam);
    PipeBarrier<PIPE_V>();
    Ln<float, false>(dst, dst, MASK_PLACEHOLDER, 1, unaryParam);
    PipeBarrier<PIPE_V>();
    Mul<float, false>(dst, src1, dst, MASK_PLACEHOLDER, 1, binaryParam);
    PipeBarrier<PIPE_V>();
    Exp<float, false>(dst, dst, MASK_PLACEHOLDER, 1, unaryParam);
}

// determine negative sign in tmpTensor1
__aicore__ inline void DetermineSign(const LocalTensor<float>& src1, const AscPowerFParams& param,
    const UnaryRepeatParams& unaryParam, const BinaryRepeatParams& binaryParam)
{
    Abs<float, false>(param.tmpTensor1, src1, MASK_PLACEHOLDER, 1, unaryParam);
    PipeBarrier<PIPE_V>();
    CastFloat2Float(param.tmpTensor1, param.tmpTensor1, RoundMode::CAST_RINT, unaryParam);
    PipeBarrier<PIPE_V>();
    Muls<float, false>(param.tmpTensor2, param.tmpTensor1, 0.5f, MASK_PLACEHOLDER, 1, unaryParam);
    PipeBarrier<PIPE_V>();
    CastFloat2Float(param.tmpTensor2, param.tmpTensor2, RoundMode::CAST_FLOOR, unaryParam);
    PipeBarrier<PIPE_V>();
    Muls<float, false>(param.tmpTensor2, param.tmpTensor2, 2.0f, MASK_PLACEHOLDER, 1, unaryParam);
    PipeBarrier<PIPE_V>();
    Sub<float, false>(param.tmpTensor1, param.tmpTensor1, param.tmpTensor2, MASK_PLACEHOLDER, 1, binaryParam);
    PipeBarrier<PIPE_V>();
    Muls<float, false>(param.tmpTensor1, param.tmpTensor1, -2.0f, MASK_PLACEHOLDER, 1, unaryParam);
    PipeBarrier<PIPE_V>();
    Adds<float, false>(param.tmpTensor1, param.tmpTensor1, 1.0f, MASK_PLACEHOLDER, 1, unaryParam);
    PipeBarrier<PIPE_V>();
    CastFloat2Float(param.tmpTensor1, param.tmpTensor1, RoundMode::CAST_RINT, unaryParam);
}

// generate mask(tmpMask1) for some ones results
__aicore__ inline void GenMaskForOne(const LocalTensor<float>& src0, const LocalTensor<float>& src1,
    const AscPowerFParams& param, const UnaryRepeatParams& unaryParam,
    const BinaryRepeatParams& binaryParam, const uint32_t calCount)
{
    NotNumUnion notNum;
    uint8_t repeat = DivCeil(calCount * sizeof(float), ONE_REPEAT_BYTE_SIZE);

    Abs<float, false>(param.tmpTensor1, src1, MASK_PLACEHOLDER, 1, unaryParam);
    PipeBarrier<PIPE_V>();
    CompareScalar<float, uint8_t, false>(param.tmpMask1, param.tmpTensor1, static_cast<float>(0), CMPMODE::NE,
        MASK_PLACEHOLDER, repeat, unaryParam);
    PipeBarrier<PIPE_V>();
    CompareScalar<float, uint8_t, false>(param.tmpMask2, src0, static_cast<float>(1), CMPMODE::NE,
        MASK_PLACEHOLDER, repeat, unaryParam);
    PipeBarrier<PIPE_V>();
    SetVectorMask<float>(0, ConstCeil(calCount, sizeof(uint16_t) * ONE_BYTE_BIT_SIZE));
    And<uint16_t, false>(param.tmpMask1.ReinterpretCast<uint16_t>(), param.tmpMask2.ReinterpretCast<uint16_t>(),
        param.tmpMask1.ReinterpretCast<uint16_t>(), MASK_PLACEHOLDER, 1, binaryParam);
    SetVectorMask<float>(0, calCount);
    PipeBarrier<PIPE_V>();
    CompareScalar<float, uint8_t, false>(param.tmpMask2, src0, static_cast<float>(-1), CMPMODE::NE,
        MASK_PLACEHOLDER, repeat, unaryParam);
    PipeBarrier<PIPE_V>();
    notNum.i = F32_INF;
    CompareScalar<float, uint8_t, false>(param.tmpMask3, param.tmpTensor1, notNum.f, CMPMODE::NE,
        MASK_PLACEHOLDER, repeat, unaryParam);
    PipeBarrier<PIPE_V>();
    SetVectorMask<float>(0, ConstCeil(calCount, sizeof(uint16_t) * ONE_BYTE_BIT_SIZE));
    Or<uint16_t, false>(param.tmpMask2.ReinterpretCast<uint16_t>(), param.tmpMask3.ReinterpretCast<uint16_t>(),
        param.tmpMask2.ReinterpretCast<uint16_t>(), MASK_PLACEHOLDER, 1, binaryParam);
    PipeBarrier<PIPE_V>();
    And<uint16_t, false>(param.tmpMask1.ReinterpretCast<uint16_t>(), param.tmpMask2.ReinterpretCast<uint16_t>(),
        param.tmpMask1.ReinterpretCast<uint16_t>(), MASK_PLACEHOLDER, 1, binaryParam);
    SetVectorMask<float>(0, calCount);
}

__aicore__ inline void GenMaskForNan(const LocalTensor<float>& src0,
    const LocalTensor<float>& src1, const AscPowerFParams& param,
    const UnaryRepeatParams& unaryParam, const BinaryRepeatParams& binaryParam, const uint32_t calCount)
{
    NotNumUnion notNum;
    notNum.i = F32_INF;
    uint8_t repeat = DivCeil(calCount * sizeof(float), ONE_REPEAT_BYTE_SIZE);

    Abs<float, false>(param.tmpTensor1, src1, MASK_PLACEHOLDER, 1, unaryParam);
    PipeBarrier<PIPE_V>();
    CompareScalar<float, uint8_t, false>(param.tmpMask1, param.tmpTensor1, notNum.f, CMPMODE::EQ,
        MASK_PLACEHOLDER, repeat, unaryParam);
    PipeBarrier<PIPE_V>();
    SetVectorMask<float>(0, ConstCeil(calCount, sizeof(uint16_t) * ONE_BYTE_BIT_SIZE));
    Or<uint16_t, false>(param.tmpMask1.ReinterpretCast<uint16_t>(),
        param.finiteIntegerYMask.ReinterpretCast<uint16_t>(), param.tmpMask1.ReinterpretCast<uint16_t>(),
        MASK_PLACEHOLDER, 1, binaryParam);
    SetVectorMask<float>(0, calCount);
    CompareScalar<float, uint8_t, false>(param.tmpMask2, src0, static_cast<float>(0), CMPMODE::GE,
        MASK_PLACEHOLDER, repeat, unaryParam);
    notNum.i = F32_NEG_INF;
    CompareScalar<float, uint8_t, false>(param.tmpMask3, src0, notNum.f, CMPMODE::EQ,
        MASK_PLACEHOLDER, repeat, unaryParam);
    PipeBarrier<PIPE_V>();
    SetVectorMask<float>(0, ConstCeil(calCount, sizeof(uint16_t) * ONE_BYTE_BIT_SIZE));
    Or<uint16_t, false>(param.tmpMask2.ReinterpretCast<uint16_t>(), param.tmpMask3.ReinterpretCast<uint16_t>(),
        param.tmpMask2.ReinterpretCast<uint16_t>(), MASK_PLACEHOLDER, 1, binaryParam);
    PipeBarrier<PIPE_V>();
    Or<uint16_t, false>(param.tmpMask1.ReinterpretCast<uint16_t>(), param.tmpMask2.ReinterpretCast<uint16_t>(),
        param.tmpMask1.ReinterpretCast<uint16_t>(), MASK_PLACEHOLDER, 1, binaryParam);
    SetVectorMask<float>(0, calCount);
}

// generate select mask(tmpMask1) for sign
__aicore__ inline void GenMaskForSign(const LocalTensor<float>& src0, const LocalTensor<float>& src1,
    const AscPowerFParams& param, const UnaryRepeatParams& unaryParam,
    const BinaryRepeatParams& binaryParam, const uint32_t calCount)
{
    constexpr float intThreshold = 0.00000001f;
    const uint8_t repeat = DivCeil(calCount * sizeof(float), ONE_REPEAT_BYTE_SIZE);

    GrepSignBit(param.tmpMask1, src0, param.tmpTensor2, param.tmpTensor4, unaryParam, binaryParam, calCount);
    PipeBarrier<PIPE_V>();
    Abs<float, false>(param.tmpTensor2, src1, MASK_PLACEHOLDER, 1, unaryParam);
    PipeBarrier<PIPE_V>();
    CastFloat2Float(param.tmpTensor3, param.tmpTensor2, RoundMode::CAST_RINT, unaryParam);
    PipeBarrier<PIPE_V>();
    Sub<float, false>(param.tmpTensor3, param.tmpTensor2, param.tmpTensor3, MASK_PLACEHOLDER, 1, binaryParam);
    PipeBarrier<PIPE_V>();
    Abs<float, false>(param.tmpTensor3, param.tmpTensor3, MASK_PLACEHOLDER, 1, unaryParam);
    PipeBarrier<PIPE_V>();
    CompareScalar<float, uint8_t, false>(param.finiteIntegerYMask, param.tmpTensor3, static_cast<float>(intThreshold),
        CMPMODE::LT, MASK_PLACEHOLDER, repeat, unaryParam);
    PipeBarrier<PIPE_V>();
    SetVectorMask<float>(0, ConstCeil(calCount, sizeof(uint16_t) * ONE_BYTE_BIT_SIZE));
    And<uint16_t, false>(param.tmpMask1.ReinterpretCast<uint16_t>(),
        param.finiteIntegerYMask.ReinterpretCast<uint16_t>(), param.tmpMask1.ReinterpretCast<uint16_t>(),
        MASK_PLACEHOLDER, 1, binaryParam);
    SetVectorMask<float>(0, calCount);
}

__aicore__ inline void CommonPowerF(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor0,
    const LocalTensor<float>& srcTensor1, const LocalTensor<float>& tmpScalar,
    const AscPowerFParams& powerParam, const uint32_t calCount)
{
    const UnaryRepeatParams unaryParam;
    const BinaryRepeatParams binaryParam;

    PipeBarrier<PIPE_V>();
    InitDst(dstTensor, srcTensor0, srcTensor1, unaryParam, binaryParam);
    PipeBarrier<PIPE_V>();
    DetermineSign(srcTensor1, powerParam, unaryParam, binaryParam);
    PipeBarrier<PIPE_V>();
    GenMaskForSign(srcTensor0, srcTensor1, powerParam, unaryParam, binaryParam, calCount);
    PipeBarrier<PIPE_V>();
    VselPowerTensorScalar(powerParam.tmpTensor2, powerParam.tmpMask1, powerParam.tmpTensor1, tmpScalar,
        SELMODE::VSEL_TENSOR_SCALAR_MODE, 1, binaryParam, calCount);
    PipeBarrier<PIPE_V>();
    Mul<float, false>(dstTensor, powerParam.tmpTensor2, dstTensor, MASK_PLACEHOLDER, 1, binaryParam);
    PipeBarrier<PIPE_V>();
    GenMaskForOne(srcTensor0, srcTensor1, powerParam, unaryParam, binaryParam, calCount);
    PipeBarrier<PIPE_V>();
    VselPowerTensorScalar(dstTensor, powerParam.tmpMask1, dstTensor, tmpScalar,
        SELMODE::VSEL_TENSOR_SCALAR_MODE, 1, binaryParam, calCount);
    PipeBarrier<PIPE_V>();
    GenMaskForNan(srcTensor0, srcTensor1, powerParam, unaryParam, binaryParam, calCount);
    PipeBarrier<PIPE_V>();
    VselPowerTensorScalar(dstTensor, powerParam.tmpMask1, dstTensor, tmpScalar[ONE_BLK_SIZE / sizeof(float)],
        SELMODE::VSEL_TENSOR_SCALAR_MODE, 1, binaryParam, calCount);
    PipeBarrier<PIPE_V>();
}

} // namespace AscendC
#endif // IMPL_MATH_POWER_POWER_FLOAT_IMPL_H