/* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lgamma_common_basic_impl.h
 * \brief
 */
#ifndef IMPL_MATH_LGAMMA_LGAMMA_COMMOM_BASIC_IMPL_H
#define IMPL_MATH_LGAMMA_LGAMMA_COMMOM_BASIC_IMPL_H

#if __CCE_AICORE__ == 220
#include "lgamma_v220_impl.h"
#elif __CCE_AICORE__ == 200
#include "lgamma_v200_impl.h"
#endif
#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 220
namespace AscendC {
__aicore__ inline void LGammaCalcMulAdd(const LocalTensor<float> &tmp, const LocalTensor<float> &src,
    const UnaryRepeatParams &unaryParams, const BinaryRepeatParams binaryParams, const float params[],
    const size_t paramLen)
{
    // tmp = x * params[0]  + params[1]
    Muls<float, false>(tmp, src, params[0], MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Adds<float, false>(tmp, tmp, params[1], MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // A strange phenomenon, when the length is params0715Len, 0.98289764
    // the result of inf will occasionally appear in the for loop, and write it separately as OK
    for (size_t i = 2U; i < paramLen && i < params0715Len - 1U; ++i) {
        // tmp = tmp * x + param[i]
        Mul<float, false>(tmp, tmp, src, MASK_PLACEHOLDER, 1, binaryParams);
        PipeBarrier<PIPE_V>();
        Adds<float, false>(tmp, tmp, params[i], MASK_PLACEHOLDER, 1, unaryParams);
        PipeBarrier<PIPE_V>();
    }

    if (paramLen == params0715Len) {
        Mul<float, false>(tmp, tmp, src, MASK_PLACEHOLDER, 1, binaryParams);
        PipeBarrier<PIPE_V>();
        Adds<float, false>(tmp, tmp, params[params0715Len - 1U], MASK_PLACEHOLDER, 1, unaryParams);
        PipeBarrier<PIPE_V>();
    }

    // tmp = tmp * x
    Mul<float, false>(tmp, tmp, src, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
}

// cal result of 0.7 <= x < 1.5 on tmp1, Ln return inf when x is 0
__aicore__ inline void LGamma007(const LocalTensor<float> &src, const LGammaFParams &params)
{
    // tmp1 = MulAdd(x)
    LGammaCalcMulAdd(params.tmp1, src, params.unaryParams, params.binaryParams, params007, params007Len);

    // tmp1 = -ln(tmp1 * x + x)
    Mul<float, false>(params.tmp1, params.tmp1, src, MASK_PLACEHOLDER, 1, params.binaryParams);
    PipeBarrier<PIPE_V>();
    Add<float, false>(params.tmp1, params.tmp1, src, MASK_PLACEHOLDER, 1, params.binaryParams);
    PipeBarrier<PIPE_V>();
    Ln<float, false>(params.tmp1, params.tmp1, MASK_PLACEHOLDER, 1, params.unaryParams);
    PipeBarrier<PIPE_V>();
    Muls<float, false>(params.tmp1, params.tmp1, fn1, MASK_PLACEHOLDER, 1, params.unaryParams);
    PipeBarrier<PIPE_V>();
}

// cal result of 0.7 <= x < 1.5 on tmp2
__aicore__ inline void LGamma0715(const LocalTensor<float> &src, const LGammaFParams &params)
{
    // tmp1 = 1.0 - x
    Muls<float, false>(params.tmp1, src, fn1, MASK_PLACEHOLDER, 1, params.unaryParams);
    PipeBarrier<PIPE_V>();
    Adds<float, false>(params.tmp1, params.tmp1, f1, MASK_PLACEHOLDER, 1, params.unaryParams);
    PipeBarrier<PIPE_V>();

    // tmp2 = MulAdd(tmp1)
    LGammaCalcMulAdd(params.tmp2, params.tmp1, params.unaryParams, params.binaryParams, params0715, params0715Len);
}

// cal result of 1.5 <= x < 3 on tmp2
__aicore__ inline void LGamma153(const LocalTensor<float> &src, const LGammaFParams &params)
{
    // tmp1 = x - 2.0
    Adds<float, false>(params.tmp1, src, fn2, MASK_PLACEHOLDER, 1, params.unaryParams);
    PipeBarrier<PIPE_V>();

    // tmp2 = MulAdd(tmp1)
    LGammaCalcMulAdd(params.tmp2, params.tmp1, params.unaryParams, params.binaryParams, params153, params153Len);
}

// cal result of 3 <= x < 5.8 on tmp3
__aicore__ inline void LGamma358(const LocalTensor<float> &src, const LGammaFParams &params)
{
    // tmp1 = x - 3.0
    Adds<float, false>(params.tmp1, src, fn3, MASK_PLACEHOLDER, 1, params.unaryParams);
    PipeBarrier<PIPE_V>();

    // tmp2 = MulAdd(tmp1)
    LGammaCalcMulAdd(params.tmp2, params.tmp1, params.unaryParams, params.binaryParams, params378X1, params378X1Len);
    // tmp2 = tmp2 - 143033.40625
    constexpr float ftmp2 = -143033.40625;
    Adds<float, false>(params.tmp2, params.tmp2, ftmp2, MASK_PLACEHOLDER, 1, params.unaryParams);
    PipeBarrier<PIPE_V>();

    // tmp3 = MulAdd(tmp1)
    LGammaCalcMulAdd(params.tmp3, params.tmp1, params.unaryParams, params.binaryParams, params378X2, params378X2Len);
    // tmp3 = tmp3 - 206353.578125
    constexpr float ftmp3 = -206353.578125;
    Adds<float, false>(params.tmp3, params.tmp3, ftmp3, MASK_PLACEHOLDER, 1, params.unaryParams);
    PipeBarrier<PIPE_V>();

    // tmp3 = tmp3 / tmp2 + tmp1
    Div<float, false>(params.tmp3, params.tmp2, params.tmp3, MASK_PLACEHOLDER, 1, params.binaryParams);
    PipeBarrier<PIPE_V>();
    Add<float, false>(params.tmp3, params.tmp3, params.tmp1, MASK_PLACEHOLDER, 1, params.binaryParams);
    PipeBarrier<PIPE_V>();
}

// cal result of x >= 5.8 on tmp1
__aicore__ inline void LGamma58(const LocalTensor<float> &src, const LGammaFParams &params)
{
    // tmp1 = ln(x) * 0.5
    Ln<float, false>(params.tmp1, src, MASK_PLACEHOLDER, 1, params.unaryParams);
    PipeBarrier<PIPE_V>();
    Muls<float, false>(params.tmp1, params.tmp1, f05, MASK_PLACEHOLDER, 1, params.unaryParams);
    PipeBarrier<PIPE_V>();

    // tmp2 = tmp1 * (x - 0.5)
    Adds<float, false>(params.tmp2, src, fn05, MASK_PLACEHOLDER, 1, params.unaryParams);
    PipeBarrier<PIPE_V>();
    Mul<float, false>(params.tmp2, params.tmp2, params.tmp1, MASK_PLACEHOLDER, 1, params.binaryParams);
    PipeBarrier<PIPE_V>();

    // tmp3 = 1/x
    Reciprocal<float, false>(params.tmp3, src, MASK_PLACEHOLDER, 1, params.unaryParams);
    PipeBarrier<PIPE_V>();

    // tmp4 = tmp3 * tmp3
    Mul<float, false>(params.tmp4, params.tmp3, params.tmp3, MASK_PLACEHOLDER, 1, params.binaryParams);
    PipeBarrier<PIPE_V>();

    // tmp1 = (MulAdd(tmp4) + 0.0833332762122) * tmp3 + 0.91893851757
    LGammaCalcMulAdd(params.tmp1, params.tmp4, params.unaryParams, params.binaryParams, params58, params58Len);
    constexpr float ftmp1 = 0.0833332762122;
    Adds<float, false>(params.tmp1, params.tmp1, ftmp1, MASK_PLACEHOLDER, 1, params.unaryParams);
    PipeBarrier<PIPE_V>();
    Mul<float, false>(params.tmp1, params.tmp1, params.tmp3, MASK_PLACEHOLDER, 1, params.binaryParams);
    PipeBarrier<PIPE_V>();
    constexpr float ftmp2 = 0.91893851757;
    Adds<float, false>(params.tmp1, params.tmp1, ftmp2, MASK_PLACEHOLDER, 1, params.unaryParams);
    PipeBarrier<PIPE_V>();

    // tmp1 = tmp1 + tmp2 + tmp2 - x
    Add<float, false>(params.tmp1, params.tmp1, params.tmp2, MASK_PLACEHOLDER, 1, params.binaryParams);
    PipeBarrier<PIPE_V>();
    Add<float, false>(params.tmp1, params.tmp1, params.tmp2, MASK_PLACEHOLDER, 1, params.binaryParams);
    PipeBarrier<PIPE_V>();
    Sub<float, false>(params.tmp1, params.tmp1, src, MASK_PLACEHOLDER, 1, params.binaryParams);
    PipeBarrier<PIPE_V>();
}

// gen mask, src < scalar set 1, other set 0, used tmp1
__aicore__ inline void LGammaGenLTMask(
    const LocalTensor<uint8_t> &mask, const LocalTensor<float> &src, const LGammaFParams &params, const float scalar)
{
    Duplicate<float, false>(params.tmp1, scalar, MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();

    uint8_t repeat = DivCeil(params.splitSize * sizeof(float), ONE_REPEAT_BYTE_SIZE);
    Compare<float, uint8_t, false>(mask, src, params.tmp1, CMPMODE::LT, MASK_PLACEHOLDER, repeat, params.binaryParams);
    PipeBarrier<PIPE_V>();
}

// gen mask, src >= scalar set 1, other set 0, used tmp1
__aicore__ inline void LGammaGenGEMask(
    const LocalTensor<uint8_t> &mask, const LocalTensor<float> &src, const LGammaFParams &params, const float scalar)
{
    Duplicate<float, false>(params.tmp1, scalar, MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();

    uint8_t repeat = DivCeil(params.splitSize * sizeof(float), ONE_REPEAT_BYTE_SIZE);
    Compare<float, uint8_t, false>(mask, src, params.tmp1, CMPMODE::GE, MASK_PLACEHOLDER, repeat, params.binaryParams);
    PipeBarrier<PIPE_V>();
}

// gen mask on params.mask, min > src >= max set 1, other set 0, used tmp1
__aicore__ inline void LGammaGenRangeMask(
    const LocalTensor<float> &src, const LGammaFParams &params, const float min, const float max)
{
    LGammaGenLTMask(params.mask, src, params, max);
    LGammaGenGEMask(params.tmpMask1, src, params, min);

    SetVectorMask<float>(0, ConstCeil(params.splitSize, sizeof(uint16_t) * ONE_BYTE_BIT_SIZE));
    And<uint16_t, false>(params.mask.ReinterpretCast<uint16_t>(),
        params.tmpMask1.ReinterpretCast<uint16_t>(),
        params.mask.ReinterpretCast<uint16_t>(),
        MASK_PLACEHOLDER,
        1,
        params.binaryParams);
    SetVectorMask<float>(0, params.splitSize);
    PipeBarrier<PIPE_V>();
}

// Select the value of src at mask 1, and accumulate the result onto dst, used tmp1
__aicore__ inline void LGammaSelect(const LocalTensor<float> &dst, const LocalTensor<float> &src,
    const LocalTensor<uint8_t> &mask, const LGammaFParams &params)
{
    SetCmpMask<float>(params.tmpScalar);
    PipeBarrier<PIPE_V>();
    Select<float, uint8_t>(params.tmp1, mask, src, 1, params.binaryParams);
    PipeBarrier<PIPE_V>();
    Add<float, false>(dst, params.tmp1, dst, MASK_PLACEHOLDER, 1, params.binaryParams);
    PipeBarrier<PIPE_V>();
}

// tmp6 is |x|, res on tmp5
__aicore__ inline void LGammaPositive(const LGammaFParams &params)
{
    Duplicate<float, false>(params.tmp5, 0.0f, MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();

    // cal and select 0 <= x < 0.7 res
    LGammaGenLTMask(params.mask, params.tmp6, params, f07);
    LGamma007(params.tmp6, params);  // res on tmp1
    LGammaSelect(params.tmp5, params.tmp1, params.mask, params);

    // cal and select 0.7 <= x < 1.5 res
    LGammaGenRangeMask(params.tmp6, params, f07, f15);
    LGamma0715(params.tmp6, params);  // res on tmp2
    LGammaSelect(params.tmp5, params.tmp2, params.mask, params);

    // cal and select 1.5 <= x < 3 res
    LGammaGenRangeMask(params.tmp6, params, f15, f3);
    LGamma153(params.tmp6, params);  // res on tmp2
    LGammaSelect(params.tmp5, params.tmp2, params.mask, params);

    // cal and select 3 <= x < 5.8 res
    LGammaGenRangeMask(params.tmp6, params, f3, f58);
    LGamma358(params.tmp6, params);  // res on tmp3
    LGammaSelect(params.tmp5, params.tmp3, params.mask, params);

    // cal and select 5.8 <= x < inf res
    NotNumUnion notNum;
    notNum.i = F32_INF;
    LGammaGenRangeMask(params.tmp6, params, f58, notNum.f);
    LGamma58(params.tmp6, params);  // res on tmp1
    LGammaSelect(params.tmp5, params.tmp1, params.mask, params);

    // cal and select x >= inf res
    LGammaGenGEMask(params.mask, params.tmp6, params, notNum.f);
    LGammaSelect(params.tmp5, params.tmp6, params.mask, params);
}

// cal tmp val on tmp2, mask for odd, tmpMask1 for even, tmp6 is |x|
__aicore__ inline void LGammaCalNegTmp1(const LGammaFParams &params)
{
    // tmp2 = floor(tmp6 + tmp6 + 0.5)
    Add<float, false>(params.tmp2, params.tmp6, params.tmp6, MASK_PLACEHOLDER, 1, params.binaryParams);
    PipeBarrier<PIPE_V>();
    Adds<float, false>(params.tmp2, params.tmp2, f05, MASK_PLACEHOLDER, 1, params.unaryParams);
    PipeBarrier<PIPE_V>();
    LGammaFloor(params.tmp2, params.tmp2);

    // tmp3 = tmp2 - floor(tmp2 / 2) * 2
    Muls<float, false>(params.tmp3, params.tmp2, f05, MASK_PLACEHOLDER, 1, params.unaryParams);
    PipeBarrier<PIPE_V>();
    LGammaFloor(params.tmp3, params.tmp3);
    Muls<float, false>(params.tmp3, params.tmp3, f2, MASK_PLACEHOLDER, 1, params.unaryParams);
    PipeBarrier<PIPE_V>();
    Sub<float, false>(params.tmp3, params.tmp2, params.tmp3, MASK_PLACEHOLDER, 1, params.binaryParams);
    PipeBarrier<PIPE_V>();

    // tmp3 > 0.5 for even, set mask
    LGammaGenGEMask(params.mask, params.tmp3, params, f05);
    SetVectorMask<float>(0, ConstCeil(params.splitSize, sizeof(uint16_t) * ONE_BYTE_BIT_SIZE));
    // tmpMask1 for even
    Not<uint16_t, false>(params.tmpMask1.ReinterpretCast<uint16_t>(),
        params.mask.ReinterpretCast<uint16_t>(),
        MASK_PLACEHOLDER,
        1,
        params.unaryParams);
    SetVectorMask<float>(0, params.splitSize);
    PipeBarrier<PIPE_V>();

    // tmp2 = (tmp2 * -0.5 + tmp6) * pi
    Muls<float, false>(params.tmp2, params.tmp2, fn05, MASK_PLACEHOLDER, 1, params.unaryParams);
    PipeBarrier<PIPE_V>();
    Add<float, false>(params.tmp2, params.tmp2, params.tmp6, MASK_PLACEHOLDER, 1, params.binaryParams);
    PipeBarrier<PIPE_V>();
    Muls<float, false>(params.tmp2, params.tmp2, fPi, MASK_PLACEHOLDER, 1, params.unaryParams);
    PipeBarrier<PIPE_V>();
}

// input is tmp2, mask for odd, tmpMask1 for even, result on tmp2
__aicore__ inline void LGammaCalNegTmp2(const LGammaFParams &params)
{
    // tmp3 = tmp2 * tmp2
    Mul<float, false>(params.tmp3, params.tmp2, params.tmp2, MASK_PLACEHOLDER, 1, params.binaryParams);
    PipeBarrier<PIPE_V>();

    // tmp1 = MulAdd(tmp3) * tmp2 + tmp2, for even
    LGammaCalcMulAdd(
        params.tmp1, params.tmp3, params.unaryParams, params.binaryParams, negParamsEven, negParamsEvenLen);
    Mul<float, false>(params.tmp1, params.tmp1, params.tmp2, MASK_PLACEHOLDER, 1, params.binaryParams);
    PipeBarrier<PIPE_V>();
    Add<float, false>(params.tmp1, params.tmp1, params.tmp2, MASK_PLACEHOLDER, 1, params.binaryParams);
    PipeBarrier<PIPE_V>();

    // init tmp2 to 0
    Duplicate<float, false>(params.tmp2, 0.0f, MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
    LGammaSelect(params.tmp2, params.tmp1, params.tmpMask1, params);

    // tmp1 = MulAdd(tmp3) + 1.0, for odd
    LGammaCalcMulAdd(params.tmp1, params.tmp3, params.unaryParams, params.binaryParams, negParamsOdd, negParamsOddLen);
    Adds<float, false>(params.tmp1, params.tmp1, f1, MASK_PLACEHOLDER, 1, params.unaryParams);
    PipeBarrier<PIPE_V>();
    LGammaSelect(params.tmp2, params.tmp1, params.mask, params);
}

// Get final result, tmp3 save for inf
__aicore__ inline void LGammaCalNegTmp3(const LGammaFParams &params)
{
    // tmp2 = 1.1447298526763916015625 - ln(|tmp2| * tmp6) - tmp5
    Abs<float, false>(params.tmp1, params.tmp2, MASK_PLACEHOLDER, 1, params.unaryParams);
    PipeBarrier<PIPE_V>();
    // Tmp3 is inter var, for too negative and inf
    Mul<float, false>(params.tmp3, params.tmp1, params.tmp6, MASK_PLACEHOLDER, 1, params.binaryParams);
    PipeBarrier<PIPE_V>();
    Ln<float, false>(params.tmp1, params.tmp3, MASK_PLACEHOLDER, 1, params.unaryParams);
    PipeBarrier<PIPE_V>();
    Muls<float, false>(params.tmp1, params.tmp1, fn1, MASK_PLACEHOLDER, 1, params.unaryParams);
    PipeBarrier<PIPE_V>();
    constexpr float ftmp = 1.1447298526763916015625;
    Adds<float, false>(params.tmp1, params.tmp1, ftmp, MASK_PLACEHOLDER, 1, params.unaryParams);
    PipeBarrier<PIPE_V>();
    Muls<float, false>(params.tmp2, params.tmp5, fn1, MASK_PLACEHOLDER, 1, params.unaryParams);
    PipeBarrier<PIPE_V>();
    Add<float, false>(params.tmp2, params.tmp1, params.tmp2, MASK_PLACEHOLDER, 1, params.binaryParams);
    PipeBarrier<PIPE_V>();
}

// Near zero negative, input as |x|, result on tmp1
__aicore__ inline void LGammaCalMinNeg(const LocalTensor<float> &src, const LGammaFParams &params)
{
    // tmp1 = -ln(src)
    Ln<float, false>(params.tmp1, src, MASK_PLACEHOLDER, 1, params.unaryParams);
    PipeBarrier<PIPE_V>();
    Muls<float, false>(params.tmp1, params.tmp1, fn1, MASK_PLACEHOLDER, 1, params.unaryParams);
    PipeBarrier<PIPE_V>();
}

// cal for x < 0, result on tmp4, tmp6 is |x|, tmp5 is pos res
__aicore__ inline void LGammaNegative(const LGammaFParams &params)
{
    Duplicate<float, false>(params.tmp4, 0.0f, MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();

    // cal res for 9.99999968266e-20 < x < 0
    constexpr float minf = 9.99999968266e-20;
    LGammaGenLTMask(params.mask, params.tmp6, params, minf);
    LGammaCalMinNeg(params.tmp6, params);  // res on tmp1
    LGammaSelect(params.tmp4, params.tmp1, params.mask, params);

    // cal temp var, mask for odd, tmpMask1 for even, result on tmp2
    LGammaCalNegTmp1(params);
    // cal odd and even, result on tmp2
    LGammaCalNegTmp2(params);
    // Get final result on tmp2, tmp3 save for inf
    LGammaCalNegTmp3(params);

    // tmp1 = floor(tmp6)
    LGammaFloor(params.tmp1, params.tmp6);

    // Gen mask for negative non-int
    uint8_t repeat = DivCeil(params.splitSize * sizeof(float), ONE_REPEAT_BYTE_SIZE);
    Compare<float, uint8_t, false>(
        params.mask, params.tmp1, params.tmp6, CMPMODE::NE, MASK_PLACEHOLDER, repeat, params.binaryParams);
    PipeBarrier<PIPE_V>();

    // Gen mask for tmp3 is not inf
    NotNumUnion notNum;
    notNum.i = F32_INF;
    Duplicate<float, false>(params.tmp1, notNum.f, MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
    Compare<float, uint8_t, false>(
        params.tmpMask1, params.tmp3, params.tmp1, CMPMODE::LT, MASK_PLACEHOLDER, repeat, params.binaryParams);
    PipeBarrier<PIPE_V>();

    // cal and
    SetVectorMask<float>(0, ConstCeil(params.splitSize, sizeof(uint16_t) * ONE_BYTE_BIT_SIZE));
    And<uint16_t, false>(params.mask.ReinterpretCast<uint16_t>(), params.tmpMask1.ReinterpretCast<uint16_t>(),
        params.mask.ReinterpretCast<uint16_t>(), MASK_PLACEHOLDER, 1, params.binaryParams);
    PipeBarrier<PIPE_V>();
    SetVectorMask<float>(0, params.splitSize);

    // cal x >= 9.99999968266e-20 mask, cal and, select
    LGammaGenGEMask(params.tmpMask1, params.tmp6, params, minf);
    SetVectorMask<float>(0, ConstCeil(params.splitSize, sizeof(uint16_t) * ONE_BYTE_BIT_SIZE));
    And<uint16_t, false>(params.mask.ReinterpretCast<uint16_t>(), params.tmpMask1.ReinterpretCast<uint16_t>(),
        params.mask.ReinterpretCast<uint16_t>(), MASK_PLACEHOLDER, 1, params.binaryParams);
    PipeBarrier<PIPE_V>();
    SetVectorMask<float>(0, params.splitSize);
    LGammaSelect(params.tmp4, params.tmp2, params.mask, params);

    // cal not
    SetVectorMask<float>(0, ConstCeil(params.splitSize, sizeof(uint16_t) * ONE_BYTE_BIT_SIZE));
    Not<uint16_t, false>(params.tmpMask1.ReinterpretCast<uint16_t>(), params.mask.ReinterpretCast<uint16_t>(),
        MASK_PLACEHOLDER, 1, params.unaryParams);
    PipeBarrier<PIPE_V>();
    SetVectorMask<float>(0, params.splitSize);

    // and for tmp6 >= 9.99999968266e-20, res is inf
    LGammaGenGEMask(params.mask, params.tmp6, params, minf);
    SetVectorMask<float>(0, ConstCeil(params.splitSize, sizeof(uint16_t) * ONE_BYTE_BIT_SIZE));
    And<uint16_t, false>(params.tmpMask1.ReinterpretCast<uint16_t>(), params.tmpMask1.ReinterpretCast<uint16_t>(),
        params.mask.ReinterpretCast<uint16_t>(), MASK_PLACEHOLDER, 1, params.binaryParams);
    PipeBarrier<PIPE_V>();
    SetVectorMask<float>(0, params.splitSize);

    Duplicate<float, false>(params.tmp2, notNum.f, MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
    LGammaSelect(params.tmp4, params.tmp2, params.tmpMask1, params);
}

template <bool isReuseSource = false>
__aicore__ inline void LGammaInitFParams(
    const LocalTensor<float> &tmp, const uint32_t splitSize, const LocalTensor<float> &src, LGammaFParams &params)
{
    params.tmp1 = tmp;
    params.tmp2 = tmp[splitSize];
    params.tmp3 = params.tmp2[splitSize];
    params.tmp4 = params.tmp3[splitSize];
    params.tmp5 = params.tmp4[splitSize];
    params.tmp6 = params.tmp5[splitSize];
    if constexpr (isReuseSource) {
        params.mask = params.tmp6[splitSize].ReinterpretCast<uint8_t>();
        params.tmpScalar = src.ReinterpretCast<float>();
    } else {
        params.tmpScalar = params.tmp6[splitSize];
        params.mask = params.tmpScalar[splitSize].ReinterpretCast<uint8_t>();
    }
    params.tmpMask1 = params.mask[splitSize];
    params.tmpMask2 = params.tmpMask1[splitSize];
    params.tmpMask3 = params.tmpMask2[splitSize];

    params.tmp1.SetSize(splitSize);
    params.tmp2.SetSize(splitSize);
    params.tmp3.SetSize(splitSize);
    params.tmp4.SetSize(splitSize);
    params.tmp5.SetSize(splitSize);
    params.tmp6.SetSize(splitSize);
    params.mask.SetSize(splitSize);
    params.tmpMask1.SetSize(splitSize);
    params.tmpMask2.SetSize(splitSize);
    params.tmpMask3.SetSize(splitSize);
    params.tmpScalar.SetSize(splitSize);

    params.splitSize = splitSize;
}
}  // namespace AscendC
#endif
#endif  // IMPL_MATH_LGAMMA_LGAMMA_COMMOM_BASIC_IMPL_H