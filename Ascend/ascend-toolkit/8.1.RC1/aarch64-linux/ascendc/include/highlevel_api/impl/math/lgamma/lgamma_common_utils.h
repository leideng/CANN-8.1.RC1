/* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lgamma_common_utils.h
 * \brief
 */
#ifndef IMPL_MATH_LGAMMA_LGAMMA_COMMOM_UTILS_H
#define IMPL_MATH_LGAMMA_LGAMMA_COMMOM_UTILS_H

#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 220
namespace AscendC {
namespace {
constexpr float f05 = 0.5;
constexpr float f07 = 0.7;
constexpr float f15 = 1.5;
constexpr float f58 = 5.8;
constexpr float f1 = 1.0;
constexpr float f2 = 2.0;
constexpr float f3 = 3.0;
constexpr float fn05 = -0.5;
constexpr float fn1 = -1.0;
constexpr float fn2 = -2.0;
constexpr float fn3 = -3.0;
constexpr float fPi = 3.1415927410125732421875;
constexpr uint32_t FLOAT_NOREUSE_CALC_PROC = 8;
constexpr uint32_t FLOAT_REUSE_CALC_PROC = 7;
constexpr uint32_t LGAMMA_HALF_CALC_PROCEDURE = 13;
constexpr uint32_t i2 = 2;
constexpr uint32_t i4 = 4;
constexpr uint32_t i6 = 6;
constexpr uint32_t i7 = 7;
constexpr uint32_t i16 = 16;
constexpr float PI = 3.14159265358979323846264338327950288;
constexpr float t0 = 0.0f;
constexpr float t4 = 4.0f;
constexpr float t5 = 5.0f;
constexpr float t01 = 0.1f;
constexpr float t12 = 12.0f;
constexpr float N01 = -0.1f;

constexpr size_t params007Len = 7U;
constexpr float params007[params007Len] = {0.00358751555905,
    -0.00547128543258,
    -0.0446271263063,
    0.167317703366,
    -0.0421359799802,
    -0.655867278576,
    0.577215373516};

constexpr size_t params0715Len = 11U;
constexpr float params0715[params0715Len] = {0.0458826646209,
    0.103739671409,
    0.122803635895,
    0.127524212003,
    0.143216684461,
    0.169343575835,
    0.207407936454,
    0.27058750391,
    0.400685429573,
    0.82246696949,
    0.577215671539};

constexpr size_t params153Len = 10U;
constexpr float params153[params153Len] = {4.95984932058e-05,
    -0.000220894842641,
    0.000541314249858,
    -0.00120451697148,
    0.00288425176404,
    -0.00738275796175,
    0.0205813199282,
    -0.067352488637,
    0.322467029095,
    0.422784328461};

constexpr size_t params378X1Len = 4U;
constexpr size_t params378X2Len = 3U;
constexpr float params378X1[params378X1Len] = {-748.890319824, -12349.7421875, -41061.375, -48310.6640625};
constexpr float params378X2[params378X2Len] = {-259.250976562, -10777.1796875, -92685.046875};

constexpr size_t params58Len = 2U;
constexpr float params58[params58Len] = {0.000777830660809, -0.00277765537612};

constexpr size_t negParamsOddLen = 4U;
constexpr float negParamsOdd[negParamsOddLen] = {0.00002427957952022552490234375,
    -0.001388786011375486850738525390625,
    0.0416667275130748748779296875,
    -0.4999999701976776123046875};
constexpr size_t negParamsEvenLen = 3U;
constexpr float negParamsEven[negParamsEvenLen] = {
    -0.000195746586541645228862762451171875, 0.0083327032625675201416015625, -0.16666662693023681640625};
}  // namespace

struct LGammaFParams {
    __aicore__ LGammaFParams()
    {}
    LocalTensor<float> tmp1;
    LocalTensor<float> tmp2;
    LocalTensor<float> tmp3;
    LocalTensor<float> tmp4;
    LocalTensor<float> tmp5;
    LocalTensor<float> tmp6;
    LocalTensor<float> tmpScalar;
    LocalTensor<uint8_t> mask;
    LocalTensor<uint8_t> tmpMask1;
    LocalTensor<uint8_t> tmpMask2;
    LocalTensor<uint8_t> tmpMask3;
    UnaryRepeatParams unaryParams;
    BinaryRepeatParams binaryParams;
    uint32_t splitSize;
};

}  // namespace AscendC
#endif
#endif  // IMPL_MATH_LGAMMA_LGAMMA_COMMOM_utils_H