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
 * \file math_util.h
 * \brief
 */

#ifndef IMPL_MATMUL_TILING_MATH_UTIL_H
#define IMPL_MATMUL_TILING_MATH_UTIL_H

#include <array>
#include <cstdint>
#include <vector>
namespace matmul_tiling {
class MathUtil {
public:
    static bool IsEqual(float leftValue, float rightValue);
    static int32_t CeilDivision(int32_t num1, int32_t num2);
    static int32_t Align(int32_t num1, int32_t num2);
    static int32_t AlignDown(int32_t num1, int32_t num2);
    static bool CheckMulOverflow(int32_t a, int32_t b, int32_t &c);
    static int32_t MapShape(int32_t shape, bool roundUpFlag = true);
    static void AddFactor(std::vector<int32_t> &dimsFactors, int32_t dim);
    static void GetFactorCnt(const int32_t shape, int32_t &factorCnt, const int32_t factorStart,
        const int32_t factorEnd);
    static void GetFactorLayerCnt(const int32_t shape, int32_t &factorCnt, const int32_t factorStart,
        const int32_t factorEnd);
    static bool CheckFactorNumSatisfy(const int32_t dim);
    static int32_t FindBestSingleCore(const int32_t oriShape, const int32_t mappedShape, const int32_t coreNum,
        bool isKDim);
    static void GetFactors(std::vector<int32_t> &factorList, int32_t srcNum, int32_t minFactor, int32_t maxFactor);
    static void GetFactors(std::vector<int32_t> &factorList, int32_t srcNum, int32_t maxFactor);
    static void GetBlockFactors(std::vector<int32_t> &factorList, const int32_t oriShape, const int32_t mpShape,
        const int32_t coreNum, const int32_t maxNum);
    static int32_t GetNonFactorMap(std::vector<int32_t> &factorList, int32_t srcNum, int32_t maxFactor);
};
} // namespace matmul_tiling
#endif // _MATH_UTIL_H_