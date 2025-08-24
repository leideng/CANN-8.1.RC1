/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_swiglu_quant_utils.h
 * \brief
 */
#ifndef ASCENDC_GROUPED_MATMUL_SWIGLU_QUANT_UTILS_H
#define ASCENDC_GROUPED_MATMUL_SWIGLU_QUANT_UTILS_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace GROUPED_MATMUL_SWIGLU_QUANT {
using namespace AscendC;
constexpr uint32_t INT8_BITS = 8;  // a int8 number has 8 bits
constexpr int32_t MKN_LIST_LEN = 128;  // 128: predefined array legnth
constexpr uint32_t UB_BLOCK_UNIT_SIZE = 32;  // 32: a block has 32 bytes data
constexpr uint32_t THRESHOLD_BLOCK_NUM = 8;
constexpr uint32_t UB_BLOCK_DOUBLE_UNIT_SIZE = 64;  // 64: a block has 64 bytes data
constexpr uint32_t HALF_UB_BLOCK_UNIT_SIZE = UB_BLOCK_UNIT_SIZE / 2;  // 2: a float16 data has two bytes
constexpr uint32_t SINGLE_CORE_M = 128;
constexpr uint32_t SINGLE_CORE_N = 256;
constexpr uint32_t SINGLE_CORE_K = 7168;
constexpr uint32_t BASIC_M = 128;
constexpr uint32_t BASIC_N = 256;
constexpr uint32_t BASIC_K = 128;
constexpr uint32_t STEP_M = 1;
constexpr uint32_t STEP_N = 1;
constexpr uint32_t STEP_Ka = 4;
constexpr uint32_t STEP_Kb = 4;
constexpr uint32_t DEPTH_A1 = 8;
constexpr uint32_t DEPTH_B1 = 8;
constexpr MatmulConfig NZ_CFG_MDL = GetMDLConfig(false, false, 0, true, false, false, false);
constexpr MatmulConfig GetMMCFG() { 
    MatmulConfig MM_CFG = NZ_CFG_MDL;
    MM_CFG.singleCoreM = SINGLE_CORE_M; 
    MM_CFG.singleCoreN= SINGLE_CORE_N; 
    MM_CFG.singleCoreK= SINGLE_CORE_K;
    MM_CFG.basicM= BASIC_M; 
    MM_CFG.basicN= BASIC_N; 
    MM_CFG.basicK= BASIC_K;
    return MM_CFG;
}

constexpr static MatmulApiStaticTiling GetMMTiling(const MatmulApiStaticTiling& mmTiling)
{
  MatmulApiStaticTiling tiling = mmTiling;
  tiling.stepM = STEP_M;
  tiling.stepN = STEP_N;
  tiling.stepKa = STEP_Ka;
  tiling.stepKb = STEP_Kb;
  tiling.depthA1 = DEPTH_A1;
  tiling.depthB1 = DEPTH_B1;
  return tiling;
}
template<class AT_, class BT_, class CT_, class BiasT_, const MatmulConfig& MM_CFG>
struct MMImplType {
  using AT = AT_;
  using BT = BT_;
  using CT = CT_;
  using BiasT = BiasT_;
  static constexpr MatmulConfig cfg = GetMMCFG();
  static constexpr MatmulApiStaticTiling mdl = GetMMTiling(GetMatmulApiTiling<AT, BT, CT, BiasT>(cfg));
  using MT = matmul::MatmulImpl<AT, BT, CT, BiasT, mdl>;
};

struct MNConfig {
  uint32_t m = 0;
  uint32_t k = 0;
  uint32_t n = 0;
  uint32_t baseM = 0;
  uint32_t baseN = 0;
  uint32_t mIdx = 0;
  uint32_t nIdx = 0;
  uint32_t blockDimM = 0;
  uint32_t blockDimN = 0;
  uint32_t singleM = 0;
  uint32_t singleN = 0;
  uint64_t wBaseOffset = 0;
  uint64_t nAxisBaseOffset = 0;
  uint64_t mAxisBaseOffset = 0;
  uint64_t xBaseOffset = 0;
  uint64_t yBaseOffset = 0;
  uint64_t wOutOffset = 0;
  uint64_t workSpaceOffset = 0;
};

struct VecConfig {
  int64_t M = 0;
  int64_t usedCoreNum = 0;
  int64_t startOffset = 0;
  int64_t curOffset = 0;
  int64_t startIdx = 0;
  int64_t curIdx = 0;
  int64_t taskNum = 0;
  int64_t curGroupIdx = 0;
  int64_t outLoopNum = 0;
  int64_t innerLoopNum = 0;
  int64_t tailLoopNum = 0;
  int64_t nextUpadteInterVal = 0;
};

template <uint32_t base, typename T = uint32_t>
__aicore__ inline T AlignUp(T a) {
  return (a + base - 1) / base * base;
}

template <typename T>
__aicore__ inline T AlignUp(T a, T base) {
  return (a + base - 1) / base * base;
}

template <typename T>
__aicore__ inline T AlignDown(T a, T base) {
  if (unlikely(base == 0)) {
    return a;
  }
  return a / base * base;
}

template <>
__aicore__ inline uint32_t AlignUp<4, uint32_t>(uint32_t a) {
  // to be Multiple of 4, result should be in a format of b(xxxx,x100).
  // This means last two bits should be zero, requiring that
  // result = num & b(1111,1100) = num & (~3).
  // &(~3) operator may reduces num into the range [num, num - 3].
  // As the result should be no less than a (result >= a), it means num - 3 >= a in the worst case.
  // In this case, num >= a+3. On the other hand, num should also be less then a+4, otherwise,
  // the result will not be least multiple of 4 for 3. In other cases like [num, num - 2],
  // num = a + 3 also satisfies the goal condition.
  return (a + 3) & ~3;  // & ~3: set last two bits of (a+3) to be zero
}

template <>
__aicore__ inline uint32_t AlignUp<8, uint32_t>(uint32_t a) {
  // In general, if we want to get the least multiple of b (b is the power of 2) for a,
  // it comes to a conclusion from the above comment: result = (a + (b - 1)) & (~b)
  return (a + 7) & ~7;  // & ~7: set last four bits of (a+7) to be zero
}

template <>
__aicore__ inline uint32_t AlignUp<16, uint32_t>(uint32_t a) {
  // In general, if we want to get the least multiple of b (b is the power of 2) for a,
  // it comes to a conclusion from the above comment: result = (a + (b - 1)) & (~b)
  return (a + 15) & ~15;  // & ~15: set last four bits of (a+15) to be zero
}

template <>
__aicore__ inline uint32_t AlignUp<32, uint32_t>(uint32_t a) {
  // refer to the above comments.
  return (a + 31) & ~31;  // & ~31: set last five bits of (a+31) to be zero}
}

}  // namespace GROUPED_MATMUL

#endif  // ASCENDC_GROUPED_MATMUL_UTILS_H
