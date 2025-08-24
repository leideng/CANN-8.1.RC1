/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file angle_v2_base.h
 * \brief
 */
#ifndef _ANGLE_V2_BASE_H_
#define _ANGLE_V2_BASE_H_

#include <cmath>
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace AngleV2N {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t COEFFICENT = 2;

struct ConstData {
  const float const_pi = M_PI;
  const float const_pi_by_two = const_pi / 2;
  const float const_pi_by_four = const_pi / 4;
  const float const_pi_by_three_quarters = const_pi * 3 / 4;

  const float const_neg_pi = -const_pi;
  const float const_neg_pi_by_two = -const_pi_by_two;
  const float const_neg_pi_by_four = -const_pi_by_four;
  const float const_neg_pi_by_three_quarters = -const_pi_by_three_quarters;
};

template <typename yType>
class AngleV2Base {
 public:
  __aicore__ inline AngleV2Base() {
  }
  __aicore__ inline void BaseMemberDataInit(const AngleV2TilingData* __restrict tilingData) {
    totalLength = tilingData->totalLength;
    alignNum = tilingData->alignNum;
    formerNum = tilingData->formerNum;
    formerLength = tilingData->formerLength;
    tailNum = tilingData->tailNum;
    tailLength = tilingData->tailLength;
    totalLengthAligned = tilingData->totalLengthAligned;
    tileLength = tilingData->tileLength;
    mask = tilingData->dataPerRepeat;
    
    if (GetBlockIdx() < formerNum) {
      blockLength = formerLength;
      offset = blockLength * GetBlockIdx();
    } else {
      blockLength = tailLength;
      offset = formerLength * formerNum + tailLength * (GetBlockIdx() - formerNum);
    }
    CalTileLength();
    tileNum = blockLength / tileLength;
    lastTileLength = blockLength % tileLength;
  }

  __aicore__ inline void CalTileLength() {
#if (__CCE_AICORE__ >= 200)
    // calculate tileLength for 910B
    if (blockLength <= tileLength) {
        tileLength = (blockLength + mask - 1) / mask * mask;
    }
#else
    tileLength = mask;
#endif
  }

  template<typename T>
  __aicore__ inline void DoSelect(LocalTensor<T> &dstLocal, LocalTensor<uint8_t> &selMask,
                                  LocalTensor<T> &src0Local, LocalTensor<T> &src1Local, uint64_t mask,
                                  uint8_t repeatTimes) {
#if (__CCE_AICORE__ >= 200)
    // Select for 910B mode 2
    Select(dstLocal, selMask, src0Local, src1Local, SELMODE::VSEL_TENSOR_TENSOR_MODE, mask, repeatTimes, repeatParams);
#else
    // Select for 910 mode 0
    Select(dstLocal, selMask, src0Local, src1Local, SELMODE::VSEL_CMPMASK_SPR, mask, repeatTimes, repeatParams);
#endif
  }

 protected:
  uint32_t totalLength, alignNum, totalLengthAligned;
  uint32_t formerNum, formerLength, tailNum, tailLength;
  uint32_t blockLength, offset, tileNum, lastTileLength;
  uint32_t tileLength = 64;
  uint64_t mask = 64;
  BinaryRepeatParams repeatParams = {1, 1, 1, 8, 8, 8};
  UnaryRepeatParams CastDownParams= {1, 1, 4, 8};
  UnaryRepeatParams CastKeepParams= {1, 1, 8, 8};
  UnaryRepeatParams CastHighParams= {1, 1, 8, 4};
  uint16_t dupDstBlockStride = 1;
  uint8_t dupDstRepeatStride = 8;
};
} // AngleV2N
#endif  // _ANGLE_V2_BASE_H_
