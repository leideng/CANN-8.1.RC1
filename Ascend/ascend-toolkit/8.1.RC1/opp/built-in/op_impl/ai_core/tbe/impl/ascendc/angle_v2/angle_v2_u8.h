/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * \file angle_v2_u8.h
 * \brief
 */
#ifndef _ANGLE_V2_U8_H_
#define _ANGLE_V2_U8_H_

#include "angle_v2_base.h"

namespace AngleV2N {
using namespace AscendC;

template <typename yType>
class AngleV2U8 : public AngleV2Base<yType> {
 public:
  __aicore__ inline AngleV2U8() {
  }
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AngleV2TilingData* __restrict tilingData) {
    this->BaseMemberDataInit(tilingData);
    repeatTimes = (this->tileLength + this->mask - 1) / this->mask;
    blockLen = this->tileLength / dataPerBlock;

    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(y) + this->offset, this->blockLength);

    // pipe alloc memory to queue, the unit is Bytes
    pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(yType));
  }

  __aicore__ inline void Process() {
    LocalTensor<yType> zeroTensor = outQueue.AllocTensor<yType>();
    Duplicate(zeroTensor, static_cast<yType>(0.0), this->mask, repeatTimes, this->dupDstBlockStride,
              this->dupDstRepeatStride);

    // loop count need to be doubled, due to double buffer
    for (int32_t i = 0; i < this->tileNum; i++) {
      int32_t coreOffset = i * this->tileLength;
      DataCopy(yGm[coreOffset], zeroTensor, {1, blockLen, 0, 0});
    }

    if (this->lastTileLength > 0) {
      int32_t coreOffset = this->blockLength - this->lastTileLength;
      repeatTimes = (this->lastTileLength + this->mask - 1) / this->mask;
      blockLen = this->lastTileLength / dataPerBlock;
      DataCopy(yGm[coreOffset], zeroTensor, {1, blockLen, 0, 0});
    }
  }

 private:
  TPipe pipe;
  GlobalTensor<yType> yGm;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
  uint8_t repeatTimes;
  int32_t dataPerBlock = 32 / sizeof(yType);
  uint16_t blockLen = 1;
};
} // AngleV2N
#endif  // _ANGLE_V2_U8_H_
