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
 * \file angle_v2.h
 * \brief
 */
#ifndef _ANGLE_V2_H_
#define _ANGLE_V2_H_

#include "angle_v2_base.h"

namespace AngleV2N {
using namespace AscendC;

template <typename yType>
class AngleV2 : public AngleV2Base<yType> {
 public:
  __aicore__ inline AngleV2() {
  }
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AngleV2TilingData* __restrict tilingData) {
    this->BaseMemberDataInit(tilingData);
    repeatTimes = (this->tileLength + this->mask - 1) / this->mask;
    blockLen = this->tileLength / dataPerBlock;

    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(x) + this->offset, this->blockLength);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(y) + this->offset, this->blockLength);

    // pipe alloc memory to queue, the unit is Bytes
    pipe.InitBuffer(inQueue, BUFFER_NUM, this->tileLength * sizeof(yType));
    pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(yType));

    pipe.InitBuffer(maskBuf1, this->tileLength * sizeof(uint8_t));
    pipe.InitBuffer(zeroBuf, this->tileLength * sizeof(yType));
    pipe.InitBuffer(piBuf, this->tileLength * sizeof(yType));
    pipe.InitBuffer(nanBuf, this->tileLength * sizeof(yType));
  }

  __aicore__ inline void Process() {
    BufferGet();
    // loop count need to be doubled, due to double buffer
    for (int32_t i = 0; i < this->tileNum; i++) {
      int32_t coreOffset = i * this->tileLength;
      CopyIn(coreOffset);
      Compute(this->tileLength);
      CopyOut(coreOffset);
    }

    if (this->lastTileLength > 0) {
      int32_t coreOffset = this->blockLength - this->lastTileLength;
      repeatTimes = (this->lastTileLength + this->mask - 1) / this->mask;
      blockLen = this->lastTileLength / dataPerBlock;
      CopyIn(coreOffset);
      Compute(this->lastTileLength);
      CopyOut(coreOffset);
    }
  }

 private:
  __aicore__ inline void BufferGet() {
    zeroTensor = zeroBuf.Get<yType>();
    piTensor = piBuf.Get<yType>();
    nanTensor = nanBuf.Get<yType>();
    mask1 = maskBuf1.Get<uint8_t>();

    Duplicate(zeroTensor, static_cast<yType>(0.0), this->mask, repeatTimes, this->dupDstBlockStride,
              this->dupDstRepeatStride);
    Duplicate(piTensor, static_cast<yType>(constData.const_pi), this->mask, repeatTimes, this->dupDstBlockStride,
              this->dupDstRepeatStride);
    Duplicate(nanTensor, static_cast<yType>(NAN), this->mask, repeatTimes, this->dupDstBlockStride,
              this->dupDstRepeatStride);
  }

  __aicore__ inline void CopyIn(int32_t coreOffset) {
    // alloc tensor from queue memory
    LocalTensor<yType> xLocal = inQueue.AllocTensor<yType>();
    // copy progress_th tile from global tensor to local tensor
    DataCopy(xLocal, xGm[coreOffset], {1, blockLen, 0, 0});
    // enque input tensors to VECIN queue
    inQueue.EnQue(xLocal);
  }

  __aicore__ inline void Compute(int32_t calCount) {
    // deque input tensors from VECIN queue
    LocalTensor<yType> input = inQueue.DeQue<yType>();
    LocalTensor<yType> result = outQueue.AllocTensor<yType>();

    // result = if input >= 0 then 0 else pi
    Compare(mask1, input, zeroTensor, CMPMODE::GE, this->mask, repeatTimes, this->repeatParams);
    this->DoSelect(result, mask1, zeroTensor, piTensor, this->mask, repeatTimes);

    // select nan
    Compare(mask1, input, input, CMPMODE::EQ, this->mask, repeatTimes, this->repeatParams);
    this->DoSelect(result, mask1, result, nanTensor, this->mask, repeatTimes);

    // enque the output tensor to VECOUT queue
    outQueue.EnQue<yType>(result);
    // free input tensors for reuse
    inQueue.FreeTensor(input);
  }

  __aicore__ inline void CopyOut(int32_t coreOffset) {
    // deque output tensor from VECOUT queue
    LocalTensor<yType> result = outQueue.DeQue<yType>();
    // copy progress_th tile from local tensor to global tensor
    DataCopy(yGm[coreOffset], result, {1, blockLen, 0, 0});
    // free output tensor for reuse
    outQueue.FreeTensor(result);
  }

 private:
  TPipe pipe;
  ConstData constData;
  uint8_t repeatTimes;
  GlobalTensor<yType> xGm, yGm;

  TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
  TBuf<TPosition::VECCALC> maskBuf1, piBuf, nanBuf, zeroBuf;

  LocalTensor<yType> zeroTensor, piTensor, nanTensor;
  LocalTensor<uint8_t> mask1;
  int32_t dataPerBlock = 32 / sizeof(yType);
  uint16_t blockLen = 1;
};
} // AngleV2N
#endif  // _ANGLE_V2_H_
