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
 * \file angle_v2_int.h
 * \brief
 */
#ifndef _ANGLE_V2_INT_H_
#define _ANGLE_V2_INT_H_

#include "angle_v2_base.h"

namespace AngleV2N {
using namespace AscendC;

template <typename xType, typename yType>
class AngleV2Int : public AngleV2Base<yType> {
 public:
  __aicore__ inline AngleV2Int() {
  }
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AngleV2TilingData* __restrict tilingData) {
    this->BaseMemberDataInit(tilingData);
    repeatTimes = (this->tileLength + this->mask - 1) / this->mask;

    InitDataAddress(x, y);

    // pipe alloc memory to queue, the unit is Bytes
    pipe.InitBuffer(inQueue, BUFFER_NUM, this->tileLength * sizeof(xType));
    pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(yType));

    pipe.InitBuffer(maskBuf1, this->tileLength * sizeof(uint8_t));
    pipe.InitBuffer(zeroBuf, this->tileLength * sizeof(yType));
    pipe.InitBuffer(piBuf, this->tileLength * sizeof(yType));

    if constexpr (std::is_same<xType, int8_t>::value) {
      pipe.InitBuffer(halfBuf, this->tileLength * sizeof(half));
    } 
#if (__CCE_AICORE__ < 200)
    if constexpr (std::is_same<xType, int16_t>::value) {
      pipe.InitBuffer(bufB16, COEFFICENT * this->tileLength * sizeof(half));
      pipe.InitBuffer(halfBuf, this->tileLength * sizeof(half));
    } else if constexpr (std::is_same<xType, int64_t>::value) {
      pipe.InitBuffer(bufB32, COEFFICENT * this->tileLength * sizeof(yType));
      pipe.InitBuffer(bufB16, COEFFICENT * this->tileLength * sizeof(half));
      pipe.InitBuffer(halfBuf, this->tileLength * sizeof(half));
    }
#endif
  }

  __aicore__ inline void InitDataAddress(GM_ADDR x, GM_ADDR y) {
    uint32_t current_offset = this->offset;
    lastTileLengthOut = this->lastTileLength;
    lastTileLengthIn = this->lastTileLength;

    uint32_t alignNumFp32 = 32 / sizeof(yType);
    uint32_t totalLengthAlignedFP32 = (this->totalLength + alignNumFp32 - 1) / alignNumFp32 * alignNumFp32;
    uint32_t diffLength = this->totalLengthAligned - totalLengthAlignedFP32;
    if constexpr (std::is_same<xType, int8_t>::value) {
      maskInt8 = this->mask * COEFFICENT;
      repeatTimesInt8 = repeatTimes % COEFFICENT ? repeatTimes / COEFFICENT + 1 : repeatTimes / COEFFICENT;

      if (GetBlockIdx() == this->formerNum + this->tailNum - 1) {
          if (this->offset < diffLength) {
            // back the compute elements when the address can not back
            lastTileLengthOut =  this->lastTileLength - diffLength;
          } else {
            // address back to avoid data stampede
            current_offset = this->offset - diffLength;
          }
      }
    } else if constexpr (std::is_same<xType, int16_t>::value) {
      if (GetBlockIdx() == this->formerNum + this->tailNum - 1) {
          if (this->offset < diffLength) {
            // back the compute elements when the address can not back
            lastTileLengthOut =  this->lastTileLength - diffLength;
          } else {
            // address back to avoid data stampede
            current_offset = this->offset - diffLength;
          }
      }
    } else if constexpr (std::is_same<xType, int64_t>::value) {
      maskInt64 = this->mask / COEFFICENT;
      repeatTimesInt64 = repeatTimes * COEFFICENT;

      uint32_t alignNumInt64 = 32 / sizeof(xType);
      uint32_t totalLengthAlignedInt64 = (this->totalLength + alignNumInt64 - 1) / alignNumInt64 * alignNumInt64;
      diffLength = this->totalLengthAligned - totalLengthAlignedInt64;
      if (GetBlockIdx() == this->formerNum + this->tailNum - 1) {
        if (this->offset < diffLength) {
          // back the compute elements when the address can not back
          lastTileLengthIn = this->lastTileLength - diffLength;
        } else {
          // address back to avoid data stampede
          current_offset = this->offset - diffLength;
        }
      }
    }

    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ xType*>(x) + current_offset, this->blockLength);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(y) + current_offset, this->blockLength);
  }

  __aicore__ inline void Process() {
    BufferGet();
    uint32_t dataPerBlockIn = 32 / sizeof(xType);
    uint32_t dataPerBlockOut = 32 / sizeof(yType);

    blockLenIn = this->tileLength / dataPerBlockIn;
    blockLenOut = this->tileLength / dataPerBlockOut;

    // loop count need to be doubled, due to double buffer
    for (int32_t i = 0; i < this->tileNum; i++) {
      int32_t coreOffset = i * this->tileLength;
      CopyIn(coreOffset);
      Compute();
      CopyOut(coreOffset);
    }

    if (this->lastTileLength > 0) {
      int32_t coreOffset = this->blockLength - this->lastTileLength;
      repeatTimes = (this->lastTileLength + this->mask - 1) / this->mask;
      
      blockLenIn = lastTileLengthIn / dataPerBlockIn;
      blockLenOut = lastTileLengthOut / dataPerBlockOut;
      CopyIn(coreOffset);
      Compute();
      CopyOut(coreOffset);
    }
  }

 private:
  __aicore__ inline void BufferGet() {
    mask1 = maskBuf1.Get<uint8_t>();
    zeroTensor = zeroBuf.Get<yType>();
    piTensor = piBuf.Get<yType>();

    if constexpr (std::is_same<xType, int8_t>::value) {
      halfTensor = halfBuf.Get<half>();
    }

#if (__CCE_AICORE__ < 200)
    if constexpr (std::is_same<xType, int16_t>::value) {
      halfTensor = halfBuf.Get<half>();
      halfTensorDouble = bufB16.Get<half>();
    } else if constexpr (std::is_same<xType, int64_t>::value) {
      halfTensor = halfBuf.Get<half>();
      halfTensorDouble = bufB16.Get<half>();
      castTensor = bufB32.Get<yType>();
    }
#endif

    Duplicate(zeroTensor, static_cast<yType>(0.0), this->mask, repeatTimes, this->dupDstBlockStride,
              this->dupDstRepeatStride);
    Duplicate(piTensor, static_cast<yType>(constData.const_pi), this->mask, repeatTimes, this->dupDstBlockStride,
              this->dupDstRepeatStride);
  }

  __aicore__ inline void CopyIn(int32_t coreOffset) {
    // alloc tensor from queue memory
    LocalTensor<xType> xLocal = inQueue.AllocTensor<xType>();
    // copy progress_th tile from global tensor to local tensor
    DataCopy(xLocal, xGm[coreOffset], {1, blockLenIn, 0, 0});
    // enque input tensors to VECIN queue
    inQueue.EnQue(xLocal);
  }

  __aicore__ inline void Cast2Float(LocalTensor<xType> input, LocalTensor<yType> result) {
    if constexpr (std::is_same<xType, int8_t>::value) {
      // Cast from int8_t to float16
      Cast(halfTensor, input, RoundMode::CAST_NONE, maskInt8, repeatTimesInt8, this->CastHighParams);
      // Cast from float16 to float32
      Cast(result, halfTensor, RoundMode::CAST_NONE, this->mask, repeatTimes, this->CastHighParams);
    } else if constexpr (std::is_same<xType, int16_t>::value) {
#if (__CCE_AICORE__ >= 200)
      // Cast for 910B from int16_t to float
      Cast(result, input, RoundMode::CAST_NONE, this->mask, repeatTimes, this->CastHighParams);
#else
      // ReinterpretCast from int16 to int8, then cast to half
      Cast(halfTensorDouble, input.template ReinterpretCast<int8_t>(), RoundMode::CAST_NONE, this->mask * COEFFICENT,
           repeatTimes, this->CastHighParams);
      // Duplicate odds elements as 0
      Duplicate(halfTensorDouble, static_cast<half>(0.0), maskDup910, repeatTimes, this->dupDstBlockStride,
                this->dupDstRepeatStride);
      PairReduceSum(halfTensor, halfTensorDouble, 1, pairReduceSumCount, 1, 1, pairReduceSumRepStride);
      // Cast from half to float
      Cast(result, halfTensor, RoundMode::CAST_NONE, this->mask, repeatTimes, this->CastHighParams);
#endif
    } else if constexpr (std::is_same<xType, int32_t>::value) {
      // Cast from int32_t to float
      Cast(result, input, RoundMode::CAST_NONE, this->mask, repeatTimes, this->CastKeepParams);
    } else if constexpr (std::is_same<xType, int64_t>::value) {
#if (__CCE_AICORE__ >= 200)
      // Cast for 910B from int64_t to float
      Cast(result, input, RoundMode::CAST_FLOOR, maskInt64, repeatTimesInt64, this->CastDownParams);
#else
      // ReinterpretCast from int64 to int32, then cast to float
      Cast(castTensor, input.template ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, this->mask,
           repeatTimes * COEFFICENT, this->CastKeepParams);
      // Cast from float to half
      Cast(halfTensorDouble, castTensor, RoundMode::CAST_NONE, this->mask, repeatTimes * COEFFICENT,
            this->CastDownParams);
      // Duplicate odds elements as 0
      Duplicate(halfTensorDouble, static_cast<half>(0.0), maskDup910, repeatTimes, this->dupDstBlockStride,
                this->dupDstRepeatStride);
      PairReduceSum(halfTensor, halfTensorDouble, 1, pairReduceSumCount, 1, 1, pairReduceSumRepStride);
      // Cast from half to float
      Cast(result, halfTensor, RoundMode::CAST_NONE, this->mask, repeatTimes, this->CastHighParams);
#endif
    }
  }

  __aicore__ inline void Compute() {
    // deque input tensors from VECIN queue
    LocalTensor<xType> input = inQueue.DeQue<xType>();
    LocalTensor<yType> result = outQueue.AllocTensor<yType>();

    Cast2Float(input, result);

    // result = if input >= 0 then 0 else pi
    Compare(mask1, result, zeroTensor, CMPMODE::GE, this->mask, repeatTimes, this->repeatParams);
    this->DoSelect(result, mask1, zeroTensor, piTensor, this->mask, repeatTimes);

    // enque the output tensor to VECOUT queue
    outQueue.EnQue<yType>(result);
    // free input tensors for reuse
    inQueue.FreeTensor(input);
  }

  __aicore__ inline void CopyOut(int32_t coreOffset) {
    // deque output tensor from VECOUT queue
    LocalTensor<yType> result = outQueue.DeQue<yType>();
    // copy progress_th tile from local tensor to global tensor
    DataCopy(yGm[coreOffset], result, {1, blockLenOut, 0, 0});
    // free output tensor for reuse
    outQueue.FreeTensor(result);
  }

 private:
  TPipe pipe;
  ConstData constData;
  uint8_t repeatTimes;
  GlobalTensor<xType> xGm;
  GlobalTensor<yType> yGm;

  TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
  TBuf<TPosition::VECCALC> maskBuf1, piBuf, zeroBuf;

  LocalTensor<yType> zeroTensor, piTensor;
  LocalTensor<uint8_t> mask1;
  uint16_t blockLenIn = 1;
  uint16_t blockLenOut = 1;
  uint32_t lastTileLengthOut = 64;
  uint32_t lastTileLengthIn = 64;

  uint64_t maskInt8;
  uint8_t repeatTimesInt8;
  LocalTensor<yType> castTensor;
  uint64_t maskInt64;
  uint8_t repeatTimesInt64;
  uint64_t maskDup910[2] = {6148914691236517205, 6148914691236517205};
  TBuf<TPosition::VECCALC> bufB32, bufB16, halfBuf;
  LocalTensor<half> halfTensor, halfTensorDouble;
  int32_t pairReduceSumCount = 128;
  int32_t pairReduceSumRepStride = 8;
};
} // AngleV2N
#endif  // _ANGLE_V2_INT_H_
