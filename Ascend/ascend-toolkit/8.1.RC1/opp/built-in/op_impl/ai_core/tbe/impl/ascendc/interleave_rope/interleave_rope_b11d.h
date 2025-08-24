/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file interleave_rope_b11d.h
 * \brief
 */
#ifndef _INTERLEAVE_ROPE_B11D_H_
#define _INTERLEAVE_ROPE_B11D_H_
#include "../inc/platform.h"

namespace InterleaveRope {
using namespace AscendC;

constexpr uint64_t SPLIT_BATCH = 0;
constexpr uint64_t SPLIT_NS = 1;

template <typename T>
class KernelInterleaveRopeB11D {
 public:
  __aicore__ inline KernelInterleaveRopeB11D(TPipe* pipe, const InterleaveRopeTilingData* tiling)
      : pipe_(pipe), tilingData_(tiling) {
  }

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y) {
    /*
     * For each block, process
     * x:   [B, N, S, 64]
     * cos: [B, 1, 1, 64]
     * sin: [B, 1, 1, 64]
     * y:   [B, N, S, 64]
     */

    numHead_ = tilingData_->numHead;
    seqLength_ = tilingData_->seqLength;
    NS_ = numHead_ * seqLength_;

    batchsPerBlock_ = tilingData_->batchsPerBlock;
    curBlockBatchs_ = tilingData_->batchsPerBlock;
    batchLoops_ = tilingData_->batchLoops;
    batchPerLoop_ = tilingData_->batchPerLoop;
    batchLastLoop_ = tilingData_->batchLastLoop;
    hiddenDimLoops_ = tilingData_->hiddenDimLoopsPerBlock;
    hiddenDimCountPerLoop_ = tilingData_->hiddenDimCountPerLoopPerBlock;
    hiddenDimCountLastLoop_ = tilingData_->hiddenDimCountLastLoopPerBlock;
    if (GetBlockIdx() == GetBlockNum() - 1) {
      curBlockBatchs_ = tilingData_->batchsLastBlock;
      hiddenDimLoops_ = tilingData_->hiddenDimLoopsLastBlock;
      hiddenDimCountPerLoop_ = tilingData_->hiddenDimCountPerLoopLastBlock;
      hiddenDimCountLastLoop_ = tilingData_->hiddenDimCountLastLoopLastBlock;
    }

    if (tilingData_->splitAxis == SPLIT_BATCH) {
      sinCosIdxOffsetBase_ = GetBlockIdx() * batchsPerBlock_;
    }

    xGm.SetGlobalBuffer((__gm__ T*)x);
    yGm.SetGlobalBuffer((__gm__ T*)y);
    cosGm.SetGlobalBuffer((__gm__ T*)cos);
    sinGm.SetGlobalBuffer((__gm__ T*)sin);

    // init pipe
    pipe_->InitBuffer(inQueueX, 1, hiddenDimCountPerLoop_ * hiddenDim * sizeof(T));
    pipe_->InitBuffer(outQueueY, 1, hiddenDimCountPerLoop_ * hiddenDim * sizeof(float) * numTwo);
    pipe_->InitBuffer(inQueueCos, 1, hiddenDim * sizeof(float) * numTwo);
    pipe_->InitBuffer(inQueueSin, 1, hiddenDim * sizeof(float) * numTwo);

    pipe_->InitBuffer(bufferReal, hiddenDimCountPerLoop_ * hiddenDimHalf * sizeof(float) * numTwo);
    pipe_->InitBuffer(bufferImag, hiddenDimCountPerLoop_ * hiddenDimHalf * sizeof(float) * numTwo);
    pipe_->InitBuffer(buffer_, hiddenDimCountPerLoop_ * hiddenDim * sizeof(float));
  }

  __aicore__ inline void Process() {
    for (int64_t i = 0; i < curBlockBatchs_; i++) {
      Rope(i);
    }
  }

  __aicore__ inline void Rope(int64_t idx) {
    int64_t sinCosIdxOffset = sinCosIdxOffsetBase_ + idx;
    // rope
    LocalTensor<float> cosLocal = inQueueCos.AllocTensor<float>();
    DataCopy(cosLocal[hiddenDim].template ReinterpretCast<T>(), cosGm[sinCosIdxOffset * hiddenDim], hiddenDim);
    
    inQueueCos.EnQue(cosLocal);
    cosLocal = inQueueCos.DeQue<float>();
    Cast(cosLocal, cosLocal[hiddenDim].template ReinterpretCast<T>(), RoundMode::CAST_NONE, hiddenDim);

    LocalTensor<float> sinLocal = inQueueSin.AllocTensor<float>();
    DataCopy(sinLocal[hiddenDim].template ReinterpretCast<T>(), sinGm[sinCosIdxOffset * hiddenDim], hiddenDim);
    inQueueSin.EnQue(sinLocal);
    sinLocal = inQueueSin.DeQue<float>();
    Cast(sinLocal, sinLocal[hiddenDim].template ReinterpretCast<T>(), RoundMode::CAST_NONE, hiddenDim);
    int64_t factor = hiddenDimCountPerLoop_;
    int64_t batchOffset = tilingData_->splitAxis == SPLIT_BATCH ? (GetBlockIdx() * batchsPerBlock_ + idx) * NS_ : idx * NS_ + GetBlockIdx() * tilingData_->hiddenDimCountPerBlock;
    for (int i = 0; i < hiddenDimLoops_; i++) {
      if (i == hiddenDimLoops_ - 1) {
        factor = hiddenDimCountLastLoop_;
      }
      int64_t xOffset = (batchOffset + i * hiddenDimCountPerLoop_) * hiddenDim;

      // load xLocal
      LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
      DataCopy(xLocal, xGm[xOffset], factor * hiddenDim);
      inQueueX.EnQue(xLocal);
      xLocal = inQueueX.DeQue<T>();

      // split Real and Imag
      LocalTensor<float> realLocal = bufferReal.Get<float>();
      LocalTensor<float> imagLocal = bufferImag.Get<float>();
      LocalTensor<float> buf_ = buffer_.Get<float>();
      uint64_t rsvdCnt = 0;
      GatherMask(realLocal[hiddenDimCountPerLoop_ * hiddenDimHalf].template ReinterpretCast<T>(), xLocal, 1, true, factor * hiddenDim, {1, 1, 8, 0}, rsvdCnt);
      GatherMask(imagLocal[hiddenDimCountPerLoop_ * hiddenDimHalf].template ReinterpretCast<T>(), xLocal, numTwo, true, factor * hiddenDim, {1, 1, 8, 0}, rsvdCnt);
      Cast(realLocal, realLocal[hiddenDimCountPerLoop_ * hiddenDimHalf].template ReinterpretCast<T>(), RoundMode::CAST_NONE, factor * hiddenDimHalf);
      Cast(imagLocal, imagLocal[hiddenDimCountPerLoop_ * hiddenDimHalf].template ReinterpretCast<T>(), RoundMode::CAST_NONE, factor * hiddenDimHalf);
      inQueueX.FreeTensor(xLocal);

      uint64_t mask[numTwo] = {0xffffffff, 0};  // mask hiddenDimHalf Elements
      LocalTensor<float> outLocal = outQueueY.AllocTensor<float>();
      Mul(outLocal, realLocal, cosLocal, mask, factor * numTwo, {1, 1, 1, 8, 4, 0});
      Mul(outLocal[hiddenDimHalf], imagLocal, cosLocal[hiddenDimHalf], mask, factor * numTwo, {1, 1, 1, 8, 4, 0});
      PipeBarrier<PIPE_V>();

      Muls<float>(imagLocal, imagLocal, -1.0f, factor * hiddenDimHalf);
      PipeBarrier<PIPE_V>();
      Mul(buf_, imagLocal, sinLocal, mask, factor * numTwo, {1, 1, 1, 8, 4, 0});
      Mul(buf_[hiddenDimHalf], realLocal, sinLocal[hiddenDimHalf], mask, factor * numTwo, {1, 1, 1, 8, 4, 0});
      PipeBarrier<PIPE_V>();

      Add(outLocal, outLocal, buf_, factor * hiddenDim);
      PipeBarrier<PIPE_V>();
      Cast(outLocal[hiddenDimCountPerLoop_ * hiddenDim].template ReinterpretCast<T>(), outLocal, RoundMode::CAST_RINT, factor * hiddenDim);
      PipeBarrier<PIPE_V>();

      outQueueY.EnQue(outLocal);
      outLocal = outQueueY.DeQue<float>();
      DataCopy(yGm[xOffset], outLocal[hiddenDimCountPerLoop_ * hiddenDim].template ReinterpretCast<T>(), factor * hiddenDim);
      outQueueY.FreeTensor(outLocal);
    }
    inQueueCos.FreeTensor(cosLocal);
    inQueueSin.FreeTensor(sinLocal);
  }

 private:
  TPipe* pipe_ = nullptr;
  const InterleaveRopeTilingData* tilingData_;
  GlobalTensor<T> xGm;
  GlobalTensor<T> yGm;
  GlobalTensor<T> cosGm;
  GlobalTensor<T> sinGm;

  TQue<QuePosition::VECIN, 1> inQueueX;
  TQue<QuePosition::VECIN, 1> inQueueCos;
  TQue<QuePosition::VECIN, 1> inQueueSin;
  TQue<QuePosition::VECOUT, 1> outQueueY;
  TBuf<TPosition::VECCALC> bufferReal;
  TBuf<TPosition::VECCALC> bufferImag;
  TBuf<TPosition::VECCALC> buffer_;
  int64_t numHead_ = 0;
  int64_t seqLength_ = 0;
  int64_t NS_ = 0;

  constexpr static int64_t hiddenDim = 64;
  constexpr static int64_t hiddenDimHalf = 32;
  constexpr static int64_t numTwo = 2;
  int64_t batchLoops_ = 0;
  int64_t batchPerLoop_ = 0;
  int64_t batchLastLoop_ = 0;
  int64_t batchsPerBlock_ = 0;
  int64_t curBlockBatchs_ = 0;
  int64_t hiddenDimLoops_ = 0;
  int64_t hiddenDimCountPerLoop_ = 0;
  int64_t hiddenDimCountLastLoop_ = 0;
  int64_t sinCosIdxOffsetBase_ = 0;
};
}  // namespace InterleaveRope

#endif  // _INTERLEAVE_ROPE_B11D_H_