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
 * \file interleave_rope_fixed_bnsd_b11d.h
 * \brief
 */
#ifndef _INTERLEAVE_ROPE_FIX_BNSD_B11D_H_
#define _INTERLEAVE_ROPE_FIX_BNSD_B11D_H_
#include "../inc/platform.h"

namespace InterleaveRope {
using namespace AscendC;

template <typename T>
class KernelInterleaveRopeFixBNSD {
 public:
  __aicore__ inline KernelInterleaveRopeFixBNSD(TPipe* pipe, const InterleaveRopeTilingData* tiling)
      : pipe_(pipe), tilingData_(tiling) {
  }

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y) {
    /*
     * original shape:
     * x:   [32, 32, 1, 64]
     * cos: [32, 1,  1, 64]
     * sin: [32, 1,  1, 64]
     * y:   [32, 32, 1, 64]
     * For each block, process
     * x:   [4, 32, 1, 64]
     * cos: [4, 1,  1, 64]
     * sin: [4, 1,  1, 64]
     * y:   [4, 32, 1, 64]
     */

    // init global memory
    xGm.SetGlobalBuffer((__gm__ T*)x + GetBlockIdx() * tilingData_->batchsPerBlock * numHead * hiddenDim, tilingData_->batchsPerBlock * numHead * hiddenDim);
    cosGm.SetGlobalBuffer((__gm__ T*)cos + GetBlockIdx() * tilingData_->batchsPerBlock * hiddenDim, tilingData_->batchsPerBlock * hiddenDim);
    sinGm.SetGlobalBuffer((__gm__ T*)sin + GetBlockIdx() * tilingData_->batchsPerBlock * hiddenDim, tilingData_->batchsPerBlock * hiddenDim);
    yGm.SetGlobalBuffer((__gm__ T*)y + GetBlockIdx() * tilingData_->batchsPerBlock * numHead * hiddenDim, tilingData_->batchsPerBlock * numHead * hiddenDim);
    
    // init pipe
    pipe_->InitBuffer(inQueueX, 1, tilingData_->batchsPerBlock * numHead * hiddenDim * sizeof(T));
    pipe_->InitBuffer(outQueueY, 1, tilingData_->batchsPerBlock * numHead * hiddenDim * sizeof(float) * numTwo);

    pipe_->InitBuffer(inQueueCos, 1, tilingData_->batchsPerBlock * hiddenDim * sizeof(float) * numTwo);
    pipe_->InitBuffer(inQueueSin, 1, tilingData_->batchsPerBlock * hiddenDim * sizeof(float) * numTwo);

    pipe_->InitBuffer(bufferReal, tilingData_->batchsPerBlock * numHead * hiddenDimHalf * sizeof(float) * numTwo);
    pipe_->InitBuffer(bufferImag, tilingData_->batchsPerBlock * numHead * hiddenDimHalf * sizeof(float) * numTwo);
    pipe_->InitBuffer(buffer_, tilingData_->batchsPerBlock * numHead * hiddenDim * sizeof(float) * numTwo);
  }

  __aicore__ inline void Process() {
    Rope();
  }

  __aicore__ inline void Rope() {
    // load
    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    DataCopy(xLocal, xGm, tilingData_->batchsPerBlock * numHead * hiddenDim);
    inQueueX.EnQue(xLocal);
    xLocal = inQueueX.DeQue<T>();

    // split Real and Imag
    LocalTensor<float> realLocal = bufferReal.Get<float>();
    LocalTensor<float> imagLocal = bufferImag.Get<float>();
    LocalTensor<float> buf_ = buffer_.Get<float>();
    uint64_t rsvdCnt = 0;
    GatherMask(realLocal[tilingData_->batchsPerBlock * numHead * hiddenDimHalf].template ReinterpretCast<T>(), xLocal, 1, true, tilingData_->batchsPerBlock * numHead * hiddenDim, {1, 1, 8, 0}, rsvdCnt);
    GatherMask(imagLocal[tilingData_->batchsPerBlock * numHead * hiddenDimHalf].template ReinterpretCast<T>(), xLocal, numTwo, true, tilingData_->batchsPerBlock * numHead * hiddenDim, {1, 1, 8, 0}, rsvdCnt);
    pipe_barrier(PIPE_V);
    Cast(realLocal, realLocal[tilingData_->batchsPerBlock * numHead * hiddenDimHalf].template ReinterpretCast<T>(), RoundMode::CAST_NONE, tilingData_->batchsPerBlock * numHead * hiddenDimHalf);
    Cast(imagLocal, imagLocal[tilingData_->batchsPerBlock * numHead * hiddenDimHalf].template ReinterpretCast<T>(), RoundMode::CAST_NONE, tilingData_->batchsPerBlock * numHead * hiddenDimHalf);
    
    inQueueX.FreeTensor(xLocal);

    // rope
    LocalTensor<float> cosLocal = inQueueCos.AllocTensor<float>();
    DataCopy(cosLocal[tilingData_->batchsPerBlock * hiddenDim].template ReinterpretCast<T>(), cosGm, tilingData_->batchsPerBlock * hiddenDim);

    inQueueCos.EnQue(cosLocal);
    cosLocal = inQueueCos.DeQue<float>();
    
    Cast(cosLocal, cosLocal[tilingData_->batchsPerBlock * hiddenDim].template ReinterpretCast<T>(), RoundMode::CAST_NONE, tilingData_->batchsPerBlock * hiddenDim);
    pipe_barrier(PIPE_V);
    uint64_t mask[numTwo] = { 0xffffffff, 0};
    LocalTensor<float> outLocal = outQueueY.AllocTensor<float>();
    for (int64_t i = 0; i < tilingData_->batchsPerBlock; i++) {
      Mul(outLocal[i * numHead * hiddenDim], realLocal[i * numHead * hiddenDimHalf], cosLocal[i * hiddenDim], mask, hiddenDim, {1, 1, 1, 8, 4, 0});
      Mul(outLocal[i * numHead * hiddenDim + numHead], imagLocal[i * numHead * hiddenDimHalf], cosLocal[i * hiddenDim + hiddenDimHalf], mask, hiddenDim, {1, 1, 1, 8, 4, 0});
    }
    pipe_barrier(PIPE_V);
    inQueueCos.FreeTensor(cosLocal);

    LocalTensor<float> sinLocal = inQueueSin.AllocTensor<float>();

    DataCopy(sinLocal[tilingData_->batchsPerBlock * hiddenDim].template ReinterpretCast<T>(), sinGm, tilingData_->batchsPerBlock * hiddenDim);

    inQueueSin.EnQue(sinLocal);
    sinLocal = inQueueSin.DeQue<float>();
    Cast(sinLocal, sinLocal[tilingData_->batchsPerBlock * hiddenDim].template ReinterpretCast<T>(), RoundMode::CAST_NONE, tilingData_->batchsPerBlock * hiddenDim);
    pipe_barrier(PIPE_V);
    Muls<float>(imagLocal, imagLocal, -1.0f, tilingData_->batchsPerBlock * numHead * hiddenDimHalf);
    pipe_barrier(PIPE_V);
    for(int64_t i = 0; i < tilingData_->batchsPerBlock; i++) {
      Mul(buf_[i * numHead * hiddenDim], imagLocal[i * numHead * hiddenDimHalf], sinLocal[i * hiddenDim], mask, hiddenDim, {1, 1, 1, 8, 4, 0});
      Mul(buf_[i * numHead * hiddenDim + hiddenDimHalf], realLocal[i * numHead * hiddenDimHalf], sinLocal[i * hiddenDim + hiddenDimHalf], mask, hiddenDim, {1, 1, 1, 8, 4, 0});
    }
    pipe_barrier(PIPE_V);
    inQueueSin.FreeTensor(sinLocal);
    Add(outLocal, outLocal, buf_, tilingData_->batchsPerBlock * numHead * hiddenDim);
    pipe_barrier(PIPE_V);
    Cast(outLocal[tilingData_->batchsPerBlock * numHead * hiddenDim].template ReinterpretCast<T>(), outLocal, RoundMode::CAST_RINT, tilingData_->batchsPerBlock * numHead * hiddenDim);
    outQueueY.EnQue(outLocal);
    outLocal = outQueueY.DeQue<float>();
    DataCopy(yGm, outLocal[tilingData_->batchsPerBlock * numHead * hiddenDim].template ReinterpretCast<T>(), tilingData_->batchsPerBlock * numHead * hiddenDim);
    outQueueY.FreeTensor(outLocal);
  }

 private:
  TPipe* pipe_ = nullptr;
  const InterleaveRopeTilingData* tilingData_;
  GlobalTensor<T> xGm, yGm, cosGm, sinGm;

  TQue<QuePosition::VECIN, 1> inQueueX, inQueueCos, inQueueSin;
  TQue<QuePosition::VECOUT, 1> outQueueY;
  TBuf<TPosition::VECCALC> bufferReal, bufferImag, buffer_;

  constexpr static int64_t hiddenDim = 64;
  constexpr static int64_t hiddenDimHalf = 32;
  constexpr static int64_t numHead = 32;
  constexpr static int64_t seqLength = 1;
  constexpr static int64_t numTwo = 2;
};
}  // namespace InterleaveRope

#endif  // _INTERLEAVE_ROPE_FIX_BNSD_B11D_H_