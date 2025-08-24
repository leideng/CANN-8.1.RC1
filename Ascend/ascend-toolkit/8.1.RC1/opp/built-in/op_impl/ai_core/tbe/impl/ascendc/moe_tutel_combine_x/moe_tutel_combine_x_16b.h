/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
 * \file moe_tutel_combine_x_16b.h
 * \brief
 */
#ifndef _MOE_TUTEL_COMBINE_X_16B_H_
#define _MOE_TUTEL_COMBINE_X_16B_H_
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
using namespace AscendC;

template <typename T>
class MoeTutelCombineX {
 public:
  __aicore__ inline MoeTutelCombineX() {
  }
  __aicore__ inline void Init(GM_ADDR y_grad, GM_ADDR gates, GM_ADDR indices, GM_ADDR locations, GM_ADDR x_grad,
                              MoeTutelCombineXTilingData* tilingData) {
    samples_ = tilingData->samples;
    hidden_ = tilingData->hidden;
    capacity_ = tilingData->capacity;
    batchSize = tilingData->batchSize;
    taskNumberBlock = tilingData->taskNumPerBlock;
    taskNum = samples_;
    curBlockIdx = GetBlockIdx();
    startOffset = curBlockIdx * taskNumberBlock;
    endOffset = (curBlockIdx + 1) * taskNumberBlock;

    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x_grad), samples_ * hidden_);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y_grad), batchSize * capacity_ * hidden_);

    indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices), batchSize * samples_);
    locationsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(locations), batchSize * samples_);

    gatesGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gates), batchSize * hidden_);

    pipe.InitBuffer(inQueueX, 1, hidden_ * sizeof(T));

    pipe.InitBuffer(inQueueGates, 1, innerOffset * sizeof(T));
    pipe.InitBuffer(inQueueY, 1, hidden_ * sizeof(T));
    pipe.InitBuffer(inQueueIndices, 1, innerOffset * sizeof(int32_t));
    pipe.InitBuffer(inQueueLocations, 1, innerOffset * sizeof(int32_t));

    pipe.InitBuffer(mulQueue, 1, hidden_ * sizeof(T));

    pipe.InitBuffer(tmpXCastUb, hidden_ * sizeof(float));
    pipe.InitBuffer(tmpYCastUb, hidden_ * sizeof(float));
    pipe.InitBuffer(tmpGateCastUb, innerOffset * sizeof(float));
    pipe.InitBuffer(tmpZCastUb, hidden_ * sizeof(float));
  }

  __aicore__ inline void Process() {
    if (endOffset > taskNum) {
      endOffset = taskNum;
    }
    for (int32_t batch_id = 0; batch_id < batchSize; batch_id++) {
      int32_t base_offset = batch_id * samples_;
      for (int32_t taskIdx = startOffset; taskIdx < endOffset; taskIdx += innerOffset) {
        Compute(taskIdx + base_offset, taskIdx);
      }
    }
  }

 private:
  __aicore__ inline void Compute(int32_t taskOffset, int32_t i_offset) {
    LocalTensor<int32_t> indicesLocal = inQueueIndices.AllocTensor<int32_t>();
    LocalTensor<int32_t> locationsLocal = inQueueLocations.AllocTensor<int32_t>();
    LocalTensor<T> gatesLocal = inQueueGates.AllocTensor<T>();
    pipe_barrier(PIPE_ALL);
    DataCopy(indicesLocal, indicesGm[taskOffset], innerOffset);
    DataCopy(locationsLocal, locationsGm[taskOffset], innerOffset);
    DataCopy(gatesLocal, gatesGm[taskOffset], innerOffset);
    pipe_barrier(PIPE_ALL);

    LocalTensor<float> gateFloatLocal = tmpGateCastUb.Get<float>();
    Cast(gateFloatLocal, gatesLocal, RoundMode::CAST_NONE, innerOffset);
    pipe_barrier(PIPE_ALL);

    int32_t inner_task = innerOffset;
    if (i_offset + innerOffset > taskNum) {
      inner_task = taskNum - i_offset;
    }

    for (int32_t i = 0; i < inner_task; i++) {
      LocalTensor<float> zTmpLocal = tmpZCastUb.Get<float>();
      LocalTensor<T> mulLocal = mulQueue.AllocTensor<T>();
      LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
      LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();

      LocalTensor<float> xFloatLocal = tmpXCastUb.Get<float>();
      LocalTensor<float> yFloatLocal = tmpYCastUb.Get<float>();

      idx0 = indicesLocal.GetValue(i);
      idx1 = locationsLocal.GetValue(i);
      gates_ = gateFloatLocal.GetValue(i);
      pipe_barrier(PIPE_ALL);
      if (idx0 >= 0 && idx1 < capacity_) {
        offset = idx0 * capacity_ + idx1;
        DataCopy(xLocal, xGm[(i_offset + i) * hidden_], hidden_);
        DataCopy(yLocal, yGm[offset * hidden_], hidden_);
        pipe_barrier(PIPE_ALL);

        Cast(xFloatLocal, xLocal, RoundMode::CAST_NONE, hidden_);
        Cast(yFloatLocal, yLocal, RoundMode::CAST_NONE, hidden_);
        pipe_barrier(PIPE_ALL);
        Muls(zTmpLocal, yFloatLocal, gates_, hidden_);
        pipe_barrier(PIPE_ALL);
        Add(zTmpLocal, zTmpLocal, xFloatLocal, hidden_);
        pipe_barrier(PIPE_ALL);
        Cast(mulLocal, zTmpLocal, RoundMode::CAST_ROUND, hidden_);
        mulQueue.EnQue(mulLocal);
        CopyOut((i_offset + i) * hidden_);
      }
      mulQueue.FreeTensor(mulLocal);
      inQueueX.FreeTensor(xLocal);
      inQueueY.FreeTensor(yLocal);
    }
    inQueueIndices.FreeTensor(indicesLocal);
    inQueueLocations.FreeTensor(locationsLocal);
    inQueueGates.FreeTensor(gatesLocal);
  }

  __aicore__ inline void CopyOut(int32_t taskOffset) {
    LocalTensor<T> mulLocal = mulQueue.DeQue<T>();
    DataCopy(xGm[taskOffset], mulLocal, hidden_);
  }

 private:
  TPipe pipe;
  int32_t blockNum = {0};
  int32_t batchSize = {0};
  int32_t curBlockIdx = {0};
  int32_t taskNumberBlock = {0};
  int32_t startOffset = {0};
  int32_t endOffset = {0};
  int32_t innerOffset = {16};
  int32_t samples_;
  int32_t capacity_;
  int32_t hidden_;
  int32_t taskNum;
  float gates_;
  int32_t idx0, idx1, offset;

  GlobalTensor<T> xGm, yGm, gatesGm;
  GlobalTensor<int32_t> indicesGm, locationsGm;

  TQue<QuePosition::VECIN, 1> inQueueGates, inQueueY, inQueueIndices, inQueueLocations, inQueueX;
  TQue<QuePosition::VECOUT, 1> mulQueue;
  TBuf<> tmpXCastUb, tmpYCastUb, tmpGateCastUb, tmpZCastUb;
};
#endif  // _MOE_TUTEL_COMBINE_X_16B_H_