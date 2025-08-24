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
 * \file moe_tutel_dispatch.h
 * \brief
 */
#ifndef MOE_TUTEL_DISPATCH_
#define MOE_TUTEL_DISPATCH_

#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

template <typename T>
class MoeTutelDispatch {
 public:
  __aicore__ inline MoeTutelDispatch() {
  }
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR gates, GM_ADDR indices, GM_ADDR locations, GM_ADDR y,
                              MoeTutelDispatchTilingData* tilingData) {
    batchSize_ = tilingData->batchSize;
    samples_ = tilingData->samples;
    hidden_ = tilingData->hidden;
    capacity_ = tilingData->capacity;
    taskNum = tilingData->taskNum;
    taskNumPerBlock = tilingData->taskNumPerBlock;
    curBlockIdx = GetBlockIdx();
    startOffset = curBlockIdx * taskNumPerBlock;
    endOffset = (curBlockIdx + 1) * taskNumPerBlock;

    if (endOffset > taskNum) {
      endOffset = taskNum;
    }

    xGm.SetGlobalBuffer((__gm__ T*)x);
    gatesGm.SetGlobalBuffer((__gm__ T*)gates);
    indicesGm.SetGlobalBuffer((__gm__ int32_t*)indices);
    locationsGm.SetGlobalBuffer((__gm__ int32_t*)locations);

    yGm.SetGlobalBuffer((__gm__ T*)y);

    pipe.InitBuffer(inQueueX, BUFFER_NUM, hidden_ * sizeof(T));
    pipe.InitBuffer(inQueueGates, BUFFER_NUM, innerOffset * sizeof(T));
    pipe.InitBuffer(inQueueIndices, BUFFER_NUM, innerOffset * sizeof(int32_t));
    pipe.InitBuffer(inQueueLocations, BUFFER_NUM, innerOffset * sizeof(int32_t));

    pipe.InitBuffer(outQueueY, BUFFER_NUM, hidden_ * sizeof(T));
  }

  __aicore__ inline void Process() {
    Compute();
  }

 private:
  __aicore__ inline void Compute() {
    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    LocalTensor<T> gatesLocal = inQueueGates.AllocTensor<T>();
    LocalTensor<int32_t> indicesLocal = inQueueIndices.AllocTensor<int32_t>();
    LocalTensor<int32_t> locationsLocal = inQueueLocations.AllocTensor<int32_t>();

    LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

    for (int32_t taskOffset = startOffset; taskOffset < endOffset; taskOffset += innerOffset) {
      DataCopy(gatesLocal, gatesGm[taskOffset], innerOffset);
      DataCopy(indicesLocal, indicesGm[taskOffset], innerOffset);
      DataCopy(locationsLocal, locationsGm[taskOffset], innerOffset);

      int32_t innerTask = innerOffset;
      if (taskOffset + innerOffset > taskNum) {
        innerTask = taskNum - taskOffset;
      }

      for (int32_t i = 0; i < innerTask; ++i) {
        indices_ = indicesLocal.GetValue(i);
        locations_ = locationsLocal.GetValue(i);
        gates_ = gatesLocal.GetValue(i);
        pipe_barrier(PIPE_ALL);
        if (locations_ < capacity_ && indices_ >= 0) {
          if (static_cast<float>(gates_) != 0) {
            offset_y = indices_ * capacity_ + locations_;
            DataCopy(xLocal, xGm[((taskOffset + i) % samples_) * hidden_], hidden_);
            pipe_barrier(PIPE_ALL);
            DataCopy(yLocal, xLocal, hidden_);
            DataCopy(yGm[offset_y * hidden_], yLocal, hidden_);
          }
        }
      }
    }
    inQueueX.FreeTensor(xLocal);
    inQueueGates.FreeTensor(gatesLocal);
    inQueueIndices.FreeTensor(indicesLocal);
    inQueueLocations.FreeTensor(locationsLocal);

    outQueueY.FreeTensor(yLocal);
  }

 private:
  TPipe pipe;
  int32_t curBlockIdx = {0};
  int32_t startOffset = {0};
  int32_t endOffset = {0};
  int32_t innerOffset = {16};
  int32_t taskNum;
  int32_t taskNumPerBlock;
  int32_t batchSize_;
  int32_t samples_;
  int32_t capacity_;
  int32_t hidden_;
  int32_t indices_;
  int32_t locations_;
  int32_t offset_y;
  T gates_;

  GlobalTensor<T> xGm, gatesGm, yGm;
  GlobalTensor<int32_t> indicesGm, locationsGm;

  TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueGates, inQueueIndices, inQueueLocations;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
};
#endif  // MOE_TUTEL_DISPATCH
