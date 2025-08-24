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
 * \file apply_came_part4_pre.h
 * \brief
 */

#ifndef ASCENDC_APPLY_CAME_PART4_PRE_H
#define ASCENDC_APPLY_CAME_PART4_PRE_H

#include "kernel_operator.h"

using namespace AscendC;

template <typename T>
class ApplyCamePart4Pre {
 public:
  __aicore__ inline ApplyCamePart4Pre() {
  }
  __aicore__ inline void Init(GM_ADDR rIn, GM_ADDR workspace, const ApplyCamePart4TilingData* tilingData);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void ParseTilingData(const ApplyCamePart4TilingData* tilingData);
  __aicore__ inline void ComputSumR(uint64_t gmOffsets, uint64_t rNumPerLoop);
  __aicore__ inline void ComputSumRWithOnce(uint64_t gmOffsets, uint64_t rNumPerLoop);

 private:
  TPipe pipe_;
  TQue<QuePosition::VECIN, 1> rcInQue_;
  TBuf<QuePosition::VECCALC> castInRBuf_;
  TBuf<QuePosition::VECCALC> tmpBuf_;

  GlobalTensor<float> sumRWorkspace_;
  GlobalTensor<T> rInGm_;

  int64_t nShape_ = 0;
  int64_t totalCoreNum_ = 0;
  uint32_t MAX_ONCE_NUM = 0;
  uint32_t CAST_MAX_NUM = 0;

  static constexpr bool FLOAT16_SCENE = (sizeof(T) == sizeof(half));
  static constexpr int32_t TMPBUF_MAX_NUM = 2048;
};

template <typename T>
__aicore__ inline void ApplyCamePart4Pre<T>::Init(GM_ADDR rIn, GM_ADDR workspace,
                                                  const ApplyCamePart4TilingData* tilingData) {
  // init tiling data
  ParseTilingData(tilingData);
  rInGm_.SetGlobalBuffer((__gm__ T*)rIn);

  // buffer申请初始化
  MAX_ONCE_NUM = 128 * 1024 / sizeof(T);
  if constexpr (FLOAT16_SCENE) {
    MAX_ONCE_NUM = 56 * 1024 / sizeof(T);
    CAST_MAX_NUM = 112 * 1024 / sizeof(float);
    pipe_.InitBuffer(castInRBuf_, CAST_MAX_NUM * sizeof(float));
  }

  pipe_.InitBuffer(rcInQue_, 1, MAX_ONCE_NUM * sizeof(T));
  pipe_.InitBuffer(tmpBuf_, TMPBUF_MAX_NUM * sizeof(float));

  sumRWorkspace_.SetGlobalBuffer((__gm__ float*)workspace + (totalCoreNum_ * 32) / sizeof(float));
}

template <typename T>
__aicore__ inline void ApplyCamePart4Pre<T>::ParseTilingData(const ApplyCamePart4TilingData* tilingData) {
  nShape_ = tilingData->n;
  totalCoreNum_ = tilingData->totalCoreNum;
}

template <typename T>
__aicore__ inline void ApplyCamePart4Pre<T>::Process() {
  if (g_coreType == AIC) {
    return;
  }
  if (GetBlockIdx() == 0) {
    InitOutput<float>(sumRWorkspace_, 1, (float)0);
    uint64_t loop_time = (nShape_ + MAX_ONCE_NUM - 1) / MAX_ONCE_NUM;
    uint64_t pre_ele_num = (nShape_ + loop_time - 1) / loop_time;
    uint64_t last_ele_num = nShape_ - pre_ele_num * (loop_time - 1);
    uint64_t gmOffsets = 0;

    if (loop_time == 1) {
      ComputSumRWithOnce(gmOffsets, last_ele_num);
    } else {
      for (int64_t i = 0; i < loop_time - 1; i++) {
        gmOffsets = i * pre_ele_num;
        ComputSumR(gmOffsets, pre_ele_num);
      }
      ComputSumR((loop_time - 1) * pre_ele_num, last_ele_num);
    }
  }

  SyncAll();
}

template <typename T>
__aicore__ inline void ApplyCamePart4Pre<T>::ComputSumR(uint64_t gmOffsets, uint64_t rNumPerLoop) {
  LocalTensor<T> rcInUb = rcInQue_.AllocTensor<T>();
  DataCopyPad(rcInUb, rInGm_[gmOffsets], {1, (uint32_t)(rNumPerLoop * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
  rcInQue_.EnQue(rcInUb);
  rcInQue_.DeQue<T>();

  LocalTensor<float> workLocal = tmpBuf_.Get<float>();

  if constexpr (FLOAT16_SCENE) {
    LocalTensor<float> inputCast = castInRBuf_.Get<float>();
    Cast(inputCast, rcInUb, RoundMode::CAST_NONE, rNumPerLoop);
    PipeBarrier<PIPE_V>();

    ReduceSum(inputCast, inputCast, workLocal, rNumPerLoop);
    event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

    SetAtomicAdd<float>();
    DataCopyPad(sumRWorkspace_, inputCast, {1, (uint16_t)(1 * sizeof(float)), 0, 0});
    SetAtomicNone();
  } else {
    ReduceSum(rcInUb, rcInUb, workLocal, rNumPerLoop);
    event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

    SetAtomicAdd<float>();
    DataCopyPad(sumRWorkspace_, rcInUb, {1, (uint16_t)(1 * sizeof(float)), 0, 0});
    SetAtomicNone();
  }

  rcInQue_.FreeTensor(rcInUb);
}

template <typename T>
__aicore__ inline void ApplyCamePart4Pre<T>::ComputSumRWithOnce(uint64_t gmOffsets, uint64_t rNumPerLoop) {
  LocalTensor<T> rcInUbOnce = rcInQue_.AllocTensor<T>();
  DataCopyPad(rcInUbOnce, rInGm_[gmOffsets], {1, (uint32_t)(rNumPerLoop * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
  rcInQue_.EnQue(rcInUbOnce);
  rcInQue_.DeQue<T>();

  LocalTensor<float> workLocal = tmpBuf_.Get<float>();
  if constexpr (FLOAT16_SCENE) {
    LocalTensor<float> inputCast = castInRBuf_.Get<float>();
    Cast(inputCast, rcInUbOnce, RoundMode::CAST_NONE, rNumPerLoop);
    PipeBarrier<PIPE_V>();

    ReduceSum(inputCast, inputCast, workLocal, rNumPerLoop);
    event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

    DataCopyPad(sumRWorkspace_, inputCast, {1, (uint16_t)(1 * sizeof(float)), 0, 0});
  } else {
    ReduceSum(rcInUbOnce, rcInUbOnce, workLocal, rNumPerLoop);
    event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

    DataCopyPad(sumRWorkspace_, rcInUbOnce, {1, (uint16_t)(1 * sizeof(float)), 0, 0});
  }

  rcInQue_.FreeTensor(rcInUbOnce);
}
#endif