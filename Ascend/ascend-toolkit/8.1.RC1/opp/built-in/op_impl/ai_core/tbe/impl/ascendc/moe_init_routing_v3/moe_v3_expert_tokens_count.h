/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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
 * \file moe_v3_expert_tokens_count.h
 * \brief
 */
#ifndef MOE_V3_EXPERT_TOKENS_COUNT_H
#define MOE_V3_EXPERT_TOKENS_COUNT_H

#include "moe_v3_common.h"
#include "kernel_operator.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

class ExpertTokensCount {
 public:
  __aicore__ inline ExpertTokensCount(){};
  __aicore__ inline void Init(GM_ADDR expandedRowIdx, GM_ADDR expertTokensCount, GM_ADDR workspace,
                              const MoeInitRoutingV3TilingData* tilingData, TPipe* tPipe);
  __aicore__ inline void Process();

 private:
  __aicore__ inline void CopyIn(int64_t loop, int64_t curLoopElements);
  __aicore__ inline void Compute(int64_t curLoopElements);
  __aicore__ inline void CopyOut();

  __aicore__ inline void expertCountCopyIn();
  __aicore__ inline void expertCountCompute();
  __aicore__ inline void expertCountCopyOut();

 private:
  GlobalTensor<int32_t> sortedexpertIdxGm_;
  GlobalTensor<int32_t> expertCountTempGm_;
  GlobalTensor<int64_t> expertTokensCountGm_;
  GlobalTensor<int32_t> expertTotalCountGm_;
  GlobalTensor<int32_t> expandedRowIdxGm_;
  TPipe* pipe_;

  TQue<QuePosition::VECIN, 1> sortedExpertIdxInQueue_;
  TQue<QuePosition::VECOUT, 1> expertCountOutToTempQueue_;
  TQue<QuePosition::VECIN, 1> expertCountTempInQueue_;
  TQue<QuePosition::VECOUT, 1> expertIdxCountOutQueue_;
  TQue<QuePosition::VECOUT, 1> expertTotalCountQueue_;

  const MoeV3ExpertTokensCountTilingData* expertTokensCountTilingData_;
  int64_t blockIdx_;
  int64_t needCoreNum_;
  int64_t perCoreElements_;
  int64_t curCoreElements_ = 0;
  int64_t expertStart_ = 0;
  int64_t expertEnd_ = 0;
  int64_t actualExpertNum_ = 0;
  int64_t coreLoopsNum_ = 0;
  int64_t perCorePerLoopElements_ = 0;
  int64_t perCoreLastLoopElements_ = 0;
  int64_t actualExpertTotalNum_ = 0;
};

__aicore__ inline void ExpertTokensCount::Init(GM_ADDR expandedRowIdx, GM_ADDR expertTokensCount, GM_ADDR workspace,
                                               const MoeInitRoutingV3TilingData* tilingData, TPipe* tPipe) {
  pipe_ = tPipe;
  expertTokensCountTilingData_ = &(tilingData->expertTokensCountTilingDataOp);
  blockIdx_ = GetBlockIdx();
  needCoreNum_ = expertTokensCountTilingData_->needCoreNum;
  perCoreElements_ = expertTokensCountTilingData_->perCoreElements;
  expertStart_ = tilingData->expertStart;
  expertEnd_ = tilingData->expertEnd;
  actualExpertNum_ = tilingData->actualExpertNum;

  if (blockIdx_ < needCoreNum_ - 1) {
    curCoreElements_ = expertTokensCountTilingData_->perCoreElements;
    coreLoopsNum_ = expertTokensCountTilingData_->perCoreLoops;
    perCorePerLoopElements_ = expertTokensCountTilingData_->perCorePerLoopElements;
    perCoreLastLoopElements_ = expertTokensCountTilingData_->perCoreLastLoopElements;
  } else if (blockIdx_ == needCoreNum_ - 1) {
    curCoreElements_ = expertTokensCountTilingData_->lastCoreElements;
    coreLoopsNum_ = expertTokensCountTilingData_->lastCoreLoops;
    perCorePerLoopElements_ = expertTokensCountTilingData_->lastCorePerLoopElements;
    perCoreLastLoopElements_ = expertTokensCountTilingData_->lastCoreLastLoopElements;
  }
  sortedexpertIdxGm_.SetGlobalBuffer((__gm__ int32_t*)workspace + blockIdx_ * perCoreElements_, curCoreElements_);
  expertTokensCountGm_.SetGlobalBuffer((__gm__ int64_t*)expertTokensCount, actualExpertNum_);
  expertCountTempGm_.SetGlobalBuffer(
      (__gm__ int32_t*)workspace + Align(tilingData->n * tilingData->k, sizeof(int32_t)) * 2, actualExpertNum_);
  expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t*)workspace +
                                          Align(tilingData->n * tilingData->k, sizeof(int32_t)) * 2 +
                                          Align(actualExpertNum_, sizeof(int32_t)),
                                      actualExpertNum_);

  expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t*)expandedRowIdx + blockIdx_ * perCoreElements_);
  if ((tilingData->rowIdxType == GATHER) && (blockIdx_ < needCoreNum_)) {
    InitGlobalMemory(expandedRowIdxGm_, curCoreElements_, -1);
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
  }

  int64_t sortedExpertIdxInLen = Max(perCorePerLoopElements_, perCoreLastLoopElements_);
  pipe_->InitBuffer(sortedExpertIdxInQueue_, 1, AlignBytes(sortedExpertIdxInLen, sizeof(int32_t)));
  pipe_->InitBuffer(expertCountOutToTempQueue_, 1, AlignBytes(actualExpertNum_, sizeof(int32_t)));
  pipe_->InitBuffer(expertCountTempInQueue_, 1, AlignBytes(actualExpertNum_, sizeof(int32_t)));
  pipe_->InitBuffer(expertIdxCountOutQueue_, 1, AlignBytes(actualExpertNum_, sizeof(int64_t)));
  pipe_->InitBuffer(expertTotalCountQueue_, 1, AlignBytes(1, sizeof(int32_t)));
}

__aicore__ inline void ExpertTokensCount::Process() {
  if (blockIdx_ < needCoreNum_) {
    for (int64_t i = 0; i < coreLoopsNum_; i++) {
      int64_t perLoopElements = (i == (coreLoopsNum_ - 1)) ? perCoreLastLoopElements_ : perCorePerLoopElements_;
      CopyIn(i, perLoopElements);
      Compute(perLoopElements);
      CopyOut();
    }
  }

  SyncAll();
  /* copy expert tokens count result from worksapce to output GM. */
  if (blockIdx_ == 0) {
    expertCountCopyIn();
    expertCountCompute();
    expertCountCopyOut();
  }
  SyncAll();
}

__aicore__ inline void ExpertTokensCount::CopyIn(int64_t loop, int64_t curLoopElements) {
  LocalTensor<int32_t> sortedExpertIdxInLocal = sortedExpertIdxInQueue_.AllocTensor<int32_t>();
  DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(curLoopElements * sizeof(int32_t)),
                                   0, 0, 0};
  DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
  int64_t sortedexpertIdxOffset = loop * perCorePerLoopElements_;
  DataCopyPad(sortedExpertIdxInLocal, sortedexpertIdxGm_[sortedexpertIdxOffset], dataCopyParams, dataCopyPadParams);
  sortedExpertIdxInQueue_.EnQue(sortedExpertIdxInLocal);
}

__aicore__ inline void ExpertTokensCount::Compute(int64_t curLoopElements) {
  LocalTensor<int32_t> sortedExpertIdxInLocal = sortedExpertIdxInQueue_.DeQue<int32_t>();
  LocalTensor<int32_t> expertCountOutLocal = expertCountOutToTempQueue_.AllocTensor<int32_t>();
  Duplicate(expertCountOutLocal.ReinterpretCast<int32_t>(), static_cast<int32_t>(0),
            static_cast<int32_t>(actualExpertNum_));
  event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
  SetFlag<HardEvent::V_S>(eventIDVToS);
  WaitFlag<HardEvent::V_S>(eventIDVToS);
  int64_t i = 0;
  int32_t lastExpertId = sortedExpertIdxInLocal.GetValue(0);
  int32_t lastIndex = 0;
  for (i = 1; i < curLoopElements; i++) {
    if ((lastExpertId >= expertEnd_) || (lastExpertId < expertStart_)) {
      break;
    }
    int32_t curExpertId = sortedExpertIdxInLocal.GetValue(i);
    if (curExpertId != lastExpertId || curExpertId >= expertEnd_) {
      expertCountOutLocal.SetValue(lastExpertId - expertStart_, i - lastIndex);
      lastIndex = i;
      lastExpertId = curExpertId;
    }
  }
  if ((i == curLoopElements) && ((lastExpertId >= expertStart_) && (lastExpertId < expertEnd_))) {
    expertCountOutLocal.SetValue(lastExpertId - expertStart_, i - lastIndex);
  }

  event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
  SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
  WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
  expertCountOutToTempQueue_.EnQue<int32_t>(expertCountOutLocal);
  sortedExpertIdxInQueue_.FreeTensor(sortedExpertIdxInLocal);
}

__aicore__ inline void ExpertTokensCount::CopyOut() {
  LocalTensor<int32_t> expertCountOutLocal = expertCountOutToTempQueue_.DeQue<int32_t>();

  DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>((actualExpertNum_) * sizeof(int32_t)), 0,
                               0, 0};
  SetAtomicAdd<int32_t>();
  DataCopyPad(expertCountTempGm_, expertCountOutLocal, copyParams);
  SetAtomicNone();
  expertCountOutToTempQueue_.FreeTensor(expertCountOutLocal);
}

__aicore__ inline void ExpertTokensCount::expertCountCopyIn() {
  LocalTensor<int32_t> expertCountTempInLocal = expertCountTempInQueue_.AllocTensor<int32_t>();

  DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1),
                                   static_cast<uint32_t>((actualExpertNum_) * sizeof(int32_t)), 0, 0, 0};
  DataCopyPadExtParams dataCopyPadParams{false, 0, 0, 0};
  DataCopyPad(expertCountTempInLocal, expertCountTempGm_, dataCopyParams, dataCopyPadParams);
  expertCountTempInQueue_.EnQue(expertCountTempInLocal);
}

__aicore__ inline void ExpertTokensCount::expertCountCompute() {
  LocalTensor<int32_t> expertCountTempInLocal = expertCountTempInQueue_.DeQue<int32_t>();
  LocalTensor<int64_t> expertCountOutLocal = expertIdxCountOutQueue_.AllocTensor<int64_t>();
  LocalTensor<int32_t> expertTotalCountLocal = expertTotalCountQueue_.AllocTensor<int32_t>();
  event_t eventIDMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
  SetFlag<HardEvent::MTE2_S>(eventIDMte2ToS);
  WaitFlag<HardEvent::MTE2_S>(eventIDMte2ToS);
  for (int64_t i = 0; i < actualExpertNum_; i++) {
    int64_t expertCount = static_cast<int64_t>(expertCountTempInLocal.GetValue(i));
    expertCountOutLocal.SetValue(i, expertCount);
    actualExpertTotalNum_ += expertCount;
  }
  expertTotalCountLocal.SetValue(0, static_cast<int32_t>(actualExpertTotalNum_));
  event_t eventIDSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
  SetFlag<HardEvent::S_MTE3>(eventIDSToMte3);
  WaitFlag<HardEvent::S_MTE3>(eventIDSToMte3);
  expertIdxCountOutQueue_.EnQue<int64_t>(expertCountOutLocal);
  expertTotalCountQueue_.EnQue<int32_t>(expertTotalCountLocal);
  expertCountTempInQueue_.FreeTensor(expertCountTempInLocal);
}

__aicore__ inline void ExpertTokensCount::expertCountCopyOut() {
  LocalTensor<int64_t> expertCountOutLocal = expertIdxCountOutQueue_.DeQue<int64_t>();
  LocalTensor<int32_t> expertTotalCountLocal = expertTotalCountQueue_.DeQue<int32_t>();
  DataCopyExtParams copyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(actualExpertNum_ * sizeof(int64_t)), 0,
                               0, 0};
  DataCopyPad(expertTokensCountGm_, expertCountOutLocal, copyParams);
  copyParams.blockLen = sizeof(int32_t);
  DataCopyPad(expertTotalCountGm_, expertTotalCountLocal, copyParams);
  expertIdxCountOutQueue_.FreeTensor(expertCountOutLocal);
  expertTotalCountQueue_.FreeTensor(expertTotalCountLocal);
}

}  // namespace MoeInitRoutingV3
#endif  // MOE_V3_EXPERT_TOKENS_COUNT_H