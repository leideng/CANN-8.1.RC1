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
 * \file moe_v3_sort_base.h
 * \brief
 */
#ifndef MOE_V3_SORT_BASE_H
#define MOE_V3_SORT_BASE_H

#include "kernel_operator.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

class MoeSortBase {
 public:
  __aicore__ inline MoeSortBase(){};
  __aicore__ inline int64_t GetSyncRound();

 protected:
  __aicore__ inline void CleanWSCache();
  __aicore__ inline void SyncAll();

 protected:
  TPipe* pipe;
  TQue<QuePosition::VECIN, 1> sortDataCopyInQueue;
  TQue<QuePosition::VECOUT, 1> sortDataCopyOutQueue;
  TBuf<TPosition::VECCALC> tempBuffer;
  TBuf<TPosition::VECCALC> sortedBuffer;

  GlobalTensor<int32_t> expertIdxGm;
  GlobalTensor<int32_t> expendedRowIdxGm;
  GlobalTensor<int32_t> sortedExpertForSourceRowGm;
  GlobalTensor<int32_t> expandDstToSrcRowGm;
  GlobalTensor<int32_t> sortedexpertIdxGm;
  GlobalTensor<int32_t> expertCountTempGm;

  int64_t tileLength;
  int64_t bufferNum = 1;
  int64_t totalLength;
  int64_t coreNum;

  int64_t expertStart_ = 0;
  int64_t expertEnd_ = 0;
  int64_t n;
  int64_t k;
  int64_t rowIdxType_ = 0;

  static constexpr int64_t SYNC_GM_NUM = 2;
  static constexpr int64_t WORK_GM_NUM = 2;
  static constexpr int64_t DST_BLK_STRIDE = 1;
  static constexpr int64_t DST_REP_STRIDE = 8;
};

__aicore__ inline void MoeSortBase::SyncAll() {
  AscendC::SyncAll();
}

}  // namespace MoeInitRoutingV3
#endif  // MOE_V3_SORT_BASE_H