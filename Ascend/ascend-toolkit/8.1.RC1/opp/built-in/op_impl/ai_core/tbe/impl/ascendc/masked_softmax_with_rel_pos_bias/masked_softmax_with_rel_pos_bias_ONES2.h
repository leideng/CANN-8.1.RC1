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
 * \file masked_softmax_with_rel_pos_bias_ONES2.h
 * \brief
 */
#ifndef ASCENDC_MASKED_SOFTMAX_WITH_RELPOSIBIAS_ONES2_H
#define ASCENDC_MASKED_SOFTMAX_WITH_RELPOSIBIAS_ONES2_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace MaskedSoftmaxWithRelPosBias {

using namespace AscendC;

template <typename T>
class MaskedSoftmaxWithRelPosBiasONES2 {
public:
  __aicore__ inline MaskedSoftmaxWithRelPosBiasONES2(
      const MaskedSoftmaxWithRelPosBiasTilingData* __restrict__ MaskedSoftmaxWithRelPosBiasONES2TilingData)
          : tilingData(MaskedSoftmaxWithRelPosBiasONES2TilingData){};
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR attenMask, GM_ADDR bias, GM_ADDR y) {
    if (GetBlockIdx() < tilingData->tailStartCoreIdx) {
      offset = GetBlockIdx() * tilingData->singleCoreSize;
      batchSize = tilingData->singleCoreSize;
    } else {
      offset = GetBlockIdx() * (tilingData->singleCoreSize - 1) + tilingData->tailStartCoreIdx;
      batchSize = tilingData->singleCoreSize - 1;
    }

    typeSize = sizeof(T);
    stackNum = tilingData->stackNum;  // 65535 is for DataCopyParams
    loopCount = batchSize / stackNum;
    loopTailSize = batchSize - batchSize / stackNum * stackNum;

    yGm.SetGlobalBuffer((__gm__ T*)y + offset, batchSize);
    pipe.InitBuffer(vecOutQueue, 1, (stackNum * typeSize / 32 + 1) * 32); // 32 is for aligned
  }

  __aicore__ inline void Process() {
    for (uint32_t i = 0; i < loopCount; i++) {
      Compute(i, stackNum);
      CopyOut(i, stackNum);
    }
    if (loopTailSize > 0) {
      Compute(loopCount, loopTailSize);
      CopyOut(loopCount, loopTailSize);
    }
  }

private:
  __aicore__ inline void Compute(int32_t i, uint32_t num) {
    LocalTensor<T> yLocal = vecOutQueue.AllocTensor<T>();

    T scalar = 1.0;
    Duplicate<T>(yLocal, scalar, num);

    vecOutQueue.EnQue<T>(yLocal);
  }

  __aicore__ inline void CopyOut(int32_t i, uint32_t num) {
    LocalTensor<T> yLocal = vecOutQueue.DeQue<T>();
    DataCopyParams copyParamsLast{(uint16_t)(1), (uint16_t)(num * typeSize), (uint16_t)(0), (uint16_t)(0)};
#if __CCE_AICORE__ == 220
    DataCopyPad(yGm[i * stackNum], yLocal, copyParamsLast);
#endif
    vecOutQueue.FreeTensor(yLocal);
  }

private:
  TPipe pipe;
  TQue<QuePosition::VECOUT, 1> vecOutQueue;
  uint32_t offset, batchSize;
  uint32_t stackNum, loopCount, loopTailSize, typeSize;
  GlobalTensor<T> yGm;
  const MaskedSoftmaxWithRelPosBiasTilingData* __restrict__ tilingData;
};

}
#endif
