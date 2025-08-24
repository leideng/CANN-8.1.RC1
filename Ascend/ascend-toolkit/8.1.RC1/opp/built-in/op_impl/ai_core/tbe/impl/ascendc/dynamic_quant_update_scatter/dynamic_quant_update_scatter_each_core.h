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
 * \file dynamic_quant_update_scatter_each_core.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_UPDATE_SCATTER_EACH_CORE_H
#define DYNAMIC_QUANT_UPDATE_SCATTER_EACH_CORE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "dynamic_quant_update_scatter_comm.h"

namespace DynamicQuantUpdateScatterND {
using namespace AscendC;

template <typename T1, typename T2, typename T3>
class DynamicQuantUpdateScatterEachCore : DynamicQuantUpdateScatterBase<T1, T2, T3> {
 public:
  __aicore__ inline DynamicQuantUpdateScatterEachCore() {
  }

  __aicore__ inline void Init(GM_ADDR var, GM_ADDR varScale, GM_ADDR indices, GM_ADDR updates, GM_ADDR smoothScales,
                              const DynamicQuantUpdateScatterTilingData* tilingPtr) {
    int64_t eachCoreBsNumElements = tilingPtr->srcBsStride * tilingPtr->eachCoreBsNum;
    this->InitBase(var, varScale, indices, smoothScales, tilingPtr);
    int64_t varScaleOutAlignElemens =
        (this->varScaleOutElemens * sizeof(float) + THIRTY_TWO) / THIRTY_TWO * THIRTY_TWO / sizeof(float);
    this->InitUpdateGM(updates, this->blockIdx * eachCoreBsNumElements, this->coreBsNumElements);
    this->InitUpdatesInQueue(this->coreBsNumElements * sizeof(T3));
    this->InitSmoothScalesInQueue(tilingPtr->varOrigLastDimSize * sizeof(T3),
                                  tilingPtr->varOrigLastDimSize * sizeof(float));
    this->InitVarOutQueue(this->coreBsNumElements * sizeof(T1));
    this->InitvarScaleOutQueue(varScaleOutAlignElemens * sizeof(float));
    this->InitTempF32(this->varScaleOutElemens * tilingPtr->varOrigLastDimSize * sizeof(float));
    this->InitTempI32(tilingPtr->varOrigLastDimSize * sizeof(int32_t));

#if defined(ORIG_DTYPE_UPDATES) && (ORIG_DTYPE_UPDATES == DT_FLOAT16 || ORIG_DTYPE_UPDATES == DT_BF16)
    this->InitUpdateF32(this->coreBsNumElements * sizeof(float));
#endif
  }

  __aicore__ inline void Process(const DynamicQuantUpdateScatterTilingData* tilingPtr) {
    if (this->blockIdx < tilingPtr->coreNum) {
      LoopProcess(tilingPtr);
    }
  }

 private:
  __aicore__ inline void LoopProcess(const DynamicQuantUpdateScatterTilingData* tilingPtr) {
    CopyIn(tilingPtr);
    Compute(tilingPtr);
    this->CopyOutByBatchs(this->coreBatchNum, tilingPtr);
  }

  __aicore__ inline void CopyIn(const DynamicQuantUpdateScatterTilingData* tilingPtr) {
    this->CopyIndicesIn(tilingPtr);
    this->CopySmoothScalesIn(0, tilingPtr->varOrigLastDimSize);
    this->CopyUpdatesIn(0, this->coreBsNumElements);
  }

  __aicore__ inline void Compute(const DynamicQuantUpdateScatterTilingData* tilingPtr) {
    LocalTensor<T3> updatesLocal = this->updatesInQueue.template DeQue<T3>();
    LocalTensor<T3> smoothScalesLocal;
    LocalTensor<float> tempUpdatesCast;
    LocalTensor<float> tempsmoothScalesCast;

#if defined(ORIG_DTYPE_UPDATES) && (ORIG_DTYPE_UPDATES == DT_FLOAT16 || ORIG_DTYPE_UPDATES == DT_BF16)
    tempUpdatesCast = this->updateF32.template Get<float>();
    Cast(tempUpdatesCast, updatesLocal, RoundMode::CAST_NONE, this->coreBsNumElements);
    if (this->hasSmoothScales) {
      smoothScalesLocal = this->smoothScalesInQueue.template DeQue<T3>();
      tempsmoothScalesCast = this->smoothScaleF32.template Get<float>();
      Cast(tempsmoothScalesCast, smoothScalesLocal[0], RoundMode::CAST_NONE, tilingPtr->varOrigLastDimSize);
    }
    this->ComputeQuant(tempUpdatesCast, tempsmoothScalesCast, tilingPtr, this->varScaleOutElemens);
#else
    if (this->hasSmoothScales) {
      smoothScalesLocal = this->smoothScalesInQueue.template DeQue<T3>();
    }
    this->ComputeQuant(updatesLocal, smoothScalesLocal, tilingPtr, this->varScaleOutElemens);
#endif

    this->updatesInQueue.FreeTensor(updatesLocal);
    if (this->hasSmoothScales) {
      this->smoothScalesInQueue.FreeTensor(smoothScalesLocal);
    }

    return;
  }

 private:
};
}  // namespace DynamicQuantUpdateScatterND
#endif  // DYNAMIC_QUANT
