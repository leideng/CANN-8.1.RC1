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
 * \file batch_norm_v3_base.h
 * \brief
 */

#ifndef BATCH_NORM_V3_BASE_H
#define BATCH_NORM_V3_BASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace BatchNormV3Ops {
using namespace AscendC;

template <typename T1, typename T2>
class BatchNormV3Base {
 public:
  __aicore__ inline BatchNormV3Base() {
  }

 protected:
  /* global memory address */
  GlobalTensor<T1> xGm;
  GlobalTensor<T2> weightGm;
  GlobalTensor<T2> biasGm;
  GlobalTensor<float> runningMeanGm;
  GlobalTensor<float> runningVarGm;

  GlobalTensor<T1> yGm;
  GlobalTensor<float> saveMeanGm;
  GlobalTensor<float> saveVarGm;
  GlobalTensor<float> runningMeanOutGm;
  GlobalTensor<float> runningVarOutGm;

  /* variable */
  float epsilon = 1e-5;
  float momentum = 0.1;
  float momentumReverse;
  float batchVarScale;

  /* ascendc variable */
  TPipe* pipe_ = nullptr;
  uint32_t blockIdx = GetBlockIdx();
  uint32_t useCoreNum = GetBlockNum();
  // 公共函数声明
};
// 公共函数实现

}  // namespace BatchNormV3Ops
#endif  // BATCH_NORM_V3_BASE_H