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
 * \file instance_norm_nchw_base.h
 * \brief
 */

#ifndef __INSTANCE_NORM_NCHW_BASE_CLASS_H_
#define __INSTANCE_NORM_NCHW_BASE_CLASS_H_

#include "instance_norm_helper.h"

template <typename T, int TILING_KEY, int BUFFER_NUM = 1>
class KernelInstanceNormNCHWBase {
 public:
  __aicore__ inline KernelInstanceNormNCHWBase() {
  }

 protected:
  __aicore__ inline void InitBaseParams(const InstanceNormV3TilingData* tiling) {
    this->cAxis = tiling->C;
    this->reduceNums = tiling->reduceNums;
    this->nAxisPerCore = tiling->nAxisPerCore;
    this->nLoops = (GetBlockIdx() < tiling->useCoreNums - 1) ? tiling->nAxisPerCore : tiling->nAxisPerCoreTail;
    this->ubFactor = tiling->ubFactor;
    this->cAxisFactor = tiling->cAxisFactor;
    this->avgFactor = tiling->avgFactor;
    this->eps = tiling->epsilon;
  }

  __aicore__ inline void InitInGlobalTensors(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta) {
    this->xGm.SetGlobalBuffer((__gm__ T*)(x) + GetBlockIdx() * this->nAxisPerCore * this->cAxis * this->reduceNums,
                              this->nLoops * this->cAxis * this->reduceNums);
    this->gammaGm.SetGlobalBuffer((__gm__ T*)gamma, this->cAxis);
    this->betaGm.SetGlobalBuffer((__gm__ T*)beta, this->cAxis);
  }

  __aicore__ inline void InitOutGlobalTensors(GM_ADDR y, GM_ADDR mean, GM_ADDR variance) {
    this->yGm.SetGlobalBuffer((__gm__ T*)(y) + GetBlockIdx() * this->nAxisPerCore * this->cAxis * this->reduceNums,
                              this->nLoops * this->cAxis * this->reduceNums);
    this->meanGm.SetGlobalBuffer((__gm__ T*)mean + GetBlockIdx() * this->nAxisPerCore * this->cAxis,
                                 this->nLoops * this->cAxis);
    this->varianceGm.SetGlobalBuffer((__gm__ T*)variance + GetBlockIdx() * this->nAxisPerCore * this->cAxis,
                                     this->nLoops * this->cAxis);
  }

  __aicore__ inline void InitWorkSpaceGlobalTensors(GM_ADDR workspace) {
  }

 protected:
  GlobalTensor<T> xGm;
  GlobalTensor<T> gammaGm;
  GlobalTensor<T> betaGm;
  GlobalTensor<T> yGm;
  GlobalTensor<T> meanGm;
  GlobalTensor<T> varianceGm;

  uint32_t cAxis;
  uint32_t reduceNums;
  uint32_t nLoops;
  uint32_t nAxisPerCore;

  uint32_t ubFactor;
  uint32_t cAxisFactor;

  float avgFactor;
  float eps;
};

#endif  // __INSTANCE_NORM_NCHW_BASE_CLASS_H_
