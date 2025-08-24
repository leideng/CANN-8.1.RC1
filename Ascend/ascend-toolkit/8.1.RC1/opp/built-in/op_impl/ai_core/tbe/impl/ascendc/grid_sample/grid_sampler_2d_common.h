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
 * \file grid_sampler_2d_common.h
 * \brief
 */
#ifndef GRID_SAMPLER_2D_COMMON
#define  GRID_SAMPLER_2D_COMMON

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "grid_sample_common.h"

namespace GridSample {

using namespace AscendC;

struct InputTensorStruct2D {
  LocalTensor<float> iXFpUb;
  LocalTensor<float> iYFpUb;
  LocalTensor<int32_t> iXIntUb;
  LocalTensor<int32_t> iYIntUb;

  __aicore__ inline InputTensorStruct2D() {
  }

  __aicore__ inline InputTensorStruct2D(LocalTensor<float> iXFpUb, LocalTensor<float> iYFpUb,
                                      LocalTensor<int32_t> iXIntUb, LocalTensor<int32_t> iYIntUb)
      : iXFpUb(iXFpUb), iYFpUb(iYFpUb), iXIntUb(iXIntUb), iYIntUb(iYIntUb) {
  }
};

struct ProcessParam2D {
  int32_t nIdx = 0;
  int32_t hwIdx = 0;
  int32_t calHWElems = 0;

  __aicore__ inline ProcessParam2D() {
  }
};

struct PointParam2D {
  int32_t loopElems = 0;
  int32_t loopOffset = 0;
  int64_t outBaseOffset = 0;
  int32_t maskOffset = 0;
  int32_t cIdx = 0;
  int32_t calCElems = 0;
  int32_t channelAlign = 0;
};
}  // namespace GridSample
#endif  //  GRID_SAMPLER_2D_COMMON