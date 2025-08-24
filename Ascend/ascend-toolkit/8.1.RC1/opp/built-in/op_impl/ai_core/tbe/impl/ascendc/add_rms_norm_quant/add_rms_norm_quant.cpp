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
 * \file add_rms_norm_quant.cpp
 * \brief
 */
#include "add_rms_norm_quant.h"
#include "add_rms_norm_quant_split_d.h"
#include "add_rms_norm_quant_single_n.h"
using namespace AscendC;

#ifndef DTYPE_ZERO_POINTS1
#define DTYPE_ZERO_POINTS1 int32_t
#endif

#define INIT_AND_PROCESS                                                                          \
  do {                                                                                            \
    op.Init(x1, x2, gamma, scales1, scales2, zero_points1, zero_points2, y1, y2, x, &tilingData); \
    op.Process();                                                                                 \
} while(0)                                                                                        \

extern "C" __global__ __aicore__ void add_rms_norm_quant(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR scales1,
                                                         GM_ADDR scales2, GM_ADDR zero_points1, GM_ADDR zero_points2,
                                                         GM_ADDR y1, GM_ADDR y2, GM_ADDR x, GM_ADDR workspace,
                                                         GM_ADDR tiling) {
  TPipe pipe;
  GET_TILING_DATA(tilingData, tiling);
  if (TILING_KEY_IS(0)) {
    KernelAddRmsNormQuant<DTYPE_X1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1> op(&pipe);
    INIT_AND_PROCESS;
  } else if (TILING_KEY_IS(1)) {
    KernelAddRmsNormQuantSplitD<DTYPE_X1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1> op(&pipe);
    INIT_AND_PROCESS;
  } else if (TILING_KEY_IS(3)) {
    KernelAddRmsNormQuantSingleN<DTYPE_X1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1> op(&pipe);
    INIT_AND_PROCESS;
  }
}