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
 * \file instance_norm_v3.cpp
 * \brief
 */

#include "instance_norm_nchw_kernel.h"
#include "instance_norm_nchw_kernel_cut_reduce.h"

extern "C" __global__ __aicore__ void instance_norm_v3(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean,
                                                       GM_ADDR variance, GM_ADDR workspace, GM_ADDR tiling) {
  TPipe pipe;
  GET_TILING_DATA(tilingData, tiling);
  GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

#define INIT_AND_PROCESS                                                 \
  op.Init(x, gamma, beta, y, mean, variance, usrWorkspace, &tilingData); \
  op.Process()

  // SingleRowDynamic
  if (TILING_KEY_IS(1)) {
    KernelInstanceNormNCHW<DTYPE_X, 1> op(&pipe);
    INIT_AND_PROCESS;
  } else if (TILING_KEY_IS(2)) {
    KernelInstanceNormNCHWCutReduce<DTYPE_X, 1> op(&pipe);
    INIT_AND_PROCESS;
  }
}