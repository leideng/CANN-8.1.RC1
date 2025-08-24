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
 * \file rms_norm_grad.cpp
 * \brief
 */
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 100)
#include "rms_norm_grad_whole_reduce_n.h"
#include "rms_norm_grad_whole_reduce_d.h"
#else
#include "rms_norm_grad_split_d_high_precision.h"
#include "rms_norm_grad_split_n_high_precision.h"
#endif

extern "C" __global__ __aicore__ void rms_norm_grad(GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR dx,
                                                    GM_ADDR dgamma, GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tilingData, tiling);
  GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 100)
  if (TILING_KEY_IS(1)) {
    RmsNormGradWholeReduceN<DTYPE_DY> rms_norm_grad_whole_reduce_n;
    rms_norm_grad_whole_reduce_n.Init(dy, x, rstd, gamma, dx, dgamma, &tilingData);
    rms_norm_grad_whole_reduce_n.Process();
  }
  if (TILING_KEY_IS(2)) {
    RmsNormGradWholeReduceD<DTYPE_DY> rms_norm_grad_whole_reduce_d;
    rms_norm_grad_whole_reduce_d.Init(dy, x, rstd, gamma, dx, dgamma, &tilingData);
    rms_norm_grad_whole_reduce_d.Process();
  }
#else
  if (TILING_KEY_IS(1)) {
    RmsNormGradSplitNHighPrecision<DTYPE_DY, DTYPE_GAMMA> rms_norm_grad_split_n;
    rms_norm_grad_split_n.Init(dy, x, rstd, gamma, dx, dgamma, &tilingData, usrWorkspace);
    rms_norm_grad_split_n.Process();
  }
  if (TILING_KEY_IS(2)) {
    RmsNormGradSplitDHighPrecision<DTYPE_DY, DTYPE_GAMMA> rms_norm_grad_split_d;
    rms_norm_grad_split_d.Init(dy, x, rstd, gamma, dx, dgamma, &tilingData, usrWorkspace);
    rms_norm_grad_split_d.Process();
  }
#endif
}

#ifndef __CCE_KT_TEST__
void rms_norm_grad_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* dy, uint8_t* x, uint8_t* rstd,
                      uint8_t* gamma, uint8_t* dx, uint8_t* dgamma, uint8_t* workspace, uint8_t* tiling) {
  rms_norm_grad<<<blockDim, l2ctrl, stream>>>(dy, x, rstd, gamma, dx, dgamma, workspace, tiling);
}
#endif