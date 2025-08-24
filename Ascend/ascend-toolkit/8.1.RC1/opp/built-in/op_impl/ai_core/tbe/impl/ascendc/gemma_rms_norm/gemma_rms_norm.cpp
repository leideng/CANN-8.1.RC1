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
 * \file gemma_rms_norm.cpp
 * \brief
 */
#include "../rms_norm/rms_norm.h"
#include "../rms_norm/rms_norm_split_d.h"

using namespace AscendC;

#define GENERAL_OP_IMPL(templateClass, ...)                                                               \
    do {                                                                                                                \
        templateClass<__VA_ARGS__> op;                                                                                  \
        op.Init(x, gamma, y, rstd, &tilingData);                                                                                   \
        op.Process();                                                                                                   \
    } while (0)

extern "C" __global__ __aicore__ void gemma_rms_norm(
  GM_ADDR x, 
  GM_ADDR gamma, 
  GM_ADDR y, 
  GM_ADDR rstd, 
  GM_ADDR workspace, 
  GM_ADDR tiling
)
{
  GET_TILING_DATA(tilingData, tiling);
  if (TILING_KEY_IS(0)) {
    GENERAL_OP_IMPL(KernelRmsNorm, DTYPE_X, DTYPE_GAMMA);
  } else if (TILING_KEY_IS(1)) {
    GENERAL_OP_IMPL(KernelRmsNormSplitD, DTYPE_X, DTYPE_GAMMA);
  }
}
