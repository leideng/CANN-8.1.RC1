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
 * \file repeat_interleave_grad.cpp
 * \brief
 */
#include "repeat_interleave_grad.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void repeat_interleave_grad(GM_ADDR input_grad, GM_ADDR repeats, GM_ADDR output_grad, GM_ADDR workspace, GM_ADDR tiling)
{
  GET_TILING_DATA(tiling_data_in, tiling);
  const RepeatInterleaveGradTilingData *__restrict tiling_data = &tiling_data_in;
  if (TILING_KEY_IS(0)) {
    KernelRepeatInterleaveGrad<half, int32_t> op;
    op.Init(input_grad, repeats, output_grad, workspace, tiling_data);
    op.Process();
  } else if (TILING_KEY_IS(1)) {
    KernelRepeatInterleaveGrad<bfloat16_t, int32_t> op;
    op.Init(input_grad, repeats, output_grad, workspace, tiling_data);
    op.Process();
  } else if (TILING_KEY_IS(2)) {
    KernelRepeatInterleaveGrad<float, int32_t> op;
    op.Init(input_grad, repeats, output_grad, workspace, tiling_data);
    op.Process();
  } else if (TILING_KEY_IS(10)) {
    KernelRepeatInterleaveGrad<half, int64_t> op;
    op.Init(input_grad, repeats, output_grad, workspace, tiling_data);
    op.Process();
  } else if (TILING_KEY_IS(11)) {
    KernelRepeatInterleaveGrad<bfloat16_t, int64_t> op;
    op.Init(input_grad, repeats, output_grad, workspace, tiling_data);
    op.Process();
  } else if (TILING_KEY_IS(12)) {
    KernelRepeatInterleaveGrad<float, int64_t> op;
    op.Init(input_grad, repeats, output_grad, workspace, tiling_data);
    op.Process();
  }
}
