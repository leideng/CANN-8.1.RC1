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
 * \file strided_slice_assign_v2.cpp
 * \brief
 */
#include "strided_slice_assign_v2.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void strided_slice_assign_v2(GM_ADDR var, GM_ADDR input_value, GM_ADDR begin, GM_ADDR end, GM_ADDR strides, GM_ADDR axes, GM_ADDR var_out, GM_ADDR workspace, GM_ADDR tiling)
{
  TPipe pipe;
  GET_TILING_DATA(tilingData, tiling);
  if (TILING_KEY_IS(1)) {
    KernelStridedSliceAssignV2<DTYPE_VAR> op(&pipe);
    op.Init(var, input_value, begin, end, strides, axes, var_out, &tilingData);
    op.Process();
  }
}
