/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
 * \file matmul_layer_norm_reduce.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "matmul_layer_norm_reduce_perf.h"

using namespace AscendC;
using namespace MatmulLayerNormReduceND;

#define MATMUL_LAYER_NORM_REDUCE_PERF_IMPL(INPUT_TYPE, BUFFER_NUM)                        \
  do {                                                                                    \
    GET_TILING_DATA_WITH_STRUCT(MatmulLayerNormReduceTilingData, tiling_data_in, tiling); \
    const MatmulLayerNormReduceTilingData* __restrict tilingData = &tiling_data_in;       \
    MatmulLayerNormReducePerf<INPUT_TYPE, BUFFER_NUM> op;                                 \
    op.Init(x1, x2, bias, add, div, y, sum, square_sum, userWs, tilingData);              \
    op.Process();                                                                         \
  } while (0)

extern "C" __global__ __aicore__ void matmul_layer_norm_reduce(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR add,
                                                               GM_ADDR div, GM_ADDR y, GM_ADDR sum, GM_ADDR square_sum,
                                                               GM_ADDR workspace, GM_ADDR tiling) {
  if (workspace == nullptr) {
    return;
  }

  GM_ADDR userWs = GetUserWorkspace(workspace);
  if (userWs == nullptr) {
    return;
  }
  MATMUL_LAYER_NORM_REDUCE_PERF_IMPL(half, 2);
  return;
}
