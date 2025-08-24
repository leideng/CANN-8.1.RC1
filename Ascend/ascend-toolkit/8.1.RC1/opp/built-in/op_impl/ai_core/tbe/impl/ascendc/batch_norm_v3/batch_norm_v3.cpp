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
 * \file batch_norm_v3.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "batch_norm_v3_welford.h"
#include "batch_norm_v3_full_reduce.h"
using namespace AscendC;
using namespace BatchNormV3Ops;

namespace {
#define BNV3_WELFORD_R0_SPLIT_NOT_ALIGN 1000
#define BNV3_WELFORD_R0_SPLIT_ALIGN 1001
#define BNV3_WELFORD_R1_SPLIT_NOT_ALIGN_RO_NOT_ALIGN 1002
#define BNV3_WELFORD_R1_SPLIT_ALIGN_RO_NOT_ALIGN 1003
#define BNV3_WELFORD_R1_SPLIT_NOT_ALIGN_RO_ALIGN 1012
#define BNV3_WELFORD_R1_SPLIT_ALIGN_RO_ALIGN 1013
#define BNV3_FULL_REDUCE_NORMAL 2000
#define BNV3_FULL_REDUCE_A_PARALLEL 2001

}  // namespace

enum BNV3WelfordSplitAlignMode : int {
  R0_SPLIT_NOT_ALIGN_MODE = 0,
  R0_SPLIT_ALIGN_MODE = 1,
  R1_SPLIT_NOT_ALIGN_MODE = 2,
  R1_SPLIT_ALIGN_MODE = 3,
};

enum BNV3WelfordR0Align : int {
  R0_NOT_ALIGN = 0,
  R0_ALIGN = 1,
};

#define BATCH_NORM_V3_WELFORD_IMPL(INPUT_X_TYPE, INPUT_WEIGHT_TYPE, SPLIT_MODE, R0_ALIGN_MODE, PIPE)      \
  do {                                                                                                    \
    GET_TILING_DATA_WITH_STRUCT(BatchNormV3WelfordTilingData, tiling_data_in, tiling);                    \
    const BatchNormV3WelfordTilingData* __restrict tilingData = &tiling_data_in;                          \
    BatchNormV3Welford<INPUT_X_TYPE, INPUT_WEIGHT_TYPE, SPLIT_MODE, R0_ALIGN_MODE> op(&PIPE);             \
    op.Init(x, weight, bias, mean, variance, y, mean_out, variance_out, save_mean, save_var, tilingData); \
    op.Process();                                                                                         \
  } while (0)

extern "C" __global__ __aicore__ void batch_norm_v3(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR mean,
                                                    GM_ADDR variance, GM_ADDR y, GM_ADDR mean_out, GM_ADDR variance_out,
                                                    GM_ADDR save_mean, GM_ADDR save_var, GM_ADDR workspace,
                                                    GM_ADDR tiling) {
  if (g_coreType == AIC) {
    return;
  }
  TPipe pipe;
  if (TILING_KEY_IS(BNV3_WELFORD_R0_SPLIT_NOT_ALIGN)) {
    BATCH_NORM_V3_WELFORD_IMPL(DTYPE_X, DTYPE_WEIGHT, BNV3WelfordSplitAlignMode::R0_SPLIT_NOT_ALIGN_MODE,
                               BNV3WelfordR0Align::R0_NOT_ALIGN, pipe);
  } else if (TILING_KEY_IS(BNV3_WELFORD_R0_SPLIT_ALIGN)) {
    BATCH_NORM_V3_WELFORD_IMPL(DTYPE_X, DTYPE_WEIGHT, BNV3WelfordSplitAlignMode::R0_SPLIT_ALIGN_MODE,
                               BNV3WelfordR0Align::R0_NOT_ALIGN, pipe);
  } else if (TILING_KEY_IS(BNV3_WELFORD_R1_SPLIT_NOT_ALIGN_RO_NOT_ALIGN)) {
    BATCH_NORM_V3_WELFORD_IMPL(DTYPE_X, DTYPE_WEIGHT, BNV3WelfordSplitAlignMode::R1_SPLIT_NOT_ALIGN_MODE,
                               BNV3WelfordR0Align::R0_NOT_ALIGN, pipe);
  } else if (TILING_KEY_IS(BNV3_WELFORD_R1_SPLIT_ALIGN_RO_NOT_ALIGN)) {
    BATCH_NORM_V3_WELFORD_IMPL(DTYPE_X, DTYPE_WEIGHT, BNV3WelfordSplitAlignMode::R1_SPLIT_ALIGN_MODE,
                               BNV3WelfordR0Align::R0_NOT_ALIGN, pipe);
  } else if (TILING_KEY_IS(BNV3_WELFORD_R1_SPLIT_NOT_ALIGN_RO_ALIGN)) {
    BATCH_NORM_V3_WELFORD_IMPL(DTYPE_X, DTYPE_WEIGHT, BNV3WelfordSplitAlignMode::R1_SPLIT_NOT_ALIGN_MODE,
                               BNV3WelfordR0Align::R0_ALIGN, pipe);
  } else if (TILING_KEY_IS(BNV3_WELFORD_R1_SPLIT_ALIGN_RO_ALIGN)) {
    BATCH_NORM_V3_WELFORD_IMPL(DTYPE_X, DTYPE_WEIGHT, BNV3WelfordSplitAlignMode::R1_SPLIT_ALIGN_MODE,
                               BNV3WelfordR0Align::R0_ALIGN, pipe);
  } else if (TILING_KEY_IS(BNV3_FULL_REDUCE_NORMAL)) {
    GET_TILING_DATA_WITH_STRUCT(BatchNormV3FullReduceTilingData, tiling_data_in, tiling);
    const BatchNormV3FullReduceTilingData* __restrict tilingData = &tiling_data_in;
    BatchNormV3FullReduce<DTYPE_X, DTYPE_WEIGHT, 0> op(&pipe);
    op.Init(x, weight, bias, mean, variance, y, mean_out, variance_out, save_mean, save_var, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(BNV3_FULL_REDUCE_A_PARALLEL)) {
    GET_TILING_DATA_WITH_STRUCT(BatchNormV3FullReduceTilingData, tiling_data_in, tiling);
    const BatchNormV3FullReduceTilingData* __restrict tilingData = &tiling_data_in;
    BatchNormV3FullReduce<DTYPE_X, DTYPE_WEIGHT, 1> op(&pipe);
    op.Init(x, weight, bias, mean, variance, y, mean_out, variance_out, save_mean, save_var, tilingData);
    op.Process();
  }
}