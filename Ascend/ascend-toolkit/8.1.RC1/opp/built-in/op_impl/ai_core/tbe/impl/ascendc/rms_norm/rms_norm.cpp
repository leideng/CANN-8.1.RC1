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
 * \file rms_norm.cpp
 * \brief
 */
#include "rms_norm.h"
#include "rms_norm_split_d.h"
#include "rms_norm_whole_reduce_sum.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__  != 100
#include "rms_norm_merge_n.h"
#include "rms_norm_single_row.h"
#endif

using namespace AscendC;

#define RMSNORM_TILING_NORMAL 0
#define RMSNORM_TILING_SPILT_D 1
#define RMSNORM_TILING_MERGE_N 2
#define RMSNORM_TILING_SINGLE_ROW 3

#define GENERAL_OP_IMPL(templateClass, ...)                                                               \
    do {                                                                                                                \
        templateClass<__VA_ARGS__> op;                                                                                  \
        op.Init(x, gamma, y, rstd, &tilingData);                                                                                   \
        op.Process();                                                                                                   \
    } while (0)

extern "C" __global__ __aicore__ void rms_norm(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling)
{
  GET_TILING_DATA(tilingData, tiling);
  if (TILING_KEY_IS(RMSNORM_TILING_NORMAL)) {
    GENERAL_OP_IMPL(KernelRmsNorm, DTYPE_X, DTYPE_GAMMA);
  } else if (TILING_KEY_IS(RMSNORM_TILING_SPILT_D)) {
    GENERAL_OP_IMPL(KernelRmsNormSplitD, DTYPE_X, DTYPE_GAMMA);
#if defined(__CCE_AICORE__) && __CCE_AICORE__  != 100
  } else if (TILING_KEY_IS(RMSNORM_TILING_MERGE_N)) {
    GENERAL_OP_IMPL(KernelRmsNormMergeN, DTYPE_X, DTYPE_GAMMA);
  } else if (TILING_KEY_IS(RMSNORM_TILING_SINGLE_ROW)) {
    GENERAL_OP_IMPL(KernelRmsNormSingleRow, DTYPE_X, DTYPE_GAMMA);
#endif
  } else if (TILING_KEY_IS(1110)) {
    GENERAL_OP_IMPL(KernelRmsNormWholeReduceSum, half, true);
  } else if (TILING_KEY_IS(1120)) {
    GENERAL_OP_IMPL(KernelRmsNormWholeReduceSum, float, true);
  } else if (TILING_KEY_IS(1130)) {
    GENERAL_OP_IMPL(KernelRmsNormWholeReduceSum, bfloat16_t, true);
  } else if (TILING_KEY_IS(1010)) {
    GENERAL_OP_IMPL(KernelRmsNormWholeReduceSum, half, false);
  } else if (TILING_KEY_IS(1020)) {
    GENERAL_OP_IMPL(KernelRmsNormWholeReduceSum, float, false);
  } else if (TILING_KEY_IS(1030)) {
    GENERAL_OP_IMPL(KernelRmsNormWholeReduceSum, bfloat16_t, false);
  }
}
