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
 * \file inplace_add_rms_norm.cpp
 * \brief
 */
#include "../add_rms_norm/add_rms_norm.h"
#include "../add_rms_norm/add_rms_norm_split_d.h"
#include "../add_rms_norm/add_rms_norm_merge_n.h"
#include "../add_rms_norm/add_rms_norm_multi_n.h"
#include "../add_rms_norm/add_rms_norm_single_n.h"

using namespace AscendC;

#define GENERAL_OP_IMPL(templateClass, ...)          \
  do {                                               \
    templateClass<__VA_ARGS__> op(&pipe);            \
    op.Init(x1, x2, gamma, y, rstd, x, &tilingData); \
    op.Process();                                    \
  } while (0)

extern "C" __global__ __aicore__ void inplace_add_rms_norm(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd,
                                                   GM_ADDR x, GM_ADDR workspace, GM_ADDR tiling) {
  TPipe pipe;
  GET_TILING_DATA(tilingData, tiling);
  if (TILING_KEY_IS(10)) {
    GENERAL_OP_IMPL(KernelAddRmsNorm, half);
  } else if (TILING_KEY_IS(20)) {
    GENERAL_OP_IMPL(KernelAddRmsNorm, float);
  } else if (TILING_KEY_IS(30)) {
    GENERAL_OP_IMPL(KernelAddRmsNorm, bfloat16_t);
  } else if (TILING_KEY_IS(11)) {
    GENERAL_OP_IMPL(KernelAddRmsNormSplitD, half);
  } else if (TILING_KEY_IS(21)) {
    GENERAL_OP_IMPL(KernelAddRmsNormSplitD, float);
  } else if (TILING_KEY_IS(31)) {
    GENERAL_OP_IMPL(KernelAddRmsNormSplitD, bfloat16_t);
  } else if (TILING_KEY_IS(12)) {
    GENERAL_OP_IMPL(KernelAddRmsNormMergeN, half);
  } else if (TILING_KEY_IS(22)) {
    GENERAL_OP_IMPL(KernelAddRmsNormMergeN, float);
  } else if (TILING_KEY_IS(32)) {
    GENERAL_OP_IMPL(KernelAddRmsNormMergeN, bfloat16_t);
  } else if (TILING_KEY_IS(13)) {
    GENERAL_OP_IMPL(KernelAddRmsNormSingleN, half);
  } else if (TILING_KEY_IS(23)) {
    GENERAL_OP_IMPL(KernelAddRmsNormSingleN, float);
  } else if (TILING_KEY_IS(33)) {
    GENERAL_OP_IMPL(KernelAddRmsNormSingleN, bfloat16_t);
  } else if (TILING_KEY_IS(14)) {
    GENERAL_OP_IMPL(KernelAddRmsNormMultiN, half);
  }
}