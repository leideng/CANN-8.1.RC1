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
 * \file tome_merge.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "tome_merge.h"

using namespace AscendC;
using namespace TomeMergeND;

extern "C" __global__ __aicore__ void tome_merge(GM_ADDR TokenA, GM_ADDR TokenB, GM_ADDR TOPK_Indice, GM_ADDR Arg_Max,
                                                 GM_ADDR unmergeTokenA, GM_ADDR unReduceTokenB, GM_ADDR unReduceCount,
                                                 GM_ADDR workspace, GM_ADDR tiling) {
  if (TokenA == nullptr || TokenB == nullptr || TOPK_Indice == nullptr || Arg_Max == nullptr ||
      unmergeTokenA == nullptr || unReduceTokenB == nullptr || unReduceCount == nullptr) {
    return;
  }

  GET_TILING_DATA(tilingData, tiling);

  TomeMerge op(tilingData.batch, tilingData.hiddenSize, tilingData.topR, tilingData.seqlenA, tilingData.seqlenB);

#if defined(__DAV_M200__) || defined(__DAV_C220_VEC__)
  op.Init(TokenA, TokenB, TOPK_Indice, Arg_Max, unmergeTokenA, unReduceTokenB, unReduceCount);
  op.Process();
#endif
}