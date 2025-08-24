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
 * \file tome_unmerge.cpp
 * \brief
 */
#include "tome_unmerge.hpp"

extern "C" __global__ __aicore__ void tome_unmerge(GM_ADDR attention, GM_ADDR ori_index_a, GM_ADDR ori_index_b,
                                                   GM_ADDR topk_indice, GM_ADDR arg_max, GM_ADDR unzip_token,
                                                   GM_ADDR workspace, GM_ADDR tiling) {
  if (attention == nullptr || ori_index_a == nullptr || ori_index_b == nullptr || topk_indice == nullptr ||
      arg_max == nullptr || unzip_token == nullptr) {
    return;
  }
  GET_TILING_DATA(tiling_data, tiling);
  TomeUnmerge op(tiling_data.batch, tiling_data.hiddenSize, tiling_data.topR,
                 tiling_data.seqlenA, tiling_data.seqlenB);
#if defined(__DAV_M200__) || defined(__DAV_C220_VEC__)
  op.Init(attention, ori_index_a, ori_index_b, topk_indice, arg_max, unzip_token);
  op.Process();
#endif
}