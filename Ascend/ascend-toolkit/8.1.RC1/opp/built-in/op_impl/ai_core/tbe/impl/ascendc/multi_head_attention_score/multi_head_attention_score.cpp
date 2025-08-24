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
 * \file multi_head_attention_score.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "multi_head_attention_score_split_b_n.h"
extern "C" __global__ __aicore__ void multi_head_attention_score(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* pse_shift, __gm__ uint8_t* dropMask,
    __gm__ uint8_t* paddingMask, __gm__ uint8_t* attenMask, __gm__ uint8_t* softmax_out, __gm__ uint8_t* attention_out,
    __gm__ uint8_t* workspace, __gm__ uint8_t* tiling) {
  TPipe tPipe;
  set_mask_norm();

  /*
  拆解TilingData 数据；
  **/
  uint32_t tmp_block_idx = GetBlockIdx();
  const TCubeTiling* __restrict bmm1tiling;
  const TCubeTiling* __restrict bmm2tiling;
  GET_TILING_DATA(tiling_data_in, tiling);
  const MultiHeadAttentionScoreTilingData* __restrict tiling_data = &tiling_data_in;

  bmm1tiling = &(tiling_data->bmm1TilingData);
  bmm2tiling = &(tiling_data->bmm2TilingData);

  /*
  获取Op可用WorkSpace空间
  **/
  __gm__ uint8_t* user = GetUserWorkspace(workspace);
  if (TILING_KEY_IS(1)) {
    AttentionScoreSplitBN<half> op;
    REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling);
    op.Init(query, key, value, pse_shift, dropMask, attenMask, softmax_out, attention_out, user, tiling_data, &tPipe);
    op.Process();
  } else if (TILING_KEY_IS(2)) {
    AttentionScoreSplitBN<float> op;
    REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling);
    op.Init(query, key, value, pse_shift, dropMask, attenMask, softmax_out, attention_out, user, tiling_data, &tPipe);
    op.Process();
  }
}
