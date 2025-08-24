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
 * \file multi_head_attention_score_grad.cpp
 * \brief
 */
#include "kernel_operator.h"

#include "multi_head_attention_score_grad_split_b_n_nd_nd_nd.h"

// implementation of kernel function
extern "C" __global__ __aicore__ void multi_head_attention_score_grad(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* dy, __gm__ uint8_t* pse_shift,
    __gm__ uint8_t* drop_mask, __gm__ uint8_t* padding_mask, __gm__ uint8_t* atten_mask, __gm__ uint8_t* softmax_in,
    __gm__ uint8_t* attention_in, __gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* dpse,
    __gm__ uint8_t* workspace, __gm__ uint8_t* tiling_data) {
#if __CCE_AICORE__ == 220
#ifdef __DAV_C220_CUBE__
  set_l1_3d_size(0);
  set_padding(0);
#elif defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_atomic_none();
  set_vector_mask((uint64_t)-1, (uint64_t)-1);
  // set_rpn_offset(1.0f, 1.0h);
#else
#endif
#endif

  TPipe pipeIn;
  __gm__ uint8_t* user = GetUserWorkspace(workspace);

  GET_TILING_DATA(tiling_data_in, tiling_data);
  const MultiHeadAttentionScoreGradTilingData* __restrict tilingData = &tiling_data_in;

  const TCubeTiling* __restrict bmm1tiling;
  const TCubeTiling* __restrict bmm2tiling;
  const TCubeTiling* __restrict bmm3tiling;
  const TCubeTiling* __restrict bmm4tiling;

  bmm1tiling = &(tilingData->mm1TilingData);
  bmm2tiling = &(tilingData->mm2TilingData);
  bmm3tiling = &(tilingData->mm3TilingData);
  bmm4tiling = &(tilingData->mm4TilingData);

  set_mask_norm();

  if (TILING_KEY_IS(1)) {
    MultiHeadAttentionScoreGradBNNdNdNdTranspose<half> op;
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm2tiling, op.mm3, bmm3tiling, op.mm4,
                      bmm4tiling);
    op.Init(key, value, dy, query, drop_mask, atten_mask, softmax_in, dq, dk, dv, dpse, user, tilingData, &pipeIn);
    op.Process();
  } else if (TILING_KEY_IS(3)) {
    MultiHeadAttentionScoreGradBNNdNdNdTranspose<float> op;
    REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm2, bmm2tiling, op.mm3, bmm3tiling, op.mm4,
                      bmm4tiling);
    op.Init(key, value, dy, query, drop_mask, atten_mask, softmax_in, dq, dk, dv, dpse, user, tilingData, &pipeIn);
    op.Process();
  }
}
