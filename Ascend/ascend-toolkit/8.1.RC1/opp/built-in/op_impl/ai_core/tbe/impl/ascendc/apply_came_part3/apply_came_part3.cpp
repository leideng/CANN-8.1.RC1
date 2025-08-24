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
 * \file apply_came_part3.cpp
 * \brief
 */
#include "apply_came_part3_fp32.cpp"
#include "apply_came_part3_fp16.cpp"
#include "apply_came_part3_post.h"


extern "C" __global__ __aicore__ void apply_came_part3(GM_ADDR u, GM_ADDR m_in, GM_ADDR eps, GM_ADDR beta1,
                                                       GM_ADDR clip_threshold, GM_ADDR sum_square_u,
                                                       GM_ADDR global_shape, GM_ADDR m_out, GM_ADDR sum_u_r,
                                                       GM_ADDR sum_u_c, GM_ADDR sum_u_rc, GM_ADDR workspace,
                                                       GM_ADDR cameTiling) {
  if (workspace == nullptr) {
    return;
  }
  GM_ADDR userWS = GetUserWorkspace(workspace);
  if (userWS == nullptr) {
    return;
  }
  ENABLE_DETERMINISTIC();

  GET_TILING_DATA(tiling_data, cameTiling);

  CamePart3InOut camePart3InOut;
  camePart3InOut.u = u;
  camePart3InOut.mIn = m_in;
  camePart3InOut.eps = eps;
  camePart3InOut.beta1 = beta1;
  camePart3InOut.clipThreshold = clip_threshold;
  camePart3InOut.sumSquareU = sum_square_u;
  camePart3InOut.globalShape = global_shape;
  camePart3InOut.mOut = m_out;
  camePart3InOut.sumUR = sum_u_r;
  camePart3InOut.sumUC = sum_u_c;
  camePart3InOut.sumURC = sum_u_rc;

  if (TILING_KEY_IS(10000000)) {
    ApplyCamePart3FP32 op;
    op.Init(camePart3InOut, userWS, &tiling_data);
    op.Process();
    pipe_barrier(PIPE_ALL);
    ApplyCamePart3Post<float> op_post;
    op_post.Init(camePart3InOut, userWS, &tiling_data);
    op_post.Process();
  } else if (TILING_KEY_IS(10000001)) {
    ApplyCamePart3FP16<half> op;
    op.Init(camePart3InOut, userWS, &tiling_data);
    op.Process();
    pipe_barrier(PIPE_ALL);
    ApplyCamePart3Post<float> op_post;
    op_post.Init(camePart3InOut, userWS, &tiling_data);
    op_post.Process();
  } else if (TILING_KEY_IS(10000002)) {
    ApplyCamePart3FP16<bfloat16_t> op;
    op.Init(camePart3InOut, userWS, &tiling_data);
    op.Process();
    pipe_barrier(PIPE_ALL);
    ApplyCamePart3Post<float> op_post;
    op_post.Init(camePart3InOut, userWS, &tiling_data);
    op_post.Process();
  }
}
