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
 * \file ring_attention_update.cpp
 * \brief
 */
#include "ring_attention_update.h"
#include "ring_attention_update_tnd.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void ring_attention_update(
  GM_ADDR prevAttnOut, GM_ADDR prevSoftmaxMax, GM_ADDR prevSoftmaxSum,
  GM_ADDR curAttnOut, GM_ADDR curSoftmaxMax, GM_ADDR curSoftmaxSum,
  GM_ADDR actualSeqQlen,
  GM_ADDR attnOut, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
  GM_ADDR workspace, GM_ADDR tiling)
{
  TPipe pipe;
  GET_TILING_DATA(tilingDataIn, tiling);
  const RingAttentionUpdateTilingData* __restrict tilingData = &tilingDataIn;
  GM_ADDR userWorkspace = GetUserWorkspace(workspace);

  if (TILING_KEY_IS(0)) {
    KernelRingAttentionUpdate<half> op;
    op.Init(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum,
            curAttnOut, curSoftmaxMax, curSoftmaxSum,
            actualSeqQlen,
            attnOut, softmaxMax, softmaxSum,
            userWorkspace, tilingData, &pipe);
    op.Process();
  } else if (TILING_KEY_IS(1)) {
    KernelRingAttentionUpdate<bfloat16_t> op;
    op.Init(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum,
            curAttnOut, curSoftmaxMax, curSoftmaxSum,
            actualSeqQlen,
            attnOut, softmaxMax, softmaxSum,
            userWorkspace, tilingData, &pipe);
    op.Process();
  } else if (TILING_KEY_IS(2)) {
    KernelRingAttentionUpdate<float> op;
    op.Init(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum,
            curAttnOut, curSoftmaxMax, curSoftmaxSum,
            actualSeqQlen,
            attnOut, softmaxMax, softmaxSum,
            userWorkspace, tilingData, &pipe);
    op.Process();
  } else if (TILING_KEY_IS(10)) {
    KernelRingAttentionUpdateTND<half> op;
    op.Init(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum,
            curAttnOut, curSoftmaxMax, curSoftmaxSum,
            actualSeqQlen,
            attnOut, softmaxMax, softmaxSum,
            userWorkspace, tilingData, &pipe);
    op.Process();
  } else if (TILING_KEY_IS(11)) {
    KernelRingAttentionUpdateTND<bfloat16_t> op;
    op.Init(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum,
            curAttnOut, curSoftmaxMax, curSoftmaxSum,
            actualSeqQlen,
            attnOut, softmaxMax, softmaxSum,
            userWorkspace, tilingData, &pipe);
    op.Process();
  } else if (TILING_KEY_IS(12)) {
    KernelRingAttentionUpdateTND<float> op;
    op.Init(prevAttnOut, prevSoftmaxMax, prevSoftmaxSum,
            curAttnOut, curSoftmaxMax, curSoftmaxSum,
            actualSeqQlen,
            attnOut, softmaxMax, softmaxSum,
            userWorkspace, tilingData, &pipe);
    op.Process();
  }
}
