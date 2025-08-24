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
 * \file apply_rotary_pos_emb.cpp
 * \brief
 */
#include "apply_rotary_pos_emb_small.h"
#include "apply_rotary_pos_emb_compute_ab.h"
#include "apply_rotary_pos_emb_compute_ab_cast.h"


using namespace ApplyRotaryPosEmb;

extern "C" __global__ __aicore__ void apply_rotary_pos_emb(GM_ADDR q, GM_ADDR k, GM_ADDR cos, GM_ADDR sin,
                                                           GM_ADDR q_out, GM_ADDR k_out, GM_ADDR workspace,
                                                           GM_ADDR tiling) {
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);
    #if ORIG_DTYPE_QUERY != DT_BF16
        if (TILING_KEY_IS(1)) {
            ApplyRotaryPosEmb::ARPESmall<DTYPE_QUERY, DTYPE_QUERY> op;
            op.Init(q, k, cos, sin, q_out, k_out, userWS, &tilingData);
            op.Process(&tilingData);
        } else if (TILING_KEY_IS(3)) {
            ApplyRotaryPosEmb::ARPEComputeAB<DTYPE_QUERY, DTYPE_QUERY> op;
            op.Init(q, k, cos, sin, q_out, k_out, userWS, &tilingData);
            op.Process(&tilingData);
        } else if (TILING_KEY_IS(4)) {
            ApplyRotaryPosEmb::ARPEComputeABCast<DTYPE_QUERY, DTYPE_QUERY> op;
            op.Init(q, k, cos, sin, q_out, k_out, userWS, &tilingData);
            op.Process(&tilingData);
        } else {
            return;
        }
    #else
        if (TILING_KEY_IS(1)) {
            ApplyRotaryPosEmb::ARPESmall<DTYPE_QUERY, float> op;
            op.Init(q, k, cos, sin, q_out, k_out, userWS, &tilingData);
            op.Process(&tilingData);
        } else if (TILING_KEY_IS(4)) {
            ApplyRotaryPosEmb::ARPEComputeABCast<DTYPE_QUERY, float> op;
            op.Init(q, k, cos, sin, q_out, k_out, userWS, &tilingData);
            op.Process(&tilingData);
        } else {
            return;
        }
    #endif
}

