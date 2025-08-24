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
 * \file moe_compute_expert_tokens.cpp
 * \brief
 */
#include "moe_compute_expert_tokens_int32_l.h"
#include "moe_compute_expert_tokens_int32_s.h"
#include "moe_compute_expert_tokens_int32_ss.h"
#include "moe_compute_expert_tokens_int32_m.h"

using namespace MoeCompute;

extern "C" __global__ __aicore__ void moe_compute_expert_tokens(
    GM_ADDR sortExperts, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(1001)) {
        MoeCompute::MoeComputeExpertTokensInt32SS<int32_t> op;
        op.Init(sortExperts, out, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1002)) {
        MoeCompute::MoeComputeExpertTokensInt32M<int32_t> op;
        op.Init(sortExperts, out, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1003)) {
        MoeCompute::MoeComputeExpertTokensInt32L<int32_t> op;
        op.Init(sortExperts, out, userWS, &tilingData);
        op.Process();
    }
}