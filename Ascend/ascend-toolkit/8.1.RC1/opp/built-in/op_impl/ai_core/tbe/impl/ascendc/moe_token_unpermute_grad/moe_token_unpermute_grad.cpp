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
 * \file moe_token_unpermute_grad.cpp
 * \brief
 */
#include "moe_token_unpermute_grad_prob_none.h"
#include "moe_token_unpermute_grad_prob_not_none.h"
using namespace AscendC;
using namespace MoeTokenUnpermuteGrad;
extern "C" __global__ __aicore__ void moe_token_unpermute_grad(GM_ADDR permuted_tokens, GM_ADDR unpermuted_tokens_grad, GM_ADDR sorted_indices, GM_ADDR probs, GM_ADDR permuted_tokens_grad, GM_ADDR probs_grad, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    // 0: padded_mode = False, 不存在prob, bf16
    // 1: padded_mode = False, 存在prob, bf16
    // 10: padded_mode = True, 不存在prob, bf16
    // 11: padded_mode = True, 存在prob, bf16
    // 100: padded_mode = False, 不存在prob, fp16
    // 101: padded_mode = False, 存在prob, fp16
    // 110: padded_mode = True, 不存在prob, fp16
    // 111: padded_mode = True, 存在prob, fp16
    // 200: padded_mode = False, 不存在prob, fp32
    // 201: padded_mode = False, 存在prob, fp32
    // 210: padded_mode = True, 不存在prob, fp32
    // 211: padded_mode = True, 存在prob, fp32
    if (TILING_KEY_IS(0) || TILING_KEY_IS(10)) {
        MoeTokenUnpermuteGradProbNone<bfloat16_t, int32_t> moeTokenUnpermuteGradProbNoneOp;
        moeTokenUnpermuteGradProbNoneOp.Init(permuted_tokens, unpermuted_tokens_grad, sorted_indices, probs, permuted_tokens_grad, probs_grad, tilingData);
        moeTokenUnpermuteGradProbNoneOp.Process();
    } else if (TILING_KEY_IS(1) || TILING_KEY_IS(11)) {
        MoeTokenUnpermuteGradProbNotNone<bfloat16_t, int32_t> moeTokenUnpermuteGradProbNotNoneOp;
        moeTokenUnpermuteGradProbNotNoneOp.Init(permuted_tokens, unpermuted_tokens_grad, sorted_indices, probs, permuted_tokens_grad, probs_grad, tilingData);
        moeTokenUnpermuteGradProbNotNoneOp.Process();
    } else if (TILING_KEY_IS(100) || TILING_KEY_IS(110)) {
        MoeTokenUnpermuteGradProbNone<half, int32_t> moeTokenUnpermuteGradProbNoneOp;
        moeTokenUnpermuteGradProbNoneOp.Init(permuted_tokens, unpermuted_tokens_grad, sorted_indices, probs, permuted_tokens_grad, probs_grad, tilingData);
        moeTokenUnpermuteGradProbNoneOp.Process();
    } else if (TILING_KEY_IS(101) || TILING_KEY_IS(111)) {
        MoeTokenUnpermuteGradProbNotNone<half, int32_t> moeTokenUnpermuteGradProbNotNoneOp;
        moeTokenUnpermuteGradProbNotNoneOp.Init(permuted_tokens, unpermuted_tokens_grad, sorted_indices, probs, permuted_tokens_grad, probs_grad, tilingData);
        moeTokenUnpermuteGradProbNotNoneOp.Process();
    } else if (TILING_KEY_IS(200) || TILING_KEY_IS(210)) {
        MoeTokenUnpermuteGradProbNone<float, int32_t> moeTokenUnpermuteGradProbNoneOp;
        moeTokenUnpermuteGradProbNoneOp.Init(permuted_tokens, unpermuted_tokens_grad, sorted_indices, probs, permuted_tokens_grad, probs_grad, tilingData);
        moeTokenUnpermuteGradProbNoneOp.Process();
    } else if (TILING_KEY_IS(201) || TILING_KEY_IS(211)) {
        MoeTokenUnpermuteGradProbNotNone<float, int32_t> moeTokenUnpermuteGradProbNotNoneOp;
        moeTokenUnpermuteGradProbNotNoneOp.Init(permuted_tokens, unpermuted_tokens_grad, sorted_indices, probs, permuted_tokens_grad, probs_grad, tilingData);
        moeTokenUnpermuteGradProbNotNoneOp.Process();
    }
}