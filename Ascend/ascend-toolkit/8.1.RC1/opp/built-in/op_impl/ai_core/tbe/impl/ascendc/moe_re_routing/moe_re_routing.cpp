/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file moe_re_routing.cpp
 * \brief
 */

#include "moe_re_routing.h"
#include "kernel_operator.h"

#define MOE_RE_ROUTING_TOKENS_INT8_EXPERTS_INT32_SCALES_FLOAT32 100000UL
#define MOE_RE_ROUTING_TOKENS_INT8_EXPERTS_INT64_SCALES_FLOAT32 100010UL
#define MOE_RE_ROUTING_TOKENS_FP16_EXPERTS_INT32_SCALES_FLOAT32 100100UL
#define MOE_RE_ROUTING_TOKENS_FP16_EXPERTS_INT64_SCALES_FLOAT32 100110UL
#define MOE_RE_ROUTING_TOKENS_BF16_EXPERTS_INT32_SCALES_FLOAT32 100200UL
#define MOE_RE_ROUTING_TOKENS_BF16_EXPERTS_INT64_SCALES_FLOAT32 100210UL

extern "C" __global__ __aicore__ void moe_re_routing(GM_ADDR tokens,
                                                     GM_ADDR expertTokensNumPerRank,
                                                     GM_ADDR perTokenScales,
                                                     GM_ADDR permuteTokens,
                                                     GM_ADDR permutePerTokenScales,
                                                     GM_ADDR permuteTokenIdx,
                                                     GM_ADDR expertTokenNum,
                                                     GM_ADDR workspace, GM_ADDR tiling) {
  TPipe pipe;
  GET_TILING_DATA(tiling_data, tiling);
  if (TILING_KEY_IS(MOE_RE_ROUTING_TOKENS_INT8_EXPERTS_INT32_SCALES_FLOAT32)) {
    KernelMoeReRouting<int8_t, int32_t, float> op(&pipe, &tiling_data);
    op.Init(tokens, expertTokensNumPerRank, perTokenScales, permuteTokens,
            permutePerTokenScales, permuteTokenIdx, expertTokenNum);
    op.Process();
  } else if (TILING_KEY_IS(MOE_RE_ROUTING_TOKENS_INT8_EXPERTS_INT64_SCALES_FLOAT32)) {
    KernelMoeReRouting<int8_t, int64_t, float> op(&pipe, &tiling_data);
    op.Init(tokens, expertTokensNumPerRank, perTokenScales, permuteTokens,
            permutePerTokenScales, permuteTokenIdx, expertTokenNum);
    op.Process();
  } else if (TILING_KEY_IS(MOE_RE_ROUTING_TOKENS_FP16_EXPERTS_INT32_SCALES_FLOAT32)) {
    KernelMoeReRouting<half, int32_t, float> op(&pipe, &tiling_data);
    op.Init(tokens, expertTokensNumPerRank, perTokenScales, permuteTokens,
            permutePerTokenScales, permuteTokenIdx, expertTokenNum);
    op.Process();
  } else if (TILING_KEY_IS(MOE_RE_ROUTING_TOKENS_FP16_EXPERTS_INT64_SCALES_FLOAT32)) {
    KernelMoeReRouting<half, int64_t, float> op(&pipe, &tiling_data);
    op.Init(tokens, expertTokensNumPerRank, perTokenScales, permuteTokens,
            permutePerTokenScales, permuteTokenIdx, expertTokenNum);
    op.Process();
  } else if (TILING_KEY_IS(MOE_RE_ROUTING_TOKENS_BF16_EXPERTS_INT32_SCALES_FLOAT32)) {
    KernelMoeReRouting<bfloat16_t, int32_t, float> op(&pipe, &tiling_data);
    op.Init(tokens, expertTokensNumPerRank, perTokenScales, permuteTokens,
            permutePerTokenScales, permuteTokenIdx, expertTokenNum);
    op.Process();
  } else if (TILING_KEY_IS(MOE_RE_ROUTING_TOKENS_BF16_EXPERTS_INT64_SCALES_FLOAT32)) {
    KernelMoeReRouting<bfloat16_t, int64_t, float> op(&pipe, &tiling_data);
    op.Init(tokens, expertTokensNumPerRank, perTokenScales, permuteTokens,
            permutePerTokenScales, permuteTokenIdx, expertTokenNum);
    op.Process();
  }
}