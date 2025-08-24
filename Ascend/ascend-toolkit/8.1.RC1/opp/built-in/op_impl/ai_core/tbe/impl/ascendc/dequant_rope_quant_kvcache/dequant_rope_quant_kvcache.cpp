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
 * \file dequant_rope_quant_kvcache.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "dequant_rope_quant_kvcache.h"
using namespace AscendC;
using namespace DequantRopeQuantKvcache;

#ifndef DTYPE_BIAS
#define DTYPE_BIAS half
#endif

extern "C" __global__ __aicore__ void dequant_rope_quant_kvcache(GM_ADDR x, GM_ADDR cos,
                                                                 GM_ADDR sin, GM_ADDR k_cache,
                                                                 GM_ADDR v_cache, GM_ADDR indices,
                                                                 GM_ADDR scale_k, GM_ADDR scale_v,
                                                                 GM_ADDR offset_k, GM_ADDR offset_v,
                                                                 GM_ADDR weight_scale, GM_ADDR activation_scale,
                                                                 GM_ADDR bias, GM_ADDR q,
                                                                 GM_ADDR k, GM_ADDR v,
                                                                 GM_ADDR k_cache_ref, GM_ADDR v_cache_ref,
                                                                 GM_ADDR workspace, GM_ADDR tiling) {

  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  GET_TILING_DATA(tilingData, tiling);
  if (TILING_KEY_IS(0)) {
    RopeQuantKvcacheV2<DTYPE_X, float, DTYPE_COS> op(&tilingData);
    op.Init(x, cos, sin, k_cache, v_cache, indices, weight_scale,
            activation_scale, bias,scale_k, scale_v, offset_k, offset_v,
            q, k, v, k_cache_ref, v_cache_ref); 
    op.Process();
  } else if (TILING_KEY_IS(1)) {
    RopeQuantKvcacheV2<DTYPE_X, half, DTYPE_COS> op(&tilingData);
    op.Init(x, cos, sin, k_cache, v_cache, indices, weight_scale,
            activation_scale, bias,scale_k, scale_v, offset_k, offset_v,
            q, k, v, k_cache_ref, v_cache_ref); 
    op.Process();
  } else if (TILING_KEY_IS(2)) {
    RopeQuantKvcacheV2<DTYPE_X, int32_t, DTYPE_COS> op(&tilingData);
    op.Init(x, cos, sin, k_cache, v_cache, indices, weight_scale,
            activation_scale, bias,scale_k, scale_v, offset_k, offset_v,
            q, k, v, k_cache_ref, v_cache_ref); 
    op.Process();
  } else if (TILING_KEY_IS(3)) {
    RopeQuantKvcacheV2<DTYPE_X, bfloat16_t, DTYPE_COS> op(&tilingData);
    op.Init(x, cos, sin, k_cache, v_cache, indices, weight_scale,
            activation_scale, bias,scale_k, scale_v, offset_k, offset_v,
            q, k, v, k_cache_ref, v_cache_ref); 
    op.Process();
  }
}