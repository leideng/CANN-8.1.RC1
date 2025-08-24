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
 * \file rope_with_sin_cos_cache.cpp
 * \brief rope_with_sin_cos_cache.cpp
 */

#include "kernel_operator.h"
#include "rope_with_sin_cos_cache_fp32.h"
#include "rope_with_sin_cos_cache_f_bf16.h"

using namespace AscendC;
using namespace RopeWithSinCosCache;

extern "C" __global__ __aicore__ void rope_with_sin_cos_cache(
                                                            GM_ADDR position_id,
                                                            GM_ADDR query_in,
                                                            GM_ADDR key_in,
                                                            GM_ADDR cos_sin_cache,
                                                            GM_ADDR query_out,
                                                            GM_ADDR key_out,
                                                            GM_ADDR workspace,
                                                            GM_ADDR tiling
                                                            ) {
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    TPipe pipe;
    #if ORIG_DTYPE_QUERYIN == DT_BF16
        if (TILING_KEY_IS(20)) {
            TPipe* ptr = &pipe;
            if (ptr != nullptr) {
                RopeWithSinCosCacheFP16<bfloat16_t> op;
                op.Init(position_id, query_in, key_in, cos_sin_cache, query_out, key_out, tilingData, ptr);
                op.Process();
            }
        }
    #elif ORIG_DTYPE_QUERYIN == DT_FLOAT16
        if (TILING_KEY_IS(21)) {
            TPipe* ptr = &pipe;
            if (ptr != nullptr) {
                RopeWithSinCosCacheFP16<half> op;
                op.Init(position_id, query_in, key_in, cos_sin_cache, query_out, key_out, tilingData, ptr);
                op.Process();
            }
        }
    #elif ORIG_DTYPE_QUERYIN == DT_FLOAT
        if (TILING_KEY_IS(22)) {
            TPipe* ptr = &pipe;
            if (ptr != nullptr) {
                RopeWithSinCosCacheF32<float> op;
                op.Init(position_id, query_in, key_in, cos_sin_cache, query_out, key_out, tilingData, ptr);
                op.Process();
            }
        }
    #endif
}