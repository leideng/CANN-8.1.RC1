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
 * \file embedding_bag.cpp
 * \brief
 */

#include "embedding_bag.h"
#include "embedding_bag_fp16.h"


using namespace AscendC;
    // kernel function
extern "C" __global__ __aicore__ void embedding_bag(GM_ADDR weight, GM_ADDR indices, GM_ADDR offsets,
                                GM_ADDR per_sample_weights, GM_ADDR y, GM_ADDR offset2bag,
                                GM_ADDR bag_size, GM_ADDR max_indices, GM_ADDR workspace, GM_ADDR tiling) {
    if (workspace == nullptr || GetUserWorkspace(workspace) == nullptr) {
        return;
    }
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR gmTensor[8] = {weight, indices, offsets, per_sample_weights, y, offset2bag, bag_size, max_indices};
    if (TILING_KEY_IS(1)) {
        EmbeddingBag<float,int> op(gmTensor, tilingData, pipe);
        op.Process();
    } 
    if (TILING_KEY_IS(2)) {
        EmbeddingBagFP16<half,int> op(gmTensor, tilingData, pipe);
        op.Process();
    }  
#if __CCE_AICORE__ >= 220
    if (TILING_KEY_IS(3)) {
        EmbeddingBagFP16<bfloat16_t,int> op(gmTensor, tilingData, pipe);
        op.Process();
    }
#endif
}
