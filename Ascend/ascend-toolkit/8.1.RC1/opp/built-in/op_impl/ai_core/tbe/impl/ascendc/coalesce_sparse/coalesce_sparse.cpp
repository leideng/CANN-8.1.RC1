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
 * \file coalesce_sparse.cpp
 * \brief
 */
#include "coalesce_sparse.h"

extern "C" __global__ __aicore__ void coalesce_sparse(GM_ADDR unique_len, GM_ADDR unique_indices, GM_ADDR indices, GM_ADDR values, GM_ADDR new_indices, GM_ADDR new_value, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    const CoalesceSparseTilingData* __restrict tilingDevice = &tilingData;
    if (TILING_KEY_IS(0)) {
        KernelCoalesceSparse<int64_t, int64_t, float> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(1)) {
        KernelCoalesceSparse<int64_t, int64_t, int32_t> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(2)) {
        KernelCoalesceSparse<int64_t, int64_t, half> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(3)) {
        KernelCoalesceSparse<int64_t, int32_t, float> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(4)) {
        KernelCoalesceSparse<int64_t, int32_t, int32_t> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(5)) {
        KernelCoalesceSparse<int64_t, int32_t, half> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(6)) {
        KernelCoalesceSparse<int32_t, int64_t, float> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(7)) {
        KernelCoalesceSparse<int32_t, int64_t, int32_t> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(8)) {
        KernelCoalesceSparse<int32_t, int64_t, half> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(9)) {
        KernelCoalesceSparse<int32_t, int32_t, float> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(10)) {
        KernelCoalesceSparse<int32_t, int32_t, int32_t> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }else if (TILING_KEY_IS(11)) {
        KernelCoalesceSparse<int32_t, int32_t, half> op;
        op.Init(unique_indices, indices, values, new_indices, new_value, tilingDevice);
        op.Process();
    }
}