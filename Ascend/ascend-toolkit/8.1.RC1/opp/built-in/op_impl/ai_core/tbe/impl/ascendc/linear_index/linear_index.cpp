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
 * \file linear_index.cpp
 * \brief
 */
#include "linear_index.h"

#define CALL_OP_IMPL(...)                                                \
  do {                                                                   \
    KernelLinearIndex<__VA_ARGS__> op;                                   \
    op.Init(tilingDevice, &pipe, indices, output);                       \
    op.Process();                                                        \
  } while (0)

extern "C" __global__ __aicore__ void linear_index(GM_ADDR indices, GM_ADDR var, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    const LinearIndexTilingData* __restrict tilingDevice = &tiling_data;
    TPipe pipe;

    if (TILING_KEY_IS(1)) {
        CALL_OP_IMPL(int, 0);
    } else if (TILING_KEY_IS(2)) {
        CALL_OP_IMPL(int64_t, 0);
    } else if (TILING_KEY_IS(11)) {
        CALL_OP_IMPL(int, 1);
    } else if (TILING_KEY_IS(12)) {
        CALL_OP_IMPL(int64_t, 1);
    } else if (TILING_KEY_IS(21)) {
        CALL_OP_IMPL(int, 2);
    } else if (TILING_KEY_IS(22)) {
        CALL_OP_IMPL(int64_t, 2);
    }
}
