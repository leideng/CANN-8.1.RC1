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
 * \file scatter_add_with_sorted.cpp
 * \brief
 */
#include "scatter_add_float_with_sorted.h"
#include "scatter_add_int_with_sorted.h"
#include "scatter_with_sorted.h"

#define CALL_OP_IMPL_SCATTER(...)                                        \
  do {                                                                   \
    KernelScatterWithSorted<__VA_ARGS__> op;                             \
    op.Init(tilingDevice, &pipe, var, value, sorted_index, pos, output); \
    op.Process();                                                        \
  } while (0)

#define CALL_OP_IMPL_FLOAT(...)                                          \
  do {                                                                   \
    KernelScatterAddFloatWithSorted<__VA_ARGS__> op;                     \
    op.Init(tilingDevice, &pipe, var, value, sorted_index, pos, output); \
    op.Process();                                                        \
  } while (0)

#define CALL_OP_IMPL_INT(...)                                            \
  do {                                                                   \
    KernelScatterAddIntWithSorted<__VA_ARGS__> op;                       \
    op.Init(tilingDevice, &pipe, var, value, sorted_index, output);      \
    op.Process();                                                        \
  } while (0)

extern "C" __global__ __aicore__ void scatter_add_with_sorted(GM_ADDR var, GM_ADDR value, GM_ADDR sorted_index, 
                                                              GM_ADDR pos, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    const ScatterAddWithSortedTilingData* __restrict tilingDevice = &tiling_data;
    TPipe pipe;

    if (TILING_KEY_IS(1)) {
        CALL_OP_IMPL_SCATTER(float);
    } else if (TILING_KEY_IS(2)) {
        CALL_OP_IMPL_SCATTER(half);
    } else if (TILING_KEY_IS(3)) {
        CALL_OP_IMPL_SCATTER(int);
    } else if (TILING_KEY_IS(4)) {
        CALL_OP_IMPL_SCATTER(uint8_t);
    } else if (TILING_KEY_IS(5)) {
        CALL_OP_IMPL_SCATTER(int8_t);
    } else if (TILING_KEY_IS(6)) {
        CALL_OP_IMPL_SCATTER(bfloat16_t);
    } else if (TILING_KEY_IS(11)) {
        CALL_OP_IMPL_FLOAT(float);
    } else if (TILING_KEY_IS(12)) {
        CALL_OP_IMPL_FLOAT(half);
    } else if (TILING_KEY_IS(13)) {
        CALL_OP_IMPL_INT(int);
    } else if (TILING_KEY_IS(14)) {
        CALL_OP_IMPL_INT(uint8_t);
    } else if (TILING_KEY_IS(15)) {
        CALL_OP_IMPL_INT(int8_t);
    } else if (TILING_KEY_IS(16)) {
        CALL_OP_IMPL_FLOAT(bfloat16_t);
    }
}
