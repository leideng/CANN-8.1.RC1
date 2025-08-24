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
 * \file diag_flat.cpp
 * \brief
 */
 
#include "diag_flat_nd_to_2d.h"
#include "diag_flat_nd_to_2d_with_few.h"
#include "diag_flat_nd_to_2d_b16_more64.h"
#include "diag_flat_nd_to_2d_b16_less.h"


using namespace DiagFlat;

extern "C" __global__ __aicore__ void diag_flat(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    if (workspace == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);

    // input number less than 64
    if (TILING_KEY_IS(101)) {
        // dtype is complex32/uint32/int32/float32
        if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
            DiagFlat::DiagFlatNDTo2DWithFew<int32_t> op;
            op.Init(input, output, workspace, &tilingData);
            op.Process();
        // dtype is complex64/uint64/int64/float64
        } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
            DiagFlat::DiagFlatNDTo2DWithFew<int64_t> op;
            op.Init(input, output, workspace, &tilingData);
            op.Process();
        // dtype is int16/uin16/float16
        } else {
            DiagFlat::DiagFlatNDTo2DWithFew<DTYPE_X> op;
            op.Init(input, output, workspace, &tilingData);
            op.Process();
        }
    // dtype is complex128, input number is less than 64
    } else if (TILING_KEY_IS(102)) {
            DiagFlat::DiagFlatND2To2DB16Less64<int64_t> op;
            op.Init(input, output, workspace, &tilingData);
            op.Process();
    // input number more than 64
    } else if (TILING_KEY_IS(103)) {
            // dtype is complex32/uint32/int32/float32
            if constexpr (sizeof(DTYPE_X) == sizeof(int32_t)) {
                DiagFlat::DiagFlatNDTo2D<int32_t> op;
                op.Init(input, output, workspace, &tilingData);
                op.Process();
            // dtype is complex64/uint64/int64/float64
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int64_t)) {
                DiagFlat::DiagFlatNDTo2D<int64_t> op;
                op.Init(input, output, workspace, &tilingData);
                op.Process();
            // dtype is int16/uin16/float16
            } else if constexpr (sizeof(DTYPE_X) == sizeof(int16_t)) {
                DiagFlat::DiagFlatNDTo2D<int16_t> op;
                op.Init(input, output, workspace, &tilingData);
                op.Process();
            } else {
                DiagFlat::DiagFlatNDTo2D<int8_t> op;
                op.Init(input, output, workspace, &tilingData);
                op.Process();
            }
    // input number more than 64, input type is complex128
    } else if (TILING_KEY_IS(104)) {
            DiagFlat::DiagFlatND2To2DB16More64<int64_t> op;
            op.Init(input, output, workspace, &tilingData);
            op.Process();
    }
}