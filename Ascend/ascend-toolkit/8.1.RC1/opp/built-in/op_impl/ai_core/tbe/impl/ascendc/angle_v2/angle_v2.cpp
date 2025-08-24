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
 * \file angle_v2.cpp
 * \brief
 */
#include "angle_v2_complex.h"
#include "angle_v2_u8.h"
#include "angle_v2_int.h"
#include "angle_v2.h"

using namespace AngleV2N;

extern "C" __global__ __aicore__ void angle_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        AngleV2N::AngleV2Complex<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        AngleV2N::AngleV2<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        AngleV2N::AngleV2<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        AngleV2N::AngleV2U8<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(5)) {
        AngleV2N::AngleV2U8<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(6)) {
        AngleV2N::AngleV2Int<int8_t, float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(7)) {
        AngleV2N::AngleV2Int<int16_t, float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(8)) {
        AngleV2N::AngleV2Int<int32_t, float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(9)) {
        AngleV2N::AngleV2Int<int64_t, float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
