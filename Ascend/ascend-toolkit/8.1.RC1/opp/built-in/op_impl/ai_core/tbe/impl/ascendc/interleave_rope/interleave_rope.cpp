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
 * \file interleave_rope.cpp
 * \brief
 */

#include "interleave_rope_fixed_bnsd_b11d.h"
#include "interleave_rope_b11d.h"
#include "interleave_rope_b1sd.h"
#include "interleave_rope_split_s.h"

using namespace InterleaveRope;
#define INTERLEAVE_ROPE_FIXED_BNSD_B11D 1000
#define INTERLEAVE_ROPE_B11D 2000
#define INTERLEAVE_ROPE_B1SD 3000
#define INTERLEAVE_ROPE_SPLIT_S 4000

extern "C" __global__ __aicore__ void interleave_rope(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR y, GM_ADDR workspace,
                                                  GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(INTERLEAVE_ROPE_FIXED_BNSD_B11D)) {
        KernelInterleaveRopeFixBNSD<DTYPE_X> op(&pipe, &tilingData);
        op.Init(x, cos, sin, y);
        op.Process();
    } else if (TILING_KEY_IS(INTERLEAVE_ROPE_B11D)) {
        KernelInterleaveRopeB11D<DTYPE_X> op(&pipe, &tilingData);
        op.Init(x, cos, sin, y);
        op.Process();
    } else if (TILING_KEY_IS(INTERLEAVE_ROPE_B1SD)) {
        KernelInterleaveRopeB1SD<DTYPE_X> op(&pipe, &tilingData);
        op.Init(x, cos, sin, y);
        op.Process();
    } else if (TILING_KEY_IS(INTERLEAVE_ROPE_SPLIT_S)) {
        KernelInterleaveRopeSplitS<DTYPE_X> op(&pipe, &tilingData);
        op.Init(x, cos, sin, y);
        op.Process();
    }
}