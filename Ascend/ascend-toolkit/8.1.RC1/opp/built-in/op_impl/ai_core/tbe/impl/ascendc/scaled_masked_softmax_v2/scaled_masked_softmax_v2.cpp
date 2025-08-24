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
 * \file scaled_masked_softmax_v2.cpp
 * \brief
 */

#include "scaled_masked_softmax_v2.h"

extern "C" __global__ __aicore__ void scaled_masked_softmax_v2(GM_ADDR x, GM_ADDR mask, GM_ADDR y,
                                                            GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(0)) {
        AscendC::ScaledMaskedSoftmaxV2<float> op;
        op.Init(x, mask, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        AscendC::ScaledMaskedSoftmaxV2<half> op;
        op.Init(x, mask, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        AscendC::ScaledMaskedSoftmaxV2<bfloat16_t> op;
        op.Init(x, mask, y, tilingData);
        op.Process();
    }
}