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
 * \file fake_quant_affine_cachemask.cpp
 * \brief
 */
#include "fake_quant_affine_cachemask_fp32.h"
#include "fake_quant_affine_cachemask_fp16.h"

using namespace FakeQuantAffineCachemaskN;

extern "C" __global__ __aicore__ void fake_quant_affine_cachemask(GM_ADDR x, GM_ADDR scale,
    GM_ADDR zero_point, GM_ADDR y, GM_ADDR mask, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        FakeQuantAffineCachemaskN::FakeQuantAffineCachemaskFp32<float> op;
        op.Init(x, scale, zero_point, y, mask, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        FakeQuantAffineCachemaskN::FakeQuantAffineCachemaskFp16<half> op;
        op.Init(x, scale, zero_point, y, mask, &tilingData);
        op.Process();
    }
}