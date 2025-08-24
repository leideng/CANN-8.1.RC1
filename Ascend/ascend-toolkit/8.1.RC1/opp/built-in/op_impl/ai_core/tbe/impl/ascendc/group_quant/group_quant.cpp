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
 * \file group_quant.cpp
 * \brief
 */

#include "group_quant_base.h"

using namespace GroupQuant;

extern "C" __global__ __aicore__ void group_quant(GM_ADDR x, GM_ADDR scale, GM_ADDR groupIndex, GM_ADDR offset,
                                                  GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(0)) {
        // The dtype of input offset must be same as scale.
        // Input offset is optional, so set offset dtype by scale dtype.
        GroupQuant::GroupQuantBase<DTYPE_X, DTYPE_SCALE, DTYPE_GROUP_INDEX, DTYPE_SCALE, DTYPE_Y> op;
        op.Init(x, scale, groupIndex, offset, y, &tilingData);
        op.Process();
    }
}