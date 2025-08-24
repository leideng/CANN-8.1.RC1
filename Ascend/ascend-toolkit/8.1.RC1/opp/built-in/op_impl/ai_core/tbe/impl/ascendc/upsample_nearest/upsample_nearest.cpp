/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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
 * \file upsample_nearest.cpp
 * \brief
 */
#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 300
#include "upsample_nearest_310p.h"
#else
#include "upsample_nearest.h"
#endif

using namespace UpsampleNearest;

extern "C" __global__ __aicore__ void upsample_nearest(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    const UpsampleNearestTilingData *__restrict tilingDataParams = &tilingData;
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

#define INIT_PROCESS                                                                                    \
        op.Init(input, output, userWS, &tilingData);                                                    \
        op.Process()
#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 300
    if (TILING_KEY_IS(1000) || TILING_KEY_IS(1001) || TILING_KEY_IS(1002)) {
        if (tilingDataParams->dataType == 2) {
            UpsampleNearestND310p<half, 0> op;
            INIT_PROCESS;}
        if (tilingDataParams->dataType == 4) {
            UpsampleNearestND310p<float, 0> op;
            INIT_PROCESS;}}
#else
    if (TILING_KEY_IS(1000)) {
        if (tilingDataParams->dataType == 2) {
            UpsampleNearestND<half, 0> op;
            INIT_PROCESS;}
        if (tilingDataParams->dataType == 4) {
            UpsampleNearestND<float, 0> op;
            INIT_PROCESS;}
    } else if (TILING_KEY_IS(1001)) {
        if (tilingDataParams->dataType == 2) {
            UpsampleNearestND<half, 1> op;
            INIT_PROCESS;}
        if (tilingDataParams->dataType == 4) {
            UpsampleNearestND<float, 1> op;
            INIT_PROCESS;}
    } else if (TILING_KEY_IS(1002)) {
        if (tilingDataParams->dataType == 2) {
            UpsampleNearestND<half, 2> op;
            INIT_PROCESS;}
        if (tilingDataParams->dataType == 4) {
            UpsampleNearestND<float, 2> op;
            INIT_PROCESS;}
    } else if (TILING_KEY_IS(1003)) {
        if (tilingDataParams->dataType == 2) {
            UpsampleNearestND<half, 3> op;
            INIT_PROCESS;}
        if (tilingDataParams->dataType == 4) {
            UpsampleNearestND<float, 3> op;
            INIT_PROCESS;}
    }
#endif
}
