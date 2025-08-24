/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
 * \file stack_group_points.cpp
 * \brief
 */
 
#include "stack_group_points.h"

extern "C" __global__ __aicore__ void stack_group_points(GM_ADDR features, GM_ADDR features_batch_cnt, GM_ADDR indices,
                                                          GM_ADDR indices_batch_cnt, GM_ADDR y, GM_ADDR workspace,
                                                          GM_ADDR tiling) {
  GET_TILING_DATA(Tiling_Data, tiling);
  if (TILING_KEY_IS(0)) {
    StackGroupPoints<half> op;
    op.Init(features, features_batch_cnt, indices, indices_batch_cnt, y, workspace, Tiling_Data.m, Tiling_Data.b,
    Tiling_Data.c, Tiling_Data.n, Tiling_Data.nsample, Tiling_Data.res, Tiling_Data.reminder, Tiling_Data.featuresSize, 
    Tiling_Data.indicesSize, Tiling_Data.fbcSize, Tiling_Data.ibcSize, Tiling_Data.outLength,Tiling_Data.actCore, 
    Tiling_Data.standard);
    op.Process();
  } else if (TILING_KEY_IS(1)) {
    StackGroupPoints<float> op;
    op.Init(features, features_batch_cnt, indices, indices_batch_cnt, y, workspace, Tiling_Data.m, Tiling_Data.b,
    Tiling_Data.c, Tiling_Data.n, Tiling_Data.nsample, Tiling_Data.res, Tiling_Data.reminder, Tiling_Data.featuresSize, 
    Tiling_Data.indicesSize, Tiling_Data.fbcSize, Tiling_Data.ibcSize, Tiling_Data.outLength, Tiling_Data.actCore, 
    Tiling_Data.standard);
    op.Process();
  } 
}