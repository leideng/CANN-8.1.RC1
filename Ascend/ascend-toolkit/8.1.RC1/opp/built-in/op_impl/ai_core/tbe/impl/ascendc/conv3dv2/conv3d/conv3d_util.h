/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
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
 * \file conv3d_util.h
 * \brief
 */

#ifndef CONV3D_UTIL_H
#define CONV3D_UTIL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"
#include "../conv_common/conv_util.h"

using namespace AscendC;

namespace conv3d {
const static uint64_t LOAD2D_MAX_REPEAT_TIMES = 255;
const static uint8_t RIGHT_MOVE_8 = 8;
const static uint32_t L0A_SIZE = 65536;
const static uint32_t L0B_SIZE = 65536;

static __aicore__ inline uint64_t GetCurrentKD(uint64_t tilingKL1, uint64_t cin, uint64_t khxKw)
{
    return conv::CeilDIV(tilingKL1, cin * khxKw);
}
}  // namespace conv3d
#endif  // __CONV3D_UTIL_H__
