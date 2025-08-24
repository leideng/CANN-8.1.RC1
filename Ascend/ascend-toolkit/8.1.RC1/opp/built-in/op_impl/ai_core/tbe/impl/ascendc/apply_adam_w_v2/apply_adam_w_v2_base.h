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
 * \file apply_adam_w_v2_base.h
 * \brief
 */

#ifndef APPLYADAM_W_V2_BASE_H
#define APPLYADAM_W_V2_BASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace ApplyAdamWV2 {
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BYTE_ONE_BLOCK = 32;
constexpr int32_t BLOCK_SIZE_FOR_FLOAT32 = 8;
constexpr int32_t IN_BUFFER_NUM = 5;
constexpr int32_t OUT_BUFFER_NUM = 4;
constexpr int32_t VAR_ORDER_IN_LOCAL_TENSOR = 0;
constexpr int32_t EXP_AVG_ORDER_IN_LOCAL_TENSOR = 1;
constexpr int32_t EXP_AVG_SQ_ORDER_IN_LOCAL_TENSOR = 2;
constexpr int32_t MAX_GRAD_NORM_ORDER_IN_LOCAL_TENSOR = 3;
constexpr int32_t GRAD_NORM_ORDER_IN_LOCAL_TENSOR = 4;
constexpr int32_t MAX_GRAD_NORM_ORDER_IN_OUT_LOCAL_TENSOR = 3;
}  // namespace ApplyAdamWV2

#endif  // APPLYADAM_W_V2_BASE_H