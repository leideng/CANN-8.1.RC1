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
 * \file adaptive_avg_pool3d_grad_common.h
 * \brief
 */
#ifndef ADAPTIVE_AVG_POOL3D_GRAD_COMMON_H
#define ADAPTIVE_AVG_POOL3D_GRAD_COMMON_H


#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
using namespace AscendC;

__aicore__ inline uint64_t start_index(uint64_t a, uint64_t b, uint64_t c)
{
    ASSERT_MSG(b != 0, "Division by zero error!");
    return (a / b) * c + ((a % b) * c) / b;
}

__aicore__ inline uint64_t end_index(uint64_t a, uint64_t b, uint64_t c)
{
    ASSERT_MSG(b != 0, "Division by zero error!");
    return 1 + ((a + 1) * c - 1) / b;
}

#endif //ADAPTIVE_AVG_POOL3D_GRAD_COMMON_H