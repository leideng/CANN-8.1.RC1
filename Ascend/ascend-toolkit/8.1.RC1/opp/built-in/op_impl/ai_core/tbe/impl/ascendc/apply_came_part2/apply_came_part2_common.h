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
 * \file apply_came_part2_common.h
 * \brief
 */
#ifndef _APPLY_CAME_PART2_COMMON_H_
#define _APPLY_CAME_PART2_COMMON_H_

#include "kernel_operator.h"

const int32_t BUFFER_NUM = 1;
const int32_t BLOCK_SIZE = 32;
const int32_t CONFUSION_TRANSPOSE_ALIGHNED_NUM = 16;
const int32_t CALC_SIZE = 256;
const int32_t HALF_CALC_SIZE = CALC_SIZE / 2;
const int32_t INT64_PER_BLOCK = 4;
const int32_t WORKSPACE_ALIGNED_SIZE = 512;
const int32_t FLOAT_SIZE = 4;

template <typename T1, typename T2>
__aicore__ inline T1 CeilAlign(T1 a, T2 b) {
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b * b;
}

template <typename T1, typename T2> 
__aicore__ inline T1 CeilDiv(T1 a, T2 b) {
    if (b == 0) {
        return 0;
    }

    return (a + b - 1) / b;
};

template <typename T1, typename T2> 
__aicore__ inline T1 Max(T1 a, T2 b) {
    return a > b ? a : b;
};

#endif // _APPLY_CAME_PART2_COMMON_H_
