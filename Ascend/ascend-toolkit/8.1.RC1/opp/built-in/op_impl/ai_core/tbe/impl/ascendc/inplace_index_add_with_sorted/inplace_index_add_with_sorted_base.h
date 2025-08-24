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
 * \file inplace_index_add_with_sorted_base.h
 * \brief
 */

#ifndef INPLACE_INDEX_ADD_WITH_SORTED_BASE_H_
#define INPLACE_INDEX_ADD_WITH_SORTED_BASE_H_

#include "kernel_operator.h"
#define IS_CAST_FLOAT ((is_same<T, half>::value) || (is_same<T, bfloat16_t>::value))
using namespace AscendC;

template <typename Tp, Tp v>
struct integral_constant {
  static constexpr Tp value = v;
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template <typename, typename>
struct is_same : public false_type {};
template <typename Tp>
struct is_same<Tp, Tp> : public true_type {};

constexpr int64_t BUFFER_NUM = 1; // tensor num for each queue
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t INDEX_UB_NUM = 1536;

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 a, T2 b) {
    a = int64_t(a);
    b = int64_t(b);
    return T1(b == 0 ? a : (a + b - 1) / b);
};

template <typename T1, typename T2>
__aicore__ inline T1 CeilAlignA2B(T1 a, T2 b) {
    a = int64_t(a);
    b = int64_t(b);
    return T1(b == 0 ? a : CeilDiv(a, b) * b);
};

#endif  // INPLACE_INDEX_ADD_WITH_SORTED_BASE_H_