/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file topk_common_utils.h
 * \brief
 */
#ifndef IMPL_SORT_TOPK_TOPK_COMMON_UTILS_H
#define IMPL_SORT_TOPK_TOPK_COMMON_UTILS_H

#if __CCE_AICORE__ >= 200
namespace {
constexpr uint16_t MIN_SORT32_SIZE = 32;
constexpr uint16_t MIN_RPSORT16_SIZE = 16;
constexpr uint32_t BUF_LIST_SIZE = 2;
constexpr uint32_t MRG_MAX_ARRAY_SIZE = 15;
constexpr uint32_t MRGSORT_VALID_QUEUE = 4;
constexpr uint32_t MRGSORT_VALID_TWO = 2;
constexpr uint32_t MRGSORT_VALID_TWO_OFFSET = 769;
constexpr uint32_t TWO = 2;
constexpr uint32_t THREE = 3;
constexpr uint32_t FOUR = 4;
constexpr uint32_t FIVE = 5;
constexpr uint32_t SIX = 6;
constexpr uint32_t SEVEN = 7;
constexpr uint32_t EIGHT = 8;
constexpr uint32_t NINE = 9;
constexpr uint32_t TWELVE = 12;
constexpr uint32_t SIXTEEN = 16;
constexpr uint32_t THIRTY_TWO = 32;
constexpr uint32_t FORTYEIGHT = 48;
constexpr uint32_t VREDUCEV2_HALF_MASK = 128;
constexpr uint32_t VREDUCEV2_FOUR_BYTE_MASK = 64;
constexpr uint32_t SRC1_STACK_TENSORSIZE = 10;
constexpr uint32_t SRC1_STACK_VAL_OFFSET = 16;
constexpr uint32_t TOPK_INNER_ALIGN_LEN = 32;
constexpr uint32_t TOPK_NORMAL_INNER_MAX_HALF_LEN = 2048;
constexpr uint32_t TOPK_NSMALL_INNER_LEN = 32;
constexpr uint32_t TOPK_NORMAL_INNER_MAX_LEN = 4096;
}  // namespace

namespace AscendC {
struct TopKInfo {
    int32_t outter = 1;
    int32_t inner;  // inner = 32-byte alignment of n
    int32_t n;      // actual length of the tensor
};

enum class TopKMode {
    TOPK_NORMAL,
    TOPK_NSMALL,
};
}
#endif

#endif  // IMPL_SORT_TOPK_TOPK_COMMON_UTILS_H