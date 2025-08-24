/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_common_utils.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_SOFTMAX_COMMON_UTILS_H
#define IMPL_ACTIVATION_SOFTMAX_SOFTMAX_COMMON_UTILS_H
#include "kernel_pop_stack_buffer.h"
#include "lib/activation/softmax_utils.h"

namespace AscendC {

constexpr uint8_t SOFTMAX_BASIC_TILE_NUM = 8;
constexpr uint8_t SOFTMAX_COMPUTE_DIM = 2;
constexpr uint8_t SOFTMAXGRAD_COMPUTE_DIM = 3;
constexpr uint8_t SOFTMAXFLASH_COMPUTE_DIM = 4;
constexpr uint8_t SOFTMAX_INNER_SHAPE_DIM = 2; // default ND ,2 dimension
constexpr uint32_t FLOAT_NUM_PER_BLK = ONE_BLK_SIZE / B32_BYTE_SIZE;
constexpr uint32_t HALF_NUM_PER_BLK = ONE_BLK_SIZE / B16_BYTE_SIZE;
constexpr uint32_t HALF_REPEAT_STRIDE = DEFAULT_REPEAT_STRIDE / B16_BYTE_SIZE;
constexpr uint32_t SCALAR_STACK_DEPTH = 8;
constexpr uint32_t SOFTMAX_SHAPE_NZ_BASIC_COUNT = 16;
constexpr uint32_t SOFTMAX_NZ_TILING_NEEDBLOCK = 3;
constexpr uint32_t SOFTMAX_MAX_REPEAT_STRIDE = MAX_REPEAT_TIMES * DEFAULT_REPEAT_STRIDE;
constexpr uint32_t SOFTMAX_MAX_REPEAT_CLC_FLOAT_NUM = MAX_REPEAT_TIMES * FLOAT_REPEAT_SIZE;
constexpr uint32_t SOFTMAX_MAX_REPEAT_CLC_HALF_NUM = MAX_REPEAT_TIMES * HALF_REPEAT_SIZE;
constexpr uint32_t SOFTMAX_SPECIAL_BASICBLOCK_LEN = FLOAT_REPEAT_SIZE * SOFTMAX_BASIC_TILE_NUM * SOFTMAX_COMPUTE_DIM;
constexpr uint32_t SOFTMAX_SUB_DIV_ROW_COLUMN_SIZE = 192;
constexpr uint32_t SOFTMAX_FLOAT_SPECIAL_BLOCKREDUCE_LEN = DEFAULT_BLOCK_SIZE * HALF_FACTOR;

struct SoftMaxParams {
    uint32_t srcM{ 0 };
    uint32_t srcK{ 0 };
    uint32_t oriSrcM{ 0 };
    uint32_t oriSrcK{ 0 };
    uint32_t loopCnt{ 1 };
    uint32_t splitMeanCnt{ 8 };
    float alpha{ 0.9375 };
};
struct LastAxisShapeND {
    uint32_t m;
    uint32_t k;
};

struct ReduceLastND {
    uint32_t originalSrcM;
    uint32_t originalSrcK;
    uint32_t srcM;
    uint32_t srcK;
    uint32_t dstM;
    uint32_t dstK;
};

struct BroadCastLastND {
    uint32_t dstM;
    uint32_t dstK;
    uint32_t srcM;
    uint32_t srcK;
};

struct SoftMaxShapeInfo {
    uint32_t srcM{ 0 };
    uint32_t srcK{ 0 };
    uint32_t oriSrcM{ 0 };
    uint32_t oriSrcK{ 0 };
};

}; // namespace AscendC
#endif // IMPL_ACTIVATION_SOFTMAX_SOFTMAX_COMMON_UTILS_H