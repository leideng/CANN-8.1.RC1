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
 * \file layernorm_grad_tiling.h
 * \brief
 */
#ifndef LIB_NORMALIZATION_LAYERNORM_GRAD_TILING_H
#define LIB_NORMALIZATION_LAYERNORM_GRAD_TILING_H

#include "graph/tensor.h"
#include "layernorm_grad_tilingdata.h"
namespace AscendC {
constexpr uint32_t LAYERNORM_GRAD_ONE_BLOCK_SIZE_OF_FLOAT = 8;
constexpr uint32_t LAYERNORM_GRAD_B32_BYTE_SIZE = 4;
constexpr uint32_t LAYERNORM_GRAD_B16_BYTE_SIZE = 2;
constexpr uint32_t LAYERNORM_GRAD_THREE_TIMES = 3;
constexpr uint32_t LAYERNORM_GRAD_TWO_TIMES = 2;
constexpr uint32_t LAYERNORM_GRAD_REUSE_FLOAT_BUF_NUM = 4;
constexpr uint32_t LAYERNORM_GRAD_HALF_BUF_NUM = 9;
constexpr uint32_t LAYERNORM_GRAD_FLOAT_BUF_NUM = 6;
constexpr uint32_t LAYERNORM_GRAD_B32_DATA_NUM_PER_BLOCK = 8;
constexpr uint32_t LAYERNORM_GRAD_B16_DATA_NUM_PER_BLOCK = 16;

void GetLayerNormGradMaxMinTmpSize(const ge::Shape &srcShape, const uint32_t typeSize, const bool isReuseSource,
    uint32_t &maxValue, uint32_t &minValue);

void GetLayerNormGradNDTilingInfo(const ge::Shape srcShape, const uint32_t stackBufferSize,
    const uint32_t typeSize, const bool isReuseSource, optiling::LayerNormGradTiling &tiling);
}
#endif // LIB_NORMALIZATION_LAYERNORM_GRAD_TILING_H
