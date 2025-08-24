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
 * \file basic_block_config.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCHMATMUL_V2_BASIC_BLOCK_CONFIG_H
#define WEIGHT_QUANT_BATCHMATMUL_V2_BASIC_BLOCK_CONFIG_H

#include "../tool.h"
#include "../weight_quant_batch_matmul_v2_constant.h"
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "lib/matmul_intf.h"


namespace WeightQuantBatchMatmulV2 {

constexpr static uint16_t WEIGHT_F16_UB_NZ_STRIDE = 65;

struct WqmmConfig {
    bool aTrans;
    bool bTrans;
    QuantType antiQuantType;
    bool hasAntiQuantOffset;
    QuantType quantType;
    CubeFormat weightFormat;
};

struct BasicBlockOffsetParam {
    uint32_t mL1Size;
    uint32_t kaL1Size;
    uint32_t kbL1Size;
    uint32_t nL1Size;

    uint64_t mOffset;
    uint64_t nOffset;

    uint64_t mSize;
    uint64_t kSize;
    uint64_t nSize;
};

struct VecAntiQuantConfig {
    int32_t ubMte2BufferNum = 2;
    int32_t ubMte2InnerSize = 512;
};

struct L1DbConfig {
    uint32_t aF16L1DbOffset;
    uint32_t biasL1DbOffset;
    uint32_t weightF16L1DbOffset;
};

struct UbConsumeConfig {
    uint32_t l1RequireVfComputeRealK;
    uint32_t l1RequireVfComputeRealN;
    uint32_t kWeightLowBitUbOffset;
    uint32_t nWeightLowBitUbOffset;
};

struct L1ConsumeConfig {
    uint32_t l1SplitTwoVecExternalOffset;
    uint32_t l1RealExternalLen;
};

}
#endif // WEIGHT_QUANT_BATCHMATMUL_V2_BASIC_BLOCK_CONFIG_H