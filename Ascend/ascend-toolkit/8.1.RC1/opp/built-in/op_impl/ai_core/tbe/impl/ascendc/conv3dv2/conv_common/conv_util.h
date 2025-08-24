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
 * \file conv_util.h
 * \brief
 */

#ifndef CONV_UTIL_H
#define CONV_UTIL_H

#include "kernel_utils.h"

using namespace AscendC;

namespace conv {
const static uint32_t K0_BIAS = 8;
const static uint32_t NUM_TWO = 2;
const static uint32_t BLOCK_L0_N = 16;
const static uint32_t BLOCK_L0_M = 16;
const static uint32_t DATA_COPY_OP_LEN = 16;
const static uint64_t C0_SIZE = 32;
const static uint64_t FRACTAL_SIZE = 512;
const static uint64_t BT_SIZE = 64;
const static uint64_t SIZE_OF_FP16 = 2;
const static uint64_t PAD_IDX_T = 2;
const static uint64_t PAD_IDX_B = 3;
const static uint64_t PAD_IDX_L = 0;
const static uint64_t PAD_IDX_R = 1;
const static uint64_t MAX_PAD_R = 255;
const static uint64_t FMAP_BATCH_DIM = 0;
const static uint64_t FMAP_CIN_DIM = 1;
const static uint64_t FMAP_H_DIM = 2;
const static uint64_t FMAP_W_DIM = 3;
const static uint64_t KERNEL_COUT_DIM = 0;
const static uint64_t KERNEL_H_DIM = 2;
const static uint64_t KERNEL_W_DIM = 3;
const static uint64_t OUTPUT_H_DIM = 2;
const static uint64_t OUTPUT_W_DIM = 3;
const static uint32_t BLOCK_SIZE = 512;
const static uint32_t AL1_BLOCK_SIZE = 32;
const static uint32_t BT_BLOCK_SIZE = 64;
const static uint64_t F8_NUM_IN_F16 = 2;
const static uint64_t MAX_UINT16 = 65535;

static __aicore__ inline size_t AlignB(uint64_t a, uint64_t b)
{
    return ((a + b - 1) / b) * b;
}

static __aicore__ inline size_t CeilDIV(uint64_t a, uint64_t b)
{
    return (a + b - 1) / b;
}

template <class Intf>
static __aicore__ inline size_t GetInputkInOneC0Block()
{
    return C0_SIZE / sizeof(typename Intf::FmapT);
}

enum class IterateOrder {
    ORDER_MTERFIRST = 0,
    ORDER_NTERFIRST,
    UNDEF,
};
}  // namespace conv
#endif  // __CONV3D_UTIL_H__
