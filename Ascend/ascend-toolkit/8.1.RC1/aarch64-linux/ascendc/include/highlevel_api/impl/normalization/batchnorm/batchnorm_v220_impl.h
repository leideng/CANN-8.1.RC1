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
 * \file batchnorm_v220_impl.h
 * \brief
 */
#ifndef IMPL_NORMALIZATION_BATCHNORM_BATCHNORM_V220_IMPL_H
#define IMPL_NORMALIZATION_BATCHNORM_BATCHNORM_V220_IMPL_H

#include "kernel_tensor.h"
#include "kernel_tiling/kernel_tiling.h"

namespace AscendC {
constexpr uint32_t FLOAT_BLOCK_NUM_V220 = 8;
constexpr uint32_t BRC_ADDS_LOOP = 7;
constexpr uint32_t BASIC_BLOCK_LEN_V220 = 64;

__aicore__ inline void BrcFirstBlockByAdds(const LocalTensor<float>& dst, const uint32_t repeat,
    const uint32_t firstOffset, UnaryRepeatParams& addsUnaryParams, const BatchNormParams<float>& params)
{
    for (uint32_t m = 0; m < repeat; m++) {
        for (uint32_t i = 0; i < params.oriBloop; i++) {
            Adds<float, false>(dst[firstOffset + m * firstOffset], dst, 0, MASK_PLACEHOLDER, MAX_REPEAT_TIMES,
                addsUnaryParams);
        }
        if (params.oriBTail) {
            Adds<float, false>(dst[firstOffset + m * firstOffset], dst, 0, MASK_PLACEHOLDER, (uint8_t)params.oriBTail,
                addsUnaryParams);
        }
    }
    PipeBarrier<PIPE_V>();
    ResetMask();
    addsUnaryParams.srcBlkStride = DEFAULT_BLK_STRIDE;
    for (uint32_t m = 0; m < (params.basicLoop - 1); m++) {
        for (uint32_t i = 0; i < params.oriBloop; i++) {
            Adds<float, false>(dst[BASIC_BLOCK_LEN_V220 + m * BASIC_BLOCK_LEN_V220 + i * params.oriBTmpLoopOffset],
                dst[i * params.oriBTmpLoopOffset], 0, MASK_PLACEHOLDER, MAX_REPEAT_TIMES, addsUnaryParams);
        }
        if (params.oriBTail) {
            Adds<float, false>(dst[BASIC_BLOCK_LEN_V220 + m * BASIC_BLOCK_LEN_V220 + params.oriBTmpTailOffset],
                dst[params.oriBTmpTailOffset], 0, MASK_PLACEHOLDER, (uint8_t)params.oriBTail, addsUnaryParams);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <bool isBasicBlock = false>
__aicore__ inline void BrcFirstDimByBrcb(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const BatchNormTiling& tiling, const BatchNormParams<float>& params)
{
    BrcbRepeatParams repeatParams;
    repeatParams.dstBlkStride = (uint16_t)tiling.shCurLengthBlockNum;
    repeatParams.dstRepStride = tiling.shCurLength * FLOAT_BLOCK_NUM_V220 / FLOAT_BLOCK_NUM_V220;

    // brcb: 1 FP32 A -> 1 block contains 8 FP32 A, after 8 blocks, do the same to the next FP32 B
    Brcb(dst, src, (uint8_t)params.brcRepeatTimes, repeatParams);
    PipeBarrier<PIPE_V>();
    // add: 8 FP32 A (1 block) -> 64 FP32 A (8 blocks) ->shCurLength FP32 A
    SetVectorMask<float, MaskMode::NORMAL>(FLOAT_BLOCK_NUM_V220);
    UnaryRepeatParams addsUnaryParams;
    addsUnaryParams.dstBlkStride = DEFAULT_BLK_STRIDE;
    addsUnaryParams.srcBlkStride = 0;
    addsUnaryParams.dstRepStride = (uint8_t)tiling.shCurLengthBlockNum;
    addsUnaryParams.srcRepStride = (uint8_t)tiling.shCurLengthBlockNum;
    BrcFirstBlockByAdds(dst, BRC_ADDS_LOOP, FLOAT_BLOCK_NUM_V220, addsUnaryParams, params);
}
} // namespace AscendC
#endif // IMPL_NORMALIZATION_BATCHNORM_BATCHNORM_V220_IMPL_H
