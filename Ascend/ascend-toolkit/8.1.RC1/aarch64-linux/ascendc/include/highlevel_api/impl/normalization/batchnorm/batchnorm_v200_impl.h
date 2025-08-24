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
 * \file batchnorm_v200_impl.h
 * \brief
 */
#ifndef IMPL_NORMALIZATION_BATCHNORM_BATCHNORM_V200_IMPL_H
#define IMPL_NORMALIZATION_BATCHNORM_BATCHNORM_V200_IMPL_H

#include "kernel_tensor.h"
#include "kernel_tiling/kernel_tiling.h"

namespace AscendC {
constexpr uint32_t FLOAT_BLOCK_NUM_V200 = 8;
constexpr uint32_t TRANSDATA_ADDS_LOOP = 3;
constexpr uint32_t TRANSDATA_ADDS_OFFSET = 16;
constexpr uint32_t NCHW_ADDR_LIST_SIZE = 16;
constexpr uint32_t NCHW_LIST_SIZE_DIV_TWO = 8;
constexpr uint32_t BASIC_BLOCK_LEN_V200 = 64;

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
            Adds<float, false>(dst[BASIC_BLOCK_LEN_V200 + m * BASIC_BLOCK_LEN_V200 + i * params.oriBTmpLoopOffset],
                dst[i * params.oriBTmpLoopOffset], 0, MASK_PLACEHOLDER, MAX_REPEAT_TIMES, addsUnaryParams);
        }
        if (params.oriBTail) {
            Adds<float, false>(dst[BASIC_BLOCK_LEN_V200 + m * BASIC_BLOCK_LEN_V200 + params.oriBTmpTailOffset],
                dst[params.oriBTmpTailOffset], 0, MASK_PLACEHOLDER, (uint8_t)params.oriBTail, addsUnaryParams);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <bool isBasicBlock = false>
__aicore__ inline void BrcFirstDimByTransData(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const BatchNormTiling& tiling, const BatchNormParams<float>& batchnormParams)
{
    uint16_t dstRepStride = tiling.shCurLength * FLOAT_BLOCK_NUM_V200 / FLOAT_BLOCK_NUM_V200;
    uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
    uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
    for (int32_t n = 0; n < NCHW_ADDR_LIST_SIZE; n++) {
        srcLocalList[n] = (uint64_t)src.GetPhyAddr();
    }
    for (int32_t n = 0; n < NCHW_LIST_SIZE_DIV_TWO; n++) {
        dstLocalList[2 * n] = (uint64_t)dst[n * tiling.shCurLength].GetPhyAddr();
        dstLocalList[2 * n + 1] = (uint64_t)dst[n * tiling.shCurLength + FLOAT_BLOCK_NUM_V200].GetPhyAddr();
    }
    TransDataTo5HDParams params(false, false, (uint8_t)batchnormParams.brcRepeatTimes, dstRepStride, 1);
    if (batchnormParams.brcRepeatTimes == 1) {
        params.dstRepStride = 0;
        params.srcRepStride = 0;
    }
    TransDataTo5HD<float>(dstLocalList, srcLocalList, params);
    PipeBarrier<PIPE_V>();
    SetVectorMask<float, MaskMode::NORMAL>(TRANSDATA_ADDS_OFFSET);
    UnaryRepeatParams addsUnaryParams;
    addsUnaryParams.srcBlkStride = 0;
    addsUnaryParams.dstRepStride = (uint8_t)tiling.shCurLengthBlockNum;
    addsUnaryParams.srcRepStride = (uint8_t)tiling.shCurLengthBlockNum;
    BrcFirstBlockByAdds(dst, TRANSDATA_ADDS_LOOP, TRANSDATA_ADDS_OFFSET, addsUnaryParams, batchnormParams);
}
} // namespace AscendC
#endif // IMPL_NORMALIZATION_BATCHNORM_BATCHNORM_V200_IMPL_H
