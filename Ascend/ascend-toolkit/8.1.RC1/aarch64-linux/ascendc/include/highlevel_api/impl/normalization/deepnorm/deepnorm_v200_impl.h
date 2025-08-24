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
 * \file deepnorm_v200_impl.h
 * \brief
 */
#ifndef IMPL_NORMALIZATION_DEEPNORM_DEEPNORM_V200_IMPL_H
#define IMPL_NORMALIZATION_DEEPNORM_DEEPNORM_V200_IMPL_H

#include "kernel_tensor.h"

namespace AscendC {
namespace DeepNormAPI {
constexpr uint32_t BASIC_BLOCK_HLENGTH = 64;    // when isBasicBlock, H must be N * 64
constexpr uint32_t BASIC_BLOCK_BSLENGTH = 8;    // when isBasicBlock, B*S must be N * 8
constexpr uint32_t FLOAT_PER_BLOCK = 8;
constexpr uint8_t HALF_REPEAT_STRIDE = 4;       // DEFAULT_REPEAT_STRIDE / sizeof(half)
constexpr float SQRT_EXPONENT = -0.5;
// use this function only when bsLength is 8 * n (n > 0)
__aicore__ inline void DeepNormBasicBlockTransdata(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const uint32_t bsLength, const uint32_t hLength)
{
    const uint32_t hbase = BASIC_BLOCK_HLENGTH;
    const uint32_t bsbase = BASIC_BLOCK_BSLENGTH;
    const uint32_t offset = FLOAT_PER_BLOCK;
    __ubuf__ float* dstAddr = (__ubuf__ float*)dst.GetPhyAddr();
    __ubuf__ float* srcAddr = (__ubuf__ float*)src.GetPhyAddr();
    // for transdata, if repeatTimes == 1, both repeatstride must be 0
    uint8_t base = (bsLength % 16 == 0) ? 2 : 1;     // how many groups of 8 number can brcb in one transdata cmd
    uint8_t repeatTimes = bsLength / (BASIC_BLOCK_BSLENGTH * base);
    uint16_t dstRepeatStride = (repeatTimes == 1) ? 0 : BASIC_BLOCK_HLENGTH * base;
    uint16_t srcRepeatStride = (repeatTimes == 1) ? 0 : base;

    // 16 num: repeatTimes: bsLength / 8,  dstRepStride: 64,  srcRepStride: 1
    // 8  num: repeatTimes: bsLength / 16, dstRepStride: 128, srcRepStride: 2
    TransDataTo5HDParams params(false, false, repeatTimes, dstRepeatStride, srcRepeatStride);

    SetMaskNorm();
    ResetMask();

    // use transdata to simulate broadcast  ->  each num 64 times
    uint64_t dstList[NCHW_CONV_ADDR_LIST_SIZE];
    uint64_t srcList[NCHW_CONV_ADDR_LIST_SIZE];
    constexpr uint32_t step = 2;
    constexpr uint32_t transdataOffset = 8;
    constexpr uint32_t range = NCHW_CONV_ADDR_LIST_SIZE / step;
    if (bsLength % 16 == 0) { // b * s = n * 16
        for (uint32_t i = 0; i < range; i++) {
            dstList[i * step] = (uint64_t)dst[i * hbase].GetPhyAddr();
            dstList[i * step + 1] = (uint64_t)dst[i * hbase + transdataOffset * hbase].GetPhyAddr();
            srcList[i] = (uint64_t)src.GetPhyAddr();
            srcList[i + transdataOffset] = (uint64_t)src[bsbase].GetPhyAddr();
        }
    } else {  // b * s = n * 8
        // when use 8 num in transdata, only dst[0, 2, 4, 6, 8, 10, 12, 14] value matters.
        for (uint32_t i = 0; i < range; i++) {
            dstList[i * step] = (uint64_t)dst[i * hbase].GetPhyAddr();
            dstList[i * step + 1] = (uint64_t)dst[i * hbase + offset].GetPhyAddr();
            srcList[i] = (uint64_t)src.GetPhyAddr();
            srcList[i + transdataOffset] = (uint64_t)src.GetPhyAddr();
        }
    }
    TransDataTo5HD<float>(dstList, srcList, params);
    PipeBarrier<PIPE_V>();

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, bsLength * BASIC_BLOCK_HLENGTH);
    // 8 FP32 A -> 64 FP32 A
    Adds<float, false>(dst, dst, 0, MASK_PLACEHOLDER, 1, {1, 0, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();
}

// use brcb for calculation while calculating variance    inputMean: outputMean  inputX: input inputX / tmpTensorA
__aicore__ inline void DeepNormVarianceBasicBlockByBrcb(const LocalTensor<float>& inputX,
    const LocalTensor<float>& inputMean, const DeepNormTiling& tiling, const DeepNormParams<float>& params)
{
    const uint8_t num = tiling.hLength / BASIC_BLOCK_HLENGTH;     // isBasicBlock gurantee hLength = n * 64

    // meanX = broadcast inputMean(B*S) -> (B*S*H)
    // assume B*S*H is 2*8*128, then only broadcast to B*S*64, part 64~128 will reuse part 0~64
    DeepNormBasicBlockTransdata(params.tempTensorC, inputMean, tiling.bsCurLength, tiling.hLength);

    BinaryRepeatParams binaryParams;
    binaryParams.dstRepStride = num * DEFAULT_REPEAT_STRIDE;
    binaryParams.src0RepStride = num * DEFAULT_REPEAT_STRIDE;

    SetVectorMask<float, MaskMode::COUNTER>(0, tiling.bshCurLength / num);
    for (uint32_t i = 0; i < num; i++) {
        // subX = inputX - meanX
        Sub<float, false>(params.tempTensorB[i * BASIC_BLOCK_HLENGTH], inputX[i * BASIC_BLOCK_HLENGTH],
            params.tempTensorC, MASK_PLACEHOLDER, 1, binaryParams);
    }
    PipeBarrier<PIPE_V>();
    SetVectorMask<float, MaskMode::COUNTER>(0, tiling.bshCurLength);
}

// xSubMean: tmpTensorB
__aicore__ inline void DeepNormOutputBasicBlockByBrcb(const LocalTensor<float>& xSubMean, const DeepNormTiling& tiling,
    const DeepNormParams<float>& params)
{
    const uint8_t num = tiling.hLength / BASIC_BLOCK_HLENGTH;     // isBasicBlock gurantee hLength = n * 64
    const UnaryRepeatParams unaryParams;
    BinaryRepeatParams binaryParams;
    binaryParams.dstRepStride = num * DEFAULT_REPEAT_STRIDE;
    binaryParams.src1RepStride = num * DEFAULT_REPEAT_STRIDE;

    // var: broadcast expX(B*S) -> expX(B*S*H)
    // assume B*S*H is 2*8*128, then only broadcast to B*S*64, part 64~128 will reuse part 0~64
    DeepNormBasicBlockTransdata(params.tempTensorC, params.tempTensorA, tiling.bsCurLength, tiling.hLength);

    SetVectorMask<float, MaskMode::COUNTER>(0, tiling.bshCurLength / num);
    for (uint32_t i = 0; i < num; i++) {
        // resX = expX * subX
        Mul<float, false>(params.tempTensorA[i * BASIC_BLOCK_HLENGTH], params.tempTensorC,
            xSubMean[i * BASIC_BLOCK_HLENGTH], MASK_PLACEHOLDER, 1, binaryParams);
    }
    PipeBarrier<PIPE_V>();

    // copy to expected output: tmpTensorC
    SetVectorMask<float, MaskMode::COUNTER>(0, tiling.bshCurLength);
    Adds<float, false>(params.tempTensorC, params.tempTensorA, 0.0, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
}
} // namespace DeepNormAPI
} // namespace AscendC
#endif // IMPL_NORMALIZATION_DEEPNORM_DEEPNORM_V200_IMPL_H
