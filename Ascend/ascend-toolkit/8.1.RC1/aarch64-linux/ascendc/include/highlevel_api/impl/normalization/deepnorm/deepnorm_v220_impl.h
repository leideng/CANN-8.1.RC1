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
 * \file deepnorm_v220_impl.h
 * \brief
 */
#ifndef IMPL_NORMALIZATION_DEEPNORM_DEEPNORM_V220_IMPL_H
#define IMPL_NORMALIZATION_DEEPNORM_DEEPNORM_V220_IMPL_H

#include "kernel_tensor.h"

namespace AscendC {
namespace DeepNormAPI {
constexpr uint32_t BASIC_BLOCK_HLENGTH = 64;    // when isBasicBlock, H must be N * 64
constexpr uint32_t BASIC_BLOCK_BSLENGTH = 8;    // when isBasicBlock, B*S must be N * 8
constexpr uint32_t FLOAT_PER_BLOCK = 8;
constexpr uint8_t HALF_REPEAT_STRIDE = 4;       // DEFAULT_REPEAT_STRIDE / sizeof(half)
constexpr float SQRT_EXPONENT = -0.5;

// brcb 1 num to 64 num. BasicBlock h is 64 * n, therefore always use the same 64 num.
// Note: dst must not be the same tensor with src tensor! When total num > 64, dst = src will have wrong calculation
__aicore__ inline void DeepNormBasicBlockVbrcb(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const uint32_t bsLength)
{
    constexpr uint16_t brcbDstBlkStride = 8;   // leave 8 blocks (64 num) space for one num
    constexpr uint16_t brcbDstRepStride = 64;
    constexpr uint16_t addSrcBlkStride = 0;    // in Adds, one block used 8 times
    const uint8_t repeatTimes = bsLength / 8;
    SetMaskNorm();
    ResetMask();

    BrcbRepeatParams brcbParams(brcbDstBlkStride, brcbDstRepStride);
    // brcb: 1 FP32 A -> 1 block contains 8 FP32 A, after 8 blocks, do the same to the next FP32 B
    Brcb<float>(dst, src, repeatTimes, brcbParams);
    PipeBarrier<PIPE_V>();

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, bsLength * BASIC_BLOCK_HLENGTH);

    // add: 8 FP32 A (1 block) -> 64 FP32 A (8 blocks)
    Adds<float, false>(dst, dst, 0, MASK_PLACEHOLDER, 1,
        {DEFAULT_BLK_STRIDE, addSrcBlkStride, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();
}

// use brcb for calculation while calculating variance    inputMean: outputMean  inputX: input inputX / tmpTensorA
__aicore__ inline void DeepNormVarianceBasicBlockByBrcb(const LocalTensor<float>& inputX,
    const LocalTensor<float>& inputMean, const DeepNormTiling& tiling, const DeepNormParams<float>& params)
{
    const uint8_t num = tiling.hLength / BASIC_BLOCK_HLENGTH;     // isBasicBlock gurantee hLength = n * 64

    // meanX = broadcast inputMean(B*S) -> (B*S*H)
    // assume B*S*H is 2*8*128, then only broadcast to B*S*64, part 64~128 will reuse part 0~64
    DeepNormBasicBlockVbrcb(params.tempTensorC, inputMean, tiling.bsCurLength);

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
    DeepNormBasicBlockVbrcb(params.tempTensorC, params.tempTensorA, tiling.bsCurLength);

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
#endif // IMPL_NORMALIZATION_DEEPNORM_DEEPNORM_V220_IMPL_H
