/* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_flashv3_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_V220_SOFTMAX_FLASHV3_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_V220_SOFTMAX_FLASHV3_IMPL_H

#include "softmax_impl.h"

namespace AscendC {

__aicore__ inline void SoftmaxFlashV3ReduceSumImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const LocalTensor<float>& rowMeanLocal, const LocalTensor<float>& rowMeanGlobalTmp,
    const LocalTensor<float>& meanTmp, const struct ReduceLastND& reduceParam, const uint32_t& baseK,
    const uint32_t& reduceSize)
{
    const uint32_t splitCount = reduceParam.originalSrcK / FLOAT_REPEAT_SIZE;
    const uint32_t tailSrcK = reduceParam.originalSrcK % FLOAT_REPEAT_SIZE;
    const uint16_t srcRepeatStride = reduceParam.srcK / FLOAT_NUM_PER_BLK;

    for (uint32_t i = 0; i < reduceParam.originalSrcM; i++) {
        BlockReduceSum<float, false>(dst[FLOAT_REPEAT_SIZE * i], src[i * reduceParam.srcK], FLOAT_NUM_PER_BLK,
            MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
    }
    uint8_t remainRepeat = splitCount - FLOAT_NUM_PER_BLK;
    if (remainRepeat != 0) {
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < reduceParam.originalSrcM; j++) {
        Add<float, false>(dst[j * FLOAT_REPEAT_SIZE], src[SOFTMAX_FLOAT_SPECIAL_BLOCKREDUCE_LEN + j * reduceParam.srcK],
            dst[j * FLOAT_REPEAT_SIZE], 1, remainRepeat, { 1, 1, 1, 0, DEFAULT_REPEAT_STRIDE, 0 });
        }
    }
    if (tailSrcK != 0) {
      PipeBarrier<PIPE_V>();
      TailAddImpl(dst, src, reduceParam, tailSrcK, srcRepeatStride, splitCount);
      ResetMask();  
    }
    PipeBarrier<PIPE_V>();
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, reduceParam.originalSrcM * FLOAT_REPEAT_SIZE);
    BlockReduceSum<float, false>(rowMeanLocal, dst, 1, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE); // M * 8
    PipeBarrier<PIPE_V>();
    UnaryRepeatParams unaryParams;
    if (baseK != 0) {
        Muls<float, false>(rowMeanLocal, rowMeanLocal, static_cast<float>(1.0f / baseK), MASK_PLACEHOLDER, 1, unaryParams); // M * 8
    }
    PipeBarrier<PIPE_V>();
    BlockReduceSum<float, false>(rowMeanGlobalTmp, rowMeanLocal, 1, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE); // M * 1
    PipeBarrier<PIPE_V>();
    const uint32_t repeat = (reduceParam.originalSrcM + BRCB_BROADCAST_NUMBER - 1) / BRCB_BROADCAST_NUMBER;
    Brcb(dst, rowMeanGlobalTmp, (uint8_t)repeat, { 1, BRCB_BROADCAST_NUMBER }); // M * 8
    PipeBarrier<PIPE_V>();
    SetVectorMask<float, MaskMode::COUNTER>(0, reduceParam.originalSrcM * reduceParam.dstK);
    Copy<float, false>(rowMeanGlobalTmp, dst, MASK_PLACEHOLDER, 1, {1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();
    SetVectorMask<float, MaskMode::COUNTER>(0, reduceSize);
    Muls<float, false>(meanTmp, rowMeanGlobalTmp, static_cast<float>(1.0f / FLOAT_NUM_PER_BLK), MASK_PLACEHOLDER, 1, unaryParams); // M * 8
    SetMaskNorm();
    ResetMask();
}

__aicore__ inline void ModifyInputImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const LocalTensor<float>& meanTmp, const LocalTensor<float>& workLocal, const LocalTensor<float>& tmpBuffer2,
    const ReduceLastND& reduceParam, const SoftMaxTiling& tiling, const SoftMaxParams& params, const uint32_t& reduceSize)
{
    const LocalTensor<float>& tmpBuffer1 = workLocal[tiling.splitSize + tiling.reduceSize]; // splitM * 64
    const LocalTensor<float>& rowMeanLocal = workLocal[tiling.splitSize + tiling.splitM * 64 + tiling.reduceSize];
    const LocalTensor<float>& rowMeanGlobalTmp = workLocal[tiling.splitSize + tiling.splitM * 64 + tiling.reduceSize * 2];
    const uint32_t baseK = reduceParam.originalSrcK / params.splitMeanCnt;

    SoftmaxFlashV3ReduceSumImpl(tmpBuffer1, src, rowMeanLocal, rowMeanGlobalTmp, meanTmp, reduceParam, baseK, reduceSize);
    PipeBarrier<PIPE_V>();
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, reduceSize);
    Sub<float, false>(rowMeanGlobalTmp, meanTmp, rowMeanLocal, MASK_PLACEHOLDER, 1,
        { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
    UnaryRepeatParams unaryParams;
    Muls<float, false>(rowMeanGlobalTmp, rowMeanGlobalTmp, static_cast<float>(params.alpha / (1.0f - params.alpha)),
        MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < reduceParam.originalSrcM; i++) {
        Brcb(tmpBuffer1, rowMeanGlobalTmp[i * params.splitMeanCnt], 1, {1, DEFAULT_REPEAT_STRIDE});
        PipeBarrier<PIPE_V>();
        SetVectorMask<float, MaskMode::COUNTER>(0, baseK);
        const CopyRepeatParams copyRepeatParams = {1, 0, (uint16_t)(baseK / FLOAT_NUM_PER_BLK), 1};
        Copy<float, false>(tmpBuffer2, tmpBuffer1, MASK_PLACEHOLDER, params.splitMeanCnt, copyRepeatParams);
        PipeBarrier<PIPE_V>();
        SetVectorMask<float, MaskMode::COUNTER>(0, reduceParam.srcK);
        Sub<float, false>(src[i * reduceParam.srcK], src[i * reduceParam.srcK], tmpBuffer2, MASK_PLACEHOLDER, 1,
            { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();
    }
    SetMaskNorm();
    ResetMask();
}
__aicore__ inline void ModifyMaxImpl(const LocalTensor<float>& maxTensor, const LocalTensor<float>& meanTensor,
    const LocalTensor<float>& src, const LocalTensor<float>& meanTmp, const LocalTensor<float>& maxTmp,
    const LocalTensor<float>& shiftVal, const SoftMaxParams& params, const uint32_t& reduceSize)
{
    float scalar = params.alpha / (1 - params.alpha);

    Sub(shiftVal, meanTmp, meanTensor, reduceSize);
    PipeBarrier<PIPE_V>();
    Muls(shiftVal, shiftVal, scalar, reduceSize);
    PipeBarrier<PIPE_V>();
    Add(maxTensor, maxTmp, shiftVal, reduceSize);
}

__aicore__ inline void SoftmaxFlashV3NoUpdateImpl(const LocalTensor<half>& dst, const LocalTensor<float>& meanTensor,
    const LocalTensor<float>& expSumTensor, const LocalTensor<float>& maxTensor, const LocalTensor<half>& src,
    const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling, const SoftMaxParams& params,
    const ReduceLastND& reduceParam, const uint32_t& offset1, const uint32_t& offset2, const uint32_t& splitSize,
    const uint32_t& reduceSize)
{
    const LocalTensor<float>& tmpBuffer0 = workLocal;
    const LocalTensor<float>& meanTmp = workLocal[tiling.splitSize];
    const LocalTensor<float>& tmpBuffer1 = workLocal[tiling.splitSize + tiling.reduceSize]; // splitM * 64
    const LocalTensor<float>& maxTmp = workLocal[tiling.splitSize + tiling.splitM * 64 + tiling.reduceSize];
    const LocalTensor<float>& shiftCurr = workLocal[tiling.splitSize + tiling.splitM * 64 + tiling.reduceSize * 2];
    const LocalTensor<float>& tmpBuffer2 = workLocal[tiling.splitSize + tiling.splitM * 64 + tiling.reduceSize * 3]; // srcK

    Cast(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, splitSize);
    PipeBarrier<PIPE_V>();
    ModifyInputImpl(tmpBuffer0, tmpBuffer0, meanTmp, workLocal, tmpBuffer2, reduceParam, tiling, params, reduceSize);
    PipeBarrier<PIPE_V>();
    Copy(meanTensor[offset2], meanTmp, reduceSize, 1, {1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();
    ResetMask();
    // rowmax
    NewReduceMaxLastNDImpl(maxTmp, tmpBuffer0, tmpBuffer1, reduceParam);
    PipeBarrier<PIPE_V>();
    // shiftCurr = (rowMeanGlobal - mean) * (a / 1 - a), max = maxTmp + shiftCurr
    ModifyMaxImpl(maxTensor[offset2], meanTensor[offset2], tmpBuffer0, meanTmp, maxTmp, shiftCurr, params, reduceSize);
    PipeBarrier<PIPE_V>();
    // max' = max - shiftCurr
    Sub(shiftCurr, maxTensor[offset2], shiftCurr, reduceSize);
    PipeBarrier<PIPE_V>();
    // y = x - max'
    GenericSubNDImpl(tmpBuffer0, tmpBuffer0, shiftCurr, reduceParam.originalSrcM, tiling.srcK, tiling.reduceK);
    PipeBarrier<PIPE_V>();
    Exp(tmpBuffer0, tmpBuffer0, splitSize);
    PipeBarrier<PIPE_V>();
    Cast(dst[offset1], tmpBuffer0, FLOAT2HALF_ROUND_MODE, splitSize);
    PipeBarrier<PIPE_V>();
    NewReduceSumLastNDImpl(expSumTensor[offset2], tmpBuffer0, tmpBuffer1, reduceParam);
}

__aicore__ inline void SoftmaxFlashV3TailImpl(const SoftMaxTiling& tiling, ReduceLastND& reduceParam,
    uint32_t& offset1, uint32_t& offset2, uint32_t& splitSize, uint32_t& reduceSize)
{
    offset2 = tiling.rangeM * tiling.reduceSize;
    offset1 = tiling.rangeM * tiling.splitSize;
    splitSize = tiling.tailSplitSize;
    reduceSize = tiling.tailReduceSize;
    reduceParam.originalSrcM = tiling.tailM;
    reduceParam.srcM = tiling.tailM;
    reduceParam.dstM = tiling.tailM;
}

__aicore__ inline void SoftmaxFlashV3NoUpdateExtImpl(const LocalTensor<half>& dst, const LocalTensor<float>& meanTensor,
    const LocalTensor<float>& expSumTensor, const LocalTensor<float>& maxTensor, const LocalTensor<half>& src,
    const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling,
    const SoftMaxParams& params, ReduceLastND& reduceParam)
{
    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint32_t splitSize = tiling.splitSize;
    uint32_t reduceSize = tiling.reduceSize;

    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        SoftmaxFlashV3NoUpdateImpl(dst, meanTensor, expSumTensor, maxTensor, src, workLocal, tiling, params, reduceParam,
            offset1, offset2, splitSize, reduceSize);
        offset1 += tiling.splitSize;
        offset2 += tiling.reduceSize;
        PipeBarrier<PIPE_V>();
    }

    if (tiling.tailM != 0) {
        SoftmaxFlashV3TailImpl(tiling, reduceParam, offset1, offset2, splitSize, reduceSize);
        SoftmaxFlashV3NoUpdateImpl(dst, meanTensor, expSumTensor, maxTensor, src, workLocal, tiling, params, reduceParam,
            offset1, offset2, splitSize, reduceSize);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void SoftmaxFlashV3UpdateMeanImpl(const LocalTensor<float>& meanTensor,
    const LocalTensor<float>& inMeanTensor, const LocalTensor<float>& meanTmp,
    const SoftMaxParams& params, const uint32_t& reduceSize)
{
    // mean = (inmean * (n - 1) + mean) / n
    Muls(meanTensor, inMeanTensor, params.loopCnt - 1.0f, reduceSize);
    PipeBarrier<PIPE_V>();
    Add(meanTensor, meanTensor, meanTmp, reduceSize);
    PipeBarrier<PIPE_V>();
    Muls(meanTensor, meanTensor, static_cast<float>(1.0f / params.loopCnt), reduceSize);
}

__aicore__ inline void SoftmaxFlashV3UpdateImpl(const LocalTensor<half>& dst, const LocalTensor<float>& meanTensor,
    const LocalTensor<float>& expSumTensor, const LocalTensor<float>& maxTensor, const LocalTensor<half>& src,
    const LocalTensor<half>& expMaxTensor, const LocalTensor<float>& inMeanTensor, const LocalTensor<float>& inExpSumTensor,
    const LocalTensor<float>& inMaxTensor, const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling,
    const SoftMaxParams& params, const ReduceLastND& reduceParam, const uint32_t& offset1, const uint32_t& offset2,
    const uint32_t& splitSize, const uint32_t& reduceSize)
{
    const LocalTensor<float>& tmpBuffer0 = workLocal;
    const LocalTensor<float>& meanTmp = workLocal[tiling.splitSize];
    const LocalTensor<float>& tmpBuffer1 = workLocal[tiling.splitSize + tiling.reduceSize];
    const LocalTensor<float>& maxTmp = workLocal[tiling.splitSize + tiling.splitM * 64 + tiling.reduceSize];
    const LocalTensor<float>& shiftCurr = workLocal[tiling.splitSize + tiling.splitM * 64 + tiling.reduceSize * 2];
    const LocalTensor<float>& shiftPrev = workLocal[tiling.splitSize + tiling.splitM * 64 + tiling.reduceSize * 3];
    const LocalTensor<float>& tmpBuffer2 = workLocal[tiling.splitSize + tiling.splitM * 64 + tiling.reduceSize * 4];

    Cast(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, splitSize);
    PipeBarrier<PIPE_V>();
    ModifyInputImpl(tmpBuffer0, tmpBuffer0, meanTmp, workLocal, tmpBuffer2, reduceParam, tiling, params, reduceSize);
    PipeBarrier<PIPE_V>();
    SoftmaxFlashV3UpdateMeanImpl(meanTensor[offset2], inMeanTensor[offset2], meanTmp, params, reduceSize);
    PipeBarrier<PIPE_V>();
    // rowmax
    NewReduceMaxLastNDImpl(maxTmp, tmpBuffer0, tmpBuffer1, reduceParam);
    PipeBarrier<PIPE_V>();
    // shiftCurr = (rowMeanGlobal - mean) * (a / 1 - a)
    ModifyMaxImpl(maxTensor[offset2], meanTensor[offset2], tmpBuffer0, meanTmp, maxTmp, shiftCurr, params, reduceSize);
    PipeBarrier<PIPE_V>();
    // shiftPrev = (inmean - mean) * (a / 1 - a)
    ModifyMaxImpl(maxTmp, meanTensor[offset2], tmpBuffer0, inMeanTensor[offset2], inMaxTensor[offset2], shiftPrev,
        params, reduceSize);
    PipeBarrier<PIPE_V>();
    // max = max(maxTmp + shiftCurr, inmax + shiftPrev)
    Max(maxTensor[offset2], maxTensor[offset2], maxTmp, reduceSize);
    PipeBarrier<PIPE_V>();
    // em = inmax - max
    Sub(maxTmp, inMaxTensor[offset2], maxTensor[offset2], reduceSize);
    PipeBarrier<PIPE_V>();
    // em' = em + shiftPrev
    Add(maxTmp, shiftPrev, maxTmp, reduceSize);
    PipeBarrier<PIPE_V>();
    Exp(maxTmp, maxTmp, reduceSize);
    PipeBarrier<PIPE_V>();
    Mul(expSumTensor[offset2], maxTmp, inExpSumTensor[offset2], reduceSize);
    PipeBarrier<PIPE_V>();
    // max' = max - shiftCurr
    Sub(shiftCurr, maxTensor[offset2], shiftCurr, reduceSize);
    PipeBarrier<PIPE_V>();
    // y = x - max'
    GenericSubNDImpl(tmpBuffer0, tmpBuffer0, shiftCurr, reduceParam.originalSrcM, tiling.srcK, tiling.reduceK);
    PipeBarrier<PIPE_V>();
    Exp(tmpBuffer0, tmpBuffer0, splitSize);
    PipeBarrier<PIPE_V>();
    Cast(dst[offset1], tmpBuffer0, FLOAT2HALF_ROUND_MODE, splitSize);
    PipeBarrier<PIPE_V>();
    NewReduceSumLastNDImpl(meanTmp, tmpBuffer0, tmpBuffer1, reduceParam);
    PipeBarrier<PIPE_V>();
    Add(expSumTensor[offset2], expSumTensor[offset2], meanTmp, reduceSize);
    PipeBarrier<PIPE_V>();
    BroadCastLastImpl(tmpBuffer0, maxTmp,
        { tiling.reduceM, B16_BYTE_SIZE * DEFAULT_REPEAT_STRIDE, tiling.reduceM, tiling.reduceK });
    PipeBarrier<PIPE_V>();
    Cast(expMaxTensor[offset2 * B16_BYTE_SIZE], tmpBuffer0, FLOAT2HALF_ROUND_MODE, reduceSize * B16_BYTE_SIZE);
}

__aicore__ inline void SoftmaxFlashV3NDExtImpl(const LocalTensor<half>& dst, const LocalTensor<float>& meanTensor,
    const LocalTensor<float>& expSumTensor, const LocalTensor<float>& maxTensor, const LocalTensor<half>& src,
    const LocalTensor<half>& expMaxTensor, const LocalTensor<float>& inMeanTensor, const LocalTensor<float>& inExpSumTensor,
    const LocalTensor<float>& inMaxTensor, const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling,
    const SoftMaxParams& params, ReduceLastND& reduceParam)
{
    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint32_t splitSize = tiling.splitSize;
    uint32_t reduceSize = tiling.reduceSize;

    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        SoftmaxFlashV3UpdateImpl(dst, meanTensor, expSumTensor, maxTensor, src, expMaxTensor, inMeanTensor, inExpSumTensor,
            inMaxTensor, workLocal, tiling, params, reduceParam, offset1, offset2, splitSize, reduceSize);
        offset1 += tiling.splitSize;
        offset2 += tiling.reduceSize;
        PipeBarrier<PIPE_V>();
    }

    if (tiling.tailM != 0) {
        SoftmaxFlashV3TailImpl(tiling, reduceParam, offset1, offset2, splitSize, reduceSize);
        SoftmaxFlashV3UpdateImpl(dst, meanTensor, expSumTensor, maxTensor, src, expMaxTensor, inMeanTensor, inExpSumTensor,
            inMaxTensor, workLocal, tiling, params, reduceParam, offset1, offset2, splitSize, reduceSize);
        PipeBarrier<PIPE_V>();
    }
}
template <typename T, typename U, bool isUpdate = false, bool isBasicBlock = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftmaxFlashV3Process(const LocalTensor<T>& dstTensor, const LocalTensor<U>& meanTensor,
    const LocalTensor<U>& expSumTensor, const LocalTensor<U>& maxTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<T>& expMaxTensor, const LocalTensor<U>& inMeanTensor, const LocalTensor<U>& inExpSumTensor,
    const LocalTensor<U>& inMaxTensor, const LocalTensor<float>& workLocal, const LastAxisShapeND& originalSrcShape,
    const SoftMaxTiling& tiling, const SoftMaxParams& params)
{
    ASCENDC_ASSERT((params.srcK >= SOFTMAX_FLOAT_SPECIAL_BLOCKREDUCE_LEN),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check shape in SoftmaxFlashV3, it should be greater than 512.");});
    ReduceLastND reduceParam = { tiling.splitM, originalSrcShape.k, tiling.splitM, tiling.splitK, tiling.reduceM,
        tiling.reduceK };
    if constexpr (!isUpdate) {
        SoftmaxFlashV3NoUpdateExtImpl(dstTensor, meanTensor, expSumTensor, maxTensor, srcTensor, workLocal,
            tiling, params, reduceParam);
    } else {
        SoftmaxFlashV3NDExtImpl(dstTensor, meanTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor, inMeanTensor,
            inExpSumTensor, inMaxTensor, workLocal, tiling, params, reduceParam);
    }
}
}
#endif // IMPL_ACTIVATION_SOFTMAX_V220_SOFTMAX_FLASHV3_IMPL_H