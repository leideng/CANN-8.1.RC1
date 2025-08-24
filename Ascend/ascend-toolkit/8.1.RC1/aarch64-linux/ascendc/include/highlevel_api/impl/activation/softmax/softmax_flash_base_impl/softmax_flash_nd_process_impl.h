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
 * \file softmax_flash_nd_process_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASH_ND_PROCESS_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASH_ND_PROCESS_IMPL_H
#include "softmax_flash_basic_block_impl.h"

namespace AscendC {

template <typename T, bool isBasicBlock = false>
__aicore__ inline void SoftmaxFlashNDImpl(const LocalTensor<T> &dst, const LocalTensor<T> &sumTensor,
    const LocalTensor<T> &maxTensor, const LocalTensor<T> &src, const LocalTensor<T> &expMaxTensor,
    const LocalTensor<T> &inSumTensor, const LocalTensor<T> &inMaxTensor, const LastAxisShapeND &originalSrcShape,
    const SoftMaxTiling &tiling)
{
    LocalTensor<float> workLocal;
    PopStackBuffer<float, TPosition::LCM>(workLocal);
    uint32_t workLocalSize = workLocal.GetSize();

    const LocalTensor<float> &tmpBuffer0 = workLocal[0];
    const LocalTensor<float> &tmpBuffer1 = workLocal[tiling.splitSize];
    const LocalTensor<float> &tmpBuffer2 = workLocal[tiling.splitSize + tiling.splitSize];
    const LocalTensor<float> &reduceSumBuffer = workLocal[tiling.splitSize + tiling.splitSize + tiling.reduceSize];
    const LocalTensor<float> &tmpBuffer4 =
        workLocal[tiling.splitSize + tiling.splitSize + tiling.reduceSize + tiling.reduceSize];
    const LocalTensor<float> &inSumTmp =
        workLocal[tiling.splitSize + tiling.splitSize + tiling.reduceSize + tiling.reduceSize + tiling.reduceSize];
    const LocalTensor<float> &inMaxTmp = workLocal[0];

    ReduceLastND reduceParam = { tiling.splitM, originalSrcShape.k, tiling.splitM,
        tiling.splitK, tiling.reduceM,     tiling.reduceK };
    BroadCastLastND brcParam = { tiling.splitM, tiling.splitK, tiling.reduceM, tiling.reduceK };
    uint32_t offset1 = 0;
    uint32_t offset2 = 0;

#if __CCE_AICORE__ == 220 && ASCENDC_CPU_DEBUG == 0
    if constexpr (isBasicBlock) {
        SoftmaxFlashBasicBlock<T>(dst, sumTensor, maxTensor, src, expMaxTensor, inSumTensor, inMaxTensor, workLocal,
            tiling);
    } else
#endif
    {
        for (uint32_t i = 0; i < tiling.rangeM; i++) {
            offset2 = i * tiling.reduceSize;
            offset1 = i * tiling.splitSize;
            Cast(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, tiling.splitSize);
            PipeBarrier<PIPE_V>();
            ReduceMaxLastNDImpl(tmpBuffer4, tmpBuffer0, reduceSumBuffer, reduceParam);
            PipeBarrier<PIPE_V>();
            BroadCastLastImpl(tmpBuffer1, tmpBuffer4, brcParam);
            PipeBarrier<PIPE_V>();
            Sub(tmpBuffer1, tmpBuffer0, tmpBuffer1, tiling.splitSize);
            PipeBarrier<PIPE_V>();
            Exp(tmpBuffer1, tmpBuffer1, tiling.splitSize);
            PipeBarrier<PIPE_V>();
            ReduceSumLastNDImpl(reduceSumBuffer, tmpBuffer1, tmpBuffer2, reduceParam);
            PipeBarrier<PIPE_V>();

            Cast(inMaxTmp, inMaxTensor[offset2], RoundMode::CAST_NONE, tiling.reduceSize);
            PipeBarrier<PIPE_V>();
            Max(tmpBuffer2, inMaxTmp, tmpBuffer4, tiling.reduceSize);
            PipeBarrier<PIPE_V>();
            Cast(maxTensor[offset2], tmpBuffer2, FLOAT2HALF_ROUND_MODE, tiling.reduceSize);
            PipeBarrier<PIPE_V>();
            Sub(tmpBuffer4, tmpBuffer4, tmpBuffer2, tiling.reduceSize);
            PipeBarrier<PIPE_V>();
            Exp(tmpBuffer4, tmpBuffer4, tiling.reduceSize);

            Sub(inMaxTmp, inMaxTmp, tmpBuffer2, tiling.reduceSize);
            PipeBarrier<PIPE_V>();
            Exp(inMaxTmp, inMaxTmp, tiling.reduceSize);

            Cast(inSumTmp, inSumTensor[offset2], RoundMode::CAST_NONE, tiling.reduceSize);
            PipeBarrier<PIPE_V>();
            Mul(inMaxTmp, inMaxTmp, inSumTmp, tiling.reduceSize);
            Mul(reduceSumBuffer, tmpBuffer4, reduceSumBuffer, tiling.reduceSize);
            PipeBarrier<PIPE_V>();
            Add(inSumTmp, inMaxTmp, reduceSumBuffer, tiling.reduceSize);
            PipeBarrier<PIPE_V>();
            Div(inMaxTmp, inMaxTmp, inSumTmp, tiling.reduceSize);
            PipeBarrier<PIPE_V>();
            Cast(expMaxTensor[offset2], inMaxTmp, FLOAT2HALF_ROUND_MODE, tiling.reduceSize);
            Cast(sumTensor[offset2], inSumTmp, FLOAT2HALF_ROUND_MODE, tiling.reduceSize);

            Div(tmpBuffer4, tmpBuffer4, inSumTmp, tiling.reduceSize);
            PipeBarrier<PIPE_V>();
            BroadCastLastImpl(tmpBuffer0, tmpBuffer4, brcParam);
            PipeBarrier<PIPE_V>();
            Mul(tmpBuffer1, tmpBuffer1, tmpBuffer0, tiling.splitSize);
            PipeBarrier<PIPE_V>();
            Cast(dst[offset1], tmpBuffer1, FLOAT2HALF_ROUND_MODE, tiling.splitSize);
        }
    }
    PipeBarrier<PIPE_V>();
    if (tiling.tailM != 0) {
        offset2 = tiling.rangeM * tiling.reduceSize;
        offset1 = tiling.rangeM * tiling.splitSize;

        Cast(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, tiling.tailSplitSize);
        PipeBarrier<PIPE_V>();
        ReduceMaxLastNDImpl(tmpBuffer4, tmpBuffer0, reduceSumBuffer, reduceParam);
        PipeBarrier<PIPE_V>();
        BroadCastLastImpl(tmpBuffer1, tmpBuffer4, brcParam);
        PipeBarrier<PIPE_V>();

        Sub(tmpBuffer1, tmpBuffer0, tmpBuffer1, tiling.tailSplitSize);
        PipeBarrier<PIPE_V>();
        Exp(tmpBuffer1, tmpBuffer1, tiling.tailSplitSize);
        PipeBarrier<PIPE_V>();
        ReduceSumLastNDImpl(reduceSumBuffer, tmpBuffer1, tmpBuffer2, reduceParam);
        PipeBarrier<PIPE_V>();

        Cast(inMaxTmp, inMaxTensor[offset2], RoundMode::CAST_NONE, tiling.tailReduceSize);
        PipeBarrier<PIPE_V>();
        Max(tmpBuffer2, inMaxTmp, tmpBuffer4, tiling.tailReduceSize);
        PipeBarrier<PIPE_V>();
        Cast(maxTensor[offset2], tmpBuffer2, FLOAT2HALF_ROUND_MODE, tiling.tailReduceSize);
        PipeBarrier<PIPE_V>();
        Sub(tmpBuffer4, tmpBuffer4, tmpBuffer2, tiling.tailReduceSize);
        PipeBarrier<PIPE_V>();
        Exp(tmpBuffer4, tmpBuffer4, tiling.tailReduceSize);

        Sub(inMaxTmp, inMaxTmp, tmpBuffer2, tiling.tailReduceSize);
        PipeBarrier<PIPE_V>();
        Exp(inMaxTmp, inMaxTmp, tiling.tailReduceSize);
        Cast(inSumTmp, inSumTensor[offset2], RoundMode::CAST_NONE, tiling.tailReduceSize);
        PipeBarrier<PIPE_V>();
        Mul(inMaxTmp, inMaxTmp, inSumTmp, tiling.tailReduceSize);
        Mul(reduceSumBuffer, tmpBuffer4, reduceSumBuffer, tiling.tailReduceSize);
        PipeBarrier<PIPE_V>();
        Add(inSumTmp, inMaxTmp, reduceSumBuffer, tiling.tailReduceSize);
        PipeBarrier<PIPE_V>();
        Div(inMaxTmp, inMaxTmp, inSumTmp, tiling.tailReduceSize);
        PipeBarrier<PIPE_V>();
        Cast(expMaxTensor[offset2], inMaxTmp, FLOAT2HALF_ROUND_MODE, tiling.tailReduceSize);
        Cast(sumTensor[offset2], inSumTmp, FLOAT2HALF_ROUND_MODE, tiling.tailReduceSize);

        Div(tmpBuffer4, tmpBuffer4, inSumTmp, tiling.tailReduceSize);
        PipeBarrier<PIPE_V>();
        BroadCastLastImpl(tmpBuffer0, tmpBuffer4, brcParam);
        PipeBarrier<PIPE_V>();
        Mul(tmpBuffer1, tmpBuffer1, tmpBuffer0, tiling.tailSplitSize);
        PipeBarrier<PIPE_V>();
        Cast(dst[offset1], tmpBuffer1, FLOAT2HALF_ROUND_MODE, tiling.tailSplitSize);
    }
}


__aicore__ inline void SoftmaxFlashNDImpl(const LocalTensor<float> &dst, const LocalTensor<float> &sumTensor,
    const LocalTensor<float> &maxTensor, const LocalTensor<float> &src, const LocalTensor<float> &expMaxTensor,
    const LocalTensor<float> &inSumTensor, const LocalTensor<float> &inMaxTensor, const LocalTensor<float> &workLocal,
    const LastAxisShapeND &originalSrcShape, const SoftMaxTiling &tiling)
{
    const LocalTensor<float> &tmpBuffer0 = workLocal[0];
    const LocalTensor<float> &tmpBuffer1 = workLocal[tiling.splitSize];
    const LocalTensor<float> &tmpBuffer2 = workLocal[tiling.splitSize + tiling.splitSize];
    const LocalTensor<float> &reduceSumBuffer = workLocal[tiling.splitSize + tiling.splitSize + tiling.reduceSize];
    const LocalTensor<float> &tmpBuffer4 =
        workLocal[tiling.splitSize + tiling.splitSize + tiling.reduceSize + tiling.reduceSize];
    const LocalTensor<float> &inSumTmp =
        workLocal[tiling.splitSize + tiling.splitSize + tiling.reduceSize + tiling.reduceSize + tiling.reduceSize];
    const LocalTensor<float> &inMaxTmp = workLocal[0];

    const ReduceLastND reduceMainParam = { tiling.splitM, originalSrcShape.k, tiling.splitM,
                                           tiling.splitK, tiling.reduceM,     tiling.reduceK };
    const ReduceLastND reduceTailParam = { tiling.tailM,  originalSrcShape.k, tiling.tailM,
                                           tiling.splitK, tiling.tailM,       tiling.reduceK };
    const BroadCastLastND mainBrcParam = { tiling.splitM, tiling.splitK, tiling.reduceM, tiling.reduceK };
    const BroadCastLastND tailBrcParam = { tiling.tailM, tiling.splitK, tiling.tailM, tiling.reduceK };

    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        offset2 = i * tiling.reduceSize;
        offset1 = i * tiling.splitSize;
        PipeBarrier<PIPE_V>();
        ReduceMaxLastNDImpl(tmpBuffer4, src[offset1], reduceSumBuffer, reduceMainParam);
        PipeBarrier<PIPE_V>();
        BroadCastLastImpl(tmpBuffer1, tmpBuffer4, mainBrcParam);
        PipeBarrier<PIPE_V>();
        Sub(tmpBuffer1, src[offset1], tmpBuffer1, tiling.splitSize);
        PipeBarrier<PIPE_V>();
        Exp(tmpBuffer1, tmpBuffer1, tiling.splitSize);
        PipeBarrier<PIPE_V>();
        ReduceSumLastNDImpl(reduceSumBuffer, tmpBuffer1, tmpBuffer2, reduceMainParam);
        PipeBarrier<PIPE_V>();

        DataCopy(inMaxTmp, inMaxTensor[offset2], tiling.reduceSize);
        PipeBarrier<PIPE_V>();
        Max(tmpBuffer2, inMaxTmp, tmpBuffer4, tiling.reduceSize);
        PipeBarrier<PIPE_V>();
        DataCopy(maxTensor[offset2], tmpBuffer2, tiling.reduceSize);
        PipeBarrier<PIPE_V>();
        Sub(tmpBuffer4, tmpBuffer4, tmpBuffer2, tiling.reduceSize);
        PipeBarrier<PIPE_V>();
        Exp(tmpBuffer4, tmpBuffer4, tiling.reduceSize);

        Sub(inMaxTmp, inMaxTmp, tmpBuffer2, tiling.reduceSize);
        PipeBarrier<PIPE_V>();
        Exp(inMaxTmp, inMaxTmp, tiling.reduceSize);

        DataCopy(inSumTmp, inSumTensor[offset2], tiling.reduceSize);
        PipeBarrier<PIPE_V>();
        Mul(inMaxTmp, inMaxTmp, inSumTmp, tiling.reduceSize);
        Mul(reduceSumBuffer, tmpBuffer4, reduceSumBuffer, tiling.reduceSize);
        PipeBarrier<PIPE_V>();
        Add(sumTensor[offset2], inMaxTmp, reduceSumBuffer, tiling.reduceSize);
        PipeBarrier<PIPE_V>();
        Div(expMaxTensor[offset2], inMaxTmp, sumTensor[offset2], tiling.reduceSize);
        PipeBarrier<PIPE_V>();
        DataCopy(sumTensor[offset2], sumTensor[offset2], tiling.reduceSize);

        Div(tmpBuffer4, tmpBuffer4, sumTensor[offset2], tiling.reduceSize);
        PipeBarrier<PIPE_V>();
        BroadCastLastImpl(tmpBuffer0, tmpBuffer4, mainBrcParam);
        PipeBarrier<PIPE_V>();
        Mul(dst[offset1], tmpBuffer1, tmpBuffer0, tiling.splitSize);
    }

    PipeBarrier<PIPE_V>();
    if (tiling.tailM != 0) {
        offset2 = tiling.rangeM * tiling.reduceSize;
        offset1 = tiling.rangeM * tiling.splitSize;

        PipeBarrier<PIPE_V>();
        ReduceMaxLastNDImpl(tmpBuffer4, src[offset1], reduceSumBuffer, reduceTailParam);
        PipeBarrier<PIPE_V>();
        BroadCastLastImpl(tmpBuffer1, tmpBuffer4, tailBrcParam);
        PipeBarrier<PIPE_V>();

        Sub(tmpBuffer1, src[offset1], tmpBuffer1, tiling.tailSplitSize);
        PipeBarrier<PIPE_V>();
        Exp(tmpBuffer1, tmpBuffer1, tiling.tailSplitSize);
        PipeBarrier<PIPE_V>();
        ReduceSumLastNDImpl(reduceSumBuffer, tmpBuffer1, tmpBuffer2, reduceTailParam);
        PipeBarrier<PIPE_V>();

        DataCopy(inMaxTmp, inMaxTensor[offset2], tiling.tailReduceSize);
        PipeBarrier<PIPE_V>();
        Max(tmpBuffer2, inMaxTmp, tmpBuffer4, tiling.tailReduceSize);
        PipeBarrier<PIPE_V>();
        DataCopy(maxTensor[offset2], tmpBuffer2, tiling.tailReduceSize);
        PipeBarrier<PIPE_V>();
        Sub(tmpBuffer4, tmpBuffer4, tmpBuffer2, tiling.tailReduceSize);
        PipeBarrier<PIPE_V>();
        Exp(tmpBuffer4, tmpBuffer4, tiling.tailReduceSize);

        Sub(inMaxTmp, inMaxTmp, tmpBuffer2, tiling.tailReduceSize);
        PipeBarrier<PIPE_V>();
        Exp(inMaxTmp, inMaxTmp, tiling.tailReduceSize);
        DataCopy(inSumTmp, inSumTensor[offset2], tiling.tailReduceSize);
        PipeBarrier<PIPE_V>();
        Mul(inMaxTmp, inMaxTmp, inSumTmp, tiling.tailReduceSize);
        Mul(reduceSumBuffer, tmpBuffer4, reduceSumBuffer, tiling.tailReduceSize);
        PipeBarrier<PIPE_V>();
        Add(sumTensor[offset2], inMaxTmp, reduceSumBuffer, tiling.tailReduceSize);
        PipeBarrier<PIPE_V>();
        Div(expMaxTensor[offset2], inMaxTmp, sumTensor[offset2], tiling.tailReduceSize);

        Div(tmpBuffer4, tmpBuffer4, sumTensor[offset2], tiling.tailReduceSize);
        PipeBarrier<PIPE_V>();
        BroadCastLastImpl(tmpBuffer0, tmpBuffer4, tailBrcParam);
        PipeBarrier<PIPE_V>();
        Mul(dst[offset1], tmpBuffer1, tmpBuffer0, tiling.tailSplitSize);
    }
}

template <typename T, bool isBasicBlock = false>
__aicore__ inline void SoftmaxFlashPostProcess(const LocalTensor<T> &dstTensor, const LocalTensor<T> &sumTensor,
    const LocalTensor<T> &maxTensor, const LocalTensor<T> &srcTensor, const LocalTensor<T> &expMaxTensor,
    const LocalTensor<T> &inSumTensor, const LocalTensor<T> &inMaxTensor, const LocalTensor<float> &workLocal,
    const LastAxisShapeND &originalSrcShape, const SoftMaxTiling &tiling, bool isUpdate = false,
    const SoftMaxShapeInfo &softmaxShapeInfo = {})
{
    const uint32_t elementNumPerBlk = ONE_BLK_SIZE / sizeof(T);
    uint32_t workLocalSize = workLocal.GetSize();
    if constexpr (sizeof(T) == sizeof(half)) {
        if (!isUpdate) {
            SoftMaxNDImpl<T, T>(dstTensor, sumTensor, maxTensor, srcTensor, workLocal, originalSrcShape, tiling);
        } else {
            SoftmaxFlashNDImpl<T, isBasicBlock>(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor, inSumTensor,
                inMaxTensor, originalSrcShape, tiling);
        }
    } else {
        if (!isUpdate) {
            SoftMaxNDImpl<T, T>(dstTensor, sumTensor, maxTensor, srcTensor, workLocal, originalSrcShape, tiling);
        } else {
#if __CCE_AICORE__ == 220 && ASCENDC_CPU_DEBUG == 0
            if constexpr (isBasicBlock) {
                SoftmaxFlashBasicBlockFloat(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor, inSumTensor,
                    inMaxTensor, workLocal, tiling);
            } else
#endif
            {
                SoftmaxFlashNDImpl(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor, inSumTensor, inMaxTensor,
                    workLocal, originalSrcShape, tiling);
            }
        }
    }
}

template <bool isBasicBlock = false>
__aicore__ inline void SoftmaxFlashNDImpl(const LocalTensor<half> &dst, const LocalTensor<float> &sumTensor,
    const LocalTensor<float> &maxTensor, const LocalTensor<half> &src, const LocalTensor<half> &expMaxTensor,
    const LocalTensor<float> &inSumTensor, const LocalTensor<float> &inMaxTensor, const LocalTensor<float> &workLocal,
    const LastAxisShapeND &originalSrcShape, const SoftMaxTiling &tiling)
{
    const LocalTensor<float> &tmpBuffer0 = workLocal[0];
    const LocalTensor<float> &tmpBuffer1 = workLocal[tiling.splitSize];
    const LocalTensor<float> &inMaxTmp = workLocal[tiling.splitSize + tiling.splitSize];
    const LocalTensor<float> &reduceSumBuffer = workLocal[tiling.splitSize + tiling.splitSize + tiling.reduceSize];
    const LocalTensor<float> &tmpBuffer4 =
        workLocal[tiling.splitSize + tiling.splitSize + tiling.reduceSize + tiling.reduceSize];
    const LocalTensor<float> &inSumTmp =
        workLocal[tiling.splitSize + tiling.splitSize + tiling.reduceSize + tiling.reduceSize + tiling.reduceSize];

    ReduceLastND reduceParam = { tiling.splitM, originalSrcShape.k, tiling.splitM,
        tiling.splitK, tiling.reduceM,     tiling.reduceK };
    BroadCastLastND brcParam = { tiling.splitM, tiling.splitK, tiling.reduceM, tiling.reduceK };
    uint32_t offset1 = 0;
    uint32_t offset2 = 0;

#if __CCE_AICORE__ == 220 && ASCENDC_CPU_DEBUG == 0
    if constexpr (isBasicBlock) {
        SoftmaxFlashBasicBlock(dst, sumTensor, maxTensor, src, expMaxTensor, inSumTensor, inMaxTensor, workLocal,
            tiling);
    } else
#endif
    {
        for (uint32_t i = 0; i < tiling.rangeM; i++) {
            offset2 = i * tiling.reduceSize;
            offset1 = i * tiling.splitSize;
            Cast(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, tiling.splitSize);
            PipeBarrier<PIPE_V>();
            ReduceMaxLastNDImpl(tmpBuffer4, tmpBuffer0, reduceSumBuffer, reduceParam);
            PipeBarrier<PIPE_V>();
            BroadCastLastImpl(tmpBuffer1, tmpBuffer4, brcParam);
            PipeBarrier<PIPE_V>();
            Sub(tmpBuffer1, tmpBuffer0, tmpBuffer1, tiling.splitSize);
            PipeBarrier<PIPE_V>();
            Exp(tmpBuffer1, tmpBuffer1, tiling.splitSize);
            PipeBarrier<PIPE_V>();
            ReduceSumLastNDImpl(reduceSumBuffer, tmpBuffer1, inMaxTmp, reduceParam);
            PipeBarrier<PIPE_V>();

            DataCopy(inMaxTmp, inMaxTensor[offset2], tiling.reduceSize);
            PipeBarrier<PIPE_V>();
            Max(maxTensor[offset2], inMaxTmp, tmpBuffer4, tiling.reduceSize);
            PipeBarrier<PIPE_V>();

            Sub(tmpBuffer4, tmpBuffer4, maxTensor[offset2], tiling.reduceSize);
            PipeBarrier<PIPE_V>();
            Exp(tmpBuffer4, tmpBuffer4, tiling.reduceSize);

            Sub(inMaxTmp, inMaxTmp, maxTensor[offset2], tiling.reduceSize);
            PipeBarrier<PIPE_V>();
            Exp(inMaxTmp, inMaxTmp, tiling.reduceSize);

            DataCopy(inSumTmp, inSumTensor[offset2], tiling.reduceSize);

            PipeBarrier<PIPE_V>();
            Mul(inMaxTmp, inMaxTmp, inSumTmp, tiling.reduceSize);
            Mul(reduceSumBuffer, tmpBuffer4, reduceSumBuffer, tiling.reduceSize);
            PipeBarrier<PIPE_V>();
            Add(inSumTmp, inMaxTmp, reduceSumBuffer, tiling.reduceSize);
            PipeBarrier<PIPE_V>();
            Div(inMaxTmp, inMaxTmp, inSumTmp, tiling.reduceSize);
            PipeBarrier<PIPE_V>();
            DataCopy(sumTensor[offset2], inSumTmp, tiling.reduceSize);

            // 32B copy to 64B
            BroadCastLastImpl(tmpBuffer0, inMaxTmp,
                { tiling.reduceM, HALF_NUM_PER_BLK, tiling.reduceM, tiling.reduceK });
            PipeBarrier<PIPE_V>();
            Cast(expMaxTensor[offset2 * HALF_FACTOR], tmpBuffer0, FLOAT2HALF_ROUND_MODE,
                tiling.reduceSize * HALF_FACTOR);

            Div(tmpBuffer4, tmpBuffer4, inSumTmp, tiling.reduceSize);
            PipeBarrier<PIPE_V>();
            BroadCastLastImpl(tmpBuffer0, tmpBuffer4, brcParam);
            PipeBarrier<PIPE_V>();
            Mul(tmpBuffer1, tmpBuffer1, tmpBuffer0, tiling.splitSize);
            PipeBarrier<PIPE_V>();
            Cast(dst[offset1], tmpBuffer1, FLOAT2HALF_ROUND_MODE, tiling.splitSize);
        }
    }
    PipeBarrier<PIPE_V>();
    if (tiling.tailM != 0) {
        offset2 = tiling.rangeM * tiling.reduceSize;
        offset1 = tiling.rangeM * tiling.splitSize;

        Cast(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, tiling.tailSplitSize);
        PipeBarrier<PIPE_V>();
        ReduceMaxLastNDImpl(tmpBuffer4, tmpBuffer0, reduceSumBuffer, reduceParam);
        PipeBarrier<PIPE_V>();
        BroadCastLastImpl(tmpBuffer1, tmpBuffer4, brcParam);
        PipeBarrier<PIPE_V>();

        Sub(tmpBuffer1, tmpBuffer0, tmpBuffer1, tiling.tailSplitSize);
        PipeBarrier<PIPE_V>();
        Exp(tmpBuffer1, tmpBuffer1, tiling.tailSplitSize);
        PipeBarrier<PIPE_V>();
        ReduceSumLastNDImpl(reduceSumBuffer, tmpBuffer1, inMaxTmp, reduceParam);
        PipeBarrier<PIPE_V>();

        DataCopy(inMaxTmp, inMaxTensor[offset2], tiling.tailReduceSize);
        PipeBarrier<PIPE_V>();
        Max(maxTensor[offset2], inMaxTmp, tmpBuffer4, tiling.tailReduceSize);
        PipeBarrier<PIPE_V>();

        Sub(tmpBuffer4, tmpBuffer4, maxTensor[offset2], tiling.tailReduceSize);
        PipeBarrier<PIPE_V>();
        Exp(tmpBuffer4, tmpBuffer4, tiling.tailReduceSize);

        Sub(inMaxTmp, inMaxTmp, maxTensor[offset2], tiling.tailReduceSize);
        PipeBarrier<PIPE_V>();
        Exp(inMaxTmp, inMaxTmp, tiling.tailReduceSize);
        DataCopy(inSumTmp, inSumTensor[offset2], tiling.tailReduceSize);
        PipeBarrier<PIPE_V>();
        Mul(inMaxTmp, inMaxTmp, inSumTmp, tiling.tailReduceSize);
        Mul(reduceSumBuffer, tmpBuffer4, reduceSumBuffer, tiling.tailReduceSize);
        PipeBarrier<PIPE_V>();
        Add(inSumTmp, inMaxTmp, reduceSumBuffer, tiling.tailReduceSize);
        PipeBarrier<PIPE_V>();
        Div(inMaxTmp, inMaxTmp, inSumTmp, tiling.tailReduceSize);
        PipeBarrier<PIPE_V>();

        // 32B copy to 64B
        BroadCastLastImpl(tmpBuffer0, inMaxTmp,
            { tiling.reduceM, FLOAT_NUM_PER_BLK * B16_BYTE_SIZE, tiling.reduceM, tiling.reduceK });
        PipeBarrier<PIPE_V>();
        Cast(expMaxTensor[offset2 * HALF_FACTOR], tmpBuffer0, FLOAT2HALF_ROUND_MODE,
            tiling.tailReduceSize * B16_BYTE_SIZE);
        DataCopy(sumTensor[offset2], inSumTmp, tiling.tailReduceSize);

        Div(tmpBuffer4, tmpBuffer4, inSumTmp, tiling.tailReduceSize);
        PipeBarrier<PIPE_V>();
        BroadCastLastImpl(tmpBuffer0, tmpBuffer4, brcParam);
        PipeBarrier<PIPE_V>();
        Mul(tmpBuffer1, tmpBuffer1, tmpBuffer0, tiling.tailSplitSize);
        PipeBarrier<PIPE_V>();
        Cast(dst[offset1], tmpBuffer1, FLOAT2HALF_ROUND_MODE, tiling.tailSplitSize);
    }
}

}
#endif // IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASH_ND_PROCESS_IMPL_H