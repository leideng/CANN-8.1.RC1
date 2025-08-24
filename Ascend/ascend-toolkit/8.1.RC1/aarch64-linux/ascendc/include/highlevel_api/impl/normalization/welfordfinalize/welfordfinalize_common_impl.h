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
* \file welfordfinalize_common_impl.h
* \brief
*/
#ifndef IMPL_NORMALIZATION_WELFORDFINALIZE_COMMON_IMPL_H
#define IMPL_NORMALIZATION_WELFORDFINALIZE_COMMON_IMPL_H

#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tiling/kernel_tiling.h"

namespace AscendC 
{
constexpr uint32_t OUTPUT_SIZE = 8;
constexpr uint32_t B32_LEN = 256 / sizeof(float);
constexpr uint32_t OUTPUT_MASK_B32 = 254;

struct WelfordFinalizePara
{
    uint32_t rnLength;
    uint32_t abLength;
    uint32_t headCount;
    uint32_t headCountLength;
    uint32_t tailCount;
    uint32_t tailCountLength;
    float abRec;
    float rRec;
};

struct WelfordFinalizeTilingData
{
    uint32_t computeLength;
    uint32_t round;
    uint32_t tail;
};

template <typename T>
struct WelfordFinalizeTmpTensors
{
    __aicore__ WelfordFinalizeTmpTensors() {}
    LocalTensor<T> tempOutputMean;
    LocalTensor<T> tempOutputVariance;
    LocalTensor<T> tempMean;
    LocalTensor<T> tempVariance;
};

__aicore__ inline void welfordFinalizeOutputPre(const LocalTensor<float> &outputMean, const LocalTensor<float> &outputVariance,
    const LocalTensor<float> &tempOutputMean, const LocalTensor<float> &tempOutputVariance)
{
    Adds(outputMean, tempOutputMean, static_cast<float>(0), 1);
    Adds(outputVariance, tempOutputVariance, static_cast<float>(0), 1);
    PipeBarrier<PIPE_V>();

    SetMaskNorm();
    SetVectorMask<float>(0, OUTPUT_MASK_B32);

    Duplicate<float, false>(outputMean, 0, MASK_PLACEHOLDER, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();

    Duplicate<float, false>(outputVariance, 0, MASK_PLACEHOLDER, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
}

template <bool isReuseSource = false>
__aicore__ inline void WelfordFinalizeExeVariance(const LocalTensor<float> &inputMean, const LocalTensor<float> &inputVariance,
    const LocalTensor<float> &outputVariance, const LocalTensor<int32_t> &counts, const WelfordFinalizeTmpTensors<float> &tempTensors, 
    const WelfordFinalizePara &para, const uint32_t computeLength)
{
    LocalTensor<float> tempMean = tempTensors.tempMean;
    LocalTensor<float> tempVariance = tempTensors.tempVariance;
    
    Adds(tempVariance, tempMean, static_cast<float>(0), 1);
    PipeBarrier<PIPE_V>();
    SetMaskCount();
    BroadcastLastDim(tempMean, tempVariance, 1, computeLength);

    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;
    SetVectorMask<float, MaskMode::COUNTER>(0, computeLength);

    Sub<float, false>(tempVariance, inputMean, tempMean, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Mul<float, false>(tempMean, tempVariance, tempVariance, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Cast<float, int32_t, false>(tempVariance, counts, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Mul<float, false>(tempMean, tempMean, tempVariance, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Add<float, false>(tempVariance, tempMean, inputVariance, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Muls<float, false>(tempVariance, tempVariance, para.rRec, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    LayerNormReduceSumImpl<true, false>(tempMean, outputVariance, tempVariance, 1, computeLength);
    PipeBarrier<PIPE_V>();
}

template <bool isReuseSource = false>
__aicore__ inline void WelfordFinalizeExeMean(const LocalTensor<float> &inputMean, const LocalTensor<float> &outputMean, const LocalTensor<int32_t> &counts,
    const WelfordFinalizeTmpTensors<float> &tempTensors, const WelfordFinalizePara &para, const uint32_t computeLength)
{
    LocalTensor<float> tempMean = tempTensors.tempMean;
    LocalTensor<float> tempVariance = tempTensors.tempVariance;

    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, computeLength);
    Cast<float, int32_t, false>(tempMean, counts, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Mul<float, false>(tempMean, inputMean, tempMean, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Muls<float, false>(tempMean, tempMean, para.rRec, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    LayerNormReduceSumImpl<true, false>(tempVariance, outputMean, tempMean, 1, computeLength);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void GetWelfordFinalizeOutputMeanWithTail(const LocalTensor<float> &inputMean, const LocalTensor<float> &tempTensorCal, 
    const LocalTensor<float> &tempoutputMean, const WelfordFinalizePara &para, const LocalTensor<float> &outputMean, const uint32_t computeLength, const uint32_t offset)
{
    const UnaryRepeatParams unaryParams;
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, computeLength);
    int32_t headCount = para.headCount;
    int32_t tailCount = para.tailCount;

    Muls<float, false>(tempTensorCal, inputMean, (float)tailCount, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Muls<float, false>(tempoutputMean, tempTensorCal, para.rRec, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    int32_t headComputeLength = para.headCountLength - offset;
    if (headComputeLength > static_cast<int32_t>(computeLength)) {
        headComputeLength = computeLength;
    } else if (headComputeLength < 0) {
        headComputeLength = 0;
    }

    if (headComputeLength > 0) {
        SetVectorMask<float, MaskMode::COUNTER>(0, headComputeLength);
        Muls<float, false>(tempoutputMean, tempoutputMean, (float)headCount / (float)tailCount, MASK_PLACEHOLDER, 1, unaryParams);
        PipeBarrier<PIPE_V>();
    }

    SetVectorMask<float, MaskMode::COUNTER>(0, computeLength);

    LayerNormReduceSumImpl<true, false>(tempTensorCal, outputMean, tempoutputMean, 1, computeLength);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void GetWelfordFinalizeOutputMeanNoTail(const LocalTensor<float> &inputMean, const LocalTensor<float> &tempoutputMean,
    const LocalTensor<float> &tempTensorCal, const WelfordFinalizePara &para, const LocalTensor<float> &outputMean, const uint32_t computeLength)
{
    SetMaskCount();
    const UnaryRepeatParams unaryParams;
    SetVectorMask<float, MaskMode::COUNTER>(0, computeLength);

    Muls<float, false>(tempoutputMean, inputMean, para.abRec, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    LayerNormReduceSumImpl<true, false>(tempTensorCal, outputMean, tempoutputMean, 1, computeLength);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void GetWelfordFinalizeOutputVarianceWithTail(const LocalTensor<float> &inputMean, const LocalTensor<float> &inputVariance,
    const LocalTensor<float> &tempoutputMean, const LocalTensor<float> &tempTensorCal, const WelfordFinalizePara &para, const LocalTensor<float> &outputVariance,
    const uint32_t computeLength, const uint32_t offset)
{
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, computeLength);
    int32_t headCount = para.headCount;
    int32_t tailCount = para.tailCount;

    Sub<float, false>(tempTensorCal, inputMean, tempoutputMean, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Mul<float, false>(tempoutputMean, tempTensorCal, tempTensorCal, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Muls<float, false>(tempTensorCal, tempoutputMean, (float)tailCount, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    int32_t headComputeLength = para.headCountLength - offset;
    if (headComputeLength > static_cast<int32_t>(computeLength)) {
        headComputeLength = computeLength;
    } else if (headComputeLength < 0) {
        headComputeLength = 0;
    }

    if (headComputeLength > 0) {
        SetVectorMask<float, MaskMode::COUNTER>(0, headComputeLength);
        Muls<float, false>(tempTensorCal, tempTensorCal, (float)headCount / (float)tailCount, MASK_PLACEHOLDER, 1, unaryParams);
        PipeBarrier<PIPE_V>();
    }

    SetVectorMask<float, MaskMode::COUNTER>(0, computeLength);
    Add<float, false>(tempoutputMean, inputVariance, tempTensorCal, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Muls<float, false>(tempTensorCal, tempoutputMean, para.rRec, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    LayerNormReduceSumImpl<true, false>(tempoutputMean, outputVariance, tempTensorCal, 1, computeLength);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void GetWelfordFinalizeOutputVarianceNoTail(const LocalTensor<float> &inputMean, const LocalTensor<float> &inputVariance,
    const LocalTensor<float> &tempoutputMean, const LocalTensor<float> &tempTensorCal, const WelfordFinalizePara &para, const LocalTensor<float> &outputVariance, const uint32_t computeLength)
{
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, computeLength);
    int32_t rnLength = para.rnLength;

    Sub<float, false>(tempTensorCal, inputMean, tempoutputMean, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Mul<float, false>(tempoutputMean, tempTensorCal, tempTensorCal, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Muls<float, false>(tempTensorCal, tempoutputMean, (float)rnLength, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Add<float, false>(tempoutputMean, inputVariance, tempTensorCal, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Muls<float, false>(tempTensorCal, tempoutputMean, para.rRec, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    LayerNormReduceSumImpl<true, false>(tempoutputMean, outputVariance, tempTensorCal, 1, computeLength);
    PipeBarrier<PIPE_V>();
}

template <bool isReuseSource = false>
__aicore__ inline void WelfordFinalizeExeMean(const LocalTensor<float> &inputMean, const LocalTensor<float> &outputMean,
    const WelfordFinalizeTmpTensors<float> &tempTensors, const WelfordFinalizePara &para, const uint32_t computeLength, const uint32_t offset)
{
    LocalTensor<float> tempMean = tempTensors.tempMean;
    LocalTensor<float> tempVariance = tempTensors.tempVariance;

    if (para.tailCountLength == 0 || para.tailCount == 0) {
        GetWelfordFinalizeOutputMeanNoTail(inputMean, tempMean, tempVariance, para, outputMean, computeLength);
    } else {
        ASCENDC_ASSERT((para.tailCount != 0), {
                KERNEL_LOG(KERNEL_ERROR,
                           "Failed to check para.tailCount, para.tailCount should not be zero when para.tailCountLength is non-zero!");
            });
        GetWelfordFinalizeOutputMeanWithTail(inputMean, tempVariance, tempMean, para, outputMean, computeLength, offset);
    }
}

template <bool isReuseSource = false>
__aicore__ inline void WelfordFinalizeExeVariance(const LocalTensor<float> &inputMean, const LocalTensor<float> &inputVariance,
    const LocalTensor<float> &outputVariance, WelfordFinalizeTmpTensors<float> &tempTensors, const WelfordFinalizePara &para, const uint32_t computeLength, const uint32_t offset)
{
    LocalTensor<float> tempMean = tempTensors.tempMean;
    LocalTensor<float> tempVariance = tempTensors.tempVariance;
    Adds(tempVariance, tempMean, static_cast<float>(0), 1);
    PipeBarrier<PIPE_V>();
    SetMaskCount();
    BroadcastLastDim(tempMean, tempVariance, 1, computeLength);
    if (para.tailCountLength == 0 || para.tailCount == 0) {
        GetWelfordFinalizeOutputVarianceNoTail(inputMean, inputVariance, tempMean, tempVariance, para, outputVariance, computeLength);
    } else {
        ASCENDC_ASSERT((para.tailCount != 0), {
                KERNEL_LOG(KERNEL_ERROR,
                           "Failed to check para.tailCount, para.tailCount should not be zero when para.tailCountLength is non-zero!");
            });
        GetWelfordFinalizeOutputVarianceWithTail(inputMean, inputVariance, tempMean, tempVariance, para, outputVariance, computeLength, offset);
    }
}

template <bool isReuseSource = false>
__aicore__ inline void WelfordFinalizeComputeImpl(const LocalTensor<float> &inputMean, const LocalTensor<float> &inputVariance,
    const LocalTensor<float> &outputMean, const LocalTensor<float> &outputVariance, WelfordFinalizeTmpTensors<float> &tempTensors, 
    const WelfordFinalizePara &para, WelfordFinalizeTilingData &tiling)
{
    uint32_t offset = 0;
    uint32_t outOffset = 0;
    LocalTensor<float> tempMean = tempTensors.tempMean;
    LocalTensor<float> tempOutputMean = tempTensors.tempOutputMean;
    LocalTensor<float> tempOutputVariance = tempTensors.tempOutputVariance;

    for (uint32_t i = 0; i < tiling.round; i++)
    {
        WelfordFinalizeExeMean<isReuseSource>(inputMean[offset], tempOutputMean[outOffset],
            tempTensors, para, tiling.computeLength, offset);
        offset += tiling.computeLength;
        outOffset++;
        if (outOffset == B32_LEN) {
            SetMaskNorm();
            WholeReduceSum(tempOutputMean, tempOutputMean, B32_LEN, 1, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, outOffset >> 0x3);
            PipeBarrier<PIPE_V>();
            outOffset = 1;
        }
    }

    if (tiling.tail > 0) {
        WelfordFinalizeExeMean<isReuseSource>(inputMean[offset], tempOutputMean[outOffset],
            tempTensors, para, tiling.tail, offset);
        outOffset++;
    }
    SetMaskNorm();
    WholeReduceSum(tempOutputMean, tempOutputMean, outOffset, 1, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, outOffset >> 0x3);
    PipeBarrier<PIPE_V>();

    offset = 0;
    outOffset = 0;
    for (uint32_t i = 0; i < tiling.round; i++)
    {
        Adds(tempMean, tempOutputMean, static_cast<float>(0), 1);
        PipeBarrier<PIPE_V>();
        WelfordFinalizeExeVariance<isReuseSource>(inputMean[offset], inputVariance[offset], tempOutputVariance[outOffset],
            tempTensors, para, tiling.computeLength, offset);
        offset += tiling.computeLength;
        outOffset++;
        if (outOffset == B32_LEN) {
            SetMaskNorm();
            WholeReduceSum(tempOutputVariance, tempOutputVariance, B32_LEN, 1, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, outOffset >> 0x3);
            PipeBarrier<PIPE_V>();
            outOffset = 1;
        }
    }

    if (tiling.tail > 0) {
        Adds(tempMean, tempOutputMean, static_cast<float>(0), 1);
        PipeBarrier<PIPE_V>();
        WelfordFinalizeExeVariance<isReuseSource>(inputMean[offset], inputVariance[offset], tempOutputVariance[outOffset],
            tempTensors, para, tiling.tail, offset);
        outOffset++;
    }
    SetMaskNorm();
    WholeReduceSum(tempOutputVariance, tempOutputVariance, outOffset, 1, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, outOffset >> 0x3);
    PipeBarrier<PIPE_V>();
    welfordFinalizeOutputPre(outputMean, outputVariance, tempOutputMean, tempOutputVariance);
}

template <bool isReuseSource = false>
__aicore__ inline void WelfordFinalizeComputeImpl(const LocalTensor<float> &inputMean, const LocalTensor<float> &inputVariance,
    const LocalTensor<float> &outputMean, const LocalTensor<float> &outputVariance, const LocalTensor<int32_t> &counts, 
    const WelfordFinalizeTmpTensors<float> &tempTensors, const WelfordFinalizePara &para, WelfordFinalizeTilingData &tiling)
{
    uint32_t offset = 0;
    uint32_t outOffset = 0;
    LocalTensor<float> tempMean = tempTensors.tempMean;
    LocalTensor<float> tempOutputMean = tempTensors.tempOutputMean;
    LocalTensor<float> tempOutputVariance = tempTensors.tempOutputVariance;

    for (uint32_t i = 0; i < tiling.round; i++)
    {
        WelfordFinalizeExeMean<isReuseSource>(inputMean[offset], tempOutputMean[outOffset], counts[offset],
            tempTensors, para, tiling.computeLength);
        offset += tiling.computeLength;
        outOffset++;
        if (outOffset == B32_LEN) {
            SetMaskNorm();
            WholeReduceSum(tempOutputMean, tempOutputMean, B32_LEN, 1, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, outOffset >> 0x3);
            PipeBarrier<PIPE_V>();
            outOffset = 1;
        }
    }

    if (tiling.tail > 0) {
        WelfordFinalizeExeMean<isReuseSource>(inputMean[offset], tempOutputMean[outOffset], counts[offset],
            tempTensors, para, tiling.tail);
        outOffset++;
    }
    SetMaskNorm();
    WholeReduceSum(tempOutputMean, tempOutputMean, outOffset, 1, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, outOffset >> 0x3);
    PipeBarrier<PIPE_V>();

    offset = 0;
    outOffset = 0;
    for (uint32_t i = 0; i < tiling.round; i++)
    {
        Adds(tempMean, tempOutputMean, static_cast<float>(0), 1);
        PipeBarrier<PIPE_V>();
        WelfordFinalizeExeVariance<isReuseSource>(inputMean[offset], inputVariance[offset], tempOutputVariance[outOffset], counts[offset],
            tempTensors, para, tiling.computeLength);
        offset += tiling.computeLength;
        outOffset++;
        if (outOffset == B32_LEN) {
            SetMaskNorm();
            WholeReduceSum(tempOutputVariance, tempOutputVariance, B32_LEN, 1, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, outOffset >> 0x3);
            PipeBarrier<PIPE_V>();
            outOffset = 1;
        }
    }

    if (tiling.tail > 0) {
        Adds(tempMean, tempOutputMean, static_cast<float>(0), 1);
        PipeBarrier<PIPE_V>();
        WelfordFinalizeExeVariance<isReuseSource>(inputMean[offset], inputVariance[offset], tempOutputVariance[outOffset], counts[offset],
            tempTensors, para, tiling.tail);
        outOffset++;
    }
    SetMaskNorm();
    WholeReduceSum(tempOutputVariance, tempOutputVariance, outOffset, 1, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, outOffset >> 0x3);
    PipeBarrier<PIPE_V>();
    welfordFinalizeOutputPre(outputMean, outputVariance, tempOutputMean, tempOutputVariance);
}

template <bool isReuseSource = false>
__aicore__ inline void GetWelfordFinalizeTensorInfo(const LocalTensor<float> &stackBuffer, const WelfordFinalizePara &para,
                                                    WelfordFinalizeTmpTensors<float> &tempTensors, WelfordFinalizeTilingData &tiling)
{
    // 0x2 indicates reserving twp buffers for the calculations of mean and variance
    uint32_t minTmpSize = B32_LEN * 0x2;
    // 0x2 indicates reserving two buffers for the storage of mean and variance
    const uint32_t minTmpOutSize = B32_LEN * 0x2;
    ASCENDC_ASSERT((stackBuffer.GetSize() >= (minTmpSize + minTmpOutSize)),
                { KERNEL_LOG(KERNEL_ERROR, "sharedTmpBuffer size must >= %d Bytes!", (minTmpSize + minTmpOutSize) * sizeof(float)); });

    const uint32_t expFactor = (stackBuffer.GetSize() - minTmpOutSize) / minTmpSize;
    tiling.computeLength = expFactor * B32_LEN;
    tiling.round = para.abLength / tiling.computeLength;
    tiling.tail = para.abLength % tiling.computeLength;

    tempTensors.tempOutputMean = stackBuffer;
    tempTensors.tempOutputVariance = stackBuffer[B32_LEN];
    tempTensors.tempMean = stackBuffer[minTmpOutSize];
    tempTensors.tempVariance = stackBuffer[minTmpOutSize + tiling.computeLength];
}

__aicore__ inline void welfordFinalizeCommonCheck(const LocalTensor<float> &inputMean, const LocalTensor<float> &inputVariance,
                                                const LocalTensor<float> &outputMean, const LocalTensor<float> &outputVariance, const WelfordFinalizePara &para)
{
    ASCENDC_ASSERT((inputMean.GetSize() >= para.abLength), { 
        KERNEL_LOG(KERNEL_ERROR, 
        "Failed to check the size of inputMean, size is %u, which should be larger than para.abLength!", inputMean.GetSize()); 
    });
    ASCENDC_ASSERT((inputVariance.GetSize() >= para.abLength), { 
        KERNEL_LOG(KERNEL_ERROR, 
        "Failed to check the size of inputVariance, size is %u, which should be para.abLength!", inputVariance.GetSize()); 
    });
    ASCENDC_ASSERT((outputMean.GetSize() >= OUTPUT_SIZE), { 
        KERNEL_LOG(KERNEL_ERROR, 
        "Failed to check the size of outputMean, outputMean is %u, which should be larger than 8!", outputMean.GetSize()); 
    });
    ASCENDC_ASSERT((outputVariance.GetSize() >= OUTPUT_SIZE), { 
        KERNEL_LOG(KERNEL_ERROR, 
        "Failed to check the size of outputVariance, outputVariance is %u, which should be larger than 8!", outputVariance.GetSize()); 
    });
    ASCENDC_ASSERT((para.abLength > 0 && para.abLength % (ONE_BLK_SIZE / sizeof(float)) == 0), {
        KERNEL_LOG(KERNEL_ERROR,
        "Failed to check para.abLength, abLength is %u, which should not be 0 and 32Bytes aligned(float)!", para.abLength);
    });
    ASCENDC_ASSERT((para.headCountLength + para.tailCountLength == para.abLength), {
        KERNEL_LOG(KERNEL_ERROR,
        "Failed to check para.headCountLength and para.tailCountLength, the sum of them must be equal to para.abLength!");
    });
}

template <bool isReuseSource = false>
__aicore__ inline void WelfordFinalizeImpl(const LocalTensor<float> &outputMean, const LocalTensor<float> &outputVariance,
                                        const LocalTensor<float> &inputMean, const LocalTensor<float> &inputVariance, const LocalTensor<uint8_t> &sharedTmpBuffer, const WelfordFinalizePara &para)
{
    welfordFinalizeCommonCheck(inputMean, inputVariance, outputMean, outputVariance, para);

    LocalTensor<float> stackBuffer = sharedTmpBuffer.ReinterpretCast<float>();
    WelfordFinalizeTmpTensors<float> tempTensors;
    WelfordFinalizeTilingData tiling;
    GetWelfordFinalizeTensorInfo<isReuseSource>(stackBuffer, para, tempTensors, tiling);

    SetMaskCount();
    WelfordFinalizeComputeImpl<isReuseSource>(inputMean, inputVariance, outputMean, outputVariance, tempTensors, para, tiling);

    SetMaskNorm();
    ResetMask();
}

template <bool isReuseSource = false>
__aicore__ inline void WelfordFinalizeImpl(const LocalTensor<float> &outputMean, const LocalTensor<float> &outputVariance,
                                        const LocalTensor<float> &inputMean, const LocalTensor<float> &inputVariance, const LocalTensor<int32_t> &counts, const LocalTensor<uint8_t> &sharedTmpBuffer, const WelfordFinalizePara &para)
{
    ASCENDC_ASSERT((counts.GetSize() >= para.abLength), { 
        KERNEL_LOG(KERNEL_ERROR, 
        "Failed to check the size of counts, size is %u, which should be larger than para.abLength!", counts.GetSize()); 
    });
    welfordFinalizeCommonCheck(inputMean, inputVariance, outputMean, outputVariance, para);

    LocalTensor<float> stackBuffer = sharedTmpBuffer.ReinterpretCast<float>();
    WelfordFinalizeTmpTensors<float> tempTensors;
    WelfordFinalizeTilingData tiling;
    GetWelfordFinalizeTensorInfo<isReuseSource>(stackBuffer, para, tempTensors, tiling);

    SetMaskCount();
    WelfordFinalizeComputeImpl<isReuseSource>(inputMean, inputVariance, outputMean, outputVariance, counts, tempTensors, para, tiling);

    SetMaskNorm();
    ResetMask();
}

template <bool isReuseSource = false>
__aicore__ inline void WelfordFinalizeImpl(const LocalTensor<float> &outputMean, const LocalTensor<float> &outputVariance,
                                        const LocalTensor<float> &inputMean, const LocalTensor<float> &inputVariance, const WelfordFinalizePara &para)
{
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });

    WelfordFinalizeImpl<isReuseSource>(outputMean, outputVariance, inputMean, inputVariance, sharedTmpBuffer, para);
}

template <bool isReuseSource = false>
__aicore__ inline void WelfordFinalizeImpl(const LocalTensor<float> &outputMean, const LocalTensor<float> &outputVariance, const LocalTensor<float> &inputMean,
                                        const LocalTensor<float> &inputVariance, const LocalTensor<int32_t> &counts, const WelfordFinalizePara &para)
{
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });

    WelfordFinalizeImpl<isReuseSource>(outputMean, outputVariance, inputMean, inputVariance, counts, sharedTmpBuffer, para);
}
} // namespace AscendC
#endif // IMPL_NORMALIZATION_WELFORDFINALIZE_COMMON_IMPL_H