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
 * \file batchnorm_common_impl.h
 * \brief
 */
#ifndef IMPL_NORMALIZATION_BATCHNORM_BATCHNORM_COMMON_IMPL_H
#define IMPL_NORMALIZATION_BATCHNORM_BATCHNORM_COMMON_IMPL_H
#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tiling/kernel_tiling.h"

#if __CCE_AICORE__ == 200
#include "batchnorm_v200_impl.h"
#elif __CCE_AICORE__ == 220
#include "batchnorm_v220_impl.h"
#endif
#include "batchnorm_common_pre_impl.h"

namespace AscendC {
__aicore__ inline void GetBatchNormOutputBasicBlock(const LocalTensor<float>& src, const LocalTensor<float>& output,
    const LocalTensor<float>& gamm, const LocalTensor<float>& beta, const BatchNormTiling& tiling,
    const BatchNormParams<float>& params)
{
    const BinaryRepeatParams binaryParams;
    BrcFirstDim(params.tempTensorB, gamm, tiling, params);
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, tiling.bshCurLength);
    Mul<float, false>(params.tempTensorC, params.tempTensorB, src, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
    BrcFirstDim(params.tempTensorB, beta, tiling, params);
    BinaryRepeatParams addBinaryParams;
    addBinaryParams.dstRepStride = tiling.meanVarSize / FLOAT_BLOCK_NUMBER;
    addBinaryParams.src0RepStride = (uint8_t)tiling.shCurLengthBlockNum;
    addBinaryParams.src1RepStride = (uint8_t)tiling.shCurLengthBlockNum;
    for (uint32_t m = 0; m < params.basicLoop; m++) {
        for (uint32_t i = 0; i < params.oriBloop; i++) {
            Add<float, false>(output[i * params.oriBOutLoopOffset + m * BASIC_BLOCK_LEN],
                params.tempTensorB[params.oriBTmpLoopOffset + m * BASIC_BLOCK_LEN],
                params.tempTensorC[params.oriBTmpLoopOffset + m * BASIC_BLOCK_LEN], MASK_PLACEHOLDER, MAX_REPEAT_TIMES,
                addBinaryParams);
        }
        if (params.oriBTail) {
            Add<float, false>(output[params.oriBOutTailOffset + m * BASIC_BLOCK_LEN],
                params.tempTensorB[params.oriBTmpTailOffset + m * BASIC_BLOCK_LEN],
                params.tempTensorC[params.oriBTmpTailOffset + m * BASIC_BLOCK_LEN], MASK_PLACEHOLDER,
                (uint8_t)params.oriBTail, addBinaryParams);
        }
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void GetBatchNormOutputNorm(const LocalTensor<float>& src, const LocalTensor<float>& output,
    const LocalTensor<float>& gamm, const LocalTensor<float>& beta, const BatchNormTiling& tiling,
    const BatchNormParams<float>& params)
{
    const BinaryRepeatParams binaryParams;
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, tiling.shCurLength);
    BrcFirstDimByDup(params.tempTensorB, gamm, tiling.shCurLength, tiling.originalBLength);
    SetVectorMask<float, MaskMode::COUNTER>(0, tiling.bshCurLength);
    Mul<float, false>(params.tempTensorC, params.tempTensorB, params.tempTensorC, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
    SetVectorMask<float, MaskMode::COUNTER>(0, tiling.shCurLength);
    BrcFirstDimByDup(params.tempTensorB, beta, tiling.shCurLength, tiling.originalBLength);
    for (uint32_t i = 0; i < tiling.originalBLength; i++) {
        Add<float, false>(output[i * tiling.meanVarSize], params.tempTensorB[i * tiling.shCurLength],
            params.tempTensorC[i * tiling.shCurLength], MASK_PLACEHOLDER, 1, binaryParams);
    }
    PipeBarrier<PIPE_V>();
}

template <bool isBasicBlock = false>
__aicore__ inline void GetBatchNormOutput(const LocalTensor<float>& src, const LocalTensor<float>& output,
    const LocalTensor<float>& gamm, const LocalTensor<float>& beta, const BatchNormTiling& tiling,
    const BatchNormParams<float>& params)
{
    if constexpr (isBasicBlock) {
        GetBatchNormOutputBasicBlock(src, output, gamm, beta, tiling, params);
    } else {
        GetBatchNormOutputNorm(src, output, gamm, beta, tiling, params);
    }
}

template <bool isBasicBlock = false>
__aicore__ inline void GetBatchNormOutput(const LocalTensor<float>& src, const LocalTensor<half>& output,
    const LocalTensor<half>& gamm, const LocalTensor<half>& beta, const BatchNormTiling& tiling,
    const BatchNormParams<float>& params)
{
    BinaryRepeatParams binaryParams;
    if constexpr (isBasicBlock) {
        CastGammBeta<isBasicBlock>(params.tempTensorA, gamm, tiling, params);
        BrcFirstDim(params.tempTensorB, params.tempTensorA, tiling, params);
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, tiling.bshCurLength);
        Mul<float, false>(params.tempTensorC, params.tempTensorB, src, MASK_PLACEHOLDER, 1, binaryParams);
        PipeBarrier<PIPE_V>();
        CastGammBeta<isBasicBlock>(params.tempTensorA, beta, tiling, params);
        BrcFirstDim(params.tempTensorB, params.tempTensorA, tiling, params);
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, tiling.bshCurLength);
        Add<float, false>(params.tempTensorB, params.tempTensorB, params.tempTensorC, MASK_PLACEHOLDER, 1,
            binaryParams);
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
        ResetMask();
    } else {
        CastGammBeta<isBasicBlock>(params.tempTensorA, gamm, tiling, params);
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, tiling.shCurLength);
        BrcFirstDimByDup(params.tempTensorB, params.tempTensorA, tiling.shCurLength, tiling.originalBLength);
        SetVectorMask<float, MaskMode::COUNTER>(0, tiling.bshCurLength);
        Mul<float, false>(params.tempTensorC, params.tempTensorB, params.tempTensorC, MASK_PLACEHOLDER, 1,
            binaryParams);
        PipeBarrier<PIPE_V>();
        CastGammBeta<isBasicBlock>(params.tempTensorA, beta, tiling, params);
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, tiling.shCurLength);
        BrcFirstDimByDup(params.tempTensorB, params.tempTensorA, tiling.shCurLength, tiling.originalBLength);
        PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < tiling.originalBLength; i++) {
            Add<float, false>(params.tempTensorB[i * tiling.shCurLength], params.tempTensorB[i * tiling.shCurLength],
                params.tempTensorC[i * tiling.shCurLength], MASK_PLACEHOLDER, 1, binaryParams);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <bool isBasicBlock = false>
__aicore__ inline void GetOutputMeanVariance(const LocalTensor<half>& dst, const LocalTensor<float>& src,
    const BatchNormTiling& tiling, const BatchNormParams<float>& params)
{
    if constexpr (isBasicBlock) {
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, tiling.shCurLength);
        UnaryRepeatParams unaryParams;
        unaryParams.dstRepStride = (uint8_t)tiling.castHalfRepStride;
        Cast<half, float, false>(dst, src, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams);
        SetMaskNorm();
        ResetMask();
    } else {
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, tiling.shCurLength);
        UnaryRepeatParams unaryParams;
        unaryParams.dstRepStride = (uint8_t)tiling.castHalfRepStride;
        Cast<half, float, false>(dst, src, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams);
    }
    PipeBarrier<PIPE_V>();
}

template <bool isReuseSource = false>
__aicore__ inline void GetBatchNormInfo(const LocalTensor<half>& inputX, const LocalTensor<half>& outputMean,
    const LocalTensor<half>& outputVariance, const LocalTensor<float>& stackBuffer, const BatchNormTiling& tiling,
    BatchNormParams<float>& params)
{
    params.meanTmpTensor = stackBuffer[tiling.meanTmpTensorPos];
    params.varianceTmpTensor = stackBuffer[tiling.varianceTmpTensorPos];
    params.tempTensorA = stackBuffer[tiling.firstTmpStartPos];
    params.tempTensorB = stackBuffer[tiling.secondTmpStartPos];
    params.tempTensorC = stackBuffer[tiling.thirdTmpStartPos];
    ASCENDC_ASSERT((tiling.thirdTmpStartPos + tiling.oneTmpSize <= tiling.tmpBufSize), {
        KERNEL_LOG(KERNEL_ERROR, "thirdTmpStartPos + oneTmpSize (%d) should be  <= tmpBufSize (%d)",
            tiling.thirdTmpStartPos + tiling.oneTmpSize, tiling.tmpBufSize);
    });
    StackBufferChecker(stackBuffer, tiling);
}

template <bool isReuseSource = false>
__aicore__ inline void GetBatchNormInfo(const LocalTensor<float>& inputX, const LocalTensor<float>& outputMean,
    const LocalTensor<float>& outputVariance, const LocalTensor<float>& stackBuffer, const BatchNormTiling& tiling,
    BatchNormParams<float>& params)
{
    params.meanTmpTensor = outputMean;
    params.varianceTmpTensor = outputVariance;

    params.tempTensorA = stackBuffer[tiling.firstTmpStartPos];
    params.tempTensorB = stackBuffer[tiling.secondTmpStartPos];
    params.tempTensorC = stackBuffer[tiling.thirdTmpStartPos];
    ASCENDC_ASSERT((tiling.thirdTmpStartPos + tiling.oneTmpSize <= tiling.tmpBufSize), {
        KERNEL_LOG(KERNEL_ERROR, "thirdTmpStartPos + oneTmpSize (%d) should be  <= tmpBufSize (%d)",
            tiling.thirdTmpStartPos + tiling.oneTmpSize, tiling.tmpBufSize);
    });
    StackBufferChecker(stackBuffer, tiling);
}

template <bool isBasicBlock = false>
__aicore__ inline void BatchNormExeImpl(const LocalTensor<float>& inputX, const LocalTensor<float>& gamm,
    const LocalTensor<float>& beta, const LocalTensor<float>& output, const LocalTensor<float>& outputMean,
    const LocalTensor<float>& outputVariance, const LocalTensor<float>& tmpOutputMean,
    const LocalTensor<float>& tmpOutputVariance, const float epsilon, const BatchNormTiling& tiling,
    const BatchNormParams<float>& params)
{
    constexpr bool needCast = false;
    if constexpr (isBasicBlock) {
        GetBatchNormOutputMean<isBasicBlock, needCast>(tmpOutputMean, inputX, tiling, params);
        GetBatchNormOutputVariance<isBasicBlock, needCast>(tmpOutputVariance, inputX, tmpOutputMean, tiling, params);
        GetBatchNormOutputPre<isBasicBlock>(tmpOutputVariance, params.tempTensorC, epsilon, tiling, params);
        GetBatchNormOutput<isBasicBlock>(params.tempTensorC, output, gamm, beta, tiling, params);
    } else {
        // 1、outputMean
        GetBatchNormOutputMean<isBasicBlock, needCast>(tmpOutputMean, inputX, tiling, params);
        // 2、outpouVariance
        GetBatchNormOutputVariance<isBasicBlock, needCast>(tmpOutputVariance, inputX, tmpOutputMean, tiling, params);
        // 3、gamma beta vmul+vadd
        GetBatchNormOutputPre<isBasicBlock>(tmpOutputVariance, params.tempTensorC, epsilon, tiling, params);
        // 4、output
        GetBatchNormOutput<isBasicBlock>(params.tempTensorC, output, gamm, beta, tiling, params);
    }
}

template <bool isBasicBlock = false>
__aicore__ inline void BatchNormExeImpl(const LocalTensor<half>& inputX, const LocalTensor<half>& gamm,
    const LocalTensor<half>& beta, const LocalTensor<half>& output, const LocalTensor<half>& outputMean,
    const LocalTensor<half>& outputVariance, const LocalTensor<float>& tmpOutputMean,
    const LocalTensor<float>& tmpOutputVariance, const half epsilon, const BatchNormTiling& tiling,
    const BatchNormParams<float>& params)
{
    constexpr bool needCast = true;
    UnaryRepeatParams unaryParams;
    if constexpr (isBasicBlock) {
        SetMaskNorm();
        ResetMask();
        CastInput<isBasicBlock>(params.tempTensorA, inputX, tiling, params);
        // 1、outputMean
        GetBatchNormOutputMean<isBasicBlock, needCast>(tmpOutputMean, params.tempTensorA, tiling, params);
        // 2、cast mean from half->float
        GetOutputMeanVariance<isBasicBlock>(outputMean, tmpOutputMean, tiling, params);
        // 3、outpouVariance
        GetBatchNormOutputVariance<isBasicBlock, needCast>(tmpOutputVariance, params.tempTensorA, tmpOutputMean, tiling,
            params);
        // 4、cast variance from half->float
        GetOutputMeanVariance<isBasicBlock>(outputVariance, tmpOutputVariance, tiling, params);
        // 5、gamma beta vmul+vadd
        GetBatchNormOutputPre<isBasicBlock>(tmpOutputVariance, params.tempTensorC, epsilon, tiling, params);
        // 6、output
        GetBatchNormOutput<isBasicBlock>(params.tempTensorC, output, gamm, beta, tiling, params);
        // 7、cast output from half->float
        CastOutput<isBasicBlock>(output, params.tempTensorB, tiling, params);
    } else {
        CastInput<isBasicBlock>(params.tempTensorA, inputX, tiling, params);
        GetBatchNormOutputMean<isBasicBlock, needCast>(tmpOutputMean, params.tempTensorA, tiling, params);
        GetOutputMeanVariance(outputMean, tmpOutputMean, tiling, params);
        GetBatchNormOutputVariance<isBasicBlock, needCast>(tmpOutputVariance, params.tempTensorA, tmpOutputMean, tiling,
            params);
        GetOutputMeanVariance(outputVariance, tmpOutputVariance, tiling, params);
        GetBatchNormOutputPre<isBasicBlock>(tmpOutputVariance, params.tempTensorC, epsilon, tiling, params);
        GetBatchNormOutput<isBasicBlock>(params.tempTensorC, output, gamm, beta, tiling, params);
        CastOutput<isBasicBlock>(output, params.tempTensorB, tiling, params);
    }
}

template <typename T, bool isBasicBlock = false>
__aicore__ inline void BatchNormCompute(const LocalTensor<T>& inputX, const LocalTensor<T>& gamm,
    const LocalTensor<T>& beta, const LocalTensor<T>& output, const LocalTensor<T>& outputMean,
    const LocalTensor<T>& outputVariance, const T epsilon, BatchNormTiling& tiling,
    BatchNormParams<float>& params)
{
    constexpr bool needCast = IsSameType<T, half>::value;
    uint32_t mvOffset = 0;
    // update params
    GetSrcOffset<needCast>(params.srcOffset, tiling);
    GetUpdataParams(tiling, params);

    for (uint32_t index = 0; index < tiling.loopRound; index++) {
        BatchNormExeImpl<isBasicBlock>(inputX[mvOffset], gamm, beta, output[mvOffset], outputMean[mvOffset],
            outputVariance[mvOffset], params.meanTmpTensor[mvOffset], params.varianceTmpTensor[mvOffset], epsilon,
            tiling, params);
        // update mvOffset
        mvOffset += tiling.shCurLength;
    }
    if (tiling.inputTailSize > 0) {
        // update params
        tiling.bshCurLength = tiling.inputTailSize;
        tiling.shCurLength = tiling.meanVarTailSize;
        tiling.shCurLengthBlockNum = tiling.shCurLength / FLOAT_BLOCK_NUMBER;
        GetSrcOffset<needCast>(params.srcOffset, tiling);
        GetUpdataParams(tiling, params);

        BatchNormExeImpl<isBasicBlock>(inputX[tiling.inputTailPos], gamm, beta, output[tiling.inputTailPos],
            outputMean[tiling.meanVarTailPos], outputVariance[tiling.meanVarTailPos],
            params.meanTmpTensor[tiling.meanVarTailPos], params.varianceTmpTensor[tiling.meanVarTailPos], epsilon,
            tiling, params);
    }
}

template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void BatchNormImpl(const LocalTensor<T>& output, const LocalTensor<T>& outputMean,
    const LocalTensor<T>& outputVariance, const LocalTensor<T>& inputX, const LocalTensor<T>& gamm,
    const LocalTensor<T>& beta, const LocalTensor<uint8_t>& sharedTmpBuffer, const T epsilon, BatchNormTiling& tiling)
{
    ASCENDC_ASSERT((IsSameType<T, half>::value || IsSameType<T, float>::value),
                   { KERNEL_LOG(KERNEL_ERROR, "BatchNorm only support data type: float/half"); });

    ASCENDC_ASSERT((sharedTmpBuffer.GetSize() > 0), { KERNEL_LOG(KERNEL_ERROR, "sharedTmpBuffer size must > 0!"); });
    LocalTensor<float> stackBuffer = sharedTmpBuffer.ReinterpretCast<float>();

    BatchNormParams<float> params;
    GetBatchNormInfo<isReuseSource>(inputX, outputMean, outputVariance, stackBuffer, tiling, params);
    params.firstDimValueBack = tiling.firstDimValueBack;

    SetMaskCount();
    BatchNormCompute<T, isBasicBlock>(inputX, gamm, beta, output, outputMean, outputVariance, epsilon, tiling, params);

    SetMaskNorm();
    ResetMask();
}

template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void BatchNormImpl(const LocalTensor<T>& output, const LocalTensor<T>& outputMean,
    const LocalTensor<T>& outputVariance, const LocalTensor<T>& inputX, const LocalTensor<T>& gamm,
    const LocalTensor<T>& beta, const T epsilon, BatchNormTiling& tiling)
{
    // Using the Stack Space to Allocate tmpBuffer
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ret = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ret), { KERNEL_LOG(KERNEL_ERROR, "BatchNorm PopStackBuffer Error!"); });
    BatchNormImpl<T, isReuseSource, isBasicBlock>(output, outputMean, outputVariance, inputX, gamm, beta,
        sharedTmpBuffer, epsilon, tiling);
}
} // namespace AscendC
#endif // IMPL_NORMALIZATION_BATCHNORM_BATCHNORM_COMMON_IMPL_H
