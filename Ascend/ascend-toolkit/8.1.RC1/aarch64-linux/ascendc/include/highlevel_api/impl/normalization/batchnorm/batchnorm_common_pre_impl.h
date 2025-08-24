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
 * \file batchnorm_common_pre_impl.h
 * \brief
 */
#ifndef IMPL_NORMALIZATION_BATCHNORM_BATCHNORM_COMMON_PRE_IMPL_H
#define IMPL_NORMALIZATION_BATCHNORM_BATCHNORM_COMMON_PRE_IMPL_H
#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tiling/kernel_tiling.h"

#if __CCE_AICORE__ == 200
#include "batchnorm_v200_impl.h"
#elif __CCE_AICORE__ == 220
#include "batchnorm_v220_impl.h"
#endif

namespace AscendC {
constexpr uint32_t FLOAT_BLOCK_NUMBER = 8;
constexpr uint32_t BASIC_BLOCK_LEN = 64;

// element num in stackBuffer cannot be less than tiling's expected element num
__aicore__ inline void StackBufferChecker(const LocalTensor<float>& stackBuffer, const BatchNormTiling& tiling)
{
    ASCENDC_ASSERT((stackBuffer.GetSize() >= tiling.tmpBufSize), {
        KERNEL_LOG(KERNEL_ERROR, "stackBuffer.GetSize (%d) should be >= tiling.tmpBufSize (%d)", stackBuffer.GetSize(),
            tiling.tmpBufSize);
    });
}

template <bool needCast = false> __aicore__ inline void GetSrcOffset(uint32_t& srcOffset, const BatchNormTiling& tiling)
{
    if constexpr (!needCast) {
        srcOffset = tiling.meanVarSize;
    } else {
        srcOffset = tiling.shCurLength;
    }
}

__aicore__ inline void GetUpdataParams(const BatchNormTiling& tiling, BatchNormParams<float>& params)
{
    params.srcRepeatStride = params.srcOffset / FLOAT_BLOCK_NUMBER;
    params.brcRepeatTimes = tiling.originalBLength / FLOAT_BLOCK_NUMBER;
    params.oriBloop = tiling.originalBLength / MAX_REPEAT_TIMES;
    params.oriBTail = tiling.originalBLength % MAX_REPEAT_TIMES;
    params.oriBTmpLoopOffset = tiling.shCurLength * MAX_REPEAT_TIMES;
    params.oriBTmpTailOffset = params.oriBloop * params.oriBTmpLoopOffset;
    params.oriBOutLoopOffset = tiling.meanVarSize * MAX_REPEAT_TIMES;
    params.oriBOutTailOffset = params.oriBloop * params.oriBOutLoopOffset;
    params.reduceAddLoop = (tiling.originalBLength - 1) / MAX_REPEAT_TIMES;
    params.reduceAddTail = (tiling.originalBLength - 1) % MAX_REPEAT_TIMES;
    params.reduceAddTailOffset = BASIC_BLOCK_LEN + params.reduceAddLoop * params.oriBTmpLoopOffset;
    params.basicLoop = tiling.shCurLength / BASIC_BLOCK_LEN;
}

template <typename T, bool needCast = false>
__aicore__ inline void GetMainTailOffset(uint64_t& inputMainOffset, uint64_t& inputTailOffset,
    const BatchNormParams<float>& params)
{
    inputMainOffset = params.oriBTmpLoopOffset;
    inputTailOffset = params.oriBTmpTailOffset;
    if constexpr (!needCast) {
        inputMainOffset = params.oriBOutLoopOffset;
        inputTailOffset = params.oriBOutTailOffset;
    }
}

template <bool isBasicBlock = false>
__aicore__ inline void CastGammBeta(const LocalTensor<float>& dst, const LocalTensor<half>& src,
    const BatchNormTiling& tiling, const BatchNormParams<float>& params)
{
    UnaryRepeatParams castUnaryParams;
    castUnaryParams.srcRepStride = (uint8_t)tiling.castHalfRepStride;
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, tiling.originalBLength);
    Cast<float, half, false>(dst, src, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, castUnaryParams);
    // isBasicBlock->brcb/transdata; !isBasicBlock->dup,need s wait v;
    if constexpr (!isBasicBlock) {
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
    } else {
        PipeBarrier<PIPE_V>();
    }
}

template <bool isBasicBlock = false>
__aicore__ inline void CastOutput(const LocalTensor<half>& output, const LocalTensor<float>& src,
    const BatchNormTiling& tiling, const BatchNormParams<float>& params)
{
    UnaryRepeatParams unaryParams;
    if constexpr (isBasicBlock) {
        unaryParams.dstRepStride = (uint8_t)tiling.castHalfOutRepStride;
        unaryParams.srcRepStride = (uint8_t)tiling.shCurLengthBlockNum;
        for (uint32_t m = 0; m < params.basicLoop; m++) {
            for (uint32_t i = 0; i < params.oriBloop; i++) {
                Cast<half, float, false>(output[i * params.oriBOutLoopOffset + m * BASIC_BLOCK_LEN],
                    src[i * params.oriBTmpLoopOffset + m * BASIC_BLOCK_LEN], RoundMode::CAST_NONE, MASK_PLACEHOLDER,
                    MAX_REPEAT_TIMES, unaryParams);
            }
            if (params.oriBTail) {
                Cast<half, float, false>(output[params.oriBOutTailOffset + m * BASIC_BLOCK_LEN],
                    src[params.oriBTmpTailOffset + m * BASIC_BLOCK_LEN], RoundMode::CAST_NONE, MASK_PLACEHOLDER,
                    (uint8_t)params.oriBTail, unaryParams);
            }
        }
    } else {
        SetVectorMask<float, MaskMode::COUNTER>(0, tiling.shCurLength);
        unaryParams.dstRepStride = (uint8_t)tiling.castHalfRepStride;
        for (uint32_t i = 0; i < tiling.originalBLength; i++) {
            Cast<half, float, false>(output[i * tiling.meanVarSize], src[i * tiling.shCurLength], RoundMode::CAST_NONE,
                MASK_PLACEHOLDER, 1, unaryParams);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <bool isBasicBlock = false>
__aicore__ inline void CastInput(const LocalTensor<float>& dst, const LocalTensor<half>& input,
    const BatchNormTiling& tiling, const BatchNormParams<float>& params)
{
    UnaryRepeatParams unaryParams;
    if constexpr (isBasicBlock) {
        unaryParams.dstRepStride = (uint8_t)tiling.shCurLengthBlockNum;
        unaryParams.srcRepStride = (uint8_t)tiling.castHalfOutRepStride;
        for (uint32_t m = 0; m < params.basicLoop; m++) {
            for (uint32_t i = 0; i < params.oriBloop; i++) {
                Cast<float, half, false>(dst[i * params.oriBTmpLoopOffset + m * BASIC_BLOCK_LEN],
                    input[i * params.oriBOutLoopOffset + m * BASIC_BLOCK_LEN], RoundMode::CAST_NONE, MASK_PLACEHOLDER,
                    MAX_REPEAT_TIMES, unaryParams);
            }
            if (params.oriBTail) {
                Cast<float, half, false>(dst[params.oriBTmpTailOffset + m * BASIC_BLOCK_LEN],
                    input[params.oriBOutTailOffset + m * BASIC_BLOCK_LEN], RoundMode::CAST_NONE, MASK_PLACEHOLDER,
                    (uint8_t)params.oriBTail, unaryParams);
            }
        }
    } else {
        SetVectorMask<float, MaskMode::COUNTER>(0, tiling.shCurLength);
        unaryParams.srcRepStride = (uint8_t)tiling.castHalfRepStride;
        for (uint32_t i = 0; i < tiling.originalBLength; i++) {
            Cast<float, half, false>(dst[i * tiling.shCurLength], input[i * tiling.meanVarSize], RoundMode::CAST_NONE,
                MASK_PLACEHOLDER, 1, unaryParams);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <bool isBasicBlock = false>
__aicore__ inline void GetReduceAddResult(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const BatchNormTiling& tiling, const BatchNormParams<float>& params)
{
    SetMaskNorm();
    ResetMask();
    DataCopyParams datacopyParams;
    datacopyParams.blockCount = 1;
    datacopyParams.blockLen = (uint16_t)tiling.shCurLengthBlockNum;
    DataCopy<float>(dst, src, datacopyParams);
    PipeBarrier<PIPE_V>();
    BinaryRepeatParams binaryParams;
    if constexpr (isBasicBlock) {
        binaryParams.dstRepStride = 0;
        binaryParams.src0RepStride = (uint8_t)tiling.shCurLengthBlockNum;
        binaryParams.src1RepStride = 0;
        for (uint32_t m = 0; m < params.basicLoop; m++) {
            for (uint32_t i = 0; i < params.reduceAddLoop; i++) {
                Add<float, false>(dst[m * BASIC_BLOCK_LEN],
                    src[tiling.shCurLength + i * params.oriBTmpLoopOffset + m * BASIC_BLOCK_LEN],
                    dst[m * BASIC_BLOCK_LEN], MASK_PLACEHOLDER, MAX_REPEAT_TIMES, binaryParams);
            }
            if (params.reduceAddTail) {
                Add<float, false>(dst[m * BASIC_BLOCK_LEN], src[tiling.shCurLength + m * BASIC_BLOCK_LEN],
                    dst[m * BASIC_BLOCK_LEN], MASK_PLACEHOLDER, (uint8_t)params.reduceAddTail, binaryParams);
            }
        }
    } else {
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, tiling.shCurLength);
        for (uint32_t i = 1; i < tiling.originalBLength; i++) {
            Add<float, false>(dst, dst, src[i * tiling.shCurLength], MASK_PLACEHOLDER, (uint8_t)1, binaryParams);
            PipeBarrier<PIPE_V>();
        }
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void BrcFirstDimByDup(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const uint32_t shLength, const uint32_t bLength)
{
    for (uint32_t i = 0; i < bLength; i++) {
        Duplicate<float, false>(dst[i * shLength], float(src.GetValue(i)), MASK_PLACEHOLDER, (uint8_t)1, (uint16_t)1,
            (uint8_t)DEFAULT_REPEAT_STRIDE);
    }
    PipeBarrier<PIPE_V>();
}

template <bool isBasicBlock = false>
__aicore__ inline void BrcFirstDim(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const BatchNormTiling& tiling, const BatchNormParams<float>& params)
{
    SetMaskNorm();
    ResetMask();
#if __CCE_AICORE__ == 220
    BrcFirstDimByBrcb<isBasicBlock>(dst, src, tiling, params);
#elif __CCE_AICORE__ == 200
    BrcFirstDimByTransData<isBasicBlock>(dst, src, tiling, params);
#endif
}

template <bool isBasicBlock = false, bool needCast = false>
__aicore__ inline void GetBatchNormOutputMean(const LocalTensor<float>& outputMean, const LocalTensor<float>& inputX,
    const BatchNormTiling& tiling, const BatchNormParams<float>& params)
{
    UnaryRepeatParams unaryParams;
    if constexpr (isBasicBlock) {
        SetMaskNorm();
        ResetMask();
        unaryParams.dstRepStride = (uint8_t)tiling.shCurLengthBlockNum;
        unaryParams.srcRepStride = params.srcRepeatStride;
        uint64_t inputMainOffset = 0;
        uint64_t inputTailOffset = 0;
        GetMainTailOffset<float, needCast>(inputMainOffset, inputTailOffset, params);
        for (uint32_t m = 0; m < params.basicLoop; m++) {
            for (uint32_t i = 0; i < params.oriBloop; i++) {
                Muls<float, false>(params.tempTensorC[i * params.oriBTmpLoopOffset + m * BASIC_BLOCK_LEN],
                    inputX[i * inputMainOffset + m * BASIC_BLOCK_LEN], params.firstDimValueBack, MASK_PLACEHOLDER,
                    MAX_REPEAT_TIMES, unaryParams);
            }
            if (params.oriBTail) {
                Muls<float, false>(params.tempTensorC[params.oriBTmpTailOffset + m * BASIC_BLOCK_LEN],
                    inputX[inputTailOffset + m * BASIC_BLOCK_LEN], params.firstDimValueBack, MASK_PLACEHOLDER,
                    (uint8_t)params.oriBTail, unaryParams);
            }
        }
        PipeBarrier<PIPE_V>();
        GetReduceAddResult<isBasicBlock>(outputMean, params.tempTensorC, tiling, params);
    } else {
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(0, tiling.shCurLength);
        for (uint32_t i = 0; i < tiling.originalBLength; i++) {
            Muls<float, false>(params.tempTensorC[i * tiling.shCurLength], inputX[i * params.srcOffset],
                params.firstDimValueBack, MASK_PLACEHOLDER, 1, unaryParams);
        }
        PipeBarrier<PIPE_V>();
        // reduce batch axis
        GetReduceAddResult<isBasicBlock>(outputMean, params.tempTensorC, tiling, params);
    }
}

template <bool needCast = false>
__aicore__ inline void GetBatchNormOutputVarianceBasicBlock(const LocalTensor<float>& outputVariance,
    const LocalTensor<float>& inputX, const LocalTensor<float>& outputMean, const BatchNormTiling& tiling,
    const BatchNormParams<float>& params)
{
    BinaryRepeatParams subBinaryParams;
    const BinaryRepeatParams mulBinaryParams;
    const UnaryRepeatParams mulsUnaryParams;
    SetMaskNorm();
    ResetMask();
    subBinaryParams.src0RepStride = params.srcRepeatStride;
    subBinaryParams.src1RepStride = 0;
    subBinaryParams.dstRepStride = (uint8_t)tiling.shCurLengthBlockNum;

    uint64_t inputMainOffset = 0;
    uint64_t inputTailOffset = 0;
    GetMainTailOffset<float, needCast>(inputMainOffset, inputTailOffset, params);
    for (uint32_t m = 0; m < params.basicLoop; m++) {
        for (uint32_t i = 0; i < params.oriBloop; i++) {
            Sub<float, false>(params.tempTensorC[i * params.oriBTmpLoopOffset + m * BASIC_BLOCK_LEN],
                inputX[i * inputMainOffset + m * BASIC_BLOCK_LEN], outputMean[m * BASIC_BLOCK_LEN], MASK_PLACEHOLDER,
                MAX_REPEAT_TIMES, subBinaryParams);
            PipeBarrier<PIPE_V>();
        }
        if (params.oriBTail) {
            Sub<float, false>(params.tempTensorC[params.oriBTmpTailOffset + m * BASIC_BLOCK_LEN],
                inputX[inputTailOffset + m * BASIC_BLOCK_LEN], outputMean[m * BASIC_BLOCK_LEN], MASK_PLACEHOLDER,
                (uint8_t)params.oriBTail, subBinaryParams);
            PipeBarrier<PIPE_V>();
        }
    }
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, tiling.bshCurLength);
    Mul<float, false>(params.tempTensorB, params.tempTensorC, params.tempTensorC, MASK_PLACEHOLDER, 1, mulBinaryParams);
    PipeBarrier<PIPE_V>();
    Muls<float, false>(params.tempTensorA, params.tempTensorB, params.firstDimValueBack, MASK_PLACEHOLDER, 1,
        mulsUnaryParams);
    PipeBarrier<PIPE_V>();
    GetReduceAddResult<true>(outputVariance, params.tempTensorA, tiling, params);
    SetMaskNorm();
    ResetMask();
}

__aicore__ inline void GetBatchNormOutputVarianceNorm(const LocalTensor<float>& outputVariance,
    const LocalTensor<float>& inputX, const LocalTensor<float>& outputMean, const BatchNormTiling& tiling,
    const BatchNormParams<float>& params)
{
    const BinaryRepeatParams binaryParams;
    const UnaryRepeatParams mulsUnaryParams;
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, tiling.shCurLength);
    // 1、sub shape:[s,h]->[b,s,h]
    for (uint32_t i = 0; i < tiling.originalBLength; i++) {
        Sub<float, false>(params.tempTensorC[i * tiling.shCurLength], inputX[i * params.srcOffset], outputMean,
            MASK_PLACEHOLDER, 1, binaryParams);
    }
    PipeBarrier<PIPE_V>();
    // 2、(x-meanX)*(x-meanX)   shape:[b,s,h]
    SetVectorMask<float, MaskMode::COUNTER>(0, tiling.bshCurLength);
    Mul<float, false>(params.tempTensorB, params.tempTensorC, params.tempTensorC, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
    // 3、(x-meanX)*(x-meanX) * (1/m)  shape:[b,s,h]
    Muls<float, false>(params.tempTensorA, params.tempTensorB, params.firstDimValueBack, MASK_PLACEHOLDER, 1,
        mulsUnaryParams);
    PipeBarrier<PIPE_V>();
    // 4、reduce batch axis  shape:[s,h]
    GetReduceAddResult<false>(outputVariance, params.tempTensorA, tiling, params);
}

template <bool isBasicBlock = false, bool needCast = false>
__aicore__ inline void GetBatchNormOutputVariance(const LocalTensor<float>& outputVariance,
    const LocalTensor<float>& inputX, const LocalTensor<float>& outputMean, const BatchNormTiling& tiling,
    const BatchNormParams<float>& params)
{
    if constexpr (isBasicBlock) {
        GetBatchNormOutputVarianceBasicBlock<needCast>(outputVariance, inputX, outputMean, tiling, params);
    } else {
        GetBatchNormOutputVarianceNorm(outputVariance, inputX, outputMean, tiling, params);
    }
}

__aicore__ inline void GetBatchNormOutputPreProcess(const LocalTensor<float>& addSrc,
    const LocalTensor<float>& addDst, const LocalTensor<float>& tmpDst, const float epsilon, 
    const BatchNormTiling& tiling)
{
    const UnaryRepeatParams unaryParams;
    constexpr float exponent = -0.5;
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, tiling.shCurLength);
    Adds<float, false>(addDst, addSrc, epsilon, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Ln<float, false>(tmpDst, addDst, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Muls<float, false>(addDst, tmpDst, exponent, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Exp<float, false>(tmpDst, addDst, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void GetBatchNormOutputPreBasicBlock(const LocalTensor<float>& addSrc,
    const LocalTensor<float>& addDst, const LocalTensor<float>& tmpDst, const LocalTensor<float>& dst,
    const float epsilon, const BatchNormTiling& tiling, const BatchNormParams<float>& params)
{
    GetBatchNormOutputPreProcess(addSrc, addDst, tmpDst, epsilon, tiling);
    SetMaskNorm();
    ResetMask();
    BinaryRepeatParams binaryParams;
    binaryParams.src0RepStride = 0;
    binaryParams.src1RepStride = (uint8_t)tiling.shCurLengthBlockNum;
    binaryParams.dstRepStride = (uint8_t)tiling.shCurLengthBlockNum;
    for (uint32_t m = 0; m < params.basicLoop; m++) {
        for (uint32_t i = 0; i < params.oriBloop; i++) {
            Mul<float, false>(dst[i * params.oriBTmpLoopOffset + m * BASIC_BLOCK_LEN], tmpDst[m * BASIC_BLOCK_LEN],
                dst[i * params.oriBTmpLoopOffset + m * BASIC_BLOCK_LEN], MASK_PLACEHOLDER, MAX_REPEAT_TIMES,
                binaryParams);
        }
        if (params.oriBTail) {
            Mul<float, false>(dst[params.oriBTmpTailOffset + m * BASIC_BLOCK_LEN], tmpDst[m * BASIC_BLOCK_LEN],
                dst[params.oriBTmpTailOffset + m * BASIC_BLOCK_LEN], MASK_PLACEHOLDER, (uint8_t)params.oriBTail,
                binaryParams);
        }
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void GetBatchNormOutputPreNorm(const LocalTensor<float>& addSrc, const LocalTensor<float>& addDst,
    const LocalTensor<float>& tmpDst, const LocalTensor<float>& dst, const float epsilon, const BatchNormTiling& tiling,
    const BatchNormParams<float>& params)
{
    GetBatchNormOutputPreProcess(addSrc, addDst, tmpDst, epsilon, tiling);
    const BinaryRepeatParams binaryParams;
    for (uint32_t i = 0; i < tiling.originalBLength; i++) {
        Mul<float, false>(dst[i * tiling.shCurLength], dst[i * tiling.shCurLength], tmpDst, MASK_PLACEHOLDER, 1,
            binaryParams);
    }
    PipeBarrier<PIPE_V>();
}

template <bool isBasicBlock = false>
__aicore__ inline void GetBatchNormOutputPre(const LocalTensor<float>& src, const LocalTensor<float>& dst,
    const float epsilon, const BatchNormTiling& tiling, const BatchNormParams<float>& params)
{
    if constexpr (isBasicBlock) {
        GetBatchNormOutputPreBasicBlock(src, params.tempTensorA, params.tempTensorB, dst, epsilon, tiling, params);
    } else {
        GetBatchNormOutputPreNorm(src, params.tempTensorA, params.tempTensorB, dst, epsilon, tiling, params);
    }
}
} // namespace AscendC
#endif // IMPL_NORMALIZATION_BATCHNORM_BATCHNORM_COMMON_PRE_IMPL_H
