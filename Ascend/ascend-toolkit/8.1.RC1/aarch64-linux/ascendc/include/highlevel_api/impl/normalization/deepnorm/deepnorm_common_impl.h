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
 * \file deepnorm_common_impl.h
 * \brief
 */
#ifndef IMPL_NORMALIZATION_DEEPNORM_DEEPNORM_COMMON_IMPL_H
#define IMPL_NORMALIZATION_DEEPNORM_DEEPNORM_COMMON_IMPL_H

#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#if __CCE_AICORE__ == 220
#include "deepnorm_v220_impl.h"
#elif __CCE_AICORE__ == 200
#include "deepnorm_v200_impl.h"
#endif

namespace AscendC {
namespace DeepNormAPI {
template <typename T, bool isBasicBlock = false>
__aicore__ inline bool IsDeepNormParamValid(DeepNormTiling& tiling)
{
    ASCENDC_ASSERT((IsSameType<T, half>::value || IsSameType<T, float>::value),
        {KERNEL_LOG(KERNEL_ERROR, "DeepNorm only support data type: float/half");
        return false;});
    ASCENDC_ASSERT(tiling.oneTmpSize > 0,
        {KERNEL_LOG(KERNEL_ERROR, "In DeepNorm, Reduce axis is too long to put it in Pop Stack Buffer!");
        return false;});

    const bool hDivBy64 = (tiling.hLength % BASIC_BLOCK_HLENGTH == 0) &&
        (tiling.originalHLength % BASIC_BLOCK_HLENGTH == 0);
    const bool bsDivBy8 = ((tiling.bLength * tiling.sLength) % BASIC_BLOCK_BSLENGTH == 0);
    if constexpr (isBasicBlock) {
        ASCENDC_ASSERT(hDivBy64 && bsDivBy8,
            {KERNEL_LOG(KERNEL_ERROR, "In DeepNorm, when isBasicBlock is true, input must have hLength % 64 = 0, " \
                "originalHLegnth % 64 = 0 and (bLength * sLength) % 8 = 0 !");
            return false;});
    }

    return true;
}

__aicore__ inline void IsStackBufferValid(const LocalTensor<float>& stackBuffer, const DeepNormTiling& tiling)
{
    ASCENDC_ASSERT((stackBuffer.GetSize() >= tiling.tmpBufSize), {
        KERNEL_LOG(KERNEL_ERROR, "stackBuffer.GetSize (%d) should be >= tiling.tmpBufSize (%d)", stackBuffer.GetSize(),
            tiling.tmpBufSize);
    });
}

// check one tmpBuffer size is 8 * H * n (n>=1), if so, can use brcb for fast computing when isBasicBlock
__aicore__ inline bool IsBasicBlockTmp8HBetter(const DeepNormTiling& tiling)
{
    bool bs8Check = (tiling.oneTmpSize % (tiling.hLength * BASIC_BLOCK_BSLENGTH)) == 0;   // tmpBuffer is 8*H*n
    // bs for loop is worse than n in (64*n)
    bool bsWorse = tiling.bsCurLength > (tiling.hLength / BASIC_BLOCK_HLENGTH);
    return bs8Check && bsWorse;
}


// FP16 not reuseable, needs extra tmpTensor to store tmp FP32 outputMean and outputVariance
template <bool isReuseSource = false>
__aicore__ inline void GetDeepNormTensorInfo(const LocalTensor<half>& inputX, const LocalTensor<half>& outputMean,
    const LocalTensor<half>& outputVariance, const LocalTensor<float>& stackBuffer, const DeepNormTiling& tiling,
    DeepNormParams<float>& params)
{
    params.tempTensorA = stackBuffer[tiling.firstTmpStartPos];
    params.tempTensorB = stackBuffer[tiling.secondTmpStartPos];
    params.tempTensorC = stackBuffer[tiling.thirdTmpStartPos];
    params.meanTmpTensor = stackBuffer[tiling.meanTmpTensorPos];
    params.varianceTmpTensor = stackBuffer[tiling.varianceTmpTensorPos];

    ASCENDC_ASSERT((tiling.thirdTmpStartPos + tiling.oneTmpSize <= tiling.tmpBufSize), {
        KERNEL_LOG(KERNEL_ERROR, "thirdTmpStartPos(%d) + oneTmpSize (%d) should be <= tmpBufSize (%d)",
            tiling.thirdTmpStartPos, tiling.oneTmpSize, tiling.tmpBufSize);
    });

    IsStackBufferValid(stackBuffer, tiling);
}

// FP32 reuse outputMean and outputVariance tensor. When isReuseSource, use inputX as one of the tmpTensor
template <bool isReuseSource = false>
__aicore__ inline void GetDeepNormTensorInfo(const LocalTensor<float>& inputX, const LocalTensor<float>& outputMean,
    const LocalTensor<float>& outputVariance, const LocalTensor<float>& stackBuffer, const DeepNormTiling& tiling,
    DeepNormParams<float>& params)
{
    params.meanTmpTensor = outputMean;
    params.varianceTmpTensor = outputVariance;

    if constexpr (isReuseSource) {
        params.tempTensorA = inputX;
        params.tempTensorB = stackBuffer[tiling.firstTmpStartPos];
        params.tempTensorC = stackBuffer[tiling.secondTmpStartPos];
        ASCENDC_ASSERT((tiling.secondTmpStartPos + tiling.oneTmpSize <= tiling.tmpBufSize), {
            KERNEL_LOG(KERNEL_ERROR, "secondTmpStartPos + oneTmpSize (%d) should be <= tmpBufSize (%d)",
                tiling.secondTmpStartPos + tiling.oneTmpSize, tiling.tmpBufSize);
        });
    } else {
        params.tempTensorA = stackBuffer[tiling.firstTmpStartPos];
        params.tempTensorB = stackBuffer[tiling.secondTmpStartPos];
        params.tempTensorC = stackBuffer[tiling.thirdTmpStartPos];
        ASCENDC_ASSERT((tiling.thirdTmpStartPos + tiling.oneTmpSize <= tiling.tmpBufSize), {
            KERNEL_LOG(KERNEL_ERROR, "thirdTmpStartPos + oneTmpSize (%d) should be  <= tmpBufSize (%d)",
                tiling.thirdTmpStartPos + tiling.oneTmpSize, tiling.tmpBufSize);
        });
    }

    IsStackBufferValid(stackBuffer, tiling);
}

// (inputX * alpha -> mulsX)  and  (mulsX + inputGx -> X1), store X1 result in tempTensorA
__aicore__ inline void DeepNormExec(const LocalTensor<half>& inputX, const LocalTensor<half>& inputGx,
    const LocalTensor<half>& output, const half alpha, const DeepNormTiling& tiling,
    const DeepNormParams<float>& params)
{
    UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;
    SetVectorMask<float, MaskMode::COUNTER>(0, tiling.bshCurLength);

    // inputX + inputGx both cast to FP32 to gurantee precision
    unaryParams.srcRepStride = HALF_REPEAT_STRIDE;
    Cast<float, half, false>(params.tempTensorA, inputX, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Cast<float, half, false>(params.tempTensorC, inputGx, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    unaryParams.srcRepStride = DEFAULT_REPEAT_STRIDE;
    // 1. alpha * x => mulsX
    Muls<float, false>(params.tempTensorA, params.tempTensorA, static_cast<float>(alpha), MASK_PLACEHOLDER, 1,
        unaryParams);
    PipeBarrier<PIPE_V>();

    // 2. mulsX + gxLocal => X1
    Add<float, false>(params.tempTensorA, params.tempTensorC, params.tempTensorA, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
}

// (inputX * alpha -> mulsX) and (mulsX + inputGx -> X1), store X1 result in tempTensorA
__aicore__ inline void DeepNormExec(const LocalTensor<float>& inputX, const LocalTensor<float>& inputGx,
    const LocalTensor<float>& output, const float alpha, const DeepNormTiling& tiling,
    const DeepNormParams<float>& params)
{
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;
    SetVectorMask<float, MaskMode::COUNTER>(0, tiling.bshCurLength);

    // 1. alpha * x => mulsX
    Muls<float, false>(params.tempTensorB, inputX, alpha, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // 2. mulsX + gxLocal => X1
    Add<float, false>(params.tempTensorA, inputGx, params.tempTensorB, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
}

// output: return tensor,  tmp: tmpTensor for remaining calculation
__aicore__ inline void DeepNormBasicBlockReduceSum(const LocalTensor<float>& output, const LocalTensor<float>& tmp,
    const LocalTensor<float>& input, const UnaryRepeatParams& unaryParams, const uint32_t bsLength,
    const uint32_t hLength)
{
    const uint8_t num = hLength / BASIC_BLOCK_HLENGTH;     // isBasicBlock gurantee hLength = n * 64

    BinaryRepeatParams binaryParams;
    binaryParams.dstRepStride = num * DEFAULT_REPEAT_STRIDE;
    binaryParams.src0RepStride = num * DEFAULT_REPEAT_STRIDE;
    binaryParams.src1RepStride = num * DEFAULT_REPEAT_STRIDE;

    // Assume input is bs * (64 * n) always use part 1 (bs*64) to add with part 2, 3 etc, then apply reducesum
    //  input:  | a1 a2 ..|                  | a1+b1  a2+b2 ..|    | a1+b1+c1  a2+b2+c2 ..|
    //  bs*192  | b1 b2 ..|  (a~c is 3*64)=> | b1     b2    ..| => | b1        b2       ..| => reducesum
    //          | c1 c2 ..|                  | c1     c2    ..|    | c1        c2       ..|
    SetVectorMask<float, MaskMode::COUNTER>(0, bsLength * BASIC_BLOCK_HLENGTH);
    for (uint32_t i = 1; i < num; i++) {
        Add<float, false>(input, input[i * BASIC_BLOCK_HLENGTH], input, MASK_PLACEHOLDER, 1, binaryParams);
        PipeBarrier<PIPE_V>();
    }
    RepeatReduceSum<float, false>(output, input, 1, 1, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE,
        num * DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();

    RepeatReduceSum<float, false>(tmp, input, 1, 1, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE,
        num * DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
    SetVectorMask<float, MaskMode::COUNTER>(0, bsLength);
}

// reduceSum   B * S * H -> B * S
// dstMvTmp: output tensor, dst: tmp tensor for following calculation
__aicore__ inline void DeepNormReduceSumImpl(const LocalTensor<float>& dstMVTmp, const LocalTensor<float>& dst,
    const LocalTensor<float>& src, const uint32_t bsLength, const uint32_t hLength, const uint32_t originalHLength)
{
    for (uint32_t i = 0; i < bsLength; i++) {
        uint32_t totalNum = originalHLength;
        LocalTensor<float> srcTmp = src[i * hLength];
        LocalTensor<float> dstTmp = dst[i * hLength];

        while (totalNum > 1) {
            SetVectorMask<float, MaskMode::COUNTER>(0, totalNum);

            // final reducesum result put in output
            if (totalNum <= ONE_REPEAT_FLOAT_SIZE) {
                RepeatReduceSum<float, false>(dstMVTmp[i], srcTmp, 1, 1, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE,
                    DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
                PipeBarrier<PIPE_V>();
                dstTmp = dst[i];
            }

            RepeatReduceSum<float, false>(dstTmp, srcTmp, 1, 1, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE,
                DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
            PipeBarrier<PIPE_V>();

            totalNum = DivCeil(totalNum, ONE_REPEAT_FLOAT_SIZE);
            srcTmp = dstTmp;
        }
    }

    SetVectorMask<float, MaskMode::COUNTER>(0, bsLength);
}


// B*S*H and H tensor calculation    mode=0: add, mode=1: mul
template <bool isBasicBlock = false, uint8_t mode = 0>
__aicore__ inline void DeepNormBshHCalc(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
    const LocalTensor<float>& src1, const uint32_t bsLength, const uint32_t hLength)
{
    if constexpr(isBasicBlock) {
        const uint32_t loop = hLength / BASIC_BLOCK_HLENGTH;   // n * 64
        const uint16_t repStride = hLength / FLOAT_PER_BLOCK;
        BinaryRepeatParams binaryParams(1, 1, 1, repStride, repStride, 0);

        SetVectorMask<float, MaskMode::COUNTER>(0, bsLength * BASIC_BLOCK_HLENGTH);
        for (uint32_t i = 0; i < loop; i++) {
            uint32_t offset = i * BASIC_BLOCK_HLENGTH;
            if constexpr(mode) {
                Mul<float, false>(dst[offset], src0[offset], src1[offset], MASK_PLACEHOLDER, 1, binaryParams);
            } else {
                Add<float, false>(dst[offset], src0[offset], src1[offset], MASK_PLACEHOLDER, 1, binaryParams);
            }
        }
        PipeBarrier<PIPE_V>();
        SetVectorMask<float, MaskMode::COUNTER>(0, hLength);
    } else {
        BinaryRepeatParams binaryParams;
        for (uint32_t i = 0; i < bsLength; i++) {
            uint32_t offset = i * hLength;
            if constexpr(mode) {
                Mul<float, false>(dst[offset], src0[offset], src1, MASK_PLACEHOLDER, 1, binaryParams);
            } else {
                Add<float, false>(dst[offset], src0[offset], src1, MASK_PLACEHOLDER, 1, binaryParams);
            }
        }
        PipeBarrier<PIPE_V>();
    }
}


// tmpMean: temporary tensor used to store mean result for following calculation. It is stored in tmpTensorC
// outputMean: mean result to return
template <bool isBasicBlock = false>
__aicore__ inline void GetDeepNormOutputMean(const LocalTensor<float>& tmpMean, const LocalTensor<float>& inputX,
    const DeepNormTiling& tiling, const DeepNormParams<float>& params, const LocalTensor<float>& outputMean)
{
    const UnaryRepeatParams unaryParams;
    SetVectorMask<float, MaskMode::COUNTER>(0, tiling.bshCurLength);
    // inputX * (1 / num)
    Muls<float, false>(params.tempTensorC, inputX, params.lastDimValueBack, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // reduceSum -> meanX
    if constexpr(isBasicBlock) {
        DeepNormBasicBlockReduceSum(outputMean, tmpMean, params.tempTensorC, unaryParams, tiling.bsCurLength,
            tiling.hLength);
    } else {
        DeepNormReduceSumImpl(outputMean, tmpMean, params.tempTensorC, tiling.bsCurLength, tiling.hLength,
            tiling.originalHLength);
    }
}

// use for loop for calculation B*S*H and B*S
__aicore__ inline void DeepNormVarianceByForLoop(const LocalTensor<float>& inputX, const LocalTensor<float>& inputMean,
    const DeepNormTiling& tiling, const DeepNormParams<float>& params, const UnaryRepeatParams& unaryParams)
{
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    SetVectorMask<float, MaskMode::COUNTER>(0, tiling.hLength);   //  B*S*H - B*S  => B S H
    for (uint32_t i = 0; i < tiling.bsCurLength; i++) {
        Adds<float, false>(params.tempTensorB[i * tiling.hLength], inputX[i * tiling.hLength],
            (float)((inputMean.GetValue(i))*(-1)), MASK_PLACEHOLDER, 1, unaryParams);
    }
    PipeBarrier<PIPE_V>();

    SetVectorMask<float, MaskMode::COUNTER>(0, tiling.bshCurLength);
}

// use for loop for calculation B*S*H and B*S
__aicore__ inline void DeepNormOutputByForLoop(const LocalTensor<float>& xSubMean, const DeepNormTiling& tiling,
    const DeepNormParams<float>& params, const UnaryRepeatParams& unaryParams)
{
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    SetVectorMask<float, MaskMode::COUNTER>(0, tiling.hLength);   //  (B*S*H) * (B*S)  => B S H
    for (uint32_t i = 0; i < tiling.bsCurLength; i++) {
        Muls<float, false>(params.tempTensorC[i * tiling.hLength], xSubMean[i * tiling.hLength],
            (float)params.tempTensorA.GetValue(i), MASK_PLACEHOLDER, 1, unaryParams);
    }
    PipeBarrier<PIPE_V>();

    SetVectorMask<float, MaskMode::COUNTER>(0, tiling.bshCurLength);
}

// tmpVariance: temporary tensor used to store variance result for following calculation. It is stored in tmpTensorC
// outputVariance: variance result to return
template <bool isBasicBlock = false>
__aicore__ inline void GetDeepNormOutputVariance(const LocalTensor<float>& tmpVariance,
    const LocalTensor<float>& inputX, const LocalTensor<float>& inputMean, const DeepNormTiling& tiling,
    const DeepNormParams<float>& params, const LocalTensor<float>& outputVariance)
{
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;

    if constexpr(isBasicBlock) {
        if ((IsBasicBlockTmp8HBetter(tiling))) {
            DeepNormVarianceBasicBlockByBrcb(inputX, inputMean, tiling, params);
        } else {
            DeepNormVarianceByForLoop(inputX, inputMean, tiling, params, unaryParams);
        }
    } else {
        DeepNormVarianceByForLoop(inputX, inputMean, tiling, params, unaryParams);
    }

    // subX2 = subX * subX
    Mul<float, false>(params.tempTensorC, params.tempTensorB, params.tempTensorB, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    // mulsX = subX2 * (1/num)
    Muls<float, false>(params.tempTensorA, params.tempTensorC, params.lastDimValueBack, MASK_PLACEHOLDER, 1,
        unaryParams);
    PipeBarrier<PIPE_V>();

    // reduceSum -> variance
    if constexpr(isBasicBlock) {
        DeepNormBasicBlockReduceSum(outputVariance, tmpVariance, params.tempTensorA, unaryParams, tiling.bsCurLength,
            tiling.hLength);
    }
    else {
        DeepNormReduceSumImpl(outputVariance, tmpVariance, params.tempTensorA, tiling.bsCurLength, tiling.hLength,
            tiling.originalHLength);
    }
}

// calculate output before calculation with gamma and beta. Result is stored in tmpTensorC
template <bool isBasicBlock = false>
__aicore__ inline void GetDeepNormOutputPre(const LocalTensor<float>& xSubMean,
    const LocalTensor<float>& inputVariance, const float epsilon, const DeepNormTiling& tiling,
    const DeepNormParams<float>& params)
{
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;

    SetVectorMask<float, MaskMode::COUNTER>(0, tiling.bsCurLength);
    // addX = var + epsilon
    Adds<float, false>(params.tempTensorA, inputVariance, epsilon, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // all 1 tensor
    Duplicate<float, false>(params.tempTensorC, float(1.0), 1, 1, 1, 8);
    PipeBarrier<PIPE_V>();

    // SqrtX = sqrt(addX)
    Sqrt<float, false>(params.tempTensorA, params.tempTensorA, 1, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // 1 / SqrtX
    Div<float, false>(params.tempTensorA, params.tempTensorC, params.tempTensorA, 1, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    if constexpr(isBasicBlock) {
        // only when tmpTensor is 8*H*n, then can use brcb for basicblock
        if ((IsBasicBlockTmp8HBetter(tiling))) {
            DeepNormOutputBasicBlockByBrcb(xSubMean, tiling, params);
            return;
        }
    }
    DeepNormOutputByForLoop(xSubMean, tiling, params, unaryParams);
}

// inputY: result calculated after GetDeepNormOutputPre
template <bool isBasicBlock = false>
__aicore__ inline void GetDeepNormOutput(const LocalTensor<half>& output, const LocalTensor<float>& inputY,
    const LocalTensor<half>& gamm, const LocalTensor<half>& beta, const DeepNormTiling& tiling,
    const DeepNormParams<float>& params)
{
    UnaryRepeatParams unaryParams;

    SetVectorMask<float, MaskMode::COUNTER>(0, tiling.hLength);
    // 1. FP16 gamma cast -> FP32
    unaryParams.srcRepStride = HALF_REPEAT_STRIDE;
    Cast<float, half, false>(params.tempTensorA, gamm, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // 2. inputY (B*S*H) * gamm (H) -> mulX (B*S*H)
    DeepNormBshHCalc<isBasicBlock, 1>(params.tempTensorB, inputY, params.tempTensorA, tiling.bsCurLength,
        tiling.hLength);

    // 3. FP16 Beta cast -> FP32
    Cast<float, half, false>(params.tempTensorC, beta, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    // 4. mulX (B*S*H) + Beta(H) -> resX
    DeepNormBshHCalc<isBasicBlock, 0>(params.tempTensorA, params.tempTensorB, params.tempTensorC, tiling.bsCurLength,
        tiling.hLength);

    SetVectorMask<float, MaskMode::COUNTER>(0, tiling.bshCurLength);
    unaryParams.srcRepStride = DEFAULT_REPEAT_STRIDE;
    unaryParams.dstRepStride = HALF_REPEAT_STRIDE;

    // 5. FP32 resX cast -> FP16 output
    Cast<half, float, false>(output, params.tempTensorA, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
}

template <bool isBasicBlock = false>
__aicore__ inline void GetDeepNormOutput(const LocalTensor<float>& output, const LocalTensor<float>& inputY,
    const LocalTensor<float>& gamm, const LocalTensor<float>& beta, const DeepNormTiling& tiling,
    const DeepNormParams<float>& params)
{
    SetVectorMask<float, MaskMode::COUNTER>(0, tiling.hLength);
    DeepNormBshHCalc<isBasicBlock, 1>(params.tempTensorA, inputY, gamm, tiling.bsCurLength, tiling.hLength);
    DeepNormBshHCalc<isBasicBlock, 0>(output, params.tempTensorA, beta, tiling.bsCurLength, tiling.hLength);
}

// cast FP32 tmpMean and tmpVariance in params to FP16 outputMean and outputVariance
__aicore__ inline void GetDeepNormOutputMeanVariance(const LocalTensor<half>& outputMean,
    const LocalTensor<half>& outputVariance, const DeepNormTiling& tiling, const DeepNormParams<float>& params)
{
    UnaryRepeatParams unaryParams;
    unaryParams.dstRepStride = HALF_REPEAT_STRIDE;
    SetVectorMask<float, MaskMode::COUNTER>(0, tiling.meanVarSize);

    Cast<half, float, false>(outputMean, params.meanTmpTensor, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Cast<half, float, false>(outputVariance, params.varianceTmpTensor, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        unaryParams);
    PipeBarrier<PIPE_V>();
}


// For FP16 scenes, calculated FP32 X1(tmpTensorA) from DeepNormExec will be set as inputX
template <bool isBasicBlock = false>
__aicore__ inline void DeepNormLayerNormExec(const LocalTensor<float>& inputX, const LocalTensor<half>& gamm,
    const LocalTensor<half>& beta, const LocalTensor<half>& output, const LocalTensor<float>& outputMean,
    const LocalTensor<float>& outputVariance, const half epsilon, const DeepNormTiling& tiling,
    const DeepNormParams<float>& params)
{
    GetDeepNormOutputMean<isBasicBlock>(params.tempTensorC, params.tempTensorA, tiling, params, outputMean);
    GetDeepNormOutputVariance<isBasicBlock>(params.tempTensorC, params.tempTensorA, outputMean, tiling, params,
        outputVariance);
    GetDeepNormOutputPre<isBasicBlock>(params.tempTensorB, params.tempTensorC, static_cast<float>(epsilon), tiling,
        params);
    GetDeepNormOutput<isBasicBlock>(output, params.tempTensorC, gamm, beta, tiling, params);
}

template <bool isBasicBlock = false>
__aicore__ inline void DeepNormLayerNormExec(const LocalTensor<float>& inputX, const LocalTensor<float>& gamm,
    const LocalTensor<float>& beta, const LocalTensor<float>& output, const LocalTensor<float>& outputMean,
    const LocalTensor<float>& outputVariance, const float epsilon, const DeepNormTiling& tiling,
    const DeepNormParams<float>& params)
{
    GetDeepNormOutputMean<isBasicBlock>(params.tempTensorC, inputX, tiling, params, outputMean);
    GetDeepNormOutputVariance<isBasicBlock>(params.tempTensorC, inputX, outputMean, tiling, params, outputVariance);
    GetDeepNormOutputPre<isBasicBlock>(params.tempTensorB, params.tempTensorC, epsilon, tiling, params);
    GetDeepNormOutput<isBasicBlock>(output, params.tempTensorC, gamm, beta, tiling, params);
}

template <typename T, bool isBasicBlock = false>
__aicore__ inline void DeepNormND(const LocalTensor<T>& inputX, const LocalTensor<T>& inputGx,
    const LocalTensor<T>& gamm, const LocalTensor<T>& beta, const LocalTensor<T>& output,
    const LocalTensor<T>& outputMean, const LocalTensor<T>& outputVariance, const T alpha, const T epsilon,
    DeepNormTiling& tiling, const DeepNormParams<float>& params)
{
    uint32_t BSHOffset = 0;
    uint32_t BSOffset = 0;

    for (uint32_t index = 0; index < tiling.loopRound; index++) {
        DeepNormExec(inputX[BSHOffset], inputGx[BSHOffset], output, alpha, tiling, params);
        DeepNormLayerNormExec<isBasicBlock>(params.tempTensorA, gamm, beta, output[BSHOffset],
            params.meanTmpTensor[BSOffset], params.varianceTmpTensor[BSOffset], epsilon, tiling, params);
        BSHOffset += tiling.inputRoundSize;
        BSOffset += tiling.meanVarRoundSize;
    }

    if (tiling.inputTailSize > 0) {
        tiling.bshCurLength = tiling.inputTailSize;
        tiling.bsCurLength = tiling.meanVarTailSize;

        BSHOffset = tiling.inputTailPos;
        BSOffset = tiling.meanVarTailPos;
        DeepNormExec(inputX[BSHOffset], inputGx[BSHOffset], output, alpha, tiling, params);
        DeepNormLayerNormExec<isBasicBlock>(params.tempTensorA, gamm, beta, output[BSHOffset],
            params.meanTmpTensor[BSOffset], params.varianceTmpTensor[BSOffset], epsilon, tiling, params);
    }

    // need to cast FP32 result to FP16 result
    if constexpr(IsSameType<T, half>::value) {
        GetDeepNormOutputMeanVariance(outputMean, outputVariance, tiling, params);
    }
}

template <typename T, bool isReuseSrc, bool isBasicBlock>
__aicore__ inline void DeepNormImpl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& meanLocal,
    const LocalTensor<T>& rstdLocal, const LocalTensor<T>& srcLocal, const LocalTensor<T>& gxLocal,
    const LocalTensor<T>& betaLocal, const LocalTensor<T>& gammaLocal, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const T alpha, const T epsilon, DeepNormTiling& tiling)
{
    if (!DeepNormAPI::IsDeepNormParamValid<T, isBasicBlock>(tiling)) {
        return;
    }
    ASCENDC_ASSERT((sharedTmpBuffer.GetSize() > 0), { KERNEL_LOG(KERNEL_ERROR, "sharedTmpBuffer size must > 0!"); });
    LocalTensor<float> stackBuffer = sharedTmpBuffer.ReinterpretCast<float>();

    DeepNormParams<float> deepnormParams;
    DeepNormAPI::GetDeepNormTensorInfo<isReuseSrc>(srcLocal, meanLocal, rstdLocal, stackBuffer, tiling, deepnormParams);
    deepnormParams.lastDimValueBack = tiling.lastDimValueBack;

    SetMaskCount();
    DeepNormAPI::DeepNormND<T, isBasicBlock>(srcLocal, gxLocal, gammaLocal, betaLocal, dstLocal, meanLocal, rstdLocal,
        alpha, epsilon, tiling, deepnormParams);
    SetMaskNorm();
    ResetMask();
}

template <typename T, bool isReuseSrc, bool isBasicBlock>
__aicore__ inline void DeepNormImpl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& meanLocal,
    const LocalTensor<T>& rstdLocal, const LocalTensor<T>& srcLocal, const LocalTensor<T>& gxLocal,
    const LocalTensor<T>& betaLocal, const LocalTensor<T>& gammaLocal, const T alpha, const T epsilon,
    DeepNormTiling& tiling)
{
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    DeepNormImpl<T, isReuseSrc, isBasicBlock>(dstLocal, meanLocal, rstdLocal, srcLocal, gxLocal, betaLocal,
        gammaLocal, sharedTmpBuffer, alpha, epsilon, tiling);
}

} // namespace DeepNormAPI
} // namespace AscendC
#endif // IMPL_NORMALIZATION_DEEPNORM_DEEPNORM_COMMON_IMPL_H
