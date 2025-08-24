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
 * \file groupnorm_common_impl.h
 * \brief
 */

#ifndef IMPL_NORMALIZATION_GROUPNORM_GROUPNORM_COMMON_IMPL_H
#define IMPL_NORMALIZATION_GROUPNORM_GROUPNORM_COMMON_IMPL_H

#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tiling/kernel_tiling.h"

namespace AscendC {
    namespace {
        constexpr uint32_t GROUPNORM_MASK_MAX_VAL = 64;
        constexpr uint32_t GROUPNORM_MASK_SMALLEST_VAL = 8;
        constexpr uint32_t GROUPNORM_MASK_STEP_VAL = 8;
        constexpr uint32_t GROUPNORM_ONE_BLK_SIZE = 8;
    } // namespace

template <typename T> struct GroupNormParams
{
    __aicore__ GroupNormParams(){};
    LocalTensor<T> tempTensorA;
    LocalTensor<T> tempTensorB;
    LocalTensor<T> tempTensorC;
    LocalTensor<T> meanTmpTensor;
    LocalTensor<T> varianceTmpTensor;
};

__aicore__ inline uint32_t GetGroupNormWholeReduceMask1(const GroupNormTiling& tiling)
{
    uint32_t mask1{0};
    if (tiling.dhwAlignSize > GROUPNORM_MASK_MAX_VAL) {
        mask1 = GROUPNORM_MASK_MAX_VAL;
        while (mask1 != 0 && tiling.dhwAlignSize % mask1 != 0) {
            mask1 -= GROUPNORM_MASK_STEP_VAL;
        }
        return mask1;
    }
    return tiling.dhwAlignSize;
}

__aicore__ inline void GetGroupNormOutputMean(const LocalTensor<float>& x_in,
    const LocalTensor<float>& tmp, const LocalTensor<float>& mean,
    const GroupNormTiling& tiling)
{
    for (uint32_t i = 0; i < tiling.bsCurLength; ++i) {
        uint32_t buffIndex = i * tiling.dhwAlignSize;
        ReduceSum<float>(mean[i], x_in[buffIndex], tmp[buffIndex], tiling.dhwAlignSize);
    }
    PipeBarrier<PIPE_V>();  

    Muls(mean, mean, tiling.factor, tiling.bsCurLength);

    // mean will be used to GetValue() to get scalar value
    auto eventIdVToS = GetTPipePtr()->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
}

__aicore__ inline void GetGroupNormOutputVar(const LocalTensor<float>& x_in,
    const LocalTensor<float>& tmp1, const LocalTensor<float>& tmp2,
    const LocalTensor<float>& mean, const LocalTensor<float>& var, const GroupNormTiling& tiling)
{
    for (uint32_t i = 0; i < tiling.d * tiling.bsCurLength; ++i) {
        uint32_t buffIndex = i * tiling.hwAlignSize;
        Adds(tmp1[buffIndex], x_in[buffIndex], -1.0f * mean.GetValue(i / tiling.d), tiling.hw);
    }
    PipeBarrier<PIPE_V>();

    Mul(tmp2, tmp1, tmp1, tiling.bshCurLength);
    PipeBarrier<PIPE_V>();

    for (uint32_t i = 0; i < tiling.bsCurLength; ++i) {
        uint32_t buffIndex = i * tiling.dhwAlignSize;
        ReduceSum<float>(var[i], tmp2[buffIndex], tmp2[buffIndex], tiling.dhwAlignSize);
    }
    PipeBarrier<PIPE_V>();      

    Muls(var, var, tiling.factor, tiling.bsCurLength);
    PipeBarrier<PIPE_V>();      
}

__aicore__ inline void GetGroupNormOutputPre(const LocalTensor<float>& inout,
    const LocalTensor<float>& tmp, const LocalTensor<float>& variance,
    const GroupNormTiling& tiling, const float epsilon)
{
    Adds(tmp, variance, epsilon, tiling.bsCurLength);
    PipeBarrier<PIPE_V>();
    Ln(tmp, tmp, tiling.bsCurLength);
    PipeBarrier<PIPE_V>();
    // Multiply by -0.5f to convert the logarithmic result to the logarithm of the reciprocal of the standard deviation
    Muls(tmp, tmp, -0.5f, tiling.bsCurLength);
    PipeBarrier<PIPE_V>();
    Exp(tmp, tmp, tiling.bsCurLength);

    auto eventIdVToS = GetTPipePtr()->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    // pre norm
    for (uint32_t i = 0; i < tiling.bsCurLength; ++i) {
        uint32_t buffIndex = i * tiling.dhwAlignSize;
        Muls(inout[buffIndex], inout[buffIndex], tmp.GetValue(i), tiling.dhwAlignSize);
    }

    // tmp will be written later 
    auto eventIdSToV = GetTPipePtr()->FetchEventID(HardEvent::S_V);
    SetFlag<HardEvent::V_S>(eventIdSToV);
    WaitFlag<HardEvent::V_S>(eventIdSToV);

    PipeBarrier<PIPE_V>();
}

__aicore__ inline void GetGroupNormOutput(const LocalTensor<float>& inout,
    const LocalTensor<float>& gamma, const LocalTensor<float>& beta,
    const GroupNormTiling& tiling, const int32_t loopCount)
{
    size_t channelIndex = loopCount * tiling.meanVarRoundSize * tiling.d;
    for (uint32_t channel_offset = 0; channel_offset < tiling.bsCurLength * tiling.d; ++channel_offset) {
        Muls(inout[channel_offset * tiling.hwAlignSize], inout[channel_offset * tiling.hwAlignSize],
        gamma.GetValue(channelIndex % tiling.c), tiling.hw);
        channelIndex += 1;
    }    
    PipeBarrier<PIPE_V>();      

    channelIndex = loopCount * tiling.meanVarRoundSize * tiling.d;
    for (uint32_t channel_offset = 0; channel_offset < tiling.bsCurLength * tiling.d; ++channel_offset) {
        Adds(inout[channel_offset * tiling.hwAlignSize], inout[channel_offset * tiling.hwAlignSize],
        beta.GetValue(channelIndex % tiling.c), tiling.hw);
        channelIndex += 1;
    }    
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void GroupNormExe(const LocalTensor<half>& inputX,
    const LocalTensor<half>& gamma, const LocalTensor<half>& beta,
    const LocalTensor<half>& output, const LocalTensor<float>& outputMean, const LocalTensor<float>& outputVariance,
    const half epsilon, const GroupNormTiling& tiling, const GroupNormParams<float>& params, const int32_t loopCount)
{
    LocalTensor<float> tempTensorA = params.tempTensorA;
    LocalTensor<float> tempTensorB = params.tempTensorB;
    LocalTensor<float> tempTensorC = params.tempTensorC;
    Duplicate(tempTensorA, 0.0f, tiling.bshCurLength);
    PipeBarrier<PIPE_V>();
    Cast<float, half>(tempTensorB, inputX, RoundMode::CAST_NONE, tiling.inputRoundSize);
    PipeBarrier<PIPE_V>();

    GetGroupNormOutputMean(tempTensorB, tempTensorC, outputMean, tiling);

    GetGroupNormOutputVar(tempTensorB, tempTensorB, tempTensorC, outputMean, outputVariance, tiling);

    GetGroupNormOutputPre(tempTensorB, tempTensorA, outputVariance, tiling, static_cast<float>(epsilon));

    Cast<float, half>(tempTensorA, gamma, RoundMode::CAST_NONE, tiling.c);
    PipeBarrier<PIPE_V>();
    Cast<float, half>(tempTensorC, beta, RoundMode::CAST_NONE, tiling.c);
    PipeBarrier<PIPE_V>();

    GetGroupNormOutput(tempTensorB, tempTensorA, tempTensorC, tiling, loopCount);
   
    Cast<half, float>(output, tempTensorB, RoundMode::CAST_NONE, tiling.inputRoundSize);
    PipeBarrier<PIPE_V>();
}


__aicore__ inline void GroupNormExe(const LocalTensor<float>& inputX,
    const LocalTensor<float>& gamma, const LocalTensor<float>& beta,
    const LocalTensor<float>& output, const LocalTensor<float>& outputMean, const LocalTensor<float>& outputVariance,
    const float epsilon, const GroupNormTiling& tiling, const GroupNormParams<float>& params, const int32_t loopCount)
{
    LocalTensor<float> tempTensorA = params.tempTensorA;
    LocalTensor<float> tempTensorB = params.tempTensorB;
    LocalTensor<float> tempTensorC = params.tempTensorC;

    GetGroupNormOutputMean(inputX, output, outputMean, tiling);

    Duplicate(output, 0.0f, tiling.bshCurLength);
    PipeBarrier<PIPE_V>();

    GetGroupNormOutputVar(inputX, output, tempTensorC, outputMean, outputVariance, tiling);

    GetGroupNormOutputPre(output, tempTensorA, outputVariance, tiling, epsilon);

    GetGroupNormOutput(output, gamma, beta, tiling, loopCount);
}

__aicore__ inline void GroupNormExeSmallShape(const LocalTensor<half>& inputX,
    const LocalTensor<half>& gamma, const LocalTensor<half>& beta,
    const LocalTensor<half>& output, const LocalTensor<float>& outputMean, const LocalTensor<float>& outputVariance,
    const half epsilon, const GroupNormTiling& tiling, const GroupNormParams<float>& params, const int32_t loopCount)
{
    LocalTensor<float> tempTensorA = params.tempTensorA;
    LocalTensor<float> tempTensorB = params.tempTensorB;
    LocalTensor<float> tempTensorC = params.tempTensorC;
    Duplicate(tempTensorA, 0.0f, tiling.inputRoundSize * tiling.numberOfTmpBuf);
    PipeBarrier<PIPE_V>();

    Cast<float, half>(tempTensorB, inputX, RoundMode::CAST_NONE, tiling.inputRoundSize);
    PipeBarrier<PIPE_V>();

    uint32_t mask1 = GetGroupNormWholeReduceMask1(tiling);
    ASCENDC_ASSERT((mask1 > 0), { KERNEL_LOG(KERNEL_ERROR, "mask1 must > 0!"); });

    uint32_t repeat1 = tiling.dhwAlignSize / mask1 * tiling.meanVarRoundSize;
    uint32_t mask2 = tiling.dhwAlignSize / mask1 * GROUPNORM_MASK_SMALLEST_VAL;
    PipeBarrier<PIPE_V>();  

    WholeReduceSum<float, true>(tempTensorC, tempTensorB, mask1, repeat1, GROUPNORM_MASK_SMALLEST_VAL, DEFAULT_BLK_STRIDE, mask1 / GROUPNORM_MASK_SMALLEST_VAL);
    PipeBarrier<PIPE_V>();

    WholeReduceSum<float, true>(outputMean, tempTensorC, mask2, tiling.bsCurLength, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, mask2 / GROUPNORM_MASK_SMALLEST_VAL);
    PipeBarrier<PIPE_V>();

    Muls(outputMean, outputMean, tiling.factor, tiling.bsCurLength);
    auto eventIdVToS = GetTPipePtr()->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    for (uint32_t i = 0; i < tiling.bsCurLength; ++i) {
        uint32_t buffIndex = i * tiling.dhwAlignSize;
        Adds(tempTensorB[buffIndex], tempTensorB[buffIndex], -1.0f * outputMean.GetValue(i), tiling.hw, tiling.d,
        {1, 1, static_cast<uint8_t>(tiling.hwAlignSize / GROUPNORM_ONE_BLK_SIZE), static_cast<uint8_t>(tiling.hwAlignSize / GROUPNORM_ONE_BLK_SIZE)});
    }
    PipeBarrier<PIPE_V>();  

    Mul(tempTensorC, tempTensorB, tempTensorB, tiling.bshCurLength);
    PipeBarrier<PIPE_V>();

    WholeReduceSum<float, true>(tempTensorA, tempTensorC, mask1, repeat1, GROUPNORM_MASK_SMALLEST_VAL, DEFAULT_BLK_STRIDE, mask1 / GROUPNORM_MASK_SMALLEST_VAL);
    PipeBarrier<PIPE_V>();

    WholeReduceSum<float, true>(outputVariance, tempTensorA, mask2, tiling.bsCurLength, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, mask2 / GROUPNORM_MASK_SMALLEST_VAL);
    PipeBarrier<PIPE_V>();

    Muls(outputVariance, outputVariance, tiling.factor, tiling.bsCurLength);
    PipeBarrier<PIPE_V>();

    GetGroupNormOutputPre(tempTensorB, tempTensorA, outputVariance, tiling, static_cast<float>(epsilon));

    Cast<float, half>(tempTensorA, gamma, RoundMode::CAST_NONE, tiling.c);
    PipeBarrier<PIPE_V>();
    Cast<float, half>(tempTensorC, beta, RoundMode::CAST_NONE, tiling.c);
    PipeBarrier<PIPE_V>();

    GetGroupNormOutput(tempTensorB, tempTensorA, tempTensorC, tiling, loopCount);
   
    Cast<half, float>(output, tempTensorB, RoundMode::CAST_NONE, tiling.inputRoundSize);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void GroupNormExeSmallShape(const LocalTensor<float>& inputX,
    const LocalTensor<float>& gamma, const LocalTensor<float>& beta,
    const LocalTensor<float>& output, const LocalTensor<float>& outputMean, const LocalTensor<float>& outputVariance,
    const float epsilon, const GroupNormTiling& tiling, const GroupNormParams<float>& params, const int32_t loopCount)
{
    LocalTensor<float> tempTensorA = params.tempTensorA;
    LocalTensor<float> tempTensorB = params.tempTensorB;
    LocalTensor<float> tempTensorC = params.tempTensorC;
    Duplicate(output, 0.0f, tiling.inputRoundSize);
    PipeBarrier<PIPE_V>();
    Duplicate(tempTensorC, 0.0f, tiling.inputRoundSize);
    PipeBarrier<PIPE_V>();
    uint32_t mask1 = GetGroupNormWholeReduceMask1(tiling);
    ASCENDC_ASSERT((mask1 > 0), { KERNEL_LOG(KERNEL_ERROR, "mask1 must > 0!"); });

    uint32_t repeat1 = tiling.dhwAlignSize / mask1 * tiling.meanVarRoundSize;
    uint32_t mask2 = tiling.dhwAlignSize / mask1 * GROUPNORM_MASK_SMALLEST_VAL;
    PipeBarrier<PIPE_V>();

    WholeReduceSum<float, true>(tempTensorC, inputX, mask1, repeat1, GROUPNORM_MASK_SMALLEST_VAL, DEFAULT_BLK_STRIDE, mask1 / GROUPNORM_MASK_SMALLEST_VAL);
    PipeBarrier<PIPE_V>();

    WholeReduceSum<float, true>(outputMean, tempTensorC, mask2, tiling.bsCurLength, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, mask2 / GROUPNORM_MASK_SMALLEST_VAL);
    PipeBarrier<PIPE_V>();

    Muls(outputMean, outputMean, tiling.factor, tiling.bsCurLength);
    auto eventIdVToS = GetTPipePtr()->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    auto repeatStride = tiling.hwAlignSize / GROUPNORM_ONE_BLK_SIZE;
    for (uint32_t i = 0; i < tiling.bsCurLength; ++i) {
        uint32_t buffIndex = i * tiling.dhwAlignSize;
        Adds(output[buffIndex], inputX[buffIndex], -1.0f * outputMean.GetValue(i), tiling.hw, tiling.d,
        {1, 1, static_cast<uint8_t>(repeatStride), static_cast<uint8_t>(repeatStride)});
    }
    PipeBarrier<PIPE_V>();

    Mul(tempTensorC, output, output, tiling.bshCurLength);
    PipeBarrier<PIPE_V>();

    Duplicate(tempTensorA, 0.0f, tiling.inputRoundSize);
    PipeBarrier<PIPE_V>();

    WholeReduceSum<float, true>(tempTensorA, tempTensorC, mask1, repeat1, GROUPNORM_MASK_SMALLEST_VAL, DEFAULT_BLK_STRIDE, mask1 / GROUPNORM_MASK_SMALLEST_VAL);
    PipeBarrier<PIPE_V>();

    WholeReduceSum<float, true>(outputVariance, tempTensorA, mask2, tiling.bsCurLength, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, mask2 / GROUPNORM_MASK_SMALLEST_VAL);
    PipeBarrier<PIPE_V>();

    Muls(outputVariance, outputVariance, tiling.factor, tiling.bsCurLength);
    PipeBarrier<PIPE_V>();
    GetGroupNormOutputPre(output, tempTensorA, outputVariance, tiling, epsilon);

    GetGroupNormOutput(output, gamma, beta, tiling, loopCount);
}

template <bool isReuseSource = false>
__aicore__ inline void GetGroupNormNDTensorInfo(const LocalTensor<half>& inputX,
    const LocalTensor<half>& outputMean, const LocalTensor<half>& outputVariance,
    const LocalTensor<float>& stackBuffer, const GroupNormTiling& tiling, GroupNormParams<float>& params)
{
    params.tempTensorA = stackBuffer[tiling.firstTmpStartPos];
    params.tempTensorB = stackBuffer[tiling.secondTmpStartPos];
    params.tempTensorC = stackBuffer[tiling.thirdTmpStartPos];
    params.meanTmpTensor = stackBuffer[tiling.meanTmpTensorPos];
    params.varianceTmpTensor = stackBuffer[tiling.varianceTmpTensorPos];

    ASCENDC_ASSERT((tiling.thirdTmpStartPos + tiling.oneTmpSize <= tiling.tmpBufSize), {
        KERNEL_LOG(KERNEL_ERROR, "thirdTmpStartPos + oneTmpSize is (%d) should <= tmpBufSize is (%d)",
        tiling.thirdTmpStartPos + tiling.oneTmpSize, tiling.tmpBufSize);
    });
    ASCENDC_ASSERT((stackBuffer.GetSize() >= tiling.tmpBufSize), {
        KERNEL_LOG(KERNEL_ERROR, "stackBuffer.GetSize is (%d) should >= tmpBufSize is (%d)",
        stackBuffer.GetSize(), tiling.tmpBufSize);
    });
}

template <bool isReuseSource = false>
__aicore__ inline void GetGroupNormNDTensorInfo(const LocalTensor<float>& inputX,
    const LocalTensor<float>& outputMean, const LocalTensor<float>& outputVariance,
    const LocalTensor<float>& stackBuffer, const GroupNormTiling& tiling, GroupNormParams<float>& params)
{
    params.meanTmpTensor = outputMean;
    params.varianceTmpTensor = outputVariance;

    if constexpr (isReuseSource) {
        params.tempTensorA = inputX;
        params.tempTensorB = stackBuffer[tiling.firstTmpStartPos];
        params.tempTensorC = stackBuffer[tiling.secondTmpStartPos];

        ASCENDC_ASSERT((tiling.secondTmpStartPos + tiling.oneTmpSize <= tiling.tmpBufSize), {
            KERNEL_LOG(KERNEL_ERROR, "secondTmpStartPos + oneTmpSize is (%d) should <= tmpBufSize is (%d)",
            tiling.secondTmpStartPos + tiling.oneTmpSize, tiling.tmpBufSize);
        });
    } else {
        params.tempTensorA = stackBuffer[tiling.firstTmpStartPos];
        params.tempTensorB = stackBuffer[tiling.secondTmpStartPos];
        params.tempTensorC = stackBuffer[tiling.thirdTmpStartPos];
       
        ASCENDC_ASSERT((tiling.thirdTmpStartPos + tiling.oneTmpSize <= tiling.tmpBufSize), {
            KERNEL_LOG(KERNEL_ERROR, "thirdTmpStartPos + oneTmpSize is (%d) should <= tmpBufSize is (%d)",
            tiling.thirdTmpStartPos + tiling.oneTmpSize, tiling.tmpBufSize);
        });
    }
   
    ASCENDC_ASSERT((stackBuffer.GetSize() >= tiling.tmpBufSize), {
        KERNEL_LOG(KERNEL_ERROR, "stackBuffer.GetSize is (%d) should >= tmpBufSize is (%d)",
        stackBuffer.GetSize(), tiling.tmpBufSize);
    });
}

__aicore__ inline void GetOutputMeanVariance(const LocalTensor<half>& outputMean,
    const LocalTensor<half>& outputVariance, const GroupNormTiling& tiling, const GroupNormParams<float>& params)
{
    Cast<half, float>(outputMean, params.meanTmpTensor, RoundMode::CAST_NONE, tiling.n * tiling.g);
    Cast<half, float>(outputVariance, params.varianceTmpTensor, RoundMode::CAST_NONE, tiling.n * tiling.g);
}

template <typename T>
__aicore__ inline void GroupNormNDCommon(const LocalTensor<T>& inputX,
    const LocalTensor<T>& gamma, const LocalTensor<T>& beta,
    const LocalTensor<T>& output, const LocalTensor<T>& outputMean, const LocalTensor<T>& outputVariance,
    const T epsilon, GroupNormTiling& tiling, const GroupNormParams<float>& params)
{
    uint32_t inputOffset = 0;
    uint32_t mvOffset = 0;

    if (tiling.smallShape) {
        for (uint32_t index = 0; index < tiling.loopRound; index++) {
            GroupNormExeSmallShape(inputX[inputOffset], gamma, beta, output[inputOffset],
            params.meanTmpTensor[mvOffset],
            params.varianceTmpTensor[mvOffset], epsilon, tiling, params, index);

            inputOffset += tiling.inputRoundSize;
            mvOffset += tiling.meanVarRoundSize;
        }
    } else {
        for (uint32_t index = 0; index < tiling.loopRound; index++) {
            GroupNormExe(inputX[inputOffset], gamma, beta, output[inputOffset],
            params.meanTmpTensor[mvOffset],
            params.varianceTmpTensor[mvOffset], epsilon, tiling, params, index);

            inputOffset += tiling.inputRoundSize;
            mvOffset += tiling.meanVarRoundSize;
        }
    }

    if (tiling.inputTailSize > 0) {
        tiling.bshCurLength = tiling.inputTailSize;
        tiling.bsCurLength = tiling.meanVarTailSize;

        inputOffset = tiling.inputTailPos;
        mvOffset = tiling.meanVarTailPos;

        if (tiling.smallShape) {
            GroupNormExeSmallShape(inputX[inputOffset], gamma, beta, output[inputOffset],
            params.meanTmpTensor[mvOffset],
            params.varianceTmpTensor[mvOffset], epsilon, tiling, params, tiling.loopRound);
        } else {
            GroupNormExe(inputX[inputOffset], gamma, beta, output[inputOffset],
            params.meanTmpTensor[mvOffset],
            params.varianceTmpTensor[mvOffset], epsilon, tiling, params, tiling.loopRound);
        }

        // revert to normal round size from tail size, for the next iteration calculation
        tiling.bshCurLength = tiling.inputRoundSize;
        tiling.bsCurLength = tiling.meanVarRoundSize;
    }

    if constexpr (sizeof(T) == sizeof(half)) {
        GetOutputMeanVariance(outputMean, outputVariance, tiling, params);
    }
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void GroupNormImpl(const LocalTensor<T>& output,
    const LocalTensor<T>& outputMean, const LocalTensor<T>& outputVariance,
    const LocalTensor<T>& inputX, const LocalTensor<T>& gamma, const LocalTensor<T>& beta,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const T epsilon, GroupNormTiling& tiling)
{
    ASCENDC_ASSERT((tiling.oneTmpSize > 0), { KERNEL_LOG(KERNEL_ERROR, "tiling.oneTmpSize must > 0!"); });

    if ASCEND_IS_AIC {
        return;
    }
    LocalTensor<float> stackBuffer = sharedTmpBuffer.ReinterpretCast<float>();
    ASCENDC_ASSERT((stackBuffer.GetSize() > 0),{ KERNEL_LOG(KERNEL_ERROR, "sharedTmpBuffer Size must > 0!"); });

    GroupNormParams<float> params;
    GetGroupNormNDTensorInfo<isReuseSource>(inputX, outputMean, outputVariance, stackBuffer, tiling, params);

    GroupNormNDCommon<T>(inputX, gamma, beta, output, outputMean, outputVariance, epsilon, tiling, params);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void GroupNormImpl(const LocalTensor<T>& output,
    const LocalTensor<T>& outputMean, const LocalTensor<T>& outputVariance,
    const LocalTensor<T>& inputX, const LocalTensor<T>& gamma, const LocalTensor<T>& beta,
    const T epsilon, GroupNormTiling& tiling)
{
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });

    GroupNormImpl<T, isReuseSource>(output, outputMean, outputVariance, inputX, gamma, beta, sharedTmpBuffer, epsilon, tiling);
}

} // namespace AscendC
#endif // IMPL_NORMALIZATION_GROUPNORM_GROUPNORM_COMMON_IMPL_H