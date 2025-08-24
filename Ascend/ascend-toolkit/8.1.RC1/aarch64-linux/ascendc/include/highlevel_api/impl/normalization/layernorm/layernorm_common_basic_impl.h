/* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file layernorm_common_basic_impl.h
 * \brief
 */
#ifndef IMPL_NORMALIZATION_LAYERNORM_LAYERNORM_COMMON_BASIC_IMPL_H
#define IMPL_NORMALIZATION_LAYERNORM_LAYERNORM_COMMON_BASIC_IMPL_H
#include "lib/normalization/layernorm_utils.h"
namespace AscendC {
constexpr uint32_t MASK_LOW_6BITS = 0x3f;
constexpr uint32_t MASK_HIGH_26BITS = 0xFFFFFFC0;
template <typename T>
struct LayerNormRstdTmpTensorParams {
    __aicore__ LayerNormRstdTmpTensorParams(){};
    LocalTensor<T> tempTensorA;
    LocalTensor<T> tempTensorB;
    LocalTensor<T> varianceTmpTensor;
};

struct LayerNormPara {
    uint32_t aLength;
    uint32_t rLength;
    uint32_t rLengthWithPadding;
};

struct WelfordUpdateParam {
    uint32_t rnLength;
    uint32_t abLength;
    uint32_t abComputeLength;
    float nRec;
};

template <bool isRelocate = true, bool isTransposeDst = false>
__aicore__ inline void LayerNormReduceSumImpl(const LocalTensor<float>& dstMVTmp, const LocalTensor<float>& dst,
    const LocalTensor<float>& src, const uint32_t bsLength, const uint32_t hLength)
{
    ResetMask();
    SetMaskNorm();
    // Contract the horizontal axis to one repeat length 64 (2^6)
    constexpr uint32_t rightShiftSix = 6;
    if (hLength > ONE_REPEAT_FLOAT_SIZE) {
        uint32_t addRepeatTime = (hLength >> rightShiftSix) - 1;
        uint32_t addTailNumber = (hLength & MASK_LOW_6BITS);
        if ((hLength & MASK_LOW_6BITS) == 0) {
            for (uint32_t i = 0; i < bsLength * hLength; i += hLength) {
                LocalTensor<float> dstTmp = src[i];
                LocalTensor<float> srcTmp = src[i + ONE_REPEAT_FLOAT_SIZE];
                Add(dstTmp, srcTmp, dstTmp, ONE_REPEAT_FLOAT_SIZE, addRepeatTime,
                    { DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, 0, DEFAULT_REPEAT_STRIDE, 0 });
                PipeBarrier<PIPE_V>();
            }
        } else if (addRepeatTime > 0) {
            for (uint32_t i = 0; i < bsLength * hLength; i += hLength) {
                LocalTensor<float> dstTmp = src[i];
                LocalTensor<float> srcTmp = src[i + ONE_REPEAT_FLOAT_SIZE];
                LocalTensor<float> srcTailTmp = src[i + (hLength & MASK_HIGH_26BITS)];
                Add(dstTmp, srcTmp, dstTmp, ONE_REPEAT_FLOAT_SIZE, addRepeatTime,
                    { DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, 0, DEFAULT_REPEAT_STRIDE, 0 });
                PipeBarrier<PIPE_V>();
                Add(dstTmp, srcTailTmp, dstTmp, addTailNumber, 1,
                    { DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, 0, DEFAULT_REPEAT_STRIDE, 0 });
                PipeBarrier<PIPE_V>();
            }
        } else {
            for (uint32_t i = 0; i < bsLength * hLength; i += hLength) {
                LocalTensor<float> dstTmp = src[i];
                LocalTensor<float> srcTailTmp = src[i + (hLength & MASK_HIGH_26BITS)];
                Add(dstTmp, srcTailTmp, dstTmp, addTailNumber, 1,
                    { DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, 0, DEFAULT_REPEAT_STRIDE, 0 });
                PipeBarrier<PIPE_V>();
            }
        }
    }

    uint32_t repeatTime = bsLength;
    uint32_t cursorSrc = 0;
    uint32_t wholeReduceSumHLength = (hLength > ONE_REPEAT_FLOAT_SIZE) ? ONE_REPEAT_FLOAT_SIZE : hLength;
    constexpr uint32_t rightShiftThree = 3;
    const uint32_t reduceSumSrcRepeatStride = hLength >> rightShiftThree;

    while (repeatTime >= MAX_REPEAT_TIMES) {
        LocalTensor<float> srcTmp = src[cursorSrc * MAX_REPEAT_TIMES * hLength];
        LocalTensor<float> dstTmp = dst[cursorSrc * MAX_REPEAT_TIMES * hLength];
        if constexpr (isRelocate) {
            WholeReduceSum<float>(dstMVTmp[cursorSrc * MAX_REPEAT_TIMES], srcTmp, wholeReduceSumHLength,
                MAX_REPEAT_TIMES, 1, DEFAULT_BLK_STRIDE, reduceSumSrcRepeatStride);
        }
        WholeReduceSum<float>(dstTmp, srcTmp, wholeReduceSumHLength, MAX_REPEAT_TIMES, hLength, DEFAULT_BLK_STRIDE,
            reduceSumSrcRepeatStride);
        PipeBarrier<PIPE_V>();
        repeatTime -= MAX_REPEAT_TIMES;
        ++cursorSrc;
    }

    uint32_t reduceSumSrcRepeatTimeTail = bsLength - cursorSrc * MAX_REPEAT_TIMES;
    if (reduceSumSrcRepeatTimeTail > 0) {
        LocalTensor<float> srcTmp = src[cursorSrc * MAX_REPEAT_TIMES * hLength];
        LocalTensor<float> dstTmp = dst[cursorSrc * MAX_REPEAT_TIMES * hLength];
        if constexpr (isRelocate) {
            WholeReduceSum<float>(dstMVTmp[cursorSrc * MAX_REPEAT_TIMES], srcTmp, wholeReduceSumHLength,
                reduceSumSrcRepeatTimeTail, 1, DEFAULT_BLK_STRIDE, reduceSumSrcRepeatStride);
        }
        WholeReduceSum<float>(dstTmp, srcTmp, wholeReduceSumHLength, reduceSumSrcRepeatTimeTail, hLength,
            DEFAULT_BLK_STRIDE, reduceSumSrcRepeatStride);
        PipeBarrier<PIPE_V>();
    }

    SetMaskCount();
}

__aicore__ inline void BroadcastLastDim(const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const uint32_t bsLength, const uint32_t hLength)
{
    SetVectorMask<uint8_t, MaskMode::COUNTER>(0, hLength);

    SetCmpMask<float>(src);
    PipeBarrier<PIPE_V>();

    LocalTensor<int16_t> maskLocal = src.ReinterpretCast<int16_t>();

    const UnaryRepeatParams unaryParams;
    Muls<int16_t, false>(maskLocal, maskLocal, 0, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    const BinaryRepeatParams binaryParams;
    Select<float, int16_t>(dst, maskLocal, dst, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    for (uint32_t i = 1; i < bsLength; i++) {
        SetCmpMask<float>(src[i * hLength]);
        PipeBarrier<PIPE_V>();

        Select<float, int16_t>(dst[i * hLength], maskLocal, dst, 1, binaryParams);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void DuplicateMulImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
    const LocalTensor<float>& src1, const uint32_t bsLength, const uint32_t hLength)
{
    const BinaryRepeatParams binaryParams;
    for (uint32_t i = 0; i < bsLength; i++) {
        Mul<float, false>(dst[i * hLength], src0[i * hLength], src1, MASK_PLACEHOLDER, 1, binaryParams);
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void DuplicateAddImpl(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
    const LocalTensor<float>& src1, const uint32_t bsLength, const uint32_t hLength)
{
    const BinaryRepeatParams binaryParams;
    for (uint32_t i = 0; i < bsLength; i++) {
        Add<float, false>(dst[i * hLength], src0[i * hLength], src1, MASK_PLACEHOLDER, 1, binaryParams);
    }
    PipeBarrier<PIPE_V>();
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void GetLayerNormNDTensorInfo(const LocalTensor<T>& inputX, const LocalTensor<T>& outputMean,
    const LocalTensor<T>& outputVariance, const LocalTensor<float>& stackBuffer, const LayerNormTiling& tiling,
    LayerNormParams<float>& params)
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
        KERNEL_LOG(KERNEL_ERROR, "stackBuffer.GetSize is (%d) should >= tiling.tmpBufSize is (%d)",
            stackBuffer.GetSize(), tiling.tmpBufSize);
    });
}

template <>
__aicore__ inline void GetLayerNormNDTensorInfo<float, false>(const LocalTensor<float> &inputX,
    const LocalTensor<float> &outputMean, const LocalTensor<float> &outputVariance,
    const LocalTensor<float> &stackBuffer, const LayerNormTiling &tiling, LayerNormParams<float> &params)
{
    params.meanTmpTensor = outputMean;
    params.varianceTmpTensor = outputVariance;

    params.tempTensorA = stackBuffer[tiling.firstTmpStartPos];
    params.tempTensorB = stackBuffer[tiling.secondTmpStartPos];
    params.tempTensorC = stackBuffer[tiling.thirdTmpStartPos];

    ASCENDC_ASSERT((tiling.thirdTmpStartPos + tiling.oneTmpSize <= tiling.tmpBufSize), {
        KERNEL_LOG(KERNEL_ERROR, "thirdTmpStartPos + oneTmpSize is (%d) should <= tmpBufSize is (%d)",
            tiling.thirdTmpStartPos + tiling.oneTmpSize, tiling.tmpBufSize);
    });

    ASCENDC_ASSERT((stackBuffer.GetSize() >= tiling.tmpBufSize), {
        KERNEL_LOG(KERNEL_ERROR, "stackBuffer.GetSize is (%d) >= tiling.tmpBufSize is (%d)", stackBuffer.GetSize(),
            tiling.tmpBufSize);
    });
}

template <>
__aicore__ inline void GetLayerNormNDTensorInfo<float, true>(const LocalTensor<float> &inputX,
    const LocalTensor<float> &outputMean, const LocalTensor<float> &outputVariance,
    const LocalTensor<float> &stackBuffer, const LayerNormTiling &tiling, LayerNormParams<float> &params)
{
    params.meanTmpTensor = outputMean;
    params.varianceTmpTensor = outputVariance;

    params.tempTensorA = inputX;
    params.tempTensorB = stackBuffer[tiling.firstTmpStartPos];
    params.tempTensorC = stackBuffer[tiling.secondTmpStartPos];

    ASCENDC_ASSERT((tiling.secondTmpStartPos + tiling.oneTmpSize <= tiling.tmpBufSize), {
        KERNEL_LOG(KERNEL_ERROR, "secondTmpStartPos + oneTmpSize is (%d) should <= tmpBufSize is (%d)",
            tiling.secondTmpStartPos + tiling.oneTmpSize, tiling.tmpBufSize);
    });

    ASCENDC_ASSERT((stackBuffer.GetSize() >= tiling.tmpBufSize), {
        KERNEL_LOG(KERNEL_ERROR, "stackBuffer.GetSize is (%d) >= tiling.tmpBufSize is (%d)", stackBuffer.GetSize(),
            tiling.tmpBufSize);
    });
}

__aicore__ inline void GetOutputMeanVariance(const LocalTensor<half>& outputMean,
    const LocalTensor<half>& outputVariance, const LayerNormTiling& tiling, const LayerNormParams<float>& params)
{
    SetVectorMask<uint8_t, MaskMode::COUNTER>(0, tiling.meanVarSize);

    UnaryRepeatParams unaryParams;
    unaryParams.dstRepStride = DEFAULT_REPEAT_STRIDE / sizeof(half);

    Cast<half, float, false>(outputMean, params.meanTmpTensor, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Cast<half, float, false>(outputVariance, params.varianceTmpTensor, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        unaryParams);
    PipeBarrier<PIPE_V>();
}

template <bool isReuseSource = false>
__aicore__ inline void GetLayerNormRstdTensorInfo(const LocalTensor<float>& stackBuffer,
    const LayerNormSeparateTiling& tiling, LayerNormRstdTmpTensorParams<float>& params)
{
    params.tempTensorA = stackBuffer[tiling.firstTmpStartPos];
    params.tempTensorB = stackBuffer[tiling.secondTmpStartPos];
    ASCENDC_ASSERT((tiling.secondTmpStartPos + tiling.oneTmpSize <= tiling.tmpBufSize), {
        KERNEL_LOG(KERNEL_ERROR, "secondTmpStartPos + oneTmpSize is (%d) should <= tmpBufSize is (%d)",
            tiling.secondTmpStartPos + tiling.oneTmpSize, tiling.tmpBufSize);
    });
    ASCENDC_ASSERT((stackBuffer.GetSize() >= tiling.tmpBufSize), {
        KERNEL_LOG(KERNEL_ERROR, "stackBuffer.GetSize is (%d) should >= tiling.tmpBufSize is (%d)",
            stackBuffer.GetSize(), tiling.tmpBufSize);
    });
}

template <typename U, typename T, const LayerNormConfig& config = LNCFG_NORM>
__aicore__ inline void CheckLayerNormRstd(const LocalTensor<float> stackBuffer, const LayerNormPara& para) {
    static_assert(SupportType<T, half, float>(), "current data type is not supported on current device!");
    if constexpr (IsSameType<T, half>::value) {
        static_assert(SupportType<U, half, float>(), "current data type is not supported on current device!");
    } else if constexpr (IsSameType<T, float>::value) {
        static_assert(SupportType<U, float>(), "current data type is not supported on current device!");
    }
    static_assert(config.isOnlyOutput == false, "current data type is not supported on current device!");
    ASCENDC_ASSERT((stackBuffer.GetSize() > para.rLengthWithPadding),
        { KERNEL_LOG(KERNEL_ERROR, "sharedTmpBuffer size must > LayerNormMinTmpSize!"); });
}

template <typename T>
__aicore__ inline void LayerNormPreProc(const LocalTensor<T>& inputX, const LocalTensor<float>& stackBuffer,
    const LayerNormPara& para)
{
    const LocalTensor<T> tempTensor = stackBuffer.ReinterpretCast<T>();
    Duplicate(tempTensor, (T)0, para.rLengthWithPadding);
    PipeBarrier<PIPE_V>();
    Adds(tempTensor, tempTensor, (T)1, para.rLength);
    PipeBarrier<PIPE_V>();
    for (int i = 0; i < para.aLength; i++) {
        Mul(inputX[i * para.rLengthWithPadding], inputX[i * para.rLengthWithPadding], tempTensor,
            para.rLengthWithPadding);
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void WelfordUpdateComputeMean(const LocalTensor<float>& tmpVreg, const LocalTensor<float>& src,
    const LocalTensor<float>& inMean, const LocalTensor<float>& outVreg, const LocalTensor<float>& outMean,
    const UnaryRepeatParams unaryParams, const BinaryRepeatParams binaryParams, const WelfordUpdateParam &para)
{
    PipeBarrier<PIPE_V>();
    Sub<float, false>(tmpVreg, src, inMean, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
    Muls<float, false>(outVreg, tmpVreg, static_cast<float>(para.nRec), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Add<float, false>(outMean, outVreg, inMean, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void WelfordUpdateComputeVar(const LocalTensor<float>& tmpVreg, const LocalTensor<float>& inVar,
    const LocalTensor<float>& outVar, const UnaryRepeatParams unaryParams, const BinaryRepeatParams binaryParams,
    const WelfordUpdateParam &para)
{
    PipeBarrier<PIPE_V>();
    Add<float, false>(outVar, tmpVreg, inVar, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
}

template <typename T, typename U, bool isReuseSource = false>
__aicore__ inline constexpr uint32_t WelfordUpdateGetTmpSize()
{
    if constexpr (sizeof(T) == sizeof(half)) {
        return 0x3;
    }

    if constexpr (isReuseSource) {
        return 1;
    }
    return 0x2;
}

__aicore__ inline void GetLayerNormOutputMean(const LocalTensor<float>& outputMean, const LocalTensor<float>& inputX,
    const LayerNormTiling& tiling, const LayerNormParams<float>& params, const LocalTensor<float>& tmpMean)
{
    SetVectorMask<uint8_t, MaskMode::COUNTER>(0, tiling.bshCurLength);

    const UnaryRepeatParams unaryParams;
    Muls<float, false>(params.tempTensorC, inputX, tiling.lastDimValueBack, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    LayerNormReduceSumImpl(tmpMean, outputMean, params.tempTensorC, tiling.bsCurLength, tiling.hLength);
}

__aicore__ inline void GetLayerNormOutputVariance(const LocalTensor<float>& outputVariance,
    const LocalTensor<float>& inputX, const LocalTensor<float>& inputMean, const LayerNormTiling& tiling,
    const LayerNormParams<float>& params, const LocalTensor<float>& tmpVariance)
{
    LocalTensor<float> tempTensorA = params.tempTensorA;
    LocalTensor<float> tempTensorB = params.tempTensorB;
    LocalTensor<float> tempTensorC = params.tempTensorC;

    BroadcastLastDim(tempTensorC, inputMean, tiling.bsCurLength, tiling.hLength);

    SetVectorMask<uint8_t, MaskMode::COUNTER>(0, tiling.bshCurLength);

    const BinaryRepeatParams binaryParams;
    Sub<float, false>(tempTensorB, inputX, tempTensorC, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    Mul<float, false>(tempTensorC, tempTensorB, tempTensorB, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    const UnaryRepeatParams unaryParams;
    Muls<float, false>(tempTensorA, tempTensorC, tiling.lastDimValueBack, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    LayerNormReduceSumImpl(tmpVariance, outputVariance, tempTensorA, tiling.bsCurLength, tiling.hLength);
    PipeBarrier<PIPE_V>();
}

template <typename U>
__aicore__ inline void WelfordUpdateInplaceCompute(const LocalTensor<U>& outMean, const LocalTensor<U>& outVar,
    const LocalTensor<U>& inMean, const LocalTensor<U>& inVar, const WelfordUpdateParam &para, uint32_t alignNum)
{
    uint32_t inPlaceLength = AlignUp(para.abLength - para.abComputeLength, alignNum);
    uint32_t dstOffset = para.abLength - inPlaceLength;

    DataCopy(outMean[dstOffset], inMean[dstOffset], inPlaceLength);
    DataCopy(outVar[dstOffset], inVar[dstOffset], inPlaceLength);
    PipeBarrier<PIPE_V>();
}
__aicore__ inline void WelfordUpdateInplace(const LocalTensor<float>& outMean, const LocalTensor<float>& outVar,
    const LocalTensor<float>& inMean, const LocalTensor<float>& inVar, const WelfordUpdateParam &para)
{
    WelfordUpdateInplaceCompute(outMean, outVar, inMean, inVar, para, B32_DATA_NUM_PER_BLOCK);
}

__aicore__ inline void WelfordUpdateInplace(const LocalTensor<half>& outMean, const LocalTensor<half>& outVar,
    const LocalTensor<half>& inMean, const LocalTensor<half>& inVar, const WelfordUpdateParam &para)
{
    WelfordUpdateInplaceCompute(outMean, outVar, inMean, inVar, para, B16_DATA_NUM_PER_BLOCK);
}

__aicore__ inline void GetLayerNormOutputPre(const LocalTensor<float>& xSubMean,
    const LocalTensor<float>& inputVariance, const float epsilon, const LayerNormTiling& tiling,
    const LayerNormParams<float>& params)
{
    const float exponent = -0.5;
    LocalTensor<float> tempTensorA = params.tempTensorA;
    LocalTensor<float> tempTensorB = params.tempTensorB;
    LocalTensor<float> tempTensorC = params.tempTensorC;

    BroadcastLastDim(tempTensorA, inputVariance, tiling.bsCurLength, tiling.hLength);

    SetVectorMask<uint8_t, MaskMode::COUNTER>(0, tiling.bshCurLength);

    const UnaryRepeatParams unaryParams;
    Adds<float, false>(tempTensorC, tempTensorA, epsilon, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Sqrt<float, false>(tempTensorA, tempTensorC, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    SetVectorMask<uint8_t, MaskMode::COUNTER>(0, B32_DATA_NUM_PER_BLOCK);
    Duplicate<float, false>(tempTensorC, 1, MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();

    SetVectorMask<uint8_t, MaskMode::COUNTER>(0, tiling.bshCurLength);
    Div<float, false>(tempTensorA, tempTensorC, tempTensorA, MASK_PLACEHOLDER, 1,
        { 1, 0, 1, DEFAULT_REPEAT_STRIDE, 0, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();

    const BinaryRepeatParams binaryParams;
    Mul<float, false>(tempTensorC, tempTensorA, xSubMean, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
}

}  // namespace AscendC
#endif  // IMPL_NORMALIZATION_LAYERNORM_LAYERNORM_COMMON_BASIC_IMPL_H