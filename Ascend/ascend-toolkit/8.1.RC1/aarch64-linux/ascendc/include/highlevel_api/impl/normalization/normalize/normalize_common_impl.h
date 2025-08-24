/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file normalize_common_impl.h
 * \brief
 */
#ifndef IMPL_NORMALIZATION_NORMALIZE_NORMALIZE_COMMON_IMPL_H
#define IMPL_NORMALIZATION_NORMALIZE_NORMALIZE_COMMON_IMPL_H

#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "lib/normalization/normalize_utils.h"

namespace AscendC {
const float DEFAULT_EPSILON = 1e-5;

struct NormalizePara {
    uint32_t aLength;
    uint32_t rLength;
    uint32_t rLengthWithPadding;
};

template <typename T>
struct NormalizeTmpTensor {
    __aicore__ NormalizeTmpTensor(){};
    LocalTensor<T> tempTensorA;
    LocalTensor<T> tempTensorB;
    LocalTensor<T> gammaTmpTensor;
    LocalTensor<T> betaTmpTensor;
};

template <typename U, typename T>
__aicore__ inline constexpr bool IsDtypeValid()
{
    // T for input, U for gamma + beta. The precision of U cannot be less than T
    constexpr bool isValid1 = (IsSameType<T, float>::value) && (IsSameType<U, float>::value);
    constexpr bool isValid2 = (IsSameType<T, half>::value) && (IsSameType<U, half>::value);
    constexpr bool isValid3 = (IsSameType<T, half>::value) && (IsSameType<U, float>::value);
    return isValid1 || isValid2 || isValid3;
}

template <const NormalizeConfig& config>
__aicore__ inline bool CheckParams(const NormalizePara& para)
{
    static_assert(config.isOnlyOutput == false, "isOnlyOutput must be set false for now.");
    static_assert(config.aLength != 1, "aLength in config must not be 1.");
    if constexpr(config.aLength != -1) {
        ASCENDC_ASSERT((config.aLength == para.aLength), {KERNEL_LOG(KERNEL_ERROR,
            "config.aLength must equal to para.aLength.");});
    }
    return true;
}

template <typename U, typename T>
__aicore__ inline void GetNormalizeTensorInfo(const LocalTensor<float>& stackBuffer, const NormalizePara& para,
    NormalizeTmpTensor<float>& tempTensor, uint32_t& N)
{
    // total needed space: A(rstd) + 2R(gamma + beta, half->float) + 2R * N (based on space left)
    // min space is at least A FP32. Thus use A FP32 calculate first, then these A FP32 used for 2R + 2R * N calculation
    ASCENDC_ASSERT((stackBuffer.GetSize() >= para.aLength), {KERNEL_LOG(KERNEL_ERROR,
        "StackBuffer size must be at least A FP32!");});
    uint32_t R2 = para.rLengthWithPadding * 2;   // 2 * R
    if constexpr(IsSameType<U, float>::value) {
        N = stackBuffer.GetSize() / R2;
        N = (N >= para.aLength) ? para.aLength : N;
        ASCENDC_ASSERT((N > 0), {KERNEL_LOG(KERNEL_ERROR, "Stackbuffer size is too small!");});
        tempTensor.tempTensorA = stackBuffer[0];                                    // NR length tmp buffer 1
        tempTensor.tempTensorB = stackBuffer[N * para.rLengthWithPadding];          // NR length tmp buffer 2
    } else {
        N = (stackBuffer.GetSize() - R2) / R2;
        N = (N >= para.aLength) ? para.aLength : N;
        ASCENDC_ASSERT((N > 0), {KERNEL_LOG(KERNEL_ERROR, "Stackbuffer size is too small!");});
        tempTensor.tempTensorA = stackBuffer[0];                                    // NR length tmp buffer 1
        tempTensor.tempTensorB = stackBuffer[N * para.rLengthWithPadding];          // NR length tmp buffer 2
        tempTensor.gammaTmpTensor = stackBuffer[R2 * N];                            // R    
        tempTensor.betaTmpTensor = stackBuffer[R2 * N + para.rLengthWithPadding];   // R
    }
}

__aicore__ inline void GetNormalizeOutputRstd(const LocalTensor<float>& dstRstd, const LocalTensor<float>& srcVar,
    const NormalizeTmpTensor<float>& tmpTensor, const UnaryRepeatParams& unaryParams,
    const BinaryRepeatParams& binaryParams, const NormalizePara& para, const float epsilon)
{
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, para.aLength);

    // 1. Variance + espilon ==> AddsX
    Adds<float, false>(dstRstd, srcVar, epsilon, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    // 2. Rsqrt(AddsX) = 1 / Sqrt(AddsX) ==> dstRstd
    Duplicate<float, false>(tmpTensor.tempTensorA, float(1), 1, 1, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
    Sqrt<float, false>(dstRstd, dstRstd, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Div<float, false>(dstRstd, tmpTensor.tempTensorA, dstRstd, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
}

template <typename U>
__aicore__ inline void CastTensor(const LocalTensor<U>& src, const LocalTensor<float>& castRes)
{
    // cast FP16 -> FP32
    Cast<float, U, false>(castRes, src, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, {1, 1, DEFAULT_REPEAT_STRIDE,
        HALF_DEFAULT_REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();
}

template <typename U, const NormalizeConfig& config>
__aicore__ inline void CastGammaBeta(const LocalTensor<U>& gamma, const LocalTensor<U>& beta,
    const NormalizeTmpTensor<float>& tmpTensor, const NormalizePara& para)
{
    SetVectorMask<float, MaskMode::COUNTER>(0, para.rLength);
    if constexpr(!config.isNoGamma) {
        CastTensor<U>(gamma, tmpTensor.gammaTmpTensor);
    }
    if constexpr(!config.isNoBeta) {
        CastTensor<U>(beta, tmpTensor.betaTmpTensor);
    }
}

template <typename T>
__aicore__ inline void CastSrc(const LocalTensor<T>& srcX, const NormalizeTmpTensor<float>& tmpTensor,
    const NormalizePara& para, const uint32_t N)
{
    SetVectorMask<float, MaskMode::COUNTER>(0, N * para.rLengthWithPadding);
    if constexpr(IsSameType<T, float>::value) {
        Adds<float, false>(tmpTensor.tempTensorA, srcX, (float)0, MASK_PLACEHOLDER, 1, {1, 1, DEFAULT_REPEAT_STRIDE,
            DEFAULT_REPEAT_STRIDE});
    } else {
        Cast<float, T, false>(tmpTensor.tempTensorA, srcX, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, {1, 1,
            DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE});
    }
    PipeBarrier<PIPE_V>();
}

// brcbTmp: Var[A], Rstd[A] => [N, R] to calculate with srcX [N, R]
__aicore__ inline void GetNormalizeOutputPre(const LocalTensor<float>& srcX, const LocalTensor<float>& srcMean,
    const LocalTensor<float>& srcRstd, const LocalTensor<float>& brcbTmp, const LocalTensor<float>& dstVmuls,
    const NormalizePara& para, const BinaryRepeatParams& binaryParams, const uint32_t N, const uint32_t NBase)
{
    // 1. SrcX [N, R] - (srcMean[A] after brcb [N, R]) ==> SubX [N, R]
    SetVectorMask<float, MaskMode::COUNTER>(0, para.rLength);
    auto eventId = GetTPipePtr()->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(eventId);
    WaitFlag<HardEvent::V_S>(eventId);
    for (uint32_t i = 0; i < N; i++) {
        float value = srcMean.GetValue(i + NBase);
        eventId = GetTPipePtr()->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(eventId);
        WaitFlag<HardEvent::S_V>(eventId);
        Duplicate<float, false>(brcbTmp[i * para.rLengthWithPadding], value, 1, 1, DEFAULT_BLK_STRIDE,
            DEFAULT_REPEAT_STRIDE);
    }
    PipeBarrier<PIPE_V>();

    SetVectorMask<float, MaskMode::COUNTER>(0, N * para.rLengthWithPadding);
    Sub<float, false>(srcX, srcX, brcbTmp, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    // 2. SubX [N, R] * (srcRstd[A] after brcb [N, R])
    SetVectorMask<float, MaskMode::COUNTER>(0, para.rLength);
    for (uint32_t i = 0; i < N; i++) {
        float value = srcRstd.GetValue(i + NBase);
        eventId = GetTPipePtr()->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(eventId);
        WaitFlag<HardEvent::S_V>(eventId);
        Duplicate<float, false>(brcbTmp[i * para.rLengthWithPadding], value, 1, 1, DEFAULT_BLK_STRIDE,
            DEFAULT_REPEAT_STRIDE);
    }
    PipeBarrier<PIPE_V>();

    SetVectorMask<float, MaskMode::COUNTER>(0, N * para.rLengthWithPadding);
    Mul<float, false>(dstVmuls, srcX, brcbTmp, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
}

template <typename T, const NormalizeConfig& config>
__aicore__ inline void GetNormalizeOutput(const LocalTensor<float>& srcxFP32, const LocalTensor<float>& gammaFP32,
    const LocalTensor<float>& betaFP32, const LocalTensor<T>& output, const NormalizePara& para,
    const UnaryRepeatParams& unaryParams, const BinaryRepeatParams& binaryParams, const uint32_t N)
{
    SetVectorMask<float, MaskMode::COUNTER>(0, para.rLength);
    for (uint32_t i = 0; i < N; i++) {
        if constexpr(!config.isNoGamma) {
            Mul<float, false>(srcxFP32[i * para.rLengthWithPadding], srcxFP32[i * para.rLengthWithPadding], gammaFP32,
                MASK_PLACEHOLDER, 1, binaryParams);
            PipeBarrier<PIPE_V>();
        }
        if constexpr(!config.isNoBeta) {
            Add<float, false>(srcxFP32[i * para.rLengthWithPadding], srcxFP32[i * para.rLengthWithPadding], betaFP32,
                MASK_PLACEHOLDER, 1, binaryParams);
            PipeBarrier<PIPE_V>();
        }
    }
    SetVectorMask<float, MaskMode::COUNTER>(0, N * para.rLengthWithPadding);
    if constexpr(IsSameType<T, float>::value) {
        Adds<float, false>(output, srcxFP32, (float)0, MASK_PLACEHOLDER, 1, unaryParams);
    } else {
        Cast<T, float, false>(output, srcxFP32, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, {1, 1,
            HALF_DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
    }
    PipeBarrier<PIPE_V>();
}

template <typename U, typename T, bool isReuseSource, const NormalizeConfig& config>
__aicore__ inline void NormalizeImpl(const LocalTensor<T>& output, const LocalTensor<float>& outputRstd,
    const LocalTensor<float>& inputMean, const LocalTensor<float>& inputVariance, const LocalTensor<T>& inputX,
    const LocalTensor<U>& gamma, const LocalTensor<U>& beta, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const float epsilon, const NormalizePara& para)
{
    static_assert(IsDtypeValid<U, T>(), "Failed to check dtype in Normalize, current api support dtype combination is "
        "T: float, U: float; T: half, U: half / float.");
    bool res = CheckParams<config>(para);

    const UnaryRepeatParams unaryParam;
    const BinaryRepeatParams binaryParam;
    NormalizeTmpTensor<float> tmpTensor;
    LocalTensor<float> stackBuffer = sharedTmpBuffer.ReinterpretCast<float>();
    uint32_t N = 0;
    GetNormalizeTensorInfo<U, T>(stackBuffer, para, tmpTensor, N);
    uint32_t mainRepeatTimes = para.aLength / N;
    uint32_t tailN = para.aLength % N;

    GetNormalizeOutputRstd(outputRstd, inputVariance, tmpTensor, unaryParam, binaryParam, para, epsilon);
    if constexpr(IsSameType<U, half>::value) {
        CastGammaBeta<U, config>(gamma, beta, tmpTensor, para);
    }

    for (uint32_t i = 0; i < mainRepeatTimes; i++) {
        uint32_t index = para.rLengthWithPadding * N * i;
        CastSrc<T>(inputX[index], tmpTensor, para, N);
        GetNormalizeOutputPre(tmpTensor.tempTensorA, inputMean, outputRstd, tmpTensor.tempTensorB,
            tmpTensor.tempTensorA, para, binaryParam, N, N * i);
        if constexpr(IsSameType<U, float>::value) {
            GetNormalizeOutput<T, config>(tmpTensor.tempTensorA, gamma, beta,
                output[index], para, unaryParam, binaryParam, N);
        } else {
            GetNormalizeOutput<T, config>(tmpTensor.tempTensorA, tmpTensor.gammaTmpTensor, tmpTensor.betaTmpTensor,
                output[index], para, unaryParam, binaryParam, N);
        }
    }

    if (tailN > 0) {
        uint32_t index = para.rLengthWithPadding * N * mainRepeatTimes;
        CastSrc<T>(inputX[index], tmpTensor, para, tailN);
        GetNormalizeOutputPre(tmpTensor.tempTensorA, inputMean, outputRstd, tmpTensor.tempTensorB,
            tmpTensor.tempTensorA, para, binaryParam, tailN, N * mainRepeatTimes);
        if constexpr(IsSameType<U, float>::value) {
            GetNormalizeOutput<T, config>(tmpTensor.tempTensorA, gamma, beta,
                output[index], para, unaryParam, binaryParam, tailN);
        } else {
            GetNormalizeOutput<T, config>(tmpTensor.tempTensorA, tmpTensor.gammaTmpTensor, tmpTensor.betaTmpTensor,
                output[index], para, unaryParam, binaryParam, tailN);
        }
    }

    SetMaskNorm();
    ResetMask();
}

template <typename U, typename T, bool isReuseSource, const NormalizeConfig& config>
__aicore__ inline void NormalizeImpl(const LocalTensor<T>& output, const LocalTensor<float>& outputRstd,
    const LocalTensor<float>& inputMean, const LocalTensor<float>& inputVariance, const LocalTensor<T>& inputX,
    const LocalTensor<U>& gamma, const LocalTensor<U>& beta, const float epsilon, const NormalizePara& para)
{
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    NormalizeImpl<U, T, isReuseSource, config>(output, outputRstd, inputMean, inputVariance, inputX, gamma, beta,
        sharedTmpBuffer, epsilon, para);
}

} // namespace AscendC
#endif // IMPL_NORMALIZATION_NORMALIZE_NORMALIZE_COMMON_IMPL_H
