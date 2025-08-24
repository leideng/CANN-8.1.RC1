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
 * \file layernormgradbeta_common_impl.h
 * \brief
 */
#ifndef IMPL_NORMALIZATION_LAYERNORMGRAD_LAYERNORMGRADBETA_COMMON_IMPL_H
#define IMPL_NORMALIZATION_LAYERNORMGRAD_LAYERNORMGRADBETA_COMMON_IMPL_H

#include "kernel_tensor.h"
#include "kernel_tiling/kernel_tiling.h"

namespace AscendC {
struct LayerNormGradBetaParams {
    __aicore__ LayerNormGradBetaParams(){};

    uint32_t bLength = 0;
    uint32_t sLength = 0;
    uint32_t hLength = 0;
    uint32_t originalHLength = 0;

    uint32_t bshCurLength = 0;
    uint32_t bsCurLength = 0;
    uint32_t hCurLength = 0;

    LocalTensor<float> gammaTempTensor;
    LocalTensor<float> betaTempTensor;
    LocalTensor<float> inputDyTmpTensor;
    LocalTensor<float> resForGammaTmpTensor;
};


template <bool isClearDst = false>
__aicore__ inline void ReduceSumFirstN(const LocalTensor<float> &dst, const LocalTensor<float> &src,
    const uint32_t bsLength, const uint32_t hLength)
{
    SetVectorMask<float, MaskMode::COUNTER>(0, hLength);
    uint32_t startIndex = 0;
    if constexpr (isClearDst) {
        const UnaryRepeatParams unaryRepeatParams;
        Adds<float, false>(dst, src, static_cast<float>(0), MASK_PLACEHOLDER, 1, unaryRepeatParams);
        startIndex = 1;
        PipeBarrier<PIPE_V>();
    }

    const BinaryRepeatParams binaryParams;
    for (; startIndex < bsLength; startIndex++) {
        Add<float, false>(dst, src[startIndex * hLength], dst, MASK_PLACEHOLDER, 1, binaryParams);
        PipeBarrier<PIPE_V>();
    }
}

template <bool isClearDst = false>
__aicore__ inline void ComputeProcess(const LocalTensor<float> &resForGamma, const LocalTensor<float> &inputDy,
    const LocalTensor<float> &outputPdGamma, const LocalTensor<float> &outputPdBeta,
    const LayerNormGradBetaParams &params)
{
    const LocalTensor<float> &resForGammaTmpTensor = params.resForGammaTmpTensor;

    SetVectorMask<float, MaskMode::COUNTER>(0, params.bshCurLength);

    const BinaryRepeatParams binaryParams;
    // res = inputDy * resForGamma
    Mul<float, false>(resForGammaTmpTensor, inputDy, resForGamma, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    // res = np.sum(res)
    ReduceSumFirstN<isClearDst>(outputPdGamma, resForGammaTmpTensor, params.bsCurLength, params.hCurLength);
    // res = np.sum(inputDy)
    ReduceSumFirstN<isClearDst>(outputPdBeta, inputDy, params.bsCurLength, params.hCurLength);
}

template <bool isClearDst = false>
__aicore__ inline void ComputeProcess(const LocalTensor<half> &resForGamma, const LocalTensor<half> &inputDy,
    const LocalTensor<half> &outputPdGamma, const LocalTensor<half> &outputPdBeta,
    const LayerNormGradBetaParams &params)
{
    const LocalTensor<float> &inputDyTmpTensor = params.inputDyTmpTensor;
    const LocalTensor<float> &resForGammaTmpTensor = params.resForGammaTmpTensor;

    const LocalTensor<float> &gammaTempTensor = params.gammaTempTensor;
    const LocalTensor<float> &betaTempTensor = params.betaTempTensor;

    SetVectorMask<half, MaskMode::COUNTER>(0, params.bshCurLength);

    UnaryRepeatParams unaryParams;
    unaryParams.srcRepStride = DEFAULT_REPEAT_STRIDE / sizeof(half);
    // In the half scenario, to ensure the precision, the value needs to be converted to a float value for calculation
    // and then converted back to the half value.
    Cast<float, half, false>(inputDyTmpTensor, inputDy, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Cast<float, half, false>(resForGammaTmpTensor, resForGamma, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    ComputeProcess<isClearDst>(resForGammaTmpTensor, inputDyTmpTensor, gammaTempTensor, betaTempTensor, params);

    SetVectorMask<float, MaskMode::COUNTER>(0, params.hCurLength);

    unaryParams.srcRepStride = DEFAULT_REPEAT_STRIDE;
    unaryParams.dstRepStride = DEFAULT_REPEAT_STRIDE / sizeof(half);

    Cast<half, float, false>(outputPdGamma, gammaTempTensor, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Cast<half, float, false>(outputPdBeta, betaTempTensor, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void LayerNormGradBetaComputeND(const LocalTensor<T> &resForGamma, const LocalTensor<T> &inputDy,
    const LocalTensor<T> &outputPdGamma, const LocalTensor<T> &outputPdBeta, const LayerNormGradBetaTiling &tiling,
    LayerNormGradBetaParams &params)
{
    ComputeProcess<true>(resForGamma, inputDy, outputPdGamma, outputPdBeta, params);

    uint32_t inputOffset = tiling.oneCalSize;

    for (uint32_t index = 1; index < tiling.loopRound; index++) {
        ComputeProcess(resForGamma[inputOffset], inputDy[inputOffset], outputPdGamma, outputPdBeta, params);
        inputOffset += tiling.oneCalSize;
    }

    if (tiling.inputTailSize > 0) {
        params.bshCurLength = tiling.inputTailSize;
        params.bsCurLength = tiling.bsTailSize;

        ComputeProcess(resForGamma[tiling.inputTailPos], inputDy[tiling.inputTailPos], outputPdGamma, outputPdBeta,
            params);
    }
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void GetLayerNormGradBetaTensorInfo(const LocalTensor<T> &resForGamma, const LocalTensor<T> &inputDy,
    const LocalTensor<float> &stackBuffer, const LayerNormGradBetaTiling &tiling, LayerNormGradBetaParams &params)
{
    params.bLength = tiling.bLength;
    params.sLength = tiling.sLength;
    params.hLength = tiling.hLength;
    params.originalHLength = tiling.originalHLength;

    params.bshCurLength = tiling.bshCurLength;
    params.bsCurLength = tiling.bsCurLength;
    params.hCurLength = tiling.originalHLength;

    if constexpr (sizeof(T) == sizeof(half)) {
        params.gammaTempTensor = stackBuffer[tiling.gammaTempTensorPos];
        params.betaTempTensor = stackBuffer[tiling.betaTempTensorPos];
        params.inputDyTmpTensor = stackBuffer[tiling.inputDyTmpTensorPos];
        params.resForGammaTmpTensor = stackBuffer[tiling.resForGammaTmpTensorPos];

        ASCENDC_ASSERT((tiling.resForGammaTmpTensorPos + tiling.oneCalSize <= tiling.stackBufferSize), {
            KERNEL_LOG(KERNEL_ERROR, "resForGammaTmpTensorPos + oneCalSize is (%d) should <= stackBufferSize is (%d)",
                tiling.resForGammaTmpTensorPos + tiling.oneCalSize, tiling.stackBufferSize);
        });
    }

    if constexpr (sizeof(T) == sizeof(float)) {
        if constexpr (isReuseSource) {
            params.resForGammaTmpTensor = resForGamma;
        } else {
            params.resForGammaTmpTensor = stackBuffer[tiling.resForGammaTmpTensorPos];

            ASCENDC_ASSERT((tiling.resForGammaTmpTensorPos + tiling.oneCalSize <= tiling.stackBufferSize), {
                KERNEL_LOG(KERNEL_ERROR,
                    "resForGammaTmpTensorPos + oneCalSize is (%d) should <= stackBufferSize is (%d)",
                    tiling.resForGammaTmpTensorPos + tiling.oneCalSize, tiling.stackBufferSize);
            });
        }
    }

    ASCENDC_ASSERT((stackBuffer.GetSize() >= tiling.stackBufferSize), {
        KERNEL_LOG(KERNEL_ERROR, "stackBuffer.GetSize is (%d) >= tiling.stackBufferSize is (%d)", stackBuffer.GetSize(),
            tiling.stackBufferSize);
    });
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void LayerNormGradBetaImpl(const LocalTensor<T> &outputPdGamma, const LocalTensor<T> &outputPdBeta,
    const LocalTensor<T> &resForGamma, const LocalTensor<T> &inputDy, const LocalTensor<uint8_t> &sharedTmpBuffer,
    const LayerNormGradBetaTiling &tiling)
{
    TRACE_START(TraceId::LayerNormGradBeta);
    ASCENDC_ASSERT((tiling.oneCalSize > 0), { KERNEL_LOG(KERNEL_ERROR, "tiling.oneCalSize must > 0!"); });

    if ASCEND_IS_AIC {
        TRACE_STOP(TraceId::LayerNormGradBeta);
        return;
    }

    LocalTensor<float> stackBuffer = sharedTmpBuffer.ReinterpretCast<float>();
    LayerNormGradBetaParams params;
    GetLayerNormGradBetaTensorInfo<T, isReuseSource>(resForGamma, inputDy, stackBuffer, tiling, params);

    SetMaskCount();
    LayerNormGradBetaComputeND(resForGamma, inputDy, outputPdGamma, outputPdBeta, tiling, params);

    SetMaskNorm();
    ResetMask();
    TRACE_STOP(TraceId::LayerNormGradBeta);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void LayerNormGradBetaImpl(const LocalTensor<T> &outputPdGamma, const LocalTensor<T> &outputPdBeta,
    const LocalTensor<T> &resForGamma, const LocalTensor<T> &inputDy, LayerNormGradBetaTiling &tiling)
{
    LocalTensor<uint8_t> sharedTmpBuffer; // partial derivation
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });

    LayerNormGradBetaImpl<T, isReuseSource>(outputPdGamma, outputPdBeta, resForGamma, inputDy, sharedTmpBuffer, tiling);
}
} // namespace AscendC
#endif // IMPL_NORMALIZATION_LAYERNORMGRAD_LAYERNORMGRADBETA_COMMON_IMPL_H