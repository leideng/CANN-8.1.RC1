/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file simple_softmax_common_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_SIMPLE_SOFTMAX_COMMON_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_SIMPLE_SOFTMAX_COMMON_IMPL_H

namespace AscendC {
__aicore__ inline void SimpleSoftMaxGenericNZImpl(const LocalTensor<float>& dst, const LocalTensor<float>& inSumTensor,
    const LocalTensor<float>& inMaxTensor, const LocalTensor<float>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling, const uint32_t& offset1, const uint32_t& offset2, const uint32_t& splitCount)
{
    const uint32_t splitOffset = tiling.splitM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    LocalTensor<float> tmpBuffer0 = workLocal;
    LocalTensor<float> tmpBuffer1 = workLocal[tiling.reduceSize + tiling.reduceSize];
    const uint16_t originalSrcM = splitCount / SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t splitNZBlockCount = tiling.srcK / SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t lastSplitNZBlockOffset = splitOffset * (splitNZBlockCount - 1);

    DataCopyParams copyParams = { originalSrcM, 1, 0, 1 };
    DataCopy(tmpBuffer0, inMaxTensor[offset2], copyParams);
    DataCopy(tmpBuffer0[FLOAT_NUM_PER_BLK], inMaxTensor[offset2], copyParams);
    DataCopy(tmpBuffer1, inSumTensor[offset2], copyParams);
    DataCopy(tmpBuffer1[FLOAT_NUM_PER_BLK], inSumTensor[offset2], copyParams);

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);
    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Sub<float, false>(dst[offset1 + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT],
            src[offset1 + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT], tmpBuffer0, MASK_PLACEHOLDER, 1,
            { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
    PipeBarrier<PIPE_V>();
    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Exp<float, false>(dst[offset1 + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT],
            dst[offset1 + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT], MASK_PLACEHOLDER, 1,
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
    PipeBarrier<PIPE_V>();
    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Div<float, false>(dst[offset1 + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT],
            dst[offset1 + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT], tmpBuffer1, MASK_PLACEHOLDER,
            1, { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
    SetMaskNorm();
    ResetMask();
}

__aicore__ inline void SimpleSoftMaxGenericNZImpl(const LocalTensor<half>& dst, const LocalTensor<half>& inSumTensor,
    const LocalTensor<half>& inMaxTensor, const LocalTensor<half>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling, const uint32_t& offset1, const uint32_t& offset2, const uint32_t& splitCount)
{
    LocalTensor<float> tmpBuffer0 = workLocal;
    LocalTensor<float> tmpBuffer1 = workLocal[tiling.splitSize];
    const uint32_t splitOffset = tiling.splitM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t splitNZBlockCount = tiling.srcK / SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t lastSplitNZBlockOffset = splitOffset * (splitNZBlockCount - 1);
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);

    Cast<float, half, false>(tmpBuffer1, inMaxTensor[offset2], RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE });

    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Cast<float, half, false>(tmpBuffer0[splitOffset * j],
            src[offset1 + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT], RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
            { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE });
    }
    PipeBarrier<PIPE_V>();

    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Sub<float, false>(tmpBuffer0[splitOffset * j], tmpBuffer0[splitOffset * j], tmpBuffer1, MASK_PLACEHOLDER, 1,
            { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
    PipeBarrier<PIPE_V>();
    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Exp<float, false>(tmpBuffer0[splitOffset * j], tmpBuffer0[splitOffset * j], MASK_PLACEHOLDER, 1,
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }

    Cast<float, half, false>(tmpBuffer1, inSumTensor[offset2], RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE });

    PipeBarrier<PIPE_V>();
    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Div<float, false>(tmpBuffer0[splitOffset * j], tmpBuffer0[splitOffset * j], tmpBuffer1, MASK_PLACEHOLDER, 1,
            { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
    PipeBarrier<PIPE_V>();
    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Cast<half, float, false>(dst[offset1 + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT],
            tmpBuffer0[splitOffset * j], FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER, 1,
            { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
    SetMaskNorm();
    ResetMask();
}

__aicore__ inline void SimpleSoftMaxGenericNZImpl(const LocalTensor<half>& dst, const LocalTensor<float>& inSumTensor,
    const LocalTensor<float>& inMaxTensor, const LocalTensor<half>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling, const uint32_t& offset1, const uint32_t& offset2, const uint32_t& splitCount)
{
    LocalTensor<float> tmpBuffer0 = workLocal;
    LocalTensor<float> tmpBuffer1 = workLocal[tiling.splitSize];
    const uint32_t splitOffset = tiling.splitM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint16_t originalSrcM = splitCount / SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t splitNZBlockCount = tiling.srcK / SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    const uint32_t lastSplitNZBlockOffset = splitOffset * (splitNZBlockCount - 1);

    DataCopyParams copyParams = { originalSrcM, 1, 0, 1 };
    DataCopy(tmpBuffer1, inMaxTensor[offset2], copyParams);
    DataCopy(tmpBuffer1[FLOAT_NUM_PER_BLK], inMaxTensor[offset2], copyParams);

    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);

    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Cast<float, half, false>(tmpBuffer0[splitOffset * j],
            src[offset1 + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT], RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
            { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE });
    }
    PipeBarrier<PIPE_V>();
    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Sub<float, false>(tmpBuffer0[splitOffset * j], tmpBuffer0[splitOffset * j], tmpBuffer1, MASK_PLACEHOLDER, 1,
            { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
    PipeBarrier<PIPE_V>();
    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Exp<float, false>(tmpBuffer0[splitOffset * j], tmpBuffer0[splitOffset * j], MASK_PLACEHOLDER, 1,
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
    SetMaskNorm();
    ResetMask();

    DataCopy(tmpBuffer1, inSumTensor[offset2], copyParams);
    DataCopy(tmpBuffer1[FLOAT_NUM_PER_BLK], inSumTensor[offset2], copyParams);

    PipeBarrier<PIPE_V>();
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, splitCount);
    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Div<float, false>(tmpBuffer0[splitOffset * j], tmpBuffer0[splitOffset * j], tmpBuffer1, MASK_PLACEHOLDER, 1,
            { 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
    PipeBarrier<PIPE_V>();
    for (uint32_t j = 0; j < splitNZBlockCount; j++) {
        Cast<half, float, false>(dst[offset1 + j * tiling.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT],
            tmpBuffer0[splitOffset * j], FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER, 1,
            { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
    SetMaskNorm();
    ResetMask();
}

template <typename T1, typename T2>
__aicore__ inline void SimpleSoftMaxNZImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& inSumTensor,
    const LocalTensor<T2>& inMaxTensor, const LocalTensor<T1>& src, const LocalTensor<float> workLocal,
    const SoftMaxTiling& tiling)
{
    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint32_t splitCount = tiling.splitM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;

    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        offset1 = i * splitCount;
        offset2 = i * tiling.reduceSize;
        SimpleSoftMaxGenericNZImpl(dst, inSumTensor, inMaxTensor, src, workLocal, tiling, offset1, offset2, splitCount);
    }
    PipeBarrier<PIPE_V>();
    if (tiling.tailM != 0) {
        offset1 = tiling.rangeM * splitCount;
        offset2 = tiling.rangeM * tiling.reduceSize;
        splitCount = tiling.tailM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
        SimpleSoftMaxGenericNZImpl(dst, inSumTensor, inMaxTensor, src, workLocal, tiling, offset1, offset2, splitCount);
    }
}

__aicore__ inline void SimpleSoftMaxBasicBlock(const LocalTensor<half>& dst, const LocalTensor<half>& expSumTensor,
    const LocalTensor<half>& maxTensor, const LocalTensor<half>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling)
{
    const LocalTensor<float>& tmpBuffer0 = workLocal;
    const LocalTensor<float>& tmpBuffer1 = workLocal[tiling.splitSize];

    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint8_t repeatTimes = (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE);
    uint8_t offset = (uint8_t)(FLOAT_NUM_PER_BLK * (tiling.splitK / FLOAT_REPEAT_SIZE));
    const uint32_t splitBlock = tiling.splitK / FLOAT_REPEAT_SIZE;
    const uint8_t reduceCeilValue = (uint8_t)(DivCeil(tiling.reduceSize, FLOAT_REPEAT_SIZE));
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        offset2 = i * tiling.reduceSize;
        offset1 = i * tiling.splitSize;

        Cast<float, half, false>(tmpBuffer1, maxTensor[offset2], RoundMode::CAST_NONE, MASK_PLACEHOLDER,
            reduceCeilValue, { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE });
        Cast<float, half, false>(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, MASK_PLACEHOLDER, repeatTimes,
            { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE });

        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < splitBlock; ++j) {
            Sub<float, false>(tmpBuffer0[FLOAT_REPEAT_SIZE * j], tmpBuffer0[FLOAT_REPEAT_SIZE * j], tmpBuffer1,
                MASK_PLACEHOLDER, (uint8_t)(tiling.splitM), { 1, 1, 0, offset, offset, HALF_FACTOR });
        }
        PipeBarrier<PIPE_V>();
        Exp<float, false>(tmpBuffer0, tmpBuffer0, MASK_PLACEHOLDER, (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE),
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });

        Cast<float, half, false>(tmpBuffer1, expSumTensor[offset2], RoundMode::CAST_NONE, MASK_PLACEHOLDER,
            reduceCeilValue, { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < splitBlock; ++j) {
            Div<float, false>(tmpBuffer0[FLOAT_REPEAT_SIZE * j], tmpBuffer0[FLOAT_REPEAT_SIZE * j], tmpBuffer1,
                MASK_PLACEHOLDER, (uint8_t)(tiling.splitM), { 1, 1, 0, offset, offset, HALF_FACTOR });
        }
        PipeBarrier<PIPE_V>();
        Cast<half, float, false>(dst[offset1], tmpBuffer0, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER, repeatTimes,
            { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
}

__aicore__ inline void SimpleSoftMaxBasicBlock(const LocalTensor<float>& dst, const LocalTensor<float>& expSumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<float>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling)
{
#if ASCENDC_CPU_DEBUG == 1
    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint8_t offset = (uint8_t)(FLOAT_NUM_PER_BLK * (tiling.splitK / FLOAT_REPEAT_SIZE));
    const uint32_t splitBlock = tiling.splitK / FLOAT_REPEAT_SIZE;
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        offset2 = i * tiling.reduceSize;
        offset1 = i * tiling.splitSize;

        for (uint32_t j = 0; j < splitBlock; ++j) {
            Sub<float, false>(dst[offset1 + FLOAT_REPEAT_SIZE * j], src[offset1 + FLOAT_REPEAT_SIZE * j],
                maxTensor[offset2], MASK_PLACEHOLDER, (uint8_t)(tiling.splitM), { 1, 1, 0, offset, offset, 1 });
        }
        PipeBarrier<PIPE_V>();
        Exp<float, false>(dst[offset1], dst[offset1], MASK_PLACEHOLDER, (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE),
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();

        for (uint32_t j = 0; j < splitBlock; ++j) {
            Div<float, false>(dst[offset1 + FLOAT_REPEAT_SIZE * j], dst[offset1 + FLOAT_REPEAT_SIZE * j],
                expSumTensor[offset2], MASK_PLACEHOLDER, (uint8_t)(tiling.splitM), { 1, 1, 0, offset, offset, 1 });
        }
    }
#else
    const uint32_t splitBlock = tiling.srcK / FLOAT_REPEAT_SIZE;
    const uint8_t repstride = tiling.srcK / FLOAT_NUM_PER_BLK;
    PipeBarrier<PIPE_V>();
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, tiling.srcM * FLOAT_REPEAT_SIZE);
    for (uint32_t j = 0; j < splitBlock; ++j) {
        Sub<float, false>(dst[FLOAT_REPEAT_SIZE * j], src[FLOAT_REPEAT_SIZE * j], maxTensor, MASK_PLACEHOLDER, 1,
            { 1, 1, 0, repstride, repstride, 1 });
    }
    SetVectorMask<float, MaskMode::COUNTER>(0, tiling.srcSize);
    PipeBarrier<PIPE_V>();
    Exp<float, false>(dst, dst, MASK_PLACEHOLDER, 1, { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();

    SetVectorMask<float, MaskMode::COUNTER>(0, tiling.srcM * FLOAT_REPEAT_SIZE);
    for (uint32_t j = 0; j < splitBlock; ++j) {
        Div<float, false>(dst[FLOAT_REPEAT_SIZE * j], dst[FLOAT_REPEAT_SIZE * j], expSumTensor, MASK_PLACEHOLDER, 1,
            { 1, 1, 0, repstride, repstride, 1 });
    }
    SetMaskNorm();
    ResetMask();
#endif
}

template <const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SimpleSoftMaxGenericNDImpl(const LocalTensor<half>& dst, const LocalTensor<half>& inSumTensor,
    const LocalTensor<half>& inMaxTensor, const LocalTensor<half>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling, const uint32_t offset1, const uint32_t offset2, const uint32_t curSplitM)
{
    const LocalTensor<float>& tmpBuffer0 = workLocal;
    const LocalTensor<float>& tmpBuffer2 = workLocal[tiling.splitSize];
    const uint32_t splitSize = curSplitM * tiling.splitK;
    const uint32_t reduceSize = curSplitM * tiling.reduceK;
    if constexpr (config.oriSrcM == 0 || config.oriSrcK == 0) {
        Cast(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, splitSize);
        Cast(tmpBuffer2, inMaxTensor[offset2], RoundMode::CAST_NONE, reduceSize);
        PipeBarrier<PIPE_V>();

        GenericSubNDImpl(tmpBuffer0, tmpBuffer0, tmpBuffer2, curSplitM, tiling.srcK, tiling.reduceK);

        PipeBarrier<PIPE_V>();
        Exp(tmpBuffer0, tmpBuffer0, splitSize);

        Cast(tmpBuffer2, inSumTensor[offset2], RoundMode::CAST_NONE, reduceSize);
        PipeBarrier<PIPE_V>();
        GenericDivNDImpl(tmpBuffer0, tmpBuffer0, tmpBuffer2, curSplitM, tiling.srcK, tiling.reduceK);

        PipeBarrier<PIPE_V>();
        Cast(dst[offset1], tmpBuffer0, FLOAT2HALF_ROUND_MODE, splitSize);
    } else {
        uint32_t splitK = 0;
        if constexpr (config.oriSrcK % HALF_NUM_PER_BLK == 0) {
            splitK = config.oriSrcK;
        } else {
            splitK = AlignUp(config.oriSrcK, HALF_NUM_PER_BLK);
        }
        Cast(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, splitSize);
        Cast(tmpBuffer2, inMaxTensor[offset2], RoundMode::CAST_NONE, reduceSize);
        PipeBarrier<PIPE_V>();

        GenericSubNDImpl(tmpBuffer0, tmpBuffer0, tmpBuffer2, curSplitM, splitK,
            DEFAULT_REPEAT_STRIDE * HALF_FACTOR);

        PipeBarrier<PIPE_V>();
        Exp(tmpBuffer0, tmpBuffer0, splitSize);

        Cast(tmpBuffer2, inSumTensor[offset2], RoundMode::CAST_NONE, reduceSize);
        PipeBarrier<PIPE_V>();
        GenericDivNDImpl(tmpBuffer0, tmpBuffer0, tmpBuffer2, curSplitM, splitK,
            DEFAULT_REPEAT_STRIDE * HALF_FACTOR);

        PipeBarrier<PIPE_V>();
        Cast(dst[offset1], tmpBuffer0, FLOAT2HALF_ROUND_MODE, splitSize);
    }
}

template <const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SimpleSoftMaxGenericNDImpl(const LocalTensor<float>& dst, const LocalTensor<float>& inSumTensor,
    const LocalTensor<float>& inMaxTensor, const LocalTensor<float>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling, const uint32_t offset1, const uint32_t offset2, const uint32_t curSplitM)
{
    const uint32_t splitSize = curSplitM * tiling.splitK;

    if constexpr (config.oriSrcM == 0 || config.oriSrcK == 0) {
        GenericSubNDImpl(dst[offset1], src[offset1], inMaxTensor[offset2], curSplitM, tiling.srcK, tiling.reduceK);
        PipeBarrier<PIPE_V>();
        Exp(dst[offset1], dst[offset1], splitSize);
        PipeBarrier<PIPE_V>();
        GenericDivNDImpl(dst[offset1], dst[offset1], inSumTensor[offset2], curSplitM, tiling.srcK, tiling.reduceK);
    } else {
        uint32_t splitK = 0;
        if constexpr (config.oriSrcK % FLOAT_NUM_PER_BLK == 0) {
            splitK = config.oriSrcK;
        } else {
            splitK = AlignUp(config.oriSrcK, FLOAT_NUM_PER_BLK);
        }
        GenericSubNDImpl(dst[offset1], src[offset1], inMaxTensor[offset2], curSplitM, splitK,
            DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
        Exp(dst[offset1], dst[offset1], splitSize);
        PipeBarrier<PIPE_V>();
        GenericDivNDImpl(dst[offset1], dst[offset1], inSumTensor[offset2], curSplitM, splitK,
            DEFAULT_REPEAT_STRIDE);
    }
}

template <typename T, bool isBasicBlock = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SimpleSoftMaxNDImpl(const LocalTensor<T>& dst, const LocalTensor<T>& inSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<T>& src, const LocalTensor<float> workLocal,
    const SoftMaxTiling& tiling)
{
    if constexpr (isBasicBlock) {
        SimpleSoftMaxBasicBlock(dst, inSumTensor, inMaxTensor, src, workLocal, tiling);
    } else {
        if constexpr (sizeof(T) == sizeof(float)) {
            SimpleSoftMaxGenericNDImpl<config>(dst, inSumTensor, inMaxTensor, src, workLocal, tiling, 0, 0,
                tiling.srcM);
        } else {
            uint32_t offset1 = 0;
            uint32_t offset2 = 0;
            PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < tiling.rangeM; i++) {
                offset1 = i * tiling.splitSize;
                offset2 = i * tiling.reduceSize;
                SimpleSoftMaxGenericNDImpl<config>(dst, inSumTensor, inMaxTensor, src, workLocal, tiling, offset1,
                    offset2, tiling.splitM);
            }
            PipeBarrier<PIPE_V>();
            if (tiling.tailM != 0) {
                offset1 = tiling.rangeM * tiling.splitSize;
                offset2 = tiling.rangeM * tiling.reduceSize;
                SimpleSoftMaxGenericNDImpl<config>(dst, inSumTensor, inMaxTensor, src, workLocal, tiling, offset1,
                    offset2, tiling.tailM);
            }
        }
    }
}

__aicore__ inline void SimpleSoftMaxBasicBlock(const LocalTensor<half>& dst, const LocalTensor<float>& inSumTensor,
    const LocalTensor<float>& inMaxTensor, const LocalTensor<half>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling)
{
    const LocalTensor<float>& tmpBuffer0 = workLocal;

    uint32_t offset1 = 0;
    uint32_t offset2 = 0;
    uint8_t repeatTimes = (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE);
    uint8_t offset = (uint8_t)(FLOAT_NUM_PER_BLK * (tiling.splitK / FLOAT_REPEAT_SIZE));
    const uint32_t splitBlock = tiling.splitK / FLOAT_REPEAT_SIZE;
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < tiling.rangeM; i++) {
        offset2 = i * tiling.reduceSize;
        offset1 = i * tiling.splitSize;

        Cast<float, half, false>(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, MASK_PLACEHOLDER, repeatTimes,
            { 1, 1, DEFAULT_REPEAT_STRIDE, HALF_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < splitBlock; ++j) {
            Sub<float, false>(tmpBuffer0[FLOAT_REPEAT_SIZE * j], tmpBuffer0[FLOAT_REPEAT_SIZE * j],
                inMaxTensor[offset2], MASK_PLACEHOLDER, (uint8_t)(tiling.splitM), { 1, 1, 0, offset, offset, 1 });
        }
        PipeBarrier<PIPE_V>();
        Exp<float, false>(tmpBuffer0, tmpBuffer0, MASK_PLACEHOLDER, (uint8_t)(tiling.splitSize / FLOAT_REPEAT_SIZE),
            { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < splitBlock; ++j) {
            Div<float, false>(tmpBuffer0[FLOAT_REPEAT_SIZE * j], tmpBuffer0[FLOAT_REPEAT_SIZE * j],
                inSumTensor[offset2], MASK_PLACEHOLDER, (uint8_t)(tiling.splitM), { 1, 1, 0, offset, offset, 1 });
        }
        PipeBarrier<PIPE_V>();
        Cast<half, float, false>(dst[offset1], tmpBuffer0, FLOAT2HALF_ROUND_MODE, MASK_PLACEHOLDER, repeatTimes,
            { 1, 1, HALF_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    }
}

template <const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SimpleSoftMaxGenericNDImpl(const LocalTensor<half>& dst, const LocalTensor<float>& inSumTensor,
    const LocalTensor<float>& inMaxTensor, const LocalTensor<half>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling, const uint32_t offset1, const uint32_t offset2, const uint32_t curSplitM)
{
    const LocalTensor<float>& tmpBuffer0 = workLocal;
    const uint32_t splitSize = curSplitM * tiling.splitK;
    if constexpr (config.oriSrcM == 0 || config.oriSrcK == 0) {
        Cast(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, splitSize);
        PipeBarrier<PIPE_V>();
#if __CCE_AICORE__ == 200
        GenericSubNDImpl(tmpBuffer0, tmpBuffer0, inMaxTensor[offset2], curSplitM, tiling.srcK,
            DEFAULT_REPEAT_STRIDE);
#elif __CCE_AICORE__ == 220
        GenericSubNDImpl(tmpBuffer0, tmpBuffer0, inMaxTensor[offset2], curSplitM, tiling.srcK, tiling.reduceK);
#endif
        PipeBarrier<PIPE_V>();
        Exp(tmpBuffer0, tmpBuffer0, tiling.splitSize);
        PipeBarrier<PIPE_V>();
#if __CCE_AICORE__ == 200
        GenericDivNDImpl(tmpBuffer0, tmpBuffer0, inSumTensor[offset2], curSplitM, tiling.srcK,
            DEFAULT_REPEAT_STRIDE);
#elif __CCE_AICORE__ == 220
        GenericDivNDImpl(tmpBuffer0, tmpBuffer0, inSumTensor[offset2], curSplitM, tiling.srcK, tiling.reduceK);
#endif
        PipeBarrier<PIPE_V>();
        Cast(dst[offset1], tmpBuffer0, FLOAT2HALF_ROUND_MODE, splitSize);
    } else {
        uint32_t splitK = 0;
        if constexpr (config.oriSrcK % FLOAT_NUM_PER_BLK == 0) {
            splitK = config.oriSrcK;
        } else {
            splitK = AlignUp(config.oriSrcK, FLOAT_NUM_PER_BLK);
        }
        Cast(tmpBuffer0, src[offset1], RoundMode::CAST_NONE, splitSize);
        PipeBarrier<PIPE_V>();
        GenericSubNDImpl(tmpBuffer0, tmpBuffer0, inMaxTensor[offset2], curSplitM, splitK,
            DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
        Exp(tmpBuffer0, tmpBuffer0, tiling.splitSize);
        PipeBarrier<PIPE_V>();
        GenericDivNDImpl(tmpBuffer0, tmpBuffer0, inSumTensor[offset2], curSplitM, splitK,
            DEFAULT_REPEAT_STRIDE);
        PipeBarrier<PIPE_V>();
        Cast(dst[offset1], tmpBuffer0, FLOAT2HALF_ROUND_MODE, splitSize);
    }
}

template <typename T, bool isBasicBlock = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SimpleSoftMaxNDImpl(const LocalTensor<half>& dst, const LocalTensor<float>& inSumTensor,
    const LocalTensor<float>& inMaxTensor, const LocalTensor<half>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling)
{
    if constexpr (isBasicBlock) {
        SimpleSoftMaxBasicBlock(dst, inSumTensor, inMaxTensor, src, workLocal, tiling);
    } else {
        uint32_t offset1 = 0;
        uint32_t offset2 = 0;
        PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < tiling.rangeM; i++) {
            offset1 = i * tiling.splitSize;
            offset2 = i * tiling.reduceSize;
            SimpleSoftMaxGenericNDImpl<config>(dst, inSumTensor, inMaxTensor, src, workLocal, tiling, offset1,
                offset2, tiling.splitM);
        }
        PipeBarrier<PIPE_V>();
        if (tiling.tailM != 0) {
            offset1 = tiling.rangeM * tiling.splitSize;
            offset2 = tiling.rangeM * tiling.reduceSize;
            SimpleSoftMaxGenericNDImpl<config>(dst, inSumTensor, inMaxTensor, src, workLocal, tiling, offset1,
                offset2, tiling.tailM);
        }
    }
}
}
#endif // IMPL_ACTIVATION_SOFTMAX_SIMPLE_SOFTMAX_COMMON_IMPL_H