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
 * \file topk_v200_impl.h
 * \brief
 */
#ifndef IMPL_SORT_TOPK_TOPK_V200_IMPL_H
#define IMPL_SORT_TOPK_TOPK_V200_IMPL_H
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
#include <iostream>
#include <type_traits>
#include "kernel_log.h"
#endif

#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tiling/kernel_tiling.h"
#include "topk_common_utils.h"

namespace AscendC {
#if ASCENDC_CPU_DEBUG
template <typename T, bool isInitIndex = false, enum TopKMode topkMode = TopKMode::TOPK_NORMAL>
void TopkInputCheck(const int32_t k, const TopKInfo &topKInfo)
{
    bool ans = (1 <= k) && (k <= topKInfo.n);
    ASCENDC_ASSERT(ans, {
        KERNEL_LOG(KERNEL_ERROR, "The value of k must be greater than or equal to 1 and less than or equal to inner.");
    });
    ans = (1 <= topKInfo.n) && (topKInfo.n <= topKInfo.inner);
    ASCENDC_ASSERT(ans, {
        KERNEL_LOG(KERNEL_ERROR, "The value of n must be greater than or equal to 1 and less than or equal to inner.");
    });
    ans = (std::is_same<T, half>::value) || (std::is_same<T, float>::value);
    ASCENDC_ASSERT(ans, {KERNEL_LOG(KERNEL_ERROR, "type must be half or float");});
    ans = (topKInfo.inner % TOPK_INNER_ALIGN_LEN == 0);
    ASCENDC_ASSERT(ans, {KERNEL_LOG(KERNEL_ERROR, "The value of inner must be an integer multiple of 32.");});

    if ((sizeof(T) == sizeof(half)) && (!isInitIndex)) {
        ans = (topKInfo.inner <= TOPK_NORMAL_INNER_MAX_HALF_LEN);
        ASCENDC_ASSERT(ans, {KERNEL_LOG(KERNEL_ERROR, "The maximum inner value is 2048.");});
    }

    if constexpr (topkMode == TopKMode::TOPK_NORMAL) {
        ans = (topKInfo.inner <= TOPK_NORMAL_INNER_MAX_LEN);
        ASCENDC_ASSERT(ans, {KERNEL_LOG(KERNEL_ERROR, "The maximum value supported by inner is 4096.");});
    }
    if constexpr (topkMode == TopKMode::TOPK_NSMALL) {
        ans = (topKInfo.inner == TOPK_NSMALL_INNER_LEN);
        ASCENDC_ASSERT(ans, {KERNEL_LOG(KERNEL_ERROR, "The value of inner must be an equal 32.");});
    }
}
#endif

template <typename T, bool isInitIndex>
__aicore__ inline void ProposalConcatValIdx(const LocalTensor<T> &srcLocal, const LocalTensor<T> &tmpLocal,
    const TopkTiling &tilling, const TopKInfo &topKInfo, const bool isLargest, const int outterIdx,
    const UnaryRepeatParams unaryParams)
{
    int offset = outterIdx * topKInfo.inner;
    // srcLocal -> Region Proposal
    if (!isLargest) {
        SetVectorMask<T, MaskMode::COUNTER>(0, topKInfo.inner);
        Muls<T, false>(tmpLocal[tilling.innerDataSize], srcLocal[offset], T(-1), MASK_PLACEHOLDER, 1, unaryParams);
        PipeBarrier<PIPE_V>();
        ProposalConcat<T>(tmpLocal, tmpLocal[tilling.innerDataSize], tilling.sortRepeat, 4);
    } else {
        ProposalConcat<T>(tmpLocal, srcLocal[offset], tilling.sortRepeat, 4);
    }
    ProposalConcat<T>(tmpLocal, tmpLocal[tilling.srcIndexOffset], tilling.sortRepeat, 0);
    PipeBarrier<PIPE_V>();
}

template <typename T, bool isInitIndex>
__aicore__ inline void TmpLocalSort16(const LocalTensor<T> &srcLocal, const LocalTensor<T> &tmpLocal,
    const TopkTiling &tilling, const TopKInfo &topKInfo, const bool isLargest, const int outterIdx,
    const UnaryRepeatParams unaryParams)
{
    // The number of inners is divided into inner/16 groups for sorting. Each iteration completes one group of 16
    // numbers of sorting.
    ProposalConcatValIdx<T, isInitIndex>(srcLocal, tmpLocal, tilling, topKInfo, isLargest, outterIdx, unaryParams);
    RpSort16<T>(tmpLocal[tilling.innerDataSize], tmpLocal, tilling.sortRepeat);
    PipeBarrier<PIPE_V>();
    const DataCopyParams intriParams = {static_cast<uint16_t>(tilling.copyUbToUbBlockCount), 1, 0, 0};
    DataCopy(tmpLocal, tmpLocal[tilling.innerDataSize], intriParams);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void MrgSort4FourQueueSort(
    const LocalTensor<T> &tmpLocal, const TopkTiling &tilling, const TopKInfo &topKInfo, uint16_t &z, int32_t &dstIdx)
{
    int32_t mrgFourQueueCount = 0;
    uint16_t innerU16Type = static_cast<uint16_t>(topKInfo.inner);
    for (; z * MRGSORT_VALID_QUEUE <= innerU16Type; z *= MRGSORT_VALID_QUEUE) {
        auto src = (mrgFourQueueCount % TWO == 1) ? tmpLocal[tilling.innerDataSize] : tmpLocal[0];
        dstIdx = (mrgFourQueueCount + 1) % TWO;
        auto dst = (dstIdx == 1) ? tmpLocal[tilling.innerDataSize] : tmpLocal[0];
        mrgFourQueueCount += 1;
        uint16_t elementLengths[MRG_SORT_ELEMENT_LEN] = {z, z, z, z};
        struct MrgSort4Info srcInfo(elementLengths, false, 0b1111, tilling.mrgSortRepeat / z);
        struct MrgSortSrcList<T> srcList(src, src[z * tilling.mrgSortSrc1offset], src[z * tilling.mrgSortSrc2offset],
            src[z * tilling.mrgSortSrc3offset]);
        MrgSort4<T>(dst, srcList, srcInfo);
        PipeBarrier<PIPE_V>();
        const DataCopyParams intriParams = {static_cast<uint16_t>(tilling.copyUbToUbBlockCount), 1, 0, 0};
        DataCopy(src, dst, intriParams);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void MrgSort4TwoQueueSort(const LocalTensor<T> &tmpLocal, const TopkTiling &tilling,
    const TopKInfo &topKInfo, const uint16_t z, int32_t &dstIdx)
{
    int32_t arrayCount = 0;
    if (z < topKInfo.inner) {
        // MRG_MAX_ARRAY_SIZE is the number of four-way merging times is multiplied by three;
        int32_t mrgArray[MRG_MAX_ARRAY_SIZE] = {0};
        int32_t tmpInner = topKInfo.inner;
        for (int32_t i = z; i >= MIN_RPSORT16_SIZE; i /= MRGSORT_VALID_QUEUE) {
            int32_t count;
            for (count = 0; count < tmpInner / i; ++count) {
                mrgArray[arrayCount++] = i;
            }
            tmpInner -= count * i;
        }

        uint16_t mrgSortedLen = 0;
        int32_t tmpDstIdx = dstIdx;
        for (int32_t i = 0; i < arrayCount - 1; ++i) {
            auto src = ((tmpDstIdx + i) % TWO == 1) ? tmpLocal[tilling.innerDataSize] : tmpLocal[0];
            dstIdx = (tmpDstIdx + 1 + i) % TWO;
            auto dst = (dstIdx == 1) ? tmpLocal[tilling.innerDataSize] : tmpLocal[0];
            mrgSortedLen += static_cast<uint16_t>(mrgArray[i]);
            uint16_t elementLengths[MRG_SORT_ELEMENT_LEN] = {
                static_cast<uint16_t>(mrgSortedLen), static_cast<uint16_t>(mrgArray[i + 1]), 0, 0};
            struct MrgSort4Info srcInfo(elementLengths, false, 0b0011, 1);
            struct MrgSortSrcList<T> srcList(src, src[mrgSortedLen * 8], src, src);
            MrgSort4<T>(dst, srcList, srcInfo);
            PipeBarrier<PIPE_V>();
        }
    }
}

template <typename T>
__aicore__ inline void ProposalExtractValIdx(const LocalTensor<T> &dstValueLocal,
    const LocalTensor<int32_t> &dstIndexLocal, const LocalTensor<T> &tmpLocal, const TopkTiling &tilling,
    const int32_t dstIdx, const int32_t dstOffsetFourBytes, const int outterIdx, const TopKInfo &topKInfo,
    const int32_t k, const UnaryRepeatParams unaryParams)
{
    int32_t tmpLocalDstOffset = tilling.innerDataSize * dstIdx;
    int32_t tmpLocalSrcOffset = tilling.innerDataSize * (1 - dstIdx);
    int32_t dstValOffset = dstOffsetFourBytes;
    ProposalExtract<T>(tmpLocal[tmpLocalSrcOffset], tmpLocal[tmpLocalDstOffset], tilling.sortRepeat, 4);
    PipeBarrier<PIPE_V>();
    if constexpr (sizeof(T) == sizeof(half)) {
        dstValOffset = outterIdx * tilling.kAlignTwoBytes;
    }
    SetVectorMask<T, MaskMode::COUNTER>(0, k);
    Adds<T, false>(dstValueLocal[dstValOffset], tmpLocal[tmpLocalSrcOffset], 0, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    ProposalExtract<T>(tmpLocal[tmpLocalSrcOffset], tmpLocal[tmpLocalDstOffset], tilling.sortRepeat, 0);
    PipeBarrier<PIPE_V>();
    SetVectorMask<T, MaskMode::COUNTER>(0, topKInfo.inner);
    int32_t castOffset = 0;
    if constexpr (sizeof(T) == sizeof(half)) {
        castOffset = topKInfo.inner;
        // half->float
        LocalTensor<float> tempBufferFloat = tmpLocal[tmpLocalSrcOffset + castOffset].template
                                             ReinterpretCast<float>();
        Cast<float, half, false>(tempBufferFloat,
            tmpLocal[tmpLocalSrcOffset],
            RoundMode::CAST_NONE,
            MASK_PLACEHOLDER,
            1,
            {1, 1, DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE});
        PipeBarrier<PIPE_V>();
    }
    int32_t tmpOffset = tmpLocalSrcOffset + castOffset;
    // float -> int32_t
    LocalTensor<int32_t> tempBufferInt32 = tmpLocal[tmpOffset].template ReinterpretCast<int32_t>();
    LocalTensor<float> tempBufferFloat = tmpLocal[tmpOffset].template ReinterpretCast<float>();
    Cast<int32_t, float, false>(tempBufferInt32,
        tempBufferFloat,
        RoundMode::CAST_ROUND,
        MASK_PLACEHOLDER,
        1,
        {1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();
    SetVectorMask<T, MaskMode::COUNTER>(0, k);
    Adds<int32_t, false>(dstIndexLocal[dstOffsetFourBytes], tempBufferInt32, 0, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
}

template <typename T, bool isInitIndex>
__aicore__ inline void CastIdxTypeToValType(const LocalTensor<int32_t> &srcIndexLocal, const LocalTensor<T> &tmpLocal,
    const TopkTiling &tilling, const TopKInfo &topKInfo, const int32_t mask, const int32_t addrOffset)
{
    SetVectorMask<T, MaskMode::COUNTER>(0, mask);
    LocalTensor<float> tempBufferFloat = tmpLocal[addrOffset].template ReinterpretCast<float>();
    if constexpr (!isInitIndex) {
        // int32_t -> float
        LocalTensor<int32_t> tempBufferInt32 = tmpLocal[addrOffset].template ReinterpretCast<int32_t>();
        Cast<float, int32_t, false>(tempBufferFloat,
            tempBufferInt32,
            RoundMode::CAST_NONE,
            MASK_PLACEHOLDER,
            1,
            {1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
        PipeBarrier<PIPE_V>();
    } else {
        // int32_t -> float
        Cast<float, int32_t, false>(tempBufferFloat,
            srcIndexLocal,
            RoundMode::CAST_NONE,
            MASK_PLACEHOLDER,
            1,
            {1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
        PipeBarrier<PIPE_V>();
    }
    if constexpr (sizeof(T) == sizeof(half)) {
        // float->half
        LocalTensor<half> tempBufferHalf = tmpLocal[addrOffset].template ReinterpretCast<half>();
        Cast<half, float, false>(tempBufferHalf,
            tempBufferFloat,
            RoundMode::CAST_NONE,
            MASK_PLACEHOLDER,
            1,
            {1, 1, HALF_DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
        PipeBarrier<PIPE_V>();
    }
}

template <typename T, bool isInitIndex, bool isHasfinish>
__aicore__ inline void TopKCompute(const LocalTensor<T> &dstValueLocal, const LocalTensor<int32_t> &dstIndexLocal,
    const LocalTensor<T> &srcLocal, const LocalTensor<int32_t> &srcIndexLocal, const LocalTensor<bool> &finishLocal,
    const LocalTensor<T> &tmpLocal, const int32_t k, const TopkTiling &tilling, const TopKInfo &topKInfo,
    const bool isLargest)
{
    CastIdxTypeToValType<T, isInitIndex>(
        srcIndexLocal, tmpLocal, tilling, topKInfo, topKInfo.inner, tilling.srcIndexOffset);
    const UnaryRepeatParams unaryParams;
    for (int j = 0; j < topKInfo.outter; ++j) {
        int32_t dstOffsetFourBytes = j * tilling.kAlignFourBytes;
        int32_t dstIdx = 0;
        TmpLocalSort16<T, isInitIndex>(srcLocal, tmpLocal, tilling, topKInfo, isLargest, j, unaryParams);
        uint16_t z = MIN_RPSORT16_SIZE;
        MrgSort4FourQueueSort<T>(tmpLocal, tilling, topKInfo, z, dstIdx);
        MrgSort4TwoQueueSort<T>(tmpLocal, tilling, topKInfo, z, dstIdx);
        ProposalExtractValIdx<T>(
            dstValueLocal, dstIndexLocal, tmpLocal, tilling, dstIdx, dstOffsetFourBytes, j, topKInfo, k, unaryParams);
        if constexpr (isHasfinish) {
            auto eventID = GetTPipePtr()->FetchEventID(HardEvent::V_S);
            SetFlag<HardEvent::V_S>(eventID);
            WaitFlag<HardEvent::V_S>(eventID);
            if (finishLocal.GetValue(j)) {
                SetVectorMask<T, MaskMode::COUNTER>(0, k);
                Duplicate<int32_t, false>(dstIndexLocal[dstOffsetFourBytes],
                    static_cast<int32_t>(topKInfo.n),
                    MASK_PLACEHOLDER,
                    1,
                    1,
                    DEFAULT_REPEAT_STRIDE);
                PipeBarrier<PIPE_V>();
            }
        }
    }
}

template <typename T, bool isInitIndex>
__aicore__ inline void ProposalConcatValIdxSmall(
    const LocalTensor<T> &srcLocal, const LocalTensor<T> &tmpLocal, const TopkTiling &tilling, const bool isLargest)
{
    if (!isLargest) {
        ProposalConcat<T>(tmpLocal, tmpLocal[tilling.innerDataSize], tilling.sortRepeat, 4);
    } else {
        ProposalConcat<T>(tmpLocal, srcLocal, tilling.sortRepeat, 4);
    }
    ProposalConcat<T>(tmpLocal, tmpLocal[tilling.topkNSmallSrcIndexOffset], tilling.sortRepeat, 0);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void CastIdxToInt(const LocalTensor<T> &tmpLocal, const TopkTiling &tilling)
{
    SetVectorMask<T, MaskMode::COUNTER>(0, tilling.allDataSize);
    if constexpr (sizeof(T) == sizeof(half)) {
        // half->float
        LocalTensor<float> tempBufferFloat = tmpLocal[tilling.srcIndexOffset].template ReinterpretCast<float>();
        Cast<float, half, false>(tempBufferFloat,
            tmpLocal[tilling.topkNSmallSrcIndexOffset],
            RoundMode::CAST_NONE,
            MASK_PLACEHOLDER,
            1,
            {1, 1, DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE});
        PipeBarrier<PIPE_V>();
    }
    // float -> int32_t
    LocalTensor<int32_t> tempBufferInt32 = tmpLocal[tilling.srcIndexOffset].template ReinterpretCast<int32_t>();
    LocalTensor<float> tempBufferFloat = tmpLocal[tilling.srcIndexOffset].template ReinterpretCast<float>();
    Cast<int32_t, float, false>(tempBufferInt32,
        tempBufferFloat,
        RoundMode::CAST_ROUND,
        MASK_PLACEHOLDER,
        1,
        {1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void TopKNSmallGetFloatTopKValue(const LocalTensor<T> &dstValueLocal,
    const LocalTensor<int32_t> &dstIndexLocal, const LocalTensor<T> &tmpLocal, const TopkTiling &tilling,
    const TopKInfo &topKInfo)
{
    LocalTensor<uint32_t> src1stackTensor = tmpLocal.template ReinterpretCast<uint32_t>();
    src1stackTensor.SetSize(DEFAULT_BLK_NUM);
    SetVectorMask<T, MaskMode::COUNTER>(0, DEFAULT_BLK_NUM);
    Duplicate<uint32_t, false>(
        src1stackTensor, static_cast<uint32_t>(0), MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
    auto eventID = GetTPipePtr()->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(eventID);
    WaitFlag<HardEvent::V_S>(eventID);
    SetVectorMask<T, MaskMode::COUNTER>(0, 1);
    Duplicate<uint32_t, false>(
        src1stackTensor, static_cast<uint32_t>(tilling.vreduceValMask0), MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
    SetVectorMask<float>(topKInfo.inner);
    struct GatherMaskParams gatherMaskParams(DEFAULT_BLK_STRIDE, topKInfo.outter, HALF_DEFAULT_REPEAT_STRIDE, 0);
    uint64_t rsvdCnt = 0;
    GatherMask<float, uint32_t>(dstValueLocal, tmpLocal[tilling.innerDataSize], src1stackTensor,
                                false, 0, gatherMaskParams, rsvdCnt);
    LocalTensor<T> tempBuffer = dstIndexLocal.template ReinterpretCast<T>();
    struct GatherMaskParams gatherMaskParams2(DEFAULT_BLK_STRIDE, topKInfo.outter, HALF_DEFAULT_REPEAT_STRIDE, 0);
    GatherMask<T, uint32_t>(tempBuffer, tmpLocal[tilling.srcIndexOffset], src1stackTensor,
                            false, 0, gatherMaskParams2, rsvdCnt);
}

template <typename T>
__aicore__ inline void TopKNSmallGetHalfTopKValue(const LocalTensor<T> &dstValueLocal,
    const LocalTensor<int32_t> &dstIndexLocal, const LocalTensor<T> &tmpLocal, const TopkTiling &tilling,
    const TopKInfo &topKInfo)
{
    LocalTensor<uint16_t> src1stackTensor = tmpLocal.template ReinterpretCast<uint16_t>();
    src1stackTensor.SetSize(DEFAULT_BLK_NUM);
    SetVectorMask<T, MaskMode::COUNTER>(0, DEFAULT_BLK_NUM);
    Duplicate<uint16_t, false>(
        src1stackTensor, static_cast<uint16_t>(0), MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
    auto eventID = GetTPipePtr()->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(eventID);
    WaitFlag<HardEvent::V_S>(eventID);
    SetVectorMask<T, MaskMode::COUNTER>(0, 2);
    Duplicate<uint16_t, false>(src1stackTensor, static_cast<uint16_t>(tilling.vreduceValMask1), MASK_PLACEHOLDER, 1, 1, 
        DEFAULT_REPEAT_STRIDE);
    SetVectorMask<T, MaskMode::COUNTER>(0, 1);
    Duplicate<uint16_t, false>(src1stackTensor, static_cast<uint16_t>(tilling.vreduceValMask0), MASK_PLACEHOLDER, 1, 1, 
        DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
    SetVectorMask<half>(topKInfo.inner);
    struct GatherMaskParams gatherMaskParams(DEFAULT_BLK_STRIDE, topKInfo.outter, ONE_FOURTH_DEFAULT_REPEAT_STRIDE, 0);
    uint64_t rsvdCnt = 0;
    GatherMask<half, uint16_t>(dstValueLocal, tmpLocal[tilling.innerDataSize], src1stackTensor, false, 0, 
        gatherMaskParams, rsvdCnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint32_t> indexStackTensor = tmpLocal.template ReinterpretCast<uint32_t>();
    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(0, 2);
    Duplicate<uint32_t, false>(indexStackTensor, static_cast<uint32_t>(0), MASK_PLACEHOLDER, 1, 1, 
        DEFAULT_REPEAT_STRIDE);
    SetVectorMask<T, MaskMode::COUNTER>(0, 1);
    Duplicate<uint32_t, false>(indexStackTensor, static_cast<uint32_t>(tilling.vreduceIdxMask0), MASK_PLACEHOLDER, 1, 1,
        DEFAULT_REPEAT_STRIDE);
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
    SetVectorMask<float>(topKInfo.inner);
    LocalTensor<float> tempBuffer, src0Buffer, src1Buffer;
    tempBuffer = dstIndexLocal.template ReinterpretCast<float>();
    src0Buffer = tmpLocal[tilling.srcIndexOffset].template ReinterpretCast<float>();
    struct GatherMaskParams gatherMaskParams2(DEFAULT_BLK_STRIDE, topKInfo.outter, HALF_DEFAULT_REPEAT_STRIDE, 0);
    GatherMask<float, uint32_t>(tempBuffer, src0Buffer, indexStackTensor,
                                false, 0, gatherMaskParams2, rsvdCnt);
}

template <typename T, bool isInitIndex, bool isHasfinish>
__aicore__ inline void TopKNSmallCompute(const LocalTensor<T> &dstValueLocal, const LocalTensor<int32_t> &dstIndexLocal,
    const LocalTensor<T> &srcLocal, const LocalTensor<int32_t> &srcIndexLocal, const LocalTensor<bool> &finishLocal,
    const LocalTensor<T> &tmpLocal, const int32_t k, const TopkTiling &tilling, const TopKInfo &topKInfo,
    const bool isLargest)
{
    CastIdxTypeToValType<T, isInitIndex>(
        srcIndexLocal, tmpLocal, tilling, topKInfo, tilling.allDataSize, tilling.topkNSmallSrcIndexOffset);
    ProposalConcatValIdxSmall<T, isInitIndex>(srcLocal, tmpLocal, tilling, isLargest);
    RpSort16<T>(tmpLocal[tilling.innerDataSize], tmpLocal, tilling.sortRepeat);
    PipeBarrier<PIPE_V>();
    // When the number of participants in the topk is greater than 16, the merging sorting is performed.
    if (topKInfo.n > MIN_RPSORT16_SIZE) {
        for (int j = 0; j < topKInfo.outter; ++j) {
            auto src = tmpLocal[tilling.innerDataSize + tilling.mrgSortSrc1offset * j];
            uint16_t elementLengths[MRG_SORT_ELEMENT_LEN] = {MIN_RPSORT16_SIZE, MIN_RPSORT16_SIZE, 0, 0};
            struct MrgSort4Info srcInfo(elementLengths, false, 0b0011, 1);
            struct MrgSortSrcList<T> srcList(src, src[tilling.mrgSortTwoQueueSrc1Offset], src, src);
            MrgSort4<T>(tmpLocal[tilling.mrgSortSrc1offset * j], srcList, srcInfo);
        }
    } else {
        const DataCopyParams intriParams = {static_cast<uint16_t>(tilling.copyUbToUbBlockCount), 1, 0, 0};
        DataCopy(tmpLocal, tmpLocal[tilling.innerDataSize], intriParams);
    }
    PipeBarrier<PIPE_V>();
    ProposalExtract<T>(tmpLocal[tilling.innerDataSize], tmpLocal, tilling.sortRepeat, 4);
    ProposalExtract<T>(tmpLocal[tilling.topkNSmallSrcIndexOffset], tmpLocal, tilling.sortRepeat, 0);
    PipeBarrier<PIPE_V>();
    CastIdxToInt<T>(tmpLocal, tilling);

    if constexpr (sizeof(T) == sizeof(float)) {
        TopKNSmallGetFloatTopKValue<T>(dstValueLocal, dstIndexLocal, tmpLocal, tilling, topKInfo);
    } else {
        TopKNSmallGetHalfTopKValue<T>(dstValueLocal, dstIndexLocal, tmpLocal, tilling, topKInfo);
    }
}

__aicore__ inline void CopyData(LocalTensor<int32_t> indexLocalTmp, const TopKInfo &topKInfo)
{
    SetMaskNorm();
    ResetMask();
    SetVectorMask<int32_t>(topKInfo.inner);
    const UnaryRepeatParams unaryParams = {1, 1, 4, 0};
    Adds<int32_t, false>(indexLocalTmp[topKInfo.inner], indexLocalTmp, 0, MASK_PLACEHOLDER, topKInfo.outter - 1, 
        unaryParams);
    PipeBarrier<PIPE_V>();
}

}  // namespace AscendC

#endif  // IMPL_SORT_TOPK_TOPK_V200_IMPL_H