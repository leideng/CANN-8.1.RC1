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
 * \file topk_v220_impl.h
 * \brief
 */
#ifndef IMPL_SORT_TOPK_TOPK_V220_IMPL_H
#define IMPL_SORT_TOPK_TOPK_V220_IMPL_H
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
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
    ASCENDC_ASSERT(ans, {
        KERNEL_LOG(KERNEL_ERROR, "type must be half or float");
    });
    ans = (topKInfo.inner % TOPK_INNER_ALIGN_LEN == 0);
    ASCENDC_ASSERT(ans, {
        KERNEL_LOG(KERNEL_ERROR, "The value of inner must be an integer multiple of 32.");
    });

    if constexpr (topkMode == TopKMode::TOPK_NORMAL) {
        ans = (topKInfo.inner <= TOPK_NORMAL_INNER_MAX_LEN);
        ASCENDC_ASSERT(ans, {
            KERNEL_LOG(KERNEL_ERROR, "The maximum value supported by inner is 4096.");
        });
    }
    if constexpr (topkMode == TopKMode::TOPK_NSMALL) {
        ans = (topKInfo.inner == TOPK_NSMALL_INNER_LEN);
        ASCENDC_ASSERT(ans, {
            KERNEL_LOG(KERNEL_ERROR, "The value of inner must be an equal 32.");
        });
    }
}
#endif

template <typename T>
__aicore__ inline void MrgFourQueueSort(const LocalTensor<T> &tmpLocal, const TopkTiling &tilling,
    const TopKInfo &topKInfo, uint16_t &z, int32_t &mrgFourQueueCount, int32_t &dstIdx)
{
    uint16_t innerU16Type = static_cast<uint16_t>(topKInfo.inner);
    for (; z * MRGSORT_VALID_QUEUE <= innerU16Type; z *= MRGSORT_VALID_QUEUE) {
        auto src = (mrgFourQueueCount % TWO == 1) ? tmpLocal[tilling.innerDataSize] : tmpLocal[0];
        dstIdx = (mrgFourQueueCount + 1) % TWO;
        auto dst = (dstIdx == 1) ? tmpLocal[tilling.innerDataSize] : tmpLocal[0];
        mrgFourQueueCount += 1;
        uint16_t elementLengths[MRG_SORT_ELEMENT_LEN] = {z, z, z, z};
        struct MrgSort4Info srcInfo(elementLengths, false, 0b1111, tilling.mrgSortRepeat / z);
        struct MrgSortSrcList<T> srcList(src,
            src[z * tilling.mrgSortSrc1offset],
            src[z * tilling.mrgSortSrc2offset],
            src[z * tilling.mrgSortSrc3offset]);
        MrgSort<T>(dst, srcList, srcInfo);
        PipeBarrier<PIPE_V>();
        const DataCopyParams intriParams = {static_cast<uint16_t>(tilling.copyUbToUbBlockCount), 1, 0, 0};
        DataCopy(src, dst, intriParams);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void MrgTwoQueueSort(const LocalTensor<T> &tmpLocal, const TopkTiling &tilling,
    const TopKInfo &topKInfo, const uint16_t z, const int32_t mrgFourQueueCount, int32_t &dstIdx, const int32_t k)
{
    int32_t arrayCount = 0;
    if (z < topKInfo.inner) {
        // MRG_MAX_ARRAY_SIZE is the number of four-way merging times is multiplied by three;
        int32_t mrgArray[MRG_MAX_ARRAY_SIZE] = {0};
        int32_t tmpInner = topKInfo.inner;
        for (int32_t i = z; i >= MIN_SORT32_SIZE; i /= MRGSORT_VALID_QUEUE) {
            int32_t count;
            for (count = 0; count < tmpInner / i; ++count) {
                mrgArray[arrayCount++] = i;
            }
            tmpInner -= count * i;
        }
        uint16_t mrgSortedLen = 0;
        for (int32_t i = 0; i < arrayCount - 1; ++i) {
            auto src = ((mrgFourQueueCount + i) % TWO == 1) ? tmpLocal[tilling.innerDataSize] : tmpLocal[0];
            dstIdx = (mrgFourQueueCount + 1 + i) % TWO;
            auto dst = (dstIdx == 1) ? tmpLocal[tilling.innerDataSize] : tmpLocal[0];
            mrgSortedLen += static_cast<uint16_t>(mrgArray[i]);
            uint64_t tmpMrgSortedLen = mrgSortedLen;
            uint64_t tmpMrgArray = mrgArray[i + 1];
            if (mrgSortedLen > k) {
                tmpMrgSortedLen = k;
            }
            if (mrgArray[i + 1] > k) {
                tmpMrgArray = k;
            }
            uint16_t elementLengths[MRG_SORT_ELEMENT_LEN] = {
                static_cast<uint16_t>(tmpMrgSortedLen), static_cast<uint16_t>(tmpMrgArray), 0, 0};
            struct MrgSort4Info srcInfo(elementLengths, false, 0b0011, 1);
            struct MrgSortSrcList<T> srcList(src, src[mrgSortedLen * tilling.mrgSortTwoQueueSrc1Offset], src, src);
            MrgSort<T>(dst, srcList, srcInfo);
            PipeBarrier<PIPE_V>();
        }
    }
}

template <typename T>
__aicore__ inline void GatherDstValAndDstIdx(const LocalTensor<T> &dstValueLocal,
    const LocalTensor<int32_t> &dstIndexLocal, const LocalTensor<T> &tmpLocal, const TopkTiling &tilling,
    const int32_t dstIdx, const int32_t dstOffsetFourBytes, const int outterIdx)
{
    uint64_t rsvdCnt = 0;
    int32_t tmpLocalDstOffset = tilling.innerDataSize * dstIdx;
    if constexpr (sizeof(T) == sizeof(float)) {
        // Get Value, The index of the odd position is obtained for each repeat.
        struct GatherMaskParams reducev2Params(DEFAULT_BLK_STRIDE, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_BLK_STRIDE);
        GatherMask<T>(dstValueLocal[dstOffsetFourBytes], tmpLocal[tmpLocalDstOffset], REDUCEV2_MODE_ONE,
                      true, tilling.maskVreducev2FourBytes, reducev2Params, rsvdCnt);
    } else {
        int32_t dstOffsetTwoBytes = outterIdx * tilling.kAlignTwoBytes;
        // Get Value. The first element is used for every four elements in each repeat.
        struct GatherMaskParams reducev2Params(DEFAULT_BLK_STRIDE, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_BLK_STRIDE);
        GatherMask<T>(dstValueLocal[dstOffsetTwoBytes], tmpLocal[tmpLocalDstOffset], REDUCEV2_MODE_THREE,
                      true, tilling.maskVreducev2TwoBytes, reducev2Params, rsvdCnt);
    }
    PipeBarrier<PIPE_V>();
    // Get Index, The index of the even position is obtained for each repeat.
    LocalTensor<float> tempBuffer = dstIndexLocal[dstOffsetFourBytes].template ReinterpretCast<float>();
    LocalTensor<float> tempBufferLocal = tmpLocal[tmpLocalDstOffset].template ReinterpretCast<float>();
    struct GatherMaskParams reducev2Params(DEFAULT_BLK_STRIDE, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_BLK_STRIDE);
    GatherMask<float>(tempBuffer, tempBufferLocal, REDUCEV2_MODE_TWO, true,
                      tilling.maskVreducev2FourBytes, reducev2Params, rsvdCnt);
    PipeBarrier<PIPE_V>();
    SetMaskCount();
}

template <typename T, bool isInitIndex>
__aicore__ inline void TmpLocalSort32(const LocalTensor<T> &srcLocal, const LocalTensor<int32_t> &srcIndexLocal,
    const LocalTensor<T> &tmpLocal, const TopkTiling &tilling, const TopKInfo &topKInfo, const bool isLargest,
    const int outterIdx, const UnaryRepeatParams unaryParams)
{
    // The number of inners is divided into inner/32 groups for sorting. Each iteration completes one group of 32
    // numbers of sorting.
    int offset = outterIdx * topKInfo.inner;
    if (!isLargest) {
        SetVectorMask<T, MaskMode::COUNTER>(0, topKInfo.inner);
        Muls<T, false>(tmpLocal[tilling.innerDataSize], srcLocal[offset], T(-1), MASK_PLACEHOLDER, 1, unaryParams);
        PipeBarrier<PIPE_V>();
        if constexpr (!isInitIndex) {
            LocalTensor<uint32_t> tempBufferUint32 = tmpLocal[tilling.srcIndexOffset].template
                                                     ReinterpretCast<uint32_t>();
            Sort32<T>(tmpLocal, tmpLocal[tilling.innerDataSize], tempBufferUint32, tilling.sortRepeat);
        } else {
            LocalTensor<uint32_t> tempBufferUint32 = srcIndexLocal.template ReinterpretCast<uint32_t>();
            Sort32<T>(tmpLocal, tmpLocal[tilling.innerDataSize], tempBufferUint32, tilling.sortRepeat);
        }
    } else {
        if constexpr (!isInitIndex) {
            LocalTensor<uint32_t> tempBufferUint32 = tmpLocal[tilling.srcIndexOffset].template
                                                     ReinterpretCast<uint32_t>();
            Sort32<T>(tmpLocal, srcLocal[offset], tempBufferUint32, tilling.sortRepeat);
        } else {
            LocalTensor<uint32_t> tempBufferUint32 = srcIndexLocal.template ReinterpretCast<uint32_t>();
            Sort32<T>(tmpLocal, srcLocal[offset], tempBufferUint32, tilling.sortRepeat);
        }
    }
    PipeBarrier<PIPE_V>();
    const DataCopyParams intriParams = {static_cast<uint16_t>(tilling.copyUbToUbBlockCount), 1, 0, 0};
    DataCopy(tmpLocal[tilling.innerDataSize], tmpLocal, intriParams);
    PipeBarrier<PIPE_V>();
}

template <typename T, bool isInitIndex, bool isHasfinish>
__aicore__ inline void TopKCompute(const LocalTensor<T> &dstValueLocal, const LocalTensor<int32_t> &dstIndexLocal,
    const LocalTensor<T> &srcLocal, const LocalTensor<int32_t> &srcIndexLocal, const LocalTensor<bool> &finishLocal,
    const LocalTensor<T> &tmpLocal, const int32_t k, const TopkTiling &tilling, const TopKInfo &topKInfo,
    const bool isLargest)
{
    const UnaryRepeatParams unaryParams;
    for (int j = 0; j < topKInfo.outter; ++j) {
        int32_t dstOffsetFourBytes = j * tilling.kAlignFourBytes;
        TmpLocalSort32<T, isInitIndex>(srcLocal, srcIndexLocal, tmpLocal, tilling, topKInfo, isLargest, j, unaryParams);

        int32_t mrgFourQueueCount = 0;
        uint16_t z = MIN_SORT32_SIZE;
        int32_t dstIdx = 0;
        MrgFourQueueSort<T>(tmpLocal, tilling, topKInfo, z, mrgFourQueueCount, dstIdx);
        MrgTwoQueueSort<T>(tmpLocal, tilling, topKInfo, z, mrgFourQueueCount, dstIdx, k);
        GatherDstValAndDstIdx(dstValueLocal, dstIndexLocal, tmpLocal, tilling, dstIdx, dstOffsetFourBytes, j);

        if constexpr (isHasfinish) {
            bool finishValue = finishLocal.GetValue(j);
            auto eventID = GetTPipePtr()->FetchEventID(HardEvent::S_V);
            SetFlag<HardEvent::S_V>(eventID);
            WaitFlag<HardEvent::S_V>(eventID);

            if (finishValue) {
                SetVectorMask<T, MaskMode::COUNTER>(0, k);
                Duplicate<int32_t, false>(dstIndexLocal[dstOffsetFourBytes],
                    static_cast<int32_t>(topKInfo.n),
                    MASK_PLACEHOLDER,
                    1,
                    1,
                    DEFAULT_REPEAT_STRIDE);
            }
        }
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void TopKNSmallGetFloatTopKValue(const LocalTensor<float> &dstValueLocal,
    const LocalTensor<int32_t> &dstIndexLocal, const LocalTensor<float> &tmpLocal, const TopkTiling &tilling,
    const TopKInfo &topKInfo)
{
    LocalTensor<uint32_t> src1stackTensor = tmpLocal[tilling.topkMrgSrc1MaskSizeOffset].template
                                            ReinterpretCast<uint32_t>();
    auto eventID = GetTPipePtr()->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(eventID);
    WaitFlag<HardEvent::V_S>(eventID);
    src1stackTensor.SetSize(SRC1_STACK_TENSORSIZE);
    // 0b01010101010101010101010101010101
    src1stackTensor.SetValue(0, tilling.vreduceValMask0);
    // 0b01010101010101010101010101010101
    src1stackTensor.SetValue(1, tilling.vreduceValMask1);

    src1stackTensor.SetValue(EIGHT, tilling.vreduceIdxMask0);
    src1stackTensor.SetValue(NINE, tilling.vreduceIdxMask1);

    eventID = GetTPipePtr()->FetchEventID(HardEvent::S_V);
    SetFlag<HardEvent::S_V>(eventID);
    WaitFlag<HardEvent::S_V>(eventID);
    // Get Value, The index of the odd position is obtained for each repeat.
    struct GatherMaskParams reducev2Params(DEFAULT_BLK_STRIDE, topKInfo.outter, DEFAULT_REPEAT_STRIDE, 0);
    uint64_t rsvdCnt = 0;
    GatherMask<float, uint32_t>(dstValueLocal, tmpLocal, src1stackTensor,
                                true, VREDUCEV2_FOUR_BYTE_MASK, reducev2Params, rsvdCnt);

    // Get Index, The index of the even position is obtained for each repeat.
    LocalTensor<float> tempBuffer = dstIndexLocal.template ReinterpretCast<float>();
    struct GatherMaskParams reducev2Params2(DEFAULT_BLK_STRIDE, topKInfo.outter, DEFAULT_REPEAT_STRIDE, 0);
    GatherMask<float, uint32_t>(tempBuffer, tmpLocal,
                                tmpLocal[tilling.topkMrgSrc1MaskSizeOffset + EIGHT].ReinterpretCast<uint32_t>(),
                                true, VREDUCEV2_FOUR_BYTE_MASK, reducev2Params2, rsvdCnt);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void TopKNSmallGetHalfTopKValue(const LocalTensor<half> &dstValueLocal,
    const LocalTensor<int32_t> &dstIndexLocal, const LocalTensor<half> &tmpLocal, const TopkTiling &tilling,
    const TopKInfo &topKInfo)
{
    LocalTensor<uint16_t> src1stackTensor = tmpLocal[tilling.topkMrgSrc1MaskSizeOffset].template
                                            ReinterpretCast<uint16_t>();
    auto eventID = GetTPipePtr()->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(eventID);
    WaitFlag<HardEvent::V_S>(eventID);
    src1stackTensor.SetSize(EIGHT);
    src1stackTensor.SetValue(0, tilling.vreducehalfValMask0);
    src1stackTensor.SetValue(1, tilling.vreducehalfValMask1);
    src1stackTensor.SetValue(TWO, tilling.vreducehalfValMask2);
    src1stackTensor.SetValue(THREE, tilling.vreducehalfValMask3);
    src1stackTensor.SetValue(FOUR, tilling.vreducehalfValMask4);
    src1stackTensor.SetValue(FIVE, tilling.vreducehalfValMask5);
    src1stackTensor.SetValue(SIX, tilling.vreducehalfValMask6);
    src1stackTensor.SetValue(SEVEN, tilling.vreducehalfValMask7);
    LocalTensor<uint32_t> indexstackTensor = tmpLocal[tilling.topkMrgSrc1MaskSizeOffset + SRC1_STACK_VAL_OFFSET].
                                             template ReinterpretCast<uint32_t>();;
    indexstackTensor.SetSize(TWO);
    indexstackTensor.SetValue(0, tilling.vreduceIdxMask0);
    indexstackTensor.SetValue(1, tilling.vreduceIdxMask1);
    eventID = GetTPipePtr()->FetchEventID(HardEvent::S_V);
    SetFlag<HardEvent::S_V>(eventID);
    WaitFlag<HardEvent::S_V>(eventID);
    // Get Value. The first element is used for every four elements in each repeat.
    struct GatherMaskParams reducev2Params(DEFAULT_BLK_STRIDE, topKInfo.outter, DEFAULT_REPEAT_STRIDE, 0);
    uint64_t rsvdCnt = 0;
    GatherMask<half, uint16_t>(dstValueLocal, tmpLocal, src1stackTensor,
                               true, VREDUCEV2_HALF_MASK, reducev2Params, rsvdCnt);
    PipeBarrier<PIPE_V>();
    // Get Index, The index of the even position is obtained for each repeat.
    LocalTensor<float> tempBufferIndex = dstIndexLocal.template ReinterpretCast<float>();
    LocalTensor<float> tempBufferLocal = tmpLocal.template ReinterpretCast<float>();
    struct GatherMaskParams reducev2Params2(DEFAULT_BLK_STRIDE, topKInfo.outter, DEFAULT_REPEAT_STRIDE, 0);
    GatherMask<float, uint32_t>(tempBufferIndex, tempBufferLocal,
        tempBufferLocal[(tilling.topkMrgSrc1MaskSizeOffset + SRC1_STACK_VAL_OFFSET) / TWO].ReinterpretCast<uint32_t>(),
        true, VREDUCEV2_FOUR_BYTE_MASK, reducev2Params2, rsvdCnt);
    PipeBarrier<PIPE_V>();
}

template <typename T, bool isInitIndex, bool isHasfinish>
__aicore__ inline void TopKNSmallCompute(const LocalTensor<T> &dstValueLocal, const LocalTensor<int32_t> &dstIndexLocal,
    const LocalTensor<T> &srcLocal, const LocalTensor<int32_t> &srcIndexLocal, const LocalTensor<bool> &finishLocal,
    const LocalTensor<T> &tmpLocal, const int32_t k, const TopkTiling &tilling, const TopKInfo &topKInfo,
    const bool isLargest)
{
    if (!isLargest) {
        if (!isInitIndex) {
            // Repeat = inner * outter / 32, inner=32
            LocalTensor<uint32_t> tempBufferUint32 = tmpLocal[tilling.topkNSmallSrcIndexOffset].template
                                                     ReinterpretCast<uint32_t>();
            Sort32<T>(tmpLocal, tmpLocal[tilling.innerDataSize], tempBufferUint32, topKInfo.outter);
        } else {
            // Repeat = inner * outter / 32, inner=32
            LocalTensor<uint32_t> tempBufferUint32 = srcIndexLocal.template ReinterpretCast<uint32_t>();
            Sort32<T>(tmpLocal, tmpLocal[tilling.innerDataSize], tempBufferUint32, topKInfo.outter);
        }
    } else {
        if (!isInitIndex) {
            // Repeat = inner * outter / 32, inner=32
            LocalTensor<uint32_t> tempBufferUint32 = tmpLocal[tilling.topkNSmallSrcIndexOffset].template
                                                     ReinterpretCast<uint32_t>();
            Sort32<T>(tmpLocal, srcLocal, tempBufferUint32, topKInfo.outter);
        } else {
            // Repeat = inner * outter / 32, inner=32
            LocalTensor<uint32_t> tempBufferUint32 = srcIndexLocal.template ReinterpretCast<uint32_t>();
            Sort32<T>(tmpLocal, srcLocal, tempBufferUint32, topKInfo.outter);
        }
    }
    PipeBarrier<PIPE_V>();

    if constexpr (sizeof(T) == sizeof(float)) {
        TopKNSmallGetFloatTopKValue(dstValueLocal, dstIndexLocal, tmpLocal, tilling, topKInfo);
    } else {
        TopKNSmallGetHalfTopKValue(dstValueLocal, dstIndexLocal, tmpLocal, tilling, topKInfo);
    }
}

__aicore__ inline void CopyData(LocalTensor<int32_t> indexLocalTmp, const TopKInfo &topKInfo)
{
    Copy(indexLocalTmp[topKInfo.inner], indexLocalTmp, THIRTY_TWO, topKInfo.outter - 1, {1, 1, FOUR, 0});
    PipeBarrier<PIPE_V>();
}

}  // namespace AscendC

#endif  // IMPL_SORT_TOPK_TOPK_V220_IMPL_H