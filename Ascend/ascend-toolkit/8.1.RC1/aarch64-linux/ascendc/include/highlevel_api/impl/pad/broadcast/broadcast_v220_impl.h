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
 * \file broadcast_v220_impl.h
 * \brief
 */
#ifndef IMPL_PAD_BROADCAST_BROADCAST_V220_IMPL_H
#define IMPL_PAD_BROADCAST_BROADCAST_V220_IMPL_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"

namespace AscendC {
constexpr uint32_t BRCB_ONE_SIZE = 8;
constexpr uint32_t BRCB_HALF_MAX_REPEATE_TIMES = 254;
constexpr uint32_t BRCB_FLOAT_MAX_REPEATE_TIMES = 255;
constexpr uint8_t GATHER_MASK_PATTERN = 7;

template <typename T>
__aicore__ inline void BrcbToOneBlock(const LocalTensor<T> &srcLocal, const uint32_t firstDim,
    uint32_t oneBlockElementNum, LocalTensor<T> &brcbOneBlockTempBuffer)
{
    const uint32_t brcbRepeatTime = (firstDim + BRCB_ONE_SIZE - 1) / BRCB_ONE_SIZE;
    uint32_t brcbMaxRepeateTimes = BRCB_HALF_MAX_REPEATE_TIMES;
    if constexpr (sizeof(T) == sizeof(float)) {
        brcbMaxRepeateTimes = BRCB_FLOAT_MAX_REPEATE_TIMES;
    }
    const uint32_t brcbCount = brcbRepeatTime / brcbMaxRepeateTimes;
    const uint32_t tailBrcbRepeateTime = brcbRepeatTime % brcbMaxRepeateTimes;
    uint32_t brcbSrcOffset = 0;
    uint32_t brcbOneBlockTempBufferOffset = 0;
    for (uint32_t i = 0; i < brcbCount; i++) {
        Brcb(brcbOneBlockTempBuffer[brcbOneBlockTempBufferOffset],
            srcLocal[brcbSrcOffset],
            brcbMaxRepeateTimes,
            {1, DEFAULT_REPEAT_STRIDE});
        brcbOneBlockTempBufferOffset += brcbMaxRepeateTimes * oneBlockElementNum * BRCB_ONE_SIZE;
        brcbSrcOffset += brcbMaxRepeateTimes * BRCB_ONE_SIZE;
    }
    if (tailBrcbRepeateTime != 0) {
        Brcb(brcbOneBlockTempBuffer[brcbOneBlockTempBufferOffset],
            srcLocal[brcbSrcOffset],
            tailBrcbRepeateTime,
            {1, DEFAULT_REPEAT_STRIDE});
    }
    PipeBarrier<PIPE_V>();
}

template <typename T, bool isReuseSource>
__aicore__ inline void TwoDimBroadCastLastDimAlign220(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    LocalTensor<T> &tmpBuffer, const uint32_t firstDim, const uint32_t blockDim)
{
    constexpr uint32_t oneBlockElementNum = ONE_BLK_SIZE / sizeof(T);
    BrcbToOneBlock(srcLocal, firstDim, oneBlockElementNum, tmpBuffer);
    SetVectorMask<T, MaskMode::COUNTER>(blockDim);
    const CopyRepeatParams copyRepeatParams = {1, 0, (uint16_t)(blockDim / oneBlockElementNum), 1};  // overflow check
    uint32_t CopyCounts = firstDim / MAX_REPEAT_TIMES;
    uint32_t dstOffset = 0;
    uint32_t brcbOneBlockTempBufferOffset = 0;
    for (uint32_t i = 0; i < CopyCounts; i++) {
        Copy<T, false>(dstLocal[dstOffset],
            tmpBuffer[brcbOneBlockTempBufferOffset],
            MASK_PLACEHOLDER,
            MAX_REPEAT_TIMES,
            copyRepeatParams);
        dstOffset += MAX_REPEAT_TIMES * blockDim;
        brcbOneBlockTempBufferOffset += MAX_REPEAT_TIMES * oneBlockElementNum;
    }
    uint32_t tailsCopyRepeateTimes = firstDim % MAX_REPEAT_TIMES;
    if (tailsCopyRepeateTimes != 0) {
        Copy<T, false>(dstLocal[dstOffset],
            tmpBuffer[brcbOneBlockTempBufferOffset],
            MASK_PLACEHOLDER,
            tailsCopyRepeateTimes,
            copyRepeatParams);
    }
    PipeBarrier<PIPE_V>();
}

template <typename T, bool isReuseSource>
__aicore__ inline void TwoDimBroadCastLastDimNotAlign220(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    LocalTensor<T> &tmpBuffer, const uint32_t firstDim, const uint32_t blockDim)
{
    constexpr uint32_t oneBlockElementNum = ONE_BLK_SIZE / sizeof(T);
    BrcbToOneBlock(srcLocal, firstDim, oneBlockElementNum, tmpBuffer);
    const uint32_t blockDimAlignBlockNum = (blockDim + oneBlockElementNum - 1) / oneBlockElementNum;
    const uint32_t blockDimAlign = blockDimAlignBlockNum * oneBlockElementNum;
    SetVectorMask<T, MaskMode::COUNTER>(blockDimAlign);
    const CopyRepeatParams copyRepeatParams = {1, 0, (uint16_t)blockDimAlignBlockNum, 1};
    uint32_t CopyCounts = firstDim / MAX_REPEAT_TIMES;
    uint32_t dstOffset = 0;
    uint32_t brcbOneBlockTempBufferOffset = 0;
    auto copyTempBuffer = tmpBuffer[firstDim * oneBlockElementNum];
    for (uint32_t i = 0; i < CopyCounts; i++) {
        Copy<T, false>(copyTempBuffer[dstOffset],
            tmpBuffer[brcbOneBlockTempBufferOffset],
            MASK_PLACEHOLDER,
            MAX_REPEAT_TIMES,
            copyRepeatParams);
        dstOffset += MAX_REPEAT_TIMES * blockDimAlign;
        brcbOneBlockTempBufferOffset += MAX_REPEAT_TIMES * oneBlockElementNum;
    }
    uint32_t tailsCopyRepeateTimes = firstDim % MAX_REPEAT_TIMES;
    if (tailsCopyRepeateTimes != 0) {
        Copy<T, false>(copyTempBuffer[dstOffset],
            tmpBuffer[brcbOneBlockTempBufferOffset],
            MASK_PLACEHOLDER,
            tailsCopyRepeateTimes,
            copyRepeatParams);
    }
    PipeBarrier<PIPE_V>();
    const GatherMaskParams gatherMaskParams = {
        1, (uint16_t)firstDim, (uint16_t)blockDimAlignBlockNum, 0};  // uint32 cast to uint16
    uint64_t rsvdCnt = 0;
    GatherMask(dstLocal, copyTempBuffer, GATHER_MASK_PATTERN, true, blockDim, gatherMaskParams, rsvdCnt);
    SetMaskCount();
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void GetAlignLoopNumbers(const uint32_t firstDim, const uint32_t blockDim,
    const uint32_t tmpBufferSize, uint32_t &oneRepeateSize, uint32_t &rangeM, uint32_t &tailM)
{
    constexpr uint32_t oneBlockElementNum = ONE_BLK_SIZE / sizeof(T);
    constexpr uint32_t minBrcbTempBufferSize = oneBlockElementNum * oneBlockElementNum;
    constexpr uint32_t minTmpBufferSize = minBrcbTempBufferSize;
    ASCENDC_ASSERT((tmpBufferSize >= minTmpBufferSize), {
        KERNEL_LOG(KERNEL_ERROR,
            "tmpBufferSize can't smaller than minTmpBufferSize, tmpBufferSize is %u, minTmpBufferSize is %u!",
            tmpBufferSize,
            minTmpBufferSize);
    });
    oneRepeateSize = tmpBufferSize / minTmpBufferSize * oneBlockElementNum;
    rangeM = firstDim / oneRepeateSize;
    tailM = firstDim - oneRepeateSize * rangeM;
}

template <typename T>
__aicore__ inline void GetNotAlignLoopNumbers(const uint32_t firstDim, const uint32_t blockDim,
    const uint32_t tmpBufferSize, uint32_t &oneRepeateSize, uint32_t &rangeM, uint32_t &tailM)
{
    constexpr uint32_t oneBlockElementNum = ONE_BLK_SIZE / sizeof(T);
    constexpr uint32_t minBrcbTempBufferSize = oneBlockElementNum * oneBlockElementNum;
    const uint32_t blockDimAlignBlockNum = (blockDim + oneBlockElementNum - 1) / oneBlockElementNum;
    const uint32_t blockDimAlign = blockDimAlignBlockNum * oneBlockElementNum;
    const uint32_t minCopyTempBufferSize = oneBlockElementNum * blockDimAlign;
    const uint32_t minTmpBufferSize = minBrcbTempBufferSize + minCopyTempBufferSize;
    ASCENDC_ASSERT((tmpBufferSize >= minTmpBufferSize), {
        KERNEL_LOG(KERNEL_ERROR,
            "tmpBufferSize can't smaller than minTmpBufferSize, tmpBufferSize is %u, minTmpBufferSize is %u!",
            tmpBufferSize,
            minTmpBufferSize);
    });
    oneRepeateSize = tmpBufferSize / minTmpBufferSize * oneBlockElementNum;
    rangeM = firstDim / oneRepeateSize;
    tailM = firstDim - oneRepeateSize * rangeM;
}

template <typename T, int32_t dim, int32_t axis, bool isReuseSource = false>
__aicore__ inline void TwoDimBroadCastLastDim(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const uint32_t dstShape[dim], const uint32_t srcShape[dim], LocalTensor<T> &tmpBuffer)
{
    const auto firstDim = dstShape[0];
    const auto blockDim = dstShape[axis];
    uint32_t oneRepeateSize = 0;
    uint32_t rangeM = 0;
    uint32_t tailM = 0;
    uint32_t dstLocalOffset = 0;
    uint32_t srcLocalOffset = 0;
    if (blockDim * sizeof(T) % ONE_BLK_SIZE == 0) {
        GetAlignLoopNumbers<T>(firstDim, blockDim, tmpBuffer.GetSize(), oneRepeateSize, rangeM, tailM);
        for (uint32_t i = 0; i < rangeM; i++) {
            TwoDimBroadCastLastDimAlign220<T, isReuseSource>(
                dstLocal[dstLocalOffset], srcLocal[srcLocalOffset], tmpBuffer, oneRepeateSize, blockDim);
            dstLocalOffset += oneRepeateSize * blockDim;
            srcLocalOffset += oneRepeateSize;
        }

        if (tailM != 0) {
            TwoDimBroadCastLastDimAlign220<T, isReuseSource>(
                dstLocal[dstLocalOffset], srcLocal[srcLocalOffset], tmpBuffer, tailM, blockDim);
        }
    } else {
        GetNotAlignLoopNumbers<T>(firstDim, blockDim, tmpBuffer.GetSize(), oneRepeateSize, rangeM, tailM);
        for (uint32_t i = 0; i < rangeM; i++) {
            TwoDimBroadCastLastDimNotAlign220<T, isReuseSource>(
                dstLocal[dstLocalOffset], srcLocal[srcLocalOffset], tmpBuffer, oneRepeateSize, blockDim);
            dstLocalOffset += oneRepeateSize * blockDim;
            srcLocalOffset += oneRepeateSize;
        }
        if (tailM != 0) {
            TwoDimBroadCastLastDimNotAlign220<T, isReuseSource>(
                dstLocal[dstLocalOffset], srcLocal[srcLocalOffset], tmpBuffer, tailM, blockDim);
        }
    }
}

template <typename T>
__aicore__ inline void NoBroad(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal, const uint32_t size)
{
    SetVectorMask<T, MaskMode::COUNTER>(size);
    Copy<T, false>(dstLocal, srcLocal, MASK_PLACEHOLDER, 1, {1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();
}

}  // namespace AscendC
#endif