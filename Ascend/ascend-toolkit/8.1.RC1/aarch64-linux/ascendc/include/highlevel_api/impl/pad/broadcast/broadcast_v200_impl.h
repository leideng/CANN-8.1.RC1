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
#ifndef IMPL_PAD_BROADCAST_BROADCAST_V200_IMPL_H
#define IMPL_PAD_BROADCAST_BROADCAST_V200_IMPL_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "broadcast_common_utils.h"

namespace AscendC {
template <typename T>
__aicore__ inline void GetAlignLoopNumbers200(const uint32_t firstDim, const uint32_t blockDim, uint32_t tmpBufferSize,
    uint32_t &oneRepeateSize, uint32_t &rangeM, uint32_t &tailM)
{
    constexpr uint32_t oneBlockElementNum = ONE_BLK_SIZE / sizeof(T);
    tmpBufferSize -= oneBlockElementNum;
    ASCENDC_ASSERT(
        (tmpBufferSize > 0), { KERNEL_LOG(KERNEL_ERROR, "tmpBufferSize should bigger than oneBlockElementNum!"); });
    const uint32_t minTmpBufferSize = oneBlockElementNum * ((blockDim + ONE_VOR_BLOCK_DIM - 1) / ONE_VOR_BLOCK_DIM);
    ASCENDC_ASSERT((tmpBufferSize > minTmpBufferSize), {
        KERNEL_LOG(
            KERNEL_ERROR, "tmpBufferSize %u should bigger than minTmpBufferSize %u!", tmpBufferSize, minTmpBufferSize);
    });
    oneRepeateSize = tmpBufferSize / minTmpBufferSize * oneBlockElementNum;
    rangeM = firstDim / oneRepeateSize;
    tailM = firstDim - oneRepeateSize * rangeM;
}

template <typename T>
__aicore__ inline void BroadCastTranse(
    const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal, const uint32_t firstDim, const uint32_t blockDim)
{
    if constexpr (sizeof(T) == sizeof(float)) {
        v4dtrans((__ubuf__ uint32_t *)(dstLocal.GetPhyAddr()),
            (__ubuf__ uint32_t *)(srcLocal.GetPhyAddr()),
            (uint16_t)firstDim,
            (uint16_t)blockDim,
            false);
    } else {
        v4dtrans((__ubuf__ uint16_t *)(dstLocal.GetPhyAddr()),
            (__ubuf__ uint16_t *)(srcLocal.GetPhyAddr()),
            (uint16_t)firstDim,
            (uint16_t)blockDim,
            false);
    }
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void TwoDimBroadCastLastDimAlign200(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const LocalTensor<T> &zeroTemp, const LocalTensor<T> &tmpBuffer, const uint32_t firstDim, const uint32_t blockDim)
{
    TwoDimBroadCastDimAlign<T, isReuseSource>(tmpBuffer, srcLocal, zeroTemp, blockDim, firstDim);
    BroadCastTranse<T>(dstLocal, tmpBuffer, firstDim, blockDim);
    PipeBarrier<PIPE_V>();
}

template <typename T, int32_t dim, int32_t axis, bool isReuseSource = false>
__aicore__ inline void TwoDimBroadCastLastDim(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const uint32_t dstShape[dim], const uint32_t srcShape[dim], LocalTensor<T> &tmpBuffer)
{
    const auto firstDim = dstShape[0];
    const auto blockDim = dstShape[axis];
    constexpr uint32_t oneBlockElementNum = ONE_BLK_SIZE / sizeof(T);
    constexpr uint32_t FIRST_DIM_LOOP_LIMITE = MAX_REPEAT_NUM * oneBlockElementNum;

    auto zeroTemp = tmpBuffer;
    const uint32_t blockSize = ONE_BLK_SIZE / sizeof(T);
    Duplicate(zeroTemp.template ReinterpretCast<uint16_t>(), (uint16_t)0, ONE_BLK_SIZE / sizeof(uint16_t));
    PipeBarrier<PIPE_V>();

    if (firstDim >= FIRST_DIM_LOOP_LIMITE) {
        LoopBroadCast<T>(tmpBuffer[blockSize], srcLocal, zeroTemp, firstDim, blockDim);
        BroadCastTranse<T>(dstLocal, tmpBuffer[blockSize], firstDim, blockDim);
        PipeBarrier<PIPE_V>();
        return;
    }

    if (firstDim * sizeof(T) % ONE_BLK_SIZE == 0) {
        uint32_t oneRepeateSize = 0;
        uint32_t rangeM = 0;
        uint32_t tailM = 0;
        uint32_t dstLocalOffset = 0;
        uint32_t srcLocalOffset = 0;
        GetAlignLoopNumbers200<T>(firstDim, blockDim, tmpBuffer.GetSize(), oneRepeateSize, rangeM, tailM);
        for (uint32_t i = 0; i < rangeM; i++) {
            TwoDimBroadCastLastDimAlign200<T, isReuseSource>(dstLocal[dstLocalOffset],
                srcLocal[srcLocalOffset],
                zeroTemp,
                tmpBuffer[blockSize],
                oneRepeateSize,
                blockDim);
            dstLocalOffset += oneRepeateSize * blockDim;
            srcLocalOffset += oneRepeateSize;
        }

        if (tailM != 0) {
            TwoDimBroadCastLastDimAlign200<T, isReuseSource>(
                dstLocal[dstLocalOffset], srcLocal[srcLocalOffset], zeroTemp, tmpBuffer[blockSize], tailM, blockDim);
        }
    } else {
        KERNEL_LOG(KERNEL_ERROR, "Non-alignment is not supported.");
    }
}

template <typename T>
__aicore__ inline void NoBroad(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal, const uint32_t size)
{
    SetVectorMask<T, MaskMode::COUNTER>(size);
    DataCopy<T>(dstLocal, srcLocal, size);
    PipeBarrier<PIPE_V>();
}

}  // namespace AscendC

#endif