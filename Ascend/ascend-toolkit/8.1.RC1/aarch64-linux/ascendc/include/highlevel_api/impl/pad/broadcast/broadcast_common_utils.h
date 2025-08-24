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
 * \file #inlcude "broadcast_common_utils.h"
 * \brief
 */
#ifndef IMPL_PAD_BROADCAST_BROADCAST_COMMON_UTILS_H
#define IMPL_PAD_BROADCAST_BROADCAST_COMMON_UTILS_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"

namespace AscendC {
constexpr uint32_t ONE_VOR_BLOCK_DIM = 8;
constexpr uint32_t ELEMENT_NUM_FOR_UINT16 = 16;
constexpr int32_t FLOAT_ELEMENT_NUM = 2;
constexpr uint32_t REPEAT_STRIDE_NUM = 8;
constexpr uint32_t MAX_REPEAT_NUM = 255;

template <typename T, bool isReuseSource = false>
__aicore__ inline void TwoDimBroadCastDimAlign(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const LocalTensor<T> &zeroTemp, const uint32_t firstDim, const uint32_t blockDim)
{
    int32_t dtypeCount = 1;
    if constexpr (sizeof(T) == sizeof(float)) {
        dtypeCount = FLOAT_ELEMENT_NUM;
    }
    uint32_t orCounts = firstDim / ONE_VOR_BLOCK_DIM;
    constexpr uint32_t oneBlockElementNum = ONE_BLK_SIZE / sizeof(T);
    uint8_t repeateTimes = blockDim / oneBlockElementNum;
    SetMaskNorm();
    SetVectorMask<uint16_t, MaskMode::NORMAL>(ONE_VOR_BLOCK_DIM * ELEMENT_NUM_FOR_UINT16);
    uint8_t dstBlkStride = blockDim * dtypeCount / ELEMENT_NUM_FOR_UINT16;
    BinaryRepeatParams binaryParams(dstBlkStride, 0, 0, 1, 1, 0);
    uint32_t transTmpBufferOffset = 0;
    for (uint32_t i = 0; i < orCounts; i++) {
        Or<uint16_t, false>(dstLocal[transTmpBufferOffset].template ReinterpretCast<uint16_t>(),
            srcLocal.template ReinterpretCast<uint16_t>(),
            zeroTemp.template ReinterpretCast<uint16_t>(),
            MASK_PLACEHOLDER,
            repeateTimes,
            binaryParams);
        transTmpBufferOffset += ONE_VOR_BLOCK_DIM * blockDim;
    }
    uint32_t orCountsTail = firstDim - orCounts * ONE_VOR_BLOCK_DIM;
    if (orCountsTail > 0) {
        SetMaskNorm();
        SetVectorMask<uint16_t, MaskMode::NORMAL>(orCountsTail * ELEMENT_NUM_FOR_UINT16);
        Or<uint16_t, false>(dstLocal[transTmpBufferOffset].template ReinterpretCast<uint16_t>(),
                            srcLocal.template ReinterpretCast<uint16_t>(),
                            zeroTemp.template ReinterpretCast<uint16_t>(),
                            MASK_PLACEHOLDER,
                            repeateTimes,
                            binaryParams);
    }
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void LoopBroadCast(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const LocalTensor<T> &zeroTemp, const uint32_t firstDim, const uint32_t blockDim)
{
    int32_t dtypeCount = 1;
    if constexpr (sizeof(T) == sizeof(float)) {
        dtypeCount = FLOAT_ELEMENT_NUM;
    }
    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(firstDim * dtypeCount);
    BinaryRepeatParams binaryParams(1, 1, 0, REPEAT_STRIDE_NUM, REPEAT_STRIDE_NUM, 0);
    uint32_t temBufferOffset = 0;
    for (uint32_t i = 0; i < blockDim; i++) {
        Or<uint16_t, false>(dstLocal[temBufferOffset].template ReinterpretCast<uint16_t>(),
            srcLocal.template ReinterpretCast<uint16_t>(),
            zeroTemp.template ReinterpretCast<uint16_t>(),
            MASK_PLACEHOLDER,
            1,
            binaryParams);
        temBufferOffset += firstDim;
    }
    PipeBarrier<PIPE_V>();
}

}  // namespace AscendC

#endif  // IMPL_PAD_BROADCAST_BROADCAST_COMMON_UTILS_H