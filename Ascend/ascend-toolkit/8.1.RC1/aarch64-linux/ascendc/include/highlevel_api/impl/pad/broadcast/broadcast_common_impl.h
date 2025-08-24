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
 * \file broadcast_common_impl.h
 * \brief
 */
#ifndef IMPL_PAD_BROADCAST_BROADCAST_COMMON_IMPL_H
#define IMPL_PAD_BROADCAST_BROADCAST_COMMON_IMPL_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "broadcast_common_utils.h"
#if __CCE_AICORE__ == 220
#include "broadcast_v220_impl.h"
#else
#include "broadcast_v200_impl.h"
#endif
#if __CCE_AICORE__ >= 200

namespace AscendC {
constexpr uint32_t TWO_DIM = 2;

template <typename T, int32_t dim, int32_t axis, bool isReuseSource = false>
__aicore__ inline void TwoDimBroadCastFirstDim(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const uint32_t dstShape[dim], const uint32_t srcShape[dim], LocalTensor<T> &tmpBuffer)
{
    const uint32_t firstDim = dstShape[0];
    const uint32_t blockDim = dstShape[1];
    ASCENDC_ASSERT(
        (blockDim * sizeof(T) % ONE_BLK_SIZE == 0), { KERNEL_LOG(KERNEL_ERROR, "Non-alignment is not supported!"); });

    constexpr uint32_t oneBlockElementNum = ONE_BLK_SIZE / sizeof(T);
    constexpr uint32_t FIRST_DIM_LOOP_LIMITE = MAX_REPEAT_NUM * oneBlockElementNum;

    auto zeroTemp = tmpBuffer;
    Duplicate(zeroTemp.template ReinterpretCast<uint16_t>(), (uint16_t)0, ONE_BLK_SIZE / sizeof(uint16_t));
    PipeBarrier<PIPE_V>();

    if (blockDim >= FIRST_DIM_LOOP_LIMITE) {
        LoopBroadCast<T>(dstLocal, srcLocal, zeroTemp, blockDim, firstDim);
        return;
    }

    TwoDimBroadCastDimAlign<T, isReuseSource>(dstLocal, srcLocal, zeroTemp, firstDim, blockDim);
}

template <typename T, int32_t dim, int32_t axis, bool isReuseSource = false>
__aicore__ inline void BroadCastCompute(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const uint32_t dstShape[dim], const uint32_t srcShape[dim], LocalTensor<T> &tmpBuffer)
{
    uint32_t srcSize = 1;
    uint32_t dstSize = 1;
    for (uint32_t i = 0; i < dim; i++) {
        srcSize *= srcShape[i];
        dstSize *= dstShape[i];
    }

    if (srcSize == dstSize) {
        NoBroad(dstLocal, srcLocal, dstSize);
    } else if (srcSize == 1) {
        TEventID event1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event1);
        WaitFlag<HardEvent::V_S>(event1);
        auto scalar = srcLocal.GetValue(0);
        TEventID event2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event2);
        WaitFlag<HardEvent::S_V>(event2);
        Duplicate(dstLocal, scalar, dstSize);
        PipeBarrier<PIPE_V>();
    } else {
        if constexpr (dim == TWO_DIM) {
            if constexpr (axis == 1) {  // last broadcast
                TwoDimBroadCastLastDim<T, dim, axis, false>(dstLocal, srcLocal, dstShape, srcShape, tmpBuffer);
            } else {  // first broadcast
                TwoDimBroadCastFirstDim<T, dim, axis, false>(dstLocal, srcLocal, dstShape, srcShape, tmpBuffer);
            }
        } else {
            KERNEL_LOG(KERNEL_ERROR, "Now only support dim = 1 or dim =2, but we get dim= %d", dim);
        }
    }
    SetMaskCount();
}

template <typename T, int32_t dim, int32_t axis, bool isReuseSource = false>
__aicore__ inline void BroadCast(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const uint32_t dstShape[dim], const uint32_t srcShape[dim], LocalTensor<uint8_t> &sharedTmpBuffer)
{
    constexpr uint32_t HALF_ONE_BLK_SIZE = 16;
    if ASCEND_IS_AIC {
        return;
    }
    TRACE_START(TraceId::BroadCast);
    if constexpr (sizeof(T) == 1) {
        LocalTensor<half> tmpBuffer = sharedTmpBuffer.ReinterpretCast<half>();
        uint32_t srcSize = 1;
        uint32_t dstSize = 1;
        for (uint32_t i = 0; i < dim; i++) {
            srcSize *= srcShape[i];
            dstSize *= dstShape[i];
        }
        auto srcTempBuffer = tmpBuffer;
        const uint32_t alignSrcSize =  ((srcSize + HALF_ONE_BLK_SIZE - 1) / HALF_ONE_BLK_SIZE) * HALF_ONE_BLK_SIZE;
        const uint32_t alignDstSize =  ((dstSize + HALF_ONE_BLK_SIZE - 1) / HALF_ONE_BLK_SIZE) * HALF_ONE_BLK_SIZE;
        auto dstTempBuffer = tmpBuffer[alignSrcSize];
        auto tempTempBuffer = dstTempBuffer[alignDstSize];
        SetMaskCount();
        SetVectorMask<T, MaskMode::COUNTER>(srcSize);
        Cast<half, T, false>(srcTempBuffer,
            srcLocal,
            RoundMode::CAST_NONE,
            MASK_PLACEHOLDER,
            1,
            {1, 1, DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE});
        PipeBarrier<PIPE_V>();
        // After BroadCastCompute, Reset to Counter model
        BroadCastCompute<half, dim, axis, isReuseSource>(
            dstTempBuffer, srcTempBuffer, dstShape, srcShape, tempTempBuffer);
        SetVectorMask<T, MaskMode::COUNTER>(dstSize);
        Cast<T, half, false>(dstLocal,
            dstTempBuffer,
            RoundMode::CAST_NONE,
            MASK_PLACEHOLDER,
            1,
            {1, 1, HALF_DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
        ResetMask();
    } else {
        LocalTensor<T> tmpBuffer = sharedTmpBuffer.ReinterpretCast<T>();
        SetMaskCount();
        BroadCastCompute<T, dim, axis, isReuseSource>(dstLocal, srcLocal, dstShape, srcShape, tmpBuffer);
        SetMaskNorm();
        ResetMask();
    }
    TRACE_STOP(TraceId::BroadCast);
}

template <typename T, int32_t dim, int32_t axis, bool isReuseSource = false>
__aicore__ inline void BroadCast(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const uint32_t dstShape[dim], const uint32_t srcShape[dim])
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    BroadCast<T, dim, axis, isReuseSource>(dstLocal, srcLocal, dstShape, srcShape, sharedTmpBuffer);
}
}  // namespace AscendC

#endif

#endif  // ASCENDC_MODULE_OPERATOR_BROADCAST_INTERFACE_H