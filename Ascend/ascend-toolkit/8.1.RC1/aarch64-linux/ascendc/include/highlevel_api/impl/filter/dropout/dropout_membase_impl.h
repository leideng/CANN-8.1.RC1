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
 * \file kernel_operator_dropout_mem_base.h
 * \brief
 */
#ifndef IMPL_FILTER_DROPOUT_DROPOUT_MEMBASE_IMPL_H
#define IMPL_FILTER_DROPOUT_DROPOUT_MEMBASE_IMPL_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "kernel_pop_stack_buffer.h"
#include "../../common/check.h"

namespace AscendC {
template <typename T> __aicore__ inline void DropOutBitModeInit(const LocalTensor<T>& sharedTmpBuffer)
{
    ASCENDC_ASSERT((sharedTmpBuffer.GetSize() > 0),
                   { KERNEL_LOG(KERNEL_ERROR, "sharedTmpBuffer.GetSize() must > 0!"); });

    ResetMask();
    LocalTensor<int16_t> stackBuffer = sharedTmpBuffer.template ReinterpretCast<int16_t>();

    // reset cmp mask vsel mode 1, sharedTmpBuffer length need >= 256 Byte
    UnaryRepeatParams unaryParams;
    Muls<int16_t, false>(stackBuffer, stackBuffer, 0, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    SetCmpMask<int16_t>(stackBuffer);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void DropOutBitModeInit()
{
    LocalTensor<int16_t> stackBuffer;
    bool ans = PopStackBuffer<int16_t, TPosition::LCM>(stackBuffer);
    ASCENDC_ASSERT((ans),
                   { KERNEL_LOG(KERNEL_ERROR, "DropOutBitModeInit PopStackBuffer<int16_t, TPosition::LCM> Error!"); });
    DropOutBitModeInit(stackBuffer);
}

template <typename T, bool isInitBitMode = false>
__aicore__ inline void DropOutBitMode(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& maskLocal, const LocalTensor<uint8_t>& sharedTmpBuffer, const T divValue,
    const uint32_t dataSize)
{
    if constexpr (isInitBitMode == false) {
        DropOutBitModeInit(sharedTmpBuffer);
    }

    SetMaskCount();
    SetVectorMask<uint8_t, MaskMode::COUNTER>(0, dataSize);

    const BinaryRepeatParams binaryParams;
    Select<T, uint8_t>(dstLocal, maskLocal, srcLocal, 1, binaryParams);
    PipeBarrier<PIPE_V>();

    const UnaryRepeatParams unaryParams;
    Muls<T, false>(dstLocal, dstLocal, static_cast<T>(divValue), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    SetMaskNorm();
    ResetMask();
}

__aicore__ inline void DropOutByteModeCalc(const LocalTensor<half>& dstLocal, const LocalTensor<half>& srcLocal,
    const LocalTensor<uint8_t>& maskLocal, const half divValue, const DropOutParams<half, float>& params)
{
    const LocalTensor<half>& stackBuffer = params.firstLocal;

    UnaryRepeatParams unaryParams;
    unaryParams.srcRepStride = DEFAULT_REPEAT_STRIDE / sizeof(half);

    Cast<half, uint8_t, false>(stackBuffer, maskLocal, RoundMode::CAST_NONE, MASK_PLACEHOLDER, params.repeatTimes,
        unaryParams);
    PipeBarrier<PIPE_V>();

    const BinaryRepeatParams binaryParams;
    Mul<half, false>(dstLocal, stackBuffer, srcLocal, MASK_PLACEHOLDER, params.repeatTimes, binaryParams);
    PipeBarrier<PIPE_V>();

    unaryParams.srcRepStride = DEFAULT_REPEAT_STRIDE;
    Muls<half, false>(dstLocal, dstLocal, divValue, MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void DropOutByteModeCalc(const LocalTensor<float>& dstLocal, const LocalTensor<float>& srcLocal,
    const LocalTensor<uint8_t>& maskLocal, const float divValue, const DropOutParams<half, float>& params)
{
    const LocalTensor<half>& firstLocal = params.firstLocal;
    const LocalTensor<float>& secondLocal = params.secondLocal;

    UnaryRepeatParams unaryParams;
    unaryParams.srcRepStride = DEFAULT_REPEAT_STRIDE / sizeof(half);

    Cast<half, uint8_t, false>(firstLocal, maskLocal, RoundMode::CAST_NONE, MASK_PLACEHOLDER, params.repeatTimes,
        unaryParams);
    PipeBarrier<PIPE_V>();

    Cast<float, half, false>(secondLocal, firstLocal, RoundMode::CAST_NONE, MASK_PLACEHOLDER, params.repeatTimes,
        unaryParams);
    PipeBarrier<PIPE_V>();

    const BinaryRepeatParams binaryParams;
    Mul<float, false>(dstLocal, secondLocal, srcLocal, MASK_PLACEHOLDER, params.repeatTimes, binaryParams);
    PipeBarrier<PIPE_V>();

    unaryParams.srcRepStride = DEFAULT_REPEAT_STRIDE;
    Muls<float, false>(dstLocal, dstLocal, divValue, MASK_PLACEHOLDER, params.repeatTimes, unaryParams);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void DropOutByteModeSetTmpBuffer(LocalTensor<half>& firstLocal,
    const LocalTensor<uint8_t>& sharedTmpBuffer, DropOutParams<half, float>& params)
{
    firstLocal = sharedTmpBuffer.ReinterpretCast<half>();

    params.stackBufferSize = firstLocal.GetSize();
    params.stackBufferSize = params.stackBufferSize / ONE_BLK_SIZE * ONE_BLK_SIZE;

    params.maxRepeatSize = MAX_REPEAT_HALF_SIZE;
    params.oneRepeatSize = ONE_REPEAT_HALF_SIZE;
}

__aicore__ inline void DropOutByteModeSetTmpBuffer(LocalTensor<half>& firstLocal, LocalTensor<float>& secondLocal,
    const LocalTensor<uint8_t>& sharedTmpBuffer, DropOutParams<half, float>& params)
{
    uint32_t popBufferLen = sharedTmpBuffer.GetSize();
    constexpr uint32_t cutBufLen = sizeof(float) + sizeof(half);
    params.stackBufferSize = popBufferLen / cutBufLen / ONE_BLK_SIZE * ONE_BLK_SIZE;

    firstLocal = sharedTmpBuffer.ReinterpretCast<half>();
    firstLocal.SetSize(params.stackBufferSize);

    secondLocal = sharedTmpBuffer[params.stackBufferSize * sizeof(half)].ReinterpretCast<float>();
    secondLocal.SetSize(params.stackBufferSize);

    params.maxRepeatSize = MAX_REPEAT_FLOAT_SIZE;
    params.oneRepeatSize = ONE_REPEAT_FLOAT_SIZE;
}

template <typename T>
__aicore__ inline void DropOutByteMode(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& maskLocal, const LocalTensor<uint8_t>& sharedTmpBuffer, const T divValue,
    const uint32_t dataSize)
{
    DropOutParams<half, float> params;
    params.dataSize = dataSize;

    if constexpr (sizeof(T) == sizeof(half)) {
        DropOutByteModeSetTmpBuffer(params.firstLocal, sharedTmpBuffer, params);
    } else {
        DropOutByteModeSetTmpBuffer(params.firstLocal, params.secondLocal, sharedTmpBuffer, params);
    }
#if ASCENDC_CPU_DEBUG
    CheckTmpBufferSize(params.stackBufferSize, 0, sharedTmpBuffer.GetSize());
#endif
    const uint32_t round = params.dataSize / params.stackBufferSize;
    const uint32_t tail = params.dataSize % params.stackBufferSize;

    SetMaskCount();
    SetVectorMask<uint8_t, MaskMode::COUNTER>(0, params.stackBufferSize);

    uint32_t offset = 0;
    for (uint32_t i = 0; i < round; i++) {
        DropOutByteModeCalc(dstLocal[offset], srcLocal[offset], maskLocal[offset], divValue, params);
        offset = offset + params.stackBufferSize;
    }

    if (tail != 0) {
        SetVectorMask<uint8_t, MaskMode::COUNTER>(0, tail);
        DropOutByteModeCalc(dstLocal[offset], srcLocal[offset], maskLocal[offset], divValue, params);
    }

    SetMaskNorm();
    ResetMask();
}
} // namespace AscendC
#endif // IMPL_FILTER_DROPOUT_DROPOUT_MEMBASE_IMPL_H
