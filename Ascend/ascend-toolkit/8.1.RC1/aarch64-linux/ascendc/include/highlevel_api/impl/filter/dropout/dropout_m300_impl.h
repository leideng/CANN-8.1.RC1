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
 * \file dropout_m300_impl.h
 * \brief
 */
#ifndef IMPL_FILTER_DROPOUT_DROPOUT_M300_IMPL_H
#define IMPL_FILTER_DROPOUT_DROPOUT_M300_IMPL_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "kernel_pop_stack_buffer.h"

namespace AscendC {
template <typename T, bool isInitBitMode = false>
__aicore__ inline void DropOutBitMode(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& maskLocal, const LocalTensor<uint8_t>& sharedTmpBuffer, const T divValue,
    const uint32_t dataSize)
{
    (void)sharedTmpBuffer;
    Select<T, uint8_t>(dstLocal, maskLocal, srcLocal, (T)0, SELMODE::VSEL_TENSOR_SCALAR_MODE, dataSize);
    Muls<T>(dstLocal, dstLocal, static_cast<T>(divValue), dataSize);
}

template <typename T, bool isInitBitMode = false>
__aicore__ inline void DropOutBitMode(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& maskLocal, const LocalTensor<uint8_t>& sharedTmpBuffer, const T divValue,
    const DropOutShapeInfo& info)
{
    (void)sharedTmpBuffer;
    GatherMaskParams reducev2Params;
    reducev2Params.repeatTimes = info.firstAxis;
    reducev2Params.src0RepeatStride = info.maskLastAxis / ONE_BLK_SIZE;
    reducev2Params.src1RepeatStride = 0;
    uint64_t rsvdCnt = 0;
    int32_t validCount = info.srcLastAxis / ONE_BYTE_BIT_SIZE / sizeof(uint16_t);

    LocalTensor<uint16_t> maskTmpLocal = maskLocal.ReinterpretCast<uint16_t>();
    // Src1 is Placeholder
    GatherMask<uint16_t>(maskTmpLocal, maskTmpLocal, REDUCEV2_MODE_SEVEN, true, validCount, reducev2Params, rsvdCnt);

    DropOutBitMode<T, true>(dstLocal, srcLocal, maskLocal, sharedTmpBuffer, divValue,
        info.firstAxis * info.srcLastAxis);
}

__aicore__ inline void DropOutByteModeCalc(const LocalTensor<half>& dstLocal, const LocalTensor<half>& srcLocal,
    const LocalTensor<uint8_t>& maskLocal, const half divValue, const DropOutParams<half, float>& params)
{
    const LocalTensor<half>& stackBuffer = params.firstLocal;
    Cast<half, uint8_t>(stackBuffer, maskLocal, RoundMode::CAST_NONE, params.currentSize);
    Mul<half>(dstLocal, stackBuffer, srcLocal, params.currentSize);
    Muls<half>(dstLocal, dstLocal, divValue, params.currentSize);
}

__aicore__ inline void DropOutByteModeCalc(const LocalTensor<float>& dstLocal, const LocalTensor<float>& srcLocal,
    const LocalTensor<uint8_t>& maskLocal, const float divValue, const DropOutParams<half, float>& params)
{
    const LocalTensor<half>& firstLocal = params.firstLocal;
    const LocalTensor<float>& secondLocal = params.secondLocal;

    Cast<half, uint8_t>(firstLocal, maskLocal, RoundMode::CAST_NONE, params.currentSize);
    Cast<float, half>(secondLocal, firstLocal, RoundMode::CAST_NONE, params.currentSize);
    Mul<float>(dstLocal, secondLocal, srcLocal, params.currentSize);
    Muls<float>(dstLocal, dstLocal, divValue, params.currentSize);
}

__aicore__ inline void DropOutByteModeSetTmpBuffer(LocalTensor<half>& firstLocal,
    const LocalTensor<uint8_t>& sharedTmpBuffer, DropOutParams<half, float>& params)
{
    firstLocal = sharedTmpBuffer.ReinterpretCast<half>();

    params.stackBufferSize = firstLocal.GetSize();
    params.stackBufferSize = params.stackBufferSize / ONE_BLK_SIZE * ONE_BLK_SIZE;

    firstLocal.SetSize(params.stackBufferSize);

    params.maxRepeatSize = MAX_REPEAT_HALF_SIZE;
    params.oneRepeatSize = ONE_REPEAT_HALF_SIZE;
}

__aicore__ inline void DropOutByteModeSetTmpBuffer(LocalTensor<half>& firstLocal,
    LocalTensor<float>& secondLocal, const LocalTensor<uint8_t>& sharedTmpBuffer, DropOutParams<half, float>& params)
{
    uint32_t popBufferLen = sharedTmpBuffer.GetSize();
    constexpr uint32_t cutBufLen = sizeof(float) + sizeof(half);
    params.stackBufferSize = popBufferLen / cutBufLen / ONE_BLK_SIZE * ONE_BLK_SIZE;

    firstLocal = sharedTmpBuffer.ReinterpretCast<half>();
    firstLocal.SetSize(params.stackBufferSize);

    secondLocal = sharedTmpBuffer[params.stackBufferSize * sizeof(half)].ReinterpretCast<float>();
    secondLocal.SetSize(params.stackBufferSize);

    params.oneRepeatSize = ONE_REPEAT_FLOAT_SIZE;
    params.maxRepeatSize = MAX_REPEAT_FLOAT_SIZE;
}

template <typename T>
__aicore__ inline void DropOutByteMode(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& maskLocal, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const T divValue, const uint32_t dataSize)
{
    DropOutParams<half, float> params;
    params.dataSize = dataSize;

    if constexpr (sizeof(T) == sizeof(half)) {
        DropOutByteModeSetTmpBuffer(params.firstLocal, sharedTmpBuffer, params);
    } else {
        DropOutByteModeSetTmpBuffer(params.firstLocal, params.secondLocal, sharedTmpBuffer, params);
    }

    const uint32_t round = params.dataSize / params.stackBufferSize;
    const uint32_t tail = params.dataSize % params.stackBufferSize;

    uint32_t offset = 0;
    params.currentSize = params.stackBufferSize;
    for (uint32_t i = 0; i < round; i++) {
        DropOutByteModeCalc(dstLocal[offset], srcLocal[offset], maskLocal[offset], divValue, params);
        offset = offset + params.stackBufferSize;
    }

    if (tail != 0) {
        params.currentSize = tail;
        DropOutByteModeCalc(dstLocal[offset], srcLocal[offset], maskLocal[offset], divValue, params);
    }
}

template <typename T>
__aicore__ inline void DropOutByteMode(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& maskLocal, const LocalTensor<uint8_t>& sharedTmpBuffer, const T divValue,
    const DropOutShapeInfo& info)
{
    GatherMaskParams reducev2Params;
    reducev2Params.repeatTimes = info.firstAxis;
    reducev2Params.src0RepeatStride = info.maskLastAxis / ONE_BLK_SIZE;
    reducev2Params.src1RepeatStride = 0;
    uint64_t rsvdCnt = 0;
    uint32_t validCount = info.srcLastAxis / sizeof(uint16_t);

    LocalTensor<uint16_t> maskTmpLocal = maskLocal.ReinterpretCast<uint16_t>();
    // Src1 is Not Enabled
    GatherMask<uint16_t>(maskTmpLocal, maskTmpLocal, REDUCEV2_MODE_SEVEN, true, validCount, reducev2Params, rsvdCnt);

    DropOutByteMode(dstLocal, srcLocal, maskLocal, sharedTmpBuffer, divValue, info.firstAxis * info.srcLastAxis);
}
} // namespace AscendC
#endif // IMPL_FILTER_DROPOUT_DROPOUT_M300_IMPL_H
