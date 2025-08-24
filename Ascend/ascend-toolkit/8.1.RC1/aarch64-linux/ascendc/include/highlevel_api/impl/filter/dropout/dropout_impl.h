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
 * \file dropout_impl.h
 * \brief
 */
#ifndef IMPL_FILTER_DROPOUT_DROPOUT_IMPL_H
#define IMPL_FILTER_DROPOUT_DROPOUT_IMPL_H

#if __CCE_AICORE__ <= 200
#include "dropout_m200_impl.h"
#elif __CCE_AICORE__ == 220
#include "dropout_c220_impl.h"
#elif __CCE_AICORE__ == 300
#include "dropout_m300_impl.h"
#endif

namespace AscendC {
#pragma begin_pipe(V)
template <typename T, bool isInitBitMode = false, uint32_t dropOutMode = 0>
__aicore__ inline void DropOutOpt(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& maskLocal, const LocalTensor<uint8_t>& sharedTmpBuffer, const float keepProb,
    const DropOutShapeInfo& info)
{
    float divValue = 1.0;
    divValue = divValue / keepProb;

    const uint32_t dataSize = info.firstAxis * info.srcLastAxis;

    if constexpr (dropOutMode == DROPOUT_MODE_BYTE_MISALIGN) {
        DropOutByteMode(dstLocal, srcLocal, maskLocal, sharedTmpBuffer, static_cast<T>(divValue), info);
    } else if constexpr (dropOutMode == DROPOUT_MODE_BYTE_ALIGN) {
        DropOutByteMode(dstLocal, srcLocal, maskLocal, sharedTmpBuffer, static_cast<T>(divValue), dataSize);
    } else if constexpr (dropOutMode == DROPOUT_MODE_BIT_ALIGN) {
        DropOutBitMode<T, isInitBitMode>(dstLocal, srcLocal, maskLocal, sharedTmpBuffer, static_cast<T>(divValue),
            dataSize);
    } else if constexpr (dropOutMode == DROPOUT_MODE_BIT_MISALIGN) {
        DropOutBitMode<T, isInitBitMode>(dstLocal, srcLocal, maskLocal, sharedTmpBuffer, static_cast<T>(divValue),
            info);
    }
}

template <typename T, bool isInitBitMode = false, uint32_t dropOutMode = 0>
__aicore__ inline void DropOutImpl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& maskLocal, const LocalTensor<uint8_t>& sharedTmpBuffer, const float keepProb,
    const DropOutShapeInfo& info)
{
    TRACE_START(TraceId::DropOut);
    ASCENDC_ASSERT((info.firstAxis > 0), { KERNEL_LOG(KERNEL_ERROR, "info.firstAxis must > 0!"); });
    ASCENDC_ASSERT((info.srcLastAxis > 0), { KERNEL_LOG(KERNEL_ERROR, "info.srcLastAxis must > 0!"); });
    ASCENDC_ASSERT((info.maskLastAxis > 0), { KERNEL_LOG(KERNEL_ERROR, "info.maskLastAxis must > 0!"); });
    ASCENDC_ASSERT((sharedTmpBuffer.GetSize() > 0),
                   { KERNEL_LOG(KERNEL_ERROR, "sharedTmpBuffer.GetSize() must > 0!"); });

    if constexpr (dropOutMode != 0) {
        DropOutOpt<T, isInitBitMode, dropOutMode>(dstLocal, srcLocal, maskLocal, sharedTmpBuffer, keepProb, info);
    } else if (info.srcLastAxis < info.maskLastAxis) {
        DropOutOpt<T, isInitBitMode, DROPOUT_MODE_BYTE_MISALIGN>(dstLocal, srcLocal, maskLocal, sharedTmpBuffer,
            keepProb, info);
    } else if (info.srcLastAxis == info.maskLastAxis) {
        DropOutOpt<T, isInitBitMode, DROPOUT_MODE_BYTE_ALIGN>(dstLocal, srcLocal, maskLocal, sharedTmpBuffer, keepProb,
            info);
    } else if (info.srcLastAxis == (info.maskLastAxis * ONE_BYTE_BIT_SIZE)) {
        DropOutOpt<T, isInitBitMode, DROPOUT_MODE_BIT_ALIGN>(dstLocal, srcLocal, maskLocal, sharedTmpBuffer, keepProb,
            info);
    } else {
        DropOutOpt<T, isInitBitMode, DROPOUT_MODE_BIT_MISALIGN>(dstLocal, srcLocal, maskLocal, sharedTmpBuffer,
            keepProb, info);
    }
    TRACE_STOP(TraceId::DropOut);
}

template <typename T, bool isInitBitMode = false, uint32_t dropOutMode = 0>
__aicore__ inline void DropOutImpl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& maskLocal, const float keepProb, const DropOutShapeInfo& info)
{
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });

    DropOutImpl<T, isInitBitMode, dropOutMode>(dstLocal, srcLocal, maskLocal, sharedTmpBuffer, keepProb, info);
}
#pragma end_pipe
} // namespace AscendC
#endif // IMPL_FILTER_DROPOUT_DROPOUT_IMPL_H
