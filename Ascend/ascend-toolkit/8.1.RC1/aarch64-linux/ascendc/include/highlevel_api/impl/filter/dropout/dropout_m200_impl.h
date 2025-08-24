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
 * \file dropout_m200_impl.h
 * \brief
 */
#ifndef IMPL_FILTER_DROPOUT_DROPOUT_M200_IMPL_H
#define IMPL_FILTER_DROPOUT_DROPOUT_M200_IMPL_H

#include "dropout_membase_impl.h"

namespace AscendC {
template <typename T, bool isInitBitMode = false>
__aicore__ inline void DropOutBitMode(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& maskLocal, const LocalTensor<uint8_t>& sharedTmpBuffer, const T divValue,
    const DropOutShapeInfo& info)
{
    if constexpr (isInitBitMode == false) {
        DropOutBitModeInit(sharedTmpBuffer);
    }

    SetMaskCount();
    SetVectorMask<uint8_t, MaskMode::COUNTER>(0, info.srcLastAxis);

    const BinaryRepeatParams binaryParams;
    const UnaryRepeatParams unaryParams;

    for (uint32_t i = 0; i < info.firstAxis; i++) {
        Select<T, uint8_t>(dstLocal[i * info.srcLastAxis], maskLocal[i * info.maskLastAxis],
            srcLocal[i * info.srcLastAxis], 1, binaryParams);
        PipeBarrier<PIPE_V>();
    }

    SetVectorMask<uint8_t, MaskMode::COUNTER>(0, info.firstAxis * info.srcLastAxis);
    Muls<T, false>(dstLocal, dstLocal, static_cast<T>(divValue), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    SetMaskNorm();
    ResetMask();
}

template <typename T>
__aicore__ inline void DropOutByteMode(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& maskLocal, const LocalTensor<uint8_t>& sharedTmpBuffer, const T divValue,
    const DropOutShapeInfo& info)
{
    for (uint32_t i = 0; i < info.firstAxis; i++) {
        DropOutByteMode(dstLocal[i * info.srcLastAxis], srcLocal[i * info.srcLastAxis],
            maskLocal[i * info.maskLastAxis], sharedTmpBuffer, divValue, info.srcLastAxis);
    }
}
} // namespace AscendC
#endif // IMPL_FILTER_DROPOUT_DROPOUT_M200_IMPL_H
