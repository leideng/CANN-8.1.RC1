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
 * \file pad_common_impl.h
 * \brief
 */
#ifndef IMPL_PAD_PAD_PAD_COMMON_IMPL_H
#define IMPL_PAD_PAD_PAD_COMMON_IMPL_H

#if __CCE_AICORE__ <= 200
#include "pad_v200_impl.h"
#elif __CCE_AICORE__ == 220
#include "pad_v220_impl.h"
#endif

namespace AscendC {
template <typename T>
__aicore__ inline void PadImpl(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, PadParams &padParams,
    const LocalTensor<uint8_t> &sharedTmpBuffer, PadTiling &tiling)
{
    PadCompute<T>(dstTensor, srcTensor, padParams, sharedTmpBuffer, tiling);
}

template <typename T>
__aicore__ inline void UnPadImpl(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
    UnPadParams &unPadParams, LocalTensor<uint8_t> &sharedTmpBuffer, UnPadTiling &tiling)
{
    ASCENDC_ASSERT(((tiling.srcWidth) * sizeof(T) % ONE_BLK_SIZE == 0),
                   { KERNEL_LOG(KERNEL_ERROR, "width is %u, which should be 32B aligned.", tiling.srcWidth); });

    // dst width 32B aligned make no sense, nothing to do
    // dst width not 32B aligned
    ASCENDC_ASSERT(((tiling.srcWidth - unPadParams.rightPad) * sizeof(T) % ONE_BLK_SIZE != 0), {
        KERNEL_LOG(KERNEL_ERROR, "width is %u, rightPad is %u, width - rightPad should not be 32B aligned.",
            tiling.srcWidth, unPadParams.rightPad);
    });
    UnPadCompute<T>(dstTensor, srcTensor, unPadParams, sharedTmpBuffer, tiling);
}
} // namespace AscendC
#endif // IMPL_PAD_PAD_PAD_COMMON_IMPL_H
