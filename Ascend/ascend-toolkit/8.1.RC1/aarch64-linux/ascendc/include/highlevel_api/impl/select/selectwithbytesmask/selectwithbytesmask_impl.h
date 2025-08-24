/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef IMPL_SELECT_SELECT_WITH_BYTES_MASK_IMPL_H
#define IMPL_SELECT_SELECT_WITH_BYTES_MASK_IMPL_H

#include "kernel_tensor.h"
#include "kernel_utils.h"
#include "selectwithbytesmask_common_impl.h"
#if __CCE_AICORE__ == 220
#include "selectwithbytesmask_v220_impl.h"
#elif __CCE_AICORE__ == 200
#include "selectwithbytesmask_v200_impl.h"
#endif

namespace AscendC {
// Selects Values from two sources and put into dst according to the mask values.
// True: Select scalar, False: select src.
template <typename T, typename U, bool isReuseMask, bool reverse = false>
__aicore__ inline __inout_pipe__(V) void SelectWithBytesMaskImpl(const LocalTensor<T> &dst, const LocalTensor<T> &src0,
    T src1, const LocalTensor<U> &mask, const LocalTensor<uint8_t> &sharedTmpBuffer,
    const SelectWithBytesMaskShapeInfo &info)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    PipeBarrier<PIPE_V>();
    constexpr uint32_t MIN_REQUIRED_BUFFER = 1024;
    constexpr uint32_t RESERVED_BUFFER = 256;
    constexpr uint32_t MAX_CALC_BYTE_PER_LOOP = 255 * ONE_REPEAT_BYTE_SIZE;
    const uint32_t firstAxis = info.firstAxis;
    const uint32_t srcLastAxis = info.srcLastAxis;
    const uint32_t maskLastAxis = info.maskLastAxis;
    const uint32_t srcSize = src0.GetSize();

    ASCENDC_ASSERT((srcSize == firstAxis * srcLastAxis),
                   { KERNEL_LOG(KERNEL_ERROR, "ShapeInfo must be match with src Tensor size."); });
    ASCENDC_ASSERT((mask.GetSize() == firstAxis * maskLastAxis),
                   { KERNEL_LOG(KERNEL_ERROR, "ShapeInfo must be match with mask Tensor size."); });
    ASCENDC_ASSERT((maskLastAxis >= srcLastAxis),
                   { KERNEL_LOG(KERNEL_ERROR, "maskLastAxis must be greater than or equal to srcLastAxis."); });
    uint32_t bufferSize = sharedTmpBuffer.GetSize();
    ASCENDC_ASSERT((bufferSize >= MIN_REQUIRED_BUFFER), { KERNEL_LOG(KERNEL_ERROR, "bufferSize must >= 1024B!"); });
    LocalTensor<U> tmpMask = mask;
    LocalTensor<T> tmpTensor = sharedTmpBuffer.ReinterpretCast<T>();
    uint32_t tmpBufferOffset = 0;
    if constexpr (!isReuseMask) {
        if (srcLastAxis != maskLastAxis) {
            const uint32_t tmpMaskRequiredBuffer = ComputeMaskExtraBufSize(srcSize, sizeof(U));
            ASCENDC_ASSERT((bufferSize >= MIN_REQUIRED_BUFFER + tmpMaskRequiredBuffer), {
                KERNEL_LOG(KERNEL_ERROR, "unalign axis and do not reuse source must provide %d buffer",
                    MIN_REQUIRED_BUFFER + tmpMaskRequiredBuffer);
            });
            tmpMask = sharedTmpBuffer.template ReinterpretCast<U>();
            tmpMask.SetSize(tmpMaskRequiredBuffer / sizeof(U));
            bufferSize -= tmpMaskRequiredBuffer;
            tmpBufferOffset = tmpMaskRequiredBuffer;
        }
    }
    // Remove tmp buffer, which is reserved for tail part computation.
    bufferSize -= RESERVED_BUFFER;
    uint32_t loopSize = bufferSize / sizeof(half) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    if (loopSize > MAX_CALC_BYTE_PER_LOOP / sizeof(half)) {
        loopSize = MAX_CALC_BYTE_PER_LOOP / sizeof(half);
    }

    SetMaskCount();
    InitScalarSelectMask(tmpTensor, src1);

    SelectWithBytesMaskProcess<T, U, reverse>(dst, src0, src1, mask, tmpMask, sharedTmpBuffer, info, tmpBufferOffset,
        loopSize);
    SetMaskNorm();
    ResetMask();
}
} // namespace AscendC
#endif // IMPL_SELECT_SELECT_WITH_BYTES_MASK_IMPL_H
