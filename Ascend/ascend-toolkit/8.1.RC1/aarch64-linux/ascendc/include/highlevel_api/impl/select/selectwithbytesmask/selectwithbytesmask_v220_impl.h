/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef IMPL_SELECT_SELECT_WITH_BYTES_MASK_V220_IMPL_H
#define IMPL_SELECT_SELECT_WITH_BYTES_MASK_V220_IMPL_H
#include "kernel_tensor.h"
#include "kernel_utils.h"
#include "selectwithbytesmask_common_impl.h"

namespace AscendC {
__aicore__ inline uint32_t ComputeMaskExtraBufSize(const uint32_t srcSize, const uint32_t typeSize)
{
    return AlignUp(srcSize * typeSize, ONE_BLK_SIZE);
}

template <typename T, typename U, bool reverse = false>
__aicore__ inline void SelectWithBytesMaskProcess(const LocalTensor<T> &dst, const LocalTensor<T> &src0, T src1,
    const LocalTensor<U> &mask, const LocalTensor<U> &tmpMask, const LocalTensor<uint8_t> &sharedTmpBuffer,
    const SelectWithBytesMaskShapeInfo &info, const uint32_t tmpBufferOffset, const uint32_t loopSize)
{
    if (info.srcLastAxis != info.maskLastAxis) {
        RemoveRedundantMask(tmpMask, mask, sharedTmpBuffer, info);
        PipeBarrier<PIPE_V>();
    }

    SelectWithBytesMaskLoopImpl<T, U, reverse>(dst, src0, src1, tmpMask, sharedTmpBuffer[tmpBufferOffset], loopSize,
        src0.GetSize(), 0, 0);
}
} // namespace AscendC
#endif // IMPL_SELECT_SELECT_WITH_BYTES_MASK_V220_IMPL_H
