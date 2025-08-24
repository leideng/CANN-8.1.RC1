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
 * \file round_common_impl.h
 * \brief
 */
#ifndef IMPL_MATH_ROUND_ROUND_COMMON_IMPL_H
#define IMPL_MATH_ROUND_ROUND_COMMON_IMPL_H
#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../common/check.h"

#if __CCE_AICORE__ == 220
#include "round_v220_impl.h"
#elif __CCE_AICORE__ == 200
#include "round_v200_impl.h"
#endif

#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 220
namespace AscendC {

template <typename T, bool isReuseSource = false>
__aicore__ inline void RoundImpl(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t calCount)
{
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");

    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Round");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Round");

    ASCENDC_ASSERT((std::is_same<T, float>::value || std::is_same<T, half>::value), {
        KERNEL_LOG( KERNEL_ERROR, "Failed to check the data types, current api support data types are half/float."); });
    SetMaskCount();
    if constexpr (sizeof(T) == sizeof(half)) {
        RoundComputeCount<half, false>(dstTensor, srcTensor, sharedTmpBuffer, calCount);
    } else {
        RoundComputeCount<float, false>(dstTensor, srcTensor, sharedTmpBuffer, calCount);
    }
    SetMaskNorm();
    ResetMask();
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void RoundImpl(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
    const uint32_t calCount)
{
    // Using the stack space to allocate tmpbuf
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ret = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ret), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    RoundImpl<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, calCount);
}
} // namespace AscendC
#endif

#endif // IMPL_MATH_ROUND_ROUND_COMMON_IMPL_H