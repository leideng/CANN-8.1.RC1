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
 * \file clamp.h
 * \brief
 */
#ifndef LIB_MATH_CLAMP_H
#define LIB_MATH_CLAMP_H
#include "kernel_tensor.h"
#include "../../impl/math/clamp/clamp_common_impl.h"
#include "kernel_pop_stack_buffer.h"
 
#if __CCE_AICORE__ == 220 || __CCE_AICORE__ == 200
namespace AscendC {
/* !
 * \brief This function implements replaces numbers greater than scalar with scalar
 * (e.g. ClampMax(2) means to replace numbers greater than 2 with 2 ). For details about the interface description, see
 * https://pytorch.org/docs/stable/generated/torch.clamp.html.
 *
 * \note support data type: half and float
 *
 * \param [out] dstTensor, output LocalTensor
 * \param [in] srcTensor, input LocalTensor
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 * \param [in] calCount, amount of data to be calculated
 */

/* **************************************************************************************************
 * ClampMax                                           *
 * ************************************************************************************************* */
#pragma begin_pipe(V)
template <typename T, bool isReuseSource = false>
__aicore__ inline void ClampMax(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const T scalar, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    ClampMaxImpl<T, false>(dstTensor, srcTensor, sharedTmpBuffer, scalar, calCount);
}
 
template <typename T, bool isReuseSource = false>
__aicore__ inline void ClampMax(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const T scalar,
    const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    // Using the stack space to allocate tmpbuf
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ret = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ret), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    ClampMaxImpl<T, false>(dstTensor, srcTensor, sharedTmpBuffer, scalar, calCount);
}

/* !
 * \brief This function implements replace numbers less than scalar with scalar
 * (e.g. ClampMin(2) means to replace numbers less than 2 with 2 ). For details about the interface description, see
 * https://pytorch.org/docs/stable/generated/torch.clamp.html.
 *
 * \note support data type: half and float
 *
 * \param [out] dstTensor, output LocalTensor
 * \param [in] srcTensor, input LocalTensor
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 * \param [in] calCount, amount of data to be calculated
 */
/* **************************************************************************************************
 * ClampMin                                           *
 * ************************************************************************************************* */
 
template <typename T, bool isReuseSource = false>
__aicore__ inline void ClampMin(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const T scalar, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    ClampMinImpl<T, false>(dstTensor, srcTensor, sharedTmpBuffer, scalar, calCount);
}
 
template <typename T, bool isReuseSource = false>
__aicore__ inline void ClampMin(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const T scalar,
    const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    // Using the stack space to allocate tmpbuf
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ret = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ret), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    ClampMinImpl<T, false>(dstTensor, srcTensor, sharedTmpBuffer, scalar, calCount);
}
#pragma end_pipe
} // namespace AscendC
#endif
#endif // LIB_MATH_CLAMP_H