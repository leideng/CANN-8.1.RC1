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
 * \file acosh.h
 * \brief
 */

#ifndef LIB_MATH_ACOSH_H
#define LIB_MATH_ACOSH_H
#include "kernel_tensor.h"
#include "../../impl/math/acosh/acosh_common_impl.h"
#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 220

namespace AscendC {
#pragma begin_pipe(V)
/* !
 * \brief Returns a new tensor with the inverse hyperbolic cosine of the elements of input.
 * https://pytorch.org/docs/stable/generated/torch.acosh.html#torch.acosh.
 *
 * \note support data type: half and float
 *
 * \param [out] dstTensor, output LocalTensor
 * \param [in] srcTensor, input LocalTensor
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Acosh(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    AcoshImpl<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer);
}
/* !
 * \note support data type: half and float
 *
 * \param [out] dstTensor, output LocalTensor
 * \param [in] srcTensor, input LocalTensor
 * \param [in] calCount, amount of data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Acosh(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    AcoshImpl<T, isReuseSource>(dstTensor, srcTensor, calCount);
}
/* !
 * \note support data type: half and float
 *
 * \param [out] dstTensor, output LocalTensor
 * \param [in] srcTensor, input LocalTensor
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Acosh(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    AcoshImpl<T, isReuseSource>(dstTensor, srcTensor);
}
/* !
 * \note support data type: half and float
 *
 * \param [out] dstTensor, output LocalTensor
 * \param [in] srcTensor, input LocalTensor
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 * \param [in] calCount, amount of data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Acosh(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    AcoshImpl<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, calCount);
}
#pragma end_pipe
}
#endif
#endif