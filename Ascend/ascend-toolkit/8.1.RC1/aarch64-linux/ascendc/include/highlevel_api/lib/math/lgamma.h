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
 * \file lgamma.h
 * \brief
 */
#ifndef LIB_MATH_LGAMMA_H
#define LIB_MATH_LGAMMA_H
#include "kernel_tensor.h"
#include "../../impl/math/lgamma/lgamma_common_impl.h"
#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 220

namespace AscendC {
#pragma begin_pipe(V)
/*
 * @brief This function Computes the natural logarithm of the absolute value of the gamma function on input (e.g.
 * lgamma(1) is 0)
 * @ingroup Lgamma
 * @param [in] srcTensor, input LocalTensor
 * @param [in] sharedTmpBuffer, input local temporary Tensor
 * @param [in] calCount, amount of data to be calculated
 * @param [out] dstTensor, output LocalTensor
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Lgamma(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    LgammaCompute<isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, calCount);
}

/*
 * @brief This function Computes the natural logarithm of the absolute value of the gamma function on input (e.g.
 * lgamma(1) is 0)
 * @ingroup Lgamma
 * @param [in] srcTensor, input LocalTensor
 * @param [in] calCount, amount of data to be calculated
 * @param [out] dstTensor, output LocalTensor
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Lgamma(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    LocalTensor<uint8_t> tmp;
    const bool ret = PopStackBuffer<uint8_t, TPosition::LCM>(tmp);
    ASCENDC_ASSERT((ret), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    LgammaCompute<isReuseSource>(dstTensor, srcTensor, tmp, calCount);
}
#pragma end_pipe
}  // namespace AscendC
#endif
#endif  // LIB_MATH_LGAMMA_INTERFACE_H
