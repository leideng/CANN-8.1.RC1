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
 * \file batchnorm.h
 * \brief
 */
#ifndef LIB_NORMALIZATION_BATCHNORM_H
#define LIB_NORMALIZATION_BATCHNORM_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../impl/normalization/batchnorm/batchnorm_common_impl.h"
#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 220
namespace AscendC {
#pragma begin_pipe(V)
/* **************************************************************************************************
 * BatchNorm                                             *
 * ************************************************************************************************** */
/*
 * @ingroup BatchNorm
 * @brief compute output = gamm * (x - outputMean) * rsqrt(outputVariance + epsilon) + beta
 * @brief compute outputMean = sum(x) / batch
 * @brief compute outputVariance = sqrt(x - outputMean) / batch
 * @param [out] output output LocalTensor
 * @param [out] outputMean output LocalTensor
 * @param [out] outputVariance output  LocalTensor
 * @param [in] inputX input LocalTensor
 * @param [in] gamm input LocalTensor
 * @param [in] beta input LocalTensor
 * @param [in] sharedTmpBuffer input local temporary Tensor
 * @param [in] epsilon
 * @param [in] tiling batchnormtiling
 */
template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void BatchNorm(const LocalTensor<T>& output, const LocalTensor<T>& outputMean,
    const LocalTensor<T>& outputVariance, const LocalTensor<T>& inputX, const LocalTensor<T>& gamm,
    const LocalTensor<T>& beta, const LocalTensor<uint8_t>& sharedTmpBuffer, const T epsilon, BatchNormTiling& tiling)
{
    if ASCEND_IS_AIC {
        return;
    }
    BatchNormImpl<T, isReuseSource, isBasicBlock>(output, outputMean, outputVariance, inputX, gamm, beta,
        sharedTmpBuffer, epsilon, tiling);
}

template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void BatchNorm(const LocalTensor<T>& output, const LocalTensor<T>& outputMean,
    const LocalTensor<T>& outputVariance, const LocalTensor<T>& inputX, const LocalTensor<T>& gamm,
    const LocalTensor<T>& beta, const T epsilon, BatchNormTiling& tiling)
{
    if ASCEND_IS_AIC {
        return;
    }
    BatchNormImpl<T, isReuseSource, isBasicBlock>(output, outputMean, outputVariance, inputX, gamm, beta, epsilon,
        tiling);
}
#pragma end_pipe
} // namespace AscendC
#endif
#endif // LIB_NORMALIZATION_BATCHNORM_H
