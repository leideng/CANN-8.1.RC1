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
 * \file reglu.h
 * \brief
 */
#ifndef LIB_REGLU_REGLU_H
#define LIB_REGLU_REGLU_H
#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 220

#include "kernel_tensor.h"
#include "../../impl/activation/reglu/reglu_common_impl.h"
namespace AscendC {
#pragma begin_pipe(V)
/*
 * @brief ReGLU is an activation function which is a variant of GLU(e.g.ReGlu(1, 1) is 1).
 * Mathematical formulas: ReGlu(x1, x2) = x1 * max(0, x2)
 * @ingroup ReGlu
 * @param [out] dstTensor, output LocalTensor
 * @param [in] srcTensor, input LocalTensor
 * @param [in] sharedTmpBuffer, input local temporary Tensor
 * @param [in] calCount, amount of input data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void ReGlu(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0,
    const LocalTensor<T>& srcTensor1, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    if (g_coreType == AIC) {
        return;
    }
    ReGluImpl<T, false>(dstTensor, srcTensor0, srcTensor1, sharedTmpBuffer, calCount);
}

/*
 * @ingroup ReGlu
 * @param [out] dstTensor, output LocalTensor
 * @param [in] srcTensor, input local temporary Tensor
 * @param [in] calCount, amount of input data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void ReGlu(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0,
    const LocalTensor<T>& srcTensor1, const uint32_t calCount)
{
    if (g_coreType == AIC) {
        return;
    }
    ReGluImpl<T, false>(dstTensor, srcTensor0, srcTensor1, calCount);
}

#pragma end_pipe
}  // namespace AscendC
#endif
#endif  // LIB_REGLU_REGLU_H
