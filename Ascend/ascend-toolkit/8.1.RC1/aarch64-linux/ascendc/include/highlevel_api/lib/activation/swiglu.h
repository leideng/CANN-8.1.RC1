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
 * \file swiglu.h
 * \brief
 */

#ifndef LIB_MATH_SWIGLU_H
#define LIB_MATH_SWIGLU_H


#include "kernel_tensor.h"
#include "../../impl/activation/swiglu/swiglu_common_impl.h"

namespace AscendC {
#pragma begin_pipe(V)
/* !
 * \note support data type: half and float
 *  Function：
    swish(x) = x / (1 + e^(-βx))
    x1 = 1 + e^(-βx)
 * \param [out] dstTensor, output LocalTensor
 * \param [in] srcTensor0, input0 LocalTensor
 * \param [in] srcTensor1, input1 LocalTensor
 * \param [in] scalarValue, input scalar
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void SwiGLU(LocalTensor<T>& dstTensor, LocalTensor<T>& srcTensor0, LocalTensor<T>& srcTensor1,
                              const float& scalarValue)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    SwiGLUImpl<T, isReuseSource>(dstTensor, srcTensor0, srcTensor1, scalarValue, srcTensor0.GetSize());
}

/* !
 * \note support data type: half and float
 *
 * \param [out] dstTensor, output LocalTensor
 * \param [in] srcTensor0, input0 LocalTensor
 * \param [in] srcTensor1, input1 LocalTensor
 * \param [in] scalarValue, input scalar
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void SwiGLU(LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0,
    const LocalTensor<T>& srcTensor1, const float& scalarValue, const LocalTensor<uint8_t>& sharedTmpBuffer)
{
    SwiGLU<T, isReuseSource>(dstTensor, srcTensor0, srcTensor1, scalarValue, sharedTmpBuffer, srcTensor0.GetSize());
}

/* !
 * \note support data type: half and float
 *
 * \param [out] dstTensor, output LocalTensor
 * \param [in] srcTensor0, input0 LocalTensor
 * \param [in] srcTensor1, input1 LocalTensor
 * \param [in] scalarValue, input scalar
 * \param [in] calCount, amount of data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void SwiGLU(LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0,
    const LocalTensor<T>& srcTensor1, const float& scalarValue, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    SwiGLUImpl<T, isReuseSource>(dstTensor, srcTensor0, srcTensor1, scalarValue, calCount);
}

/* !
 * \note support data type: half and float
 *
 * \param [out] dstTensor, output LocalTensor
 * \param [in] srcTensor0, input0 LocalTensor
 * \param [in] srcTensor1, input1 LocalTensor
 * \param [in] scalarValue, input scalar
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 * \param [in] calCount, amount of data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void SwiGLU(LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0,
                              const LocalTensor<T>& srcTensor1, const float& scalarValue,
                              const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    SwiGLUImpl<T, isReuseSource>(dstTensor, srcTensor0, srcTensor1, scalarValue, sharedTmpBuffer, calCount);
}
#pragma end_pipe
}
#endif
