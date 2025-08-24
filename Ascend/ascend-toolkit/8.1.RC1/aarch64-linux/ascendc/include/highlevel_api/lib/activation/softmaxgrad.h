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
 * \file softmaxgrad.h
 * \brief SoftmaxGrad and SoftmaxGradFront api of AscendC
 */
#ifndef LIB_SOFTMAX_SOFTMAXGRAD_H
#define LIB_SOFTMAX_SOFTMAXGRAD_H

#include "kernel_tensor.h"
#include "kernel_tiling/kernel_tiling.h"
#if __CCE_AICORE__ >= 200
#include "../../impl/activation/softmax/softmax_common.h"
#include "../../impl/activation/softmax/softmax_grad_base_impl.h"

#pragma begin_pipe(V)
namespace AscendC {
/*!
 * \ingroup SoftmaxGrad
 * \brief compute process: sum = rowsum(grad * x), grad * x - sum * x
 *                         if isFront = true
 *                             y = sum
 *                         if isFront = false
 *                             y = grad * x - sum * x
 * \note support data type: half and float
 * \param [out] dstTensor: output y
 * \param [in] gradTensor: input grad
 * \param [in] srcTensor: input x
 * \param [in] softmaxShapeInfo: input x shape
 * \param [in] tiling: input softmaxtiling
 * \param [in] isFront: compute mode refer to brief
 * \param [in] isReuseSource: reserved param
 * \param [in] isDataFormatNZ: if the data format of input srcTensor is NZ
 */
template <typename T, bool isReuseSource = false, bool isDataFormatNZ = false>
__aicore__ inline void SoftmaxGrad(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor,
    const LocalTensor<T>& srcTensor, const SoftMaxTiling& tiling, bool isFront = false,
    const SoftMaxShapeInfo& softmaxShapeInfo = {})
{
    if ASCEND_IS_AIC {
        return;
    }
    TRACE_START(TraceId::SoftmaxGrad);
    SoftmaxGradImpl<T, isDataFormatNZ>(dstTensor, gradTensor, srcTensor, tiling, isFront, softmaxShapeInfo);
    TRACE_STOP(TraceId::SoftmaxGrad);
}

/*!
 * \ingroup SoftmaxGradFront
 * \brief compute process: y = rowsum(grad * x)
 * \note support data type: half and float
 * \param [out] dstTensor: output y
 * \param [in] gradTensor: input grad
 * \param [in] srcTensor: input x
 * \param [in] softmaxShapeInfo: input x shape
 * \param [in] tiling: input softmaxtiling
 * \param [in] isBasicBlock: if src shape[m,k] satisfy the condition(m%8 == 0 && k%64 == 0), you can set true to improve
 *                           performance , but it is a reserved param when isDataFormatNZ = true
 * \param [in] isDataFormatNZ: if the data format of input srcTensor is NZ
 */
template <typename T, bool isBasicBlock = false, bool isDataFormatNZ = false>
__aicore__ inline void SoftmaxGradFront(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor,
    const LocalTensor<T>& srcTensor, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
{
    if ASCEND_IS_AIC {
        return;
    }
    TRACE_START(TraceId::SoftmaxGrad);
    SoftmaxGradFrontImpl<T, isBasicBlock, isDataFormatNZ>(dstTensor, gradTensor, srcTensor, tiling, softmaxShapeInfo);
    TRACE_STOP(TraceId::SoftmaxGrad);
}

/*!
 * \ingroup SoftmaxGrad
 * \brief compute process: sum = rowsum(grad * x), grad * x - sum * x
 *                         if isFront = true
 *                             y = sum
 *                         if isFront = false
 *                             y = grad * x - sum * x
 * \note support data type: half and float
 * \param [out] dstTensor: output y
 * \param [in] gradTensor: input grad
 * \param [in] srcTensor: input x
 * \param [in] softmaxShapeInfo: input x shape
 * \param [in] sharedTmpBuffer: input local temporary Tensor,you can get the range by tilingfunc of
 *                              GetSoftMaxGradMinTmpSize/GetSoftMaxGradMaxTmpSize
 * \param [in] tiling: input softmaxtiling
 * \param [in] isFront: compute mode refer to brief
 * \param [in] isReuseSource: reserved param
 * \param [in] isDataFormatNZ: if the data format of input srcTensor is NZ
 */
template <typename T, bool isReuseSource = false, bool isDataFormatNZ = false>
__aicore__ inline void SoftmaxGrad(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling,
    bool isFront = false, const SoftMaxShapeInfo& softmaxShapeInfo = {})
{
    if ASCEND_IS_AIC {
        return;
    }
    TRACE_START(TraceId::SoftmaxGrad);
    SoftmaxGradImpl<T, isDataFormatNZ>(dstTensor, gradTensor, srcTensor, sharedTmpBuffer, tiling, isFront,
        softmaxShapeInfo);
    TRACE_STOP(TraceId::SoftmaxGrad);
}

/*!
 * \ingroup SoftmaxGradFront
 * \brief compute process: y = rowsum(grad * x)
 * \note support data type: half and float
 * \param [out] dstTensor: output y
 * \param [in] gradTensor: input grad
 * \param [in] srcTensor: input x
 * \param [in] softmaxShapeInfo: input x shape
 * \param [in] sharedTmpBuffer: input local temporary Tensor,you can get the range by tilingfunc of
 *                              GetSoftMaxGradMinTmpSize/GetSoftMaxGradMaxTmpSize
 * \param [in] tiling: input softmaxtiling
 * \param [in] isBasicBlock: if src shape[m,k] satisfy the condition(m%8 == 0 && k%64 == 0), you can set true to improve
 *                           performance , but it is a reserved param when isDataFormatNZ = true
 * \param [in] isDataFormatNZ: if the data format of input srcTensor is NZ
 */
template <typename T, bool isBasicBlock = false, bool isDataFormatNZ = false>
__aicore__ inline void SoftmaxGradFront(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling,
    const SoftMaxShapeInfo& softmaxShapeInfo = {})
{
    if ASCEND_IS_AIC {
        return;
    }
    SoftmaxGradFrontImpl<T, isBasicBlock, isDataFormatNZ>(dstTensor, gradTensor, srcTensor, sharedTmpBuffer, tiling,
        softmaxShapeInfo);
}
} // namespace AscendC
#pragma end_pipe
#endif
#endif // LIB_SOFTMAX_SOFTMAXGRAD_H
