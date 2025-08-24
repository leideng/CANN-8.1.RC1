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
 * \file simplesoftmax.h
 * \brief SimpleSoftMax api of AscendC
 */
#ifndef LIB_SOFTMAX_SIMPLESOFTMAX_H
#define LIB_SOFTMAX_SIMPLESOFTMAX_H

#include "kernel_tensor.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/activation/softmax_utils.h"
#if __CCE_AICORE__ >= 200
#include "../../impl/activation/softmax/softmax_common.h"
#include "../../impl/activation/softmax/simple_softmax_base_impl.h"

#pragma begin_pipe(V)
namespace AscendC {
/*!
 * \ingroup SimpleSoftMax
 * \brief compute process: y = exp(x-inmax)/insum
 * \note support data type: half and float
 * \param [out] dstTensor: output y
 * \param [in] inSumTensor: input insum
 * \param [in] inMaxTensor: input inmax
 * \param [in] srcTensor: input x
 * \param [in] softmaxShapeInfo: input srcTensor shape
 * \param [in] tiling: input softmaxtiling
 * \param [in] isReuseSource: reserved param
 * \param [in] isBasicBlock: if srcTensor shape[m,k] satisfy the condition(m%8 == 0 && k%64 == 0),you can set true to
 *                           improve performance, but it is a reserved param when isDataFormatNZ = true
 * \param [in] isDataFormatNZ: if the data format of input srcTensor is NZ
 */
template <typename T, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SimpleSoftMax(const LocalTensor<T>& dstTensor, const LocalTensor<T>& inSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<T>& srcTensor, const SoftMaxTiling& tiling,
    const SoftMaxShapeInfo& softmaxShapeInfo = {})
{
    if ASCEND_IS_AIC {
        return;
    }
    SimpleSoftMaxImpl<T, T, isBasicBlock, isDataFormatNZ, config>(dstTensor, inSumTensor, inMaxTensor, srcTensor,
        tiling, softmaxShapeInfo);
}

/*!
 * \ingroup SimpleSoftMax
 * \brief compute process: y = exp(x-inmax)/insum
 * \param [out] dstTensor: output y with dtype of half
 * \param [in] inSumTensor: input insum with dtype of float
 * \param [in] inMaxTensor: input inmax with dtype of float
 * \param [in] srcTensor: input x with dtype of half
 * \param [in] softmaxShapeInfo: input srcTensor shape
 * \param [in] tiling: input softmaxtiling
 * \param [in] isReuseSource: reserved param
 * \param [in] isBasicBlock: if srcTensor shape[m,k] satisfy the condition(m%8 == 0 && k%64 == 0), you can set true to
 *                           improve performance, but it is a reserved param when isDataFormatNZ = true
 * \param [in] isDataFormatNZ: if the data format of input srcTensor is NZ
 */
template <typename T, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SimpleSoftMax(const LocalTensor<half>& dstTensor, const LocalTensor<float>& inSumTensor,
    const LocalTensor<float>& inMaxTensor, const LocalTensor<half>& srcTensor, const SoftMaxTiling& tiling,
    const SoftMaxShapeInfo& softmaxShapeInfo = {})
{
    if ASCEND_IS_AIC {
        return;
    }
    SimpleSoftMaxImpl<half, float, isBasicBlock, isDataFormatNZ, config>(dstTensor, inSumTensor, inMaxTensor, srcTensor,
        tiling, softmaxShapeInfo);
}

/*!
 * \ingroup SimpleSoftMax
 * \brief compute process: y = exp(x-inmax)/insum
 * \note support data type: half and float
 * \param [out] dstTensor: output y
 * \param [in] inSumTensor: input insum
 * \param [in] inMaxTensor: input inmax
 * \param [in] srcTensor: input x
 * \param [in] sharedTmpBuffer: input local temporary Tensor,you can get the range by tilingfunc of
 *                              GetSoftMaxMinTmpSize/GetSoftMaxMaxTmpSize
 * \param [in] softmaxShapeInfo: input srcTensor shape
 * \param [in] tiling: input softmaxtiling
 * \param [in] isReuseSource: reserved param
 * \param [in] isBasicBlock: if srcTensor shape[m,k] satisfy the condition(m%8 == 0 && k%64 == 0), you can set true to
 *                           improve performance, but it is a reserved param when isDataFormatNZ = true
 * \param [in] isDataFormatNZ: if the data format of input srcTensor is NZ
 */
template <typename T, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SimpleSoftMax(const LocalTensor<T>& dstTensor, const LocalTensor<T>& inSumTensor,
    const LocalTensor<T>& inMaxTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
{
    if ASCEND_IS_AIC {
        return;
    }
    SimpleSoftMaxImpl<T, T, isBasicBlock, isDataFormatNZ, config>(dstTensor, inSumTensor, inMaxTensor, srcTensor,
        sharedTmpBuffer, tiling, softmaxShapeInfo);
}

/*!
 * \ingroup SimpleSoftMax
 * \brief compute process: y = exp(x-inmax)/insum
 * \param [out] dstTensor: output y with dtype of half
 * \param [in] inSumTensor: input insum with dtype of float
 * \param [in] inMaxTensor: input inmax with dtype of float
 * \param [in] srcTensor: input x with dtype of half
 * \param [in] sharedTmpBuffer: input local temporary Tensor,you can get the range by tilingfunc of
 *                              GetSoftMaxMinTmpSize/GetSoftMaxMaxTmpSize
 * \param [in] softmaxShapeInfo: input srcTensor shape
 * \param [in] tiling: input softmaxtiling
 * \param [in] isReuseSource: reserved param
 * \param [in] isBasicBlock: if srcTensor shape[m,k] satisfy the condition(m%8 == 0 && k%64 == 0), you can set true to
 *                           improve performance, but it is a reserved param when isDataFormatNZ = true
 * \param [in] isDataFormatNZ: if the data format of input srcTensor is NZ
 */
template <typename T, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SimpleSoftMax(const LocalTensor<half>& dstTensor, const LocalTensor<float>& inSumTensor,
    const LocalTensor<float>& inMaxTensor, const LocalTensor<half>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling,
    const SoftMaxShapeInfo& softmaxShapeInfo = {})
{
    if ASCEND_IS_AIC {
        return;
    }
    SimpleSoftMaxImpl<half, float, isBasicBlock, isDataFormatNZ, config>(dstTensor, inSumTensor, inMaxTensor, srcTensor,
        sharedTmpBuffer, tiling, softmaxShapeInfo);
}
} // namespace AscendC

#pragma end_pipe
#endif
#endif // LIB_SOFTMAX_SIMPLESOFTMAX_H