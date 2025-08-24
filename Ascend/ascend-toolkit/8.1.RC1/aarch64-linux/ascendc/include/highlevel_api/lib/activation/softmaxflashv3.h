/* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmaxflashv3.h
 * \brief Softmaxflashv3 api of AscendC
 */
#ifndef LIB_SOFTMAX_SOFTMAXFLASHV3_H
#define LIB_SOFTMAX_SOFTMAXFLASHV3_H

#include "kernel_tensor.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/activation/softmax_utils.h"
#if __CCE_AICORE__ >= 200
#include "../../impl/activation/softmax/softmax_common.h"
#include "../../impl/activation/softmax/softmax_flashv3_base_impl.h"

#pragma begin_pipe(V)
namespace AscendC {
/*!
 * \ingroup SoftmaxFlashV3
 * \brief softmax calculation in FlashAttention-2 process:
 *        if isUpdate = false:
 *            rowMeanLocal = rowsum(baseK) / baseK, rowMeanGlobal = rowsum(x), mean = rowMeanGlobal,
 *            meanTmp = rowMeanGlobal - rowMeanLocal, x = x - meanTmp * (a / 1 - a), 
 *            maxTmp = rowmax(x), shiftCurr = (rowMeanGlobal - mean) * (a / 1 - a), max = maxTmp + shiftCurr,
 *            y = exp(x-max), sum = rowsum(y)
 *        if isUpdate = true:
 *            rowMeanLocal = rowsum(baseK) / baseK, rowMeanGlobal = rowsum(x), mean = (rowMeanGlobal + inmean * (n - 1)) / n,
 *            meanTmp = rowMeanGlobal - rowMeanLocal, x = x - meanTmp * (a / 1 - a), 
 *            maxTmp = rowmax(x), shiftCurr = (rowMeanGlobal - mean) * (a / 1 - a), shiftPrev = (inmean - mean) * (a / 1 - a), max = max(maxTmp + shiftCurr, inmax + shiftPrev),
 *            y = exp(x-max), sum = rowsum(y)
 *            expmax = exp(inmax - max + shiftPrev), sum = expmax * insum + sum
 * \note support data type: T half, U float
 * \param [out] dstTensor: output y
 * \param [out] meanTensor: output mean
 * \param [out] expSumTensor: output sum
 * \param [out] maxTensor: output max
 * \param [out] expMaxTensor: output expmax, only valid when isUpdate is true
 * \param [in] srcTensor: input x
 * \param [in] inMeanTensor: input last inmean
 * \param [in] inExpSumTensor: input last insum
 * \param [in] inMaxTensor: input last inmax
 * \param [in] softmaxShapeInfo: input src shape
 * \param [in] tiling: input softmaxtiling
 * \param [in] isUpdate: whether to update the sum and expmax
 * \param [in] isReuseSource: reserved param
 * \param [in] isBasicBlock: if src shape[m,k] satisfy the condition(m%8 == 0 && k%64 == 0), you can set true to improve
 *                           performance, but it is a reserved param when isDataFormatNZ = true
 * \param [in] isDataFormatNZ: if the data format of input srcTensor is NZ
 */
template <typename T, typename U, bool isUpdate = false, bool isReuseSource = false, bool isBasicBlock = false,
    bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftmaxFlashV3(const LocalTensor<T>& dstTensor, const LocalTensor<U>& meanTensor,
    const LocalTensor<U>& expSumTensor, const LocalTensor<U>& maxTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<T>& expMaxTensor, const LocalTensor<U>& inMeanTensor, const LocalTensor<U>& inExpSumTensor,
    const LocalTensor<U>& inMaxTensor, const SoftMaxTiling& tiling, const SoftMaxParams& params)
{
    if ASCEND_IS_AIC {
        return;
    }
    TRACE_START(TraceId::SoftmaxFlashV3);
    SoftmaxFlashV3Impl<T, U, isUpdate, isBasicBlock, isDataFormatNZ, config>(dstTensor, meanTensor, expSumTensor,
        maxTensor, srcTensor, expMaxTensor, inMeanTensor, inExpSumTensor, inMaxTensor, tiling, params);
    TRACE_STOP(TraceId::SoftmaxFlashV3);
}

/*!
 * \ingroup SoftmaxFlashV3
 * \brief softmax calculation in FlashAttention-2 process:
 *        if isUpdate = false:
 *            rowMeanLocal = rowsum(baseK) / baseK, rowMeanGlobal = rowsum(x), mean = rowMeanGlobal,
 *            meanTmp = rowMeanGlobal - rowMeanLocal, x = x - meanTmp * (a / 1 - a), 
 *            maxTmp = rowmax(x), shiftCurr = (rowMeanGlobal - mean) * (a / 1 - a), max = maxTmp + shiftCurr,
 *            y = exp(x-max), sum = rowsum(y)
 *        if isUpdate = true:
 *            rowMeanLocal = rowsum(baseK) / baseK, rowMeanGlobal = rowsum(x), mean = (rowMeanGlobal + inmean * (n - 1)) / n,
 *            meanTmp = rowMeanGlobal - rowMeanLocal, x = x - meanTmp * (a / 1 - a), 
 *            maxTmp = rowmax(x), shiftCurr = (rowMeanGlobal - mean) * (a / 1 - a), shiftPrev = (inmean - mean) * (a / 1 - a), max = max(maxTmp + shiftCurr, inmax + shiftPrev),
 *            y = exp(x-max), sum = rowsum(y)
 *            expmax = exp(inmax - max + shiftPrev), sum = expmax * insum + sum
 * \note support data type: T half, U float
 * \param [out] dstTensor: output y
 * \param [out] meanTensor: output mean
 * \param [out] expSumTensor: output sum
 * \param [out] maxTensor: output max
 * \param [out] expMaxTensor: output expmax, only valid when isUpdate is true
 * \param [in] srcTensor: input x
 * \param [in] inMeanTensor: input last inmean
 * \param [in] inExpSumTensor: input last insum
 * \param [in] inMaxTensor: input last inmax
 * \param [in] softmaxShapeInfo: input src shape
 * \param [in] tiling: input softmaxtiling
 * \param [in] isUpdate: whether to update the sum and expmax
 * \param [in] isReuseSource: reserved param
 * \param [in] isBasicBlock: if src shape[m,k] satisfy the condition(m%8 == 0 && k%64 == 0), you can set true to improve
 *                           performance, but it is a reserved param when isDataFormatNZ = true
 * \param [in] isDataFormatNZ: if the data format of input srcTensor is NZ
 */
template <typename T, typename U, bool isUpdate = false, bool isReuseSource = false, bool isBasicBlock = false,
    bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftmaxFlashV3(const LocalTensor<T>& dstTensor, const LocalTensor<U>& meanTensor,
    const LocalTensor<U>& expSumTensor, const LocalTensor<U>& maxTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<T>& expMaxTensor, const LocalTensor<U>& inMeanTensor, const LocalTensor<U>& inExpSumTensor,
    const LocalTensor<U>& inMaxTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling,
    const SoftMaxParams& params)
{
    if ASCEND_IS_AIC {
        return;
    }
    TRACE_START(TraceId::SoftmaxFlashV3);
    SoftmaxFlashV3Impl<T, U, isUpdate, isBasicBlock, isDataFormatNZ, config>(dstTensor, meanTensor, expSumTensor,
        maxTensor, srcTensor, expMaxTensor, inMeanTensor, inExpSumTensor, inMaxTensor, sharedTmpBuffer, tiling, params);
    TRACE_STOP(TraceId::SoftmaxFlashV3);
}
} // namespace AscendC
#pragma end_pipe
#endif
#endif // LIB_SOFTMAX_SoftmaxFlashV3_H
