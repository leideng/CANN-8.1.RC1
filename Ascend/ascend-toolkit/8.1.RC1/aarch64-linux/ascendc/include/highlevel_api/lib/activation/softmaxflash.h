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
 * \file softmaxflash.h
 * \brief SoftmaxFlash api of AscendC, this api will be deprecated, please use SoftmaxFlashV2 api instead
 */
#ifndef LIB_SOFTMAX_SOFTMAXFLASH_H
#define LIB_SOFTMAX_SOFTMAXFLASH_H

#include "kernel_tensor.h"
#include "kernel_tiling/kernel_tiling.h"
#if __CCE_AICORE__ >= 200
#include "../../impl/activation/softmax/softmax_common.h"
#include "../../impl/activation/softmax/softmax_flash_base_impl.h"

#pragma begin_pipe(V)
namespace AscendC {
/*!
 * \ingroup SoftmaxFlash
 * \brief softmax calculation in FlashAttention process: you can refet to the paper:
 *        <Flashattention: Fast and memory-efficient exact attention with io-awareness>
 *        if isUpdate = false:
 *            max = rowmax(x),  y = exp(x-max), sum = rowsum(y), y = y / sum
 *        if isUpdate = true:
 *            max = max(rowmax(x), inmax), y = exp(x-max), sum = rowsum(y)
 *            expmax = exp(inmax - max), sum = expmax * insum + sum
 *            expmax = expmax * insum / sum, y = y / sum
 * \note support data type: half and float
 * \param [out] dstTensor: output y
 * \param [out] sumTensor: output sum
 * \param [out] maxTensor: output max
 * \param [out] expMaxTensor: output expmax, only valid when isUpdate is true
 * \param [in] srcTensor: input x
 * \param [in] inSumTensor: input last insum
 * \param [in] inMaxTensor: input last inmax
 * \param [in] softmaxShapeInfo: input src shape
 * \param [in] tiling: input softmaxtiling
 * \param [in] isUpdate: whether to update the sum and expmax
 * \param [in] isReuseSource: reserved param
 * \param [in] isBasicBlock: if src shape[m,k] satisfy the condition(m%8 == 0 && k%64 == 0),
 *                           you can set true to improve performance
 */
template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void SoftmaxFlash(const LocalTensor<T> &dstTensor, const LocalTensor<T> &sumTensor,
    const LocalTensor<T> &maxTensor, const LocalTensor<T> &srcTensor, const LocalTensor<T> &expMaxTensor,
    const LocalTensor<T> &inSumTensor, const LocalTensor<T> &inMaxTensor, const SoftMaxTiling &tiling,
    bool isUpdate = false, const SoftMaxShapeInfo &softmaxShapeInfo = {})
{
    if ASCEND_IS_AIC {
        return;
    }
    TRACE_START(TraceId::SoftmaxFlash);
    SoftmaxFlashImpl<T, isBasicBlock>(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor, inSumTensor,
        inMaxTensor, tiling, isUpdate, softmaxShapeInfo);
    TRACE_STOP(TraceId::SoftmaxFlash);
}
/* !
 * \ingroup SoftmaxFlash
 * \brief softmax calculation in FlashAttention process: you can refet to the paper:
 *        <Flashattention: Fast and memory-efficient exact attention with io-awareness>
 *        if isUpdate = false:
 *            max = rowmax(x),  y = exp(x-max), sum = rowsum(y), y = y / sum
 *        if isUpdate = true:
 *            max = max(rowmax(x), inmax), y = exp(x-max), sum = rowsum(y)
 *            expmax = exp(inmax - max), sum = expmax * insum + sum
 *            expmax = expmax * insum / sum, y = y / sum
 * \param [out] dstTensor: output y with dtype of half
 * \param [out] sumTensor: output sum with dtype of float
 * \param [out] maxTensor: output max with dtype of float
 * \param [out] expMaxTensor: output expmax with dtype of half, only valid when isUpdate is true
 * \param [in] srcTensor: input x with dtype of half
 * \param [in] inSumTensor: input last insum with dtype of float
 * \param [in] inMaxTensor: input last inmax with dtype of float
 * \param [in] softmaxShapeInfo: input src shape
 * \param [in] tiling: input softmaxtiling
 * \param [in] isUpdate: whether to update the sum and expmax
 * \param [in] isReuseSource: reserved param
 * \param [in] isBasicBlock: if src shape[m,k] satisfy the condition(m%8 == 0 && k%64 == 0),
 *                           you can set true to improve performance
 */
template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void SoftmaxFlash(const LocalTensor<half>& dstTensor, const LocalTensor<float>& sumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcTensor, const LocalTensor<half>& expMaxTensor,
    const LocalTensor<float>& inSumTensor, const LocalTensor<float>& inMaxTensor, const SoftMaxTiling& tiling,
    bool isUpdate = false, const SoftMaxShapeInfo& softmaxShapeInfo = {})
{
    if ASCEND_IS_AIC {
        return;
    }
    TRACE_START(TraceId::SoftmaxFlash);
    SoftmaxFlashImpl<T, isBasicBlock>(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor, inSumTensor,
        inMaxTensor, tiling, isUpdate, softmaxShapeInfo);
    TRACE_STOP(TraceId::SoftmaxFlash);
}
/*!
 * \ingroup SoftmaxFlash
 * \brief softmax calculation in FlashAttention process: you can refet to the paper:
 *        <Flashattention: Fast and memory-efficient exact attention with io-awareness>
 *        if isUpdate = false:
 *            max = rowmax(x),  y = exp(x-max), sum = rowsum(y), y = y / sum
 *        if isUpdate = true:
 *            max = max(rowmax(x), inmax), y = exp(x-max), sum = rowsum(y)
 *            expmax = exp(inmax - max), sum = expmax * insum + sum
 *            expmax = expmax * insum / sum, y = y / sum
 * \note support data type: half and float
 * \param [out] dstTensor: output y
 * \param [out] sumTensor: output sum
 * \param [out] maxTensor: output max
 * \param [out] expMaxTensor: output expmax, only valid when isUpdate is true
 * \param [in] srcTensor: input x
 * \param [in] inSumTensor: input last insum
 * \param [in] inMaxTensor: input last inmax
 * \param [in] sharedTmpBuffer: input local temporary Tensor, you can get the range by tilingfunc of
 *                              GetSoftMaxFlashMinTmpSize/GetSoftMaxFlashMaxTmpSize
 * \param [in] softmaxShapeInfo: input src shape
 * \param [in] tiling: input softmaxtiling
 * \param [in] isUpdate: whether to update the sum and expmax
 * \param [in] isReuseSource: reserved param
 * \param [in] isBasicBlock: if src shape[m,k] satisfy the condition(m%8 == 0 && k%64 == 0),
 *                           you can set true to improve performance
 */
template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void SoftmaxFlash(const LocalTensor<T>& dstTensor, const LocalTensor<T>& sumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor,
    const LocalTensor<T>& inSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const SoftMaxTiling& tiling, bool isUpdate = false, const SoftMaxShapeInfo& softmaxShapeInfo = {})
{
    if ASCEND_IS_AIC {
        return;
    }
    TRACE_START(TraceId::SoftmaxFlash);
    SoftmaxFlashImpl<T, isBasicBlock>(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor, inSumTensor,
        inMaxTensor, sharedTmpBuffer, tiling, isUpdate, softmaxShapeInfo);
    TRACE_STOP(TraceId::SoftmaxFlash);
}
/*!
 * \ingroup SoftmaxFlash
 * \brief softmax calculation in FlashAttention process: you can refet to the paper:
 *        <Flashattention: Fast and memory-efficient exact attention with io-awareness>
 *        if isUpdate = false:
 *            max = rowmax(x),  y = exp(x-max), sum = rowsum(y), y = y / sum
 *        if isUpdate = true:
 *            max = max(rowmax(x), inmax), y = exp(x-max), sum = rowsum(y)
 *            expmax = exp(inmax - max), sum = expmax * insum + sum
 *            expmax = expmax * insum / sum, y = y / sum
 * \param [out] dstTensor: output y with dtype of half
 * \param [out] sumTensor: output sum with dtype of float
 * \param [out] maxTensor: output max with dtype of float
 * \param [out] expMaxTensor: output expmax with dtype of half, only valid when isUpdate is true
 * \param [in] srcTensor: input x with dtype of half
 * \param [in] inSumTensor: input last insum with dtype of float
 * \param [in] inMaxTensor: input last inmax with dtype of float
 * \param [in] sharedTmpBuffer: input local temporary Tensor, you can get the range by tilingfunc of
 *                              GetSoftMaxFlashMinTmpSize/GetSoftMaxFlashMaxTmpSize
 * \param [in] softmaxShapeInfo: input src shape
 * \param [in] tiling: input softmaxtiling
 * \param [in] isUpdate: whether to update the sum and expmax
 * \param [in] isReuseSource: reserved param
 * \param [in] isBasicBlock: if src shape[m,k] satisfy the condition(m%8 == 0 && k%64 == 0),
 *                           you can set true to improve performance
 */
template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void SoftmaxFlash(const LocalTensor<half>& dstTensor, const LocalTensor<float>& sumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcTensor, const LocalTensor<half>& expMaxTensor,
    const LocalTensor<float>& inSumTensor, const LocalTensor<float>& inMaxTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, bool isUpdate = false,
    const SoftMaxShapeInfo& softmaxShapeInfo = {})
{
    if ASCEND_IS_AIC {
        return;
    }
    TRACE_START(TraceId::SoftmaxFlash);
    SoftmaxFlashImpl<T, isBasicBlock>(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor, inSumTensor,
        inMaxTensor, sharedTmpBuffer, tiling, isUpdate, softmaxShapeInfo);
    TRACE_STOP(TraceId::SoftmaxFlash);
}
} // namespace AscendC
#pragma end_pipe
#endif
#endif // LIB_SOFTMAX_SOFTMAXFLASHV2_H
