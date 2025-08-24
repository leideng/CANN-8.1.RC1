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
 * \file sum.h
 * \brief
 */
#ifndef LIB_REDUCE_SUM_H
#define LIB_REDUCE_SUM_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "../../impl/reduce/sum/sum_common_impl.h"
#if ASCENDC_CPU_DEBUG
#include "kernel_log.h"
#include <type_traits>
#endif
namespace AscendC {
/* !
 * \brief This function calculates the sum of all elements in the input tensor.
* \For details about the interface description, see https://pytorch.org/docs/stable/generated/torch.round.html.
 *
 * \note support data type: half and float
 *
 * \param [out] dstTensor, output LocalTensor
 * \param [in] srcTensor, input LocalTensor
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 * \param [in] sumParams, shape information of srcLocal
 */
#pragma begin_pipe(V)
template <typename T, int32_t reduceDim = -1, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void Sum(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const SumParams &sumParams)
{
    if ASCEND_IS_AIC {
        return;
    }
    SumCompute<T, reduceDim, isReuseSource, isBasicBlock>(dstTensor, srcTensor, sharedTmpBuffer, sumParams);
}

/* !
 * \brief This function calculates the sum of all elements in the input tensor.
* \For details about the interface description, see https://pytorch.org/docs/stable/generated/torch.round.html.
 *
 * \note support data type: half and float
 *
 * \param [out] dstTensor, output LocalTensor
 * \param [in] srcTensor, input LocalTensor
 * \param [in] sumParams, shape information of srcLocal
 */
template <typename T, int32_t reduceDim = -1, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void Sum(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const SumParams &sumParams)
{
    if ASCEND_IS_AIC {
        return;
    }
#if __CCE_AICORE__ >= 200
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    uint32_t elementNumPerBlk = ONE_BLK_SIZE / sizeof(T);  // half=16 float=8
    uint32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T);
    uint32_t repeatTimes = (sumParams.n + elementNumPerRep - 1) / elementNumPerRep;
    uint32_t finalWorkSize = (repeatTimes + elementNumPerBlk - 1) / elementNumPerBlk * elementNumPerBlk * sizeof(T);
    ASCENDC_ASSERT(
        sharedTmpBuffer.GetSize() >= finalWorkSize, { KERNEL_LOG(KERNEL_ERROR, "tmpBuffer isn't enough!"); });
    sharedTmpBuffer.SetSize(finalWorkSize);
    Sum<T, reduceDim, isReuseSource, isBasicBlock>(dstTensor, srcTensor, sharedTmpBuffer, sumParams);
#endif
}
#pragma end_pipe
}  // namespace AscendC
#endif // LIB_REDUCE_SUM_H
