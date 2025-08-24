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
 * \file mean.h
 * \brief
 */
#ifndef LIB_REDUCE_MEAN_H
#define LIB_REDUCE_MEAN_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "../../impl/reduce/mean/mean_common_impl.h"
#if ASCENDC_CPU_DEBUG
#include "kernel_log.h"
#include <type_traits>
#include <stdio.h>
#endif

namespace AscendC {
#pragma begin_pipe(V)
/* !
 * \brief This function calculates the average based on the orientation of the last axis.
 * For details about the interface description, see
 * https://pytorch.org/docs/stable/generated/torch.mean.html
 *
 * \note support data type: half and float
 *
 * \param [out] dstTensor, output LocalTensor
 * \param [in] srcTensor, input LocalTensor
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 * \param [in] meanParams, shape information of srcTensor
 */
template <typename T, typename accType = T, bool isReuseSource = false, bool isBasicBlock = false,
          int32_t reduceDim = -1>
__aicore__ inline void Mean(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor,
                            const LocalTensor<uint8_t> &sharedTmpBuffer, const MeanParams &meanParams)
{
    if ASCEND_IS_AIC
    {
        return;
    }
    ASCENDC_ASSERT(((std::is_same<T, half>::value && std::is_same<accType, half>::value) ||
                    (std::is_same<T, float>::value && std::is_same<accType, float>::value) ||
                    (std::is_same<T, half>::value && std::is_same<accType, float>::value)),
                   { KERNEL_LOG(KERNEL_ERROR, "Two conditions are supported: "
                                              "1.T is half or float , and accType is same with T; "
                                              "2.T is half and accType is float."); });
    if constexpr (sizeof(T) == sizeof(half) && sizeof(accType) == sizeof(float))
    {
        MeanCast(dstTensor, srcTensor, sharedTmpBuffer, meanParams);
    }
    else
    {
        MeanCommon(dstTensor, srcTensor, sharedTmpBuffer, meanParams);
    }
}

/* !
 * \brief This function calculates the average based on the orientation of the last axis.
 * For details about the interface description, see
 * https://pytorch.org/docs/stable/generated/torch.mean.html
 *
 * \note support data type: half and float
 *
 * \param [out] dstTensor, output LocalTensor
 * \param [in] srcTensor, input LocalTensor
 * \param [in] meanParams, shape information of srcTensor
 */
template <typename T, typename accType = T, bool isReuseSource = false, bool isBasicBlock = false,
          int32_t reduceDim = -1>
__aicore__ inline void Mean(
    const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const MeanParams &meanParams)
{
    if ASCEND_IS_AIC
    {
        return;
    }
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    Mean<T, accType>(dstTensor, srcTensor, sharedTmpBuffer, meanParams);
}
#pragma end_pipe
}  // namespace AscendC

#endif  // LIB_REDUCE_MEAN_H