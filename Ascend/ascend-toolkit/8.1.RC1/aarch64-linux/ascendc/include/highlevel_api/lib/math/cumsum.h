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
 * \file cumsum.h
 * \brief
 */
#ifndef LIB_MATH_CUMSUM_H
#define LIB_MATH_CUMSUM_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "../../impl/math/cumsum/cumsum_common_impl.h"
#include "lib/math/cumsum_utils.h"
#if ASCENDC_CPU_DEBUG
#include "kernel_log.h"
#endif
#if __CCE_AICORE__ >= 200

namespace AscendC {
#pragma begin_pipe(V)

constexpr CumSumConfig defaultCumSumConfig = {true, false, true};

/* !
 * \brief This function calculates the average based on the orientation of the last axis or fist axis.
 * For details about the interface description, see
 * https://pytorch.org/docs/stable/generated/torch.cumsum.html
 *
 * \note support data type: half and float
 *
 * \param [out] dstTensor, output LocalTensor
 * \param [out] lastRowTensor, the last row of the output LocalTensor
 * \param [in] srcTensor, input LocalTensor
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 * \param [in] cumSumInfo, shape information of srcTensor
 */

template <typename T, const CumSumConfig &config = defaultCumSumConfig>
__aicore__ inline void CumSum(LocalTensor<T> &dstTensor, LocalTensor<T> &lastRowTensor, const LocalTensor<T> &srcTensor,
    LocalTensor<uint8_t> &sharedTmpBuffer, const CumSumInfo &cumSumInfo)
{
    CumSumImpl<T, config>(dstTensor, lastRowTensor, srcTensor, sharedTmpBuffer, cumSumInfo);
}

/* !
 * \brief This function calculates the average based on the orientation of the last axis or fist axis.
 * For details about the interface description, see
 * https://pytorch.org/docs/stable/generated/torch.cumsum.html
 *
 * \note support data type: half and float
 *
 * \param [out] dstTensor, output LocalTensor
 * \param [out] lastRowTensor, the last row of the output LocalTensor
 * \param [in] srcTensor, input LocalTensor
 * \param [in] cumSumInfo, shape information of srcTensor
 */

template <typename T, const CumSumConfig &config = defaultCumSumConfig>
__aicore__ inline void CumSum(LocalTensor<T> &dstTensor, LocalTensor<T> &lastRowTensor, const LocalTensor<T> &srcTensor,
    const CumSumInfo &cumSumInfo)
{
    if ASCEND_IS_AIC {
        return;
    }
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    CumSum<T, config>(dstTensor, lastRowTensor, srcTensor, sharedTmpBuffer, cumSumInfo);
}

#pragma end_pipe
}  // namespace AscendC

#endif

#endif  // LIB_CUMSUM_H