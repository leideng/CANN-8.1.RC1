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
 * \file swish.h
 * \brief
 */
#ifndef LIB_SWISH_SWISH_H
#define LIB_SWISH_SWISH_H

#include "kernel_tensor.h"
#include "../../impl/activation/swish/swish_common_impl.h"

namespace AscendC {
/* !
 * \brief This function implements the swish function. swish(x) = x / (1 + e^(-βx))
 * (e.g. swish(0.5312) is 0.3784).
 *
 * \note support data type: half and float
 *
 * \param [out] dstTensor, output LocalTensor
 * \param [in] srcTensor, input LocalTensor
 * \param [in] dataSize, amount of data to be calculated
 * \param [in] scalarValue, The β Parameter in the Activation Function
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline __inout_pipe__(V) void Swish(
    const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal, uint32_t dataSize, const T scalarValue)
{
    // Only for AI Vector Core
    if ASCEND_IS_AIC {
        return;
    }
    SwishCompute<T, false>(dstLocal, srcLocal, dataSize, scalarValue);
}
}  // namespace AscendC
#endif  // LIB_SWISH_SWISH_H