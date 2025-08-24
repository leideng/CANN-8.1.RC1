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
 * \file silu.h
 * \brief
 */
#ifndef LIB_SILU_SILU_H
#define LIB_SILU_SILU_H

#include "kernel_tensor.h"
#include "../../impl/activation/silu/silu_common_impl.h"

namespace AscendC {
/* !
 * \brief This function implements the Sigmoid Linear Unit (SiLU) function. silu(x) = x / (1 + e^(-x))
 * (e.g. silu(1.04788) is 0.7758789). For details about the interface description, see
 * https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
 *
 * \note support data type: half and float
 *
 * \param [out] dstTensor, output LocalTensor
 * \param [in] srcTensor, input LocalTensor
 * \param [in] dataSize, amount of data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline __inout_pipe__(V) void Silu(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    uint32_t dataSize)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    SiluCompute<T, false>(dstLocal, srcLocal, dataSize);
}

}  // namespace AscendC
#endif // LIB_SILU_SILU_H
