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
 * \file exp.h
 */

#ifndef LIB_MATH_EXP_H
#define LIB_MATH_EXP_H

#include "kernel_tensor.h"
#include "../../impl/math/exp/exp_common_impl.h"

namespace AscendC {

#pragma begin_pipe(V)

/*!
 * \ingroup Exp
 * \brief Use Taylor Formula for Exp calculation
 *        exp(a) = (e^b) * (e^c) b is integer part of a, c is decimal part of a.  a = b + c
 *        e^c = 1 + c + c^2 / (2!) + .... + c^n / n!
 * \tparam T: Data type to be calculated, half or float
 * \tparam taylorExpandLevel: The number of Taylor formula terms (n above)
 * \tparam isReuseSource: Whether to reuse the buffer of srcTensor.
 *         If the value is true, srcTensor can used as tmpBuffer and the data in it will be overwritten.
 *         If the value is false, srcTensor will not be used as tmpBuffer for calculation.
 * \param [out] dstLocal: Output localTensor.
 * \param [in] srcLocal: Input localTensor
 * \param [in] calCount: The number of elements to be processed.
 */
template <typename T, uint8_t taylorExpandLevel, bool isReuseSource = false>
__aicore__ inline void Exp(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }
    static_assert((std::is_same<T, float>::value || std::is_same<T, half>::value),
        "Failed to check the data types, current api support data types are half/float.");
    ExpAPI::ExpImpl<T, taylorExpandLevel, isReuseSource>(dstLocal, srcLocal, calCount);
}

/*!
 * \ingroup Exp
 * \brief Use Taylor Formula for Exp calculation
 *        exp(a) = (e^b) * (e^c) b is integer part of a, c is decimal part of a.  a = b + c
 *        e^c = 1 + c + c^2 / (2!) + .... + c^n / n!
 * \tparam T: Data type to be calculated, half or float
 * \tparam taylorExpandLevel: The number of Taylor formula terms (n above)
 * \tparam isReuseSource: Whether to reuse the buffer of srcTensor.
 *         If the value is true, srcTensor can used as tmpBuffer and the data in it will be overwritten.
 *         If the value is false, srcTensor will not be used as tmpBuffer for calculation.
 * \param [out] dstLocal: Output localTensor.
 * \param [in] srcLocal: Input localTensor
 * \param [in] sharedTmpBuffer：extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at exp_tiling.h.
 *             Generally, the more space you allocate, the better performance you will achieve, and the performance
 *             reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it is not guaranteed
 *             that the shared space will be cleared after usage, the data could be anything.
 * \param [in] calCount: The number of elements to be processed.
 */
template <typename T, uint8_t taylorExpandLevel, bool isReuseSource = false>
__aicore__ inline void Exp(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }
    static_assert((std::is_same<T, float>::value || std::is_same<T, half>::value),
        "Failed to check the data types, current api support data types are half/float.");
    ExpAPI::ExpImpl<T, taylorExpandLevel, isReuseSource>(dstLocal, srcLocal, sharedTmpBuffer, calCount);
}

#pragma end_pipe
}  // namespace AscendC
#endif  // LIB_MATH_EXP_H
