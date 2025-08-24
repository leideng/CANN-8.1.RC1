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
 * \file tan.h
 * \brief Defines a series of interface used to do elementwise math tan calculation.
 * Formula: tan(x) = xP(x) / ((π/2 - x)(π/2 + x)Q(x))
 * The Tan function does not have an elementary function expression, first normalize x to (-π/2, π/2)
 * and then calculating by function approximation.
 * Final solution：
 *  k=round(x/π), x0=x-kπ, x0 belongs to (-π/2, π/2)
 *  π=π_0+π_1+π_2+π_3+π_4 achieve final precision compensation.
 *  Final solution：
 *  k = round(x * invpi)
 *  x -= k * pi_0
 *  x -= k * pi_1
 *  down1 = x + pio2_high // pi/2 + x
 *  down2 = x - pio2_high // x - pi/2
 *  x -= k * pi_2
 *  down1 -= k * pi_2
 *  down2 -= k * pi_2
 *  x -= k * pi_3
 *  down1 -= k * pi_3
 *  down2 -= k * pi_3
 *  x -= k * pi_4
 *  down1 -= k * pi_4
 *  down2 -= k * pi_4
 *  P(x) = (x^2 * R0 + R1) * x^2 + R2
 *  Q(x) = x^2 * R3
 *  R0 = 0.0698520831551998762793
 *  R1 = -6.8711573651634203789
 *  R2 = 61.20362572811089435388
 *  R3 = -24.8048928861126769186219
 */
#ifndef LIB_MATH_TAN_H
#define LIB_MATH_TAN_H

#if __CCE_AICORE__ == 220 || __CCE_AICORE__ == 200

#include "kernel_tensor.h"
#include "../../impl/math/tan/tan_common_impl.h"

namespace AscendC {
#pragma begin_pipe(V)
/*!
 * \ingroup Tan
 * \brief compute Tan elementwisely
 * \tparam T: half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 * this parameter is reserved, please use the default value.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] srcTensor: input LocalTensor
 * \param [in] sharedTmpBuffer: extra temporary shared space used for intermediate values among calculation process,
 * whose required space size should refer to corresponding tiling API, which is defined at tan_tiling.h.
 * Generally, the more space you allocate, the better performance you will achieve, and the performance
 * reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it is not guaranteed
 * that the shared space will be cleared after usage, the data could be anything.
 * \note src/dst Tensor must be 32B aligned, and it doesn't allow src/dst/sharedTmpBuffer tensor address overlap.
 * Input data valid range should be (-65504, 65504)
 */
 template <typename T, bool isReuseSource = false>
__aicore__ inline void Tan(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer)
{
    Tan<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, srcTensor.GetSize());
}

/*!
 * \ingroup Tan
 * \brief compute Tan elementwisely
 * \tparam T: half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 * this parameter is reserved, please use the default value.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] srcTensor: input LocalTensor
 * \param [in] sharedTmpBuffer: extra temporary shared space used for intermediate values among calculation process,
 * whose required space size should refer to corresponding tiling API, which is defined at tan_tiling.h.
 * Generally, the more space you allocate, the better performance you will achieve, and the performance
 * reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it is not guaranteed
 * that the shared space will be cleared after usage, the data could be anything.
 * \param [in] calCount: the number of elements to be processed.
 * \note src/dst Tensor must be 32B aligned, and it doesn't allow src/dst/sharedTmpBuffer tensor address overlap.
 * Input data valid range should be (-65504, 65504)
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Tan(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    TanImpl(dstTensor, srcTensor, sharedTmpBuffer, calCount);
}

/*!
 * \ingroup Tan
 * \brief compute Tan elementwisely
 * \tparam T: half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 * this parameter is reserved, please use the default value.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] srcTensor: input LocalTensor
 * \param [in] calCount: the number of elements to be processed.
 * \note src/dst Tensor must be 32B aligned, and it doesn't allow src/dst/sharedTmpBuffer tensor address overlap.
 * Input data valid range should be (-65504, 65504)
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Tan(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const uint32_t calCount)
{
    TanImpl(dstTensor, srcTensor, calCount);
}

/*!
 * \ingroup Tan
 * \brief compute Tan elementwisely
 * \tparam T: half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 * this parameter is reserved, please use the default value.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] srcTensor: input LocalTensor
 * \note src/dst Tensor must be 32B aligned, and it doesn't allow src/dst/sharedTmpBuffer tensor address overlap.
 * Input data valid range should be (-65504, 65504)
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Tan(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor)
{
    Tan<T, isReuseSource>(dstTensor, srcTensor, srcTensor.GetSize());
}
#pragma end_pipe
} // namespace AscendC

#endif

#endif // LIB_MATH_TAN_H