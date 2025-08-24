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
 * \file power.h
 * \brief Power function takes the power of base with exponent and returns a tensor with the result.
 * Mathematical formulas: Power(x, y) = x ^ y
 */
#ifndef LIB_MATH_POWER_H
#define LIB_MATH_POWER_H
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220 || __CCE_AICORE__ == 200)

#include "kernel_tensor.h"
#include "../../impl/math/power/power_common_impl.h"
namespace AscendC {
#pragma begin_pipe(V)
/*!
 * \ingroup Power
 * \brief compute Power elementwisely.
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] src0Tensor: base LocalTensor
 * \param [in] src1Tensor: exponent LocalTensor
 * \param [in] sharedTmpBuffer：extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             power_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 * \param [in] calCount: amount of output data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Power(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor,
    const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)
{
    PowerCommonImpl<T, isReuseSource>(dstTensor, src0Tensor, src1Tensor, sharedTmpBuffer, calCount);
}
/*!
 * \ingroup Power
 * \brief compute Power elementwisely.
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] src0Tensor: base LocalTensor
 * \param [in] src1Tensor: exponent LocalTensor
 * \param [in] calCount: amount of output data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Power(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor,
    const LocalTensor<T>& src1Tensor, uint32_t calCount)
{
    PowerCommonImpl<T, isReuseSource>(dstTensor, src0Tensor, src1Tensor, calCount);
}
/*!
 * \ingroup Power
 * \brief compute Power elementwisely.
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] src0Tensor: base LocalTensor
 * \param [in] src1Tensor: exponent LocalTensor
 * \param [in] sharedTmpBuffer：extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             power_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Power(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor,
    const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer)
{
    Power<T, isReuseSource>(dstTensor, src0Tensor, src1Tensor, sharedTmpBuffer, src0Tensor.GetSize());
}
/*!
 * \ingroup Power
 * \brief compute Power elementwisely.
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] src0Tensor: base LocalTensor
 * \param [in] src1Tensor: exponent LocalTensor
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Power(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor,
    const LocalTensor<T>& src1Tensor)
{
    Power<T, isReuseSource>(dstTensor, src0Tensor, src1Tensor, src0Tensor.GetSize());
}

/*!
 * \ingroup Power
 * \brief compute Power elementwisely.
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] src0Tensor: base LocalTensor
 * \param [in] src1Scalar: exponent Scalar
 * \param [in] sharedTmpBuffer：extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             power_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 * \param [in] calCount: amount of output data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Power(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const T& src1Scalar,
    const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)
{
    PowerCommonImpl<T, isReuseSource>(dstTensor, src0Tensor, src1Scalar, sharedTmpBuffer, calCount);
}
/*!
 * \ingroup Power
 * \brief compute Power elementwisely.
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] src0Tensor: base LocalTensor
 * \param [in] src1Scalar: exponent Scalar
 * \param [in] calCount: amount of output data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Power(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor,
    const T& src1Scalar, uint32_t calCount)
{
    PowerCommonImpl<T, isReuseSource>(dstTensor, src0Tensor, src1Scalar, calCount);
}
/*!
 * \ingroup Power
 * \brief compute Power elementwisely.
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] src0Tensor: base LocalTensor
 * \param [in] src1Scalar: exponent Scalar
 * \param [in] sharedTmpBuffer：extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             power_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Power(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor,
    const T& src1Scalar, const LocalTensor<uint8_t>& sharedTmpBuffer)
{
    Power<T, isReuseSource>(dstTensor, src0Tensor, src1Scalar, sharedTmpBuffer, src0Tensor.GetSize());
}
/*!
 * \ingroup Power
 * \brief compute Power elementwisely.
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] src0Tensor: base LocalTensor
 * \param [in] src1Scalar: exponent Scalar
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Power(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor,
    const T& src1Scalar)
{
    Power<T, isReuseSource>(dstTensor, src0Tensor, src1Scalar, src0Tensor.GetSize());
}

/*!
 * \ingroup Power
 * \brief compute Power elementwisely.
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] src0Scalar: base Scalar
 * \param [in] src1Tensor: exponent LocalTensor
 * \param [in] sharedTmpBuffer：extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             power_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 * \param [in] calCount: amount of output data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Power(const LocalTensor<T>& dstTensor, const T& src0Scalar, const LocalTensor<T>& src1Tensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)
{
    PowerCommonImpl<T, isReuseSource>(dstTensor, src0Scalar, src1Tensor, sharedTmpBuffer, calCount);
}
/*!
 * \ingroup Power
 * \brief compute Power elementwisely.
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] src0Scalar: input Scalar
 * \param [in] src1Tensor: exponent LocalTensor
 * \param [in] calCount: amount of output data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Power(const LocalTensor<T>& dstTensor, const T& src0Scalar,
    const LocalTensor<T>& src1Tensor, uint32_t calCount)
{
    PowerCommonImpl<T, isReuseSource>(dstTensor, src0Scalar, src1Tensor, calCount);
}
/*!
 * \ingroup Power
 * \brief compute Power elementwisely.
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] src0Scalar: base Scalar
 * \param [in] src1Tensor: exponent LocalTensor
 * \param [in] sharedTmpBuffer：extra temporary shared space used for intermediate values among calculation process,
 *             whose required space size should refer to corresponding tiling API, which is defined at
 *             power_tiling.h. Generally, the more space you allocate, the better performance you will achieve,
 *             and the performance reaches peak when buffer size is maximum(calculated by tiling function). Moreover, it
 *             is not guaranteed that the shared space will be cleared after usage, the data could be anything.
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Power(const LocalTensor<T>& dstTensor, const T& src0Scalar, const LocalTensor<T>& src1Tensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer)
{
    PowerCommonImpl<T, isReuseSource>(dstTensor, src0Scalar, src1Tensor, sharedTmpBuffer, src1Tensor.GetSize());
}
/*!
 * \ingroup Power
 * \brief compute Power elementwisely.
 * \tparam T: input dataType, support half/float
 * \tparam isReuseSource: whether allows API to modify source data, usually for performance reason,
 *         not enabled currently.
 * \param [out] dstTensor: output LocalTensor
 * \param [in] src0Scalar: base Scalar
 * \param [in] src1Tensor: exponent LocalTensor
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Power(const LocalTensor<T>& dstTensor, const T& src0Scalar, const LocalTensor<T>& src1Tensor)
{
    PowerCommonImpl<T, isReuseSource>(dstTensor, src0Scalar, src1Tensor, src1Tensor.GetSize());
}
#pragma end_pipe
}  // namespace AscendC
#endif
#endif  // LIB_MATH_POWER_H
