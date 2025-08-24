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
 * \file atanh.h
 * \brief
 */

#ifndef LIB_MATH_ATANH_INTERFACE_H
#define LIB_MATH_ATANH_INTERFACE_H
#include "kernel_tensor.h"
#include "../../impl/math/atanh/atanh_common_impl.h"

namespace AscendC {
#pragma begin_pipe(V)

/*!
 * \brief This function implements the atanh(x), which has the formula
 * atanh(x) = 0.5 * ln((1 + x) / (1 - x)) x:(-1, 1)
 * (e.g. atanh(0.76159416) is 1.00000001).
 * For details about the interface description, see https://pytorch.org/docs/stable/generated/torch.atanh.html.
 *
 * \note support data type: half and float
 *
 * \param [out] dstTensor, output LocalTensor
 * \param [in] srcTensor, input LocalTensor
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 * \param [in] calCount, amount of data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Atanh(const LocalTensor<T> &dstTensor,
    const LocalTensor<T> &srcTensor,
    const LocalTensor<uint8_t> &sharedTmpBuffer,
    const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    AtanhImpl(dstTensor, srcTensor, sharedTmpBuffer, calCount);
}

/* !
 * \ingroup atanh
 * \note support data type: half and float
 * \param [out] dstTensor, output LocalTensor
 * \param [in] srcTensor, input LocalTensor
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Atanh(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor)
{
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ret = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ret), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });

    Atanh<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, srcTensor.GetSize());
}

/* !
 * \ingroup atanh
 * \note support data type: half and float
 * \param [out] dstTensor, output LocalTensor
 * \param [in] srcTensor, input LocalTensor
 * \param [in] sharedTmpBuffer, input local temporary Tensor
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Atanh(const LocalTensor<T> &dstTensor,
    const LocalTensor<T> &srcTensor,
    const LocalTensor<uint8_t> &sharedTmpBuffer)
{
    Atanh<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, srcTensor.GetSize());
}

/* !
 * \ingroup atanh
 * \note support data type: half and float
 * \param [out] dstTensor, output LocalTensor
 * \param [in] srcTensor, input LocalTensor
 * \param [in] calCount, amount of data to be calculated
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void Atanh(const LocalTensor<T> &dstTensor, const LocalTensor<T> &srcTensor, const uint32_t calCount)
{
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ret = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ret), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });

    Atanh<T, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, calCount);
}

#pragma end_pipe
} // namespace AscendC

#endif // LIB_MATH_ATANH_INTERFACE_H