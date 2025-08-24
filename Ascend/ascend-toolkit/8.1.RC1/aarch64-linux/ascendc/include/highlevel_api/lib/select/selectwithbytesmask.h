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
 * \file selectwithbytesmask.h
 * \brief
 */
#ifndef LIB_SELECT_SELECT_WITH_BYTES_MASK_H
#define LIB_SELECT_SELECT_WITH_BYTES_MASK_H
#if __CCE_AICORE__ == 220 || __CCE_AICORE__ == 200
#include "kernel_tensor.h"
#include "kernel_utils.h"
#include "../../impl/select/selectwithbytesmask/selectwithbytesmask_impl.h"

namespace AscendC {
#pragma begin_pipe(V)
/*!
 * \ingroup SelectWithBytesMask
 * \brief Selects values from two sources(tensor/scalar) and put into dst tensor elementwisely according to the mask
 * Tensor Value.
 * True: mask tensor position value == 1, select src1 scalar.
 * False: mask tensor position value == 0, select src0 Tensor correspoding value.
 * \note support data type: T half/float, U uint8_t/int8_t/uint16_t/int16_t/uint32_t/int32_t/bool
 * Tensor shape can be considered as [first_axis, last_axis],
 * and it requries that mask_first_axis = source_first_axis, mask_last_axis >= source_first_axis
 * mask last axis should be 32B aligned and element size should be time of 16
 * src last axis should be 32B aligned
 * it doesn't allow src/sharedTmpBuffer tensor address overlap.
 * \tparam isReuseMask: whether to reuse the mask space for process
 * \param [out] dst: output LocalTensor
 * \param [in] src0: input LocalTensor
 * \param [in] src1: input scalar
 * \param [in] mask: mask LocalTensor
 * \param [in] sharedTmpBuffer: extra tmp buffer used as intermediate values among calculation process.
 * \param [in] info: shape information of input/mask tensors
 */
template <typename T, typename U, bool isReuseMask = true>
__aicore__ inline void SelectWithBytesMask(const LocalTensor<T> &dst, const LocalTensor<T> &src0, T src1,
    const LocalTensor<U> &mask, const LocalTensor<uint8_t> &sharedTmpBuffer, const SelectWithBytesMaskShapeInfo &info)
{
    SelectWithBytesMaskImpl<T, U, isReuseMask, false>(dst, src0, src1, mask, sharedTmpBuffer, info);
}

/*!
 * \ingroup SelectWithBytesMask
 * \brief Selects values from two sources(tensor/scalar) and put into dst tensor elementwisely according to the mask
 * Tensor Value.
 * True: mask tensor position value == 1, select src1 Tensor correspoding value.
 * False: mask tensor position value == 0, select src0 scalar.
 * \note support data type: T half/float, U uint8_t/int8_t/uint16_t/int16_t/uint32_t/int32_t/bool
 * Tensor shape can be considered as [first_axis, last_axis],
 * and it requries that mask_first_axis = source_first_axis, mask_last_axis >= source_first_axis
 * mask last axis should be 32B aligned and element size should be time of 16
 * src last axis should be 32B aligned
 * it doesn't allow src/sharedTmpBuffer tensor address overlap.
 * \tparam isReuseMask: whether to reuse the mask space for process
 * \param [out] dst: output LocalTensor
 * \param [in] src0: input scalar
 * \param [in] src1: input LocalTensor
 * \param [in] mask: mask LocalTensor
 * \param [in] sharedTmpBuffer: extra tmp buffer used as intermediate values among calculation process.
 * \param [in] info: shape information of input/mask tensors
 */
template <typename T, typename U, bool isReuseMask = true>
__aicore__ inline void SelectWithBytesMask(const LocalTensor<T> &dst, T src0, const LocalTensor<T> &src1,
    const LocalTensor<U> &mask, const LocalTensor<uint8_t> &sharedTmpBuffer, const SelectWithBytesMaskShapeInfo &info)
{
    SelectWithBytesMaskImpl<T, U, isReuseMask, true>(dst, src1, src0, mask, sharedTmpBuffer, info);
}
#pragma end_pipe
} // namespace AscendC
#endif
#endif // LIB_SELECT_SELECT_WITH_BYTES_MASK_H
