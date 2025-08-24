/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file inner_kernel_operator_vec_ternary_scalar_intf.cppm
 * \brief
 */
#ifndef ASCENDC_MODULE_INNER_OPERATOR_VEC_TERNARY_SCALAR_INTERFACE_H
#define ASCENDC_MODULE_INNER_OPERATOR_VEC_TERNARY_SCALAR_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_struct_unary.h"

#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_vec_ternary_scalar_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_vec_ternary_scalar_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_vec_ternary_scalar_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_vec_ternary_scalar_impl.h"
#endif
#include "kernel_check.h"

#pragma begin_pipe(V)
namespace AscendC {
/*
 * @ingroup Axpy Level 0
 * @brief dst[i] = src[i]*scalar + dst[i]
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] scalarValue input scalar number
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.srcBlkStride src block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src repeat stride
 */
template <typename T, typename U, bool isSetMask>
__aicore__ inline void Axpy(const LocalTensor<T>& dstLocal, const LocalTensor<U>& srcLocal, const U& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalarDiffType(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "Axpy")) {
        ASCENDC_REPORT_CHECK_ERROR("Axpy", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    AxpyImpl<T, U, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ U*)srcLocal.GetPhyAddr(), scalarValue, mask,
        repeatTimes, repeatParams);
}

template <typename T, typename U, bool isSetMask>
__aicore__ inline void Axpy(const LocalTensor<T>& dstLocal, const LocalTensor<U>& srcLocal, const U& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalarDiffType(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "Axpy")) {
        ASCENDC_REPORT_CHECK_ERROR("Axpy", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    AxpyImpl<T, U, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ U*)srcLocal.GetPhyAddr(), scalarValue, mask,
        repeatTimes, repeatParams);
}

/*
 * @ingroup Axpy Level 2
 * @brief dst[i] = src[i]*scalar + dst[i]
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] scalarValue input scalar number
 * @param [in] calCount number Number of data involved in calculation
 */
template <typename T, typename U>
__aicore__ inline void Axpy(const LocalTensor<T>& dstLocal, const LocalTensor<U>& srcLocal, const U& scalarValue,
    const int32_t& calCount)
{
#if ASCENDC_CPU_DEBUG
    if (!CheckFunVecBinaryScalarDiffType(dstLocal, srcLocal, scalarValue, calCount, "Axpy")) {
        ASCENDC_REPORT_CHECK_ERROR("Axpy", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    AxpyImpl<T, U>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ U*)srcLocal.GetPhyAddr(), scalarValue, calCount);
}
} // namespace AscendC
#pragma end_pipe
#endif // ASCENDC_MODULE_INNER_OPERATOR_VEC_TERNARY_SCALAR_INTERFACE_H
