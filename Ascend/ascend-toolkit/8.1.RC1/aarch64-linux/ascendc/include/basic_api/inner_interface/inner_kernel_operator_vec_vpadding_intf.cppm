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
 * \file inner_kernel_operator_vec_vpadding_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_INNER_OPERATOR_VEC_VPADDING_INTERFACE_H
#define ASCENDC_MODULE_INNER_OPERATOR_VEC_VPADDING_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_check.h"
#include "kernel_struct_unary.h"

#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_vec_vpadding_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_vec_vpadding_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_vec_vpadding_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_vec_vpadding_impl.h"
#elif defined(__DAV_M310__)
#include "dav_m310/kernel_operator_vec_vpadding_impl.h"
#endif

#pragma begin_pipe(V)
namespace AscendC {
template <typename T, bool isSetMask>
__aicore__ inline void VectorPadding(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const uint8_t padMode, const bool padSide, const uint64_t mask, const uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckVectorPadding(dstLocal, srcLocal, padMode, padSide, mask, repeatTimes, repeatParams, "VectorPadding")) {
        ASCENDC_REPORT_CHECK_ERROR("VectorPadding", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    VectorPaddingImpl<T, isSetMask>((__ubuf__ T *)dstLocal.GetPhyAddr(), (__ubuf__ T *)srcLocal.GetPhyAddr(), padMode,
        padSide, mask, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask>
__aicore__ inline void VectorPadding(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const uint8_t padMode, const bool padSide, const uint64_t mask[], const uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckVectorPadding(dstLocal, srcLocal, padMode, padSide, mask, repeatTimes, repeatParams, "VectorPadding")) {
        ASCENDC_REPORT_CHECK_ERROR("VectorPadding", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    VectorPaddingImpl<T, isSetMask>((__ubuf__ T *)dstLocal.GetPhyAddr(), (__ubuf__ T *)srcLocal.GetPhyAddr(), padMode,
        padSide, mask, repeatTimes, repeatParams);
}

template <typename T>
__aicore__ inline void VectorPadding(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const uint8_t padMode, const bool padSide, const uint32_t calCount)
{
#if ASCENDC_CPU_DEBUG
    if (!CheckVectorPadding(dstLocal, srcLocal, padMode, padSide, calCount, "VectorPadding")) {
        ASCENDC_REPORT_CHECK_ERROR("VectorPadding", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    VectorPaddingImpl((__ubuf__ T *)dstLocal.GetPhyAddr(), (__ubuf__ T *)srcLocal.GetPhyAddr(), padMode, padSide,
        calCount);
}
} // namespace AscendC
#pragma end_pipe
#endif // ASCENDC_MODULE_INNER_OPERATOR_VEC_VPADDING_INTERFACE_H