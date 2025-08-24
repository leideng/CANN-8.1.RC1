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
 * \file inner_kernel_operator_vec_mulcast_intf.cppm
 * \brief
 */
#ifndef ASCENDC_MODULE_INNER_OPERATOR_VEC_MULCAST_INTERFACE_H
#define ASCENDC_MODULE_INNER_OPERATOR_VEC_MULCAST_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_check.h"
#include "kernel_struct_binary.h"

#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_vec_mulcast_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_vec_mulcast_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_vec_mulcast_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_vec_mulcast_impl.h"
#elif defined(__DAV_M310__)
#include "dav_m310/kernel_operator_vec_mulcast_impl.h"
#endif

#pragma begin_pipe(V)
namespace AscendC {
template <typename T, typename U>
__aicore__ inline void MulCast(const LocalTensor<T> &dstLocal, const LocalTensor<U> &src0Local,
    const LocalTensor<U> &src1Local, uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    ASCENDC_ASSERT((SupportType<U, half>() && SupportType<T, int8_t, uint8_t>()), {KERNEL_LOG(KERNEL_ERROR, "Failed "
        "to check dtype in MulCast, current api support dtype combination is src: half, dst: int8_t / uint8_t");});
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncVecBinaryDiffType(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "MulCast")) {
        ASCENDC_REPORT_CHECK_ERROR("MulCast", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    MulCastCalc(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams);
}

template <typename T, typename U>
__aicore__ inline void MulCast(const LocalTensor<T> &dstLocal, const LocalTensor<U> &src0Local,
    const LocalTensor<U> &src1Local, uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams &repeatParams)
{
    ASCENDC_ASSERT((SupportType<U, half>() && SupportType<T, int8_t, uint8_t>()), {KERNEL_LOG(KERNEL_ERROR, "Failed "
        "to check dtype in MulCast, current api support dtype combination is src: half, dst: int8_t / uint8_t");});
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncVecBinaryDiffType(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "MulCast")) {
        ASCENDC_REPORT_CHECK_ERROR("MulCast", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    MulCastCalc(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams);
}

template <typename T, typename U>
__aicore__ inline void MulCast(const LocalTensor<T> &dstLocal, const LocalTensor<U> &src0Local,
    const LocalTensor<U> &src1Local, uint32_t calCount)
{
    ASCENDC_ASSERT((SupportType<U, half>() && SupportType<T, int8_t, uint8_t>()), {KERNEL_LOG(KERNEL_ERROR, "Failed "
        "to check dtype in MulCast, current api support dtype combination is src: half, dst: int8_t / uint8_t");});
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncVecBinaryDiffType(dstLocal, src0Local, src1Local, calCount, "MulCast")) {
        ASCENDC_REPORT_CHECK_ERROR("MulCast", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    MulCastCalc(dstLocal, src0Local, src1Local, calCount);
}
} // namespace AscendC
#pragma end_pipe
#endif // ASCENDC_MODULE_INNER_OPERATOR_VEC_MULCAST_INTERFACE_H