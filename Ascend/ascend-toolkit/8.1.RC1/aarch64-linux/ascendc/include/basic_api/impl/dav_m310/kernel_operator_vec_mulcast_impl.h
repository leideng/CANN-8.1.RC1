/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * \file kernel_operator_vec_mulcast_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_MULCAST_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_MULCAST_IMPL_H
#include "kernel_tensor.h"
#include "kernel_struct_binary.h"
#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif

#pragma begin_pipe(V)
namespace AscendC {
template <typename T, typename U>
__aicore__ inline void MulCastCalc(const LocalTensor<T> &dstLocal, const LocalTensor<U> &src0Local,
    const LocalTensor<U> &src1Local, uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current device don't support MulCast"); });
}

template <typename T, typename U>
__aicore__ inline void MulCastCalc(const LocalTensor<T> &dstLocal, const LocalTensor<U> &src0Local,
    const LocalTensor<U> &src1Local, uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams &repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current device don't support MulCast"); });
}

template <typename T, typename U>
__aicore__ inline void MulCastCalc(const LocalTensor<T> &dstLocal, const LocalTensor<U> &src0Local,
    const LocalTensor<U> &src1Local, uint32_t calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current device don't support MulCast"); });
}
} // namespace AscendC
#pragma end_pipe
#endif // ASCENDC_MODULE_OPERATOR_VEC_MULCAST_IMPL_H