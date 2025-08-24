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
__aicore__ inline void MulCastIntrinsicsImpl(const LocalTensor<T> &dstLocal, const LocalTensor<U> &src0Local,
    const LocalTensor<U> &src1Local, const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    if constexpr (IsSameType<T, int8_t>::value) {
        vmulconv_f162s8((__ubuf__ int8_t *)dstLocal.GetPhyAddr(), (__ubuf__ half *)src0Local.GetPhyAddr(),
            (__ubuf__ half *)src1Local.GetPhyAddr(), repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
            repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride,
            repeatParams.src1RepStride);
    } else {
        vmulconv_f162u8((__ubuf__ uint8_t *)dstLocal.GetPhyAddr(), (__ubuf__ half *)src0Local.GetPhyAddr(),
            (__ubuf__ half *)src1Local.GetPhyAddr(), repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
            repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride,
            repeatParams.src1RepStride);
    }
}

template <typename T, typename U>
__aicore__ inline void MulCastCalc(const LocalTensor<T> &dstLocal, const LocalTensor<U> &src0Local,
    const LocalTensor<U> &src1Local, uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<U>(mask);
        MulCastIntrinsicsImpl(dstLocal, src0Local, src1Local, repeatTimes, repeatParams);
    }
}

template <typename T, typename U>
__aicore__ inline void MulCastCalc(const LocalTensor<T> &dstLocal, const LocalTensor<U> &src0Local,
    const LocalTensor<U> &src1Local, uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams &repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T>(mask[1], mask[0]);
        MulCastIntrinsicsImpl(dstLocal, src0Local, src1Local, repeatTimes, repeatParams);
    }
}

template <typename T, typename U>
__aicore__ inline void MulCastCalc(const LocalTensor<T> &dstLocal, const LocalTensor<U> &src0Local,
    const LocalTensor<U> &src1Local, uint32_t calCount)
{
    if ASCEND_IS_AIV {
        BinaryRepeatParams repeatParams;
        repeatParams.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;
        set_mask_count();
        set_vector_mask(0, calCount);
        MulCastIntrinsicsImpl(dstLocal, src0Local, src1Local, 1, repeatParams);
        set_mask_norm();
        ResetMask();
    }
}
} // namespace AscendC
#pragma end_pipe
#endif // ASCENDC_MODULE_OPERATOR_VEC_MULCAST_IMPL_H