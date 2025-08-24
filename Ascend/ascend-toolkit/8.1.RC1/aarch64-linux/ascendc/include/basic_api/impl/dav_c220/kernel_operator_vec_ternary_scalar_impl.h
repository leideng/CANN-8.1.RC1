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
 * \file kernel_operator_vec_ternary_scalar_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_TERNARY_SCALAR_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_TERNARY_SCALAR_IMPL_H
#include "kernel_struct_unary.h"

namespace AscendC {

template <typename T, typename U>
__aicore__ inline void AxpyIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ U* src, U scalarValue, const uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT((SupportType<Tuple<T, U>, Tuple<half, half>, Tuple<float, float>, Tuple<float, half>>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Axpy, current api support dtype combination is src: half, "
        "dst: half / float; src: float, dst: float.");});
    vaxpy(dst, src, scalarValue, repeatTimes, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
        repeatParams.dstRepStride, repeatParams.srcRepStride);
}

// Axpy::Level 0 - mask count mode
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void AxpyImpl(__ubuf__ T* dst, __ubuf__ U* src, const U& scalarValue, const uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        if constexpr (isSetMask) {
            if (sizeof(T) > sizeof(U)) {
                AscendCUtils::SetMask<T>(mask);
            } else {
                AscendCUtils::SetMask<U>(mask);
            }
        }
        AxpyIntrinsicsImpl(dst, src, scalarValue, repeatTimes, repeatParams);
    }
}

// Axpy::Level 0 - mask bit mode
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void AxpyImpl(__ubuf__ T* dst, __ubuf__ U* src, const U& scalarValue, const uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
        AxpyIntrinsicsImpl(dst, src, scalarValue, repeatTimes, repeatParams);
    }
}

// Axpy::Level 2
template <typename T, typename U>
__aicore__ inline void AxpyImpl(__ubuf__ T* dst, __ubuf__ U* src, const U& scalarValue, const int32_t& calCount)
{
    if ASCEND_IS_AIV {
        SetMaskCount();
        AscendCUtils::SetMask<U>(0, calCount);
        if constexpr (sizeof(T) > sizeof(U)) {
            AxpyIntrinsicsImpl(dst, src, scalarValue, 1,
                { DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE / 2 });
        } else {
            AxpyIntrinsicsImpl(dst, src, scalarValue, 1,
                { DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        }
        ResetMask();
        SetMaskNorm();
    }
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_TERNARY_SCALAR_IMPL_H
