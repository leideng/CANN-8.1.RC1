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
__aicore__ inline void AxpyIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ U* src, half scalarValue, const uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    vaxpy(dst, src, scalarValue, repeatTimes, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
        repeatParams.dstRepStride, repeatParams.srcRepStride);
}
template <typename T, typename U>
__aicore__ inline void AxpyIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ U* src, float scalarValue,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    vaxpy(dst, src, scalarValue, repeatTimes, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
        repeatParams.dstRepStride, repeatParams.srcRepStride);
}

// Axpy::Level 0 - mask count mode
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void AxpyImpl(__ubuf__ T* dst, __ubuf__ U* src, const U& scalarValue, const uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        if (sizeof(T) > sizeof(U)) {
            AscendCUtils::SetMask<T>(mask);
        } else {
            AscendCUtils::SetMask<U>(mask);
        }
    }
    AxpyIntrinsicsImpl(dst, src, scalarValue, repeatTimes, repeatParams);
}

// Axpy::Level 0 - mask bit mode
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void AxpyImpl(__ubuf__ T* dst, __ubuf__ U* src, const U& scalarValue, const uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask[1], mask[0]);
    }
    AxpyIntrinsicsImpl(dst, src, scalarValue, repeatTimes, repeatParams);
}

// Axpy::Level 2
template <typename T, typename U>
__aicore__ inline void AxpyImpl(__ubuf__ T* dst, __ubuf__ U* src, const U& scalarValue, const int32_t& calCount)
{
    const int32_t typeLen = (sizeof(T) > sizeof(U)) ? sizeof(T) : sizeof(U);
    const int32_t oneRepeatNum = DEFAULT_BLOCK_SIZE / typeLen;
    IntriInfo intriInfo = AscendCUtils::CalIntriInfo(typeLen, calCount);

    struct UnaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.srcBlkStride = 1;
    repeatParams.dstRepStride = oneRepeatNum / (ONE_BLK_SIZE / sizeof(T));
    repeatParams.srcRepStride = oneRepeatNum / (ONE_BLK_SIZE / sizeof(U));
    int32_t dstOffset = 0, srcOffset = 0;
    const int32_t dstOffsetCount = MAX_REPEAT_TIMES * oneRepeatNum;
    const int32_t srcOffsetCount = MAX_REPEAT_TIMES * oneRepeatNum;
    for (int32_t i = 0; i < intriInfo.repeatRounding; i++) {
        AxpyImpl((__ubuf__ T*)(dst + dstOffset), (__ubuf__ U*)(src + srcOffset), scalarValue, oneRepeatNum,
            MAX_REPEAT_TIMES, repeatParams);
        dstOffset += i * dstOffsetCount;
        srcOffset += i * srcOffsetCount;
    }

    dstOffset = intriInfo.repeatRounding * MAX_REPEAT_TIMES * oneRepeatNum;
    srcOffset = intriInfo.repeatRounding * MAX_REPEAT_TIMES * oneRepeatNum;

    if (intriInfo.repeatRemaining != 0) {
        AxpyImpl((__ubuf__ T*)(dst + dstOffset), (__ubuf__ U*)(src + srcOffset), scalarValue, oneRepeatNum,
            intriInfo.repeatRemaining, repeatParams);
    }

    if (intriInfo.tail != 0) {
        dstOffset += intriInfo.repeatRemaining * oneRepeatNum;
        srcOffset += intriInfo.repeatRemaining * oneRepeatNum;
        AxpyImpl((__ubuf__ T*)(dst + dstOffset), (__ubuf__ U*)(src + srcOffset), scalarValue, intriInfo.tail, 1,
            repeatParams);
    }
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_TERNARY_SCALAR_IMPL_H
