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
 * \file kernel_operator_vec_unary_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_UNARY_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_UNARY_IMPL_H
#include "kernel_struct_unary.h"

namespace AscendC {
/* **************************************** Relu ****************************************** */
__aicore__ inline void ReluIntrinsicsImpl(__ubuf__ half* dst, __ubuf__ half* src, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    vrelu(dst, src, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

__aicore__ inline void ReluIntrinsicsImpl(__ubuf__ float* dst, __ubuf__ float* src, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    vrelu(dst, src, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

__aicore__ inline void ReluIntrinsicsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    vrelu(dst, src, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

/* **************************************** Exp ****************************************** */
__aicore__ inline void ExpIntrinsicsImpl(__ubuf__ half* dst, __ubuf__ half* src, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    vexp(dst, src, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

__aicore__ inline void ExpIntrinsicsImpl(__ubuf__ float* dst, __ubuf__ float* src, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    vexp(dst, src, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

/* **************************************** Ln ****************************************** */
__aicore__ inline void LnIntrinsicsImpl(__ubuf__ half* dst, __ubuf__ half* src, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    vln(dst, src, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

__aicore__ inline void LnIntrinsicsImpl(__ubuf__ float* dst, __ubuf__ float* src, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    vln(dst, src, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

/* **************************************** Abs ****************************************** */
__aicore__ inline void AbsIntrinsicsImpl(__ubuf__ half* dst, __ubuf__ half* src, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    vabs(dst, src, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

__aicore__ inline void AbsIntrinsicsImpl(__ubuf__ float* dst, __ubuf__ float* src, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    vabs(dst, src, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

__aicore__ inline void AbsIntrinsicsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    vabs(dst, src, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint16_t)repeatParams.dstRepStride, (uint16_t)repeatParams.srcRepStride);
}

/* **************************************** Reciprocal ****************************************** */
__aicore__ inline void ReciprocalIntrinsicsImpl(__ubuf__ half* dst, __ubuf__ half* src, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    vrec(dst, src, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

__aicore__ inline void ReciprocalIntrinsicsImpl(__ubuf__ float* dst, __ubuf__ float* src, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    vrec(dst, src, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

/* **************************************** Rsqrt ****************************************** */
__aicore__ inline void RsqrtIntrinsicsImpl(__ubuf__ half* dst, __ubuf__ half* src, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    vrsqrt(dst, src, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

__aicore__ inline void RsqrtIntrinsicsImpl(__ubuf__ float* dst, __ubuf__ float* src, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    vrsqrt(dst, src, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

/* **************************************** Sqrt ****************************************** */
__aicore__ inline void SqrtIntrinsicsImpl(__ubuf__ half* dst, __ubuf__ half* src, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    vsqrt(dst, src, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

__aicore__ inline void SqrtIntrinsicsImpl(__ubuf__ float* dst, __ubuf__ float* src, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    vsqrt(dst, src, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

/* **************************************** Not ****************************************** */
__aicore__ inline void NotIntrinsicsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    vnot(dst, src, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

__aicore__ inline void NotIntrinsicsImpl(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    vnot(dst, src, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

template <typename T>
__aicore__ inline void VecUnaryCompute(__ubuf__ T* dst, __ubuf__ T* src, const int32_t& calCount,
    void (*func)(__ubuf__ T*, __ubuf__ T*, uint64_t, uint8_t, const UnaryRepeatParams&))
{
    struct UnaryRepeatParams repeatParams;

    IntriInfo intriInfo = AscendCUtils::CalIntriInfo(sizeof(T), calCount);

    int32_t dstOffset = 0, srcOffset = 0;
    const auto dstOffsetCount = MAX_REPEAT_TIMES * repeatParams.dstRepStride * intriInfo.c0Count;
    const auto srcOffsetCount = MAX_REPEAT_TIMES * repeatParams.srcRepStride * intriInfo.c0Count;
    const int32_t fullMask = intriInfo.c0Count * DEFAULT_BLK_NUM;
    for (int32_t i = 0; i < intriInfo.repeatRounding; i++) {
        func((__ubuf__ T*)(dst + dstOffset), (__ubuf__ T*)(src + srcOffset), fullMask, MAX_REPEAT_TIMES, repeatParams);
        dstOffset += dstOffsetCount;
        srcOffset += srcOffsetCount;
    }

    dstOffset = (intriInfo.repeatRounding * MAX_REPEAT_TIMES) * repeatParams.dstRepStride * intriInfo.c0Count;
    srcOffset = (intriInfo.repeatRounding * MAX_REPEAT_TIMES) * repeatParams.srcRepStride * intriInfo.c0Count;

    if (intriInfo.repeatRemaining != 0) {
        func((__ubuf__ T*)(dst + dstOffset), (__ubuf__ T*)(src + srcOffset), fullMask, intriInfo.repeatRemaining,
            repeatParams);
    }

    if (intriInfo.tail != 0) {
        dstOffset = intriInfo.repeat * repeatParams.dstRepStride * intriInfo.c0Count;
        srcOffset = intriInfo.repeat * repeatParams.srcRepStride * intriInfo.c0Count;
        func((__ubuf__ T*)(dst + dstOffset), (__ubuf__ T*)(src + srcOffset), intriInfo.tail, 1, repeatParams);
    }
}

/* **************************************** Relu ****************************************** */
// Relu::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void ReluImpl(__ubuf__ T* dst, __ubuf__ T* src, uint64_t mask[], uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask[1], mask[0]);
    }
    ReluIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void ReluImpl(__ubuf__ T* dst, __ubuf__ T* src, uint64_t mask, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask);
    }
    ReluIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

// Relu::Level 2
template <typename T> __aicore__ inline void ReluImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t& calCount)
{
    VecUnaryCompute(dst, src, calCount, ReluImpl<T>);
}

/* **************************************** Exp ****************************************** */
// Exp::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void ExpImpl(__ubuf__ T* dst, __ubuf__ T* src, const uint64_t mask[], const uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask[1], mask[0]);
    }
    ExpIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void ExpImpl(__ubuf__ T* dst, __ubuf__ T* src, const uint64_t mask, const uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask);
    }
    ExpIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

// Exp::Level 2
template <typename T> __aicore__ inline void ExpImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t& calCount)
{
    VecUnaryCompute(dst, src, calCount, ExpImpl<T>);
}

/* **************************************** Ln ****************************************** */
// Ln::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void LnImpl(__ubuf__ T* dst, __ubuf__ T* src, uint64_t mask[], uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask[1], mask[0]);
    }
    LnIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void LnImpl(__ubuf__ T* dst, __ubuf__ T* src, uint64_t mask, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask);
    }
    LnIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

// Ln::Level 2
template <typename T> __aicore__ inline void LnImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t& calCount)
{
    VecUnaryCompute(dst, src, calCount, LnImpl<T>);
}

/* **************************************** Abs ****************************************** */
// Abs::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void AbsImpl(__ubuf__ T* dst, __ubuf__ T* src, uint64_t mask[], uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask[1], mask[0]);
    }
    AbsIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void AbsImpl(__ubuf__ T* dst, __ubuf__ T* src, uint64_t mask, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask);
    }
    AbsIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

// Ln::Level 2
template <typename T> __aicore__ inline void AbsImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t& calCount)
{
    VecUnaryCompute(dst, src, calCount, AbsImpl<T>);
}

/* **************************************** Reciprocal ****************************************** */
// Reciprocal::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void ReciprocalImpl(__ubuf__ T* dst, __ubuf__ T* src, uint64_t mask[], uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask[1], mask[0]);
    }
    ReciprocalIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void ReciprocalImpl(__ubuf__ T* dst, __ubuf__ T* src, uint64_t mask, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask);
    }
    ReciprocalIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

// Reciprocal::Level 2
template <typename T> __aicore__ inline void ReciprocalImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t& calCount)
{
    VecUnaryCompute(dst, src, calCount, ReciprocalImpl<T>);
}

/* **************************************** Rsqrt ****************************************** */
// Rsqrt::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void RsqrtImpl(__ubuf__ T* dst, __ubuf__ T* src, uint64_t mask[], uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask[1], mask[0]);
    }
    RsqrtIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void RsqrtImpl(__ubuf__ T* dst, __ubuf__ T* src, uint64_t mask, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask);
    }
    RsqrtIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

// Rsqrt::Level 2
template <typename T> __aicore__ inline void RsqrtImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t& calCount)
{
    VecUnaryCompute(dst, src, calCount, RsqrtImpl<T>);
}

/* **************************************** Sqrt ****************************************** */
// Sqrt::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void SqrtImpl(__ubuf__ T* dst, __ubuf__ T* src, uint64_t mask[], uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask[1], mask[0]);
    }
    SqrtIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void SqrtImpl(__ubuf__ T* dst, __ubuf__ T* src, uint64_t mask, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask);
    }
    SqrtIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

// Rsqrt::Level 2
template <typename T> __aicore__ inline void SqrtImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t& calCount)
{
    VecUnaryCompute(dst, src, calCount, SqrtImpl<T>);
}

/* **************************************** Not ****************************************** */
// Not::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void NotImpl(__ubuf__ T* dst, __ubuf__ T* src, uint64_t mask[], uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask[1], mask[0]);
    }
    NotIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void NotImpl(__ubuf__ T* dst, __ubuf__ T* src, uint64_t mask, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask);
    }
    NotIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

// Not::Level 2
template <typename T> __aicore__ inline void NotImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t& calCount)
{
    VecUnaryCompute(dst, src, calCount, NotImpl<T>);
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_UNARY_IMPL_H
