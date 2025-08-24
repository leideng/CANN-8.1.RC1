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
template <typename T>
__aicore__ inline void ReluIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, half, float, int32_t>(), "Failed to check dtype in Relu, current api support dtype "
        "combination is src and dst both: half / float / int32_t.");
    vrelu(dst, src, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

/* **************************************** Exp ****************************************** */
template <typename T>
__aicore__ inline void ExpIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, half, float>(), "Failed to check dtype in Exp, current api support dtype combination "
        "is src and dst both: half / float.");
    vexp(dst, src, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

/* **************************************** Ln ****************************************** */
template <typename T>
__aicore__ inline void LnIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, half, float>(), "Failed to check dtype in Ln, current api support dtype combination "
        "is src and dst both: half / float.");
    vln(dst, src, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

/* **************************************** Abs ****************************************** */
template <typename T>
__aicore__ inline void AbsIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, half, float>(), "Failed to check dtype in Abs, current api support dtype combination "
        "is src and dst both: half / float.");
    vabs(dst, src, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

/* **************************************** Reciprocal ****************************************** */
template <typename T>
__aicore__ inline void ReciprocalIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, half, float>(), "Failed to check dtype in Reciprocal, current api support dtype "
        "combination is src and dst both: half / float.");
    vrec(dst, src, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

/* **************************************** Rsqrt ****************************************** */
template <typename T>
__aicore__ inline void RsqrtIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, half, float>(), "Failed to check dtype in Rsqrt, current api support dtype "
        "combination is src and dst both: half / float.");
    vrsqrt(dst, src, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

/* **************************************** Sqrt ****************************************** */
template <typename T>
__aicore__ inline void SqrtIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, half, float>(), "Failed to check dtype in Sqrt, current api support dtype "
        "combination is src and dst both: half / float.");
    vsqrt(dst, src, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

/* **************************************** Not ****************************************** */
template <typename T>
__aicore__ inline void NotIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, int16_t, uint16_t>(), "Failed to check dtype in Not, current api support dtype "
        "combination is src and dst both: int16_t / uint16_t.");
    vnot(dst, src, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

/* **************************************** Relu ****************************************** */
// Relu::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void ReluImpl(__ubuf__ T* dst, __ubuf__ T* src, const uint64_t mask[], const uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
    ReluIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void ReluImpl(__ubuf__ T* dst, __ubuf__ T* src, const uint64_t mask, const uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    AscendCUtils::SetMask<T, isSetMask>(mask);
    ReluIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

// Relu::Level 2
template <typename T> __aicore__ inline void ReluImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t& calCount)
{
    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(0, calCount);
    ReluIntrinsicsImpl(dst, src, 1, {DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE,
        DEFAULT_REPEAT_STRIDE});
    SetMaskNorm();
    set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
}

/* **************************************** Exp ****************************************** */
// Exp::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void ExpImpl(__ubuf__ T* dst, __ubuf__ T* src, const uint64_t mask[], const uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
    ExpIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void ExpImpl(__ubuf__ T* dst, __ubuf__ T* src, const uint64_t mask, const uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    AscendCUtils::SetMask<T, isSetMask>(mask);
    ExpIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

// Exp::Level 2
template <typename T> __aicore__ inline void ExpImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t& calCount)
{
    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(0, calCount);
    ExpIntrinsicsImpl(dst, src, 1, {DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE,
        DEFAULT_REPEAT_STRIDE});
    SetMaskNorm();
    set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
}

/* **************************************** Ln ****************************************** */
// Ln::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void LnImpl(__ubuf__ T* dst, __ubuf__ T* src, const uint64_t mask[], const uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
    LnIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void LnImpl(__ubuf__ T* dst, __ubuf__ T* src, const uint64_t mask, const uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    AscendCUtils::SetMask<T, isSetMask>(mask);
    LnIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

// Ln::Level 2
template <typename T> __aicore__ inline void LnImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t& calCount)
{
    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(0, calCount);
    LnIntrinsicsImpl(dst, src, 1, {DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE,
        DEFAULT_REPEAT_STRIDE});
    SetMaskNorm();
    set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
}

/* **************************************** Abs ****************************************** */
// Abs::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void AbsImpl(__ubuf__ T* dst, __ubuf__ T* src, const uint64_t mask[], const uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
    AbsIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void AbsImpl(__ubuf__ T* dst, __ubuf__ T* src, const uint64_t mask, const uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    AscendCUtils::SetMask<T, isSetMask>(mask);
    AbsIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

// Ln::Level 2
template <typename T> __aicore__ inline void AbsImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t& calCount)
{
    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(0, calCount);
    AbsIntrinsicsImpl(dst, src, 1, {DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE,
        DEFAULT_REPEAT_STRIDE});
    SetMaskNorm();
    set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
}

/* **************************************** Reciprocal ****************************************** */
// Reciprocal::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void ReciprocalImpl(__ubuf__ T* dst, __ubuf__ T* src, const uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
    ReciprocalIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void ReciprocalImpl(__ubuf__ T* dst, __ubuf__ T* src, const uint64_t mask, const uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    AscendCUtils::SetMask<T, isSetMask>(mask);
    ReciprocalIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

// Reciprocal::Level 2
template <typename T> __aicore__ inline void ReciprocalImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t& calCount)
{
    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(0, calCount);
    ReciprocalIntrinsicsImpl(dst, src, 1, {DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE,
        DEFAULT_REPEAT_STRIDE});
    SetMaskNorm();
    set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
}

/* **************************************** Rsqrt ****************************************** */
// Rsqrt::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void RsqrtImpl(__ubuf__ T* dst, __ubuf__ T* src, const uint64_t mask[], const uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
    RsqrtIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void RsqrtImpl(__ubuf__ T* dst, __ubuf__ T* src, const uint64_t mask, const uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    AscendCUtils::SetMask<T, isSetMask>(mask);
    RsqrtIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

// Rsqrt::Level 2
template <typename T> __aicore__ inline void RsqrtImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t& calCount)
{
    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(0, calCount);
    RsqrtIntrinsicsImpl(dst, src, 1, {DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE,
        DEFAULT_REPEAT_STRIDE});
    SetMaskNorm();
    set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
}

/* **************************************** Sqrt ****************************************** */
// Sqrt::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void SqrtImpl(__ubuf__ T* dst, __ubuf__ T* src, const uint64_t mask[], const uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
    SqrtIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void SqrtImpl(__ubuf__ T* dst, __ubuf__ T* src, const uint64_t mask, const uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    AscendCUtils::SetMask<T, isSetMask>(mask);
    SqrtIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

// Rsqrt::Level 2
template <typename T> __aicore__ inline void SqrtImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t& calCount)
{
    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(0, calCount);
    SqrtIntrinsicsImpl(dst, src, 1, {DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE,
        DEFAULT_REPEAT_STRIDE});
    SetMaskNorm();
    set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
}

/* **************************************** Not ****************************************** */
// Not::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void NotImpl(__ubuf__ T* dst, __ubuf__ T* src, const uint64_t mask[], const uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
    NotIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void NotImpl(__ubuf__ T* dst, __ubuf__ T* src, const uint64_t mask, const uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    AscendCUtils::SetMask<T, isSetMask>(mask);
    NotIntrinsicsImpl(dst, src, repeatTimes, repeatParams);
}

// Not::Level 2
template <typename T> __aicore__ inline void NotImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t& calCount)
{
    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(0, calCount);
    NotIntrinsicsImpl(dst, src, 1, {DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE,
        DEFAULT_REPEAT_STRIDE});
    SetMaskNorm();
    set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_UNARY_IMPL_H
