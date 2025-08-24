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
 * \file kernel_operator_vec_binary_scalar_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_BINARY_SCALAR_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_BINARY_SCALAR_IMPL_H
#include "kernel_struct_unary.h"

namespace AscendC {
template <typename T>
__aicore__ inline void VecBinaryScalarCompute(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue,
    const int32_t& calCount,
    void (*func)(__ubuf__ T*, __ubuf__ T*, const T&, const uint64_t, const uint8_t, const UnaryRepeatParams&))
{
    struct UnaryRepeatParams repeatParams;
    IntriInfo intriInfo = AscendCUtils::CalIntriInfo(sizeof(T), calCount);

    uint32_t dstOffset = 0;
    uint32_t srcOffset = 0;
    const auto dstOffsetCount = MAX_REPEAT_TIMES * repeatParams.dstRepStride * intriInfo.c0Count;
    const auto srcOffsetCount = MAX_REPEAT_TIMES * repeatParams.srcRepStride * intriInfo.c0Count;

    const int32_t fullMask = intriInfo.c0Count * DEFAULT_BLK_NUM;
    for (int32_t i = 0; i < intriInfo.repeatRounding; i++) {
        func((__ubuf__ T*)(dst + dstOffset), (__ubuf__ T*)(src + srcOffset), scalarValue, fullMask, MAX_REPEAT_TIMES,
            repeatParams);
        dstOffset += dstOffsetCount;
        srcOffset += srcOffsetCount;
    }

    dstOffset = (intriInfo.repeatRounding * MAX_REPEAT_TIMES) * repeatParams.dstRepStride * intriInfo.c0Count;
    srcOffset = (intriInfo.repeatRounding * MAX_REPEAT_TIMES) * repeatParams.srcRepStride * intriInfo.c0Count;

    if (intriInfo.repeatRemaining != 0) {
        func((__ubuf__ T*)(dst + dstOffset), (__ubuf__ T*)(src + srcOffset), scalarValue, fullMask,
            intriInfo.repeatRemaining, repeatParams);
    }

    if (intriInfo.tail != 0) {
        dstOffset = intriInfo.repeat * repeatParams.dstRepStride * intriInfo.c0Count;
        srcOffset = intriInfo.repeat * repeatParams.srcRepStride * intriInfo.c0Count;
        func((__ubuf__ T*)(dst + dstOffset), (__ubuf__ T*)(src + srcOffset), scalarValue, intriInfo.tail, 1,
            repeatParams);
    }
}
template <typename T>
__aicore__ inline void ShiftRightCompute(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue,
    const int32_t& calCount,
    void (*func)(__ubuf__ T*, __ubuf__ T*, const T&, const uint64_t, const uint8_t, const UnaryRepeatParams&, bool),
    bool roundEn)
{
    struct UnaryRepeatParams repeatParams;
    IntriInfo intriInfo = AscendCUtils::CalIntriInfo(sizeof(T), calCount);

    uint32_t dstOffset = 0;
    uint32_t srcOffset = 0;
    const auto dstOffsetCount = MAX_REPEAT_TIMES * repeatParams.dstRepStride * intriInfo.c0Count;
    const auto srcOffsetCount = MAX_REPEAT_TIMES * repeatParams.srcRepStride * intriInfo.c0Count;

    const int32_t fullMask = intriInfo.c0Count * DEFAULT_BLK_NUM;
    for (int32_t i = 0; i < intriInfo.repeatRounding; i++) {
        func((__ubuf__ T*)(dst + dstOffset), (__ubuf__ T*)(src + srcOffset), scalarValue, fullMask, MAX_REPEAT_TIMES,
            repeatParams, roundEn);
        dstOffset += dstOffsetCount;
        srcOffset += srcOffsetCount;
    }

    dstOffset = (intriInfo.repeatRounding * MAX_REPEAT_TIMES) * repeatParams.dstRepStride * intriInfo.c0Count;
    srcOffset = (intriInfo.repeatRounding * MAX_REPEAT_TIMES) * repeatParams.srcRepStride * intriInfo.c0Count;

    if (intriInfo.repeatRemaining != 0) {
        func((__ubuf__ T*)(dst + dstOffset), (__ubuf__ T*)(src + srcOffset), scalarValue, fullMask,
            intriInfo.repeatRemaining, repeatParams, roundEn);
    }

    if (intriInfo.tail != 0) {
        dstOffset = intriInfo.repeat * repeatParams.dstRepStride * intriInfo.c0Count;
        srcOffset = intriInfo.repeat * repeatParams.srcRepStride * intriInfo.c0Count;
        func((__ubuf__ T*)(dst + dstOffset), (__ubuf__ T*)(src + srcOffset), scalarValue, intriInfo.tail, 1,
            repeatParams, roundEn);
    }
}
/* **************************************************************************************************
 * Adds                                             *
 * ************************************************************************************************* */
__aicore__ inline void AddsIntrinsicsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    vadds(dst, src, scalarValue, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

__aicore__ inline void AddsIntrinsicsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    vadds(dst, src, scalarValue, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

__aicore__ inline void AddsIntrinsicsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    vadds(dst, src, scalarValue, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

__aicore__ inline void AddsIntrinsicsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    vadds(dst, src, scalarValue, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}
// Adds::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
    AddsIntrinsicsImpl(dst, src, scalarValue, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    AscendCUtils::SetMask<T, isSetMask>(mask);
    AddsIntrinsicsImpl(dst, src, scalarValue, repeatTimes, repeatParams);
}

// Adds::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const int32_t& calCount)
{
    VecBinaryScalarCompute(dst, src, scalarValue, calCount, AddsImpl<T, isSetMask>);
}

/* **************************************************************************************************
 * Muls                                             *
 * ************************************************************************************************* */
__aicore__ inline void MulsIntrinsicsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    vmuls(dst, src, scalarValue, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

__aicore__ inline void MulsIntrinsicsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    vmuls(dst, src, scalarValue, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

__aicore__ inline void MulsIntrinsicsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    vmuls(dst, src, scalarValue, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

__aicore__ inline void MulsIntrinsicsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    vmuls(dst, src, scalarValue, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride);
}

// Muls::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
    MulsIntrinsicsImpl(dst, src, scalarValue, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    AscendCUtils::SetMask<T, isSetMask>(mask);
    MulsIntrinsicsImpl(dst, src, scalarValue, repeatTimes, repeatParams);
}

// Muls::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const int32_t& calCount)
{
    VecBinaryScalarCompute(dst, src, scalarValue, calCount, MulsImpl<T>);
}

/* **************************************************************************************************
 * Maxs                                             *
 * ************************************************************************************************* */
// Maxs::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const uint64_t mask[],
    uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "Maxs");
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const uint64_t mask,
    uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "Maxs");
}
// Maxs::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const int32_t& calCount)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "Maxs");
}

/* **************************************************************************************************
 * Mins                                             *
 * ************************************************************************************************* */
// Mins::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const uint64_t mask[],
    uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "Mins");
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const uint64_t mask,
    uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "Mins");
}

// Mins::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const int32_t& calCount)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "Mins");
}

/* **************************************************************************************************
 * ShiftLeft                                             *
 * ************************************************************************************************* */
// ShiftLeft::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "ShiftLeft");
}

template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "ShiftLeft");
}

// ShiftLeft::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const int32_t& calCount)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "ShiftLeft");
}

/* **************************************************************************************************
 * ShiftRight                                             *
 * ************************************************************************************************* */
// ShiftRight::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const uint64_t mask[],
    uint8_t repeatTimes, const UnaryRepeatParams& repeatParams, bool roundEn = false)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "ShiftRight");
}

template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const uint64_t mask,
    uint8_t repeatTimes, const UnaryRepeatParams& repeatParams, bool roundEn = false)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "ShiftRight");
}

// ShiftRight::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const int32_t& calCount)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "ShiftRight");
}

/* **************************************************************************************************
 * LeakyRelu                                             *
 * ************************************************************************************************* */
// LeakyRelu::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void LeakyReluImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const uint64_t mask[],
    uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LeakyRelu");
}

template <typename T, bool isSetMask = true>
__aicore__ inline void LeakyReluImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const uint64_t mask,
    uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LeakyRelu");
}

// LeakyRelu::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void LeakyReluImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const int32_t& calCount)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "LeakyRelu");
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_BINARY_SCALAR_IMPL_H
