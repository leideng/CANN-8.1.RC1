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
#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"
#include "kernel_struct_unary.h"

namespace AscendC {
/* **************************************************************************************************
 * Adds                                             *
 * ************************************************************************************************* */
// Adds::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }

    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_bool preg;
        preg = movp_b16();
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 16, strideConfig0, preg);
            vadds(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 16, strideConfig1, preg);
        }
    }
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }

    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        vector_bool preg;
        preg = movp_b32();
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 8, strideConfig0, preg);
            vadds(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 8, strideConfig1, preg);
        }
    }
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }

    __VEC_SCOPE__
    {
        vector_s32 vreg0;
        vector_s32 vreg1;
        vector_bool preg;
        preg = movp_b32();
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 8, strideConfig0, preg);
            vadds(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 8, strideConfig1, preg);
        }
    }
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }

    __VEC_SCOPE__
    {
        vector_s16 vreg0;
        vector_s16 vreg1;
        vector_bool preg;
        preg = movp_b16();
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 16, strideConfig0, preg);
            vadds(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 16, strideConfig1, preg);
        }
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b16(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 16, strideConfig0, preg);
            vadds(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 16, strideConfig1, preg);
        }
    }
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        vector_s16 vreg0;
        vector_s16 vreg1;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b16(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 16, strideConfig0, preg);
            vadds(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 16, strideConfig1, preg);
        }
    }
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        vector_s32 vreg0;
        vector_s32 vreg1;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b32(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 8, strideConfig0, preg);
            vadds(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 8, strideConfig1, preg);
        }
    }
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b32(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 8, strideConfig0, preg);
            vadds(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 8, strideConfig1, preg);
        }
    }
}

// Adds::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const int32_t& calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint32_t sregLower = (uint32_t)128;
        uint16_t repeatTimes = CeilDivision(calCount, 128);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, src, i * sregLower, NORM);
            vadds(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsts(vreg1, dst, i * sregLower, NORM_B16, preg);
        }
    }
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint32_t sregLower = (uint32_t)64;
        uint16_t repeatTimes = CeilDivision(calCount, 64);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b32(sreg, POST_UPDATE);
            vlds(vreg0, src, i * sregLower, NORM);
            vadds(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsts(vreg1, dst, i * sregLower, NORM_B32, preg);
        }
    }
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_s16 vreg0;
        vector_s16 vreg1;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint32_t sregLower = (uint32_t)128;
        uint16_t repeatTimes = CeilDivision(calCount, 128);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, src, i * sregLower, NORM);
            vadds(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsts(vreg1, dst, i * sregLower, NORM_B16, preg);
        }
    }
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_s32 vreg0;
        vector_s32 vreg1;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint32_t sregLower = (uint32_t)64;
        uint16_t repeatTimes = CeilDivision(calCount, 64);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b32(sreg, POST_UPDATE);
            vlds(vreg0, src, i * sregLower, NORM);
            vadds(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsts(vreg1, dst, i * sregLower, NORM_B32, preg);
        }
    }
}

/* **************************************************************************************************
 * Muls                                             *
 * ************************************************************************************************* */
// Muls::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }

    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_bool preg;
        preg = movp_b16();
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 16, strideConfig0, preg);
            vmuls(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 16, strideConfig1, preg);
        }
    }
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }

    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        vector_bool preg;
        vector_bool preg1;
        preg = movp_b32();
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 8, strideConfig0, preg);
            vmuls(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 8, strideConfig1, preg);
        }
    }
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }

    __VEC_SCOPE__
    {
        vector_s32 vreg0;
        vector_s32 vreg1;
        vector_bool preg;
        vector_bool preg1;
        preg = movp_b32();
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 8, strideConfig0, preg);
            vmuls(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 8, strideConfig1, preg);
        }
    }
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }

    __VEC_SCOPE__
    {
        vector_s16 vreg0;
        vector_s16 vreg1;
        vector_bool preg;
        preg = movp_b16();
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 16, strideConfig0, preg);
            vmuls(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 16, strideConfig1, preg);
        }
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b16(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 16, strideConfig0, preg);
            vmuls(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 16, strideConfig1, preg);
        }
    }
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        vector_s16 vreg0;
        vector_s16 vreg1;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b16(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 16, strideConfig0, preg);
            vmuls(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 16, strideConfig1, preg);
        }
    }
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        vector_s32 vreg0;
        vector_s32 vreg1;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b32(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 8, strideConfig0, preg);
            vmuls(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 8, strideConfig1, preg);
        }
    }
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b32(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 8, strideConfig0, preg);
            vmuls(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 8, strideConfig1, preg);
        }
    }
}

// Muls::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ T* dst, __ubuf__ T* src, const T scalarValue, const int32_t& calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint32_t sregLower = (uint32_t)128;
        uint16_t repeatTimes = CeilDivision(calCount, 128);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, src, i * sregLower, NORM);
            vmuls(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsts(vreg1, dst, i * sregLower, NORM_B16, preg);
        }
    }
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint32_t sregLower = (uint32_t)64;
        uint16_t repeatTimes = CeilDivision(calCount, 64);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b32(sreg, POST_UPDATE);
            vlds(vreg0, src, i * sregLower, NORM);
            vmuls(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsts(vreg1, dst, i * sregLower, NORM_B32, preg);
        }
    }
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_s16 vreg0;
        vector_s16 vreg1;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint32_t sregLower = (uint32_t)128;
        uint16_t repeatTimes = CeilDivision(calCount, 128);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, src, i * sregLower, NORM);
            vmuls(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsts(vreg1, dst, i * sregLower, NORM_B16, preg);
        }
    }
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_s32 vreg0;
        vector_s32 vreg1;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint32_t sregLower = (uint32_t)64;
        uint16_t repeatTimes = CeilDivision(calCount, 64);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b32(sreg, POST_UPDATE);
            vlds(vreg0, src, i * sregLower, NORM);
            vmuls(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsts(vreg1, dst, i * sregLower, NORM_B32, preg);
        }
    }
}

/* **************************************************************************************************
 * Maxs                                             *
 * ************************************************************************************************* */
// Maxs::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }

    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_bool preg;
        preg = movp_b16();
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 16, strideConfig0, preg);
            vmaxs(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 16, strideConfig1, preg);
        }
    }
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }

    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        vector_bool preg;
        preg = movp_b32();
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 8, strideConfig0, preg);
            vmaxs(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 8, strideConfig1, preg);
        }
    }
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }

    __VEC_SCOPE__
    {
        vector_s32 vreg0;
        vector_s32 vreg1;
        vector_bool preg;
        preg = movp_b32();
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 8, strideConfig0, preg);
            vmaxs(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 8, strideConfig1, preg);
        }
    }
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }

    __VEC_SCOPE__
    {
        vector_s16 vreg0;
        vector_s16 vreg1;
        vector_bool preg;
        preg = movp_b16();
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 16, strideConfig0, preg);
            vmaxs(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 16, strideConfig1, preg);
        }
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b16(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 16, strideConfig0, preg);
            vmaxs(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 16, strideConfig1, preg);
        }
    }
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        vector_s16 vreg0;
        vector_s16 vreg1;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b16(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 16, strideConfig0, preg);
            vmaxs(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 16, strideConfig1, preg);
        }
    }
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        vector_s32 vreg0;
        vector_s32 vreg1;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b32(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 8, strideConfig0, preg);
            vmaxs(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 8, strideConfig1, preg);
        }
    }
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b32(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 8, strideConfig0, preg);
            vmaxs(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 8, strideConfig1, preg);
        }
    }
}

// Maxs::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const int32_t& calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint32_t sregLower = (uint32_t)128;
        uint16_t repeatTimes = CeilDivision(calCount, 128);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, src, i * sregLower, NORM);
            vmaxs(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsts(vreg1, dst, i * sregLower, NORM_B16, preg);
        }
    }
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint32_t sregLower = (uint32_t)64;
        uint16_t repeatTimes = CeilDivision(calCount, 64);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b32(sreg, POST_UPDATE);
            vlds(vreg0, src, i * sregLower, NORM);
            vmaxs(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsts(vreg1, dst, i * sregLower, NORM_B32, preg);
        }
    }
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_s16 vreg0;
        vector_s16 vreg1;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint32_t sregLower = (uint32_t)128;
        uint16_t repeatTimes = CeilDivision(calCount, 128);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, src, i * sregLower, NORM);
            vmaxs(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsts(vreg1, dst, i * sregLower, NORM_B16, preg);
        }
    }
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_s32 vreg0;
        vector_s32 vreg1;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint32_t sregLower = (uint32_t)64;
        uint16_t repeatTimes = CeilDivision(calCount, 64);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b32(sreg, POST_UPDATE);
            vlds(vreg0, src, i * sregLower, NORM);
            vmaxs(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsts(vreg1, dst, i * sregLower, NORM_B32, preg);
        }
    }
}

/* **************************************************************************************************
 * Mins                                             *
 * ************************************************************************************************* */
// Mins::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }

    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_bool preg;
        preg = movp_b16();
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 16, strideConfig0, preg);
            vmins(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 16, strideConfig1, preg);
        }
    }
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }

    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        vector_bool preg;
        preg = movp_b32();
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 8, strideConfig0, preg);
            vmins(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 8, strideConfig1, preg);
        }
    }
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }

    __VEC_SCOPE__
    {
        vector_s32 vreg0;
        vector_s32 vreg1;
        vector_bool preg;
        preg = movp_b32();
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 8, strideConfig0, preg);
            vmins(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 8, strideConfig1, preg);
        }
    }
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }

    __VEC_SCOPE__
    {
        vector_s16 vreg0;
        vector_s16 vreg1;
        vector_bool preg;
        preg = movp_b16();
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 16, strideConfig0, preg);
            vmins(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 16, strideConfig1, preg);
        }
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b16(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 16, strideConfig0, preg);
            vmins(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 16, strideConfig1, preg);
        }
    }
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        vector_s16 vreg0;
        vector_s16 vreg1;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b16(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 16, strideConfig0, preg);
            vmins(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 16, strideConfig1, preg);
        }
    }
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        vector_s32 vreg0;
        vector_s32 vreg1;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b32(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 8, strideConfig0, preg);
            vmins(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 8, strideConfig1, preg);
        }
    }
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b32(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 8, strideConfig0, preg);
            vmins(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 8, strideConfig1, preg);
        }
    }
}

// Mins::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const int32_t& calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint32_t sregLower = (uint32_t)128;
        uint16_t repeatTimes = CeilDivision(calCount, 128);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, src, i * sregLower, NORM);
            vmins(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsts(vreg1, dst, i * sregLower, NORM_B16, preg);
        }
    }
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint32_t sregLower = (uint32_t)64;
        uint16_t repeatTimes = CeilDivision(calCount, 64);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b32(sreg, POST_UPDATE);
            vlds(vreg0, src, i * sregLower, NORM);
            vmins(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsts(vreg1, dst, i * sregLower, NORM_B32, preg);
        }
    }
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_s16 vreg0;
        vector_s16 vreg1;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint32_t sregLower = (uint32_t)128;
        uint16_t repeatTimes = CeilDivision(calCount, 128);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, src, i * sregLower, NORM);
            vmins(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsts(vreg1, dst, i * sregLower, NORM_B16, preg);
        }
    }
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_s32 vreg0;
        vector_s32 vreg1;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint32_t sregLower = (uint32_t)64;
        uint16_t repeatTimes = CeilDivision(calCount, 64);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b32(sreg, POST_UPDATE);
            vlds(vreg0, src, i * sregLower, NORM);
            vmins(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsts(vreg1, dst, i * sregLower, NORM_B32, preg);
        }
    }
}

/* **************************************************************************************************
 * ShiftLeft                                             *
 * ************************************************************************************************* */
// ShiftLeft::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = uint16_t, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src,
    uint16_t scalarValue, const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }

    __VEC_SCOPE__
    {
        vector_u16 vreg0;
        vector_u16 vreg1;
        vector_bool preg;
        preg = movp_b16();
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 16, strideConfig0, preg);
            vshls(vreg1, vreg0, (int16_t)scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 16, strideConfig1, preg);
        }
    }
}

template <typename T = uint32_t, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src,
    uint32_t scalarValue, const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }

    __VEC_SCOPE__
    {
        vector_u32 vreg0;
        vector_u32 vreg1;
        vector_bool preg;
        preg = movp_b32();
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 8, strideConfig0, preg);
            vshls(vreg1, vreg0, (int16_t)scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 8, strideConfig1, preg);
        }
    }
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }

    __VEC_SCOPE__
    {
        vector_s32 vreg0;
        vector_s32 vreg1;
        vector_bool preg;
        preg = movp_b32();
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 8, strideConfig0, preg);
            vshls(vreg1, vreg0, (int16_t)scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 8, strideConfig1, preg);
        }
    }
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }

    __VEC_SCOPE__
    {
        vector_s16 vreg0;
        vector_s16 vreg1;
        vector_bool preg;
        preg = movp_b16();
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 16, strideConfig0, preg);
            vshls(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 16, strideConfig1, preg);
        }
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = uint16_t, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src,
    uint16_t scalarValue, const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        vector_u16 vreg0;
        vector_u16 vreg1;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b16(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 16, strideConfig0, preg);
            vshls(vreg1, vreg0, (int16_t)scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 16, strideConfig1, preg);
        }
    }
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        vector_s16 vreg0;
        vector_s16 vreg1;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b16(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 16, strideConfig0, preg);
            vshls(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 16, strideConfig1, preg);
        }
    }
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        vector_s32 vreg0;
        vector_s32 vreg1;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b32(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 8, strideConfig0, preg);
            vshls(vreg1, vreg0, (int16_t)scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 8, strideConfig1, preg);
        }
    }
}

template <typename T = uint32_t, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src,
    uint32_t scalarValue, const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        vector_u32 vreg0;
        vector_u32 vreg1;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b32(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 8, strideConfig0, preg);
            vshls(vreg1, vreg0, (int16_t)scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 8, strideConfig1, preg);
        }
    }
}

// ShiftLeft::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const int32_t& calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = uint16_t, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src,
    uint16_t scalarValue, const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_u16 vreg0;
        vector_u16 vreg1;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint32_t sregLower = (uint32_t)128;
        uint16_t repeatTimes = CeilDivision(calCount, 128);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, src, i * sregLower, NORM);
            vshls(vreg1, vreg0, (int16_t)scalarValue, preg, MODE_ZEROING);
            vsts(vreg1, dst, i * sregLower, NORM_B16, preg);
        }
    }
}

template <typename T = uint32_t, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src,
    uint32_t scalarValue, const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_u32 vreg0;
        vector_u32 vreg1;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint32_t sregLower = (uint32_t)64;
        uint16_t repeatTimes = CeilDivision(calCount, 64);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b32(sreg, POST_UPDATE);
            vlds(vreg0, src, i * sregLower, NORM);
            vshls(vreg1, vreg0, (int16_t)scalarValue, preg, MODE_ZEROING);
            vsts(vreg1, dst, i * sregLower, NORM_B32, preg);
        }
    }
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_s16 vreg0;
        vector_s16 vreg1;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint32_t sregLower = (uint32_t)128;
        uint16_t repeatTimes = CeilDivision(calCount, 128);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, src, i * sregLower, NORM);
            vshls(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsts(vreg1, dst, i * sregLower, NORM_B16, preg);
        }
    }
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_s32 vreg0;
        vector_s32 vreg1;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint32_t sregLower = (uint32_t)64;
        uint16_t repeatTimes = CeilDivision(calCount, 64);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b32(sreg, POST_UPDATE);
            vlds(vreg0, src, i * sregLower, NORM);
            vshls(vreg1, vreg0, (int16_t)scalarValue, preg, MODE_ZEROING);
            vsts(vreg1, dst, i * sregLower, NORM_B32, preg);
        }
    }
}

/* **************************************************************************************************
 * ShiftRight                                             *
 * ************************************************************************************************* */
// ShiftRight::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams, bool roundEn)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = uint16_t, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src,
    uint16_t scalarValue, const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams,
    bool roundEn)
{
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }

    __VEC_SCOPE__
    {
        vector_u16 vreg0;
        vector_u16 vreg1;
        vector_bool preg;
        preg = movp_b16();
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 16, strideConfig0, preg);
            vshrs(vreg1, vreg0, (int16_t)scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 16, strideConfig1, preg);
        }
    }
}

template <typename T = uint32_t, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src,
    uint32_t scalarValue, const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams,
    bool roundEn)
{
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }

    __VEC_SCOPE__
    {
        vector_u32 vreg0;
        vector_u32 vreg1;
        vector_bool preg;
        preg = movp_b32();
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 8, strideConfig0, preg);
            vshrs(vreg1, vreg0, (int16_t)scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 8, strideConfig1, preg);
        }
    }
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams, bool roundEn)
{
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }

    __VEC_SCOPE__
    {
        vector_s32 vreg0;
        vector_s32 vreg1;
        vector_bool preg;
        preg = movp_b32();
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 8, strideConfig0, preg);
            vshrs(vreg1, vreg0, (int16_t)scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 8, strideConfig1, preg);
        }
    }
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams, bool roundEn)
{
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }

    __VEC_SCOPE__
    {
        vector_s16 vreg0;
        vector_s16 vreg1;
        vector_bool preg;
        preg = movp_b16();
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 16, strideConfig0, preg);
            vshrs(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 16, strideConfig1, preg);
        }
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams, bool roundEn)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = uint16_t, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src,
    uint16_t scalarValue, const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams,
    bool roundEn)
{
    __VEC_SCOPE__
    {
        vector_u16 vreg0;
        vector_u16 vreg1;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b16(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 16, strideConfig0, preg);
            vshrs(vreg1, vreg0, (int16_t)scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 16, strideConfig1, preg);
        }
    }
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams, bool roundEn)
{
    __VEC_SCOPE__
    {
        vector_s16 vreg0;
        vector_s16 vreg1;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b16(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 16, strideConfig0, preg);
            vshrs(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 16, strideConfig1, preg);
        }
    }
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams, bool roundEn)
{
    __VEC_SCOPE__
    {
        vector_s32 vreg0;
        vector_s32 vreg1;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b32(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 8, strideConfig0, preg);
            vshrs(vreg1, vreg0, (int16_t)scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 8, strideConfig1, preg);
        }
    }
}

template <typename T = uint32_t, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src,
    uint32_t scalarValue, const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams,
    bool roundEn)
{
    __VEC_SCOPE__
    {
        vector_u32 vreg0;
        vector_u32 vreg1;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b32(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 8, strideConfig0, preg);
            vshrs(vreg1, vreg0, (int16_t)scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 8, strideConfig1, preg);
        }
    }
}

// ShiftLeft::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ T* dst, __ubuf__ T* src, const T scalarValue, const int32_t& calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = uint16_t, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src,
    uint16_t scalarValue, const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_u16 vreg0;
        vector_u16 vreg1;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint32_t sregLower = (uint32_t)128;
        uint16_t repeatTimes = CeilDivision(calCount, 128);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, src, i * sregLower, NORM);
            vshrs(vreg1, vreg0, (int16_t)scalarValue, preg, MODE_ZEROING);
            vsts(vreg1, dst, i * sregLower, NORM_B16, preg);
        }
    }
}

template <typename T = uint32_t, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src,
    uint32_t scalarValue, const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_u32 vreg0;
        vector_u32 vreg1;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint32_t sregLower = (uint32_t)64;
        uint16_t repeatTimes = CeilDivision(calCount, 64);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b32(sreg, POST_UPDATE);
            vlds(vreg0, src, i * sregLower, NORM);
            vshrs(vreg1, vreg0, (int16_t)scalarValue, preg, MODE_ZEROING);
            vsts(vreg1, dst, i * sregLower, NORM_B32, preg);
        }
    }
}

template <typename T = int16_t, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src, int16_t scalarValue,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_s16 vreg0;
        vector_s16 vreg1;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint32_t sregLower = (uint32_t)128;
        uint16_t repeatTimes = CeilDivision(calCount, 128);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, src, i * sregLower, NORM);
            vshrs(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsts(vreg1, dst, i * sregLower, NORM_B16, preg);
        }
    }
}

template <typename T = int32_t, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src, int32_t scalarValue,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_s32 vreg0;
        vector_s32 vreg1;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint32_t sregLower = (uint32_t)64;
        uint16_t repeatTimes = CeilDivision(calCount, 64);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b32(sreg, POST_UPDATE);
            vlds(vreg0, src, i * sregLower, NORM);
            vshrs(vreg1, vreg0, (int16_t)scalarValue, preg, MODE_ZEROING);
            vsts(vreg1, dst, i * sregLower, NORM_B32, preg);
        }
    }
}

/* **************************************************************************************************
 * LeakyRelu                                             *
 * ************************************************************************************************* */
// LeakyRelu::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void LeakyReluImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask[],
    uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void LeakyReluImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }

    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        vector_bool preg;
        preg = movp_b32();
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 8, strideConfig0, preg);
            vlrelu(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 8, strideConfig1, preg);
        }
    }
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void LeakyReluImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        SetVectorMask<T>(mask[1], mask[0]);
    }

    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_bool preg;
        preg = movp_b16();
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 16, strideConfig0, preg);
            vlrelu(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 16, strideConfig1, preg);
        }
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void LeakyReluImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask,
    uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void LeakyReluImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b32(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 8, strideConfig0, preg);
            vlrelu(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 8, strideConfig1, preg);
        }
    }
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void LeakyReluImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b16(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)repeatParams.srcBlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, src + i * repeatParams.srcRepStride * 16, strideConfig0, preg);
            vlrelu(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg1, dst + i * repeatParams.dstRepStride * 16, strideConfig1, preg);
        }
    }
}

// LeakyRelu::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void LeakyReluImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const int32_t& calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void LeakyReluImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint32_t sregLower = (uint32_t)64;
        uint16_t repeatTimes = CeilDivision(calCount, 64);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b32(sreg, POST_UPDATE);
            vlds(vreg0, src, i * sregLower, NORM);
            vlrelu(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsts(vreg1, dst, i * sregLower, NORM_B32, preg);
        }
    }
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void LeakyReluImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint32_t sregLower = (uint32_t)128;
        uint16_t repeatTimes = CeilDivision(calCount, 128);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, src, i * sregLower, NORM);
            vlrelu(vreg1, vreg0, scalarValue, preg, MODE_ZEROING);
            vsts(vreg1, dst, i * sregLower, NORM_B16, preg);
        }
    }
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_BINARY_SCALAR_IMPL_H
