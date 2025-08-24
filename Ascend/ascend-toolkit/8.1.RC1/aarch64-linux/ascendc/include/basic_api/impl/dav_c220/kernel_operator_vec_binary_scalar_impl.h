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
/* **************************************************************************************************
 * Adds                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void AddsIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, half, float, int16_t, int32_t>(), "Failed to check dtype in Adds, current api support "
        "dtype combination is src and dst both: half / float / int16_t / int32_t.");
    vadds(dst, src, scalarValue, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint16_t)repeatParams.dstRepStride, (uint16_t)repeatParams.srcRepStride);
}

// Adds::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
        AddsIntrinsicsImpl(dst, src, scalarValue, repeatTimes, repeatParams);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask);
        AddsIntrinsicsImpl(dst, src, scalarValue, repeatTimes, repeatParams);
    }
}

// Adds::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const int32_t& calCount)
{
    if ASCEND_IS_AIV {
        if constexpr (!isSetMask) {
            AddsIntrinsicsImpl(dst, src, scalarValue, 1,
                { DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
            return;
        }
        SetMaskCount();
        AscendCUtils::SetMask<T>(0, calCount);
        AddsIntrinsicsImpl(dst, src, scalarValue, 1,
            { DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        ResetMask();
        SetMaskNorm();
    }
}

/* **************************************************************************************************
 * Muls                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void MulsIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, half, float, int16_t, int32_t>(), "Failed to check dtype in Muls, current api support "
        "dtype combination is src and dst both: half / float / int16_t / int32_t.");
    vmuls(dst, src, scalarValue, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint16_t)repeatParams.dstRepStride, (uint16_t)repeatParams.srcRepStride);
}

// Muls::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
        MulsIntrinsicsImpl(dst, src, scalarValue, repeatTimes, repeatParams);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask);
        MulsIntrinsicsImpl(dst, src, scalarValue, repeatTimes, repeatParams);
    }
}

// Muls::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const int32_t& calCount)
{
    if ASCEND_IS_AIV {
        if constexpr (!isSetMask) {
            MulsIntrinsicsImpl(dst, src, scalarValue, 1,
                { DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
            return;
        }
        SetMaskCount();
        AscendCUtils::SetMask<T>(0, calCount);
        MulsIntrinsicsImpl(dst, src, scalarValue, 1,
            { DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        ResetMask();
        SetMaskNorm();
    }
}

/* **************************************************************************************************
 * Maxs                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void MaxsIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, half, float, int16_t, int32_t>(), "Failed to check dtype in Maxs, current api support "
        "dtype combination is src and dst both: half / float / int16_t / int32_t.");
    vmaxs(dst, src, scalarValue, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride, false, false);
}

// Maxs::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const uint64_t mask[],
    uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
        MaxsIntrinsicsImpl(dst, src, scalarValue, repeatTimes, repeatParams);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const uint64_t mask,
    uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask);
        MaxsIntrinsicsImpl(dst, src, scalarValue, repeatTimes, repeatParams);
    }
}
// Maxs::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const int32_t& calCount)
{
    if ASCEND_IS_AIV {
        if constexpr (!isSetMask) {
            MaxsIntrinsicsImpl(dst, src, scalarValue, 1,
                { DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
            return;
        }
        SetMaskCount();
        AscendCUtils::SetMask<T>(0, calCount);
        MaxsIntrinsicsImpl(dst, src, scalarValue, 1,
            { DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        ResetMask();
        SetMaskNorm();
    }
}

/* **************************************************************************************************
 * Mins                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void MinsIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, half, float, int16_t, int32_t>(), "Failed to check dtype in Mins, current api support "
        "dtype combination is src and dst both: half / float / int16_t / int32_t.");
    vmins(dst, src, scalarValue, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint8_t)repeatParams.dstRepStride, (uint8_t)repeatParams.srcRepStride, false, false);
}

// Mins::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const uint64_t mask[],
    uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
        MinsIntrinsicsImpl(dst, src, scalarValue, repeatTimes, repeatParams);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const uint64_t mask,
    uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask);
        MinsIntrinsicsImpl(dst, src, scalarValue, repeatTimes, repeatParams);
    }
}

// Mins::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const int32_t& calCount)
{
    if ASCEND_IS_AIV {
        if constexpr (!isSetMask) {
            MinsIntrinsicsImpl(dst, src, scalarValue, 1,
                { DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
            return;
        }
        SetMaskCount();
        AscendCUtils::SetMask<T>(0, calCount);
        MinsIntrinsicsImpl(dst, src, scalarValue, 1,
            { DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        ResetMask();
        SetMaskNorm();
    }
}

/* **************************************************************************************************
 * ShiftLeft                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void ShiftLeftIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, uint16_t, uint32_t, int16_t, int32_t>(), "Failed to check dtype in ShiftLeft, current "
        "api support dtype combination is src and dst both: uint16_t / uint32_t / int16_t / int32_t.");
    // B16 must be in range [0, 16]. B32 must be in range [0, 32].
    ASCENDC_CHECK_VALUE_RANGE(scalarValue, 0, sizeof(T) * 8, "scalarValue", "ShiftLeft"); // sizeof(T) * 8 as max
    vshl(dst, src, (uint32_t)scalarValue, repeatTimes, (uint16_t)repeatParams.dstBlkStride,
        (uint16_t)repeatParams.srcBlkStride, (uint16_t)repeatParams.dstRepStride, (uint16_t)repeatParams.srcRepStride);
}

// ShiftLeft::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
        ShiftLeftIntrinsicsImpl(dst, src, scalarValue, repeatTimes, repeatParams);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask);
        ShiftLeftIntrinsicsImpl(dst, src, scalarValue, repeatTimes, repeatParams);
    }
}

// ShiftLeft::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const int32_t& calCount)
{
    if ASCEND_IS_AIV {
        if constexpr (!isSetMask) {
            ShiftLeftIntrinsicsImpl(dst, src, scalarValue, 1,
                { DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
            return;
        }
        SetMaskCount();
        AscendCUtils::SetMask<T>(0, calCount);
        ShiftLeftIntrinsicsImpl(dst, src, scalarValue, 1,
            { DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        ResetMask();
        SetMaskNorm();
    }
}

/* **************************************************************************************************
 * ShiftRight                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void ShiftRightIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams, bool roundEn)
{
    ASCENDC_ASSERT((SupportType<T, int16_t, uint16_t, int32_t, uint32_t>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to "
        "check dtype in ShiftRight, current api support dtype combination is src and dst both: int16_t, uint16_t, "
        "int32_t, uint32_t.");});
    // B16 must be in range [0, 16]. B32 must be in range [0, 32].
    ASCENDC_CHECK_VALUE_RANGE(scalarValue, 0, sizeof(T) * 8, "scalarValue", "ShiftRight"); // sizeof(T) * 8 as max

    if (roundEn) {
        if constexpr (SupportType<T, int16_t, int32_t>()) {
            vshr(dst, src, (int32_t)scalarValue, repeatTimes, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                (uint16_t)repeatParams.dstRepStride, (uint16_t)repeatParams.srcRepStride, true);
        } else {
            vshr(dst, src, (uint32_t)scalarValue, repeatTimes, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                (uint16_t)repeatParams.dstRepStride, (uint16_t)repeatParams.srcRepStride, true);
        }
    } else {
        if constexpr (SupportType<T, int16_t, int32_t>()) {
            vshr(dst, src, (int32_t)scalarValue, repeatTimes, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                (uint16_t)repeatParams.dstRepStride, (uint16_t)repeatParams.srcRepStride, false);
        } else {
            vshr(dst, src, (uint32_t)scalarValue, repeatTimes, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                (uint16_t)repeatParams.dstRepStride, (uint16_t)repeatParams.srcRepStride, false);
        }
    }
}

// ShiftRight::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const uint64_t mask[],
    uint8_t repeatTimes, const UnaryRepeatParams& repeatParams, bool roundEn = false)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
        ShiftRightIntrinsicsImpl(dst, src, scalarValue, repeatTimes, repeatParams, roundEn);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const uint64_t mask,
    uint8_t repeatTimes, const UnaryRepeatParams& repeatParams, bool roundEn = false)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask);
        ShiftRightIntrinsicsImpl(dst, src, scalarValue, repeatTimes, repeatParams, roundEn);
    }
}

// ShiftRight::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const int32_t& calCount)
{
    if ASCEND_IS_AIV {
        if constexpr (!isSetMask) {
            ShiftRightIntrinsicsImpl(dst, src, scalarValue, 1,
                { DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE }, false);
            return;
        }
        SetMaskCount();
        AscendCUtils::SetMask<T>(0, calCount);
        ShiftRightIntrinsicsImpl(dst, src, scalarValue, 1,
            { DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE }, false);
        ResetMask();
        SetMaskNorm();
    }
}

/* **************************************************************************************************
 * LeakyRelu                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void LeakyReluIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, half, float>(), "Failed to check dtype in LeakyRelu, current api support dtype "
        "combination is src and dst both: half / float.");
    vlrelu(dst, src, scalarValue, repeatTimes, (uint16_t)repeatParams.dstBlkStride, (uint16_t)repeatParams.srcBlkStride,
        (uint16_t)repeatParams.dstRepStride, (uint16_t)repeatParams.srcRepStride);
}

// LeakyRelu::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void LeakyReluImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const uint64_t mask[],
    uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
        LeakyReluIntrinsicsImpl(dst, src, scalarValue, repeatTimes, repeatParams);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void LeakyReluImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const uint64_t mask,
    uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask);
        LeakyReluIntrinsicsImpl(dst, src, scalarValue, repeatTimes, repeatParams);
    }
}

// LeakyRelu::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void LeakyReluImpl(__ubuf__ T* dst, __ubuf__ T* src, const T& scalarValue, const int32_t& calCount)
{
    if ASCEND_IS_AIV {
        if constexpr (!isSetMask) {
            LeakyReluIntrinsicsImpl(dst, src, scalarValue, 1,
                { DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
            return;
        }
        SetMaskCount();
        AscendCUtils::SetMask<T>(0, calCount);
        LeakyReluIntrinsicsImpl(dst, src, scalarValue, 1,
            { DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
        ResetMask();
        SetMaskNorm();
    }
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_BINARY_SCALAR_IMPL_H
