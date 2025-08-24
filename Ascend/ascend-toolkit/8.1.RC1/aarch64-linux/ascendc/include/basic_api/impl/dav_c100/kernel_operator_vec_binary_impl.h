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
 * \file kernel_operator_vec_binary_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_BINARY_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_BINARY_IMPL_H
#include "kernel_utils.h"
#include "kernel_struct_binary.h"

namespace AscendC {
template <typename T>
__aicore__ inline void BinaryCompute(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount,
    void (*func)(__ubuf__ T*, __ubuf__ T*, __ubuf__ T*, const uint64_t, const uint8_t, const BinaryRepeatParams&))
{
    struct BinaryRepeatParams repeatParams;
    IntriInfo intriInfo = AscendCUtils::CalIntriInfo(sizeof(T), calCount);

    uint32_t dstOffset = 0;
    uint32_t src0Offset = 0;
    uint32_t src1Offset = 0;
    const auto dstOffsetCount = MAX_REPEAT_TIMES * repeatParams.dstRepStride * intriInfo.c0Count;
    const auto src0OffsetCount = MAX_REPEAT_TIMES * repeatParams.src0RepStride * intriInfo.c0Count;
    const auto src1OffsetCount = MAX_REPEAT_TIMES * repeatParams.src1RepStride * intriInfo.c0Count;

    const int32_t fullMask = intriInfo.c0Count * DEFAULT_BLK_NUM;
    for (uint32_t i = 0; i < intriInfo.repeatRounding; i++) {
        func(dst + dstOffset, src0 + src0Offset, src1 + src1Offset, fullMask, MAX_REPEAT_TIMES, repeatParams);
        dstOffset += dstOffsetCount;
        src0Offset += src0OffsetCount;
        src1Offset += src1OffsetCount;
    }

    dstOffset = (intriInfo.repeatRounding * MAX_REPEAT_TIMES) * repeatParams.dstRepStride * intriInfo.c0Count;
    src0Offset = (intriInfo.repeatRounding * MAX_REPEAT_TIMES) * repeatParams.src0RepStride * intriInfo.c0Count;
    src1Offset = (intriInfo.repeatRounding * MAX_REPEAT_TIMES) * repeatParams.src1RepStride * intriInfo.c0Count;

    if (intriInfo.repeatRemaining != 0) {
        func(dst + dstOffset, src0 + src0Offset, src1 + src1Offset, fullMask, intriInfo.repeatRemaining, repeatParams);
    }

    if (intriInfo.tail != 0) {
        dstOffset = intriInfo.repeat * repeatParams.dstRepStride * intriInfo.c0Count;
        src0Offset = intriInfo.repeat * repeatParams.src0RepStride * intriInfo.c0Count;
        src1Offset = intriInfo.repeat * repeatParams.src1RepStride * intriInfo.c0Count;
        func(dst + dstOffset, src0 + src0Offset, src1 + src1Offset, intriInfo.tail, 1, repeatParams);
    }
}
/* **************************************************************************************************
 * Add                                             *
 * ************************************************************************************************* */
__aicore__ inline void AddIntrinsicsImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1,
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    vadd(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

__aicore__ inline void AddIntrinsicsImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1,
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    vadd(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

__aicore__ inline void AddIntrinsicsImpl(const __ubuf__ int16_t* dst, const __ubuf__ int16_t* src0,
    const __ubuf__ int16_t* src1, uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    (void)(dst);
    (void)(src0);
    (void)(src1);
    (void)(repeatTimes);
    (void)(repeatParams);
    ASCENDC_REPORT_NOT_SUPPORT(false, "Add with type int16_t");
}

__aicore__ inline void AddIntrinsicsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1,
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    vadd(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}
// Add::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void AddImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask[1], mask[0]);
    }
    AddIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void AddImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask);
    }
    AddIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
}

// Add::Level 2
template <typename T>
__aicore__ inline void AddImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    BinaryCompute(dst, src0, src1, calCount, AddImpl<T>);
}

/* **************************************************************************************************
 * Sub                                             *
 * ************************************************************************************************* */
__aicore__ inline void SubIntrinsicsImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1,
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    vsub(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

__aicore__ inline void SubIntrinsicsImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1,
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    vsub(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

__aicore__ inline void SubIntrinsicsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    vsub(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

__aicore__ inline void SubIntrinsicsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1,
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    vsub(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}
// Sub::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void SubImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask[1], mask[0]);
    }
    SubIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void SubImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask);
    }
    SubIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
}

// Sub::Level 2
template <typename T>
__aicore__ inline void SubImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    BinaryCompute(dst, src0, src1, calCount, SubImpl<T>);
}
/* **************************************************************************************************
 * Mul                                             *
 * ************************************************************************************************* */
__aicore__ inline void MulIntrinsicsImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1,
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    vmul(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

__aicore__ inline void MulIntrinsicsImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1,
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    vmul(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

__aicore__ inline void MulIntrinsicsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    vmul(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

__aicore__ inline void MulIntrinsicsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1,
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    vmul(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}
// Mul::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void MulImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask[1], mask[0]);
    }
    MulIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MulImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask);
    }
    MulIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
}

// Mul::Level 2
template <typename T>
__aicore__ inline void MulImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    BinaryCompute(dst, src0, src1, calCount, MulImpl<T>);
}
/* **************************************************************************************************
 * Div                                             *
 * ************************************************************************************************* */
__aicore__ inline void DivIntrinsicsImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1,
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    vdiv(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

__aicore__ inline void DivIntrinsicsImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1,
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    vdiv(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

// Div::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void DivImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask[1], mask[0]);
    }
    DivIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void DivImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask);
    }
    DivIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
}

// Div::Level 2
template <typename T>
__aicore__ inline void DivImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    BinaryCompute(dst, src0, src1, calCount, DivImpl<T>);
}

/* **************************************************************************************************
 * Max                                             *
 * ************************************************************************************************* */
__aicore__ inline void MaxIntrinsicsImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1,
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    vmax(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

__aicore__ inline void MaxIntrinsicsImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1,
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    vmax(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

__aicore__ inline void MaxIntrinsicsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    vmax(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

__aicore__ inline void MaxIntrinsicsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1,
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    vmax(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}
// Max::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void MaxImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask[1], mask[0]);
    }
    MaxIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MaxImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask);
    }
    MaxIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
}

// Max::Level 2
template <typename T>
__aicore__ inline void MaxImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    BinaryCompute(dst, src0, src1, calCount, MaxImpl<T>);
}

/* **************************************************************************************************
 * Min                                             *
 * ************************************************************************************************* */
__aicore__ inline void MinIntrinsicsImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1,
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    vmin(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

__aicore__ inline void MinIntrinsicsImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1,
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    vmin(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

__aicore__ inline void MinIntrinsicsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    vmin(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

__aicore__ inline void MinIntrinsicsImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1,
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    vmin(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}
// Min::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void MinImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask[1], mask[0]);
    }
    MinIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MinImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask);
    }
    MinIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
}

// Min::Level 2
template <typename T>
__aicore__ inline void MinImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    BinaryCompute(dst, src0, src1, calCount, MinImpl<T>);
}

/* **************************************************************************************************
 * And                                             *
 * ************************************************************************************************* */
__aicore__ inline void AndIntrinsicsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    vand(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}
__aicore__ inline void AndIntrinsicsImpl(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src0, __ubuf__ uint16_t* src1,
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    vand(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

// And::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void AndImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask[1], mask[0]);
    }
    AndIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void AndImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask);
    }
    AndIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
}

// And::Level 2
template <typename T>
__aicore__ inline void AndImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    BinaryCompute(dst, src0, src1, calCount, AndImpl<T>);
}

/* **************************************************************************************************
 * Or                                             *
 * ************************************************************************************************* */
__aicore__ inline void OrIntrinsicsImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    vor(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride, repeatParams.src1BlkStride,
        repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

__aicore__ inline void OrIntrinsicsImpl(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src0, __ubuf__ uint16_t* src1,
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    vor(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride, repeatParams.src1BlkStride,
        repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}
// Or::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void OrImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask[1], mask[0]);
    }
    OrIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void OrImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask);
    }
    OrIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
}

// Or::Level 2
template <typename T>
__aicore__ inline void OrImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    BinaryCompute(dst, src0, src1, calCount, OrImpl<T>);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void AddReluImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "AddRelu");
}

template <typename T, bool isSetMask = true>
__aicore__ inline void AddReluImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "AddRelu");
}

/* **************************************************************************************************
 * FusedMulAdd                                             *
 * ************************************************************************************************* */
// FusedMulAdd::Level 2
template <typename T>
__aicore__ inline void FusedMulAddImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "FusedMulAdd");
}

// FusedMulAdd::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void FusedMulAddImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "FusedMulAdd");
}

template <typename T, bool isSetMask = true>
__aicore__ inline void FusedMulAddImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "FusedMulAdd");
}

/* **************************************************************************************************
 * FusedMulAddRelu                                             *
 * ************************************************************************************************* */
// FusedMulAddRelu::Level 2
template <typename T>
__aicore__ inline void FusedMulAddReluImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1,
    const int32_t& calCount)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "FusedMulAddRelu");
}

// FusedMulAddRelu::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void FusedMulAddReluImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1,
    const uint64_t mask[], const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "FusedMulAddRelu");
}

template <typename T, bool isSetMask = true>
__aicore__ inline void FusedMulAddReluImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1,
    const uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "FusedMulAddRelu");
}

// MulAddDst::Level 2
template <typename T, typename U>
__aicore__ inline void MulAddDstImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1, const int32_t& calCount)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "MulAddDst");
}

// MulAddDst::Level 0
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void MulAddDstImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "MulAddDst");
}

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void MulAddDstImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "MulAddDst");
}

/* **************************************************************************************************
 * SubRelu                                             *
 * ************************************************************************************************* */
// SubRelu::Level 2
template <typename T>
__aicore__ inline void SubReluImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "SubRelu");
}

// SubRelu::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void SubReluImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "SubRelu");
}

template <typename T, bool isSetMask = true>
__aicore__ inline void SubReluImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "SubRelu");
}
/* **************************************************************************************************
 * AddDeqRelu                                             *
 * ************************************************************************************************* */
__aicore__ inline void AddDeqReluImpl(__ubuf__ half *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1,
    const int32_t &calCount)
{
    (void)dst;
    (void)src0;
    (void)src1;
    (void)calCount;
    ASCENDC_REPORT_NOT_SUPPORT(false, "AddDeqRelu");
}

// AddDeqRelu::Level 0
template <bool isSetMask = true>
__aicore__ inline void AddDeqReluImpl(__ubuf__ half *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1,
    const uint64_t mask[], const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    (void)dst;
    (void)src0;
    (void)src1;
    (void)mask;
    (void)repeatTimes;
    (void)repeatParams;
    ASCENDC_REPORT_NOT_SUPPORT(false, "AddDeqRelu");
}

template <bool isSetMask = true>
__aicore__ inline void AddDeqReluImpl(__ubuf__ half *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1,
    const uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    (void)dst;
    (void)src0;
    (void)src1;
    (void)mask;
    (void)repeatTimes;
    (void)repeatParams;
    ASCENDC_REPORT_NOT_SUPPORT(false, "AddDeqRelu");
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_BINARY_IMPL_H