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
 * \file kernel_operator_vec_duplicate_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_DUPLICATE_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_DUPLICATE_IMPL_H

#include <type_traits>
#include "kernel_struct_unary.h"

namespace AscendC {
template <typename T> constexpr __aicore__ inline void CheckDuplicateSupportedType()
{
    static_assert(std::is_same<T, half>::value || std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value ||
        std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value || std::is_same<T, float>::value,
        "Duplicate instr only support half/int16_t/uint16_t/int32_t/uint32_t/float type on current device");
}

template <typename T>
__aicore__ inline void DuplicateIntrinsicsImpl(__ubuf__ T* dstLocal, half scalarValue, const uint8_t repeatTimes,
    const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    vector_dup(dstLocal, scalarValue, repeatTimes, dstBlockStride, 1, dstRepeatStride, 0);
}

template <typename T>
__aicore__ inline void DuplicateIntrinsicsImpl(__ubuf__ T* dstLocal, float scalarValue, const uint8_t repeatTimes,
    const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    vector_dup(dstLocal, scalarValue, repeatTimes, dstBlockStride, 1, dstRepeatStride, 0);
}

template <typename T>
__aicore__ inline void DuplicateIntrinsicsImpl(__ubuf__ T* dstLocal, int16_t scalarValue, const uint8_t repeatTimes,
    const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    vector_dup(dstLocal, scalarValue, repeatTimes, dstBlockStride, 1, dstRepeatStride, 0);
}

template <typename T>
__aicore__ inline void DuplicateIntrinsicsImpl(__ubuf__ T* dstLocal, int32_t scalarValue, const uint8_t repeatTimes,
    const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    vector_dup(dstLocal, scalarValue, repeatTimes, dstBlockStride, 1, dstRepeatStride, 0);
}

template <typename T>
__aicore__ inline void DuplicateIntrinsicsImpl(__ubuf__ T* dstLocal, uint16_t scalarValue, const uint8_t repeatTimes,
    const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    vector_dup(dstLocal, scalarValue, repeatTimes, dstBlockStride, 1, dstRepeatStride, 0);
}

template <typename T>
__aicore__ inline void DuplicateIntrinsicsImpl(__ubuf__ T* dstLocal, uint32_t scalarValue, const uint8_t repeatTimes,
    const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    vector_dup(dstLocal, scalarValue, repeatTimes, dstBlockStride, 1, dstRepeatStride, 0);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void DuplicateImpl(__ubuf__ T* dstLocal, const T& scalarValue, uint64_t mask,
    const uint8_t repeatTimes, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask);
    }
    DuplicateIntrinsicsImpl(dstLocal, scalarValue, repeatTimes, dstBlockStride, dstRepeatStride);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void DuplicateImpl(__ubuf__ T* dstLocal, const T& scalarValue, uint64_t mask[],
    const uint8_t repeatTimes, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask[1], mask[0]);
    }
    DuplicateIntrinsicsImpl(dstLocal, scalarValue, repeatTimes, dstBlockStride, dstRepeatStride);
}

template <typename T>
__aicore__ inline void DuplicateImpl(__ubuf__ T* dstLocal, const T& scalarValue, const int32_t& calCount)
{
    IntriInfo intriInfo = AscendCUtils::CalIntriInfo(sizeof(T), calCount);
    const int32_t oneRepeatNum = DEFAULT_BLOCK_SIZE / sizeof(T);
    struct UnaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.srcBlkStride = 0;
    repeatParams.dstRepStride = 8;
    repeatParams.srcRepStride = 0;
    int32_t dstOffset = 0;
    const int32_t dstOffsetCount = MAX_REPEAT_TIMES * oneRepeatNum;
    for (int32_t i = 0; i < intriInfo.repeatRounding; i++) {
        DuplicateImpl((__ubuf__ T*)(dstLocal + dstOffset), scalarValue, oneRepeatNum, MAX_REPEAT_TIMES,
            DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
        dstOffset += dstOffsetCount;
    }
    dstOffset = intriInfo.repeatRounding * MAX_REPEAT_TIMES * oneRepeatNum;
    if (intriInfo.repeatRemaining != 0) {
        DuplicateImpl((__ubuf__ T*)(dstLocal + dstOffset), scalarValue, oneRepeatNum, intriInfo.repeatRemaining,
            DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    }
    if (intriInfo.tail != 0) {
        dstOffset += intriInfo.repeatRemaining * oneRepeatNum;
        DuplicateImpl((__ubuf__ T*)(dstLocal + dstOffset), scalarValue, intriInfo.tail, 1, DEFAULT_BLK_STRIDE,
            DEFAULT_REPEAT_STRIDE);
    }
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_DUPLICATE_IMPL_H
