/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"

namespace AscendC {
template <typename T> constexpr __aicore__ inline void CheckDuplicateSupportedType()
{
    static_assert(std::is_same<T, half>::value || std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value ||
        std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value || std::is_same<T, float>::value,
        "Duplicate instr only support half/int16_t/uint16_t/int32_t/uint32_t/float type on current device");
}

template <typename T, bool isSetMask = true>
__aicore__ inline void DuplicateImpl(__ubuf__ T* dstLocal, const T& scalarValue, uint64_t mask,
    const uint8_t repeatTimes, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

template <>
__aicore__ inline void DuplicateImpl<half, true>(__ubuf__ half* dstLocal, const half& scalarValue, uint64_t mask,
    const uint8_t repeatTimes, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    __VEC_SCOPE__
    {
        vector_f16 vreg;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b16(sreg, POST_UPDATE);
        uint32_t strideConfig = (((uint32_t)dstBlockStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vdup(vreg, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg, dstLocal + i * dstRepeatStride * 16, strideConfig, preg);
        }
    }
}

template <>
__aicore__ inline void DuplicateImpl<float, true>(__ubuf__ float* dstLocal, const float& scalarValue, uint64_t mask,
    const uint8_t repeatTimes, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    __VEC_SCOPE__
    {
        vector_f32 vreg;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b32(sreg, POST_UPDATE);
        uint32_t strideConfig = (((uint32_t)dstBlockStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vdup(vreg, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg, dstLocal + i * dstRepeatStride * 8, strideConfig, preg);
        }
    }
}


template <>
__aicore__ inline void DuplicateImpl<int16_t, true>(__ubuf__ int16_t* dstLocal, const int16_t& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    __VEC_SCOPE__
    {
        vector_s16 vreg;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b16(sreg, POST_UPDATE);
        uint32_t strideConfig = (((uint32_t)dstBlockStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vdup(vreg, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg, dstLocal + i * dstRepeatStride * 16, strideConfig, preg);
        }
    }
}

template <>
__aicore__ inline void DuplicateImpl<uint16_t, true>(__ubuf__ uint16_t* dstLocal, const uint16_t& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    __VEC_SCOPE__
    {
        vector_u16 vreg;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b16(sreg, POST_UPDATE);
        uint32_t strideConfig = (((uint32_t)dstBlockStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vdup(vreg, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg, dstLocal + i * dstRepeatStride * 16, strideConfig, preg);
        }
    }
}

template <>
__aicore__ inline void DuplicateImpl<int32_t, true>(__ubuf__ int32_t* dstLocal, const int32_t& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    __VEC_SCOPE__
    {
        vector_s32 vreg;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b32(sreg, POST_UPDATE);
        uint32_t strideConfig = (((uint32_t)dstBlockStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vdup(vreg, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg, dstLocal + i * dstRepeatStride * 8, strideConfig, preg);
        }
    }
}

template <>
__aicore__ inline void DuplicateImpl<uint32_t, true>(__ubuf__ uint32_t* dstLocal, const uint32_t& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    __VEC_SCOPE__
    {
        vector_u32 vreg;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b32(sreg, POST_UPDATE);
        uint32_t strideConfig = (((uint32_t)dstBlockStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vdup(vreg, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg, dstLocal + i * dstRepeatStride * 8, strideConfig, preg);
        }
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void DuplicateImpl(__ubuf__ T* dstLocal, const T& scalarValue, uint64_t mask[],
    const uint8_t repeatTimes, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

template <>
__aicore__ inline void DuplicateImpl<half, true>(__ubuf__ half* dstLocal, const half& scalarValue, uint64_t mask[],
    const uint8_t repeatTimes, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    __ubuf__ uint8_t* tmpBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    *((__ubuf__ uint64_t*)tmpBuf) = mask[0];
    *((__ubuf__ uint64_t*)tmpBuf + 1) = mask[1];

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    __VEC_SCOPE__
    {
        vector_f16 vreg;
        vector_bool preg;
        plds(preg, ((__ubuf__ uint32_t*)tmpBuf), 0, US);
        uint32_t strideConfig = (((uint32_t)dstBlockStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vdup(vreg, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg, dstLocal + i * dstRepeatStride * 16, strideConfig, preg);
        }
    }
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tmpBuf);
}

template <>
__aicore__ inline void DuplicateImpl<float, true>(__ubuf__ float* dstLocal, const float& scalarValue, uint64_t mask[],
    const uint8_t repeatTimes, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    __ubuf__ uint8_t* tmpBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    *((__ubuf__ uint64_t*)tmpBuf) = mask[0];
    *((__ubuf__ uint64_t*)tmpBuf + 1) = mask[1];

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    __VEC_SCOPE__
    {
        vector_f32 vreg;
        vector_bool preg0;
        vector_bool preg1;
        plds(preg0, ((__ubuf__ uint32_t*)tmpBuf), 0, US);
        punpack(preg1, preg0, LOWER);
        uint32_t strideConfig = (((uint32_t)dstBlockStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vdup(vreg, scalarValue, preg1, MODE_ZEROING);
            vsstb(vreg, dstLocal + i * dstRepeatStride * 8, strideConfig, preg1);
        }
    }
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tmpBuf);
}

template <>
__aicore__ inline void DuplicateImpl<int16_t, true>(__ubuf__ int16_t* dstLocal, const int16_t& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    __ubuf__ uint8_t* tmpBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    *((__ubuf__ uint64_t*)tmpBuf) = mask[0];
    *((__ubuf__ uint64_t*)tmpBuf + 1) = mask[1];

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    __VEC_SCOPE__
    {
        vector_s16 vreg;
        vector_bool preg;
        plds(preg, ((__ubuf__ uint32_t*)tmpBuf), 0, US);
        uint32_t strideConfig = (((uint32_t)dstBlockStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vdup(vreg, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg, dstLocal + i * dstRepeatStride * 16, strideConfig, preg);
        }
    }
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tmpBuf);
}

template <>
__aicore__ inline void DuplicateImpl<uint16_t, true>(__ubuf__ uint16_t* dstLocal, const uint16_t& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    __ubuf__ uint8_t* tmpBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    *((__ubuf__ uint64_t*)tmpBuf) = mask[0];
    *((__ubuf__ uint64_t*)tmpBuf + 1) = mask[1];

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    __VEC_SCOPE__
    {
        vector_u16 vreg;
        vector_bool preg;
        plds(preg, ((__ubuf__ uint32_t*)tmpBuf), 0, US);
        uint32_t strideConfig = (((uint32_t)dstBlockStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vdup(vreg, scalarValue, preg, MODE_ZEROING);
            vsstb(vreg, dstLocal + i * dstRepeatStride * 16, strideConfig, preg);
        }
    }
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tmpBuf);
}

template <>
__aicore__ inline void DuplicateImpl<int32_t, true>(__ubuf__ int32_t* dstLocal, const int32_t& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    __ubuf__ uint8_t* tmpBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    *((__ubuf__ uint64_t*)tmpBuf) = mask[0];
    *((__ubuf__ uint64_t*)tmpBuf + 1) = mask[1];

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    __VEC_SCOPE__
    {
        vector_s32 vreg;
        vector_bool preg0;
        vector_bool preg1;
        plds(preg0, ((__ubuf__ uint32_t*)tmpBuf), 0, US);
        punpack(preg1, preg0, LOWER);
        uint32_t strideConfig = (((uint32_t)dstBlockStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vdup(vreg, scalarValue, preg1, MODE_ZEROING);
            vsstb(vreg, dstLocal + i * dstRepeatStride * 8, strideConfig, preg1);
        }
    }
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tmpBuf);
}

template <>
__aicore__ inline void DuplicateImpl<uint32_t, true>(__ubuf__ uint32_t* dstLocal, const uint32_t& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    __ubuf__ uint8_t* tmpBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    *((__ubuf__ uint64_t*)tmpBuf) = mask[0];
    *((__ubuf__ uint64_t*)tmpBuf + 1) = mask[1];

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    __VEC_SCOPE__
    {
        vector_u32 vreg;
        vector_bool preg0;
        vector_bool preg1;
        plds(preg0, ((__ubuf__ uint32_t*)tmpBuf), 0, US);
        punpack(preg1, preg0, LOWER);
        uint32_t strideConfig = (((uint32_t)dstBlockStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vdup(vreg, scalarValue, preg1, MODE_ZEROING);
            vsstb(vreg, dstLocal + i * dstRepeatStride * 8, strideConfig, preg1);
        }
    }
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tmpBuf);
}

template <typename T>
__aicore__ inline void DuplicateImpl(__ubuf__ T* dstLocal, const T& scalarValue, const int32_t& calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

template <>
__aicore__ inline void DuplicateImpl(__ubuf__ half* dstLocal, const half& scalarValue, const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_f16 vreg;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint16_t repeatTimes = CeilDivision(calCount, 128);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vdup(vreg, scalarValue, preg, MODE_ZEROING);
            vsts(vreg, dstLocal, i * 128, NORM_B16, preg);
        }
    }
}

template <>
__aicore__ inline void DuplicateImpl(__ubuf__ float* dstLocal, const float& scalarValue, const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_f32 vreg;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint16_t repeatTimes = CeilDivision(calCount, 64);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b32(sreg, POST_UPDATE);
            vdup(vreg, scalarValue, preg, MODE_ZEROING);
            vsts(vreg, dstLocal, i * 64, NORM_B32, preg);
        }
    }
}

template <>
__aicore__ inline void DuplicateImpl(__ubuf__ int16_t* dstLocal, const int16_t& scalarValue, const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_s16 vreg;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint16_t repeatTimes = CeilDivision(calCount, 128);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vdup(vreg, scalarValue, preg, MODE_ZEROING);
            vsts(vreg, dstLocal, i * 128, NORM_B16, preg);
        }
    }
}

template <>
__aicore__ inline void DuplicateImpl(__ubuf__ uint16_t* dstLocal, const uint16_t& scalarValue, const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_u16 vreg;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint16_t repeatTimes = CeilDivision(calCount, 128);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vdup(vreg, scalarValue, preg, MODE_ZEROING);
            vsts(vreg, dstLocal, i * 128, NORM_B16, preg);
        }
    }
}

template <>
__aicore__ inline void DuplicateImpl(__ubuf__ int32_t* dstLocal, const int32_t& scalarValue, const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_s32 vreg;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint16_t repeatTimes = CeilDivision(calCount, 64);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b32(sreg, POST_UPDATE);
            vdup(vreg, scalarValue, preg, MODE_ZEROING);
            vsts(vreg, dstLocal, i * 64, NORM_B32, preg);
        }
    }
}

template <>
__aicore__ inline void DuplicateImpl(__ubuf__ uint32_t* dstLocal, const uint32_t& scalarValue, const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_u32 vreg;
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg;
        uint16_t repeatTimes = CeilDivision(calCount, 64);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b32(sreg, POST_UPDATE);
            vdup(vreg, scalarValue, preg, MODE_ZEROING);
            vsts(vreg, dstLocal, i * 64, NORM_B32, preg);
        }
    }
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_DUPLICATE_IMPL_H
