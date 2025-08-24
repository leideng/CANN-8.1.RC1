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
 * \file kernel_operator_vec_ternary_scalar_impl.h
 * \brief AscendC v300 support vaxpy level 0/2 api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_TERNARY_SCALAR_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_TERNARY_SCALAR_IMPL_H
#include "kernel_operator_common_impl.h"
#include "kernel_utils.h"
#include "kernel_struct_unary.h"

namespace AscendC {
#define NORMAL_AXPY_IMPL(dst, src, scalarValue, repeatTimes, repeatParams, preg, data_bits)                     \
    vector_f##data_bits vreg0;                                                                                  \
    vector_f##data_bits vreg1;                                                                                  \
    vector_f##data_bits vreg2;                                                                                  \
    vector_f##data_bits temp_vreg;                                                                              \
    uint32_t src_sm = (((uint32_t)repeatParams.srcBlkStride) << BLOCK_STRIDE_POS_IN_SM);                        \
    uint32_t dst_sm = (((uint32_t)repeatParams.dstBlkStride) << BLOCK_STRIDE_POS_IN_SM);                        \
    for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {                                                      \
        vsldb(vreg0, src + i * repeatParams.srcRepStride * B##data_bits##_DATA_NUM_PER_BLOCK, src_sm, preg);    \
        vmuls(temp_vreg, vreg0, scalarValue, preg, MODE_ZEROING);                                               \
        vsldb(vreg1, dst + i * repeatParams.dstRepStride * B##data_bits##_DATA_NUM_PER_BLOCK, dst_sm, preg);    \
        vadd(vreg2, temp_vreg, vreg1, preg, MODE_ZEROING);                                                      \
        vsstb(vreg2, dst + i * repeatParams.dstRepStride * B##data_bits##_DATA_NUM_PER_BLOCK, dst_sm, preg);    \
    }                                                                                                           \

#define COUNTER_AXPY_IMPL(dst, src, scalarValue, calCount, data_bits)                       \
    vector_f##data_bits vreg0;                                                              \
    vector_f##data_bits vreg1;                                                              \
    vector_f##data_bits vreg2;                                                              \
    vector_f##data_bits temp_vreg;                                                          \
    vector_bool preg;                                                                       \
    uint32_t sreg = (uint32_t)calCount;                                                     \
    uint16_t repeatTimes = CeilDivision(calCount, B##data_bits##_DATA_NUM_PER_REPEAT);      \
    for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {                                  \
        preg = plt_b##data_bits(sreg, POST_UPDATE);                                         \
        vlds(vreg0, src, i * B##data_bits##_DATA_NUM_PER_REPEAT, NORM);                     \
        vmuls(temp_vreg, vreg0, scalarValue, preg, MODE_ZEROING);                           \
        vlds(vreg1, dst, i * B##data_bits##_DATA_NUM_PER_REPEAT, NORM);                     \
        vadd(vreg2, temp_vreg, vreg1, preg, MODE_ZEROING);                                  \
        vsts(vreg2, dst, i * B##data_bits##_DATA_NUM_PER_REPEAT, NORM_B##data_bits, preg);  \
    }                                                                                       \

#define MIX_AXPY_IMPL(dst, src, scalarValue, repeatTimes, repeatParams, preg)                           \
    vector_f16 src_vreg;                                                                                \
    vector_f16 tmp_vreg;                                                                                \
    vector_f16 zero_vreg;                                                                               \
    vector_f32 cvt_vreg;                                                                                \
    vector_f32 dst_vreg;                                                                                \
    vector_f32 add_vreg;                                                                                \
    vector_f32 mul_vreg;                                                                                \
    vector_bool full_preg;                                                                              \
    uint32_t full_sreg = FULL_MASK_LEN;                                                                 \
    full_preg = plt_b16(full_sreg, POST_UPDATE);                                                        \
    vdup(zero_vreg, 0, full_preg, MODE_ZEROING);                                                        \
    uint32_t src_sm = (((uint32_t)repeatParams.srcBlkStride) << BLOCK_STRIDE_POS_IN_SM);                \
    uint32_t dst_sm = (((uint32_t)repeatParams.dstBlkStride) << BLOCK_STRIDE_POS_IN_SM);                \
    for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {                                              \
        vsldb(src_vreg, src + i * repeatParams.srcRepStride * B16_DATA_NUM_PER_BLOCK, src_sm, preg);    \
        vintlv(src_vreg, tmp_vreg, src_vreg, zero_vreg);                                                \
        vcvt(cvt_vreg, src_vreg, preg, PART_EVEN);                                                      \
        vmuls(mul_vreg, cvt_vreg, scalarValue, preg, MODE_ZEROING);                                     \
        vsldb(dst_vreg, dst + i * repeatParams.dstRepStride * B32_DATA_NUM_PER_BLOCK, dst_sm, preg);    \
        vadd(add_vreg, mul_vreg, dst_vreg, preg, MODE_ZEROING);                                         \
        vsstb(add_vreg, dst + i * repeatParams.dstRepStride * B32_DATA_NUM_PER_BLOCK, dst_sm, preg);    \
    }                                                                                                   \


__aicore__ inline void AxpyIntrinsicsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
                                          uint64_t mask, const uint8_t repeatTimes,
                                          const UnaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        vector_bool preg;
        uint32_t sreg = (uint32_t)mask;
        preg = plt_b16(sreg, POST_UPDATE);
        NORMAL_AXPY_IMPL(dst, src, scalarValue, repeatTimes, repeatParams, preg, 16);
    }
}

__aicore__ inline void AxpyIntrinsicsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
                                          uint64_t mask, const uint8_t repeatTimes,
                                          const UnaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        vector_bool preg;
        uint32_t sreg = (uint32_t)mask;
        preg = plt_b32(sreg, POST_UPDATE);
        NORMAL_AXPY_IMPL(dst, src, scalarValue, repeatTimes, repeatParams, preg, 32);
    }
}

__aicore__ inline void AxpyIntrinsicsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
                                          uint64_t mask[], const uint8_t repeatTimes,
                                          const UnaryRepeatParams& repeatParams)
{
    __ubuf__ uint64_t* tmpBuffer = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(TMP_UB_OFFSET, PLD_BUFFER_SIZE);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tmpBuffer)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tmpBuffer + 1)) = ((uint64_t)mask[1]);
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    __VEC_SCOPE__
    {
        vector_bool preg;
        plds(preg, ((__ubuf__ uint32_t*)tmpBuffer), 0, US);
        NORMAL_AXPY_IMPL(dst, src, scalarValue, repeatTimes, repeatParams, preg, 16);
    }
    AscendCUtils::FreeTemporaryBuffer<uint64_t>(tmpBuffer);
}

__aicore__ inline void AxpyIntrinsicsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
                                          uint64_t mask[], const uint8_t repeatTimes,
                                          const UnaryRepeatParams& repeatParams)
{
    __ubuf__ uint64_t* pldBuffer = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(TMP_UB_OFFSET, PLD_BUFFER_SIZE);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)pldBuffer)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)pldBuffer + 1)) = ((uint64_t)mask[1]);
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    __VEC_SCOPE__
    {
        vector_bool preg;
        plds(preg, ((__ubuf__ uint32_t*)pldBuffer), 0, US);
        punpack(preg, preg, LOWER);
        NORMAL_AXPY_IMPL(dst, src, scalarValue, repeatTimes, repeatParams, preg, 32);
    }
    AscendCUtils::FreeTemporaryBuffer<uint64_t>(pldBuffer);
}

__aicore__ inline void AxpyIntrinsicsImpl(__ubuf__ half* dst, __ubuf__ half* src, half scalarValue,
                                          const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        COUNTER_AXPY_IMPL(dst, src, scalarValue, calCount, 16);
    }
}

__aicore__ inline void AxpyIntrinsicsImpl(__ubuf__ float* dst, __ubuf__ float* src, float scalarValue,
                                          const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        COUNTER_AXPY_IMPL(dst, src, scalarValue, calCount, 32);
    }
}

__aicore__ inline void AxpyFmixImpl(__ubuf__ float* dst, __ubuf__ half* src, half scalarValue,
                                    uint64_t mask, const uint8_t repeatTimes,
                                    const UnaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        vector_bool preg;
        uint32_t sreg = (uint32_t)mask;
        preg = plt_b32(sreg, POST_UPDATE);
        MIX_AXPY_IMPL(dst, src, scalarValue, repeatTimes, repeatParams, preg);
    }
}

__aicore__ inline void AxpyFmixImpl(__ubuf__ float* dst, __ubuf__ half* src, half scalarValue,
                                    uint64_t mask[], const uint8_t repeatTimes,
                                    const UnaryRepeatParams& repeatParams)
{
    __ubuf__ uint64_t* pldsBuffer = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(TMP_UB_OFFSET, PLD_BUFFER_SIZE);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)pldsBuffer)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)pldsBuffer + 1)) = ((uint64_t)mask[1]);
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    __VEC_SCOPE__
    {
        vector_bool preg;
        plds(preg, ((__ubuf__ uint32_t*)pldsBuffer), 0, US);
        punpack(preg, preg, LOWER);
        MIX_AXPY_IMPL(dst, src, scalarValue, repeatTimes, repeatParams, preg);
    }
    AscendCUtils::FreeTemporaryBuffer<uint64_t>(pldsBuffer);
}

__aicore__ inline void AxpyFmixImpl(__ubuf__ float* dst, __ubuf__ half* src, half scalarValue,
                                    const int32_t& calCount)
{
    __VEC_SCOPE__ {
        vector_f16 src_vreg;
        vector_f16 tmp_vreg;
        vector_f16 zero_vreg;
        vector_f32 cvt_vreg;
        vector_f32 dst_vreg;
        vector_f32 add_vreg;
        vector_f32 mul_vreg;
        vector_bool preg;
        vector_bool full_preg;
        uint32_t full_sreg = FULL_MASK_LEN;
        full_preg = plt_b16(full_sreg, POST_UPDATE);
        vdup(zero_vreg, 0, full_preg, MODE_ZEROING);
        uint32_t sreg = (uint32_t)calCount;
        uint16_t repeatTimes = CeilDivision(calCount, B32_DATA_NUM_PER_REPEAT);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b32(sreg, POST_UPDATE);
            vlds(src_vreg, src, i * B32_DATA_NUM_PER_REPEAT, NORM);
            vintlv(src_vreg, tmp_vreg, src_vreg, zero_vreg);
            vcvt(cvt_vreg, src_vreg, preg, PART_EVEN);
            vmuls(mul_vreg, cvt_vreg, scalarValue, preg, MODE_ZEROING);
            vlds(dst_vreg, dst, i * B32_DATA_NUM_PER_REPEAT, NORM);
            vadd(add_vreg, mul_vreg, dst_vreg, preg, MODE_ZEROING);
            vsts(add_vreg, dst, i * B32_DATA_NUM_PER_REPEAT, NORM_B32, preg);
        }
    }
}

// Axpy::Level 0
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void AxpyImpl(__ubuf__ T* dst, __ubuf__ U* src, const U& scalarValue,
                                uint64_t mask[], const uint8_t repeatTimes,
                                const UnaryRepeatParams& repeatParams)
{
    if constexpr (sizeof(T) == sizeof(U)) {
        return AxpyIntrinsicsImpl(dst, src, scalarValue, mask, repeatTimes, repeatParams);
    } else if constexpr (sizeof(T) > sizeof(U)) {
        return AxpyFmixImpl(dst, src, scalarValue, mask, repeatTimes, repeatParams);
    }
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void AxpyImpl(__ubuf__ T* dst, __ubuf__ U* src, const U& scalarValue,
                                uint64_t mask, const uint8_t repeatTimes,
                                const UnaryRepeatParams& repeatParams)
{
    if constexpr (sizeof(T) == sizeof(U)) {
        return AxpyIntrinsicsImpl(dst, src, scalarValue, mask, repeatTimes, repeatParams);
    } else if constexpr (sizeof(T) > sizeof(U)) {
        return AxpyFmixImpl(dst, src, scalarValue, mask, repeatTimes, repeatParams);
    }
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

// Add::Level 2
template <typename T, typename U>
__aicore__ inline void AxpyImpl(__ubuf__ T* dst, __ubuf__ U* src, const U& scalarValue,
                                const int32_t& calCount)
{
    if constexpr (sizeof(T) == sizeof(U)) {
        return AxpyIntrinsicsImpl(dst, src, scalarValue, calCount);
    } else if constexpr (sizeof(T) > sizeof(U)) {
        return AxpyFmixImpl(dst, src, scalarValue, calCount);
    }
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}
}  // namespace AscendC
#endif  // ASCENDC_MODULE_OPERATOR_VEC_TERNARY_SCALAR_IMPL_H