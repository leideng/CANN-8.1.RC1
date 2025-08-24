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
 * \file kernel_operator_vec_cmpsel_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_CMPSEL_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_CMPSEL_IMPL_H

#include "kernel_utils.h"
#include "kernel_struct_binary.h"
#include "kernel_struct_unary.h"

namespace AscendC {
/* ***************************************************************************************
 * ************************************** Compare ****************************************
 * ************************************************************************************** */
#define CONTINUOUS_MODE_B16_VCMPV_VF(cmpMode)                                                      \
    __VEC_SCOPE__                                                                                  \
    {                                                                                              \
        vector_f16 vreg0;                                                                          \
        vector_f16 vreg1;                                                                          \
        uint32_t sreg = (uint32_t)mask;                                                            \
        vector_bool preg0 = plt_b16(sreg, POST_UPDATE);                                            \
        vector_bool preg1;                                                                         \
        uint32_t strideConfig0 = (((uint32_t)repeatParams.src0BlkStride) << 16);                   \
        uint32_t strideConfig1 = (((uint32_t)repeatParams.src1BlkStride) << 16);                   \
        uint32_t blockElm = ONE_BLK_SIZE / sizeof(half);                                         \
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {                                     \
            vsldb(vreg0, src0 + i * repeatParams.src0RepStride * blockElm, strideConfig0, preg0);  \
            vsldb(vreg1, src1 + i * repeatParams.src1RepStride * blockElm, strideConfig1, preg0);  \
            vcmp_##cmpMode(preg1, vreg0, vreg1, preg0);                                            \
            psts(preg1, ((__ubuf__ uint32_t *)dst + i * 4), 0, PK);                                \
        }                                                                                          \
    }

#define CONTINUOUS_MODE_B32_VCMPV_VF(cmpMode)                                                      \
    __VEC_SCOPE__                                                                                  \
    {                                                                                              \
        vector_f32 vreg0;                                                                          \
        vector_f32 vreg1;                                                                          \
        vector_f32 vreg2;                                                                          \
        vector_f32 vreg3;                                                                          \
        uint32_t sreg = (uint32_t)mask;                                                            \
        vector_bool preg0 = plt_b32(sreg, POST_UPDATE);                                            \
        vector_bool preg1;                                                                         \
        vector_bool preg2;                                                                         \
        vector_bool preg3;                                                                         \
        vector_bool preg4;                                                                         \
        uint32_t strideConfig0 = (((uint32_t)repeatParams.src0BlkStride) << 16);                   \
        uint32_t strideConfig1 = (((uint32_t)repeatParams.src1BlkStride) << 16);                   \
        uint32_t blockElm = ONE_BLK_SIZE / sizeof(float);                                        \
        for (uint16_t i = 0; i < (uint16_t)(repeatTimes / 2); ++i) {                               \
            vsldb(vreg0, src0 + i*2 * repeatParams.src0RepStride * blockElm,                       \
                  strideConfig0, preg0);                                                           \
            vsldb(vreg1, src1 + i*2 * repeatParams.src1RepStride * blockElm,                       \
                  strideConfig1, preg0);                                                           \
            vcmp_##cmpMode(preg1, vreg0, vreg1, preg0);                                            \
            vsldb(vreg2, src0 + (i*2 + 1) * repeatParams.src0RepStride * blockElm,                 \
                  strideConfig0, preg0);                                                           \
            vsldb(vreg3, src1 + (i*2 + 1) * repeatParams.src1RepStride * blockElm,                 \
                  strideConfig1, preg0);                                                           \
            vcmp_##cmpMode(preg2, vreg2, vreg3, preg0);                                            \
            pdintlv_b8(preg3, preg4, preg1, preg2) ;                                               \
            psts(preg3, ((__ubuf__ uint32_t *)dst + i * 4), 0, PK);                                \
        }                                                                                          \
        vector_bool preg5;                                                                         \
        vector_bool preg6;                                                                         \
        uint32_t offset0 = (repeatTimes / 2) * 2 * repeatParams.src0RepStride * blockElm;          \
        uint32_t offset1 = (repeatTimes / 2) * 2 * repeatParams.src1RepStride * blockElm;          \
        uint32_t offset2 = (repeatTimes / 2) * 4;                                                  \
        for (uint16_t i = 0; i < (uint16_t)(repeatTimes % 2); ++i) {                               \
            vsldb(vreg0, src0 + offset0, strideConfig0, preg0);                                    \
            vsldb(vreg1, src1 + offset1, strideConfig1, preg0);                                    \
            vcmp_##cmpMode(preg5, vreg0, vreg1, preg0);                                            \
            ppack(preg6, preg5, LOWER);                                                            \
            psts(preg6, ((__ubuf__ uint32_t *)dst + offset2), 0, PK);                              \
        }                                                                                          \
    }


#define BITS_MODE_B16_VCMPV_VF(cmpMode)                                                            \
    __VEC_SCOPE__                                                                                  \
    {                                                                                              \
        vector_f16 vreg0;                                                                          \
        vector_f16 vreg1;                                                                          \
        vector_bool preg0;                                                                         \
        plds(preg0, ((__ubuf__ uint32_t*)tempBuf), 0, US);                                         \
        vector_bool preg1;                                                                         \
        uint32_t strideConfig0 = (((uint32_t)repeatParams.src0BlkStride) << 16);                   \
        uint32_t strideConfig1 = (((uint32_t)repeatParams.src1BlkStride) << 16);                   \
        uint32_t blockElm = ONE_BLK_SIZE / sizeof(half);                                         \
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {                                     \
            vsldb(vreg0, src0 + i * repeatParams.src0RepStride * blockElm, strideConfig0, preg0);  \
            vsldb(vreg1, src1 + i * repeatParams.src1RepStride * blockElm, strideConfig1, preg0);  \
            vcmp_##cmpMode(preg1, vreg0, vreg1, preg0);                                            \
            psts(preg1, ((__ubuf__ uint32_t *)dst + i * 4), 0, PK);                                \
        }                                                                                          \
    }

#define BITS_MODE_B32_VCMPV_VF(cmpMode)                                                            \
    __VEC_SCOPE__                                                                                  \
    {                                                                                              \
        vector_f32 vreg0;                                                                          \
        vector_f32 vreg1;                                                                          \
        vector_f32 vreg2;                                                                          \
        vector_f32 vreg3;                                                                          \
        vector_bool preg0;                                                                         \
        plds(preg0, ((__ubuf__ uint32_t*)tempBuf), 0, US);                                         \
        vector_bool preg7;                                                                         \
        punpack(preg7, preg0, LOWER);                                                              \
        vector_bool preg1;                                                                         \
        vector_bool preg2;                                                                         \
        vector_bool preg3;                                                                         \
        vector_bool preg4;                                                                         \
        uint32_t strideConfig0 = (((uint32_t)repeatParams.src0BlkStride) << 16);                   \
        uint32_t strideConfig1 = (((uint32_t)repeatParams.src1BlkStride) << 16);                   \
        uint32_t blockElm = ONE_BLK_SIZE / sizeof(float);                                        \
        for (uint16_t i = 0; i < (uint16_t)(repeatTimes / 2); ++i) {                               \
            vsldb(vreg0, src0 + i * 2 * repeatParams.src0RepStride * blockElm,                     \
                  strideConfig0, preg7);                                                           \
            vsldb(vreg1, src1 + i * 2 * repeatParams.src1RepStride * blockElm,                     \
                  strideConfig1, preg7);                                                           \
            vcmp_##cmpMode(preg1, vreg0, vreg1, preg7);                                            \
            vsldb(vreg2, src0 + (i * 2 + 1) * repeatParams.src0RepStride * blockElm,               \
                  strideConfig0, preg7);                                                           \
            vsldb(vreg3, src1 + (i * 2 + 1) * repeatParams.src1RepStride * blockElm,               \
                  strideConfig1, preg7);                                                           \
            vcmp_##cmpMode(preg2, vreg2, vreg3, preg7);                                            \
            pdintlv_b8(preg3, preg4, preg1, preg2) ;                                               \
            psts(preg3, ((__ubuf__ uint32_t *)dst + i * 4), 0, PK);                                \
        }                                                                                          \
        vector_bool preg5;                                                                         \
        vector_bool preg6;                                                                         \
        uint32_t offset0 = (repeatTimes / 2) * 2 * repeatParams.src0RepStride * blockElm;          \
        uint32_t offset1 = (repeatTimes / 2) * 2 * repeatParams.src1RepStride * blockElm;          \
        uint32_t offset2 = (repeatTimes / 2) * 4;                                                  \
        for (uint16_t i = 0; i < (uint16_t)(repeatTimes % 2); ++i) {                               \
            vsldb(vreg0, src0 + offset0, strideConfig0, preg7);                                    \
            vsldb(vreg1, src1 + offset1, strideConfig1, preg7);                                    \
            vcmp_##cmpMode(preg5, vreg0, vreg1, preg7);                                            \
            ppack(preg6, preg5, LOWER);                                                            \
            psts(preg6, ((__ubuf__ uint32_t *)dst + offset2), 0, PK);                              \
        }                                                                                          \
    }

#define COUNTER_MODE_B16_VCMPV_VF(cmpMode)                                                         \
    __VEC_SCOPE__                                                                                  \
    {                                                                                              \
        vector_f16 vreg0;                                                                          \
        vector_f16 vreg1;                                                                          \
        vector_bool preg0;                                                                         \
        uint32_t sreg = (uint32_t)calCount;                                                        \
        vector_bool preg1;                                                                         \
        uint32_t repeatElm = VECTOR_REG_WIDTH / sizeof(half);                                      \
        uint16_t repeatTimes = CeilDivision(calCount, repeatElm);                                  \
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {                                     \
            preg0 = plt_b16(sreg, POST_UPDATE);                                                    \
            vlds(vreg0, src0, i * repeatElm, NORM);                                                \
            vlds(vreg1, src1, i * repeatElm, NORM);                                                \
            vcmp_##cmpMode(preg1, vreg0, vreg1, preg0);                                            \
            psts(preg1, ((__ubuf__ uint32_t *)dst + i * 4), 0, PK);                                \
        }                                                                                          \
    }


#define COUNTER_MODE_B32_VCMPV_VF(cmpMode)                                                         \
    __VEC_SCOPE__                                                                                  \
    {                                                                                              \
        vector_f32 vreg0;                                                                          \
        vector_f32 vreg1;                                                                          \
        vector_f32 vreg2;                                                                          \
        vector_f32 vreg3;                                                                          \
        vector_bool preg0;                                                                         \
        uint32_t sreg = (uint32_t)calCount;                                                        \
        vector_bool preg1;                                                                         \
        vector_bool preg2;                                                                         \
        vector_bool preg3;                                                                         \
        vector_bool preg4;                                                                         \
        uint32_t repeatElm = VECTOR_REG_WIDTH / sizeof(float);                                     \
        uint16_t repeatTimes = CeilDivision(calCount, repeatElm);                                  \
        for (uint16_t i = 0; i < (uint16_t)(repeatTimes / 2); ++i) {                               \
            preg0 = plt_b32(sreg, POST_UPDATE);                                                    \
            vlds(vreg0, src0, i * 2 * repeatElm, NORM);                                            \
            vlds(vreg1, src1, i * 2 * repeatElm, NORM);                                            \
            vcmp_##cmpMode(preg1, vreg0, vreg1, preg0);                                            \
            preg0 = plt_b32(sreg, POST_UPDATE);                                                    \
            vlds(vreg2, src0, (i * 2 + 1) * repeatElm, NORM);                                      \
            vlds(vreg3, src1, (i * 2 + 1) * repeatElm, NORM);                                      \
            vcmp_##cmpMode(preg2, vreg2, vreg3, preg0);                                            \
            pdintlv_b8(preg3, preg4, preg1, preg2);                                                \
            psts(preg3, ((__ubuf__ uint32_t *)dst + i * 4), 0, PK);                                \
        }                                                                                          \
        vector_bool preg5;                                                                         \
        vector_bool preg6;                                                                         \
        uint32_t offset0 = (repeatTimes / 2) * 2 * repeatElm;                                      \
        uint32_t offset1 = (repeatTimes / 2) * 2 * repeatElm;                                      \
        uint32_t offset2 = (repeatTimes / 2) * 4;                                                  \
        for (uint16_t i = 0; i < (uint16_t)(repeatTimes % 2); ++i) {                               \
            preg0 = plt_b32(sreg, POST_UPDATE);                                                    \
            vlds(vreg0, src0 + offset0, 0, NORM);                                                  \
            vlds(vreg1, src1 + offset1, 0, NORM);                                                  \
            vcmp_##cmpMode(preg5, vreg0, vreg1, preg0);                                            \
            ppack(preg6, preg5, LOWER);                                                            \
            psts(preg6, ((__ubuf__ uint32_t *)dst + offset2), 0, PK);                              \
        }                                                                                          \
    }

// Compare::Level 0 - mask bit mode
template <typename T = half, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvImpl(__ubuf__ U* dst, __ubuf__ half* src0, __ubuf__ half* src1,
    CMPMODE cmpMode, const uint64_t mask[], uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    switch (cmpMode) {
        case CMPMODE::LT: {
            BITS_MODE_B16_VCMPV_VF(lt);
            break;
        }
        case CMPMODE::GT: {
            BITS_MODE_B16_VCMPV_VF(gt);
            break;
        }
        case CMPMODE::EQ: {
            BITS_MODE_B16_VCMPV_VF(eq);
            break;
        }
        case CMPMODE::LE: {
            BITS_MODE_B16_VCMPV_VF(le);
            break;
        }
        case CMPMODE::GE: {
            BITS_MODE_B16_VCMPV_VF(ge);
            break;
        }
        case CMPMODE::NE: {
            BITS_MODE_B16_VCMPV_VF(ne);
            break;
        }
        default:
            break;
    }
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

template <typename T = float, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvImpl(__ubuf__ U* dst, __ubuf__ float* src0, __ubuf__ float* src1,
    CMPMODE cmpMode, const uint64_t mask[], uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    switch (cmpMode) {
        case CMPMODE::LT: {
            BITS_MODE_B32_VCMPV_VF(lt);
            break;
        }
        case CMPMODE::GT: {
            BITS_MODE_B32_VCMPV_VF(gt);
            break;
        }
        case CMPMODE::EQ: {
            BITS_MODE_B32_VCMPV_VF(eq);
            break;
        }
        case CMPMODE::LE: {
            BITS_MODE_B32_VCMPV_VF(le);
            break;
        }
        case CMPMODE::GE: {
            BITS_MODE_B32_VCMPV_VF(ge);
            break;
        }
        case CMPMODE::NE: {
            BITS_MODE_B32_VCMPV_VF(ne);
            break;
        }
        default:
            break;
    }
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

// Compare::Level 0 - mask normaL mode
template <typename T = half, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvImpl(__ubuf__ U* dst, __ubuf__ half* src0, __ubuf__ half* src1,
    CMPMODE cmpMode, const uint64_t mask, uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    switch (cmpMode) {
        case CMPMODE::LT: {
            CONTINUOUS_MODE_B16_VCMPV_VF(lt);
            break;
        }
        case CMPMODE::GT: {
            CONTINUOUS_MODE_B16_VCMPV_VF(gt);
            break;
        }
        case CMPMODE::EQ: {
            CONTINUOUS_MODE_B16_VCMPV_VF(eq);
            break;
        }
        case CMPMODE::LE: {
            CONTINUOUS_MODE_B16_VCMPV_VF(le);
            break;
        }
        case CMPMODE::GE: {
            CONTINUOUS_MODE_B16_VCMPV_VF(ge);
            break;
        }
        case CMPMODE::NE: {
            CONTINUOUS_MODE_B16_VCMPV_VF(ne);
            break;
        }
        default:
            break;
    }
}

template <typename T = float, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvImpl(__ubuf__ U* dst, __ubuf__ float* src0, __ubuf__ float* src1,
    CMPMODE cmpMode, const uint64_t mask, uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    switch (cmpMode) {
        case CMPMODE::LT: {
            CONTINUOUS_MODE_B32_VCMPV_VF(lt);
            break;
        }
        case CMPMODE::GT: {
            CONTINUOUS_MODE_B32_VCMPV_VF(gt);
            break;
        }
        case CMPMODE::EQ: {
            CONTINUOUS_MODE_B32_VCMPV_VF(eq);
            break;
        }
        case CMPMODE::LE: {
            CONTINUOUS_MODE_B32_VCMPV_VF(le);
            break;
        }
        case CMPMODE::GE: {
            CONTINUOUS_MODE_B32_VCMPV_VF(ge);
            break;
        }
        case CMPMODE::NE: {
            CONTINUOUS_MODE_B32_VCMPV_VF(ne);
            break;
        }
        default:
            break;
    }
}

// Compare::Level 2
template <typename U>
__aicore__ inline void VcmpvImpl(__ubuf__ U* dst, __ubuf__ half* src0, __ubuf__ half* src1,
    CMPMODE cmpMode, const uint32_t calCount)
{
    switch (cmpMode) {
        case CMPMODE::LT: {
            COUNTER_MODE_B16_VCMPV_VF(lt);
            break;
        }
        case CMPMODE::GT: {
            COUNTER_MODE_B16_VCMPV_VF(gt);
            break;
        }
        case CMPMODE::EQ: {
            COUNTER_MODE_B16_VCMPV_VF(eq);
            break;
        }
        case CMPMODE::LE: {
            COUNTER_MODE_B16_VCMPV_VF(le);
            break;
        }
        case CMPMODE::GE: {
            COUNTER_MODE_B16_VCMPV_VF(ge);
            break;
        }
        case CMPMODE::NE: {
            COUNTER_MODE_B16_VCMPV_VF(ne);
            break;
        }
        default:
            break;
    }
}

template <typename U>
__aicore__ inline void VcmpvImpl(__ubuf__ U* dst, __ubuf__ float* src0, __ubuf__ float* src1,
    CMPMODE cmpMode, const uint32_t calCount)
{
    switch (cmpMode) {
        case CMPMODE::LT: {
            COUNTER_MODE_B32_VCMPV_VF(lt);
            break;
        }
        case CMPMODE::GT: {
            COUNTER_MODE_B32_VCMPV_VF(gt);
            break;
        }
        case CMPMODE::EQ: {
            COUNTER_MODE_B32_VCMPV_VF(eq);
            break;
        }
        case CMPMODE::LE: {
            COUNTER_MODE_B32_VCMPV_VF(le);
            break;
        }
        case CMPMODE::GE: {
            COUNTER_MODE_B32_VCMPV_VF(ge);
            break;
        }
        case CMPMODE::NE: {
            COUNTER_MODE_B32_VCMPV_VF(ne);
            break;
        }
        default:
            break;
    }
}

// Compare written to CMPMASK
template <typename T, bool isSetMask = true>
__aicore__ inline void VcmpImpl(__ubuf__ T* src0, __ubuf__ T* src1, CMPMODE cmpMode,
    const uint64_t mask[], const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Compare written to CMPMASK is not supported on current device"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void VcmpImpl(__ubuf__ T* src0, __ubuf__ T* src1, CMPMODE cmpMode,
    const uint64_t mask, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Compare written to CMPMASK is not supported on current device"); });
}

/* ***************************************************************************************
 * *********************************** CompareScalar *************************************
 * ************************************************************************************** */
// CompareScalar::Level 0 - mask bit mode
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvsImpl(__ubuf__ U* dst, __ubuf__ T* src0, T src1, CMPMODE cmpMode,
    const uint64_t mask[], uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "CompareScalar is not supported on current device"); });
}

// CompareScalar::Level 0 - mask count mode
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvsImpl(__ubuf__ U* dst, __ubuf__ T* src0, T src1, CMPMODE cmpMode,
    const uint64_t mask, uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "CompareScalar is not supported on current device"); });
}

// CompareScalar::Level 2
template <typename T, typename U>
__aicore__ inline void VcmpvsImpl(__ubuf__ U* dst, __ubuf__ T* src0, T src1, CMPMODE cmpMode,
    const uint32_t calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "CompareScalar is not supported on current device"); });
}

/* ***************************************************************************************
 * *************************************** Select ****************************************
 * ************************************************************************************** */
// ============ select mode: 0/2 ============
template <typename T, typename U>
__aicore__ inline void VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, __ubuf__ T* src1,
    SELMODE selMode, const uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

template <typename U>
__aicore__ inline void VselImpl(__ubuf__ half* dst, __ubuf__ U* sel, __ubuf__ half* src0, __ubuf__ half* src1,
    SELMODE selMode, const uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    uint32_t blockElm = ONE_BLK_SIZE / sizeof(half);
    if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
        __VEC_SCOPE__
        {
            vector_f16 vreg0;
            vector_f16 vreg1;
            vector_f16 vreg2;
            uint32_t sreg = (uint32_t)mask;
            vector_bool preg0 = plt_b16(sreg, POST_UPDATE);
            vector_bool preg1;
            plds(preg1, ((__ubuf__ uint32_t *)sel), 0, US);
            uint32_t strideConfig0 = (((uint32_t)repeatParams.src0BlkStride) << 16);
            uint32_t strideConfig1 = (((uint32_t)repeatParams.src1BlkStride) << 16);
            uint32_t strideConfig2 = (((uint32_t)repeatParams.dstBlkStride) << 16);
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
                vsldb(vreg0, src0 + i * repeatParams.src0RepStride * blockElm, strideConfig0, preg0);
                vsldb(vreg1, src1 + i * repeatParams.src1RepStride * blockElm, strideConfig1, preg0);
                vsel(vreg2, vreg0, vreg1, preg1);
                vsstb(vreg2, dst + i * repeatParams.dstRepStride * blockElm, strideConfig2, preg0);
            }
        }
    } else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
        __VEC_SCOPE__
        {
            vector_f16 vreg0;
            vector_f16 vreg1;
            vector_f16 vreg2;
            uint32_t sreg = (uint32_t)mask;
            vector_bool preg0 = plt_b16(sreg, POST_UPDATE);
            vector_bool preg1;
            uint32_t strideConfig0 = (((uint32_t)repeatParams.src0BlkStride) << 16);
            uint32_t strideConfig1 = (((uint32_t)repeatParams.src1BlkStride) << 16);
            uint32_t strideConfig2 = (((uint32_t)repeatParams.dstBlkStride) << 16);
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
                plds(preg1, ((__ubuf__ uint32_t *)sel + i * 4), 0, US);
                vsldb(vreg0, src0 + i * repeatParams.src0RepStride * blockElm, strideConfig0, preg0);
                vsldb(vreg1, src1 + i * repeatParams.src1RepStride * blockElm, strideConfig1, preg0);
                vsel(vreg2, vreg0, vreg1, preg1);
                vsstb(vreg2, dst + i * repeatParams.dstRepStride * blockElm, strideConfig2, preg0);
            }
        }
    }
}

template <typename U>
__aicore__ inline void VselImpl(__ubuf__ float* dst, __ubuf__ U* sel, __ubuf__ float* src0, __ubuf__ float* src1,
    SELMODE selMode, const uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    uint32_t blockElm = ONE_BLK_SIZE / sizeof(float);
    if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
        __VEC_SCOPE__
        {
            vector_f32 vreg0;
            vector_f32 vreg1;
            vector_f32 vreg2;
            uint32_t sreg = (uint32_t)mask;
            vector_bool preg0 = plt_b32(sreg, POST_UPDATE);
            vector_bool preg1;
            vector_bool preg2;
            plds(preg1, ((__ubuf__ uint32_t *)sel), 0, US);
            punpack(preg2, preg1, LOWER);
            uint32_t strideConfig0 = (((uint32_t)repeatParams.src0BlkStride) << 16);
            uint32_t strideConfig1 = (((uint32_t)repeatParams.src1BlkStride) << 16);
            uint32_t strideConfig2 = (((uint32_t)repeatParams.dstBlkStride) << 16);
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
                vsldb(vreg0, src0 + i * repeatParams.src0RepStride * blockElm, strideConfig0, preg0);
                vsldb(vreg1, src1 + i * repeatParams.src1RepStride * blockElm, strideConfig1, preg0);
                vsel(vreg2, vreg0, vreg1, preg2);
                vsstb(vreg2, dst + i * repeatParams.dstRepStride * blockElm, strideConfig2, preg0);
            }
        }
    } else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
        __VEC_SCOPE__
        {
            vector_f32 vreg0;
            vector_f32 vreg1;
            vector_f32 vreg2;
            vector_f32 vreg3;
            vector_f32 vreg4;
            vector_f32 vreg5;
            uint32_t sreg = (uint32_t)mask;
            vector_bool preg0 = plt_b32(sreg, POST_UPDATE);
            vector_bool preg1;
            vector_bool preg2 = pset_b8(PAT_ALLF);
            vector_bool preg3;
            vector_bool preg4;
            uint32_t strideConfig0 = (((uint32_t)repeatParams.src0BlkStride) << 16);
            uint32_t strideConfig1 = (((uint32_t)repeatParams.src1BlkStride) << 16);
            uint32_t strideConfig2 = (((uint32_t)repeatParams.dstBlkStride) << 16);
            for (uint16_t i = 0; i < (uint16_t)(repeatTimes / 2); ++i) {
                plds(preg1, ((__ubuf__ uint32_t *)sel + i * 4), 0, US);
                pintlv_b16(preg3, preg4, preg1, preg2);
                vsldb(vreg0, src0 + i*2 * repeatParams.src0RepStride * blockElm, strideConfig0, preg0);
                vsldb(vreg1, src1 + i*2 * repeatParams.src1RepStride * blockElm, strideConfig1, preg0);
                vsel(vreg2, vreg0, vreg1, preg3);
                vsstb(vreg2, dst + i*2 * repeatParams.dstRepStride * blockElm, strideConfig2, preg0);
                vsldb(vreg3, src0 + (i*2+1) * repeatParams.src0RepStride * blockElm, strideConfig0, preg0);
                vsldb(vreg4, src1 + (i*2+1) * repeatParams.src1RepStride * blockElm, strideConfig1, preg0);
                vsel(vreg5, vreg3, vreg4, preg4);
                vsstb(vreg5, dst + (i*2+1) * repeatParams.dstRepStride * blockElm, strideConfig2, preg0);
            }

            vector_f32 vreg6;
            vector_f32 vreg7;
            vector_f32 vreg8;
            vector_bool preg5;
            vector_bool preg6;
            uint32_t offset0 = (repeatTimes / 2) * 2 * repeatParams.src0RepStride * blockElm;
            uint32_t offset1 = (repeatTimes / 2) * 2 * repeatParams.src1RepStride * blockElm;
            uint32_t offset2 = (repeatTimes / 2) * 2 * repeatParams.dstRepStride * blockElm;
            uint32_t selOffset = (repeatTimes / 2) * 4;
            for (uint16_t i = 0; i < (uint16_t)(repeatTimes % 2); ++i) {
                plds(preg5, ((__ubuf__ uint32_t *)sel + selOffset), 0, US);
                punpack(preg6, preg5, LOWER);
                vsldb(vreg6, src0 + offset0, strideConfig0, preg0);
                vsldb(vreg7, src1 + offset1, strideConfig1, preg0);
                vsel(vreg8, vreg6, vreg7, preg6);
                vsstb(vreg8, dst + offset2, strideConfig2, preg0);
            }
        }
    }
}

// select mode: 1
template <typename T, typename U>
__aicore__ inline void VselImpl(__ubuf__ T* dstLocal, __ubuf__ U* selMask, __ubuf__ T* src0Local, T src1Local,
    SELMODE selMode, const uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

template <typename U>
__aicore__ inline void VselImpl(__ubuf__ half* dst, __ubuf__ U* sel, __ubuf__ half* src0, half src1,
    SELMODE selMode, const uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_f16 vreg2;
        vbr(vreg1, src1);
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg0 = plt_b16(sreg, POST_UPDATE);
        vector_bool preg1;
        uint32_t strideConfig0 = (((uint32_t)repeatParams.src0BlkStride) << 16);
        uint32_t strideConfig1 = (((uint32_t)repeatParams.src1BlkStride) << 16);
        uint32_t strideConfig2 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        uint32_t blockElm = ONE_BLK_SIZE / sizeof(half);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            plds(preg1, ((__ubuf__ uint32_t *)sel + i * 4), 0, US);
            vsldb(vreg0, src0 + i * repeatParams.src0RepStride * blockElm, strideConfig0, preg0);
            vsel(vreg2, vreg0, vreg1, preg1);
            vsstb(vreg2, dst + i * repeatParams.dstRepStride * blockElm, strideConfig2, preg0);
        }
    }
}

template <typename U>
__aicore__ inline void VselImpl(__ubuf__ float* dst, __ubuf__ U* sel, __ubuf__ float* src0, float src1,
    SELMODE selMode, const uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    uint32_t blockElm = ONE_BLK_SIZE / sizeof(float);
    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        vector_f32 vreg2;
        vector_f32 vreg3;
        vector_f32 vreg4;
        vbr(vreg1, src1);
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg0 = plt_b32(sreg, POST_UPDATE);
        vector_bool preg1;
        vector_bool preg2 = pset_b8(PAT_ALLF);
        vector_bool preg3;
        vector_bool preg4;
        uint32_t strideConfig0 = (((uint32_t)repeatParams.src0BlkStride) << 16);
        uint32_t strideConfig2 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)(repeatTimes / 2); ++i) {
            plds(preg1, ((__ubuf__ uint32_t *)sel + i * 4), 0, US);
            pintlv_b16(preg3, preg4, preg1, preg2);
            vsldb(vreg0, src0 + i*2 * repeatParams.src0RepStride * blockElm, strideConfig0, preg0);
            vsel(vreg2, vreg0, vreg1, preg3);
            vsstb(vreg2, dst + i*2 * repeatParams.dstRepStride * blockElm, strideConfig2, preg0);
            vsldb(vreg3, src0 + (i*2+1) * repeatParams.src0RepStride * blockElm, strideConfig0, preg0);
            vsel(vreg4, vreg3, vreg1, preg4);
            vsstb(vreg4, dst + (i*2+1) * repeatParams.dstRepStride * blockElm, strideConfig2, preg0);
        }

        vector_f32 vreg5;
        vector_f32 vreg6;
        vector_bool preg5;
        vector_bool preg6;
        uint32_t offset0 = (repeatTimes / 2) * 2 * repeatParams.src0RepStride * blockElm;
        uint32_t offset2 = (repeatTimes / 2) * 2 * repeatParams.dstRepStride * blockElm;
        uint32_t selOffset = (repeatTimes / 2) * 4;
        for (uint16_t i = 0; i < (uint16_t)(repeatTimes % 2); ++i) {
            plds(preg5, ((__ubuf__ uint32_t *)sel + selOffset), 0, US);
            punpack(preg6, preg5, LOWER);
            vsldb(vreg5, src0 + offset0, strideConfig0, preg0);
            vsel(vreg6, vreg5, vreg1, preg6);
            vsstb(vreg6, dst + offset2, strideConfig2, preg0);
        }
    }
}

// select mode: 0/2
template <typename T, typename U>
__aicore__ inline void VselImpl(__ubuf__ T* dstLocal, __ubuf__ U* selMask, __ubuf__ T* src0Local, __ubuf__ T* src1Local,
    SELMODE selMode, const uint64_t mask[], const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

template <typename U>
__aicore__ inline void VselImpl(__ubuf__ half* dst, __ubuf__ U* sel, __ubuf__ half* src0, __ubuf__ half* src1,
    SELMODE selMode, const uint64_t mask[], const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    uint32_t blockElm = ONE_BLK_SIZE / sizeof(half);
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
        __VEC_SCOPE__
        {
            vector_f16 vreg0;
            vector_f16 vreg1;
            vector_f16 vreg2;
            vector_bool preg0;
            plds(preg0, ((__ubuf__ uint32_t*)tempBuf), 0, US);  // mask位宽扩展
            vector_bool preg1;
            plds(preg1, ((__ubuf__ uint32_t *)sel), 0, US);
            uint32_t strideConfig0 = (((uint32_t)repeatParams.src0BlkStride) << 16);
            uint32_t strideConfig1 = (((uint32_t)repeatParams.src1BlkStride) << 16);
            uint32_t strideConfig2 = (((uint32_t)repeatParams.dstBlkStride) << 16);
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
                vsldb(vreg0, src0 + i * repeatParams.src0RepStride * blockElm, strideConfig0, preg0);
                vsldb(vreg1, src1 + i * repeatParams.src1RepStride * blockElm, strideConfig1, preg0);
                vsel(vreg2, vreg0, vreg1, preg1);
                vsstb(vreg2, dst + i * repeatParams.dstRepStride * blockElm, strideConfig2, preg0);
            }
        }
    } else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
        __VEC_SCOPE__
        {
            vector_f16 vreg0;
            vector_f16 vreg1;
            vector_f16 vreg2;
            vector_bool preg0;
            plds(preg0, ((__ubuf__ uint32_t*)tempBuf), 0, US);  // mask位宽扩展
            vector_bool preg1;
            uint32_t strideConfig0 = (((uint32_t)repeatParams.src0BlkStride) << 16);
            uint32_t strideConfig1 = (((uint32_t)repeatParams.src1BlkStride) << 16);
            uint32_t strideConfig2 = (((uint32_t)repeatParams.dstBlkStride) << 16);
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
                plds(preg1, ((__ubuf__ uint32_t *)sel + i * 4), 0, US);
                vsldb(vreg0, src0 + i * repeatParams.src0RepStride * blockElm, strideConfig0, preg0);
                vsldb(vreg1, src1 + i * repeatParams.src1RepStride * blockElm, strideConfig1, preg0);
                vsel(vreg2, vreg0, vreg1, preg1);
                vsstb(vreg2, dst + i * repeatParams.dstRepStride * blockElm, strideConfig2, preg0);
            }
        }
    }
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

template <typename U>
__aicore__ inline void VselImpl(__ubuf__ float* dst, __ubuf__ U* sel, __ubuf__ float* src0, __ubuf__ float* src1,
    SELMODE selMode, const uint64_t mask[], const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    uint32_t blockElm = ONE_BLK_SIZE / sizeof(float);
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
        __VEC_SCOPE__
        {
            vector_f32 vreg0;
            vector_f32 vreg1;
            vector_f32 vreg2;
            vector_bool preg0;
            vector_bool preg3;
            plds(preg0, ((__ubuf__ uint32_t*)tempBuf), 0, US);
            punpack(preg3, preg0, LOWER);
            vector_bool preg1;
            vector_bool preg2;
            plds(preg1, ((__ubuf__ uint32_t *)sel), 0, US);
            punpack(preg2, preg1, LOWER);
            uint32_t strideConfig0 = (((uint32_t)repeatParams.src0BlkStride) << 16);
            uint32_t strideConfig1 = (((uint32_t)repeatParams.src1BlkStride) << 16);
            uint32_t strideConfig2 = (((uint32_t)repeatParams.dstBlkStride) << 16);
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
                vsldb(vreg0, src0 + i * repeatParams.src0RepStride * blockElm, strideConfig0, preg3);
                vsldb(vreg1, src1 + i * repeatParams.src1RepStride * blockElm, strideConfig1, preg3);
                vsel(vreg2, vreg0, vreg1, preg2);
                vsstb(vreg2, dst + i * repeatParams.dstRepStride * blockElm, strideConfig2, preg3);
            }
        }
    } else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
        __VEC_SCOPE__
        {
            vector_f32 vreg0;
            vector_f32 vreg1;
            vector_f32 vreg2;
            vector_f32 vreg3;
            vector_f32 vreg4;
            vector_f32 vreg5;
            vector_bool preg0;
            vector_bool preg7;
            plds(preg0, ((__ubuf__ uint32_t*)tempBuf), 0, US);
            punpack(preg7, preg0, LOWER);
            vector_bool preg1;
            vector_bool preg2 = pset_b8(PAT_ALLF);
            vector_bool preg3;
            vector_bool preg4;
            uint32_t strideConfig0 = (((uint32_t)repeatParams.src0BlkStride) << 16);
            uint32_t strideConfig1 = (((uint32_t)repeatParams.src1BlkStride) << 16);
            uint32_t strideConfig2 = (((uint32_t)repeatParams.dstBlkStride) << 16);
            for (uint16_t i = 0; i < (uint16_t)(repeatTimes / 2); ++i) {
                plds(preg1, ((__ubuf__ uint32_t *)sel + i * 4), 0, US);
                pintlv_b16(preg3, preg4, preg1, preg2);
                vsldb(vreg0, src0 + i*2 * repeatParams.src0RepStride * blockElm, strideConfig0, preg7);
                vsldb(vreg1, src1 + i*2 * repeatParams.src1RepStride * blockElm, strideConfig1, preg7);
                vsel(vreg2, vreg0, vreg1, preg3);
                vsstb(vreg2, dst + i*2 * repeatParams.dstRepStride * blockElm, strideConfig2, preg7);
                vsldb(vreg3, src0 + (i*2+1) * repeatParams.src0RepStride * blockElm, strideConfig0, preg7);
                vsldb(vreg4, src1 + (i*2+1) * repeatParams.src1RepStride * blockElm, strideConfig1, preg7);
                vsel(vreg5, vreg3, vreg4, preg4);
                vsstb(vreg5, dst + (i*2+1) * repeatParams.dstRepStride * blockElm, strideConfig2, preg7);
            }

            vector_f32 vreg6;
            vector_f32 vreg7;
            vector_f32 vreg8;
            vector_bool preg5;
            vector_bool preg6;
            uint32_t offset0 = (repeatTimes / 2) * 2 * repeatParams.src0RepStride * blockElm;
            uint32_t offset1 = (repeatTimes / 2) * 2 * repeatParams.src1RepStride * blockElm;
            uint32_t offset2 = (repeatTimes / 2) * 2 * repeatParams.dstRepStride * blockElm;
            uint32_t selOffset = (repeatTimes / 2) * 4;
            for (uint16_t i = 0; i < (uint16_t)(repeatTimes % 2); ++i) {
                plds(preg5, ((__ubuf__ uint32_t *)sel + selOffset), 0, US);
                punpack(preg6, preg5, LOWER);
                vsldb(vreg6, src0 + offset0, strideConfig0, preg7);
                vsldb(vreg7, src1 + offset1, strideConfig1, preg7);
                vsel(vreg8, vreg6, vreg7, preg6);
                vsstb(vreg8, dst + offset2, strideConfig2, preg7);
            }
        }
    }
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

// select mode: 1
template <typename T, typename U>
__aicore__ inline void VselImpl(__ubuf__ T* dstLocal, __ubuf__ U* selMask, __ubuf__ T* src0Local, T src1Local,
    SELMODE selMode, const uint64_t mask[], const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

template <typename U>
__aicore__ inline void VselImpl(__ubuf__ half* dst, __ubuf__ U* sel, __ubuf__ half* src0, half src1,
    SELMODE selMode, const uint64_t mask[], const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_f16 vreg2;
        vbr(vreg1, src1);
        vector_bool preg0;
        plds(preg0, ((__ubuf__ uint32_t*)tempBuf), 0, US);
        vector_bool preg1;
        uint32_t strideConfig0 = (((uint32_t)repeatParams.src0BlkStride) << 16);
        uint32_t strideConfig2 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        uint32_t blockElm = ONE_BLK_SIZE / sizeof(half);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            plds(preg1, ((__ubuf__ uint32_t *)sel + i * 4), 0, US);
            vsldb(vreg0, src0 + i * repeatParams.src0RepStride * blockElm, strideConfig0, preg0);
            vsel(vreg2, vreg0, vreg1, preg1);
            vsstb(vreg2, dst + i * repeatParams.dstRepStride * blockElm, strideConfig2, preg0);
        }
    }

    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

template <typename U>
__aicore__ inline void VselImpl(__ubuf__ float* dst, __ubuf__ U* sel, __ubuf__ float* src0, float src1,
    SELMODE selMode, const uint64_t mask[], const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        vector_f32 vreg2;
        vector_f32 vreg3;
        vector_f32 vreg4;
        vbr(vreg1, src1);
        vector_bool preg0;
        vector_bool preg7;
        plds(preg0, ((__ubuf__ uint32_t*)tempBuf), 0, US);  // mask位宽扩展
        punpack(preg7, preg0, LOWER);
        vector_bool preg1;
        vector_bool preg2 = pset_b8(PAT_ALLF);
        vector_bool preg3;
        vector_bool preg4;
        uint32_t strideConfig0 = (((uint32_t)repeatParams.src0BlkStride) << 16);
        uint32_t strideConfig2 = (((uint32_t)repeatParams.dstBlkStride) << 16);
        uint32_t blockElm = ONE_BLK_SIZE / sizeof(float);
        for (uint16_t i = 0; i < (uint16_t)(repeatTimes / 2); ++i) {
            plds(preg1, ((__ubuf__ uint32_t *)sel + i * 4), 0, US);
            pintlv_b16(preg3, preg4, preg1, preg2);
            vsldb(vreg0, src0 + i*2 * repeatParams.src0RepStride * blockElm, strideConfig0, preg7);
            vsel(vreg2, vreg0, vreg1, preg3);
            vsstb(vreg2, dst + i*2 * repeatParams.dstRepStride * blockElm, strideConfig2, preg7);
            vsldb(vreg3, src0 + (i*2+1) * repeatParams.src0RepStride * blockElm, strideConfig0, preg7);
            vsel(vreg4, vreg3, vreg1, preg4);
            vsstb(vreg4, dst + (i*2+1) * repeatParams.dstRepStride * blockElm, strideConfig2, preg7);
        }

        vector_f32 vreg5;
        vector_f32 vreg6;
        vector_bool preg5;
        vector_bool preg6;
        uint32_t offset0 = (repeatTimes / 2) * 2 * repeatParams.src0RepStride * blockElm;
        uint32_t offset2 = (repeatTimes / 2) * 2 * repeatParams.dstRepStride * blockElm;
        uint32_t selOffset = (repeatTimes / 2) * 4;
        for (uint16_t i = 0; i < (uint16_t)(repeatTimes % 2); ++i) {
            plds(preg5, ((__ubuf__ uint32_t *)sel + selOffset), 0, US);
            punpack(preg6, preg5, LOWER);
            vsldb(vreg5, src0 + offset0, strideConfig0, preg7);
            vsel(vreg6, vreg5, vreg1, preg6);
            vsstb(vreg6, dst + offset2, strideConfig2, preg7);
        }
    }

    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

template <typename T, SELMODE selMode>
__aicore__ inline void SelectCal(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, int32_t repeat, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

template <typename T, typename U>
__aicore__ inline void SelectCal(
    __ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0, int32_t repeat, const BinaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

template <typename T, typename U>
__aicore__ inline void VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0,
    T src1, SELMODE selMode, uint32_t calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

template <typename U>
__aicore__ inline void VselImpl(__ubuf__ half* dst, __ubuf__ U* sel, __ubuf__ half* src0,
    half src1, SELMODE selMode, uint32_t calCount)
{
    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_f16 vreg2;
        vbr(vreg1, src1);
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg0;
        vector_bool preg1;
        uint32_t repeatElm = VECTOR_REG_WIDTH / sizeof(half);
        uint16_t repeatTimes = CeilDivision(calCount, repeatElm);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg0 = plt_b16(sreg, POST_UPDATE);
            plds(preg1, ((__ubuf__ uint32_t *)sel + i * 4), 0, US);
            vlds(vreg0, src0, i * repeatElm, NORM);
            vsel(vreg2, vreg0, vreg1, preg1);
            vsts(vreg2, dst, i * repeatElm, NORM_B16, preg0);
        }
    }
}

template <typename U>
__aicore__ inline void VselImpl(__ubuf__ float* dst, __ubuf__ U* sel, __ubuf__ float* src0,
    float src1, SELMODE selMode, uint32_t calCount)
{
    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        vector_f32 vreg2;
        vector_f32 vreg3;
        vector_f32 vreg4;
        vbr(vreg1, src1);
        uint32_t sreg = (uint32_t)calCount;
        vector_bool preg0;
        vector_bool preg1;
        vector_bool preg2 = pset_b8(PAT_ALLF);
        vector_bool preg3;
        vector_bool preg4;
        uint32_t repeatElm = VECTOR_REG_WIDTH / sizeof(float);
        uint16_t repeatTimes = CeilDivision(calCount, repeatElm);
        for (uint16_t i = 0; i < (uint16_t)(repeatTimes / 2); ++i) {
            plds(preg1, ((__ubuf__ uint32_t *)sel + i * 4), 0, US);
            pintlv_b16(preg3, preg4, preg1, preg2);
            preg0 = plt_b32(sreg, POST_UPDATE);
            vlds(vreg0, src0, i*2 * repeatElm, NORM);
            vsel(vreg2, vreg0, vreg1, preg3);
            vsts(vreg2, dst, i*2 * repeatElm, NORM_B32, preg0);
            preg0 = plt_b32(sreg, POST_UPDATE);
            vlds(vreg3, src0, (i*2+1) * repeatElm, NORM);
            vsel(vreg4, vreg3, vreg1, preg4);
            vsts(vreg4, dst, (i*2+1) * repeatElm, NORM_B32, preg0);
        }

        vector_f32 vreg5;
        vector_f32 vreg6;
        vector_bool preg5;
        vector_bool preg6;
        uint32_t offset = (repeatTimes / 2) * 2 * repeatElm;
        uint32_t selOffset = (repeatTimes / 2) * 4;
        for (uint16_t i = 0; i < (uint16_t)(repeatTimes % 2); ++i) {
            preg0 = plt_b32(sreg, POST_UPDATE);
            plds(preg5, ((__ubuf__ uint32_t *)sel + selOffset), 0, US);
            punpack(preg6, preg5, LOWER);
            vlds(vreg5, src0, offset, NORM);
            vsel(vreg6, vreg5, vreg1, preg6);
            vsts(vreg6, dst, offset, NORM_B32, preg0);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void VselImpl(__ubuf__ T* dst, __ubuf__ U* sel, __ubuf__ T* src0,
    __ubuf__ T* src1, SELMODE selMode, uint32_t calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

template <typename U>
__aicore__ inline void VselImpl(__ubuf__ half* dst, __ubuf__ U* sel, __ubuf__ half* src0,
    __ubuf__ half* src1, SELMODE selMode, uint32_t calCount)
{
    if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
        __VEC_SCOPE__
        {
            vector_f16 vreg0;
            vector_f16 vreg1;
            vector_f16 vreg2;
            uint32_t sreg = (uint32_t)calCount;
            vector_bool preg0;
            vector_bool preg1;
            plds(preg1, ((__ubuf__ uint32_t *)sel), 0, US);
            uint32_t repeatElm = VECTOR_REG_WIDTH / sizeof(half);
            uint16_t repeatTimes = CeilDivision(calCount, repeatElm);
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
                preg0 = plt_b16(sreg, POST_UPDATE);
                vlds(vreg0, src0, i * repeatElm, NORM);
                vlds(vreg1, src1, i * repeatElm, NORM);
                vsel(vreg2, vreg0, vreg1, preg1);
                vsts(vreg2, dst, i * repeatElm, NORM_B16, preg0);
            }
        }
    } else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
        __VEC_SCOPE__
        {
            vector_f16 vreg0;
            vector_f16 vreg1;
            vector_f16 vreg2;
            uint32_t sreg = (uint32_t)calCount;
            vector_bool preg0;
            vector_bool preg1;
            uint32_t repeatElm = VECTOR_REG_WIDTH / sizeof(half);
            uint16_t repeatTimes = CeilDivision(calCount, repeatElm);
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
                preg0 = plt_b16(sreg, POST_UPDATE);
                plds(preg1, ((__ubuf__ uint32_t *)sel + i * 4), 0, US);
                vlds(vreg0, src0, i * repeatElm, NORM);
                vlds(vreg1, src1, i * repeatElm, NORM);
                vsel(vreg2, vreg0, vreg1, preg1);
                vsts(vreg2, dst, i * repeatElm, NORM_B16, preg0);
            }
        }
    }
}

template <typename U>
__aicore__ inline void VselImpl(__ubuf__ float* dst, __ubuf__ U* sel, __ubuf__ float* src0,
    __ubuf__ float* src1, SELMODE selMode, uint32_t calCount)
{
    if (selMode == SELMODE::VSEL_CMPMASK_SPR) {
        __VEC_SCOPE__
        {
            vector_f32 vreg0;
            vector_f32 vreg1;
            vector_f32 vreg2;
            uint32_t sreg = (uint32_t)calCount;
            vector_bool preg0;
            vector_bool preg1;
            vector_bool preg2;
            plds(preg1, ((__ubuf__ uint32_t *)sel), 0, US);
            punpack(preg2, preg1, LOWER);
            uint32_t repeatElm = VECTOR_REG_WIDTH / sizeof(float);
            uint16_t repeatTimes = CeilDivision(calCount, repeatElm);
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
                preg0 = plt_b32(sreg, POST_UPDATE);
                vlds(vreg0, src0, i * repeatElm, NORM);
                vlds(vreg1, src1, i * repeatElm, NORM);
                vsel(vreg2, vreg0, vreg1, preg2);
                vsts(vreg2, dst, i * repeatElm, NORM_B32, preg0);
            }
        }
    } else if (selMode == SELMODE::VSEL_TENSOR_TENSOR_MODE) {
        __VEC_SCOPE__
        {
            vector_f32 vreg0;
            vector_f32 vreg1;
            vector_f32 vreg2;
            uint32_t sreg = (uint32_t)calCount;
            vector_f32 vreg3;
            vector_f32 vreg4;
            vector_f32 vreg5;
            vector_bool preg0;
            vector_bool preg1;
            vector_bool preg2 = pset_b8(PAT_ALLF);
            vector_bool preg3;
            vector_bool preg4;
            uint32_t repeatElm = VECTOR_REG_WIDTH / sizeof(float);
            uint16_t repeatTimes = CeilDivision(calCount, repeatElm);
            for (uint16_t i = 0; i < (uint16_t)(repeatTimes / 2); ++i) {
                plds(preg1, ((__ubuf__ uint32_t *)sel + i * 4), 0, US);
                pintlv_b16(preg3, preg4, preg1, preg2);
                preg0 = plt_b32(sreg, POST_UPDATE);
                vlds(vreg0, src0, i * 2 * repeatElm, NORM);
                vlds(vreg1, src1, i * 2 * repeatElm, NORM);
                vsel(vreg2, vreg0, vreg1, preg3);
                vsts(vreg2, dst, i * 2 * repeatElm, NORM_B32, preg0);
                preg0 = plt_b32(sreg, POST_UPDATE);
                vlds(vreg3, src0, (i*2+1) * repeatElm, NORM);
                vlds(vreg4, src1, (i*2+1) * repeatElm, NORM);
                vsel(vreg5, vreg3, vreg4, preg4);
                vsts(vreg5, dst, (i*2+1) * repeatElm, NORM_B32, preg0);
            }

            vector_f32 vreg6;
            vector_f32 vreg7;
            vector_f32 vreg8;
            vector_bool preg5;
            vector_bool preg6;
            uint32_t offset = (repeatTimes / 2) * 2 * repeatElm;
            uint32_t selOffset = (repeatTimes / 2) * 4;
            for (uint16_t i = 0; i < (uint16_t)(repeatTimes % 2); ++i) {
                plds(preg5, ((__ubuf__ uint32_t *)sel + selOffset), 0, US);
                punpack(preg6, preg5, LOWER);
                preg0 = plt_b32(sreg, POST_UPDATE);
                vlds(vreg6, src0, offset, NORM);
                vlds(vreg7, src1, offset, NORM);
                vsel(vreg8, vreg6, vreg7, preg6);
                vsts(vreg8, dst, offset, NORM_B32, preg0);
            }
        }
    }
}

template <typename T>
__aicore__ inline void GetCmpMaskImpl(__ubuf__ T* dst)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "GetCmpMask is not supported on current device"); });
}

template <typename T>
__aicore__ inline void SetCmpMaskImpl(__ubuf__ T* src)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "GetCmpMask is not supported on current device"); });
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_CMPSEL_IMPL_H