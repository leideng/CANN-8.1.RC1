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
 * \file kernel_operator_vec_binary_continuous_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_BINARY_CONTINUOUS_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_BINARY_CONTINUOUS_IMPL_H
#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"

namespace AscendC {
/* **************************************************************************************************
 * Add                                             *
 * ************************************************************************************************* */
// Add::Level 2
template <typename T>
__aicore__ inline void AddImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

__aicore__ inline void AddImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_f16 vreg2;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        vector_bool preg;
        uint32_t sregLower = 128;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, 128));
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, src0, i * sregLower, NORM);
            vlds(vreg1, src1, i * sregLower, NORM);
            vadd(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vsts(vreg2, dst, i * sregLower, NORM_B16, preg);
        }
    }
}

__aicore__ inline void AddImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_s16 vreg0;
        vector_s16 vreg1;
        vector_s16 vreg2;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        vector_bool preg;
        uint32_t sregLower = 128;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, 128));
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, src0, i * sregLower, NORM);
            vlds(vreg1, src1, i * sregLower, NORM);
            vadd(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vsts(vreg2, dst, i * sregLower, NORM_B16, preg);
        }
    }
}

__aicore__ inline void AddImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_s32 vreg0;
        vector_s32 vreg1;
        vector_s32 vreg2;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        vector_bool preg;
        uint32_t sregLower = 64;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, 64));
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = plt_b32(sreg, POST_UPDATE);
            vlds(vreg0, src0, i * sregLower, NORM);
            vlds(vreg1, src1, i * sregLower, NORM);
            vadd(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vsts(vreg2, dst, i * sregLower, NORM_B32, preg);
        }
    }
}

__aicore__ inline void AddImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        vector_f32 vreg2;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        vector_bool preg;
        uint32_t sregLower = 64;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, 64));
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = plt_b32(sreg, POST_UPDATE);
            vlds(vreg0, src0, i * sregLower, NORM);
            vlds(vreg1, src1, i * sregLower, NORM);
            vadd(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vsts(vreg2, dst, i * sregLower, NORM_B32, preg);
        }
    }
}

/* **************************************************************************************************
 * Sub                                             *
 * ************************************************************************************************* */
// Sub::Level 2
template <typename T>
__aicore__ inline void SubImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

__aicore__ inline void SubImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_f16 vreg2;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        vector_bool preg;
        uint32_t sregLower = 128;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, 128));
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, src0, i * sregLower, NORM);
            vlds(vreg1, src1, i * sregLower, NORM);
            vsub(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vsts(vreg2, dst, i * sregLower, NORM_B16, preg);
        }
    }
}

__aicore__ inline void SubImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_s16 vreg0;
        vector_s16 vreg1;
        vector_s16 vreg2;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        vector_bool preg;
        uint32_t sregLower = 128;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, 128));
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, src0, i * sregLower, NORM);
            vlds(vreg1, src1, i * sregLower, NORM);
            vsub(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vsts(vreg2, dst, i * sregLower, NORM_B16, preg);
        }
    }
}

__aicore__ inline void SubImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_s32 vreg0;
        vector_s32 vreg1;
        vector_s32 vreg2;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        vector_bool preg;
        uint32_t sregLower = 64;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, 64));
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = plt_b32(sreg, POST_UPDATE);
            vlds(vreg0, src0, i * sregLower, NORM);
            vlds(vreg1, src1, i * sregLower, NORM);
            vsub(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vsts(vreg2, dst, i * sregLower, NORM_B32, preg);
        }
    }
}

__aicore__ inline void SubImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        vector_f32 vreg2;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        vector_bool preg;
        uint32_t sregLower = 64;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, 64));
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = plt_b32(sreg, POST_UPDATE);

            vlds(vreg0, src0, i * sregLower, NORM);
            vlds(vreg1, src1, i * sregLower, NORM);
            vsub(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vsts(vreg2, dst, i * sregLower, NORM_B32, preg);
        }
    }
}

/* **************************************************************************************************
 * Mul                                             *
 * ************************************************************************************************* */
// Mul::Level 2
template <typename T>
__aicore__ inline void MulImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

__aicore__ inline void MulImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_f16 vreg2;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        vector_bool preg;
        uint32_t sregLower = 128;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, 128));
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, src0, i * sregLower, NORM);
            vlds(vreg1, src1, i * sregLower, NORM);
            vmul(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vsts(vreg2, dst, i * sregLower, NORM_B16, preg);
        }
    }
}

__aicore__ inline void MulImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_s16 vreg0;
        vector_s16 vreg1;
        vector_s16 vreg2;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        vector_bool preg;
        uint32_t sregLower = 128;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, 128));
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, src0, i * sregLower, NORM);
            vlds(vreg1, src1, i * sregLower, NORM);
            vmul(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vsts(vreg2, dst, i * sregLower, NORM_B16, preg);
        }
    }
}

__aicore__ inline void MulImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_s32 vreg0;
        vector_s32 vreg1;
        vector_s32 vreg2;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        vector_bool preg;
        uint32_t sregLower = 64;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, 64));
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = plt_b32(sreg, POST_UPDATE);
            vlds(vreg0, src0, i * sregLower, NORM);
            vlds(vreg1, src1, i * sregLower, NORM);
            vmul(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vsts(vreg2, dst, i * sregLower, NORM_B32, preg);
        }
    }
}

__aicore__ inline void MulImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        vector_f32 vreg2;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        vector_bool preg;
        uint32_t sregLower = 64;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, 64));
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = plt_b32(sreg, POST_UPDATE);
            vlds(vreg0, src0, i * sregLower, NORM);
            vlds(vreg1, src1, i * sregLower, NORM);
            vmul(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vsts(vreg2, dst, i * sregLower, NORM_B32, preg);
        }
    }
}

/* **************************************************************************************************
 * Div                                             *
 * ************************************************************************************************* */
// Div::Level 2
template <typename T>
__aicore__ inline void DivImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

__aicore__ inline void DivImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_f16 vreg2;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        vector_bool preg;
        uint32_t sregLower = 128;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, 128));
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, src0, i * sregLower, NORM);
            vlds(vreg1, src1, i * sregLower, NORM);
            vdiv(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vsts(vreg2, dst, i * sregLower, NORM_B16, preg);
        }
    }
}

__aicore__ inline void DivImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        vector_f32 vreg2;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        vector_bool preg;
        uint32_t sregLower = 64;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, 64));
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = plt_b32(sreg, POST_UPDATE);
            vlds(vreg0, src0, i * sregLower, NORM);
            vlds(vreg1, src1, i * sregLower, NORM);
            vdiv(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vsts(vreg2, dst, i * sregLower, NORM_B32, preg);
        }
    }
}

/* **************************************************************************************************
 * Max                                             *
 * ************************************************************************************************* */
// Max::Level 2
template <typename T>
__aicore__ inline void MaxImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

__aicore__ inline void MaxImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_f16 vreg2;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        vector_bool preg;
        uint32_t sregLower = 128;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, 128));
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, src0, i * sregLower, NORM);
            vlds(vreg1, src1, i * sregLower, NORM);
            vmax(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vsts(vreg2, dst, i * sregLower, NORM_B16, preg);
        }
    }
}

__aicore__ inline void MaxImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_s16 vreg0;
        vector_s16 vreg1;
        vector_s16 vreg2;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        vector_bool preg;
        uint32_t sregLower = 128;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, 128));
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, src0, i * sregLower, NORM);
            vlds(vreg1, src1, i * sregLower, NORM);
            vmax(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vsts(vreg2, dst, i * sregLower, NORM_B16, preg);
        }
    }
}

__aicore__ inline void MaxImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_s32 vreg0;
        vector_s32 vreg1;
        vector_s32 vreg2;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        vector_bool preg;
        uint32_t sregLower = 64;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, 64));
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = plt_b32(sreg, POST_UPDATE);
            vlds(vreg0, src0, i * sregLower, NORM);
            vlds(vreg1, src1, i * sregLower, NORM);
            vmax(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vsts(vreg2, dst, i * sregLower, NORM_B32, preg);
        }
    }
}

__aicore__ inline void MaxImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        vector_f32 vreg2;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        vector_bool preg;
        uint32_t sregLower = 64;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, 64));
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = plt_b32(sreg, POST_UPDATE);
            vlds(vreg0, src0, i * sregLower, NORM);
            vlds(vreg1, src1, i * sregLower, NORM);
            vmax(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vsts(vreg2, dst, i * sregLower, NORM_B32, preg);
        }
    }
}

/* **************************************************************************************************
 * Min                                             *
 * ************************************************************************************************* */
// Min::Level 2
template <typename T>
__aicore__ inline void MinImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

__aicore__ inline void MinImpl(__ubuf__ half* dst, __ubuf__ half* src0, __ubuf__ half* src1, const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_f16 vreg2;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        vector_bool preg;
        uint32_t sregLower = 128;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, 128));
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, src0, i * sregLower, NORM);
            vlds(vreg1, src1, i * sregLower, NORM);
            vmin(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vsts(vreg2, dst, i * sregLower, NORM_B16, preg);
        }
    }
}

__aicore__ inline void MinImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_s16 vreg0;
        vector_s16 vreg1;
        vector_s16 vreg2;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        vector_bool preg;
        uint32_t sregLower = 128;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, 128));
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, src0, i * sregLower, NORM);
            vlds(vreg1, src1, i * sregLower, NORM);
            vmin(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vsts(vreg2, dst, i * sregLower, NORM_B16, preg);
        }
    }
}

__aicore__ inline void MinImpl(__ubuf__ int32_t* dst, __ubuf__ int32_t* src0, __ubuf__ int32_t* src1,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_s32 vreg0;
        vector_s32 vreg1;
        vector_s32 vreg2;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        vector_bool preg;
        uint32_t sregLower = 64;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, 64));
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = plt_b32(sreg, POST_UPDATE);
            vlds(vreg0, src0, i * sregLower, NORM);
            vlds(vreg1, src1, i * sregLower, NORM);
            vmin(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vsts(vreg2, dst, i * sregLower, NORM_B32, preg);
        }
    }
}

__aicore__ inline void MinImpl(__ubuf__ float* dst, __ubuf__ float* src0, __ubuf__ float* src1, const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        vector_f32 vreg2;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        vector_bool preg;
        uint32_t sregLower = 64;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, 64));
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = plt_b32(sreg, POST_UPDATE);
            vlds(vreg0, src0, i * sregLower, NORM);
            vlds(vreg1, src1, i * sregLower, NORM);
            vmin(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vsts(vreg2, dst, i * sregLower, NORM_B32, preg);
        }
    }
}

/* **************************************************************************************************
 * And                                             *
 * ************************************************************************************************* */
// And::Level 2
template <typename T>
__aicore__ inline void AndImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

__aicore__ inline void AndImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_s16 vreg0;
        vector_s16 vreg1;
        vector_s16 vreg2;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        vector_bool preg;
        uint32_t sregLower = 128;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, 128));
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, src0, i * sregLower, NORM);
            vlds(vreg1, src1, i * sregLower, NORM);
            vand(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vsts(vreg2, dst, i * sregLower, NORM_B16, preg);
        }
    }
}

__aicore__ inline void AndImpl(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src0, __ubuf__ uint16_t* src1,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_u16 vreg0;
        vector_u16 vreg1;
        vector_u16 vreg2;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        vector_bool preg;
        uint32_t sregLower = 128;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, 128));
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, src0, i * sregLower, NORM);
            vlds(vreg1, src1, i * sregLower, NORM);
            vand(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vsts(vreg2, dst, i * sregLower, NORM_B16, preg);
        }
    }
}

/* **************************************************************************************************
 * Or                                             *
 * ************************************************************************************************* */
// Or::Level 2
template <typename T>
__aicore__ inline void OrImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

__aicore__ inline void OrImpl(__ubuf__ int16_t* dst, __ubuf__ int16_t* src0, __ubuf__ int16_t* src1,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_s16 vreg0;
        vector_s16 vreg1;
        vector_s16 vreg2;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        vector_bool preg;
        uint32_t sregLower = 128;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, 128));
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, src0, i * sregLower, NORM);
            vlds(vreg1, src1, i * sregLower, NORM);
            vor(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vsts(vreg2, dst, i * sregLower, NORM_B16, preg);
        }
    }
}

__aicore__ inline void OrImpl(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src0, __ubuf__ uint16_t* src1,
    const int32_t& calCount)
{
    __VEC_SCOPE__
    {
        vector_u16 vreg0;
        vector_u16 vreg1;
        vector_u16 vreg2;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        vector_bool preg;
        uint32_t sregLower = 128;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, 128));
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            preg = plt_b16(sreg, POST_UPDATE);
            vlds(vreg0, src0, i * sregLower, NORM);
            vlds(vreg1, src1, i * sregLower, NORM);
            vor(vreg2, vreg0, vreg1, preg, MODE_ZEROING);
            vsts(vreg2, dst, i * sregLower, NORM_B16, preg);
        }
    }
}
/* **************************************************************************************************
 * FusedMulAdd                                             *
 * ************************************************************************************************* */
// FusedMulAdd::Level 2
template <typename T>
__aicore__ inline void FusedMulAddImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "FusedMulAdd is not supported on current device!"); });
}

/* **************************************************************************************************
 * FusedMulAddRelu                                             *
 * ************************************************************************************************* */
// FusedMulAddRelu::Level 2
template <typename T>
__aicore__ inline void FusedMulAddReluImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1,
    const int32_t& calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "FusedMulAddRelu is not supported on current device!"); });
}
/* **************************************************************************************************
 * MulAddDst                                             *
 * ************************************************************************************************* */
// MulAddDst::Level 2
template <typename T, typename U>
__aicore__ inline void MulAddDstImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1, const int32_t& calCount)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "MulAddDst is not supported on current device!"); });
}

} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_BINARY_CONTINUOUS_IMPL_H
