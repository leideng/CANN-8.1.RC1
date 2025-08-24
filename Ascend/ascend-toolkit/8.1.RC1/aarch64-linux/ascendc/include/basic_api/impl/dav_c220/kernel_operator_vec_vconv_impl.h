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
 * \file kernel_operator_vec_vconv_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_VCONV_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_VCONV_IMPL_H
#include "kernel_utils.h"
#include "kernel_struct_binary.h"
#include "kernel_struct_unary.h"
#include "kernel_struct_vdeq.h"

namespace AscendC {
__aicore__ inline void CastIntrinsicsImpl(__ubuf__ half* dst, __ubuf__ int32_t* src, const RoundMode& roundMode,
    uint8_t repeat, const UnaryRepeatParams& repeatParams)
{
    vconv_deq(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride, repeatParams.dstRepStride,
        repeatParams.srcRepStride);
}

__aicore__ inline void CastIntrinsicsImpl(__ubuf__ half* dst, __ubuf__ int8_t* src, const RoundMode& roundMode,
    uint8_t repeat, const UnaryRepeatParams& repeatParams)
{
    switch (roundMode) {
        case RoundMode::CAST_NONE:
            vconv_s82f16(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        default:
            ASCENDC_ASSERT((false), {
                KERNEL_LOG(KERNEL_ERROR,
                    "illegal input cast mode %d, only support CAST_NONE from int8_t to half on current device",
                    static_cast<int32_t>(roundMode));
            });
            break;
    }
}

__aicore__ inline void CastIntrinsicsImpl(__ubuf__ half* dst, __ubuf__ uint8_t* src, const RoundMode& roundMode,
    uint8_t repeat, const UnaryRepeatParams& repeatParams)
{
    switch (roundMode) {
        case RoundMode::CAST_NONE:
            vconv_u82f16(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        default:
            ASCENDC_ASSERT((false), {
                KERNEL_LOG(KERNEL_ERROR,
                    "illegal input cast mode %d, only support CAST_NONE from uint8_t to half on current device",
                    static_cast<int32_t>(roundMode));
            });
            break;
    }
}

__aicore__ inline void CastIntrinsicsImpl(__ubuf__ float* dst, __ubuf__ int32_t* src, const RoundMode& roundMode,
    uint8_t repeat, const UnaryRepeatParams& repeatParams)
{
    switch (roundMode) {
        case RoundMode::CAST_RINT:
            vconv_s322f32r(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_s322f32f(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_s322f32c(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_s322f32a(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_s322f32z(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_NONE:
            vconv_s322f32(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ODD:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "CAST_ODD from int32_t to float not supported on current device"); });
            break;
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            break;
    }
}

__aicore__ inline void CastIntrinsicsImpl(__ubuf__ float* dst, __ubuf__ half* src, const RoundMode& roundMode,
    uint8_t repeat, const UnaryRepeatParams& repeatParams)
{
    switch (roundMode) {
        case RoundMode::CAST_NONE:
            vconv_f162f32(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        default:
            ASCENDC_ASSERT((false), {
                KERNEL_LOG(KERNEL_ERROR,
                    "illegal input cast mode %d, only support CAST_NONE from half to float on current device",
                    static_cast<int32_t>(roundMode));
            });
            break;
    }
}

__aicore__ inline void CastIntrinsicsImpl(__ubuf__ int32_t* dst, __ubuf__ half* src, const RoundMode& roundMode,
    uint8_t repeat, const UnaryRepeatParams& repeatParams)
{
    switch (roundMode) {
        case RoundMode::CAST_RINT:
            vconv_f162s32r(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f162s32f(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f162s32c(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f162s32a(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f162s32z(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ODD:
            ASCENDC_ASSERT((false),
                           { KERNEL_LOG(KERNEL_ERROR, "CAST_ODD from half to int32_t not supported on current device"); });
            break;
        case RoundMode::CAST_NONE:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "CAST_NONE from half to int32_t not supported on current device"); });
            break;
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            break;
    }
}

__aicore__ inline void CastIntrinsicsImpl(__ubuf__ int8_t* dst, __ubuf__ half* src, const RoundMode& roundMode,
    uint8_t repeat, const UnaryRepeatParams& repeatParams)
{
    switch (roundMode) {
        case RoundMode::CAST_RINT:
            vconv_f162s8r(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f162s8f(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f162s8c(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f162s8a(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f162s8z(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_NONE:
            vconv_f162s8(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ODD:
            ASCENDC_ASSERT((false),
                           { KERNEL_LOG(KERNEL_ERROR, "CAST_ODD from half to int8_t not supported on current device"); });
            break;
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            break;
    }
}

__aicore__ inline void CastIntrinsicsImpl(__ubuf__ uint8_t* dst, __ubuf__ half* src, const RoundMode& roundMode,
    uint8_t repeat, const UnaryRepeatParams& repeatParams)
{
    switch (roundMode) {
        case RoundMode::CAST_RINT:
            vconv_f162u8r(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f162u8f(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f162u8c(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f162u8a(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f162u8z(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_NONE:
            vconv_f162u8(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ODD:
            ASCENDC_ASSERT((false),
                           { KERNEL_LOG(KERNEL_ERROR, "CAST_ODD from half to uint8_t not supported on current device"); });
            break;
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            break;
    }
}

__aicore__ inline void CastIntrinsicsImpl(__ubuf__ half* dst, __ubuf__ float* src, const RoundMode& roundMode,
    uint8_t repeat, const UnaryRepeatParams& repeatParams)
{
    switch (roundMode) {
        case RoundMode::CAST_RINT:
            vconv_f322f16r(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f322f16f(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f322f16c(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f322f16a(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f322f16z(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ODD:
            vconv_f322f16o(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_NONE:
            vconv_f322f16(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            break;
    }
}

__aicore__ inline void CastIntrinsicsImpl(__ubuf__ int32_t* dst, __ubuf__ float* src, const RoundMode& roundMode,
    uint8_t repeat, const UnaryRepeatParams& repeatParams)
{
    switch (roundMode) {
        case RoundMode::CAST_RINT:
            vconv_f322s32r(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f322s32f(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f322s32c(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f322s32a(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f322s32z(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ODD:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "CAST_ODD from float to int32_t not supported on current device"); });
            break;
        case RoundMode::CAST_NONE:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "CAST_NONE from float to int32_t not supported on current device"); });
            break;
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            break;
    }
}

__aicore__ inline void CastIntrinsicsImpl(__ubuf__ int16_t* dst, __ubuf__ half* src, const RoundMode& roundMode,
    uint8_t repeat, const UnaryRepeatParams& repeatParams)
{
    switch (roundMode) {
        case RoundMode::CAST_RINT:
            vconv_f162s16r(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f162s16f(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f162s16c(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f162s16a(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f162s16z(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ODD:
            ASCENDC_ASSERT((false),
                           { KERNEL_LOG(KERNEL_ERROR, "CAST_ODD from half to int16_t not supported on current device"); });
            break;
        case RoundMode::CAST_NONE:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "CAST_NONE from half to int16_t not supported on current device"); });
            break;
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            break;
    }
}

__aicore__ inline void CastIntrinsicsImpl(__ubuf__ uint8_t* dst, __ubuf__ int16_t* src, const RoundMode& roundMode,
    uint8_t repeat, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "Cast from type int16_t to uint8_t");
}

__aicore__ inline void CastIntrinsicsImpl(__ubuf__ int8_t* dst, __ubuf__ int16_t* src, const RoundMode& roundMode,
    uint8_t repeat, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "Cast from type int16_t to int8_t");
}

__aicore__ inline void CastIntrinsicsImpl(__ubuf__ half* dst, __ubuf__ int16_t* src, const RoundMode& roundMode,
    uint8_t repeat, const UnaryRepeatParams& repeatParams)
{
    switch (roundMode) {
        case RoundMode::CAST_RINT:
            vconv_s162f16r(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_s162f16f(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_s162f16c(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_s162f16a(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_s162f16z(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_NONE:
            vconv_s162f16(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ODD:
            ASCENDC_ASSERT((false),
                           { KERNEL_LOG(KERNEL_ERROR, "CAST_ODD from int16_t to half not supported on current device"); });
            break;
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            break;
    }
}

__aicore__ inline void CastIntrinsicsImpl(__ubuf__ float* dst, __ubuf__ float* src, const RoundMode& roundMode,
    uint8_t repeat, const UnaryRepeatParams& repeatParams)
{
    switch (roundMode) {
        case RoundMode::CAST_RINT:
            vconv_f322f32r(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f322f32f(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f322f32c(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f322f32a(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f322f32z(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ODD:
            ASCENDC_ASSERT((false),
                           { KERNEL_LOG(KERNEL_ERROR, "CAST_ODD from float to float not supported on current device"); });
            break;
        case RoundMode::CAST_NONE:
            ASCENDC_ASSERT((false),
                           { KERNEL_LOG(KERNEL_ERROR, "CAST_NONE from float to float not supported on current device"); });
            break;
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            break;
    }
}

__aicore__ inline void CastIntrinsicsImpl(__ubuf__ bfloat16_t* dst, __ubuf__ float* src, const RoundMode& roundMode,
    uint8_t repeat, const UnaryRepeatParams& repeatParams)
{
    switch (roundMode) {
        case RoundMode::CAST_RINT:
            vconv_f322bf16r(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f322bf16f(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f322bf16c(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f322bf16a(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f322bf16z(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ODD:
            ASCENDC_ASSERT((false), {
                KERNEL_LOG(KERNEL_ERROR, "CAST_ODD from float to bfloat16_t not supported on current device");
            });
            break;
        case RoundMode::CAST_NONE:
            ASCENDC_ASSERT((false), {
                KERNEL_LOG(KERNEL_ERROR, "CAST_NONE from float to bfloat16_t not supported on current device");
            });
            break;
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            break;
    }
}

__aicore__ inline void CastIntrinsicsImpl(__ubuf__ int64_t* dst, __ubuf__ float* src, const RoundMode& roundMode,
    uint8_t repeat, const UnaryRepeatParams& repeatParams)
{
    switch (roundMode) {
        case RoundMode::CAST_RINT:
            vconv_f322s64r(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f322s64f(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f322s64c(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f322s64a(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f322s64z(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ODD:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "CAST_ODD from float to int64_t not supported on current device"); });
            break;
        case RoundMode::CAST_NONE:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "CAST_NONE from float to int64_t not supported on current device"); });
            break;
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            break;
    }
}

__aicore__ inline void CastIntrinsicsImpl(__ubuf__ float* dst, __ubuf__ bfloat16_t* src, const RoundMode& roundMode,
    uint8_t repeat, const UnaryRepeatParams& repeatParams)
{
    switch (roundMode) {
        case RoundMode::CAST_NONE:
            vconv_bf162f32(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        default:
            ASCENDC_ASSERT((false), {
                KERNEL_LOG(KERNEL_ERROR,
                    "illegal input cast mode %d, only support CAST_NONE from bfloat16_t to float on current device",
                    static_cast<int32_t>(roundMode));
            });
            break;
    }
}

__aicore__ inline void CastIntrinsicsImpl(__ubuf__ int32_t* dst, __ubuf__ bfloat16_t* src, const RoundMode& roundMode,
    uint8_t repeat, const UnaryRepeatParams& repeatParams)
{
    switch (roundMode) {
        case RoundMode::CAST_RINT:
            vconv_bf162s32r(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_bf162s32f(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_bf162s32c(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_bf162s32a(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_bf162s32z(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ODD:
            ASCENDC_ASSERT((false), {
                KERNEL_LOG(KERNEL_ERROR, "CAST_ODD from bfloat16_t to int32_t not supported on current device");
            });
            break;
        case RoundMode::CAST_NONE:
            ASCENDC_ASSERT((false), {
                KERNEL_LOG(KERNEL_ERROR, "CAST_NONE from bfloat16_t to int32_t not supported on current device");
            });
            break;
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            break;
    }
}

__aicore__ inline void CastIntrinsicsImpl(__ubuf__ int16_t* dst, __ubuf__ float* src, const RoundMode& roundMode,
    uint8_t repeat, const UnaryRepeatParams& repeatParams)
{
    switch (roundMode) {
        case RoundMode::CAST_RINT:
            vconv_f322s16r(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f322s16f(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f322s16c(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f322s16a(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f322s16z(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ODD:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "CAST_ODD from float to int16_t not supported on current device"); });
            break;
        case RoundMode::CAST_NONE:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "CAST_NONE from float to int16_t not supported on current device"); });
            break;
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            break;
    }
}

__aicore__ inline void CastIntrinsicsImpl(__ubuf__ float* dst, __ubuf__ int16_t* src, const RoundMode& roundMode,
    uint8_t repeat, const UnaryRepeatParams& repeatParams)
{
    switch (roundMode) {
        case RoundMode::CAST_NONE:
            vconv_s162f32(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        default:
            ASCENDC_ASSERT((false), {
                KERNEL_LOG(KERNEL_ERROR,
                    "illegal input cast mode %d, only support CAST_NONE from int16_t to float on current device",
                    static_cast<int32_t>(roundMode));
            });
            break;
    }
}

__aicore__ inline void CastIntrinsicsImpl(__ubuf__ int16_t* dst, __ubuf__ int32_t* src, const RoundMode& roundMode,
    uint8_t repeat, const UnaryRepeatParams& repeatParams)
{
    switch (roundMode) {
        case RoundMode::CAST_NONE:
            vconv_s322s16(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        default:
            ASCENDC_ASSERT((false), {
                KERNEL_LOG(KERNEL_ERROR,
                    "illegal input cast mode %d, only support CAST_NONE from int32_t to int16_t on current device",
                    static_cast<int32_t>(roundMode));
            });
            break;
    }
}

__aicore__ inline void CastIntrinsicsImpl(__ubuf__ int64_t* dst, __ubuf__ int32_t* src, const RoundMode& roundMode,
    uint8_t repeat, const UnaryRepeatParams& repeatParams)
{
    switch (roundMode) {
        case RoundMode::CAST_NONE:
            vconv_s322s64(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        default:
            ASCENDC_ASSERT((false), {
                KERNEL_LOG(KERNEL_ERROR,
                    "illegal input cast mode %d, only support CAST_NONE from int32_t to int64_t on current device",
                    static_cast<int32_t>(roundMode));
            });
            break;
    }
}

__aicore__ inline void CastIntrinsicsImpl(__ubuf__ float* dst, __ubuf__ int64_t* src, const RoundMode& roundMode,
    uint8_t repeat, const UnaryRepeatParams& repeatParams)
{
    switch (roundMode) {
        case RoundMode::CAST_RINT:
            vconv_s642f32r(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_s642f32f(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_s642f32c(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_s642f32a(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_s642f32z(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ODD:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "CAST_ODD from int64_t to float not supported on current device"); });
            break;
        case RoundMode::CAST_NONE:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "CAST_NONE from int64_t to float not supported on current device"); });
            break;
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            break;
    }
}

__aicore__ inline void CastIntrinsicsImpl(__ubuf__ int32_t* dst, __ubuf__ int64_t* src, const RoundMode& roundMode,
    uint8_t repeat, const UnaryRepeatParams& repeatParams)
{
    switch (roundMode) {
        case RoundMode::CAST_NONE:
            vconv_s642s32(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        default:
            ASCENDC_ASSERT((false), {
                KERNEL_LOG(KERNEL_ERROR,
                    "illegal input cast mode %d, only support CAST_NONE from int64_t to int32_t on current device",
                    static_cast<int32_t>(roundMode));
            });
            break;
    }
}

__aicore__ inline void CastIntrinsicsImpl(__ubuf__ int4b_t* dst, __ubuf__ half* src, const RoundMode& roundMode,
    uint8_t repeat, const UnaryRepeatParams& repeatParams)
{
    switch (roundMode) {
        case RoundMode::CAST_RINT:
            vconv_f162s4r(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_FLOOR:
            vconv_f162s4f(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_CEIL:
            vconv_f162s4c(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ROUND:
            vconv_f162s4a(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_TRUNC:
            vconv_f162s4z(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_NONE:
            vconv_f162s4(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        case RoundMode::CAST_ODD:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "CAST_ODD from half to int4b_t not supported on current device"); });
            break;
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            break;
    }
}

__aicore__ inline void CastIntrinsicsImpl(__ubuf__ half* dst, __ubuf__ int4b_t* src, const RoundMode& roundMode,
    uint8_t repeat, const UnaryRepeatParams& repeatParams)
{
    switch (roundMode) {
        case RoundMode::CAST_NONE:
            vconv_s42f16(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
                repeatParams.dstRepStride, repeatParams.srcRepStride);
            break;
        default:
            ASCENDC_ASSERT((false), {
                KERNEL_LOG(KERNEL_ERROR,
                    "illegal input cast mode %d, only support CAST_NONE from int4b_t to half on current device",
                    static_cast<int32_t>(roundMode));
            });
            break;
    }
}

// check Cast  datatype
template <typename U, typename T>
__aicore__ static inline void CheckCastDatatype() {
    ASCENDC_ASSERT((SupportType<Tuple<U, T>, Tuple<int32_t, half>, Tuple<int16_t, half>, Tuple<float, half>,
        Tuple<int8_t, half>, Tuple<uint8_t, half>, Tuple<int4b_t, half>, Tuple<float, float>, Tuple<int32_t, float>,
        Tuple<half, float>, Tuple<int64_t, float>, Tuple<int16_t, float>,  Tuple<bfloat16_t, float>,
        Tuple<float, bfloat16_t>, Tuple<int32_t, bfloat16_t>, Tuple<half, int4b_t>, Tuple<half, uint8_t>,
        Tuple<half, int8_t>, Tuple<half, int16_t>, Tuple<float, int16_t>, Tuple<float, int32_t>,
        Tuple<int16_t, int32_t>, Tuple<int64_t, int32_t>, Tuple<half, int32_t>, Tuple<int32_t, int64_t>,
        Tuple<float, int64_t>>()), { KERNEL_LOG(KERNEL_ERROR,
        "Failed to check dtype in Cast, current api support dtype combination is src: half, dst: int32_t / int16_t / "
        "float / int8_t / uint8_t / int4b_t; src: float, dst: float / int32_t/ half / int64_t / int16_t / bfloat16_t; "
        "src: bfloat16_t, dst: float / int32_t; src: int4b_t, dst: half; src: uint8_t, dst: half; "
        "src: int 8_t, dst: half; src: int16_t, dst: half / float; src: int32_t, dst: float / int16_t / int64_t; "
        "src: int64_t, dst: int32_t / float / half");});
}

// Cast::Level 2
template <typename U, typename T>
__aicore__ inline void CastImpl(__ubuf__ U* dst, __ubuf__ T* src, const RoundMode& roundMode,
    const uint32_t calCount)
{
    if ASCEND_IS_AIV {
        CheckCastDatatype<U, T>();
        set_mask_count();
        set_vector_mask(0, calCount);
        if constexpr (sizeof(U) > sizeof(T)) {
            if constexpr (IsSameType<T, int4b_t>::value) {
                CastIntrinsicsImpl(
                    dst, src, roundMode, 1, {1, 1, DEFAULT_REPEAT_STRIDE, ONE_FOURTH_DEFAULT_REPEAT_STRIDE});
            } else {
                CastIntrinsicsImpl(dst, src, roundMode, 1, {1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE / 2});
            }
        } else if constexpr (sizeof(U) < sizeof(T)) {
            if constexpr (IsSameType<U, int4b_t>::value) {
                CastIntrinsicsImpl(
                    dst, src, roundMode, 1, {1, 1, ONE_FOURTH_DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
            } else {
                CastIntrinsicsImpl(dst, src, roundMode, 1, {1, 1, DEFAULT_REPEAT_STRIDE / 2, DEFAULT_REPEAT_STRIDE});
            }
        } else {
            CastIntrinsicsImpl(dst, src, roundMode, 1, {1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
        }
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    }
}

// Cast::Level 0 - mask bit mode
template <typename U, typename T, bool isSetMask = true>
__aicore__ inline void CastImpl(__ubuf__ U* dst, __ubuf__ T* src, const RoundMode& roundMode,
    const uint64_t mask[], uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        CheckCastDatatype<U, T>();
        if constexpr (isSetMask) {
            if (sizeof(U) >= sizeof(T)) {
                AscendCUtils::SetMask<T>(mask[1], mask[0]);
            } else {
                AscendCUtils::SetMask<U>(mask[1], mask[0]);
            }
        }
        CastIntrinsicsImpl(dst, src, roundMode, repeatTimes, repeatParams);
    }
}

// Cast::Level 0 - mask count mode
template <typename U, typename T, bool isSetMask = true>
__aicore__ inline void CastImpl(__ubuf__ U* dst, __ubuf__ T* src, const RoundMode& roundMode,
    const uint64_t mask, uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        CheckCastDatatype<U, T>();
        if constexpr (isSetMask) {
            if (sizeof(U) >= sizeof(T)) {
                AscendCUtils::SetMask<T>(mask);
            } else {
                AscendCUtils::SetMask<U>(mask);
            }
        }
        CastIntrinsicsImpl(dst, src, roundMode, repeatTimes, repeatParams);
    }
}

template <typename T, bool halfBlock>
__aicore__ inline void CastDeqIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ int16_t* src, uint8_t repeat,
    const UnaryRepeatParams& repeatParams)
{
    if constexpr (halfBlock) {
        vconv_deqs162b8h(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
            repeatParams.dstRepStride, repeatParams.srcRepStride);
    } else {
        vconv_deqs162b8l(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
            repeatParams.dstRepStride, repeatParams.srcRepStride);
    }
}

template <typename T, bool halfBlock>
__aicore__ inline void CastVDeqIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ int16_t* src, uint8_t repeat,
    const UnaryRepeatParams& repeatParams)
{
    if constexpr (halfBlock) {
        vconv_vdeqs162b8h(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
            repeatParams.dstRepStride, repeatParams.srcRepStride);
    } else {
        vconv_vdeqs162b8l(dst, src, repeat, repeatParams.dstBlkStride, repeatParams.srcBlkStride,
            repeatParams.dstRepStride, repeatParams.srcRepStride);
    }
}

// CastDeq::Level 2
template <typename U, typename T, bool isVecDeq, bool halfBlock>
__aicore__ inline void CastDeqImpl(__ubuf__ U* dst, __ubuf__ T* src,
    const uint32_t calCount)
{
    if ASCEND_IS_AIV {
        set_mask_count();
        set_vector_mask(0, calCount);
        struct UnaryRepeatParams repeatParams;
        if constexpr (isVecDeq) {
            CastVDeqIntrinsicsImpl<U, halfBlock>(dst, src, 1, repeatParams);
        } else {
            CastDeqIntrinsicsImpl<U, halfBlock>(dst, src, 1, repeatParams);
        }
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    }
}

// CastDeq::Level 0 - mask bit mode
template <typename U, typename T, bool isSetMask, bool isVecDeq, bool halfBlock>
__aicore__ inline void CastDeqImpl(__ubuf__ U* dst, __ubuf__ T* src,
    const uint64_t mask[], uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
        if constexpr (isVecDeq) {
            CastVDeqIntrinsicsImpl<U, halfBlock>(dst, src, repeatTimes, repeatParams);
        } else {
            CastDeqIntrinsicsImpl<U, halfBlock>(dst, src, repeatTimes, repeatParams);
        }
    }
}

// CastDeq::Level 0 - mask count mode
template <typename U, typename T, bool isSetMask, bool isVecDeq, bool halfBlock>
__aicore__ inline void CastDeqImpl(__ubuf__ U* dst, __ubuf__ T* src,
    const int32_t mask, uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask);
        if constexpr (isVecDeq) {
            CastVDeqIntrinsicsImpl<U, halfBlock>(dst, src, repeatTimes, repeatParams);
        } else {
            CastDeqIntrinsicsImpl<U, halfBlock>(dst, src, repeatTimes, repeatParams);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void AddReluCastIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1, uint8_t repeat,
    const BinaryRepeatParams& repeatParams)
{
    static_assert(SupportType<Tuple<U, T>, Tuple<half, int8_t>, Tuple<float, half>, Tuple<int16_t, int8_t>>(),
        "Failed to check dtype in AddReluCast, current api support dtype combination is src: half, dst: int8_t; src: "
        "float, dst: half; src: int16_t, dst: int8_t.");
    if constexpr (SupportType<Tuple<U, T>, Tuple<half, int8_t>>()) {
        vaddreluconv_f162s8(dst, src0, src1, repeat, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
            repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride,
            repeatParams.src1RepStride, false);
    } else if constexpr (SupportType<Tuple<U, T>, Tuple<float, half>>()) {
        vaddreluconv_f322f16(dst, src0, src1, repeat, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
            repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride,
            repeatParams.src1RepStride, false);
    } else {
        vaddreluconv_s162s8(dst, src0, src1, repeat, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
            repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride,
            repeatParams.src1RepStride, false);
    }
}

// AddReluCast::Level 0 - mask count mode
template <typename DST_TYPE, typename SRC_TYPE, bool isSetMask = true>
__aicore__ inline void AddReluCastImpl(__ubuf__ DST_TYPE* dst, __ubuf__ SRC_TYPE* src0, __ubuf__ SRC_TYPE* src1,
    const uint64_t mask, uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        if constexpr (isSetMask) {
            if constexpr (sizeof(DST_TYPE) >= sizeof(SRC_TYPE)) {
                AscendCUtils::SetMask<SRC_TYPE>(mask);
            } else {
                AscendCUtils::SetMask<DST_TYPE>(mask);
            }
        }
        AddReluCastIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}

// AddReluCast::Level 0 - mask bit mode
template <typename DST_TYPE, typename SRC_TYPE, bool isSetMask = true>
__aicore__ inline void AddReluCastImpl(__ubuf__ DST_TYPE* dst, __ubuf__ SRC_TYPE* src0, __ubuf__ SRC_TYPE* src1,
    const uint64_t mask[], uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        if constexpr (isSetMask) {
            if constexpr (sizeof(DST_TYPE) >= sizeof(SRC_TYPE)) {
                AscendCUtils::SetMask<SRC_TYPE>(mask[1], mask[0]);
            } else {
                AscendCUtils::SetMask<DST_TYPE>(mask[1], mask[0]);
            }
        }
        AddReluCastIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}

// AddReluCast::Level 2
template <typename DST_TYPE, typename SRC_TYPE>
__aicore__ inline void AddReluCastImpl(__ubuf__ DST_TYPE* dst, __ubuf__ SRC_TYPE* src0, __ubuf__ SRC_TYPE* src1,
    const uint32_t calCount)
{
    if ASCEND_IS_AIV {
        set_mask_count();
        set_vector_mask(0, calCount);
        if constexpr (sizeof(DST_TYPE) > sizeof(SRC_TYPE)) {
            AddReluCastIntrinsicsImpl(dst, src0, src1, 1, {1, 1, 1, DEFAULT_REPEAT_STRIDE,
                DEFAULT_REPEAT_STRIDE / HALF_FACTOR, DEFAULT_REPEAT_STRIDE / HALF_FACTOR});
        } else if constexpr (sizeof(DST_TYPE) < sizeof(SRC_TYPE)) {
            AddReluCastIntrinsicsImpl(dst, src0, src1, 1, {1, 1, 1, DEFAULT_REPEAT_STRIDE / HALF_FACTOR,
                DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
        } else {
            AddReluCastIntrinsicsImpl(dst, src0, src1, 1, {1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE,
                DEFAULT_REPEAT_STRIDE});
        }
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    }
}


template <typename T, typename U>
__aicore__ inline void SubReluCastIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1, uint8_t repeat,
    const BinaryRepeatParams& repeatParams)
{
    static_assert(SupportType<Tuple<U, T>, Tuple<half, int8_t>, Tuple<float, half>, Tuple<int16_t, int8_t>>(),
        "Failed to check dtype in SubReluCast, current api support dtype combination is src: half, dst: int8_t; src: "
        "float, dst: half; src: int16_t, dst: int8_t.");
    if constexpr (SupportType<Tuple<U, T>, Tuple<half, int8_t>>()) {
        vsubreluconv_f162s8(dst, src0, src1, repeat, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
            repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride,
            repeatParams.src1RepStride, false);
    } else if constexpr (SupportType<Tuple<U, T>, Tuple<float, half>>()) {
        vsubreluconv_f322f16(dst, src0, src1, repeat, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
            repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride,
            repeatParams.src1RepStride, false);
    } else {
        vsubreluconv_s162s8(dst, src0, src1, repeat, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
            repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride,
            repeatParams.src1RepStride, false);
    }
}

// SubReluCast::Level 2
template <typename DST_TYPE, typename SRC_TYPE>
__aicore__ inline void SubReluCastImpl(__ubuf__ DST_TYPE* dst, __ubuf__ SRC_TYPE* src0, __ubuf__ SRC_TYPE* src1,
    const uint32_t calCount)
{
    if ASCEND_IS_AIV {
        set_mask_count();
        set_vector_mask(0, calCount);
        if constexpr (sizeof(DST_TYPE) > sizeof(SRC_TYPE)) {
            SubReluCastIntrinsicsImpl(dst, src0, src1, 1, {1, 1, 1, DEFAULT_REPEAT_STRIDE,
                DEFAULT_REPEAT_STRIDE / HALF_FACTOR, DEFAULT_REPEAT_STRIDE / HALF_FACTOR});
        } else if constexpr (sizeof(DST_TYPE) < sizeof(SRC_TYPE)) {
            SubReluCastIntrinsicsImpl(dst, src0, src1, 1, {1, 1, 1, DEFAULT_REPEAT_STRIDE / HALF_FACTOR,
                DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
        } else {
            SubReluCastIntrinsicsImpl(dst, src0, src1, 1, {1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE,
                DEFAULT_REPEAT_STRIDE});
        }
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    }
}

// SubReluCast::Level 0 - mask count mode
template <typename DST_TYPE, typename SRC_TYPE, bool isSetMask = true>
__aicore__ inline void SubReluCastImpl(__ubuf__ DST_TYPE* dst, __ubuf__ SRC_TYPE* src0, __ubuf__ SRC_TYPE* src1,
    const uint64_t mask, uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        if constexpr (isSetMask) {
            if constexpr (sizeof(DST_TYPE) >= sizeof(SRC_TYPE)) {
                AscendCUtils::SetMask<SRC_TYPE>(mask);
            } else {
                AscendCUtils::SetMask<DST_TYPE>(mask);
            }
        }
        SubReluCastIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}

// SubReluCast::Level 0 - mask bit mode
template <typename DST_TYPE, typename SRC_TYPE, bool isSetMask = true>
__aicore__ inline void SubReluCastImpl(__ubuf__ DST_TYPE* dst, __ubuf__ SRC_TYPE* src0, __ubuf__ SRC_TYPE* src1,
    const uint64_t mask[], uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        if constexpr (isSetMask) {
            if constexpr (sizeof(DST_TYPE) >= sizeof(SRC_TYPE)) {
                AscendCUtils::SetMask<SRC_TYPE>(mask[1], mask[0]);
            } else {
                AscendCUtils::SetMask<DST_TYPE>(mask[1], mask[0]);
            }
        }
        SubReluCastIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}

__aicore__ inline uint64_t MakeDeqScaleConfig(float scale, int16_t offset, bool signMode)
{
    constexpr uint64_t signModeBit = 46;
    constexpr uint64_t offsetMask = 0x1ff;
    constexpr uint64_t offsetBit = 37;
    uint64_t config = ((static_cast<uint64_t>(signMode) << signModeBit) | ((offset & offsetMask) << offsetBit) |
                       *(reinterpret_cast<uint32_t *>(&scale)));
    return config;
}

__aicore__ inline void SetDeqScaleImpl(float scale, int16_t offset, bool signMode)
{
    set_deqscale(MakeDeqScaleConfig(scale, offset, signMode));
}

template <typename T>
__aicore__ inline void SetDeqScaleImpl(const LocalTensor<T>& vdeqTensor, const VdeqInfo& vdeqInfo)
{
    for (uint8_t i = 0; i < VDEQ_TENSOR_SIZE; i++) {
        float scale = vdeqInfo.vdeqScale[i];
        int16_t offset = vdeqInfo.vdeqOffset[i];
        bool signMode = vdeqInfo.vdeqSignMode[i];
        vdeqTensor.SetValue(i, static_cast<T>(MakeDeqScaleConfig(scale, offset, signMode)));
    }
#if ASCENDC_CPU_DEBUG
    set_deqscale((uint64_t)vdeqTensor.GetPhyAddr());
#else
    constexpr uint64_t deqAddr = 5; // 32B align
    set_deqscale(((uint64_t)vdeqTensor.GetPhyAddr()) >> deqAddr);
#endif
}

template<typename T>
__aicore__ inline void SetDeqScaleImpl(T config)
{
    set_deqscale(config);
    g_deqValue = config;
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_VCONV_IMPL_H