/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file simd.h
 * \brief
 */
#ifndef INCLUDE_SIMD_H
#define INCLUDE_SIMD_H

#include "hardware.h"
#include "kernel_operator.h"

/////////////////////////////////////////////////////
// vadd
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void add_v(AscendC::LocalTensor<DType> dst,
                             AscendC::LocalTensor<DType> src0,
                             AscendC::LocalTensor<DType> src1,
                             uint8_t repeat,
                             uint8_t dstBlockStride,
                             uint8_t src0BlockStride,
                             uint8_t src1BlockStride,
                             uint8_t dstRepeatStride,
                             uint8_t src0RepeatStride,
                             uint8_t src1RepeatStride)
{
    AscendC::Add<DType, false>(
        dst,
        src0,
        src1,
        (uint64_t)0,
        repeat,
        AscendC::BinaryRepeatParams(
            dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride));
}

/////////////////////////////////////////////////////
// vadds
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void adds_v(AscendC::LocalTensor<DType> dst,
                              AscendC::LocalTensor<DType> src,
                              DType scalarValue,
                              uint8_t repeat,
                              uint8_t dstBlockStride,
                              uint8_t srcBlockStride,
                              uint8_t dstRepeatStride,
                              uint8_t srcRepeatStride)
{
    AscendC::Adds<DType, false>(
        dst,
        src,
        scalarValue,
        (uint64_t)0,
        repeat,
        AscendC::UnaryRepeatParams(dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride));
}


/////////////////////////////////////////////////////
// vconv
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DTypeIn, typename DTypeOut>
__aicore__ inline void conv_v(AscendC::LocalTensor<DTypeOut> dst,
                              AscendC::LocalTensor<DTypeIn> src,
                              uint8_t repeat,
                              uint16_t dstBlockStride,
                              uint16_t srcBlockStride,
                              uint16_t dstRepeatStride,
                              uint16_t srcRepeatStride)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    if constexpr (std::is_same<DTypeIn, float>::value && std::is_same<DTypeOut, bfloat16_t>::value) {
        AscendC::Cast<DTypeOut, DTypeIn, false>(
            dst,
            src,
            AscendC::RoundMode::CAST_RINT,
            (uint64_t)0,
            repeat,
            AscendC::UnaryRepeatParams(dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride));
    } else {
        AscendC::Cast<DTypeOut, DTypeIn, false>(
            dst,
            src,
            AscendC::RoundMode::CAST_NONE,
            (uint64_t)0,
            repeat,
            AscendC::UnaryRepeatParams(dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride));
    }
#else
    AscendC::Cast<DTypeOut, DTypeIn, false>(
        dst,
        src,
        AscendC::RoundMode::CAST_NONE,
        (uint64_t)0,
        repeat,
        AscendC::UnaryRepeatParams(dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride));
#endif
}

/////////////////////////////////////////////////////
// vconv_f322bf16r
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DTypeIn, typename DTypeOut>
__aicore__ inline void convr_v(AscendC::LocalTensor<DTypeOut> dst,
                               AscendC::LocalTensor<DTypeIn> src,
                               uint8_t repeat,
                               uint16_t dstBlockStride,
                               uint16_t srcBlockStride,
                               uint16_t dstRepeatStride,
                               uint16_t srcRepeatStride)
{
    AscendC::Cast<DTypeOut, DTypeIn, false>(
        dst,
        src,
        AscendC::RoundMode::CAST_RINT,
        (uint64_t)0,
        repeat,
        AscendC::UnaryRepeatParams(dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride));
}


/////////////////////////////////////////////////////
// vmul
/////////////////////////////////////////////////////
template <ArchType ArchTag, typename DType>
__aicore__ inline void mul_v(AscendC::LocalTensor<DType> dst,
                             AscendC::LocalTensor<DType> src0,
                             AscendC::LocalTensor<DType> src1,
                             uint8_t repeat,
                             uint8_t dstBlockStride,
                             uint8_t src0BlockStride,
                             uint8_t src1BlockStride,
                             uint8_t dstRepeatStride,
                             uint8_t src0RepeatStride,
                             uint8_t src1RepeatStride)
{
    AscendC::Mul<DType, false>(
        dst,
        src0,
        src1,
        (uint64_t)0,
        repeat,
        AscendC::BinaryRepeatParams(
            dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride, src1RepeatStride));
}

#endif