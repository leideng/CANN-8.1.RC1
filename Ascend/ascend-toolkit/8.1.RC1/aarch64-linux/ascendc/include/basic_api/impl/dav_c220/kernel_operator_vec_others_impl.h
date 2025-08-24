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
 * \file kernel_operator_vec_others_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_OTHERS_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_OTHERS_IMPL_H

#ifndef ASCENDC_CPU_DEBUG
namespace AscendC {
template <typename T>
__aicore__ inline void AddRelu(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint64_t config)
{
    vaddrelu(dst, src0, src1, config);
}

template <typename T>
__aicore__ inline void AddRelu(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeat,
    uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
    uint8_t src0RepeatStride, uint8_t src1RepeatStride)
{
    vaddrelu(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
        src0RepeatStride, src1RepeatStride);
}

template <typename T> __aicore__ inline void CmpvsEq(__ubuf__ uint8_t *dst, __ubuf__ T *src0, T src1, uint64_t config)
{
    vcmpvs_eq(dst, src0, src1, config);
}

template <typename T>
__aicore__ inline void CmpvsEq(__ubuf__ uint8_t *dst, __ubuf__ T *src0, T src1, uint8_t repeat, uint16_t dstBlockStride,
    uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride)
{
    vcmpvs_eq(dst, src0, src1, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
}

template <typename T> __aicore__ inline void Gather(__ubuf__ T *dst, __ubuf__ uint32_t *src, uint64_t config)
{
    vgather(dst, src, config);
}

template <typename T> __aicore__ inline void Madd(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint64_t config)
{
    vmadd(dst, src0, src1, config);
}

template <typename T>
__aicore__ inline void Madd(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeat, uint8_t dstBlockStride,
    uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride, uint8_t src0RepeatStride,
    uint8_t src1RepeatStride)
{
    vmadd(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride,
        src1RepeatStride);
}

template <typename T>
__aicore__ inline void MaddRelu(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint64_t config)
{
    vmaddrelu(dst, src0, src1, config);
}

template <typename T>
__aicore__ inline void MaddRelu(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeat,
    uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
    uint8_t src0RepeatStride, uint8_t src1RepeatStride)
{
    vmaddrelu(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
        src0RepeatStride, src1RepeatStride);
}

template <typename T> __aicore__ inline void Mla(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint64_t config)
{
    vmla(dst, src0, src1, config);
}

template <typename T>
__aicore__ inline void Mla(__ubuf__ T *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t config)
{
    vmla(dst, src0, src1, config);
}

template <typename T>
__aicore__ inline void Mla(__ubuf__ T *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
    uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
    uint8_t src0RepeatStride, uint8_t src1RepeatStride, bool repeatStrideMode, bool strideSizeMode)
{
    vmla(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride,
        src1RepeatStride, repeatStrideMode, strideSizeMode);
}

template <typename T>
__aicore__ inline void Mla(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeat, uint8_t dstBlockStride,
    uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride, uint8_t src0RepeatStride,
    uint8_t src1RepeatStride, bool repeatStrideMode, bool strideSizeMode)
{
    vmla(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride,
        src1RepeatStride, repeatStrideMode, strideSizeMode);
}

template <typename T>
__aicore__ inline void Mla(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeat, uint8_t dstBlockStride,
    uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride, uint8_t src0RepeatStride,
    uint8_t src1RepeatStride)
{
    vmla(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride, src0RepeatStride,
        src1RepeatStride);
}

template <typename T>
__aicore__ inline void SubRelu(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint64_t config)
{
    vsubrelu(dst, src0, src1, config);
}

template <typename T>
__aicore__ inline void SubRelu(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t repeat,
    uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
    uint8_t src0RepeatStride, uint8_t src1RepeatStride)
{
    vsubrelu(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
        src0RepeatStride, src1RepeatStride);
}

__aicore__ inline void SubReluConvF162s8(__ubuf__ int8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1,
    uint64_t config, bool h)
{
    vsubreluconv_f162s8(dst, src0, src1, config, h);
}

__aicore__ inline void SubReluConvF162s8(__ubuf__ int8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
    uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
    uint8_t src0RepeatStride, uint8_t src1RepeatStride, bool h)
{
    vsubreluconv_f162s8(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
        src0RepeatStride, src1RepeatStride, h);
}

template <typename T>
__aicore__ inline void AddReluConvVdeqs162b8(__ubuf__ T *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1,
    uint8_t repeat, uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
    uint8_t src0RepeatStride, uint8_t src1RepeatStride, bool h)
{
    vaddreluconv_vdeqs162b8(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
        src0RepeatStride, src1RepeatStride, h);
}

template <typename T>
__aicore__ inline void MulAndCast(__ubuf__ T *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t config)
{
    vmulconv_f162s8(dst, src0, src1, config);
}

template <typename T>
__aicore__ inline void MulAndCast(__ubuf__ T *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
    uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
    uint8_t src0RepeatStride, uint8_t src1RepeatStride)
{
    vmulconv_f162s8(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
        src0RepeatStride, src1RepeatStride);
}

template <typename T>
__aicore__ inline void MulAndCast(__ubuf__ T *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
    uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
    uint8_t src0RepeatStride, uint8_t src1RepeatStride, bool repeatStrideMode, bool strideSizeMode)
{
    vmulconv_f162s8(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
        src0RepeatStride, src1RepeatStride, repeatStrideMode, strideSizeMode);
}
} // namespace AscendC
#endif
#endif // ASCENDC_MODULE_OPERATOR_VEC_OTHERS_IMPL_H