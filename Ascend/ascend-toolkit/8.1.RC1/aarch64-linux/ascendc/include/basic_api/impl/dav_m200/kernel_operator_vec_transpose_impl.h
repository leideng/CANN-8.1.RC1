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
 * \file kernel_operator_vec_transpose_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_TRANSPOSE_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_TRANSPOSE_IMPL_H
#include "kernel_struct_transpose.h"

namespace AscendC {
__aicore__ inline void TransDataTo5HDIntrinsicsImpl(__ubuf__ float* dstList[16], __ubuf__ float* srcList[16],
    const TransDataTo5HDParams& transDataTo5HDParams)
{
    scatter_vnchwconv_b32(VA0, VA2, transDataTo5HDParams.repeatTimes, transDataTo5HDParams.dstRepStride,
        transDataTo5HDParams.srcRepStride);
}

__aicore__ inline void TransDataTo5HDIntrinsicsImpl(__ubuf__ int32_t* dstList[16], __ubuf__ int32_t* srcList[16],
    const TransDataTo5HDParams& transDataTo5HDParams)
{
    scatter_vnchwconv_b32(VA0, VA2, transDataTo5HDParams.repeatTimes, transDataTo5HDParams.dstRepStride,
        transDataTo5HDParams.srcRepStride);
}

__aicore__ inline void TransDataTo5HDIntrinsicsImpl(__ubuf__ uint32_t* dstList[16], __ubuf__ uint32_t* srcList[16],
    const TransDataTo5HDParams& transDataTo5HDParams)
{
    scatter_vnchwconv_b32(VA0, VA2, transDataTo5HDParams.repeatTimes, transDataTo5HDParams.dstRepStride,
        transDataTo5HDParams.srcRepStride);
}

__aicore__ inline void TransDataTo5HDIntrinsicsImpl(__ubuf__ int16_t* dstList[16], __ubuf__ int16_t* srcList[16],
    const TransDataTo5HDParams& transDataTo5HDParams)
{
    scatter_vnchwconv_b16(VA0, VA2, transDataTo5HDParams.repeatTimes, transDataTo5HDParams.dstRepStride,
        transDataTo5HDParams.srcRepStride);
}

__aicore__ inline void TransDataTo5HDIntrinsicsImpl(__ubuf__ uint16_t* dstList[16], __ubuf__ uint16_t* srcList[16],
    const TransDataTo5HDParams& transDataTo5HDParams)
{
    scatter_vnchwconv_b16(VA0, VA2, transDataTo5HDParams.repeatTimes, transDataTo5HDParams.dstRepStride,
        transDataTo5HDParams.srcRepStride);
}

__aicore__ inline void TransDataTo5HDIntrinsicsImpl(__ubuf__ half* dstList[16], __ubuf__ half* srcList[16],
    const TransDataTo5HDParams& transDataTo5HDParams)
{
    scatter_vnchwconv_b16(VA0, VA2, transDataTo5HDParams.repeatTimes, transDataTo5HDParams.dstRepStride,
        transDataTo5HDParams.srcRepStride);
}

template <typename T>
__aicore__ inline void TransDataTo5HDB8IntrinsicsImpl(__ubuf__ T* dstList[16], __ubuf__ T* srcList[16],
    const TransDataTo5HDParams& transDataTo5HDParams)
{
    if ((transDataTo5HDParams.dstHighHalf == false) && (transDataTo5HDParams.srcHighHalf == false)) {
        scatter_vnchwconv_b8(VA0, VA2, transDataTo5HDParams.repeatTimes, transDataTo5HDParams.dstRepStride,
            transDataTo5HDParams.srcRepStride, false, false);
    } else if ((transDataTo5HDParams.dstHighHalf == false) && (transDataTo5HDParams.srcHighHalf == true)) {
        scatter_vnchwconv_b8(VA0, VA2, transDataTo5HDParams.repeatTimes, transDataTo5HDParams.dstRepStride,
            transDataTo5HDParams.srcRepStride, false, true);
    } else if ((transDataTo5HDParams.dstHighHalf == true) && (transDataTo5HDParams.srcHighHalf == true)) {
        scatter_vnchwconv_b8(VA0, VA2, transDataTo5HDParams.repeatTimes, transDataTo5HDParams.dstRepStride,
            transDataTo5HDParams.srcRepStride, true, true);
    } else {
        scatter_vnchwconv_b8(VA0, VA2, transDataTo5HDParams.repeatTimes, transDataTo5HDParams.dstRepStride,
            transDataTo5HDParams.srcRepStride, true, false);
    }
}

__aicore__ inline void TransDataTo5HDIntrinsicsImpl(__ubuf__ int8_t* dstList[16], __ubuf__ int8_t* srcList[16],
    const TransDataTo5HDParams& transDataTo5HDParams)
{
    TransDataTo5HDB8IntrinsicsImpl(dstList, srcList, transDataTo5HDParams);
}

__aicore__ inline void TransDataTo5HDIntrinsicsImpl(__ubuf__ uint8_t* dstList[16], __ubuf__ uint8_t* srcList[16],
    const TransDataTo5HDParams& transDataTo5HDParams)
{
    TransDataTo5HDB8IntrinsicsImpl(dstList, srcList, transDataTo5HDParams);
}

template<typename T>
__aicore__ inline void TransDataTo5HDIntrinsicsImpl(uint64_t dstList[16], uint64_t srcList[16],
    const TransDataTo5HDParams& transDataTo5HDParams)
{
    ASSERT(false && "TransDataTo5HD with current dtype is not supported on current device");
}

template<>
__aicore__ inline void TransDataTo5HDIntrinsicsImpl<float>(uint64_t dstList[16], uint64_t srcList[16],
    const TransDataTo5HDParams& transDataTo5HDParams)
{
    scatter_vnchwconv_b32(VA0, VA2, transDataTo5HDParams.repeatTimes, transDataTo5HDParams.dstRepStride,
        transDataTo5HDParams.srcRepStride);
}

template <>
__aicore__ inline void TransDataTo5HDIntrinsicsImpl<int32_t>(uint64_t dstList[16], uint64_t srcList[16],
    const TransDataTo5HDParams& transDataTo5HDParams)
{
    scatter_vnchwconv_b32(VA0, VA2, transDataTo5HDParams.repeatTimes, transDataTo5HDParams.dstRepStride,
        transDataTo5HDParams.srcRepStride);
}

template <>
__aicore__ inline void TransDataTo5HDIntrinsicsImpl<uint32_t>(uint64_t dstList[16], uint64_t srcList[16],
    const TransDataTo5HDParams& transDataTo5HDParams)
{
    scatter_vnchwconv_b32(VA0, VA2, transDataTo5HDParams.repeatTimes, transDataTo5HDParams.dstRepStride,
        transDataTo5HDParams.srcRepStride);
}

template <>
__aicore__ inline void TransDataTo5HDIntrinsicsImpl<int16_t>(uint64_t dstList[16], uint64_t srcList[16],
    const TransDataTo5HDParams& transDataTo5HDParams)
{
    scatter_vnchwconv_b16(VA0, VA2, transDataTo5HDParams.repeatTimes, transDataTo5HDParams.dstRepStride,
        transDataTo5HDParams.srcRepStride);
}

template <>
__aicore__ inline void TransDataTo5HDIntrinsicsImpl<uint16_t>(uint64_t dstList[16], uint64_t srcList[16],
    const TransDataTo5HDParams& transDataTo5HDParams)
{
    scatter_vnchwconv_b16(VA0, VA2, transDataTo5HDParams.repeatTimes, transDataTo5HDParams.dstRepStride,
        transDataTo5HDParams.srcRepStride);
}

template <>
__aicore__ inline void TransDataTo5HDIntrinsicsImpl<half>(uint64_t dstList[16], uint64_t srcList[16],
    const TransDataTo5HDParams& transDataTo5HDParams)
{
    scatter_vnchwconv_b16(VA0, VA2, transDataTo5HDParams.repeatTimes, transDataTo5HDParams.dstRepStride,
        transDataTo5HDParams.srcRepStride);
}

template <typename T>
__aicore__ inline void TransDataTo5HDB8IntrinsicsImpl(uint64_t dstList[16], uint64_t srcList[16],
    const TransDataTo5HDParams& transDataTo5HDParams)
{
    if ((transDataTo5HDParams.dstHighHalf == false) && (transDataTo5HDParams.srcHighHalf == false)) {
        scatter_vnchwconv_b8(VA0, VA2, transDataTo5HDParams.repeatTimes, transDataTo5HDParams.dstRepStride,
            transDataTo5HDParams.srcRepStride, false, false);
    } else if ((transDataTo5HDParams.dstHighHalf == false) && (transDataTo5HDParams.srcHighHalf == true)) {
        scatter_vnchwconv_b8(VA0, VA2, transDataTo5HDParams.repeatTimes, transDataTo5HDParams.dstRepStride,
            transDataTo5HDParams.srcRepStride, false, true);
    } else if ((transDataTo5HDParams.dstHighHalf == true) && (transDataTo5HDParams.srcHighHalf == true)) {
        scatter_vnchwconv_b8(VA0, VA2, transDataTo5HDParams.repeatTimes, transDataTo5HDParams.dstRepStride,
            transDataTo5HDParams.srcRepStride, true, true);
    } else {
        scatter_vnchwconv_b8(VA0, VA2, transDataTo5HDParams.repeatTimes, transDataTo5HDParams.dstRepStride,
            transDataTo5HDParams.srcRepStride, true, false);
    }
}

template <>
__aicore__ inline void TransDataTo5HDIntrinsicsImpl<int8_t>(uint64_t dstList[16], uint64_t srcList[16],
    const TransDataTo5HDParams& transDataTo5HDParams)
{
    TransDataTo5HDB8IntrinsicsImpl<int8_t>(dstList, srcList, transDataTo5HDParams);
}

template <>
__aicore__ inline void TransDataTo5HDIntrinsicsImpl<uint8_t>(uint64_t dstList[16], uint64_t srcList[16],
    const TransDataTo5HDParams& transDataTo5HDParams)
{
    TransDataTo5HDB8IntrinsicsImpl<uint8_t>(dstList, srcList, transDataTo5HDParams);
}

template <typename T> __aicore__ inline void SetVaReg(__ubuf__ T* dstList[16], __ubuf__ T* srcList[16])
{
    uint64_t vaRegArray1[VA_REG_ARRAY_LEN];
    uint64_t vaRegArray2[VA_REG_ARRAY_LEN];
    uint64_t vaRegArray3[VA_REG_ARRAY_LEN];
    uint64_t vaRegArray4[VA_REG_ARRAY_LEN];

    for (int32_t i = 0; i < VA_REG_ARRAY_LEN; i++) {
        vaRegArray1[i] = (uint64_t)dstList[i];
        vaRegArray2[i] = (uint64_t)dstList[VA_REG_ARRAY_LEN + i];
        vaRegArray3[i] = (uint64_t)srcList[i];
        vaRegArray4[i] = (uint64_t)srcList[VA_REG_ARRAY_LEN + i];
    }

    set_va_reg_sb(VA0, vaRegArray1);
    set_va_reg_sb(VA1, vaRegArray2);
    set_va_reg_sb(VA2, vaRegArray3);
    set_va_reg_sb(VA3, vaRegArray4);
}

__aicore__ inline void SetVaReg(uint64_t dst[NCHW_CONV_ADDR_LIST_SIZE],
    uint64_t src[NCHW_CONV_ADDR_LIST_SIZE])
{
    set_va_reg_sb(VA0, dst);
    set_va_reg_sb(VA1, dst + VA_REG_ARRAY_LEN);
    set_va_reg_sb(VA2, src);
    set_va_reg_sb(VA3, src + VA_REG_ARRAY_LEN);
}

__aicore__ inline void VldVaReg(__ubuf__ uint64_t* dst, __ubuf__ uint64_t* src)
{
    vld_va_reg(VA0, dst, L128);
    vld_va_reg(VA1, dst, H128);
    vld_va_reg(VA2, src, L128);
    vld_va_reg(VA3, src, H128);
}

template <typename T>
__aicore__ inline void TransDataTo5HDImpl(__ubuf__ T* dstList[16], __ubuf__ T* srcList[16],
    const TransDataTo5HDParams& transDataTo5HDParams)
{
    SetVaReg(dstList, srcList);
    TransDataTo5HDIntrinsicsImpl(dstList, srcList, transDataTo5HDParams);
}

template <typename T>
__aicore__ inline void TransDataTo5HDImpl(uint64_t dstList[NCHW_CONV_ADDR_LIST_SIZE],
    uint64_t srcList[NCHW_CONV_ADDR_LIST_SIZE], const TransDataTo5HDParams& transDataTo5HDParams)
{
    SetVaReg(dstList, srcList);
    TransDataTo5HDIntrinsicsImpl<T>(dstList, srcList, transDataTo5HDParams);
}

template <typename T>
__aicore__ inline void TransDataTo5HDVldVaRegImpl(
    __ubuf__ uint64_t* dst, __ubuf__ uint64_t* src, const TransDataTo5HDParams& transDataTo5HDParams)
{
    VldVaReg(dst, src);
    uint64_t dstList[NCHW_CONV_ADDR_LIST_SIZE] = { 0 };
    uint64_t srcList[NCHW_CONV_ADDR_LIST_SIZE] = { 0 };
    TransDataTo5HDIntrinsicsImpl<T>(dstList, srcList, transDataTo5HDParams);
}

// Transpose::Level 0
template <typename T> __aicore__ inline void TransposeIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src)
{
    vtranspose((__ubuf__ uint16_t*)dst, (__ubuf__ uint16_t*)src);
}

// Transpose::Level 0
template <typename T> __aicore__ inline void TransposeImpl(__ubuf__ T* dst, __ubuf__ T* src)
{
    TransposeIntrinsicsImpl((__ubuf__ uint16_t*)dst, (__ubuf__ uint16_t*)src);
}

template <typename T>
__aicore__ inline void Transpose4DImpl(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const LocalTensor<uint8_t> &sharedTmpBuffer, const TransposeParamsExt &transposeParams)
{
    uint16_t imageSize = transposeParams.hSize * transposeParams.wSize;
    uint32_t channelImageSize = imageSize * transposeParams.cSize;

    using U = typename Conditional<sizeof(T) == B8_BYTE_SIZE, uint8_t,
        typename Conditional<sizeof(T) == B16_BYTE_SIZE, uint16_t, uint32_t>::type>::type;
    if (transposeParams.transposeType == TransposeType::TRANSPOSE_NCHW2NHWC) {
        for (int i = 0; i < transposeParams.nSize; i++) {
            v4dtrans((__ubuf__ U*)dstLocal[channelImageSize * i].GetPhyAddr(),
                (__ubuf__ U*)srcLocal[channelImageSize * i].GetPhyAddr(), imageSize, transposeParams.cSize, false);
        }
    } else if (transposeParams.transposeType == TransposeType::TRANSPOSE_NHWC2NCHW) {
        for (int i = 0; i < transposeParams.nSize; i++) {
            v4dtrans((__ubuf__ U*)dstLocal[channelImageSize * i].GetPhyAddr(),
                (__ubuf__ U*)srcLocal[channelImageSize * i].GetPhyAddr(), imageSize, transposeParams.cSize, true);
        }
    }
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_TRANSPOSE_IMPL_H