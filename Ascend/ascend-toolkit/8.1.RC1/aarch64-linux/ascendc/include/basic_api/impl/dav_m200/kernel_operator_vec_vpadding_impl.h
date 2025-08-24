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
 * \file kernel_operator_vec_vpadding_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_VPADDING_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_VPADDING_IMPL_H
#include "kernel_utils.h"
#include "kernel_struct_unary.h"

namespace AscendC {

template <typename T>
__aicore__ inline void VectorPaddingIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src, uint8_t padMode, bool padSide,
    uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT((SupportType<T, half, int16_t, uint16_t, float, int32_t, uint32_t>()), { KERNEL_LOG(KERNEL_ERROR,
        "Failed to check dtype in VectorPadding, current api support dtype combination is src and dst both: half / "
        "int16_t / uint16_t / float / int32_t / uint32_t.");});
    ASCENDC_CHECK_VALUE_RANGE(padMode, 0, 2, "padMode", "VectorPadding");
    if constexpr(sizeof(T) == B16_BYTE_SIZE) {
        vpadding((__ubuf__ uint16_t*)dst, (__ubuf__ uint16_t*)src, repeatTimes, repeatParams.dstBlkStride,
            repeatParams.srcBlkStride, repeatParams.dstRepStride, repeatParams.srcRepStride,
            repeatParams.repeatStrideMode, repeatParams.strideSizeMode, padMode, padSide);
    } else if constexpr(sizeof(T) == B32_BYTE_SIZE) {
        vpadding((__ubuf__ uint32_t*)dst, (__ubuf__ uint32_t*)src, repeatTimes, repeatParams.dstBlkStride,
            repeatParams.srcBlkStride, repeatParams.dstRepStride, repeatParams.srcRepStride,
            repeatParams.repeatStrideMode, repeatParams.strideSizeMode, padMode, padSide);
    }
}

template <typename T, bool isSetMask>
__aicore__ inline void VectorPaddingImpl(__ubuf__ T* dst, __ubuf__ T* src, const uint8_t padMode, const bool padSide,
    const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    AscendCUtils::SetMask<T, isSetMask>(mask);
    VectorPaddingIntrinsicsImpl(dst, src, padMode, padSide, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask>
__aicore__ inline void VectorPaddingImpl(__ubuf__ T* dst, __ubuf__ T* src, const uint8_t padMode, const bool padSide,
    const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
    VectorPaddingIntrinsicsImpl(dst, src, padMode, padSide, repeatTimes, repeatParams);
}

template <typename T>
__aicore__ inline void VectorPaddingImpl(__ubuf__ T* dst, __ubuf__ T* src, uint8_t padMode, bool padSide,
    const uint32_t calCount)
{
    set_mask_count();
    set_vector_mask(0, calCount);
    UnaryRepeatParams repeatParams;
    VectorPaddingIntrinsicsImpl(dst, src, padMode, padSide, 1, repeatParams);
    set_mask_norm();
    set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_VPADDING_IMPL_H