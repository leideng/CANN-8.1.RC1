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

namespace AscendC {
template <typename T> constexpr __aicore__ inline void CheckDuplicateSupportedType()
{
    static_assert(SupportType<T, half, int16_t, uint16_t, int32_t, uint32_t, float>(), "Failed to check dtype in "
        "Duplicate, current api support dtype combination is src and dst both: half / int16_t / uint16_t / int32_t / "
        "uint32_t / float.");
}

template <typename T>
__aicore__ inline void DuplicateIntrinsicsImpl(__ubuf__ T* dstLocal, T scalarValue, const uint8_t repeatTimes,
    const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    vector_dup(dstLocal, scalarValue, repeatTimes, dstBlockStride, 1, dstRepeatStride, 0);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void DuplicateImpl(__ubuf__ T* dstLocal, const T& scalarValue, uint64_t mask,
    const uint8_t repeatTimes, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    AscendCUtils::SetMask<T, isSetMask>(mask);
    DuplicateIntrinsicsImpl(dstLocal, scalarValue, repeatTimes, dstBlockStride, dstRepeatStride);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void DuplicateImpl(__ubuf__ T* dstLocal, const T& scalarValue, uint64_t mask[],
    const uint8_t repeatTimes, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
    DuplicateIntrinsicsImpl(dstLocal, scalarValue, repeatTimes, dstBlockStride, dstRepeatStride);
}

template <typename T>
__aicore__ inline void DuplicateImpl(__ubuf__ T* dstLocal, const T& scalarValue, const int32_t& calCount)
{
    SetMaskCount();
    AscendCUtils::SetMask<T>(0, calCount);
    DuplicateIntrinsicsImpl(dstLocal, scalarValue, 1, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    ResetMask();
    SetMaskNorm();
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_DUPLICATE_IMPL_H
