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
 * \file kernel_operator_vec_vpadding_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_VPADDING_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_VPADDING_IMPL_H
#include "kernel_utils.h"
#include "kernel_struct_unary.h"

namespace AscendC {
template <typename T, bool isSetMask>
__aicore__ inline void VectorPaddingImpl(__ubuf__ T* dst, __ubuf__ T* src, uint8_t padMode, bool padSide,
    const uint64_t mask, uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT((false), "VectorPadding is not supported on current device");
}

template <typename T, bool isSetMask>
__aicore__ inline void VectorPaddingImpl(__ubuf__ T* dst, __ubuf__ T* src, uint8_t padMode, bool padSide,
    const uint64_t mask[], uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_ASSERT((false), "VectorPadding is not supported on current device");
}

template <typename T>
__aicore__ inline void VectorPaddingImpl(__ubuf__ T* dst, __ubuf__ T* src, uint8_t padMode, bool padSide,
    const uint32_t calCount)
{
    ASCENDC_ASSERT((false), "VectorPadding is not supported on current device");
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_VPADDING_IMPL_H