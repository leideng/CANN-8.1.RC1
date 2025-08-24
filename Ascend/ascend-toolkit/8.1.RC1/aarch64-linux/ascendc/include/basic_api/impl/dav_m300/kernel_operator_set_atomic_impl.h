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
 * \file kernel_operator_set_atomic_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_SET_ATOMIC_ADD_IMPL_H
#define ASCENDC_MODULE_OPERATOR_SET_ATOMIC_ADD_IMPL_H

namespace AscendC {
// set_atomic_none
__aicore__ inline void SetAtomicNoneImpl()
{
    set_atomic_none();
}

// set_atomic_add
template <typename T>
__aicore__ inline void SetAtomicAddImpl()
{
    ASCENDC_ASSERT((SupportType<T, half, float, int16_t, int32_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in SetAtomicAdd, current api support dtype is: "
        "half / float / int16_t / int32_t");});
}

template <> __aicore__ inline void SetAtomicAddImpl<float>()
{
    set_atomic_add();
    set_atomic_f32();
}

template <>
__aicore__ inline void SetAtomicAddImpl<half>()
{
    set_atomic_add();
    set_atomic_f16();
}

template <>
__aicore__ inline void SetAtomicAddImpl<int16_t>()
{
    set_atomic_add();
    set_atomic_s16();
}

template <>
__aicore__ inline void SetAtomicAddImpl<int32_t>()
{
    set_atomic_add();
    set_atomic_s32();
}

// set_atomic_max
template <typename T>
__aicore__ inline void SetAtomicMaxImpl()
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "SetAtomicMax");
}

// set_atomic_min
template <typename T>
__aicore__ inline void SetAtomicMinImpl()
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "SetAtomicMin");
}

// set_atomic_type
template <typename T>
__aicore__ inline void SetAtomicTypeImpl()
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "SetAtomicType");
}

template <>
__aicore__ inline void SetAtomicTypeImpl<float>()
{
    set_atomic_f32();
}

template <>
__aicore__ inline void SetAtomicTypeImpl<half>()
{
    set_atomic_f16();
}

template <>
__aicore__ inline void SetAtomicTypeImpl<int16_t>()
{
    set_atomic_s16();
}

template <>
__aicore__ inline void SetAtomicTypeImpl<int32_t>()
{
    set_atomic_s32();
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_SET_ATOMIC_ADD_IMPL_H
