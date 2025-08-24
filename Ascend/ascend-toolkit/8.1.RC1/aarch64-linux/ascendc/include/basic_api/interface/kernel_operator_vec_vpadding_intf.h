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
 * \file kernel_operator_vec_vpadding_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_VPADDING_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_VEC_VPADDING_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_struct_unary.h"
#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif

#pragma begin_pipe(V)
namespace AscendC {
template <typename T, bool isSetMask = true>
__aicore__ inline void VectorPadding(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const uint8_t padMode, const bool padSide, const uint64_t mask, const uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void VectorPadding(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const uint8_t padMode, const bool padSide, const uint64_t mask[], const uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams);

template <typename T>
__aicore__ inline void VectorPadding(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const uint8_t padMode, const bool padSide, const uint32_t calCount);
} // namespace AscendC
#pragma end_pipe
#endif // ASCENDC_MODULE_OPERATOR_VEC_VPADDING_INTERFACE_H