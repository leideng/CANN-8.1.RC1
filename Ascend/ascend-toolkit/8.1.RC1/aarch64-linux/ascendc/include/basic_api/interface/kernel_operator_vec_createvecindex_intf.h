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
 * \file kernel_operator_vec_createvecindex_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_CREATEVECINDEX_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_VEC_CREATEVECINDEX_INTERFACE_H
#include "kernel_tensor.h"
#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif

namespace AscendC {

template <typename T>
__aicore__ inline __in_pipe__(S) __out_pipe__(V) void CreateVecIndex(LocalTensor<T> &dstLocal, const T &firstValue,
    uint64_t mask, uint8_t repeatTimes, uint16_t dstBlkStride, uint8_t dstRepStride);

template <typename T>
__aicore__ inline __in_pipe__(S) __out_pipe__(V) void CreateVecIndex(LocalTensor<T> &dstLocal, const T &firstValue,
    uint64_t mask[], uint8_t repeatTimes, uint16_t dstBlkStride, uint8_t dstRepStride);

template <typename T>
__aicore__ inline __in_pipe__(S) __out_pipe__(V) void CreateVecIndex(LocalTensor<T> dstLocal, const T &firstValue,
    uint32_t calCount);
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_CREATEVECINDEX_INTERFACE_H