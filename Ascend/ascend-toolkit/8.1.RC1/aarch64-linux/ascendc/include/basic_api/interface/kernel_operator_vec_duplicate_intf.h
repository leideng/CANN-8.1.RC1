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
 * \file kernel_operator_vec_duplicate_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_DUPLICATE_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_VEC_DUPLICATE_INTERFACE_H
#include "kernel_tensor.h"

#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif
#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_vec_duplicate_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_vec_duplicate_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_vec_duplicate_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_vec_duplicate_impl.h"
#elif defined(__DAV_M310__)
#include "dav_m310/kernel_operator_vec_duplicate_impl.h"
#endif

#pragma begin_pipe(V)
namespace AscendC {
/* **************************************************************************************************
 * Duplicate                                            *
 * ************************************************************************************************* */
/*
 * @ingroup Duplicate Level 0
 * @brief dst[i] = scalar
 * @param [out] dstLocal output LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] dstBlockStride dst block stride
 * @param [in] dstRepeatStride dst repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Duplicate(const LocalTensor<T>& dstLocal, const T& scalarValue, uint64_t mask,
    const uint8_t repeatTimes, const uint16_t dstBlockStride, const uint8_t dstRepeatStride);

template <typename T, bool isSetMask = true>
__aicore__ inline void Duplicate(const LocalTensor<T>& dstLocal, const T& scalarValue, uint64_t mask[],
    const uint8_t repeatTimes, const uint16_t dstBlockStride, const uint8_t dstRepeatStride);

/*
 * @ingroup Duplicate Level 2
 * @brief dst = dst[i] = scalar
 * @param [out] dstLocal output LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Duplicate(const LocalTensor<T>& dstLocal, const T& scalarValue, const int32_t& calCount);
} // namespace AscendC
#pragma end_pipe
#endif // ASCENDC_MODULE_OPERATOR_VEC_DUPLICATE_INTERFACE_H
