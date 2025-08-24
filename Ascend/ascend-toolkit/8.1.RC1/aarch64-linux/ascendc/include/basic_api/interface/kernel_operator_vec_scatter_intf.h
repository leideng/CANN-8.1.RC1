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
 * \file kernel_operator_vec_scatter_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_SCATTER_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_VEC_SCATTER_INTERFACE_H
#include "kernel_tensor.h"

#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif
#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_vec_scatter_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_vec_scatter_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_vec_scatter_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_vec_scatter_impl.h"
#elif defined(__DAV_M310__)
#include "dav_m310/kernel_operator_vec_scatter_impl.h"
#endif

#pragma begin_pipe(V)
namespace AscendC {
/*
 * @ingroup scatter Level 0
 * @brief scatter element from dstLocal according to dstOffsetLocal
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] dstOffsetLocal input LocalTensor
 * @param [in] mask valid element count
 * @param [in] repeatTimes repeat times
 * @param [in] srcRepStride src repeat stride
 */
template <typename T>
__aicore__ inline void Scatter(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint32_t>& dstOffsetLocal, const uint32_t dstBaseAddr, const uint64_t mask,
    const uint8_t repeatTimes, const uint8_t srcRepStride);

/*
 * @ingroup scatter Level 0
 * @brief scatter element from dstLocal according to dstOffsetLocal
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] dstOffsetLocal input LocalTensor
 * @param [in] mask valid element count(bit mode)
 * @param [in] repeatTimes repeat times
 * @param [in] srcRepStride src repeat stride
 */
template <typename T>
__aicore__ inline void Scatter(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint32_t>& dstOffsetLocal, const uint32_t dstBaseAddr, const uint64_t mask[],
    const uint8_t repeatTimes, const uint8_t srcRepStride);

/*
 * @ingroup scatter Level 2
 * @brief scatter element from dstLocal according to dstOffsetLocal
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] dstOffsetLocal input LocalTensor
 * @param [in] count element count
 */
template <typename T>
__aicore__ inline void Scatter(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint32_t>& dstOffsetLocal, const uint32_t dstBaseAddr, const uint32_t count);
} // namespace AscendC
#pragma end_pipe
#endif // ASCENDC_MODULE_OPERATOR_VEC_SCATTER_INTERFACE_H