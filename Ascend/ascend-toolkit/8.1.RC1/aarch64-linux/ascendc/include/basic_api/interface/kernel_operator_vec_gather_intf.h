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
 * \file kernel_operator_vec_gather_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_GATHER_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_VEC_GATHER_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_struct_gather.h"

#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif
#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_vec_gather_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_vec_gather_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_vec_gather_impl.h"
#endif

#pragma begin_pipe(V)
namespace AscendC {
/*
 * @ingroup gatherb Level 0
 * @brief this function fetches N addresses from offsetLocal,then accesses these N addresses(plus the src0Local address)
 * @brief to get N 32Byte block, and finally writes these N blocks into dstLocal.
 * @brief gather element in the uint of block
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] offsetLocal input LocalTensor
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 */
template <typename T>
__aicore__ inline void Gatherb(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<uint32_t>& offsetLocal, const uint8_t repeatTimes, const GatherRepeatParams& repeatParams);

/*
 * @ingroup gather Level 0
 * @brief gather element from srcLocal according to srcOffsetLocal
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] srcOffsetLocal input LocalTensor
 * @param [in] srcBaseAddr base address of srcLocal
 * @param [in] mask valid element count
 * @param [in] repeatTimes repeat times
 * @param [in] dstRepStride dst repeat stride
 */
template <typename T>
__aicore__ inline void Gather(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint32_t>& srcOffsetLocal, const uint32_t srcBaseAddr, const uint64_t mask,
    const uint8_t repeatTimes, const uint16_t dstRepStride);

/*
 * @ingroup gather Level 0
 * @brief gather element from srcLocal according to srcOffsetLocal
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] srcOffsetLocal input LocalTensor
 * @param [in] srcBaseAddr base address of srcLocal
 * @param [in] mask valid element count(bit mode)
 * @param [in] repeatTimes repeat times
 * @param [in] dstRepStride dst repeat stride
 */
template <typename T>
__aicore__ inline void Gather(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint32_t>& srcOffsetLocal, const uint32_t srcBaseAddr, const uint64_t mask[],
    const uint8_t repeatTimes, const uint16_t dstRepStride);

/*
 * @ingroup gather Level 2
 * @brief gather element from srcLocal according to srcOffsetLocal
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] srcOffsetLocal input LocalTensor
 * @param [in] srcBaseAddr base address of srcLocal
 * @param [in] count element count
 */
template <typename T>
__aicore__ inline void Gather(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<uint32_t>& srcOffsetLocal, const uint32_t srcBaseAddr, const uint32_t count);
} // namespace AscendC
#pragma end_pipe
#endif // ASCENDC_MODULE_OPERATOR_VEC_GATHER_INTERFACE_H