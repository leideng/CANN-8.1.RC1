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
 * \file kernel_operator_vec_gather_mask_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_REDUCEV2_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_VEC_REDUCEV2_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_struct_gather.h"
#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_vec_gather_mask_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_vec_gather_mask_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_vec_gather_mask_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_vec_gather_mask_impl.h"
#elif defined(__DAV_M310__)
#include "dav_m310/kernel_operator_vec_gather_mask_impl.h"
#endif

#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif

namespace AscendC {
#pragma begin_pipe(V)
template <typename T, typename U, GatherMaskMode mode = defaultGatherMaskMode>
__aicore__ inline void GatherMask(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<U>& src1Pattern, const bool reduceMode, const uint32_t mask,
    const GatherMaskParams& gatherMaskParams, uint64_t& rsvdCnt);

template <typename T, GatherMaskMode mode = defaultGatherMaskMode>
__aicore__ inline void GatherMask(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const uint8_t src1Pattern, const bool reduceMode, const uint32_t mask, const GatherMaskParams& gatherMaskParams,
    uint64_t& rsvdCnt);
#pragma end_pipe

__aicore__ inline __inout_pipe__(S) int64_t GetGatherMaskRemainCount();
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_REDUCEV2_INTERFACE_H