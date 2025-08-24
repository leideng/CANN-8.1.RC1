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
 * \file inner_kernel_operator_vec_createvecindex_intf.cppm
 * \brief
 */
#ifndef ASCENDC_MODULE_INNER_OPERATOR_VEC_CREATEVECINDEX_INTERFACE_H
#define ASCENDC_MODULE_INNER_OPERATOR_VEC_CREATEVECINDEX_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_check.h"

#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_vec_createvecindex_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_vec_createvecindex_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_vec_createvecindex_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_vec_createvecindex_impl.h"
#elif defined(__DAV_M310__)
#include "dav_m310/kernel_operator_vec_createvecindex_impl.h"
#endif

namespace AscendC {
template <typename T>
__aicore__ inline __in_pipe__(S) __out_pipe__(V) void CreateVecIndex(LocalTensor<T> &dstLocal, const T &firstValue,
    uint64_t mask, uint8_t repeatTimes, uint16_t dstBlkStride, uint8_t dstRepStride)
{
    ASCENDC_ASSERT((SupportType<T, half, int16_t, float, int32_t>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to check "
        "dtype in CreateVecIndex, current api support dtype combination is dst: half / int16_t / float / int32_t");});
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncCreateVecIndex(dstLocal, mask, repeatTimes, dstBlkStride, dstRepStride, "CreateVecIndex")) {
        ASCENDC_REPORT_CHECK_ERROR("CreateVecIndex", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    CreateVecIndexCalc(dstLocal, firstValue, mask, repeatTimes, dstBlkStride, dstRepStride);
}

template <typename T>
__aicore__ inline __in_pipe__(S) __out_pipe__(V) void CreateVecIndex(LocalTensor<T> &dstLocal, const T &firstValue,
    uint64_t mask[], uint8_t repeatTimes, uint16_t dstBlkStride, uint8_t dstRepStride)
{
    ASCENDC_ASSERT((SupportType<T, half, int16_t, float, int32_t>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to check "
        "dtype in CreateVecIndex, current api support dtype combination is dst: half / int16_t / float / int32_t");});
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncCreateVecIndex(dstLocal, mask, repeatTimes, dstBlkStride, dstRepStride, "CreateVecIndex")) {
        ASCENDC_REPORT_CHECK_ERROR("CreateVecIndex", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    CreateVecIndexCalc(dstLocal, firstValue, mask, repeatTimes, dstBlkStride, dstRepStride);
}

template <typename T>
__aicore__ inline __in_pipe__(S) __out_pipe__(V) void CreateVecIndex(LocalTensor<T> dstLocal, const T &firstValue,
    uint32_t calCount)
{
    ASCENDC_ASSERT((SupportType<T, half, int16_t, float, int32_t>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to check "
        "dtype in CreateVecIndex, current api support dtype combination is dst: half / int16_t / float / int32_t");});
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncCreateVecIndex(dstLocal, calCount, "CreateVecIndex")) {
        ASCENDC_REPORT_CHECK_ERROR("CreateVecIndex", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    CreateVecIndexCalc(dstLocal, firstValue, calCount);
}
} // namespace AscendC
#endif // ASCENDC_MODULE_INNER_OPERATOR_VEC_CREATEVECINDEX_INTERFACE_H