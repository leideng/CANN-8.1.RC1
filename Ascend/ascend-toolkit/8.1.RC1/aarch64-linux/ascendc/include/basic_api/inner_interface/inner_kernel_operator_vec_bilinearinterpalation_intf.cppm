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
 * \file inner_kernel_operator_vec_bilinearinterpalation_intf.cppm
 * \brief
 */
#ifndef ASCENDC_MODULE_INNER_OPERATOR_VEC_BILINEARINTERPALATION_INTERFACE_H
#define ASCENDC_MODULE_INNER_OPERATOR_VEC_BILINEARINTERPALATION_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_check.h"

#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_vec_bilinearinterpalation_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_vec_bilinearinterpalation_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_vec_bilinearinterpalation_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_vec_bilinearinterpalation_impl.h"
#elif defined(__DAV_M310__)
#include "dav_m310/kernel_operator_vec_bilinearinterpalation_impl.h"
#endif

#pragma begin_pipe(V)
namespace AscendC {
template <typename T>
__aicore__ inline void BilinearInterpolation(const LocalTensor<T> &dstLocal, const LocalTensor<T> &src0Local,
    const LocalTensor<uint32_t> &src0OffsetLocal, const LocalTensor<T> &src1Local, uint64_t mask, uint8_t hRepeat,
    bool repeatMode, uint16_t dstBlkStride, uint16_t vROffset, uint8_t vRepeat,
    const LocalTensor<uint8_t> &sharedTmpBuffer)
{
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncBilinearInterpolation(dstLocal, src0Local, src0OffsetLocal, src1Local, mask, hRepeat, repeatMode,
        dstBlkStride, vROffset, vRepeat, "BilinearInterpolation")) {
        ASCENDC_REPORT_CHECK_ERROR("BilinearInterpolation", KernelFuncType::MASK_COUNT_MODE);
    }

    uint32_t sharedTmpBufferSize = sharedTmpBuffer.GetSize();
#if __CCE_AICORE__ == 220
    uint32_t expectedTmpBufferSize = (src0Local.GetSize() + src1Local.GetSize()) * 32;
#else
    uint32_t expectedTmpBufferSize = src0OffsetLocal.GetSize() * sizeof(uint32_t);
#endif
    ASCENDC_ASSERT((sharedTmpBufferSize >= expectedTmpBufferSize), { KERNEL_LOG(KERNEL_ERROR, "Failed to check "
        "sharedTmpBuffer size in BilinearInterpolation, its expected size is at least %u, current size is %u",
        expectedTmpBufferSize, sharedTmpBufferSize);});
#endif
    ASCENDC_ASSERT((SupportType<T, half>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in BilinearInterpolation,"
        " current api support dtype combination is src: half, dst: half");});
    ASCENDC_CHECK_VALUE_RANGE(vROffset, 128, UINT16_MAX, "vROffset", "BilinearInterpolation");
    BilinearInterpolationCalc(dstLocal, src0Local, src0OffsetLocal, src1Local, mask, hRepeat,
        repeatMode, dstBlkStride, vROffset, vRepeat, sharedTmpBuffer);
}

template <typename T>
__aicore__ inline void BilinearInterpolation(const LocalTensor<T> &dstLocal, const LocalTensor<T> &src0Local,
    const LocalTensor<uint32_t> &src0OffsetLocal, const LocalTensor<T> &src1Local, uint64_t mask[], uint8_t hRepeat,
    bool repeatMode, uint16_t dstBlkStride, uint16_t vROffset, uint8_t vRepeat,
    const LocalTensor<uint8_t> &sharedTmpBuffer)
{
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncBilinearInterpolation(dstLocal, src0Local, src0OffsetLocal, src1Local, mask, hRepeat, repeatMode,
        dstBlkStride, vROffset, vRepeat, "BilinearInterpolation")) {
        ASCENDC_REPORT_CHECK_ERROR("BilinearInterpolation", KernelFuncType::MASK_BIT_MODE);
    }

    uint32_t sharedTmpBufferSize = sharedTmpBuffer.GetSize();
#if __CCE_AICORE__ == 220
    uint32_t expectedTmpBufferSize = (src0Local.GetSize() + src1Local.GetSize()) * 32;
#else
    uint32_t expectedTmpBufferSize = src0OffsetLocal.GetSize() * sizeof(uint32_t);
#endif
    ASCENDC_ASSERT((sharedTmpBufferSize >= expectedTmpBufferSize), { KERNEL_LOG(KERNEL_ERROR, "Failed to check "
        "sharedTmpBuffer size in BilinearInterpolation, its expected size is at least %u, current size is %u",
        expectedTmpBufferSize, sharedTmpBufferSize);});
#endif
    ASCENDC_ASSERT((SupportType<T, half>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in BilinearInterpolation,"
        " current api support dtype combination is src: half, dst: half");});
    ASCENDC_CHECK_VALUE_RANGE(vROffset, 128, UINT16_MAX, "vROffset", "BilinearInterpolation");
    BilinearInterpolationCalc(dstLocal, src0Local, src0OffsetLocal, src1Local, mask, hRepeat,
        repeatMode, dstBlkStride, vROffset, vRepeat, sharedTmpBuffer);
}
} // namespace AscendC
#pragma end_pipe
#endif // ASCENDC_MODULE_INNER_OPERATOR_VEC_BILINEARINTERPALATION_INTERFACE_H