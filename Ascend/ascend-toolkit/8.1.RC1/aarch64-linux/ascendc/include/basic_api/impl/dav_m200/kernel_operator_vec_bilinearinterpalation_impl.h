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
 * \file kernel_operator_vec_bilinearinterpalation_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_BILINEARINTERPALATION_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_BILINEARINTERPALATION_IMPL_H
#include "kernel_tensor.h"
#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif

#pragma begin_pipe(V)
namespace AscendC {

template <typename T>
__aicore__ inline void BilinearInterpolationCalc(const LocalTensor<T> &dstLocal, const LocalTensor<T> &src0Local,
    const LocalTensor<uint32_t> &src0OffsetLocal, const LocalTensor<T> &src1Local, uint64_t mask, uint8_t hRepeat,
    bool repeatMode, uint16_t dstBlkStride, uint16_t vROffset, uint8_t vRepeat,
    const LocalTensor<uint8_t> &sharedTmpBuffer)
{
#if ASCENDC_CPU_DEBUG
    SetVbiSrc0Param(src0Local.GetPhyAddr(), src0Local.GetSize());
    set_mask_norm();
    AscendCUtils::SetMask<T>(mask);
    vbi(dstLocal.GetPhyAddr(), reinterpret_cast<uint16_t*>(src0OffsetLocal.GetPhyAddr()), src1Local.GetPhyAddr(),
        hRepeat, repeatMode, dstBlkStride, vROffset, vRepeat);
    ResetMask();
#else
    set_mask_count();
    set_vector_mask(0, src0OffsetLocal.GetSize());
    vadds(reinterpret_cast<__ubuf__ int32_t*>(sharedTmpBuffer.GetPhyAddr()),
        reinterpret_cast<__ubuf__ int32_t*>(src0OffsetLocal.GetPhyAddr()),
        static_cast<int32_t>(src0Local.GetPhyAddr()), 1,
        DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
    pipe_barrier(PIPE_V);
    set_mask_norm();
    AscendCUtils::SetMask<T>(mask);
    vbi(reinterpret_cast<__ubuf__ half*>(dstLocal.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint16_t*>(sharedTmpBuffer.GetPhyAddr()),
        reinterpret_cast<__ubuf__ half*>(src1Local.GetPhyAddr()),
        hRepeat, repeatMode, dstBlkStride, vROffset, vRepeat);
    ResetMask();
#endif
}

template <typename T>
__aicore__ inline void BilinearInterpolationCalc(const LocalTensor<T> &dstLocal, const LocalTensor<T> &src0Local,
    const LocalTensor<uint32_t> &src0OffsetLocal, const LocalTensor<T> &src1Local, uint64_t mask[], uint8_t hRepeat,
    bool repeatMode, uint16_t dstBlkStride, uint16_t vROffset, uint8_t vRepeat,
    const LocalTensor<uint8_t> &sharedTmpBuffer)
{
#if ASCENDC_CPU_DEBUG
    SetVbiSrc0Param(src0Local.GetPhyAddr(), src0Local.GetSize());
    set_mask_norm();
    AscendCUtils::SetMask<T>(mask[1], mask[0]);
    vbi(dstLocal.GetPhyAddr(), reinterpret_cast<uint16_t*>(src0OffsetLocal.GetPhyAddr()), src1Local.GetPhyAddr(),
        hRepeat, repeatMode, dstBlkStride, vROffset, vRepeat);
    ResetMask();
#else
    set_mask_count();
    set_vector_mask(0, src0OffsetLocal.GetSize());
    vadds(reinterpret_cast<__ubuf__ int32_t*>(sharedTmpBuffer.GetPhyAddr()),
        reinterpret_cast<__ubuf__ int32_t*>(src0OffsetLocal.GetPhyAddr()),
        static_cast<int32_t>(src0Local.GetPhyAddr()), 1,
        DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
    pipe_barrier(PIPE_V);
    set_mask_norm();
    AscendCUtils::SetMask<T>(mask[1], mask[0]);
    vbi(reinterpret_cast<__ubuf__ half*>(dstLocal.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint16_t*>(sharedTmpBuffer.GetPhyAddr()),
        reinterpret_cast<__ubuf__ half*>(src1Local.GetPhyAddr()),
        hRepeat, repeatMode, dstBlkStride, vROffset, vRepeat);
    ResetMask();
#endif
}
} // namespace AscendC
#pragma end_pipe
#endif // ASCENDC_MODULE_OPERATOR_VEC_BILINEARINTERPALATION_IMPL_H