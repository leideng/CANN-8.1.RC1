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
#include "kernel_struct_brcb.h"
#include "kernel_struct_gather.h"
#include "kernel_struct_binary.h"
#include "kernel_struct_unary.h"
#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif

#pragma begin_pipe(V)
namespace AscendC {
constexpr uint32_t brcbEleNum = 8;

template <typename T>
__aicore__ inline void BilinearInterpolationCalc(const LocalTensor<T> &dstLocal, const LocalTensor<T> &src0Local,
    const LocalTensor<uint32_t> &src0OffsetLocal, const LocalTensor<T> &src1Local, uint64_t mask, uint8_t hRepeat,
    bool repeatMode, uint16_t dstBlkStride, uint16_t vROffset, uint8_t vRepeat,
    const LocalTensor<uint8_t> &sharedTmpBuffer)
{
    auto sharedTmpBufferT = sharedTmpBuffer.ReinterpretCast<T>();
    GatherRepeatParams gatherbRepeatParams;
    uint8_t innerRepeatTimes = hRepeat * vRepeat;
    constexpr uint32_t eleCntOfOneRep = DEFAULT_REPEAT_STRIDE * ONE_BLK_SIZE / sizeof(T);
    // gatherb
    GatherbImpl((__ubuf__ uint16_t *)sharedTmpBufferT.GetPhyAddr(), (__ubuf__ uint16_t *)src0Local.GetPhyAddr(),
        (__ubuf__ uint32_t *)src0OffsetLocal.GetPhyAddr(), src0Local.GetSize(), innerRepeatTimes, gatherbRepeatParams);
    uint32_t posSrc1Brcb = hRepeat * vRepeat * DEFAULT_REPEAT_STRIDE * ONE_BLK_SIZE / sizeof(T);
    BrcbRepeatParams brcbRepeatParams;
    Brcb(sharedTmpBufferT[posSrc1Brcb], src1Local, src1Local.GetSize() / brcbEleNum, brcbRepeatParams);
    PipeBarrier<PIPE_V>();
    // mul
    BinaryRepeatParams mulRepeatParams;
    if (repeatMode == false) {
        mulRepeatParams.src0RepStride = 1;
        mulRepeatParams.src0BlkStride = 0;
    }

    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(0, innerRepeatTimes * eleCntOfOneRep);
    Mul<T, false>(sharedTmpBufferT, sharedTmpBufferT[posSrc1Brcb], sharedTmpBufferT, mask, innerRepeatTimes,
        mulRepeatParams);
    SetMaskNorm();
    ResetMask();
    PipeBarrier<PIPE_V>();

    BinaryRepeatParams addRepeatParams;
    addRepeatParams.dstRepStride = 0;
    addRepeatParams.src1RepStride = 0;
    for (int i = 0; i < vRepeat; i++) {
        if (hRepeat > 1) {
            Add(sharedTmpBufferT[i * hRepeat * eleCntOfOneRep], sharedTmpBufferT[(i * hRepeat + 1) * eleCntOfOneRep],
                sharedTmpBufferT[i * hRepeat * eleCntOfOneRep], mask, hRepeat - 1, addRepeatParams);
        }
    }
    PipeBarrier<PIPE_V>();
    // copy out
    UnaryRepeatParams addsRepeatParams;
    addsRepeatParams.srcRepStride = hRepeat * DEFAULT_REPEAT_STRIDE;
    addsRepeatParams.dstBlkStride = dstBlkStride;
    addsRepeatParams.dstRepStride = vROffset * sizeof(T) / ONE_BLK_SIZE;
    Adds(dstLocal, sharedTmpBufferT, (T)0, mask, vRepeat, addsRepeatParams);
}

template <typename T>
__aicore__ inline void BilinearInterpolationCalc(const LocalTensor<T> &dstLocal, const LocalTensor<T> &src0Local,
    const LocalTensor<uint32_t> &src0OffsetLocal, const LocalTensor<T> &src1Local, uint64_t mask[], uint8_t hRepeat,
    bool repeatMode, uint16_t dstBlkStride, uint16_t vROffset, uint8_t vRepeat,
    const LocalTensor<uint8_t> &sharedTmpBuffer)
{
    auto sharedTmpBufferT = sharedTmpBuffer.ReinterpretCast<T>();
    GatherRepeatParams gatherbRepeatParams;
    uint8_t innerRepeatTimes = hRepeat * vRepeat;
    constexpr uint32_t eleCntOfOneRep = DEFAULT_REPEAT_STRIDE * ONE_BLK_SIZE / sizeof(T);
    // gatherb
    GatherbImpl((__ubuf__ uint16_t *)sharedTmpBufferT.GetPhyAddr(), (__ubuf__ uint16_t *)src0Local.GetPhyAddr(),
        (__ubuf__ uint32_t *)src0OffsetLocal.GetPhyAddr(), src0Local.GetSize(), innerRepeatTimes, gatherbRepeatParams);
    uint32_t posSrc1Brcb = hRepeat * vRepeat * DEFAULT_REPEAT_STRIDE * ONE_BLK_SIZE / sizeof(T);
    BrcbRepeatParams brcbRepeatParams;
    Brcb(sharedTmpBufferT[posSrc1Brcb], src1Local, src1Local.GetSize() / brcbEleNum, brcbRepeatParams);
    PipeBarrier<PIPE_V>();
    // mul
    BinaryRepeatParams mulRepeatParams;
    if (repeatMode == false) {
        mulRepeatParams.src0RepStride = 1;
        mulRepeatParams.src0BlkStride = 0;
    }

    SetMaskCount();
    SetVectorMask<T, MaskMode::COUNTER>(0, innerRepeatTimes * eleCntOfOneRep);
    Mul<T, false>(sharedTmpBufferT, sharedTmpBufferT[posSrc1Brcb], sharedTmpBufferT, mask, innerRepeatTimes,
        mulRepeatParams);
    SetMaskNorm();
    ResetMask();
    PipeBarrier<PIPE_V>();

    BinaryRepeatParams addRepeatParams;
    addRepeatParams.dstRepStride = 0;
    addRepeatParams.src1RepStride = 0;
    for (int i = 0; i < vRepeat; i++) {
        if (hRepeat > 1) {
            Add(sharedTmpBufferT[i * hRepeat * eleCntOfOneRep], sharedTmpBufferT[(i * hRepeat + 1) * eleCntOfOneRep],
                sharedTmpBufferT[i * hRepeat * eleCntOfOneRep], mask, hRepeat - 1, addRepeatParams);
        }
    }
    PipeBarrier<PIPE_V>();
    // copy out
    UnaryRepeatParams addsRepeatParams;
    addsRepeatParams.srcRepStride = hRepeat * DEFAULT_REPEAT_STRIDE;
    addsRepeatParams.dstBlkStride = dstBlkStride;
    addsRepeatParams.dstRepStride = vROffset * sizeof(T) / ONE_BLK_SIZE;
    Adds(dstLocal, sharedTmpBufferT, (T)0, mask, vRepeat, addsRepeatParams);
}
} // namespace AscendC
#pragma end_pipe
#endif // ASCENDC_MODULE_OPERATOR_VEC_BILINEARINTERPALATION_IMPL_H