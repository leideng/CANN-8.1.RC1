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
 * \file kernel_operator_vec_createvecindex_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_CREATEVECINDEX_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_CREATEVECINDEX_IMPL_H
#include "kernel_tensor.h"
#include "kernel_struct_unary.h"
#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif

namespace AscendC {
constexpr int32_t maskBitNum = 64;

template <typename T>
__aicore__ inline void CreateVecIndexOneBlk(const LocalTensor<T> &dstLocal, const T &firstValue, uint32_t calCount)
{
    for (int32_t i = 0; i < static_cast<int32_t>(calCount); i++) {
        dstLocal.SetValue(i, static_cast<float>(firstValue) + static_cast<float>(i));
    }
    auto eventIdSToV = GetTPipePtr()->FetchEventID(HardEvent::S_V);
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
}

template <typename T>
__aicore__ inline void CreateVecIndexOneRep(const LocalTensor<T> &dstLocal, const T &firstValue, uint64_t mask[],
    uint16_t dstBlkStride)
{
    constexpr int32_t eleCntOfOneBlk = (ONE_BLK_SIZE / sizeof(T));
    constexpr int32_t eleCntOfOneRep = (ONE_BLK_SIZE * DEFAULT_REPEAT_STRIDE / sizeof(T));
    int32_t maskNum = 2;
    if constexpr (sizeof(T) == sizeof(half)) {
        for (int32_t i = 0; i < maskNum; i++) {
            uint64_t maskValue = 1;
            for (int j = 0; j < maskBitNum; j++) {
                if (mask[i] & maskValue) {
                    uint32_t index = i * maskBitNum + j;
                    uint32_t blkIndex = index / eleCntOfOneBlk;
                    uint32_t eleIndex = blkIndex * eleCntOfOneBlk * dstBlkStride + index % eleCntOfOneBlk;
                    dstLocal.SetValue(eleIndex, (float)firstValue + (float)(i * maskBitNum + j));
                }
                maskValue <<= 1;
            }
        }
    } else {
        uint64_t maskValue = 1;
        for (int32_t j = 0; j < maskBitNum; j++) {
            if (mask[0] & maskValue) {
                uint32_t blkIndex = j / eleCntOfOneBlk;
                uint32_t eleIndex = blkIndex * eleCntOfOneBlk * dstBlkStride + j % eleCntOfOneBlk;
                dstLocal.SetValue(eleIndex, (float)firstValue + (float)j);
            }
            maskValue <<= 1;
        }
    }
    auto eventIdSToV = GetTPipePtr()->FetchEventID(HardEvent::S_V);
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
}

template <typename T>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<T> &dstLocal, const T &firstValue, uint64_t mask,
    uint8_t repeatTimes, uint16_t dstBlkStride, uint8_t dstRepStride)
{
    // 1st block
    constexpr int32_t eleCntOfOneBlk = (ONE_BLK_SIZE / sizeof(T));
    if (mask < eleCntOfOneBlk) {
        CreateVecIndexOneBlk(dstLocal, firstValue, mask);
    } else {
        CreateVecIndexOneBlk(dstLocal, firstValue, eleCntOfOneBlk);
    }
    constexpr int32_t eleCntOfOneRep = (ONE_BLK_SIZE * DEFAULT_REPEAT_STRIDE / sizeof(T));
    UnaryRepeatParams addsParams;
    // 2~8 block
    int32_t loopN = mask / eleCntOfOneBlk - 1;
    int32_t tailSize = mask % eleCntOfOneBlk;
    int32_t blkEleStride = dstBlkStride * eleCntOfOneBlk;
    int32_t repEleStride = dstRepStride * eleCntOfOneBlk;
    for (int i = 0; i < loopN; i++) {
        Adds(dstLocal[(i + 1) * blkEleStride], dstLocal[i * blkEleStride], (T)(eleCntOfOneBlk), eleCntOfOneBlk, 1,
            addsParams);
        PipeBarrier<PIPE_V>();
    }
    addsParams.dstBlkStride = dstBlkStride;
    addsParams.srcBlkStride = dstBlkStride;
    int32_t offsetTailDst = mask / eleCntOfOneBlk * eleCntOfOneBlk * dstBlkStride;
    int32_t offsetTailSrc = offsetTailDst - eleCntOfOneBlk * dstBlkStride;
    if (tailSize > 0) {
        Adds(dstLocal[offsetTailDst], dstLocal[offsetTailSrc], (T)eleCntOfOneBlk, tailSize, 1, addsParams);
        PipeBarrier<PIPE_V>();
    }

    // 2~n repeats
    for (int i = 0; i < repeatTimes - 1; i++) {
        Adds(dstLocal[(i + 1) * repEleStride], dstLocal[i * repEleStride], (T)(eleCntOfOneRep), mask, 1, addsParams);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<T> &dstLocal, const T &firstValue, uint64_t mask[],
    uint8_t repeatTimes, uint16_t dstBlkStride, uint8_t dstRepStride)
{
    // first rep
    CreateVecIndexOneRep(dstLocal, firstValue, mask, dstBlkStride);
    // 2~n repeats
    UnaryRepeatParams addsParams;
    addsParams.dstBlkStride = dstBlkStride;
    addsParams.srcBlkStride = dstBlkStride;
    constexpr int32_t eleCntOfOneBlk = (ONE_BLK_SIZE / sizeof(T));
    constexpr int32_t eleCntOfOneRep = (ONE_BLK_SIZE * DEFAULT_REPEAT_STRIDE / sizeof(T));
    int32_t blkEleStride = dstBlkStride * eleCntOfOneBlk;
    int32_t repEleStride = dstRepStride * eleCntOfOneBlk;
    for (int i = 0; i < repeatTimes - 1; i++) {
        Adds(dstLocal[(i + 1) * repEleStride], dstLocal[i * repEleStride], (T)(eleCntOfOneRep), mask, 1, addsParams);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void CreateVecIndexCalc(LocalTensor<T> dstLocal, const T &firstValue, uint32_t calCount)
{
    // first block
    constexpr int32_t eleCntOfOneBlk = (ONE_BLK_SIZE / sizeof(T));
    if (calCount <= eleCntOfOneBlk) {
        CreateVecIndexOneBlk(dstLocal, firstValue, (uint32_t)eleCntOfOneBlk);
        return;
    }
    CreateVecIndexOneBlk(dstLocal, firstValue, (uint32_t)eleCntOfOneBlk);

    UnaryRepeatParams addsParams;
    constexpr int32_t eleCntOfOneRep = (ONE_BLK_SIZE * DEFAULT_REPEAT_STRIDE / sizeof(T));
    // 2~8 block
    int32_t loopN = 0, tailSize = 0, offsetTailDst, offsetTailSrc;
    if (calCount >= eleCntOfOneRep) {
        loopN = DEFAULT_REPEAT_STRIDE - 1;
    } else {
        loopN = calCount / eleCntOfOneBlk - 1;
        tailSize = calCount % eleCntOfOneBlk;
    }
    for (int i = 0; i < loopN; i++) {
        Adds(dstLocal[(i + 1) * eleCntOfOneBlk], dstLocal[i * eleCntOfOneBlk], (T)eleCntOfOneBlk, eleCntOfOneBlk, 1,
            addsParams);
        PipeBarrier<PIPE_V>();
    }
    offsetTailDst = calCount / eleCntOfOneBlk * eleCntOfOneBlk;
    offsetTailSrc = offsetTailDst - eleCntOfOneBlk;
    if (tailSize > 0) {
        Adds(dstLocal[offsetTailDst], dstLocal[offsetTailSrc], (T)eleCntOfOneBlk, tailSize, 1, addsParams);
        PipeBarrier<PIPE_V>();
    }
    if (calCount <= eleCntOfOneRep) {
        return;
    }
    // 2~n repeats
    loopN = calCount / eleCntOfOneRep - 1;
    tailSize = calCount % eleCntOfOneRep;
    for (int i = 0; i < loopN; i++) {
        Adds(dstLocal[(i + 1) * eleCntOfOneRep], dstLocal[i * eleCntOfOneRep], (T)(eleCntOfOneRep), eleCntOfOneRep, 1,
            addsParams);
        PipeBarrier<PIPE_V>();
    }
    offsetTailDst = calCount / eleCntOfOneRep * eleCntOfOneRep;
    offsetTailSrc = offsetTailDst - eleCntOfOneRep;
    if (tailSize > 0) {
        Adds(dstLocal[offsetTailDst], dstLocal[offsetTailSrc], (T)(eleCntOfOneRep), tailSize, 1, addsParams);
    }
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_CREATEVECINDEX_IMPL_H