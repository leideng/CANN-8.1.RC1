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
 * \file kernel_operator_vec_brcb_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_BRCB_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_BRCB_IMPL_H
#include "kernel_struct_brcb.h"

namespace AscendC {
/* **************************************************************************************************
 * Brcb                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void BrcbImplB32Impl(__ubuf__ T* dst, __ubuf__ T* src0, const uint8_t repeatTimes,
    const BrcbRepeatParams& repeatParams)
{
    // only support for dstblockStride = 1
    constexpr uint16_t oneRepeatNum = 8;
    constexpr int32_t defaultTmpSize = 256;
    constexpr uint16_t defaultLen = 8;
    constexpr uint8_t defaultStride = 8;
    uint16_t dstRptEle = ONE_BLK_SIZE / sizeof(uint32_t) * repeatParams.dstRepStride;

    __ubuf__ int32_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<int32_t>(TMP_UB_OFFSET, defaultTmpSize);
    uint16_t srcOffset = 0;
    uint16_t dstOffset = 0;
    AscendCUtils::SetMask<uint32_t>(0, FULL_MASK);
    for (uint8_t i = 0; i < repeatTimes; i++) {
        // padding data to 8 * 8
        vadds((__ubuf__ int32_t*)tempBuf, (__ubuf__ int32_t*)src0 + srcOffset, static_cast<int32_t>(0),
            static_cast<uint8_t>(1), static_cast<uint16_t>(1), static_cast<uint16_t>(0), defaultStride,
            static_cast<uint8_t>(0));
        PipeBarrier<PIPE_V>();
        // transpose
        v4dtrans((__ubuf__ uint32_t*)dst + dstOffset, (__ubuf__ uint32_t*)tempBuf, defaultLen, defaultLen, true);
        PipeBarrier<PIPE_V>();
        srcOffset += oneRepeatNum;
        dstOffset += dstRptEle;
    }
}

template <typename T>
__aicore__ inline void BrcbImplB16Impl(__ubuf__ T* dst, __ubuf__ T* src0, const uint8_t repeatTimes,
    const BrcbRepeatParams& repeatParams)
{
    // only support for dstblockStride = 1
    constexpr uint32_t tmpUbSize = 1024;
    constexpr uint16_t residualOffset = 128;
    constexpr uint8_t defaultStride = 8;
    constexpr int32_t defaultTmpSize = 512;
    constexpr uint64_t maskValue = 0x00ff00ff00ff00ff;
    constexpr uint16_t oneRepeatNum = 8;
    uint16_t dstRptEle = ONE_BLK_SIZE / sizeof(uint16_t) * repeatParams.dstRepStride;

    __ubuf__ int16_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<int16_t>(TMP_UB_OFFSET + tmpUbSize,
        defaultTmpSize);
    __ubuf__ int16_t* tempBuf1 = (__ubuf__ int16_t*)tempBuf + residualOffset * 2;

    uint16_t srcOffset = 0;
    uint16_t dstOffset = 0;
    AscendCUtils::SetMask<int16_t>(FULL_MASK, FULL_MASK);
    for (uint8_t i = 0; i < repeatTimes; i++) {
        // save the rear data(128) in tmp buffer
        vadds((__ubuf__ int16_t*)tempBuf1, (__ubuf__ int16_t*)dst + dstOffset + residualOffset, static_cast<int16_t>(0),
            static_cast<uint8_t>(1), static_cast<uint16_t>(1), static_cast<uint16_t>(1), defaultStride, defaultStride);
        // padding to 16 * 16
        AscendCUtils::SetMask<int16_t>(maskValue, maskValue);
        vadds((__ubuf__ int16_t*)tempBuf, (__ubuf__ int16_t*)src0 + srcOffset, static_cast<int16_t>(0),
            static_cast<uint8_t>(2), static_cast<uint16_t>(1), static_cast<uint16_t>(0), defaultStride,
            static_cast<uint8_t>(0));
        PipeBarrier<PIPE_V>();
        vtranspose((__ubuf__ uint16_t*)dst + dstOffset, (__ubuf__ uint16_t*)tempBuf);
        PipeBarrier<PIPE_V>();
        // restore the rear data to dst from tmp buffer
        AscendCUtils::SetMask<uint16_t>(FULL_MASK, FULL_MASK);
        vadds((__ubuf__ int16_t*)dst + dstOffset + residualOffset, (__ubuf__ int16_t*)(tempBuf1),
            static_cast<int16_t>(0), static_cast<uint8_t>(1), static_cast<uint16_t>(1), static_cast<uint16_t>(1),
            defaultStride, defaultStride);
        PipeBarrier<PIPE_V>();
        srcOffset += oneRepeatNum;
        dstOffset += dstRptEle;
    }
}

template <typename T>
__aicore__ inline void BrcbImplStrideOne(__ubuf__ T* dst, __ubuf__ T* src0, const uint8_t repeatTimes,
    const BrcbRepeatParams& repeatParams)
{
    if constexpr (sizeof(T) == sizeof(uint16_t)) {
        BrcbImplB16Impl(dst, src0, repeatTimes, repeatParams);
    } else {
        BrcbImplB32Impl(dst, src0, repeatTimes, repeatParams);
    }
}

template <typename T>
__aicore__ inline void BrcbImpl(__ubuf__ T* dst, __ubuf__ T* src0, const uint8_t repeatTimes,
    const BrcbRepeatParams& repeatParams)
{
    SetMaskNorm();
    if constexpr(sizeof(T) != B16_BYTE_SIZE && sizeof(T) != B32_BYTE_SIZE) {
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Brcb, current api support dtype "
            "combination is src and dst both: half / int16_t / uint16_t / float / int32_t / uint32_t.");});
        return;
    }
    if (likely((repeatParams.dstBlkStride == 1) || (repeatParams.dstBlkStride == 0))) {
        BrcbImplStrideOne(dst, src0, repeatTimes, repeatParams);
    } else {
        constexpr uint8_t blockNum = 8;
        uint16_t dstblkEle = ONE_BLK_SIZE / sizeof(T) * repeatParams.dstBlkStride;
        uint16_t dstRptEle = ONE_BLK_SIZE / sizeof(T) * repeatParams.dstRepStride;
        uint64_t mask = ONE_BLK_SIZE / sizeof(T);
        AscendCUtils::SetMask<T>(mask);
        event_t eventID0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        event_t eventID1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        event_t eventID2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        event_t eventID3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE2_S>(eventID1);
        WaitFlag<HardEvent::MTE2_S>(eventID1);
        SetFlag<HardEvent::V_S>(eventID2);
        WaitFlag<HardEvent::V_S>(eventID2);
        SetFlag<HardEvent::MTE3_S>(eventID3);
        WaitFlag<HardEvent::MTE3_S>(eventID3);
        for (uint8_t i = 0; i < repeatTimes; i++) {
            for (uint8_t j = 0; j < blockNum; j++) {
                T scalarValue = *((__ubuf__ T*)src0 + i * blockNum + j);
                SetFlag<HardEvent::S_V>(eventID0);
                WaitFlag<HardEvent::S_V>(eventID0);
                vector_dup((__ubuf__ T*)(dst) + i * dstRptEle + j * dstblkEle, scalarValue, static_cast<uint8_t>(1),
                    static_cast<uint16_t>(1), static_cast<uint16_t>(1), blockNum, static_cast<uint8_t>(0));
            }
        }
    }
    ResetMask();
}

} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_BRCB_IMPL_H