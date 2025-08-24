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
 * \file kernel_operator_vec_reduce_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_REDUCE_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_REDUCE_IMPL_H

namespace AscendC {
#define VCPADD_FUNC() vcpadd(vreg1, vreg0, preg, MODE_ZEROING)
#define VCGADD_FUNC() vcgadd(vreg1, vreg0, preg, MODE_ZEROING)
#define VCGMAX_FUNC() vcgmax(vreg1, vreg0, preg, MODE_ZEROING)
#define VCGMIN_FUNC() vcgmin(vreg1, vreg0, preg, MODE_ZEROING)
#define VCMAX_FUNC() vcmax(vreg1, vreg0, preg, MODE_ZEROING)
#define VCMIN_FUNC() vcmin(vreg1, vreg0, preg, MODE_ZEROING)
#define VCADD_FUNC() vcadd(vreg1, vreg0, preg, MODE_ZEROING)

#define CONTINUOUS_MODE_REDUCE_VF(reducefunc, vregType, pltType, dstStrideOffset) \
    __VEC_SCOPE__                                                                 \
    {                                                                             \
        vector_##vregType vreg0;                                                  \
        vector_##vregType vreg1;                                                  \
        vector_align ureg;                                                        \
        uint32_t sreg = (uint32_t)mask;                                           \
        vector_bool preg = plt_##pltType(sreg, POST_UPDATE);                      \
        uint32_t strideConfig = (((uint32_t)srcBlkStride) << 16);                 \
        for (uint16_t i = 0; i < (uint16_t)newRepeat; ++i) {                      \
            vsldb(vreg0, newSrc + i * srcStrideOffset, strideConfig, preg);       \
            reducefunc();                                                         \
            vstus(ureg, dstStrideOffset, vreg1, dst, POST_UPDATE);                \
            vstas(ureg, dst, dstStrideOffset*(newDstRepStride - 1), POST_UPDATE); \
        }                                                                         \
    }

#define BITBYBIT_MODE_HALF_REDUCE_VF(reducefunc, dstStrideOffset)                 \
    __VEC_SCOPE__                                                                 \
    {                                                                             \
        vector_f16 vreg0;                                                         \
        vector_f16 vreg1;                                                         \
        vector_align ureg;                                                        \
        vector_bool preg;                                                         \
        plds(preg, ((__ubuf__ uint32_t*)tempBuf), 0, US);                         \
        uint32_t strideConfig = (((uint32_t)srcBlkStride) << 16);                 \
        for (uint16_t i = 0; i < (uint16_t)newRepeat; ++i) {                      \
            vsldb(vreg0, newSrc + i * srcStrideOffset, strideConfig, preg);       \
            reducefunc();                                                         \
            vstus(ureg, dstStrideOffset, vreg1, dst, POST_UPDATE);                \
            vstas(ureg, dst, dstStrideOffset*(newDstRepStride - 1), POST_UPDATE); \
        }                                                                         \
    }

#define BITBYBIT_MODE_FLOAT_REDUCE_VF(reducefunc, dstStrideOffset)                \
    __VEC_SCOPE__                                                                 \
    {                                                                             \
        vector_f32 vreg0;                                                         \
        vector_f32 vreg1;                                                         \
        vector_align ureg;                                                        \
        vector_bool preg;                                                         \
        vector_bool preg1;                                                        \
        plds(preg1, ((__ubuf__ uint32_t*)tempBuf), 0, US);                        \
        punpack(preg, preg1, LOWER);                                              \
        uint32_t strideConfig = (((uint32_t)srcBlkStride) << 16);                 \
        for (uint16_t i = 0; i < (uint16_t)newRepeat; ++i) {                      \
            vsldb(vreg0, newSrc + i * srcStrideOffset, strideConfig, preg);       \
            reducefunc();                                                         \
            vstus(ureg, dstStrideOffset, vreg1, dst, POST_UPDATE);                \
            vstas(ureg, dst, dstStrideOffset*(newDstRepStride - 1), POST_UPDATE); \
        }                                                                         \
    }

/* **************************************** Pair Reduce Impl ****************************************** */
template <typename T, bool isSetMask = true>
__aicore__ inline void PairReduceSumImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t repeat, const int32_t mask,
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void PairReduceSumImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t repeat, const uint64_t mask[],
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void PairReduceSumImpl(__ubuf__ half* dst, __ubuf__ half* src, const int32_t repeat,
    const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    int32_t newRepeat = repeat;
    uint32_t srcStrideOffset = srcRepStride * ONE_BLK_HALF_NUM;
    uint32_t newDstRepStride = dstRepStride;
    __ubuf__ half* newSrc = src;
    if (dstRepStride == 0 && repeat > 0) {
        newRepeat = 1;
        srcStrideOffset = srcRepStride * ONE_BLK_HALF_NUM * (repeat - 1);
        newSrc += srcStrideOffset;
        newDstRepStride = 1;
    }
    BITBYBIT_MODE_HALF_REDUCE_VF(VCPADD_FUNC, FULL_MASK_LEN / HALF_FACTOR);
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void PairReduceSumImpl(__ubuf__ float* dst, __ubuf__ float* src, const int32_t repeat,
    const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    int32_t newRepeat = repeat;
    uint32_t srcStrideOffset = srcRepStride * ONE_BLK_FLOAT_NUM;
    uint32_t newDstRepStride = dstRepStride;
    __ubuf__ float* newSrc = src;
    if (dstRepStride == 0 && repeat > 0) {
        newRepeat = 1;
        srcStrideOffset = srcRepStride * ONE_BLK_FLOAT_NUM * (repeat - 1);
        newSrc += srcStrideOffset;
        newDstRepStride = 1;
    }
    BITBYBIT_MODE_FLOAT_REDUCE_VF(VCPADD_FUNC, HLAF_MASK_LEN / HALF_FACTOR);
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void PairReduceSumImpl(__ubuf__ half* dst, __ubuf__ half* src, const int32_t repeat,
    const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    int32_t newRepeat = repeat;
    uint32_t srcStrideOffset = srcRepStride * ONE_BLK_HALF_NUM;
    uint32_t newDstRepStride = dstRepStride;
    __ubuf__ half* newSrc = src;
    if (dstRepStride == 0 && repeat > 0) {
        newRepeat = 1;
        srcStrideOffset = srcRepStride * ONE_BLK_HALF_NUM * (repeat - 1);
        newSrc += srcStrideOffset;
        newDstRepStride = 1;
    }
    CONTINUOUS_MODE_REDUCE_VF(VCPADD_FUNC, f16, b16, FULL_MASK_LEN / HALF_FACTOR);
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void PairReduceSumImpl(__ubuf__ float* dst, __ubuf__ float* src, const int32_t repeat,
    const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    int32_t newRepeat = repeat;
    uint32_t srcStrideOffset = srcRepStride * ONE_BLK_FLOAT_NUM;
    uint32_t newDstRepStride = dstRepStride;
    __ubuf__ float* newSrc = src;
    if (dstRepStride == 0 && repeat > 0) {
        newRepeat = 1;
        srcStrideOffset = srcRepStride * ONE_BLK_FLOAT_NUM * (repeat - 1);
        newSrc += srcStrideOffset;
        newDstRepStride = 1;
    }
    CONTINUOUS_MODE_REDUCE_VF(VCPADD_FUNC, f32, b32, HLAF_MASK_LEN / HALF_FACTOR);
}

/* **************************************** Block Reduce Impl ****************************************** */
template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceSumImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t repeat,
    const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}
template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceSumImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t repeat, const int32_t mask,
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void BlockReduceSumImpl(__ubuf__ half* dst, __ubuf__ half* src, const int32_t repeat,
    const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    int32_t newRepeat = repeat;
    uint32_t srcStrideOffset = srcRepStride * ONE_BLK_HALF_NUM;
    uint32_t newDstRepStride = dstRepStride;
    __ubuf__ half* newSrc = src;
    if (dstRepStride == 0 && repeat > 0) {
        newRepeat = 1;
        srcStrideOffset = srcRepStride * ONE_BLK_HALF_NUM * (repeat - 1);
        newSrc += srcStrideOffset;
        newDstRepStride = 1;
    }
    BITBYBIT_MODE_HALF_REDUCE_VF(VCGADD_FUNC, DEFAULT_BLK_NUM);
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void BlockReduceSumImpl(__ubuf__ float* dst, __ubuf__ float* src, const int32_t repeat,
    const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    int32_t newRepeat = repeat;
    uint32_t srcStrideOffset = srcRepStride * ONE_BLK_FLOAT_NUM;
    uint32_t newDstRepStride = dstRepStride;
    __ubuf__ float* newSrc = src;
    if (dstRepStride == 0 && repeat > 0) {
        newRepeat = 1;
        srcStrideOffset = srcRepStride * ONE_BLK_FLOAT_NUM * (repeat - 1);
        newSrc += srcStrideOffset;
        newDstRepStride = 1;
    }
    BITBYBIT_MODE_FLOAT_REDUCE_VF(VCGADD_FUNC, DEFAULT_BLK_NUM);
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void BlockReduceSumImpl(__ubuf__ half* dst, __ubuf__ half* src, const int32_t repeat,
    const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    int32_t newRepeat = repeat;
    uint32_t srcStrideOffset = srcRepStride * ONE_BLK_HALF_NUM;
    uint32_t newDstRepStride = dstRepStride;
    __ubuf__ half* newSrc = src;
    if (dstRepStride == 0 && repeat > 0) {
        newRepeat = 1;
        srcStrideOffset = srcRepStride * ONE_BLK_HALF_NUM * (repeat - 1);
        newSrc += srcStrideOffset;
        newDstRepStride = 1;
    }
    CONTINUOUS_MODE_REDUCE_VF(VCGADD_FUNC, f16, b16, DEFAULT_BLK_NUM);
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void BlockReduceSumImpl(__ubuf__ float* dst, __ubuf__ float* src, const int32_t repeat,
    const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    int32_t newRepeat = repeat;
    uint32_t srcStrideOffset = srcRepStride * ONE_BLK_FLOAT_NUM;
    uint32_t newDstRepStride = dstRepStride;
    __ubuf__ float* newSrc = src;
    if (dstRepStride == 0 && repeat > 0) {
        newRepeat = 1;
        srcStrideOffset = srcRepStride * ONE_BLK_FLOAT_NUM * (repeat - 1);
        newSrc += srcStrideOffset;
        newDstRepStride = 1;
    }
    CONTINUOUS_MODE_REDUCE_VF(VCGADD_FUNC, f32, b32, DEFAULT_BLK_NUM);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceMaxImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t repeat,
    const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceMaxImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t repeat, const int32_t mask,
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void BlockReduceMaxImpl(__ubuf__ half* dst, __ubuf__ half* src, const int32_t repeat,
    const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    int32_t newRepeat = repeat;
    uint32_t srcStrideOffset = srcRepStride * ONE_BLK_HALF_NUM;
    uint32_t newDstRepStride = dstRepStride;
    __ubuf__ half* newSrc = src;
    if (dstRepStride == 0 && repeat > 0) {
        newRepeat = 1;
        srcStrideOffset = srcRepStride * ONE_BLK_HALF_NUM * (repeat - 1);
        newSrc += srcStrideOffset;
        newDstRepStride = 1;
    }
    BITBYBIT_MODE_HALF_REDUCE_VF(VCGMAX_FUNC, DEFAULT_BLK_NUM);
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void BlockReduceMaxImpl(__ubuf__ half* dst, __ubuf__ half* src, const int32_t repeat,
    const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    int32_t newRepeat = repeat;
    uint32_t srcStrideOffset = srcRepStride * ONE_BLK_HALF_NUM;
    uint32_t newDstRepStride = dstRepStride;
    __ubuf__ half* newSrc = src;
    if (dstRepStride == 0 && repeat > 0) {
        newRepeat = 1;
        srcStrideOffset = srcRepStride * ONE_BLK_HALF_NUM * (repeat - 1);
        newSrc += srcStrideOffset;
        newDstRepStride = 1;
    }
    CONTINUOUS_MODE_REDUCE_VF(VCGMAX_FUNC, f16, b16, DEFAULT_BLK_NUM);
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void BlockReduceMaxImpl(__ubuf__ float* dst, __ubuf__ float* src, const int32_t repeat,
    const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    int32_t newRepeat = repeat;
    uint32_t srcStrideOffset = srcRepStride * ONE_BLK_FLOAT_NUM;
    uint32_t newDstRepStride = dstRepStride;
    __ubuf__ float* newSrc = src;
    if (dstRepStride == 0 && repeat > 0) {
        newRepeat = 1;
        srcStrideOffset = srcRepStride * ONE_BLK_FLOAT_NUM * (repeat - 1);
        newSrc += srcStrideOffset;
        newDstRepStride = 1;
    }
    BITBYBIT_MODE_FLOAT_REDUCE_VF(VCGMAX_FUNC, DEFAULT_BLK_NUM);
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void BlockReduceMaxImpl(__ubuf__ float* dst, __ubuf__ float* src, const int32_t repeat,
    const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    int32_t newRepeat = repeat;
    uint32_t srcStrideOffset = srcRepStride * ONE_BLK_FLOAT_NUM;
    uint32_t newDstRepStride = dstRepStride;
    __ubuf__ float* newSrc = src;
    if (dstRepStride == 0 && repeat > 0) {
        newRepeat = 1;
        srcStrideOffset = srcRepStride * ONE_BLK_FLOAT_NUM * (repeat - 1);
        newSrc += srcStrideOffset;
        newDstRepStride = 1;
    }
    CONTINUOUS_MODE_REDUCE_VF(VCGMAX_FUNC, f32, b32, DEFAULT_BLK_NUM);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceMinImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t repeat,
    const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceMinImpl(__ubuf__ T* dst, __ubuf__ T* src, const int32_t repeat, const int32_t mask,
    const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void BlockReduceMinImpl(__ubuf__ half* dst, __ubuf__ half* src, const int32_t repeat,
    const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    int32_t newRepeat = repeat;
    uint32_t srcStrideOffset = srcRepStride * ONE_BLK_HALF_NUM;
    uint32_t newDstRepStride = dstRepStride;
    __ubuf__ half* newSrc = src;
    if (dstRepStride == 0 && repeat > 0) {
        newRepeat = 1;
        srcStrideOffset = srcRepStride * ONE_BLK_HALF_NUM * (repeat - 1);
        newSrc += srcStrideOffset;
        newDstRepStride = 1;
    }
    BITBYBIT_MODE_HALF_REDUCE_VF(VCGMIN_FUNC, DEFAULT_BLK_NUM);
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void BlockReduceMinImpl(__ubuf__ half* dst, __ubuf__ half* src, const int32_t repeat,
    const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    int32_t newRepeat = repeat;
    uint32_t srcStrideOffset = srcRepStride * ONE_BLK_HALF_NUM;
    uint32_t newDstRepStride = dstRepStride;
    __ubuf__ half* newSrc = src;
    if (dstRepStride == 0 && repeat > 0) {
        newRepeat = 1;
        srcStrideOffset = srcRepStride * ONE_BLK_HALF_NUM * (repeat - 1);
        newSrc += srcStrideOffset;
        newDstRepStride = 1;
    }
    CONTINUOUS_MODE_REDUCE_VF(VCGMIN_FUNC, f16, b16, DEFAULT_BLK_NUM);
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void BlockReduceMinImpl(__ubuf__ float* dst, __ubuf__ float* src, const int32_t repeat,
    const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    int32_t newRepeat = repeat;
    uint32_t srcStrideOffset = srcRepStride * ONE_BLK_FLOAT_NUM;
    uint32_t newDstRepStride = dstRepStride;
    __ubuf__ float* newSrc = src;
    if (dstRepStride == 0 && repeat > 0) {
        newRepeat = 1;
        srcStrideOffset = srcRepStride * ONE_BLK_FLOAT_NUM * (repeat - 1);
        newSrc += srcStrideOffset;
        newDstRepStride = 1;
    }
    BITBYBIT_MODE_FLOAT_REDUCE_VF(VCGMIN_FUNC, DEFAULT_BLK_NUM);
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void BlockReduceMinImpl(__ubuf__ float* dst, __ubuf__ float* src, const int32_t repeat,
    const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    int32_t newRepeat = repeat;
    uint32_t srcStrideOffset = srcRepStride * ONE_BLK_FLOAT_NUM;
    uint32_t newDstRepStride = dstRepStride;
    __ubuf__ float* newSrc = src;
    if (dstRepStride == 0 && repeat > 0) {
        newRepeat = 1;
        srcStrideOffset = srcRepStride * ONE_BLK_FLOAT_NUM * (repeat - 1);
        newSrc += srcStrideOffset;
        newDstRepStride = 1;
    }
    CONTINUOUS_MODE_REDUCE_VF(VCGMIN_FUNC, f32, b32, DEFAULT_BLK_NUM);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void RepeatReduceSumImpl(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const int32_t repeat,
    const int32_t elemsInOneRepeate, const int32_t dstBlkStride, const int32_t srcBlkStride, const int32_t dstRepStride,
    const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

/* **************************************** Whole Reduce Interface ****************************************** */
template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMaxImpl(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const uint64_t mask[],
    const int32_t repeat, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const ReduceOrder order)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMaxImpl(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const int32_t mask,
    const int32_t repeat, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const ReduceOrder order)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void WholeReduceMaxImpl(__ubuf__ half* dst, __ubuf__ half* src, const int32_t mask,
    const int32_t repeat, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const ReduceOrder order)
{
    int32_t newRepeat = repeat;
    uint32_t srcStrideOffset = srcRepStride * ONE_BLK_HALF_NUM;
    uint32_t newDstRepStride = dstRepStride;
    uint32_t dstStrideOffset = (dstRepStride > 1) ? 2 : 1;
    __ubuf__ half* newSrc = src;
    if (dstRepStride == 0 && repeat > 0) {
        newRepeat = 1;
        srcStrideOffset = srcRepStride * ONE_BLK_HALF_NUM * (repeat - 1);
        newSrc += srcStrideOffset;
        newDstRepStride = 1;
        dstStrideOffset = 1;
    }
    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_align ureg;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b16(sreg, POST_UPDATE);
        uint32_t strideConfig = (((uint32_t)srcBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)newRepeat; ++i) {
            vsldb(vreg0, newSrc + i * srcStrideOffset, strideConfig, preg);
            vcmax(vreg1, vreg0, preg, MODE_ZEROING);
            vstus(ureg, dstStrideOffset, vreg1, dst, POST_UPDATE);
            vstas(ureg, dst, newDstRepStride - dstStrideOffset, POST_UPDATE);
        }
    }
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void WholeReduceMaxImpl(__ubuf__ float* dst, __ubuf__ float* src, const int32_t mask,
    const int32_t repeat, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const ReduceOrder order)
{
    int32_t newRepeat = repeat;
    uint32_t srcStrideOffset = srcRepStride * ONE_BLK_FLOAT_NUM;
    uint32_t newDstRepStride = dstRepStride;
    uint32_t dstStrideOffset = (dstRepStride > 1) ? 2 : 1;
    __ubuf__ float* newSrc = src;
    if (dstRepStride == 0 && repeat > 0) {
        newRepeat = 1;
        srcStrideOffset = srcRepStride * ONE_BLK_FLOAT_NUM * (repeat - 1);
        newSrc += srcStrideOffset;
        newDstRepStride = 1;
        dstStrideOffset = 1;
    }
    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        vector_align ureg;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b32(sreg, POST_UPDATE);
        uint32_t strideConfig = (((uint32_t)srcBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)newRepeat; ++i) {
            vsldb(vreg0, newSrc + i * srcStrideOffset, strideConfig, preg);
            vcmax(vreg1, vreg0, preg, MODE_ZEROING);
            vstus(ureg, dstStrideOffset, vreg1, dst, POST_UPDATE);
            vstas(ureg, dst, newDstRepStride - dstStrideOffset, POST_UPDATE);
        }
    }
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void WholeReduceMaxImpl(__ubuf__ half* dst, __ubuf__ half* src, const uint64_t mask[],
    const int32_t repeat, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const ReduceOrder order)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    int32_t newRepeat = repeat;
    uint32_t newDstRepStride = dstRepStride;
    uint32_t dstStrideOffset = (dstRepStride > 1) ? 2 : 1;
    __ubuf__ half* newSrc = src;
    uint32_t srcStrideOffset = srcRepStride * ONE_BLK_HALF_NUM;
    if (dstRepStride == 0 && repeat > 0) {
        newRepeat = 1;
        srcStrideOffset = srcRepStride * ONE_BLK_HALF_NUM * (repeat - 1);
        newSrc += srcStrideOffset;
        newDstRepStride = 1;
        dstStrideOffset = 1;
    }
    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_align ureg;
        vector_bool preg;
        plds(preg, ((__ubuf__ uint32_t*)tempBuf), 0, US);
        uint32_t strideConfig = (((uint32_t)srcBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)newRepeat; ++i) {
            vsldb(vreg0, newSrc + i * srcStrideOffset, strideConfig, preg);
            vcmax(vreg1, vreg0, preg, MODE_ZEROING);
            vstus(ureg, dstStrideOffset, vreg1, dst, POST_UPDATE);
            vstas(ureg, dst, newDstRepStride - dstStrideOffset, POST_UPDATE);
        }
    }
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void WholeReduceMaxImpl(__ubuf__ float* dst, __ubuf__ float* src, const uint64_t mask[],
    const int32_t repeat, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const ReduceOrder order)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    int32_t newRepeat = repeat;
    uint32_t srcStrideOffset = srcRepStride * ONE_BLK_FLOAT_NUM;
    uint32_t newDstRepStride = dstRepStride;
    uint32_t dstStrideOffset = (dstRepStride > 1) ? 2 : 1;
    __ubuf__ float* newSrc = src;
    if (dstRepStride == 0 && repeat > 0) {
        newRepeat = 1;
        srcStrideOffset = srcRepStride * ONE_BLK_FLOAT_NUM * (repeat - 1);
        newSrc += srcStrideOffset;
        newDstRepStride = 1;
        dstStrideOffset = 1;
    }
    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        vector_align ureg;
        vector_bool preg;
        vector_bool preg1;
        plds(preg1, ((__ubuf__ uint32_t*)tempBuf), 0, US);
        punpack(preg, preg1, LOWER);
        uint32_t strideConfig = (((uint32_t)srcBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)newRepeat; ++i) {
            vsldb(vreg0, newSrc + i * srcStrideOffset, strideConfig, preg);
            vcmax(vreg1, vreg0, preg, MODE_ZEROING);
            vstus(ureg, dstStrideOffset, vreg1, dst, POST_UPDATE);
            vstas(ureg, dst, newDstRepStride - dstStrideOffset, POST_UPDATE);
        }
    }
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMinImpl(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const uint64_t mask[],
    const int32_t repeat, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const ReduceOrder order)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMinImpl(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const int32_t mask,
    const int32_t repeat, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const ReduceOrder order)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void WholeReduceMinImpl(__ubuf__ half* dst, __ubuf__ half* src, const int32_t mask,
    const int32_t repeat, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const ReduceOrder order)
{
    int32_t newRepeat = repeat;
    uint32_t srcStrideOffset = srcRepStride * ONE_BLK_HALF_NUM;
    uint32_t newDstRepStride = dstRepStride;
    uint32_t dstStrideOffset = (dstRepStride > 1) ? 2 : 1;
    __ubuf__ half* newSrc = src;
    if (dstRepStride == 0 && repeat > 0) {
        newRepeat = 1;
        srcStrideOffset = srcRepStride * ONE_BLK_HALF_NUM * (repeat - 1);
        newSrc += srcStrideOffset;
        newDstRepStride = 1;
        dstStrideOffset = 1;
    }
    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_align ureg;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b16(sreg, POST_UPDATE);
        uint32_t strideConfig = (((uint32_t)srcBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)newRepeat; ++i) {
            vsldb(vreg0, newSrc + i * srcStrideOffset, strideConfig, preg);
            vcmin(vreg1, vreg0, preg, MODE_ZEROING);
            vstus(ureg, dstStrideOffset, vreg1, dst, POST_UPDATE);
            vstas(ureg, dst, newDstRepStride - dstStrideOffset, POST_UPDATE);
        }
    }
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void WholeReduceMinImpl(__ubuf__ float* dst, __ubuf__ float* src, const int32_t mask,
    const int32_t repeat, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const ReduceOrder order)
{
    int32_t newRepeat = repeat;
    uint32_t srcStrideOffset = srcRepStride * ONE_BLK_FLOAT_NUM;
    uint32_t newDstRepStride = dstRepStride;
    uint32_t dstStrideOffset = (dstRepStride > 1) ? 2 : 1;
    __ubuf__ float* newSrc = src;
    if (dstRepStride == 0 && repeat > 0) {
        newRepeat = 1;
        srcStrideOffset = srcRepStride * ONE_BLK_FLOAT_NUM * (repeat - 1);
        newSrc += srcStrideOffset;
        newDstRepStride = 1;
        dstStrideOffset = 1;
    }
    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        vector_align ureg;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b32(sreg, POST_UPDATE);
        uint32_t strideConfig = (((uint32_t)srcBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)newRepeat; ++i) {
            vsldb(vreg0, newSrc + i * srcStrideOffset, strideConfig, preg);
            vcmin(vreg1, vreg0, preg, MODE_ZEROING);
            vstus(ureg, dstStrideOffset, vreg1, dst, POST_UPDATE);
            vstas(ureg, dst, newDstRepStride - dstStrideOffset, POST_UPDATE);
        }
    }
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void WholeReduceMinImpl(__ubuf__ half* dst, __ubuf__ half* src, const uint64_t mask[],
    const int32_t repeat, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const ReduceOrder order)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    int32_t newRepeat = repeat;
    uint32_t srcStrideOffset = srcRepStride * ONE_BLK_HALF_NUM;
    uint32_t newDstRepStride = dstRepStride;
    uint32_t dstStrideOffset = (dstRepStride > 1) ? 2 : 1;
    __ubuf__ half* newSrc = src;
    if (dstRepStride == 0 && repeat > 0) {
        newRepeat = 1;
        srcStrideOffset = srcRepStride * ONE_BLK_HALF_NUM * (repeat - 1);
        newSrc += srcStrideOffset;
        newDstRepStride = 1;
        dstStrideOffset = 1;
    }

    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_align ureg;
        vector_bool preg;
        plds(preg, ((__ubuf__ uint32_t*)tempBuf), 0, US);
        uint32_t strideConfig = (((uint32_t)srcBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)newRepeat; ++i) {
            vsldb(vreg0, newSrc + i * srcStrideOffset, strideConfig, preg);
            vcmin(vreg1, vreg0, preg, MODE_ZEROING);
            vstus(ureg, dstStrideOffset, vreg1, dst, POST_UPDATE);
            vstas(ureg, dst, newDstRepStride - dstStrideOffset, POST_UPDATE);
        }
    }
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void WholeReduceMinImpl(__ubuf__ float* dst, __ubuf__ float* src, const uint64_t mask[],
    const int32_t repeat, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const ReduceOrder order)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    int32_t newRepeat = repeat;
    uint32_t srcStrideOffset = srcRepStride * ONE_BLK_FLOAT_NUM;
    uint32_t newDstRepStride = dstRepStride;
    uint32_t dstStrideOffset = (dstRepStride > 1) ? 2 : 1;
    __ubuf__ float* newSrc = src;
    if (dstRepStride == 0 && repeat > 0) {
        newRepeat = 1;
        srcStrideOffset = srcRepStride * ONE_BLK_FLOAT_NUM * (repeat - 1);
        newSrc += srcStrideOffset;
        newDstRepStride = 1;
        dstStrideOffset = 1;
    }
    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        vector_align ureg;
        vector_bool preg;
        vector_bool preg1;
        plds(preg1, ((__ubuf__ uint32_t*)tempBuf), 0, US);
        punpack(preg, preg1, LOWER);
        uint32_t strideConfig = (((uint32_t)srcBlkStride) << 16);
        for (uint16_t i = 0; i < (uint16_t)newRepeat; ++i) {
            vsldb(vreg0, newSrc + i * srcStrideOffset, strideConfig, preg);
            vcmin(vreg1, vreg0, preg, MODE_ZEROING);
            vstus(ureg, dstStrideOffset, vreg1, dst, POST_UPDATE);
            vstas(ureg, dst, newDstRepStride - dstStrideOffset, POST_UPDATE);
        }
    }
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceSumImpl(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const uint64_t mask[],
    const int32_t repeat, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceSumImpl(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const int32_t mask,
    const int32_t repeat, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported on current device"); });
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void WholeReduceSumImpl(__ubuf__ half* dst, __ubuf__ half* src, const int32_t mask,
    const int32_t repeat, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    int32_t newRepeat = repeat;
    uint32_t srcStrideOffset = srcRepStride * ONE_BLK_HALF_NUM;
    uint32_t newDstRepStride = dstRepStride;
    __ubuf__ half* newSrc = src;
    if (dstRepStride == 0 && repeat > 0) {
        newRepeat = 1;
        srcStrideOffset = srcRepStride * ONE_BLK_HALF_NUM * (repeat - 1);
        newSrc += srcStrideOffset;
        newDstRepStride = 1;
    }
    CONTINUOUS_MODE_REDUCE_VF(VCADD_FUNC, f16, b16, 1);
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void WholeReduceSumImpl(__ubuf__ float* dst, __ubuf__ float* src, const int32_t mask,
    const int32_t repeat, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    int32_t newRepeat = repeat;
    uint32_t srcStrideOffset = srcRepStride * ONE_BLK_FLOAT_NUM;
    uint32_t newDstRepStride = dstRepStride;
    __ubuf__ float* newSrc = src;
    if (dstRepStride == 0 && repeat > 0) {
        newRepeat = 1;
        srcStrideOffset = srcRepStride * ONE_BLK_FLOAT_NUM * (repeat - 1);
        newSrc += srcStrideOffset;
        newDstRepStride = 1;
    }
    CONTINUOUS_MODE_REDUCE_VF(VCADD_FUNC, f32, b32, 1);
}

template <typename T = half, bool isSetMask = true>
__aicore__ inline void WholeReduceSumImpl(__ubuf__ half* dst, __ubuf__ half* src, const uint64_t mask[],
    const int32_t repeat, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    int32_t newRepeat = repeat;
    uint32_t srcStrideOffset = srcRepStride * ONE_BLK_HALF_NUM;
    uint32_t newDstRepStride = dstRepStride;
    __ubuf__ half* newSrc = src;
    if (dstRepStride == 0 && repeat > 0) {
        newRepeat = 1;
        srcStrideOffset = srcRepStride * ONE_BLK_HALF_NUM * (repeat - 1);
        newSrc += srcStrideOffset;
        newDstRepStride = 1;
    }
    BITBYBIT_MODE_HALF_REDUCE_VF(VCADD_FUNC, 1);
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

template <typename T = float, bool isSetMask = true>
__aicore__ inline void WholeReduceSumImpl(__ubuf__ float* dst, __ubuf__ float* src, const uint64_t mask[],
    const int32_t repeat, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf)) = ((uint64_t)mask[0]);
    (*(__ubuf__ uint64_t*)((__ubuf__ uint64_t*)tempBuf + 1)) = ((uint64_t)mask[1]);

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    int32_t newRepeat = repeat;
    uint32_t srcStrideOffset = srcRepStride * ONE_BLK_FLOAT_NUM;
    uint32_t newDstRepStride = dstRepStride;
    __ubuf__ float* newSrc = src;
    if (dstRepStride == 0 && repeat > 0) {
        newRepeat = 1;
        srcStrideOffset = srcRepStride * ONE_BLK_FLOAT_NUM * (repeat - 1);
        newSrc += srcStrideOffset;
        newDstRepStride = 1;
    }
    BITBYBIT_MODE_FLOAT_REDUCE_VF(VCADD_FUNC, 1);
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

/* **************************************** Reduce Interface ****************************************** */
template <typename T>
__aicore__ inline void ReduceMaxIntrinsicsImpl(__ubuf__ T* workLocal, __ubuf__ T* srcLocal, const int32_t mask,
    const int32_t repeatTimes, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMaxIntrinsicsImpl is not supported!"); });
}

template <typename T>
__aicore__ inline void ReduceMaxIntrinsicsImpl(__ubuf__ T* workLocal, __ubuf__ T* srcLocal, const uint64_t mask[],
    const int32_t repeatTimes, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMaxIntrinsicsImpl is not supported!"); });
}

__aicore__ inline void ReduceMaxIntrinsicsImpl(__ubuf__ half* workLocal, __ubuf__ half* srcLocal, const int32_t mask,
    const int32_t repeatTimes, const int32_t srcRepStride)
{
    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_align ureg;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b16(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)1) << 16);
        uint32_t strideConfig1 = (((uint32_t)0) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, srcLocal + i * srcRepStride * 16, strideConfig0, preg);
            vcmax(vreg1, vreg0, preg, MODE_ZEROING);
            vstus(ureg, 2, vreg1, workLocal, POST_UPDATE);
        }
        vstas(ureg, workLocal, 0, POST_UPDATE);
    }
}

__aicore__ inline void ReduceMaxIntrinsicsImpl(__ubuf__ half* workLocal, __ubuf__ half* srcLocal,
    const uint64_t mask[], const int32_t repeatTimes, const int32_t srcRepStride)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    *((__ubuf__ uint64_t*)tempBuf) = mask[0];
    *((__ubuf__ uint64_t*)tempBuf + 1) = mask[1];

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_align ureg;
        vector_bool preg;
        plds(preg, ((__ubuf__ uint32_t*)tempBuf), 0, US);
        uint32_t strideConfig0 = (((uint32_t)1) << 16);
        uint32_t strideConfig1 = (((uint32_t)0) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, srcLocal + i * srcRepStride * 16, strideConfig0, preg);
            vcmax(vreg1, vreg0, preg, MODE_ZEROING);
            vstus(ureg, 2, vreg1, workLocal, POST_UPDATE);
        }
        vstas(ureg, workLocal, 0, POST_UPDATE);
    }
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

__aicore__ inline void ReduceMaxIntrinsicsImpl(__ubuf__ float* workLocal, __ubuf__ float* srcLocal, const int32_t mask,
    const int32_t repeatTimes, const int32_t srcRepStride)
{
    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        vector_align ureg;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b32(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)1) << 16);
        uint32_t strideConfig1 = (((uint32_t)0) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, srcLocal + i * srcRepStride * 8, strideConfig0, preg);
            vcmax(vreg1, vreg0, preg, MODE_ZEROING);
            vstus(ureg, 2, vreg1, workLocal, POST_UPDATE);
        }
        vstas(ureg, workLocal, 0, POST_UPDATE);
    }
}

__aicore__ inline void ReduceMaxIntrinsicsImpl(__ubuf__ float* workLocal, __ubuf__ float* srcLocal,
    const uint64_t mask[], const int32_t repeatTimes, const int32_t srcRepStride)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    *((__ubuf__ uint64_t*)tempBuf) = mask[0];
    *((__ubuf__ uint64_t*)tempBuf + 1) = mask[1];

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        vector_align ureg;
        vector_bool preg;
        vector_bool preg1;
        plds(preg, ((__ubuf__ uint32_t*)tempBuf), 0, US);
        punpack(preg1, preg, LOWER);
        uint32_t strideConfig0 = (((uint32_t)1) << 16);
        uint32_t strideConfig1 = (((uint32_t)0) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, srcLocal + i * srcRepStride * 8, strideConfig0, preg1);
            vcmax(vreg1, vreg0, preg1, MODE_ZEROING);
            vstus(ureg, 2, vreg1, workLocal, POST_UPDATE);
        }
        vstas(ureg, workLocal, 0, POST_UPDATE);
    }
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

template <typename T>
__aicore__ inline void ReduceMinIntrinsicsImpl(__ubuf__ T* workLocal, __ubuf__ T* srcLocal, const int32_t mask,
    const int32_t repeatTimes, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMinIntrinsicsImpl is not supported!"); });
}

template <typename T>
__aicore__ inline void ReduceMinIntrinsicsImpl(__ubuf__ T* workLocal, __ubuf__ T* srcLocal, const uint64_t mask[],
    const int32_t repeatTimes, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMinIntrinsicsImpl is not supported!"); });
}

__aicore__ inline void ReduceMinIntrinsicsImpl(__ubuf__ half* workLocal, __ubuf__ half* srcLocal, const int32_t mask,
    const int32_t repeatTimes, const int32_t srcRepStride)
{
    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_align ureg;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b16(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)1) << 16);
        uint32_t strideConfig1 = (((uint32_t)0) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, srcLocal + i * srcRepStride * 16, strideConfig0, preg);
            vcmin(vreg1, vreg0, preg, MODE_ZEROING);
            vstus(ureg, 2, vreg1, workLocal, POST_UPDATE);
        }
        vstas(ureg, workLocal, 0, POST_UPDATE);
    }
}

__aicore__ inline void ReduceMinIntrinsicsImpl(__ubuf__ half* workLocal, __ubuf__ half* srcLocal,
    const uint64_t mask[], const int32_t repeatTimes, const int32_t srcRepStride)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    *((__ubuf__ uint64_t*)tempBuf) = mask[0];
    *((__ubuf__ uint64_t*)tempBuf + 1) = mask[1];

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_align ureg;
        vector_bool preg;
        plds(preg, ((__ubuf__ uint32_t*)tempBuf), 0, US);
        uint32_t strideConfig0 = (((uint32_t)1) << 16);
        uint32_t strideConfig1 = (((uint32_t)0) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, srcLocal + i * srcRepStride * 16, strideConfig0, preg);
            vcmin(vreg1, vreg0, preg, MODE_ZEROING);
            vstus(ureg, 2, vreg1, workLocal, POST_UPDATE);
        }
        vstas(ureg, workLocal, 0, POST_UPDATE);
    }
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

__aicore__ inline void ReduceMinIntrinsicsImpl(__ubuf__ float* workLocal, __ubuf__ float* srcLocal, const int32_t mask,
    const int32_t repeatTimes, const int32_t srcRepStride)
{
    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        vector_align ureg;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b32(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)1) << 16);
        uint32_t strideConfig1 = (((uint32_t)0) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, srcLocal + i * srcRepStride * 8, strideConfig0, preg);
            vcmin(vreg1, vreg0, preg, MODE_ZEROING);
            vstus(ureg, 2, vreg1, workLocal, POST_UPDATE);
        }
        vstas(ureg, workLocal, 0, POST_UPDATE);
    }
}

__aicore__ inline void ReduceMinIntrinsicsImpl(__ubuf__ float* workLocal, __ubuf__ float* srcLocal,
    const uint64_t mask[], const int32_t repeatTimes, const int32_t srcRepStride)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    *((__ubuf__ uint64_t*)tempBuf) = mask[0];
    *((__ubuf__ uint64_t*)tempBuf + 1) = mask[1];

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        vector_align ureg;
        vector_bool preg;
        vector_bool preg1;
        plds(preg, ((__ubuf__ uint32_t*)tempBuf), 0, US);
        punpack(preg1, preg, LOWER);
        uint32_t strideConfig0 = (((uint32_t)1) << 16);
        uint32_t strideConfig1 = (((uint32_t)0) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, srcLocal + i * srcRepStride * 8, strideConfig0, preg1);
            vcmin(vreg1, vreg0, preg1, MODE_ZEROING);
            vstus(ureg, 2, vreg1, workLocal, POST_UPDATE);
        }
        vstas(ureg, workLocal, 0, POST_UPDATE);
    }
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

template <typename T>
__aicore__ inline void ReduceSumIntrinsicsImpl(__ubuf__ T* workLocal, __ubuf__ T* srcLocal, const int32_t mask,
    const int32_t repeatTimes, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceSumIntrinsicsImpl is not supported!"); });
}

template <typename T>
__aicore__ inline void ReduceSumIntrinsicsImpl(__ubuf__ T* workLocal, __ubuf__ T* srcLocal, const uint64_t mask[],
    const int32_t repeatTimes, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceSumIntrinsicsImpl is not supported!"); });
}

__aicore__ inline void ReduceSumIntrinsicsImpl(__ubuf__ half* workLocal, __ubuf__ half* srcLocal, const int32_t mask,
    const int32_t repeatTimes, const int32_t srcRepStride)
{
    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_align ureg;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b16(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)1) << 16);
        uint32_t strideConfig1 = (((uint32_t)0) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, srcLocal + i * srcRepStride * 16, strideConfig0, preg);
            vcadd(vreg1, vreg0, preg, MODE_ZEROING);
            vstus(ureg, 1, vreg1, workLocal, POST_UPDATE);
        }
        vstas(ureg, workLocal, 0, POST_UPDATE);
    }
}

__aicore__ inline void ReduceSumIntrinsicsImpl(__ubuf__ half* workLocal, __ubuf__ half* srcLocal,
    const uint64_t mask[], const int32_t repeatTimes, const int32_t srcRepStride)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    *((__ubuf__ uint64_t*)tempBuf) = mask[0];
    *((__ubuf__ uint64_t*)tempBuf + 1) = mask[1];

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    __VEC_SCOPE__
    {
        vector_f16 vreg0;
        vector_f16 vreg1;
        vector_align ureg;
        vector_bool preg;
        plds(preg, ((__ubuf__ uint32_t*)tempBuf), 0, US);
        uint32_t strideConfig0 = (((uint32_t)1) << 16);
        uint32_t strideConfig1 = (((uint32_t)0) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, srcLocal + i * srcRepStride * 16, strideConfig0, preg);
            vcadd(vreg1, vreg0, preg, MODE_ZEROING);
            vstus(ureg, 1, vreg1, workLocal, POST_UPDATE);
        }
        vstas(ureg, workLocal, 0, POST_UPDATE);
    }
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

__aicore__ inline void ReduceSumIntrinsicsImpl(__ubuf__ float* workLocal, __ubuf__ float* srcLocal, const int32_t mask,
    const int32_t repeatTimes, const int32_t srcRepStride)
{
    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        vector_align ureg;
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b32(sreg, POST_UPDATE);
        uint32_t strideConfig0 = (((uint32_t)1) << 16);
        uint32_t strideConfig1 = (((uint32_t)0) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, srcLocal + i * srcRepStride * 8, strideConfig0, preg);
            vcadd(vreg1, vreg0, preg, MODE_ZEROING);
            vstus(ureg, 1, vreg1, workLocal, POST_UPDATE);
        }
        vstas(ureg, workLocal, 0, POST_UPDATE);
    }
}

__aicore__ inline void ReduceSumIntrinsicsImpl(__ubuf__ float* workLocal, __ubuf__ float* srcLocal,
    const uint64_t mask[], const int32_t repeatTimes, const int32_t srcRepStride)
{
    __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);
    *((__ubuf__ uint64_t*)tempBuf) = mask[0];
    *((__ubuf__ uint64_t*)tempBuf + 1) = mask[1];

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    __VEC_SCOPE__
    {
        vector_f32 vreg0;
        vector_f32 vreg1;
        vector_align ureg;
        vector_bool preg;
        vector_bool preg1;
        plds(preg, ((__ubuf__ uint32_t*)tempBuf), 0, US);
        punpack(preg1, preg, LOWER);
        uint32_t strideConfig0 = (((uint32_t)1) << 16);
        uint32_t strideConfig1 = (((uint32_t)0) << 16);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vsldb(vreg0, srcLocal + i * srcRepStride * 8, strideConfig0, preg1);
            vcadd(vreg1, vreg0, preg1, MODE_ZEROING);
            vstus(ureg, 1, vreg1, workLocal, POST_UPDATE);
        }
        vstas(ureg, workLocal, 0, POST_UPDATE);
    }
    AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);
}

template <typename T>
__aicore__ inline void ReduceSumSecondStep(__ubuf__ T* dstLocal, __ubuf__ T* workLocal,
    struct ReduceRepeatParams& params)
{
    int32_t dstOffset = 0;
    int32_t srcOffset = 0;
    int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T); // fp16=128 , fp32=64
    int32_t newRepeatTimes = params.repeatTimes / elementNumPerRep;
    int32_t leftData = params.repeatTimes % elementNumPerRep;

    if (newRepeatTimes != 0) {
        ReduceSumIntrinsicsImpl(workLocal, workLocal, elementNumPerRep, newRepeatTimes, DEFAULT_REPEAT_STRIDE);
    }

    if (leftData > 0) { // has_tail
        srcOffset = elementNumPerRep * newRepeatTimes;
        ReduceSumIntrinsicsImpl(dstLocal, workLocal + srcOffset, leftData, 1, DEFAULT_REPEAT_STRIDE);
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        *(workLocal + newRepeatTimes) = *dstLocal;
        if (newRepeatTimes != 0) {
            event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventIdSToV);
            WaitFlag<HardEvent::S_V>(eventIdSToV);
        }
    }
}

template <typename T>
__aicore__ inline void CreateSpecialFormatMask(const int32_t& maskLen, uint64_t& highMask, uint64_t& lowMask)
{
    // create mask in the "0101010101" format
    int32_t halfLen = HLAF_MASK_LEN / 2;
    for (int32_t i = 0; i < maskLen - halfLen; i++) {
        highMask = highMask << 2;
        highMask = highMask | 1;
    }
    int32_t lowMaskRange = maskLen >= halfLen ? halfLen : maskLen;
    for (int32_t i = 0; i < lowMaskRange; i++) {
        lowMask = lowMask << 2;
        lowMask = lowMask | 1;
    }
}

template <typename T>
__aicore__ inline void ReduceOperation(__ubuf__ T* workLocal, __ubuf__ T* srcLocal, struct ReduceRepeatParams& params,
    const ReduceMode& mode)
{
    if (params.maskMode == 1) {
        switch (mode) {
            case ReduceMode::REDUCE_MAX:
                ReduceMaxIntrinsicsImpl(workLocal, srcLocal, params.normalMask, params.repeatTimes,
                    params.srcRepStride);
                break;
            case ReduceMode::REDUCE_MIN:
                ReduceMinIntrinsicsImpl(workLocal, srcLocal, params.normalMask, params.repeatTimes,
                    params.srcRepStride);
                break;
            case ReduceMode::REDUCE_SUM:
                ReduceSumIntrinsicsImpl(workLocal, srcLocal, params.normalMask, params.repeatTimes,
                    params.srcRepStride);
                break;
            default:
                break;
        }
    } else {
        switch (mode) {
            case ReduceMode::REDUCE_MAX:
                ReduceMaxIntrinsicsImpl(workLocal, srcLocal, params.bitMask, params.repeatTimes, params.srcRepStride);
                break;
            case ReduceMode::REDUCE_MIN:
                ReduceMinIntrinsicsImpl(workLocal, srcLocal, params.bitMask, params.repeatTimes, params.srcRepStride);
                break;
            case ReduceMode::REDUCE_SUM:
                ReduceSumIntrinsicsImpl(workLocal, srcLocal, params.bitMask, params.repeatTimes, params.srcRepStride);
                break;
            default:
                break;
        }
    }
}

template <typename T>
__aicore__ inline void ReduceImplFirstStep(__ubuf__ T* workLocal, __ubuf__ T* srcLocal,
    struct ReduceRepeatParams& params, const ReduceMode& mode, int32_t& curData)
{
    int32_t dstOffset = 0;
    int32_t srcOffset = 0;
    int32_t range = params.repeatTimes / MAX_REPEAT_TIMES;

    for (int32_t index = 0; index < range; index++) {
        dstOffset = index * MAX_REPEAT_TIMES * VREDUCE_PER_REP_OUTPUT;
        srcOffset = index * MAX_REPEAT_TIMES * params.srcRepStride * ONE_BLK_SIZE / sizeof(T);
        struct ReduceRepeatParams newParams = params;
        newParams.repeatTimes = MAX_REPEAT_TIMES;
        ReduceOperation<T>(workLocal + dstOffset, srcLocal + srcOffset, newParams, mode);
    }
    int32_t leftRepeatTimes = params.repeatTimes % MAX_REPEAT_TIMES;
    if (leftRepeatTimes > 0) {
        dstOffset = range * MAX_REPEAT_TIMES * VREDUCE_PER_REP_OUTPUT;
        srcOffset = range * MAX_REPEAT_TIMES * params.srcRepStride * ONE_BLK_SIZE / sizeof(T);
        struct ReduceRepeatParams leftParams = params;
        leftParams.repeatTimes = leftRepeatTimes;
        ReduceOperation<T>(workLocal + dstOffset, srcLocal + srcOffset, leftParams, mode);
    }
    curData = VREDUCE_PER_REP_OUTPUT * params.repeatTimes;
}

template <typename T>
__aicore__ inline void ReduceImplSecondStep(__ubuf__ T* workLocal, const ReduceMode& mode, int32_t& curData,
    int32_t preStartPos, int32_t secondStartPos)
{
    int32_t dstOffset = 0;
    int32_t srcOffset = 0;
    int32_t newMaskLen = 0;
    int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T);
    int32_t newRepeatTimes = curData / elementNumPerRep;
    int32_t leftData = curData % elementNumPerRep;
    uint64_t highMask = 0, lowMask = 0;
    uint64_t newMask[2];
    int32_t bodyOutputCount = 0;
    int32_t tailOutputCount = 0;

    if (newRepeatTimes >= 1) {
        highMask = (sizeof(T) == sizeof(half)) ? 0x5555555555555555 : 0;
        lowMask = 0x5555555555555555;
        newMask[0] = lowMask;
        newMask[1] = highMask;
        struct ReduceRepeatParams newParams(newMask, newRepeatTimes, DEFAULT_REDUCE_DST_REP_SRIDE, DEFAULT_BLK_STRIDE,
            DEFAULT_REPEAT_STRIDE);

        ReduceOperation<T>(workLocal + secondStartPos, workLocal + preStartPos, newParams, mode);
        bodyOutputCount = newRepeatTimes * VREDUCE_PER_REP_OUTPUT;
    }
    highMask = 0;
    lowMask = 0;

    if (leftData > 0) {
        newMaskLen = leftData / VREDUCE_PER_REP_OUTPUT;
        // create mask in the "0101010101" format
        CreateSpecialFormatMask<T>(newMaskLen, highMask, lowMask);
        newMask[0] = lowMask;
        newMask[1] = highMask;
        struct ReduceRepeatParams leftParams(newMask, 1, DEFAULT_REDUCE_DST_REP_SRIDE, DEFAULT_BLK_STRIDE,
            DEFAULT_REPEAT_STRIDE);

        dstOffset = secondStartPos + bodyOutputCount;
        srcOffset = preStartPos + newRepeatTimes * elementNumPerRep;
        ReduceOperation<T>(workLocal + dstOffset, workLocal + srcOffset, leftParams, mode);
        tailOutputCount = VREDUCE_PER_REP_OUTPUT;
    }

    curData = bodyOutputCount + tailOutputCount;
}

template <typename T>
__aicore__ inline void GetIndex(__ubuf__ T* workLocal, int32_t secondStartPos, int32_t& secondIndex,
    int32_t& thirdIndex)
{
    int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T);
    if (sizeof(T) == sizeof(half)) {
        thirdIndex = *reinterpret_cast<__ubuf__ uint16_t*>(workLocal + secondStartPos + 1);
        ASSERT(thirdIndex >= 0);
        ASSERT(thirdIndex < elementNumPerRep);
        secondIndex = *reinterpret_cast<__ubuf__ uint16_t*>(workLocal + thirdIndex + 1);
        ASSERT(secondIndex >= 0);
        ASSERT(secondIndex < elementNumPerRep);
    } else {
        thirdIndex = *reinterpret_cast<__ubuf__ uint32_t*>(workLocal + secondStartPos + 1);
        ASSERT(thirdIndex >= 0);
        ASSERT(thirdIndex < elementNumPerRep);
        secondIndex = *reinterpret_cast<__ubuf__ uint32_t*>(workLocal + thirdIndex + 1);
        ASSERT(secondIndex >= 0);
        ASSERT(secondIndex < elementNumPerRep);
    }
}

template <typename T>
__aicore__ inline void GetIndex(__ubuf__ T* workLocal, int32_t secondStartPos, int32_t thirdStartPos,
    int32_t& firstIndex, int32_t& secondIndex, int32_t& thirdIndex)
{
    int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T);
    if (sizeof(T) == sizeof(half)) {
        thirdIndex = *reinterpret_cast<__ubuf__ uint16_t*>(workLocal + thirdStartPos + 1);
        ASSERT(thirdIndex >= 0);
        ASSERT(thirdIndex < elementNumPerRep);
        secondIndex = *reinterpret_cast<__ubuf__ uint16_t*>(workLocal + secondStartPos + thirdIndex + 1);
        ASSERT(secondIndex >= 0);
        ASSERT(secondIndex < elementNumPerRep);
        firstIndex = *reinterpret_cast<__ubuf__ uint16_t*>(workLocal +
            elementNumPerRep * (thirdIndex / VREDUCE_PER_REP_OUTPUT) + secondIndex + 1);
        ASSERT(firstIndex >= 0);
        ASSERT(firstIndex < elementNumPerRep);
    } else {
        thirdIndex = *reinterpret_cast<__ubuf__ uint32_t*>(workLocal + thirdStartPos + 1);
        ASSERT(thirdIndex >= 0);
        ASSERT(thirdIndex < elementNumPerRep);
        secondIndex = *reinterpret_cast<__ubuf__ uint32_t*>(workLocal + secondStartPos + thirdIndex + 1);
        ASSERT(secondIndex >= 0);
        ASSERT(secondIndex < elementNumPerRep);
        firstIndex = *reinterpret_cast<__ubuf__ uint32_t*>(workLocal +
            elementNumPerRep * (thirdIndex / VREDUCE_PER_REP_OUTPUT) + secondIndex + 1);
        ASSERT(firstIndex >= 0);
        ASSERT(firstIndex < elementNumPerRep);
    }
}

template <typename T>
__aicore__ inline void GetIndex(__ubuf__ T* workLocal, int32_t secondStartPos, int32_t thirdStartPos,
    int32_t fourthStartPos, int32_t& firstIndex, int32_t& secondIndex, int32_t& thirdIndex, int32_t& fourthIndex)
{
    int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T);
    if (sizeof(T) == sizeof(half)) {
        fourthIndex = *reinterpret_cast<__ubuf__ uint16_t*>(workLocal + fourthStartPos + 1);
        ASSERT(fourthIndex >= 0);
        ASSERT(fourthIndex < elementNumPerRep);
        thirdIndex = *reinterpret_cast<__ubuf__ uint16_t*>(workLocal + thirdStartPos + fourthIndex + 1);
        ASSERT(thirdIndex >= 0);
        ASSERT(thirdIndex < elementNumPerRep);
        secondIndex = *reinterpret_cast<__ubuf__ uint16_t*>(workLocal + secondStartPos +
            elementNumPerRep * (fourthIndex / VREDUCE_PER_REP_OUTPUT) + thirdIndex + 1);
        ASSERT(secondIndex >= 0);
        ASSERT(secondIndex < elementNumPerRep);
        firstIndex = *reinterpret_cast<__ubuf__ uint16_t*>(workLocal +
            elementNumPerRep * (elementNumPerRep * (fourthIndex / VREDUCE_PER_REP_OUTPUT) + thirdIndex) /
            VREDUCE_PER_REP_OUTPUT +
            secondIndex + 1);
        ASSERT(firstIndex >= 0);
        ASSERT(firstIndex < elementNumPerRep);
    } else {
        fourthIndex = *reinterpret_cast<__ubuf__ uint32_t*>(workLocal + fourthStartPos + 1);
        ASSERT(fourthIndex >= 0);
        ASSERT(fourthIndex < elementNumPerRep);
        thirdIndex = *reinterpret_cast<__ubuf__ uint32_t*>(workLocal + thirdStartPos + fourthIndex + 1);
        ASSERT(thirdIndex >= 0);
        ASSERT(thirdIndex < elementNumPerRep);
        secondIndex = *reinterpret_cast<__ubuf__ uint32_t*>(workLocal + secondStartPos +
            elementNumPerRep * (fourthIndex / VREDUCE_PER_REP_OUTPUT) + thirdIndex + 1);
        ASSERT(secondIndex >= 0);
        ASSERT(secondIndex < elementNumPerRep);
        firstIndex = *reinterpret_cast<__ubuf__ uint32_t*>(workLocal +
            elementNumPerRep * (elementNumPerRep * (fourthIndex / VREDUCE_PER_REP_OUTPUT) + thirdIndex) /
            VREDUCE_PER_REP_OUTPUT +
            secondIndex + 1);
        ASSERT(firstIndex >= 0);
        ASSERT(firstIndex < elementNumPerRep);
    }
}

template <typename T>
__aicore__ inline void ReduceImplThirdStep(__ubuf__ T* dstLocal, __ubuf__ T* workLocal, const int32_t srcRepStride,
    const ReduceMode& mode, int32_t& curData, int32_t& secondStartPos, int32_t& thirdStartPos)
{
    int32_t preNum = 0;
    int32_t firstIndex = 0;
    int32_t secondIndex = 0;
    int32_t thirdIndex = 0;
    int32_t fourthIndex = 0;
    int32_t dstOffset = 0;
    int32_t srcOffset = 0;
    uint64_t highMask = 0;
    uint64_t lowMask = 0;
    uint64_t newMask[2];
    int32_t offsetNumPerRep = ONE_BLK_SIZE / sizeof(T) * srcRepStride;
    int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T);
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    event_t eventIdSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    if (curData == VREDUCE_PER_REP_OUTPUT) {
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        GetIndex<T>(workLocal, secondStartPos, secondIndex, thirdIndex);
        preNum = offsetNumPerRep * (thirdIndex / VREDUCE_PER_REP_OUTPUT);
        int32_t redultIndex = secondIndex + preNum;
        *dstLocal = *(workLocal + secondStartPos);
        *(dstLocal + 1) = *reinterpret_cast<T*>(&redultIndex);
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        SetFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
        return;
    }

    int32_t newMaskLen = curData / VREDUCE_PER_REP_OUTPUT;
    CreateSpecialFormatMask<T>(newMaskLen, highMask, lowMask);
    newMask[0] = lowMask;
    newMask[1] = highMask;
    if (curData > elementNumPerRep) {
        ReduceImplSecondStep<T>(workLocal, mode, curData, secondStartPos, thirdStartPos);

        int32_t fourthStartPos =
            (((thirdStartPos + curData) * sizeof(T) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE) * ONE_BLK_SIZE / sizeof(T);
        dstOffset = fourthStartPos;
        srcOffset = thirdStartPos;
        struct ReduceRepeatParams newParams(newMask, 1, DEFAULT_REDUCE_DST_REP_SRIDE, DEFAULT_BLK_STRIDE,
            DEFAULT_REPEAT_STRIDE);

        ReduceOperation<T>(workLocal + dstOffset, workLocal + srcOffset, newParams, mode);
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        *dstLocal = *(workLocal + dstOffset);

        GetIndex<T>(workLocal, secondStartPos, thirdStartPos, fourthStartPos, firstIndex, secondIndex, thirdIndex,
            fourthIndex);
        preNum = offsetNumPerRep *
            (elementNumPerRep * (elementNumPerRep * (fourthIndex / VREDUCE_PER_REP_OUTPUT) + thirdIndex) /
            VREDUCE_PER_REP_OUTPUT +
            secondIndex) /
            VREDUCE_PER_REP_OUTPUT;
    } else {
        dstOffset = thirdStartPos;
        srcOffset = secondStartPos;
        struct ReduceRepeatParams newParams(newMask, 1, DEFAULT_REDUCE_DST_REP_SRIDE, DEFAULT_BLK_STRIDE,
            DEFAULT_REPEAT_STRIDE);
        ReduceOperation<T>(workLocal + dstOffset, workLocal + srcOffset, newParams, mode);
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        *dstLocal = *(workLocal + thirdStartPos);

        GetIndex<T>(workLocal, secondStartPos, thirdStartPos, firstIndex, secondIndex, thirdIndex);
        preNum = offsetNumPerRep * (elementNumPerRep * (thirdIndex / VREDUCE_PER_REP_OUTPUT) + secondIndex) /
            VREDUCE_PER_REP_OUTPUT;
    }

    int32_t redultIndex = firstIndex + preNum;
    *(dstLocal + 1) = *reinterpret_cast<T*>(&redultIndex);
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    SetFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
}

template <typename T>
__aicore__ inline void ReduceSumFirstStep(__ubuf__ T* workLocal, __ubuf__ T* srcLocal,
    struct ReduceRepeatParams& params)
{
    int32_t dstOffset = 0;
    int32_t srcOffset = 0;
    int32_t maxRepeatTimes = (MAX_REPEAT_TIMES - ONE_BLK_SIZE / sizeof(T) + 1);
    int32_t range = params.repeatTimes / maxRepeatTimes;

    for (int32_t index = 0; index < range; index++) {
        dstOffset = index * maxRepeatTimes;
        srcOffset = index * maxRepeatTimes * (params.srcRepStride * ONE_BLK_SIZE / sizeof(T));
        struct ReduceRepeatParams newParams = params;
        newParams.repeatTimes = maxRepeatTimes;
        ReduceOperation<T>(workLocal + dstOffset, srcLocal + srcOffset, newParams, ReduceMode::REDUCE_SUM);
    }

    int32_t leftRepeatTimes = params.repeatTimes % maxRepeatTimes;
    if (leftRepeatTimes > 0) {
        dstOffset = range * maxRepeatTimes;
        srcOffset = range * maxRepeatTimes * (params.srcRepStride * ONE_BLK_SIZE / sizeof(T));
        struct ReduceRepeatParams leftParams = params;
        leftParams.repeatTimes = leftRepeatTimes;
        ReduceOperation<T>(workLocal + dstOffset, srcLocal + srcOffset, leftParams, ReduceMode::REDUCE_SUM);
    }
}

template <typename T>
__aicore__ inline void ReduceSumFinalStep(__ubuf__ T* dstLocal, __ubuf__ T* workLocal, int32_t& secondResultNum)
{
    if (secondResultNum == 1) {
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        *(dstLocal) = *(workLocal);
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        event_t eventIdSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
    } else {
        struct ReduceRepeatParams newParams(secondResultNum, 1, DEFAULT_REDUCE_DST_REP_SRIDE, DEFAULT_BLK_STRIDE,
            DEFAULT_REPEAT_STRIDE);
        ReduceOperation<T>(dstLocal, workLocal, newParams, ReduceMode::REDUCE_SUM);
    }
}

template <typename T>
__aicore__ inline void ReduceSumImpl(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal, __ubuf__ T* workLocal,
    struct ReduceRepeatParams& params)
{
    ReduceSumFirstStep<T>(workLocal, srcLocal, params);
    ReduceSumSecondStep<T>(dstLocal, workLocal, params);
    int32_t secondResultNum = DivCeil(params.repeatTimes, ONE_REPEAT_BYTE_SIZE / sizeof(T));
    ReduceSumFinalStep<T>(dstLocal, workLocal, secondResultNum);
}

template <typename T>
__aicore__ inline void ReduceImplSecondStepNoIndex(__ubuf__ T* workLocal, const ReduceMode& mode, int32_t& curData)
{
    int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T); // fp16=128,fp32=64
    int32_t newRepeatTimes = curData / elementNumPerRep;
    int32_t leftData = curData % elementNumPerRep;
    uint64_t highMask = 0;
    uint64_t lowMask = 0;
    uint64_t newMask[2];
    if (newRepeatTimes != 0) {
        CreateSpecialFormatMask<T>(elementNumPerRep / VREDUCE_PER_REP_OUTPUT, highMask, lowMask);
        newMask[0] = lowMask;
        newMask[1] = highMask;
        struct ReduceRepeatParams newParams(newMask, newRepeatTimes, DEFAULT_REDUCE_DST_REP_SRIDE, DEFAULT_BLK_STRIDE,
            DEFAULT_REPEAT_STRIDE);
        ReduceOperation<T>(workLocal, workLocal, newParams, mode);
    }
    highMask = 0;
    lowMask = 0;
    if (leftData > 0) {
        CreateSpecialFormatMask<T>(leftData / VREDUCE_PER_REP_OUTPUT, highMask, lowMask);
        newMask[0] = lowMask;
        newMask[1] = highMask;
        struct ReduceRepeatParams leftParams(newMask, 1, DEFAULT_REDUCE_DST_REP_SRIDE, DEFAULT_BLK_STRIDE,
            DEFAULT_REPEAT_STRIDE);
        ReduceOperation<T>(workLocal + newRepeatTimes * VREDUCE_PER_REP_OUTPUT,
            workLocal + newRepeatTimes * elementNumPerRep, leftParams, mode);
        newRepeatTimes += 1;
    }
    curData = newRepeatTimes * VREDUCE_PER_REP_OUTPUT;
}

template <typename T>
__aicore__ inline void ReduceImplThirdStepNoIndex(__ubuf__ T* dstLocal, __ubuf__ T* workLocal, const ReduceMode& mode,
    int32_t& curData)
{
    uint64_t highMask = 0;
    uint64_t lowMask = 0;
    uint64_t newMask[2];
    CreateSpecialFormatMask<T>(curData / VREDUCE_PER_REP_OUTPUT, highMask, lowMask);
    newMask[0] = lowMask;
    newMask[1] = highMask;
    struct ReduceRepeatParams newParams(newMask, 1, DEFAULT_REDUCE_DST_REP_SRIDE, DEFAULT_BLK_STRIDE,
        DEFAULT_REPEAT_STRIDE);
    ReduceOperation<T>(workLocal, workLocal, newParams, mode);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    *dstLocal = *workLocal;
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    event_t eventIdSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
}

template <typename T>
__aicore__ inline void ReduceImplWithIndex(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal, __ubuf__ T* workLocal,
    struct ReduceRepeatParams& params, const ReduceMode& mode)
{
    if (params.repeatTimes == 1) {
        ReduceOperation<T>(dstLocal, srcLocal, params, mode);
    } else {
        int32_t curData = 0;
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        ReduceImplFirstStep<T>(workLocal, srcLocal, params, mode, curData);
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);

        int32_t secondStartPos = ((curData * sizeof(T) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE) * ONE_BLK_SIZE / sizeof(T);
        ReduceImplSecondStep<T>(workLocal, mode, curData, 0, secondStartPos);
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);

        int32_t thirdStartPos =
            (((secondStartPos + curData) * sizeof(T) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE) * ONE_BLK_SIZE / sizeof(T);
        ReduceImplThirdStep<T>(dstLocal, workLocal, params.srcRepStride, mode, curData, secondStartPos, thirdStartPos);
    }
}

template <typename T>
__aicore__ inline void ReduceImplNoIndex(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal, __ubuf__ T* workLocal,
    struct ReduceRepeatParams& params, const ReduceMode& mode)
{
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    if (params.repeatTimes == 1) {
        ReduceOperation<T>(workLocal, srcLocal, params, mode);
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        *dstLocal = *workLocal;
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        event_t eventIdSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
    } else {
        if (mode == ReduceMode::REDUCE_SUM) {
            ReduceSumImpl<T>(dstLocal, srcLocal, workLocal, params);
        } else {
            int32_t curData = 0;
            ReduceImplFirstStep<T>(workLocal, srcLocal, params, mode, curData);
            SetFlag<HardEvent::V_S>(eventIdVToS);
            WaitFlag<HardEvent::V_S>(eventIdVToS);

            ReduceImplSecondStepNoIndex<T>(workLocal, mode, curData);

            int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T); // fp16=128,fp32=64
            if (curData <= elementNumPerRep) {
                SetFlag<HardEvent::V_S>(eventIdVToS);
                WaitFlag<HardEvent::V_S>(eventIdVToS);
                ReduceImplThirdStepNoIndex<T>(dstLocal, workLocal, mode, curData);
                return;
            }
            SetFlag<HardEvent::V_S>(eventIdVToS);
            WaitFlag<HardEvent::V_S>(eventIdVToS);
            ReduceImplSecondStepNoIndex<T>(workLocal, mode, curData);
            if (curData <= elementNumPerRep) {
                SetFlag<HardEvent::V_S>(eventIdVToS);
                WaitFlag<HardEvent::V_S>(eventIdVToS);
                ReduceImplThirdStepNoIndex<T>(dstLocal, workLocal, mode, curData);
            }
        }
    }
}

template <typename T>
__aicore__ inline void ReduceImpl(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal, __ubuf__ T* workLocal,
    struct ReduceRepeatParams& params, bool calIndex, const ReduceMode& mode)
{
    if (calIndex) {
        ReduceImplWithIndex<T>(dstLocal, srcLocal, workLocal, params, mode);
    } else {
        ReduceImplNoIndex<T>(dstLocal, srcLocal, workLocal, params, mode);
    }
}

template <typename T>
__aicore__ inline void ReduceTailCompute(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<T>& workLocal, const int32_t count, bool calIndex, const ReduceMode& mode)
{
    int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T); // fp16=128 , fp32=64
    int32_t repeatTimes = count / elementNumPerRep;
    int32_t tailCount = count % elementNumPerRep; // tailCount  <= 128/64 repeat=1
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    T bodyValue = dstLocal.GetValue(0);
    T bodyIndex = dstLocal.GetValue(1);

    struct ReduceRepeatParams tailParams(tailCount, 1, DEFAULT_REDUCE_DST_REP_SRIDE, DEFAULT_BLK_STRIDE,
        DEFAULT_REPEAT_STRIDE);

    ReduceImpl<T>((__ubuf__ T*)dstLocal.GetPhyAddr(), // 复用dstLocal
        (__ubuf__ T*)srcLocal.GetPhyAddr(elementNumPerRep * repeatTimes), (__ubuf__ T*)workLocal.GetPhyAddr(),
        tailParams, calIndex, mode);
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    T tailValue = dstLocal.GetValue(0);
    T tailIndex = dstLocal.GetValue(1);

    // bodyresult tailresult need vcmin/vcmax again
    struct ReduceRepeatParams lastParams(2, 1, DEFAULT_REDUCE_DST_REP_SRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE);
    workLocal.SetValue(0, bodyValue);
    workLocal.SetValue(1, tailValue);
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    ReduceImpl<T>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)workLocal.GetPhyAddr(),
        (__ubuf__ T*)workLocal.GetPhyAddr(), lastParams, calIndex, mode);
    if (calIndex) {
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        T lastIndexVal = dstLocal.GetValue(1);
        uint32_t newIndex = 0;
        uint32_t lastIndex = 0;
        if constexpr (sizeof(T) == sizeof(half)) {
            lastIndex = *reinterpret_cast<uint16_t*>(&lastIndexVal);
            newIndex = elementNumPerRep * repeatTimes + *reinterpret_cast<uint16_t*>(&tailIndex);
        } else {
            lastIndex = *reinterpret_cast<uint32_t*>(&lastIndexVal);
            newIndex = elementNumPerRep * repeatTimes + *reinterpret_cast<uint32_t*>(&tailIndex);
        }
        if (lastIndex == 1) {
            dstLocal.SetValue(1, *reinterpret_cast<T*>(&newIndex));
        } else {
            dstLocal.SetValue(1, bodyIndex);
        }
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        event_t eventIdSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
    }
}

template <typename T>
__aicore__ inline void GetReduceMaxMinCountImpl(uint32_t &maxMinValue, uint32_t &maxMinIndex)
{
    ASCENDC_ASSERT((false), "GetReduceMaxMinCount is not supported on current device");
}

template <typename T>
__aicore__ inline void GetReduceMaxMinCountImpl(uint32_t &maxMinValue)
{
    ASCENDC_ASSERT((false), "GetReduceMaxMinCount is not supported on current device");
}

template <typename T>
__aicore__ inline void GetReduceMaxMinCountImpl(T &maxMinValue, T &maxMinIndex)
{
    ASCENDC_ASSERT((false), "GetReduceMaxMinCount is not supported on current device");
}

template <typename T>
__aicore__ inline void GetReduceMaxMinCountImpl(T &maxMinValue)
{
    ASCENDC_ASSERT((false), "GetReduceMaxMinCount is not supported on current device");
}

template <typename T>
__aicore__ inline T GetAccValImpl()
{
    ASCENDC_ASSERT((false), "GetAccVal is not supported on current device");
    return 0;
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_REDUCE_IMPL_H
