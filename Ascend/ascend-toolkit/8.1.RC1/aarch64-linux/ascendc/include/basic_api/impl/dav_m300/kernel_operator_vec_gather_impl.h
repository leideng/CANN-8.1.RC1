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
 * \file kernel_operator_vec_gather_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_GATHER_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_GATHER_IMPL_H
#include "kernel_struct_gather.h"

namespace AscendC {
/* **************************************************************************************************
 * Gather                                             *
 * ************************************************************************************************* */
constexpr int16_t B32_SHIFT_VAL_GATHER = 2;
constexpr int16_t B16_SHIFT_VAL_GATHER = 1;
constexpr uint32_t B32_BLK_ELE_GATHER = 8;
constexpr uint32_t B16_BLK_ELE_GATHER = 16;
constexpr uint32_t B8_BLK_ELE_GATHER = 32;
constexpr uint32_t SRC_REP64_GATHER = 64;
constexpr uint32_t SRC_REP128_GATHER = 128;
template <typename T>
__aicore__ inline void GatherImplB32(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ uint32_t *srcOffsetLocal,
                                      const uint32_t srcBaseOffset, const uint32_t count)
{
    __VEC_SCOPE__
    {
        vector_u32 indexReg;
        vector_f32 dstReg;
        uint32_t sregPlt = (uint32_t)count;
        vector_bool indexMask = pset_b32(PAT_ALL);
        vector_bool preg;
        uint16_t repeatTimes = CeilDivision(count, SRC_REP64_GATHER);

        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b32(sregPlt, POST_UPDATE);
            vlds(indexReg, srcOffsetLocal, i * SRC_REP64_GATHER, NORM);
            vshrs(indexReg, indexReg, B32_SHIFT_VAL_GATHER, indexMask, MODE_ZEROING);
            vgather2(dstReg, (__ubuf__ float *)(srcLocal + srcBaseOffset), indexReg, preg);
            vsts(dstReg, (__ubuf__ float*) dstLocal, i * SRC_REP64_GATHER, NORM_B32, preg);
        }
    }
}

template <typename T>
__aicore__ inline void GatherImplB32(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ uint32_t *srcOffsetLocal,
                                      const uint32_t srcLength, const uint32_t srcBaseOffset, const uint64_t mask,
                                      const uint8_t repeatTimes, const uint8_t srcRepStride)
{
    __VEC_SCOPE__
    {
        vector_u32 indexReg;
        vector_f32 dstReg;
        vector_bool indexMask = pset_b32(PAT_ALL);
        uint32_t maskV = (uint32_t)mask;
        vector_bool srcMask;
        srcMask = plt_b32(maskV, POST_UPDATE);

        for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
            vlds(indexReg, srcOffsetLocal, i * SRC_REP64_GATHER, NORM);
            vshrs(indexReg, indexReg, B32_SHIFT_VAL_GATHER, indexMask, MODE_ZEROING);
            vgather2(dstReg, (__ubuf__ float *)(srcLocal + srcBaseOffset), indexReg, srcMask);
            vsts(dstReg, (__ubuf__ float*) dstLocal, i * srcRepStride * B32_BLK_ELE_GATHER, NORM_B32, srcMask);
        }
    }
}

template <typename T>
__aicore__ inline void GatherImplB32(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ uint32_t *srcOffsetLocal,
                                      const uint32_t srcLength, const uint32_t srcBaseOffset, const uint64_t mask[],
                                      const uint8_t repeatTimes, const uint8_t srcRepStride)
{
    SetVectorMask<T>(mask[1], mask[0]);
    __VEC_SCOPE__
    {
        vector_u32 indexReg;
        vector_f32 dstReg;
        vector_bool indexMask = pset_b32(PAT_ALL);
        vector_bool dstPreg;
        dstPreg = movp_b32();

        for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
            vlds(indexReg, srcOffsetLocal, i * SRC_REP64_GATHER, NORM);
            vshrs(indexReg, indexReg, B32_SHIFT_VAL_GATHER, indexMask, MODE_ZEROING);
            vgather2(dstReg, (__ubuf__ float *)(srcLocal + srcBaseOffset), indexReg, dstPreg);
            vsts(dstReg, (__ubuf__ float*)dstLocal, i * srcRepStride * B32_BLK_ELE_GATHER, NORM_B32, dstPreg);
        }
    }
}

template <typename T>
__aicore__ inline void GatherImplB16(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ uint32_t *srcOffsetLocal,
                                      const uint32_t srcBaseOffset, const uint32_t count)
{
    __VEC_SCOPE__
    {
        vector_u32  indexReg;
        vector_u32  indexRegSec;
        vector_f16  dstReg;
        uint32_t sregPlt = (uint32_t)count;
        vector_bool indexMask = pset_b32(PAT_ALL);
        vector_bool preg;
        uint16_t repeatTimes = CeilDivision(count, SRC_REP128_GATHER);

        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b16(sregPlt, POST_UPDATE);
            vlds(indexReg, srcOffsetLocal, 2 * i * SRC_REP64_GATHER, NORM);
            vlds(indexRegSec, srcOffsetLocal, (2 * i + 1) * SRC_REP64_GATHER, NORM);
            vshrs(indexReg, indexReg, B16_SHIFT_VAL_GATHER, indexMask, MODE_ZEROING);
            vshrs(indexRegSec, indexRegSec, B16_SHIFT_VAL_GATHER, indexMask, MODE_ZEROING);
            vdintlv((vector_u16&)indexReg, (vector_u16&)indexRegSec, (vector_u16&)indexReg, (vector_u16&)indexRegSec);
            vgather2(dstReg, (__ubuf__ half *)(srcLocal + srcBaseOffset), (vector_u16&)indexReg, preg);
            vsts(dstReg, (__ubuf__ half *)dstLocal, i * SRC_REP128_GATHER, NORM_B16, preg);
        }
    }
}

template <typename T>
__aicore__ inline void GatherImplB16(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ uint32_t *srcOffsetLocal,
                                      const uint32_t srcLength, const uint32_t srcBaseOffset, const uint64_t mask,
                                      const uint8_t repeatTimes, const uint8_t srcRepStride)
{
    __VEC_SCOPE__
    {
        vector_u32  indexReg;
        vector_u32  indexRegSec;
        vector_u16  indexU16;
        vector_u16  lowerU16;
        vector_u16  highU16;
        vector_f16  dstReg;
        vector_bool indexMask = pset_b32(PAT_ALL);
        uint32_t sreg = (uint32_t)mask;
        vector_bool preg = plt_b16(sreg, POST_UPDATE);

        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vlds(indexReg, srcOffsetLocal, 2 * i * SRC_REP64_GATHER, NORM);
            vlds(indexRegSec, srcOffsetLocal, (2 * i + 1) * SRC_REP64_GATHER, NORM);
            vshrs(indexReg, indexReg, B16_SHIFT_VAL_GATHER, indexMask, MODE_ZEROING);
            vshrs(indexRegSec, indexRegSec, B16_SHIFT_VAL_GATHER, indexMask, MODE_ZEROING);
            vdintlv((vector_u16&)indexReg, (vector_u16&)indexRegSec, (vector_u16&)indexReg, (vector_u16&)indexRegSec);
            vgather2(dstReg, (__ubuf__ half *)(srcLocal + srcBaseOffset), (vector_u16&)indexReg, preg);
            vsts(dstReg, (__ubuf__ half *)dstLocal, i * srcRepStride * B16_BLK_ELE_GATHER, NORM_B16, preg);
        }
    }
}

template <typename T>
__aicore__ inline void GatherImplB16(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ uint32_t *srcOffsetLocal,
                                      const uint32_t srcLength, const uint32_t srcBaseOffset, const uint64_t mask[],
                                      const uint8_t repeatTimes, const uint8_t srcRepStride)
{
    SetVectorMask<T>(mask[1], mask[0]);
    __VEC_SCOPE__
    {
        vector_u32  indexReg;
        vector_u32  indexRegSec;
        vector_bool indexMask = pset_b32(PAT_ALL);
        vector_f16  dstReg;
        vector_bool dstPreg;
        dstPreg = movp_b16();

        for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
            vlds(indexReg, srcOffsetLocal, 2 * i * SRC_REP64_GATHER, NORM);
            vlds(indexRegSec, srcOffsetLocal, (2 * i + 1) * SRC_REP64_GATHER, NORM);
            vshrs(indexReg, indexReg, B16_SHIFT_VAL_GATHER, indexMask, MODE_ZEROING);
            vshrs(indexRegSec, indexRegSec, B16_SHIFT_VAL_GATHER, indexMask, MODE_ZEROING);
            vdintlv((vector_u16&)indexReg, (vector_u16&)indexRegSec, (vector_u16&)indexReg, (vector_u16&)indexRegSec);
            vgather2(dstReg, (__ubuf__ half *)(srcLocal + srcBaseOffset), (vector_u16&)indexReg, dstPreg);
            vsts(dstReg, (__ubuf__ half *)dstLocal, i * srcRepStride * B16_BLK_ELE_GATHER, NORM_B16, dstPreg);
        }
    }
}

template <typename T>
__aicore__ inline void GatherImplB8(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ uint32_t *srcOffsetLocal,
                                     const uint32_t srcBaseOffset, const uint32_t count)
{
    __VEC_SCOPE__
    {
        vector_u32  indexReg;
        vector_u32  indexRegSec;
        vector_s8  dstReg;
        uint32_t sregPlt = (uint32_t)count;
        vector_bool preg;
        uint16_t repeatTimes = CeilDivision(count, SRC_REP128_GATHER);

        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            preg = plt_b16(sregPlt, POST_UPDATE);
            vlds(indexReg, srcOffsetLocal, 2 * i * SRC_REP64_GATHER, NORM);
            vlds(indexRegSec, srcOffsetLocal, (2 * i + 1) * SRC_REP64_GATHER, NORM);
            vdintlv((vector_u16&)indexReg, (vector_u16&)indexRegSec, (vector_u16&)indexReg, (vector_u16&)indexRegSec);
            vgather2((vector_s16&)dstReg, (__ubuf__ int8_t*)(srcLocal + srcBaseOffset), (vector_u16&)indexReg, preg);
            vsts(dstReg, (__ubuf__ int8_t*)dstLocal, i * SRC_REP128_GATHER, PK_B16, preg);
        }
    }
}

template <typename T>
__aicore__ inline void GatherImplB8(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ uint32_t *srcOffsetLocal,
                                     const uint32_t srcLength, const uint32_t srcBaseOffset, const uint64_t mask,
                                     const uint8_t repeatTimes, const uint8_t srcRepStride)
{
    __VEC_SCOPE__
    {
        vector_u32  indexReg;
        vector_u32  indexRegSec;
        vector_u16  highU16;
        vector_s8  dstReg;
        vector_bool dstPreg;
        uint32_t srcMask = (uint32_t)mask;
        if (srcMask > SRC_REP128_GATHER) {
            srcMask = SRC_REP128_GATHER;
        }
        dstPreg = plt_b16(srcMask, POST_UPDATE);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vlds(indexReg, srcOffsetLocal, 2 * i * SRC_REP64_GATHER, NORM);
            vlds(indexRegSec, srcOffsetLocal, (2 * i + 1) * SRC_REP64_GATHER, NORM);
            vdintlv((vector_u16&)indexReg, (vector_u16&)indexRegSec, (vector_u16&)indexReg, (vector_u16&)indexRegSec);
            vgather2((vector_s16&)dstReg, (__ubuf__ int8_t*)srcLocal + srcBaseOffset, (vector_u16&)indexReg, dstPreg);
            vsts(dstReg, (__ubuf__ int8_t*)dstLocal, i * srcRepStride * B8_BLK_ELE_GATHER, PK_B16, dstPreg);
        }
    }
}

template <typename T>
__aicore__ inline void GatherImplB8(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ uint32_t *srcOffsetLocal,
                                     const uint32_t srcLength, const uint32_t srcBaseOffset, const uint64_t mask[],
                                     const uint8_t repeatTimes, const uint8_t srcRepStride)
{
    SetVectorMask<T>(mask[1], mask[0]);
    __VEC_SCOPE__
    {
        vector_u32  indexReg;
        vector_u32  indexRegSec;
        vector_s8  dstReg;
        vector_bool dstPreg;
        dstPreg = movp_b16();
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; ++i) {
            vlds(indexReg, srcOffsetLocal, 2 * i * SRC_REP64_GATHER, NORM);
            vlds(indexRegSec, srcOffsetLocal, (2 * i + 1) * SRC_REP64_GATHER, NORM);
            vdintlv((vector_u16&)indexReg, (vector_u16&)indexRegSec, (vector_u16&)indexReg, (vector_u16&)indexRegSec);
            vgather2((vector_s16&)dstReg, (__ubuf__ int8_t*)srcLocal + srcBaseOffset, (vector_u16&)indexReg, dstPreg);
            vsts(dstReg, (__ubuf__ int8_t*)dstLocal, i * srcRepStride * B8_BLK_ELE_GATHER, PK_B16, dstPreg);
        }
    }
}

// norm mode
template <typename T>
__aicore__ inline void GatherImpl(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal, __ubuf__ uint32_t* srcOffsetLocal,
    const uint32_t srcLength, const uint32_t srcBaseAddr, const uint64_t mask, const uint8_t repeatTimes,
    const uint8_t srcRepStride)
{
    static_assert(SupportBytes<T, 1, 2, 4>(), "Gather only support type b8/b16/b32 on current device");
    uint32_t srcBaseOffset = srcBaseAddr/sizeof(T);
    if constexpr (sizeof(T) == 1) {
        GatherImplB8(dstLocal, srcLocal, srcOffsetLocal, srcLength, srcBaseOffset, mask, repeatTimes, srcRepStride);
    } else if constexpr (sizeof(T) == 2) {
        GatherImplB16(dstLocal, srcLocal, srcOffsetLocal, srcLength, srcBaseOffset, mask, repeatTimes, srcRepStride);
    } else if constexpr (sizeof(T) == 4) {
        GatherImplB32(dstLocal, srcLocal, srcOffsetLocal, srcLength, srcBaseOffset, mask, repeatTimes, srcRepStride);
    }
}

// bit mode
template <typename T>
__aicore__ inline void GatherImpl(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal, __ubuf__ uint32_t* srcOffsetLocal,
    const uint32_t srcLength, const uint32_t srcBaseAddr, const uint64_t mask[], const uint8_t repeatTimes,
    const uint8_t srcRepStride)
{
    static_assert(SupportBytes<T, 1, 2, 4>(), "Gather only support type b8/b16/b32 on current device");
    uint32_t srcBaseOffset = srcBaseAddr/sizeof(T);
    if constexpr (sizeof(T) == 1) {
        GatherImplB8(dstLocal, srcLocal, srcOffsetLocal, srcLength, srcBaseOffset, mask, repeatTimes, srcRepStride);
    } else if constexpr (sizeof(T) == 2) {
        GatherImplB16(dstLocal, srcLocal, srcOffsetLocal, srcLength, srcBaseOffset, mask, repeatTimes, srcRepStride);
    } else if constexpr (sizeof(T) == 4) {
        GatherImplB32(dstLocal, srcLocal, srcOffsetLocal, srcLength, srcBaseOffset, mask, repeatTimes, srcRepStride);
    }
}

// counter mode
template <typename T>
__aicore__ inline void GatherImpl(__ubuf__ T *dstLocal, __ubuf__ T *srcLocal, __ubuf__ uint32_t *srcOffsetLocal,
                                   const uint32_t srcBaseAddr, const uint32_t count)
{
    static_assert(SupportBytes<T, 1, 2, 4>(), "Gather only support type b8/b16/b32 on current device");
    uint32_t srcBaseOffset = srcBaseAddr/sizeof(T);
    if constexpr (sizeof(T) == 1) {
        GatherImplB8(dstLocal, srcLocal, srcOffsetLocal, srcBaseOffset, count);
    } else if constexpr (sizeof(T) == 2) {
        GatherImplB16(dstLocal, srcLocal, srcOffsetLocal, srcBaseOffset, count);
    } else if constexpr (sizeof(T) == 4) {
        GatherImplB32(dstLocal, srcLocal, srcOffsetLocal, srcBaseOffset, count);
    }
}

/* **************************************************************************************************
 * Gatherb                                            *
 * ************************************************************************************************* */

template <typename T>
__aicore__ inline void GatherbImpl(__ubuf__ T* dstLocal, __ubuf__ T* src0Local, __ubuf__ uint32_t* offsetLocal,
    const uint32_t srcLength, uint8_t repeatTimes, const GatherRepeatParams& repeatParams)
{
    static_assert(SupportType<T, int8_t, uint8_t, int16_t, uint16_t, half, float, int32_t, uint32_t, bfloat16_t, int64_t>(),
        "Failed to check dtype in Gatherb, current api support dtype combination is "
        "src and dst both: int8_t, uint8_t, int16_t, uint16_t, half, float, int32_t, uint32_t, bfloat16_t, int64_t");
}

#define GATHERB_T8(FUNC_NAME, DATA_TYPE, REG_TYPE)                                                              \
__aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dstLocal, __ubuf__ DATA_TYPE* srcLocal,                    \
    __ubuf__ uint32_t* srcOffsetLocal, const uint32_t srcLength, uint8_t repeatTimes,                           \
    const GatherRepeatParams& repeatParams)                                                                     \
{                                                                                                               \
    __VEC_SCOPE__                                                                                               \
    {                                                                                                           \
        vector_u32 indexReg;                                                                                    \
        REG_TYPE dstReg;                                                                                        \
        vector_bool preg = pset_b8(PAT_ALL);                                                                    \
        uint32_t strideConfig = (((uint32_t)repeatParams.dstBlkStride) << 16);                                  \
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {                                                  \
            vlds(indexReg, srcOffsetLocal, i * B32_BLK_ELE_GATHER, NORM);                                       \
            vgatherb(dstReg, srcLocal, indexReg);                                                               \
            vsstb(dstReg, dstLocal + i * repeatParams.dstRepStride * B8_BLK_ELE_GATHER, strideConfig, preg);    \
        }                                                                                                       \
    }                                                                                                           \
}                                                                                                               \

#define GATHERB_T16(FUNC_NAME, DATA_TYPE, REG_TYPE)                                                             \
__aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dstLocal, __ubuf__ DATA_TYPE* srcLocal,                    \
    __ubuf__ uint32_t* srcOffsetLocal, const uint32_t srcLength, uint8_t repeatTimes,                           \
    const GatherRepeatParams& repeatParams)                                                                     \
{                                                                                                               \
    __VEC_SCOPE__                                                                                               \
    {                                                                                                           \
        vector_u32 indexReg;                                                                                    \
        REG_TYPE dstReg;                                                                                        \
        vector_bool preg = pset_b16(PAT_ALL);                                                                   \
        uint32_t strideConfig = (((uint32_t)repeatParams.dstBlkStride) << 16);                                  \
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {                                                  \
            vlds(indexReg, srcOffsetLocal, i * B32_BLK_ELE_GATHER, NORM);                                       \
            vgatherb(dstReg, srcLocal, indexReg);                                                               \
            vsstb(dstReg, dstLocal + i * repeatParams.dstRepStride * B16_BLK_ELE_GATHER, strideConfig, preg);   \
        }                                                                                                       \
    }                                                                                                           \
}

#define GATHERB_T32(FUNC_NAME, DATA_TYPE, REG_TYPE)                                                             \
__aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dstLocal, __ubuf__ DATA_TYPE* srcLocal,                    \
    __ubuf__ uint32_t* srcOffsetLocal, const uint32_t srcLength, uint8_t repeatTimes,                           \
    const GatherRepeatParams& repeatParams)                                                                     \
{                                                                                                               \
    __VEC_SCOPE__                                                                                               \
    {                                                                                                           \
        vector_u32 indexReg;                                                                                    \
        REG_TYPE dstReg;                                                                                        \
        vector_bool preg = pset_b32(PAT_ALL);                                                                   \
        uint32_t strideConfig = (((uint32_t)repeatParams.dstBlkStride) << 16);                                  \
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {                                                  \
            vlds(indexReg, srcOffsetLocal, i * B32_BLK_ELE_GATHER, NORM);                                       \
            vgatherb(dstReg, srcLocal, indexReg);                                                               \
            vsstb(dstReg, dstLocal + i * repeatParams.dstRepStride * B32_BLK_ELE_GATHER, strideConfig, preg);   \
        }                                                                                                       \
    }                                                                                                           \
}

#define GATHERB_T64(FUNC_NAME, DATA_TYPE, REG_TYPE)                                                             \
__aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dstLocal, __ubuf__ DATA_TYPE* srcLocal,                    \
    __ubuf__ uint32_t* srcOffsetLocal, const uint32_t srcLength, uint8_t repeatTimes,                           \
    const GatherRepeatParams& repeatParams)                                                                     \
{                                                                                                               \
    __VEC_SCOPE__                                                                                               \
    {                                                                                                           \
        vector_u32 indexReg;                                                                                    \
        REG_TYPE dstReg;                                                                                        \
        vector_bool preg = pset_b32(PAT_ALL);                                                                   \
        uint32_t strideConfig = (((uint32_t)repeatParams.dstBlkStride) << 16);                                  \
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {                                                  \
            vlds(indexReg, srcOffsetLocal, i * B32_BLK_ELE_GATHER, NORM);                                       \
            vgatherb(dstReg, (__ubuf__ uint32_t*)srcLocal, indexReg);                                           \
            vsstb(dstReg, (__ubuf__ uint32_t*)dstLocal + i * repeatParams.dstRepStride * B32_BLK_ELE_GATHER,    \
                strideConfig, preg);                                                                            \
        }                                                                                                       \
    }                                                                                                           \
}
GATHERB_T8(GatherbImpl, int8_t, vector_s8);
GATHERB_T8(GatherbImpl, uint8_t, vector_u8);
GATHERB_T16(GatherbImpl, half, vector_f16);
GATHERB_T16(GatherbImpl, bfloat16_t, vector_bf16);
GATHERB_T16(GatherbImpl, int16_t, vector_s16);
GATHERB_T16(GatherbImpl, uint16_t, vector_u16);
GATHERB_T32(GatherbImpl, float, vector_f32);
GATHERB_T32(GatherbImpl, uint32_t, vector_u32);
GATHERB_T32(GatherbImpl, int32_t, vector_s32);
GATHERB_T64(GatherbImpl, int64_t, vector_u32);
}  // namespace AscendC
#endif  // ASCENDC_MODULE_OPERATOR_VEC_GATHER_IMPL_H