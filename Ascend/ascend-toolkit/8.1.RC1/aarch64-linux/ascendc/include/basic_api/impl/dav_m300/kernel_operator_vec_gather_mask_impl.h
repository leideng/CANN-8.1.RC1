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
 * \file kernel_operator_vec_gather_mask_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_REDUCEV2_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_REDUCEV2_IMPL_H
#include "kernel_struct_gather.h"
 
namespace AscendC {
template <typename T>
__aicore__ inline void GatherMaskCal(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ uint16_t* src1, const bool reduceMode,
    const uint32_t mask, const GatherMaskParams& reducev2Params, uint64_t& rsvdCnt)
{
	ASSERT(false && "unsupported Gather on current device");
}
 
template <typename T>
__aicore__ inline void GatherMaskCal(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ uint32_t* src1, const bool reduceMode,
    const uint32_t mask, const GatherMaskParams& reducev2Params, uint64_t& rsvdCnt)
{
    ASSERT(false && "unsupported Gather on current device");
}

__aicore__ inline int64_t GetGatherMaskRemainCountImpl()
{
    ASCENDC_ASSERT((false), "unsupported GetGatherMaskRemainCount on current device");
    return 0;
}

#define REGISTER_GATHER_MASK_B16(data_type, vector_type)                                                               \
    __aicore__ inline void GatherMaskCal(__ubuf__ data_type* dst, __ubuf__ data_type* src0, __ubuf__ uint16_t* src1,   \
    const bool reduceMode, const uint32_t mask, const GatherMaskParams& reducev2Params, uint64_t& rsvdCnt)             \
    {                                                                                                                  \
        uint32_t blockElm = ONE_BLK_SIZE / sizeof(half);                                                             \
        if (reduceMode) {                                                                                              \
            __VEC_SCOPE__ {                                                                                            \
                vector_align ureg0;                                                                                    \
                vector_##vector_type vreg0;                                                                            \
                vector_##vector_type vreg1;                                                                            \
                vector_bool preg0;                                                                                     \
                vector_bool preg1;                                                                                     \
                vector_bool preg2;                                                                                     \
                sprclr(SPR_AR);                                                                                        \
                uint64_t hoist_let_var0 = ((uint64_t)dst);                                                             \
                uint32_t sreg0;                                                                                        \
                uint32_t strideConfig0 = (((uint32_t)reducev2Params.src0BlockStride) << 16);                           \
                uint32_t repeatElm = VECTOR_REG_WIDTH / sizeof(half);                                                  \
                uint16_t counterLoop = (mask + repeatElm - 1) / repeatElm;                                             \
                for (uint16_t i = 0; i < (uint16_t)reducev2Params.repeatTimes; ++i) {                                  \
                    sreg0 = mask;                                                                                      \
                    for (uint16_t j = 0; j < (uint16_t)counterLoop; ++j) {                                             \
                        preg0 = plt_b16(sreg0, POST_UPDATE);                                                           \
                        vsldb(vreg0, src0 + i * reducev2Params.src0RepeatStride * blockElm +                           \
                            j * 8 * reducev2Params.src0BlockStride * blockElm, strideConfig0, preg0);                  \
                        plds(preg1, ((__ubuf__ uint32_t *)src1),                                                       \
                             i*reducev2Params.src1RepeatStride * 32 + j * 16, US);                                     \
                        pmov(preg2, preg1, preg0);                                                                     \
                        vsqz(vreg1, vreg0, preg2, MODE_STORED);                                                        \
                        vstur(ureg0, vreg1, ((__ubuf__ data_type *&)hoist_let_var0), POST_UPDATE);                     \
                    }                                                                                                  \
                }                                                                                                      \
                vstar(ureg0, ((__ubuf__ data_type *)dst));                                                             \
            }                                                                                                          \
        } else {                                                                                                       \
            __VEC_SCOPE__ {                                                                                            \
                vector_align ureg0;                                                                                    \
                vector_##vector_type vreg0;                                                                            \
                vector_##vector_type vreg1;                                                                            \
                vector_bool preg0;                                                                                     \
                vector_bool preg1 = pset_b16(PAT_ALL);                                                                 \
                sprclr(SPR_AR);                                                                                        \
                uint64_t hoist_let_var0 = ((uint64_t)dst);                                                             \
                uint32_t strideConfig0 = (((uint32_t)reducev2Params.src0BlockStride) << 16);                           \
                for (uint16_t i = 0; i < (uint16_t)reducev2Params.repeatTimes; ++i) {                                  \
                    vsldb(vreg0, src0 + i * reducev2Params.src0RepeatStride * blockElm, strideConfig0, preg1);         \
                    plds(preg0, ((__ubuf__ uint32_t *)src1), i * reducev2Params.src1RepeatStride * 32, US);            \
                    vsqz(vreg1, vreg0, preg0, MODE_STORED);                                                            \
                    vstur(ureg0, vreg1, ((__ubuf__ data_type *&)hoist_let_var0), POST_UPDATE);                         \
                }                                                                                                      \
                vstar(ureg0, ((__ubuf__ data_type *)dst));                                                             \
            }                                                                                                          \
        }                                                                                                              \
        rsvdCnt = get_ar() / 2;                                                                                        \
    }                                                                                                                  \

REGISTER_GATHER_MASK_B16(half, f16)
REGISTER_GATHER_MASK_B16(uint16_t, u16)
REGISTER_GATHER_MASK_B16(int16_t, s16)

#define REGISTER_GATHER_MASK_B32(data_type, vector_type)                                                               \
    __aicore__ inline void GatherMaskCal(__ubuf__ data_type* dst, __ubuf__ data_type* src0, __ubuf__ uint32_t* src1,   \
        const bool reduceMode, const uint32_t mask, const GatherMaskParams& reducev2Params, uint64_t& rsvdCnt)         \
    {                                                                                                                  \
        uint32_t blockElm = ONE_BLK_SIZE / sizeof(float);                                                            \
        if (reduceMode) {                                                                                              \
            __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);              \
            event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));                   \
            SetFlag<HardEvent::S_V>(eventIDSToV);                                                                      \
            WaitFlag<HardEvent::S_V>(eventIDSToV);                                                                     \
            __VEC_SCOPE__ {                                                                                            \
                vector_align ureg0;                                                                                    \
                vector_##vector_type vreg0;                                                                            \
                vector_##vector_type vreg1;                                                                            \
                vector_u8 vreg2;                                                                                       \
                vector_bool preg0;                                                                                     \
                vector_bool preg1;                                                                                     \
                vector_bool preg2;                                                                                     \
                vector_bool preg3;                                                                                     \
                vector_bool preg4 = pset_b8(PAT_VL32);                                                                 \
                vector_align ureg1;                                                                                    \
                sprclr(SPR_AR);                                                                                        \
                uint64_t hoist_let_var0 = ((uint64_t)dst);                                                             \
                uint64_t hoist_let_var1 = ((uint64_t)src1);                                                            \
                uint32_t sreg0 = mask;                                                                                 \
                uint32_t sreg1;                                                                                        \
                uint32_t strideConfig0 = (((uint32_t)reducev2Params.src0BlockStride) << 16);                           \
                uint32_t repeatElm = VECTOR_REG_WIDTH / sizeof(float);                                                 \
                uint16_t counterLoop= (mask + repeatElm - 1) / repeatElm;                                              \
                for (uint16_t i = 0; i < (uint16_t)reducev2Params.repeatTimes; ++i) {                                  \
                    sreg0 = mask;                                                                                      \
                    for (uint16_t j = 0; j < (uint16_t)counterLoop; ++j) {                                             \
                        preg0 = plt_b32(sreg0, POST_UPDATE);                                                           \
                        vsldb(vreg0, src0 + i * reducev2Params.src0RepeatStride * blockElm +                           \
                            j * 8 * reducev2Params.src0BlockStride * blockElm, strideConfig0, preg0);                  \
                        sreg1 = i*reducev2Params.src1RepeatStride * 32 + j * 8;                                        \
                        vldas(ureg1, ((__ubuf__ uint8_t *)src1 + sreg1));                                              \
                        hoist_let_var1 = ((uint64_t)src1) + ((uint64_t)sreg1);                                         \
                        vldus(vreg2, ureg1, ((__ubuf__ uint8_t *&)hoist_let_var1), 0, POST_UPDATE);                    \
                        mem_bar(VST_VST);                                                                              \
                        vsts(vreg2, ((__ubuf__ uint8_t *)tempBuf), 0, NORM_B32, preg4);                                \
                        mem_bar(VST_VLD);                                                                              \
                        plds(preg1, ((__ubuf__ uint32_t *)tempBuf), 0, US);                                            \
                        punpack(preg3, preg1, LOWER);                                                                  \
                        pmov(preg2, preg3, preg0);                                                                     \
                        vsqz(vreg1, vreg0, preg2, MODE_STORED);                                                        \
                        mem_bar(VST_VST);                                                                              \
                        mem_bar(VLD_VST);                                                                              \
                        vstur(ureg0, vreg1, ((__ubuf__ data_type *&)hoist_let_var0), POST_UPDATE);                     \
                    }                                                                                                  \
                }                                                                                                      \
                vstar(ureg0, ((__ubuf__ data_type *)dst));                                                             \
            }                                                                                                          \
            AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);                                                       \
        } else {                                                                                                       \
            __VEC_SCOPE__ {                                                                                            \
                vector_align ureg0;                                                                                    \
                vector_##vector_type vreg0;                                                                            \
                vector_##vector_type vreg1;                                                                            \
                vector_bool preg0;                                                                                     \
                vector_bool preg1 = pset_b32(PAT_ALL);                                                                 \
                vector_bool preg2;                                                                                     \
                sprclr(SPR_AR);                                                                                        \
                uint64_t hoist_let_var0 = ((uint64_t)dst);                                                             \
                uint32_t strideConfig0 = (((uint32_t)reducev2Params.src0BlockStride) << 16);                           \
                for (uint16_t i = 0; i < (uint16_t)reducev2Params.repeatTimes; ++i) {                                  \
                    vsldb(vreg0, src0 + i * reducev2Params.src0RepeatStride * blockElm, strideConfig0, preg1);         \
                    plds(preg0, ((__ubuf__ uint32_t *)src1), i*reducev2Params.src1RepeatStride * 32, US);              \
                    punpack(preg2, preg0, LOWER);                                                                      \
                    vsqz(vreg1, vreg0, preg2, MODE_STORED);                                                            \
                    vstur(ureg0, vreg1, ((__ubuf__ data_type *&)hoist_let_var0), POST_UPDATE);                         \
                }                                                                                                      \
                vstar(ureg0, ((__ubuf__ data_type *)dst));                                                             \
            }                                                                                                          \
        }                                                                                                              \
        rsvdCnt = get_ar() / 4;                                                                                        \
    }                                                                                                                  \

REGISTER_GATHER_MASK_B32(float, f32)
REGISTER_GATHER_MASK_B32(uint32_t, u32)
REGISTER_GATHER_MASK_B32(int32_t, s32)


template <typename T>
__aicore__ inline void GatherMaskCal(__ubuf__ T* dst, __ubuf__ T* src0, const uint8_t src1Pattern,
    const bool reduceMode, const uint32_t mask, const GatherMaskParams& reducev2Params, uint64_t& rsvdCnt)
{
    ASSERT(false && "unsupported Gather on current device");
}

#define REGISTER_GATHER_MASK_SOLID_B16(data_type, vector_type)                                                         \
    __aicore__ inline void GatherMaskCal(__ubuf__ data_type* dst, __ubuf__ data_type* src0, const uint8_t src1Pattern, \
        const bool reduceMode, const uint32_t mask, const GatherMaskParams& reducev2Params, uint64_t& rsvdCnt)         \
    {                                                                                                                  \
        __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);                  \
        const uint32_t pattern[8] = {0, 1431655765, 2863311530, 286331153, 572662306,                                  \
                                     1145324612, 2290649224, 4294967295};                                              \
        uint32_t blockElm = ONE_BLK_SIZE / sizeof(half);                                                             \
        if (reduceMode) {                                                                                              \
            __VEC_SCOPE__ {                                                                                            \
                vector_align ureg0;                                                                                    \
                vector_##vector_type vreg0;                                                                            \
                vector_##vector_type vreg1;                                                                            \
                vector_u32 vreg2;                                                                                      \
                vector_bool preg0;                                                                                     \
                vector_bool preg1;                                                                                     \
                vector_bool preg2;                                                                                     \
                vector_bool preg3 = pset_b32(PAT_VL8);                                                                 \
                vdup(vreg2, pattern[src1Pattern], preg3, MODE_ZEROING);                                                \
                vsts(vreg2, ((__ubuf__ uint32_t *)tempBuf), 0, NORM_B32, preg3);                                       \
                mem_bar(VST_VLD);                                                                                      \
                plds(preg1, ((__ubuf__ uint32_t *)tempBuf), 0, US);                                                    \
                sprclr(SPR_AR);                                                                                        \
                uint64_t hoist_let_var0 = ((uint64_t)dst);                                                             \
                uint32_t sreg0;                                                                                        \
                uint32_t strideConfig0 = (((uint32_t)reducev2Params.src0BlockStride) << 16);                           \
                uint32_t repeatElm = VECTOR_REG_WIDTH / sizeof(half);                                                  \
                uint16_t counterLoop = (mask + repeatElm - 1) / repeatElm;                                             \
                for (uint16_t i = 0; i < (uint16_t)reducev2Params.repeatTimes; ++i) {                                  \
                    sreg0 = mask;                                                                                      \
                    for (uint16_t j = 0; j < (uint16_t)counterLoop; ++j) {                                             \
                        preg0 = plt_b16(sreg0, POST_UPDATE);                                                           \
                        vsldb(vreg0, src0 + i * reducev2Params.src0RepeatStride * blockElm +                           \
                            j * 8 * reducev2Params.src0BlockStride * blockElm, strideConfig0, preg0);                  \
                        pmov(preg2, preg1, preg0);                                                                     \
                        vsqz(vreg1, vreg0, preg2, MODE_STORED);                                                        \
                        vstur(ureg0, vreg1, ((__ubuf__ data_type *&)hoist_let_var0), POST_UPDATE);                     \
                    }                                                                                                  \
                }                                                                                                      \
                vstar(ureg0, ((__ubuf__ data_type *)dst));                                                             \
            }                                                                                                          \
        }                                                                                                              \
        else {                                                                                                         \
            __VEC_SCOPE__ {                                                                                            \
                vector_align ureg0;                                                                                    \
                vector_u32 vreg0;                                                                                      \
                vector_##vector_type vreg1;                                                                            \
                vector_##vector_type vreg2;                                                                            \
                vector_bool preg0 = pset_b32(PAT_VL8);                                                                 \
                vector_bool preg1 = pset_b16(PAT_ALL);                                                                 \
                vector_bool preg2;                                                                                     \
                vdup(vreg0, pattern[src1Pattern], preg0, MODE_ZEROING);                                                \
                vsts(vreg0, ((__ubuf__ uint32_t *)tempBuf), 0, NORM_B32, preg0);                                       \
                mem_bar(VST_VLD);                                                                                      \
                plds(preg2, ((__ubuf__ uint32_t *)tempBuf), 0, US);                                                    \
                sprclr(SPR_AR);                                                                                        \
                uint64_t hoist_let_var0 = ((uint64_t)dst);                                                             \
                uint32_t strideConfig0 = (((uint32_t)reducev2Params.src0BlockStride) << 16);                           \
                for (uint16_t i = 0; i < (uint16_t)reducev2Params.repeatTimes; ++i) {                                  \
                    vsldb(vreg1, src0 + i * reducev2Params.src0RepeatStride * blockElm, strideConfig0, preg1);         \
                    vsqz(vreg2, vreg1, preg2, MODE_STORED);                                                            \
                    vstur(ureg0, vreg2, ((__ubuf__ data_type *&)hoist_let_var0), POST_UPDATE);                         \
                }                                                                                                      \
                vstar(ureg0, ((__ubuf__ data_type *)dst));                                                             \
            }                                                                                                          \
            AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);                                                       \
        }                                                                                                              \
        rsvdCnt = get_ar() / 2;                                                                                        \
    }                                                                                                                  \

REGISTER_GATHER_MASK_SOLID_B16(half, f16)
REGISTER_GATHER_MASK_SOLID_B16(uint16_t, u16)
REGISTER_GATHER_MASK_SOLID_B16(int16_t, s16)

#define REGISTER_GATHER_MASK_SOLID_B32(data_type, vector_type)                                                         \
    __aicore__ inline void GatherMaskCal(__ubuf__ data_type* dst, __ubuf__ data_type* src0, const uint8_t src1Pattern, \
        const bool reduceMode, const uint32_t mask, const GatherMaskParams& reducev2Params, uint64_t& rsvdCnt)         \
    {                                                                                                                  \
        __ubuf__ uint8_t* tempBuf = AscendCUtils::GetTemporaryBufferAddr<uint8_t>(TMP_UB_OFFSET, 16);                  \
        const uint32_t pattern[8] = {0, 1431655765, 2863311530, 286331153, 572662306,                                  \
                                     1145324612, 2290649224, 4294967295};                                              \
        __VEC_SCOPE__ {                                                                                                \
            vector_align ureg0;                                                                                        \
            vector_u32 vreg0;                                                                                          \
            vector_##vector_type vreg1;                                                                                \
            vector_##vector_type vreg2;                                                                                \
            vector_bool preg0 = pset_b32(PAT_VL8);                                                                     \
            vector_bool preg1 = pset_b32(PAT_ALL);                                                                     \
            vector_bool preg2;                                                                                         \
            vector_bool preg3;                                                                                         \
            vdup(vreg0, pattern[src1Pattern], preg0, MODE_ZEROING);                                                    \
            vsts(vreg0, ((__ubuf__ uint32_t *)tempBuf), 0, NORM_B32, preg0);                                           \
            mem_bar(VST_VLD);                                                                                          \
            plds(preg2, ((__ubuf__ uint32_t *)tempBuf), 0, US);                                                        \
            punpack(preg3, preg2, LOWER);                                                                              \
            sprclr(SPR_AR);                                                                                            \
            uint64_t hoist_let_var0 = ((uint64_t)dst);                                                                 \
            uint32_t strideConfig0 = (((uint32_t)reducev2Params.src0BlockStride) << 16);                               \
            uint32_t blockElm = ONE_BLK_SIZE / sizeof(float);                                                        \
            for (uint16_t i = 0; i < (uint16_t)reducev2Params.repeatTimes; ++i) {                                      \
                vsldb(vreg1, src0 + i * reducev2Params.src0RepeatStride * blockElm, strideConfig0, preg1);             \
                vsqz(vreg2, vreg1, preg3, MODE_STORED);                                                                \
                vstur(ureg0, vreg2, ((__ubuf__ data_type *&)hoist_let_var0), POST_UPDATE);                             \
            }                                                                                                          \
            vstar(ureg0, ((__ubuf__ data_type *)dst));                                                                 \
        }                                                                                                              \
        AscendCUtils::FreeTemporaryBuffer<uint8_t>(tempBuf);                                                           \
        rsvdCnt = get_ar() / 4;                                                                                        \
    }                                                                                                                  \

REGISTER_GATHER_MASK_SOLID_B32(float, f32)
REGISTER_GATHER_MASK_SOLID_B32(uint32_t, u32)
REGISTER_GATHER_MASK_SOLID_B32(int32_t, s32)
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_REDUCEV2_IMPL_H
