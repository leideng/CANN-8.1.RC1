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
__aicore__ inline void GatherMaskImpl(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src0, __ubuf__ uint16_t* src1,
    const uint8_t PatternMode, const bool reduceMode, const uint32_t mask, const GatherMaskParams& reducev2Params,
    uint64_t& rsvdCnt)
{
    // reduceMode false - norm mode, mask shopuld be none
    // reduceMode true - counter mode, src1 shoud be tensor
    if (reduceMode) {
        SetMaskCount();
    } else {
        SetMaskNorm();
    }
#if ASCENDC_CPU_DEBUG
    if (reduceMode) {
        set_vector_mask(0, mask);
    } else {
        AscendCUtils::SetMask<uint16_t>(ONE_REPEAT_BYTE_SIZE / sizeof(uint16_t));
    }
#else
    set_vector_mask(0, mask);
#endif

    vreduce(dst, src0, src1, reducev2Params.repeatTimes, 1, reducev2Params.src0BlockStride, PatternMode, 8,
        reducev2Params.src0RepeatStride, reducev2Params.src1RepeatStride);
    rsvdCnt = AscendCUtils::GetRsvdCnt();
    SetMaskNorm();
}

__aicore__ inline void GatherMaskImpl(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src0, __ubuf__ uint32_t* src1,
    const uint8_t PatternMode, const bool reduceMode, const uint32_t mask, const GatherMaskParams& gatherMaskParams,
    uint64_t& rsvdCnt)
{
    if (reduceMode) {
        SetMaskCount();
    } else {
        SetMaskNorm();
    }
#if ASCENDC_CPU_DEBUG
    if (reduceMode) {
        set_vector_mask(0, mask);
    } else {
        AscendCUtils::SetMask<uint32_t>(ONE_REPEAT_BYTE_SIZE / sizeof(uint32_t));
    }
#else
    set_vector_mask(0, mask);
#endif
    vreduce(dst, src0, src1, gatherMaskParams.repeatTimes, 1, gatherMaskParams.src0BlockStride, PatternMode, 8,
        gatherMaskParams.src0RepeatStride, gatherMaskParams.src1RepeatStride);
    rsvdCnt = AscendCUtils::GetRsvdCnt();
    SetMaskNorm();
}

template <typename T>
__aicore__ inline void GatherMaskImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint8_t PatternMode,
    const GatherMaskParams& gatherMaskParams)
{
    ASCENDC_ASSERT((SupportType<T, half, uint16_t, int16_t, float, uint32_t, int32_t>()), {KERNEL_LOG(KERNEL_ERROR,
        "Failed to check dtype in GatherMask, current api support dtype combination is src and dst both: half / "
        "uint16_t / int16_t / float / uint32_t / int32_t.");});
    if (sizeof(T) == sizeof(uint16_t)) {
        vreduce(reinterpret_cast<__ubuf__ uint16_t*>(dst), reinterpret_cast<__ubuf__ uint16_t*>(src0),
            reinterpret_cast<__ubuf__ uint16_t*>(src1), gatherMaskParams.repeatTimes, 1,
            gatherMaskParams.src0BlockStride, PatternMode, 8, gatherMaskParams.src0RepeatStride,
            gatherMaskParams.src1RepeatStride);
    } else {
        vreduce(reinterpret_cast<__ubuf__ uint32_t*>(dst), reinterpret_cast<__ubuf__ uint32_t*>(src0),
            reinterpret_cast<__ubuf__ uint32_t*>(src1), gatherMaskParams.repeatTimes, 1,
            gatherMaskParams.src0BlockStride, PatternMode, 8, gatherMaskParams.src0RepeatStride,
            gatherMaskParams.src1RepeatStride);
    }
}

template <typename T>
__aicore__ inline void GatherMaskImpl(
    __ubuf__ T* dst, __ubuf__ T* src0, const uint8_t PatternMode, const GatherMaskParams& gatherMaskParams)
{
    ASCENDC_CHECK_VALUE_RANGE(PatternMode, 1, 6, "src1Pattern", "GatherMask");
    __ubuf__ T* nullsrc1 = ONE_REPEAT_BYTE_SIZE * sizeof(T) + src0;
    GatherMaskImpl(dst, src0, nullsrc1, PatternMode, gatherMaskParams);
}

template <typename T, GatherMaskMode mode = defaultGatherMaskMode>
__aicore__ inline void GatherMaskCal(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ uint16_t* src1, const bool reduceMode,
    const uint32_t mask, const GatherMaskParams& reducev2Params, uint64_t& rsvdCnt)
{
    ASCENDC_REPORT_NOT_SUPPORT(mode == GatherMaskMode::VERSION_V1, "GatherMask with mode = GatherMaskMode::V2");
    ASCENDC_ASSERT((SupportType<T, half, uint16_t, int16_t>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in "
        "GatherMask when src1Pattern is uint16_t tensor, current api support dtype combination is src and dst both: "
        "half / uint16_t / int16_t.");});
    GatherMaskImpl(reinterpret_cast<__ubuf__ uint16_t*>(dst), reinterpret_cast<__ubuf__ uint16_t*>(src0), src1, 0,
        reduceMode, mask, reducev2Params, rsvdCnt);
}

template <typename T, GatherMaskMode mode = defaultGatherMaskMode>
__aicore__ inline void GatherMaskCal(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ uint32_t* src1, const bool reduceMode,
    const uint32_t mask, const GatherMaskParams& reducev2Params, uint64_t& rsvdCnt)
{
    ASCENDC_REPORT_NOT_SUPPORT(mode == GatherMaskMode::VERSION_V1, "GatherMask with mode = GatherMaskMode::V2");
    ASCENDC_ASSERT((SupportType<T, float, uint32_t, int32_t>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in "
        "GatherMask when src1Pattern is uint32_t tensor, current api support dtype combination is src and dst both: "
        "float / uint32_t / int32_t.");});
    GatherMaskImpl(reinterpret_cast<__ubuf__ uint32_t*>(dst), reinterpret_cast<__ubuf__ uint32_t*>(src0), src1, 0,
        reduceMode, mask, reducev2Params, rsvdCnt);
}

template <typename T, GatherMaskMode mode = defaultGatherMaskMode>
__aicore__ inline void GatherMaskCal(__ubuf__ T* dst, __ubuf__ T* src0, const uint8_t src1Pattern,
    const bool reduceMode, const uint32_t mask, const GatherMaskParams& reducev2Params, uint64_t& rsvdCnt)
{
    ASCENDC_REPORT_NOT_SUPPORT(mode == GatherMaskMode::VERSION_V1, "GatherMask with mode = GatherMaskMode::V2");
    ASCENDC_CHECK_VALUE_RANGE(src1Pattern, 1, 6, "src1Pattern", "GatherMask");
    ASCENDC_ASSERT((SupportType<T, half, uint16_t, int16_t, float, uint32_t, int32_t>()), {KERNEL_LOG(KERNEL_ERROR,
        "Failed to check dtype in GatherMask, current api support dtype combination is src and dst both: half / "
        "uint16_t / int16_t / float / uint32_t / int32_t.");});
    __ubuf__ T* nullsrc1 = ONE_REPEAT_BYTE_SIZE * sizeof(T) + src0;
    if (sizeof(T) == sizeof(uint16_t)) {
        GatherMaskImpl(reinterpret_cast<__ubuf__ uint16_t*>(dst), reinterpret_cast<__ubuf__ uint16_t*>(src0),
            reinterpret_cast<__ubuf__ uint16_t*>(nullsrc1), src1Pattern, reduceMode, mask, reducev2Params, rsvdCnt);
    } else {
        GatherMaskImpl(reinterpret_cast<__ubuf__ uint32_t*>(dst), reinterpret_cast<__ubuf__ uint32_t*>(src0),
            reinterpret_cast<__ubuf__ uint32_t*>(nullsrc1), src1Pattern, reduceMode, mask, reducev2Params, rsvdCnt);
    }
}

__aicore__ inline int64_t GetGatherMaskRemainCountImpl()
{
    return get_rsvd_cnt();
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_REDUCEV2_IMPL_H
