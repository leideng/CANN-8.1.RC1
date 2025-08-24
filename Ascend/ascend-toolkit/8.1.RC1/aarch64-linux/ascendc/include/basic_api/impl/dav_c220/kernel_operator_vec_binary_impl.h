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
 * \file kernel_operator_vec_binary_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_BINARY_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_BINARY_IMPL_H
#include "kernel_utils.h"
#include "kernel_struct_binary.h"

namespace AscendC {
/* **************************************************************************************************
 * Add                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void AddIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, half, float, int16_t, int32_t>(), "Failed to check dtype in Add, current api support "
        "dtype combination is src and dst both: half / float / int16_t / int32_t.");
    vadd(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

// Add::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void AddImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
        AddIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void AddImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask);
        AddIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}

// Add::Level 2
template <typename T>
__aicore__ inline void AddImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    if ASCEND_IS_AIV {
        ASCENDC_ASSERT((SupportType<T, half, float, int16_t, int32_t>()), {KERNEL_LOG(KERNEL_ERROR,
            "Failed to check dtype in Add, current api support dtype combination is src and dst both: half / float / "
            "int16_t / int32_t.");});
        set_mask_count();
        set_vector_mask(0, calCount);
        vadd(dst, src0, src1, 1, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE,
            DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
        set_mask_norm();
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    }
}

/* **************************************************************************************************
 * Sub                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void SubIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, half, float, int16_t, int32_t>(), "Failed to check dtype in Sub, current api support "
        "dtype combination is src and dst both: half / float / int16_t / int32_t.");
    vsub(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

// Sub::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void SubImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
        SubIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void SubImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask);
        SubIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}

// Sub::Level 2
template <typename T>
__aicore__ inline void SubImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    if ASCEND_IS_AIV {
        ASCENDC_ASSERT((SupportType<T, half, float, int16_t, int32_t>()), {KERNEL_LOG(KERNEL_ERROR,
            "Failed to check dtype in Sub, current api support dtype combination is src and dst both: half / float / "
            "int16_t / int32_t.");});
        set_mask_count();
        set_vector_mask(0, calCount);
        vsub(dst, src0, src1, 1, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE,
            DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
        set_mask_norm();
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    }
}
/* **************************************************************************************************
 * Mul                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void MulIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, half, float, int16_t, int32_t>(), "Failed to check dtype in Mul, current api support "
        "dtype combination is src and dst both: half / float / int16_t / int32_t.");
    vmul(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

// Mul::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void MulImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
        MulIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MulImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask);
        MulIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}

// Mul::Level 2
template <typename T>
__aicore__ inline void MulImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    if ASCEND_IS_AIV {
        ASCENDC_ASSERT((SupportType<T, half, float, int16_t, int32_t>()), {KERNEL_LOG(KERNEL_ERROR,
            "Failed to check dtype in Mul, current api support dtype combination is src and dst both: half / float / "
            "int16_t / int32_t.");});
        set_mask_count();
        set_vector_mask(0, calCount);
        vmul(dst, src0, src1, 1, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE,
            DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
        set_mask_norm();
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    }
}
/* **************************************************************************************************
 * Div                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void DivIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, half, float>(), "Failed to check dtype in Div, current api support dtype combination "
        "is src and dst both: half / float.");
    vdiv(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

// Div::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void DivImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
        DivIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void DivImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask);
        DivIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}

// Div::Level 2
template <typename T>
__aicore__ inline void DivImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    if ASCEND_IS_AIV {
        ASCENDC_ASSERT((SupportType<T, half, float>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Div, "
            "current api support dtype combination is src and dst both: half / float.");});
        set_mask_count();
        set_vector_mask(0, calCount);
        vdiv(dst, src0, src1, 1, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE,
            DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
        set_mask_norm();
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    }
}

/* **************************************************************************************************
 * Max                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void MaxIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, half, float, int16_t, int32_t>(), "Failed to check dtype in Max, current api support "
        "dtype combination is src and dst both: half / float / int16_t / int32_t.");
    vmax(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

// Max::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void MaxImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
        MaxIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MaxImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask);
        MaxIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}

// Max::Level 2
template <typename T>
__aicore__ inline void MaxImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    if ASCEND_IS_AIV {
        ASCENDC_ASSERT((SupportType<T, half, float, int16_t, int32_t>()), {KERNEL_LOG(KERNEL_ERROR,
            "Failed to check dtype in Max, current api support dtype combination is src and dst both: half / float / "
            "int16_t / int32_t.");});
        set_mask_count();
        set_vector_mask(0, calCount);
        vmax(dst, src0, src1, 1, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE,
            DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
        set_mask_norm();
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    }
}

/* **************************************************************************************************
 * Min                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void MinIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, half, float, int16_t, int32_t>(), "Failed to check dtype in Min, current api support "
        "dtype combination is src and dst both: half / float / int16_t / int32_t.");
    vmin(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

// Min::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void MinImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
        MinIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MinImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask);
        MinIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}

// Min::Level 2
template <typename T>
__aicore__ inline void MinImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    if ASCEND_IS_AIV {
        ASCENDC_ASSERT((SupportType<T, half, float, int16_t, int32_t>()), {KERNEL_LOG(KERNEL_ERROR,
            "Failed to check dtype in Min, current api support dtype combination is src and dst both: half / float / "
            "int16_t / int32_t.");});
        set_mask_count();
        set_vector_mask(0, calCount);
        vmin(dst, src0, src1, 1, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE,
            DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
        set_mask_norm();
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    }
}

/* **************************************************************************************************
 * And                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void AndIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, int16_t, uint16_t>(), "Failed to check dtype in And, current api support dtype "
        "combination is src and dst both: int16_t / uint16_t.");
    vand(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

// And::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void AndImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
        AndIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void AndImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask);
        AndIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}

// And::Level 2
template <typename T>
__aicore__ inline void AndImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    if ASCEND_IS_AIV {
        ASCENDC_ASSERT((SupportType<T, int16_t, uint16_t, uint32_t, int32_t>()), {KERNEL_LOG(KERNEL_ERROR,
            "Failed to check dtype in And, current api support dtype combination is src and dst both: int16_t / "
            "uint16_t / uint32_t / int32_t.");});
        set_mask_count();
        set_vector_mask(0, calCount);
        // for compatibility, in older version, all dtype reinterpret_cast to int16_t
        vand((__ubuf__ int16_t*)dst, (__ubuf__ int16_t*)src0, (__ubuf__ int16_t*)src1, 1,
            DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE,
            DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
        set_mask_norm();
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    }
}

/* **************************************************************************************************
 * Or                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void OrIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, int16_t, uint16_t>(), "Failed to check dtype in Or, current api support dtype "
        "combination is src and dst both: int16_t / uint16_t.");
    vor(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride, repeatParams.src1BlkStride,
        repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

// Or::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void OrImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
        OrIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void OrImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask);
        OrIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}

// Or::Level 2
template <typename T>
__aicore__ inline void OrImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    if ASCEND_IS_AIV {
        ASCENDC_ASSERT((SupportType<T, int16_t, uint16_t, uint32_t, int32_t>()), {KERNEL_LOG(KERNEL_ERROR,
            "Failed to check dtype in Or, current api support dtype combination is src and dst both: int16_t / uint16_t"
            " / uint32_t / int32_t.");});
        set_mask_count();
        set_vector_mask(0, calCount);
        // for compatibility, in older version, all dtype reinterpret_cast to int16_t
        vor((__ubuf__ int16_t*)dst, (__ubuf__ int16_t*)src0, (__ubuf__ int16_t*)src1, 1,
            DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE,
            DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
        set_mask_norm();
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    }
}


/* **************************************************************************************************
 * AddRelu                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void AddReluIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, int16_t, half, float>(), "Failed to check dtype in AddRelu, current api support dtype "
        "combination is src and dst both: int16_t / half / float.");
    vaddrelu(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

// AddRelu::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void AddReluImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
        AddReluIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void AddReluImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask);
        AddReluIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}

// AddRelu::Level 2
template <typename T>
__aicore__ inline void AddReluImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    if ASCEND_IS_AIV {
        ASCENDC_ASSERT((SupportType<T, int16_t, half, float>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in "
            "AddRelu, current api support dtype combination is src and dst both: int16_t / half / float.");});
        set_mask_count();
        set_vector_mask(0, calCount);
        vaddrelu((__ubuf__ T*)dst, (__ubuf__ T*)src0, (__ubuf__ T*)src1, 1,
            DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE,
            DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
        set_mask_norm();
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    }
}

/* **************************************************************************************************
 * AddDeqRelu                                             *
 * ************************************************************************************************* */
struct AddDeqReluParams {
    __aicore__ AddDeqReluParams(){};

    uint32_t needTmpSize = 0;
    uint32_t calcSize = 0;
    uint32_t src0Offset = 0;
    uint32_t src1Offset = 0;
    uint32_t dstOffset = 0;
    uint32_t tailSrc0Offset = 0;
    uint32_t tailSrc1Offset = 0;
    uint32_t tailDstOffset = 0;
    uint64_t mask1;
    uint64_t mask2[2];
    uint8_t maskMode = 0;
    uint16_t mainBlock = 0;
    uint16_t tailSize = 0;

    uint8_t repeat = 0;
    uint8_t dstBlkStride = DEFAULT_BLK_STRIDE;
    uint8_t src0BlkStride = DEFAULT_BLK_STRIDE;
    uint8_t src1BlkStride = DEFAULT_BLK_STRIDE;
    uint8_t dstRepStride = DEFAULT_REPEAT_STRIDE;
    uint8_t src0RepStride = DEFAULT_REPEAT_STRIDE;
    uint8_t src1RepStride = DEFAULT_REPEAT_STRIDE;
};

__aicore__ inline void SetAddDeqReluMaskCal(AddDeqReluParams &params)
{
    if (params.maskMode == ADDDEQRELU_MASK_MODE_ONE) {
        AscendCUtils::SetMask<half>(params.mask1);
    } else if (params.maskMode == ADDDEQRELU_MASK_MODE_TWO) {
        AscendCUtils::SetMask<half>(params.mask2[1], params.mask2[0]);
    }
}

template <bool isSetMask = true>
__aicore__ inline void AddDeqReluComput(__ubuf__ half *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1,
    __ubuf__ int32_t *tmpBuffer, AddDeqReluParams &params)
{
    // 1、src1+src2(int32_t)
    vadd(tmpBuffer, src0, src1, params.repeat, DEFAULT_BLK_STRIDE, params.src0BlkStride,
        params.src1BlkStride, DEFAULT_REPEAT_STRIDE, params.src0RepStride, params.src1RepStride);
    pipe_barrier(PIPE_V);
    // 2、cast: int32_t->float
    __ubuf__ float *src0FloatTmp = reinterpret_cast<__ubuf__ float *>(src0);
    vconv_s322f32(src0FloatTmp, tmpBuffer, params.repeat, static_cast<uint16_t>(DEFAULT_BLK_STRIDE),
        static_cast<uint16_t>(DEFAULT_BLK_STRIDE), static_cast<uint16_t>(DEFAULT_REPEAT_STRIDE),
        static_cast<uint16_t>(DEFAULT_REPEAT_STRIDE));
    pipe_barrier(PIPE_V);
    // 3、Muls: castRes * (1/m)
    vmuls(src0FloatTmp, src0FloatTmp, static_cast<float>(DEQ_SHIFT_RIGHT_17_BIT), params.repeat,
        static_cast<uint16_t>(DEFAULT_BLK_STRIDE), static_cast<uint16_t>(DEFAULT_BLK_STRIDE),
        static_cast<uint16_t>(DEFAULT_REPEAT_STRIDE), static_cast<uint16_t>(DEFAULT_REPEAT_STRIDE));
    pipe_barrier(PIPE_V);
    // 4、Muls: MulsRes * (float)DeqScale
    vmuls(src0FloatTmp, src0FloatTmp, static_cast<float>(g_deqValue), params.repeat,
        static_cast<uint16_t>(DEFAULT_BLK_STRIDE), static_cast<uint16_t>(DEFAULT_BLK_STRIDE),
        static_cast<uint16_t>(DEFAULT_REPEAT_STRIDE), static_cast<uint16_t>(DEFAULT_REPEAT_STRIDE));
    pipe_barrier(PIPE_V);
    // 5、Muls: MulsRes * (float)m
    vmuls(src0FloatTmp, src0FloatTmp, static_cast<float>(DEQ_SHIFT_LEFT_17_BIT), params.repeat,
        static_cast<uint16_t>(DEFAULT_BLK_STRIDE), static_cast<uint16_t>(DEFAULT_BLK_STRIDE),
        static_cast<uint16_t>(DEFAULT_REPEAT_STRIDE), static_cast<uint16_t>(DEFAULT_REPEAT_STRIDE));
    pipe_barrier(PIPE_V);
    // 6、cast: float->half
    if constexpr (isSetMask) {
        SetAddDeqReluMaskCal(params);
    }
    __ubuf__ half *src1HalfTmp = reinterpret_cast<__ubuf__ half *>(src1);
    vconv_f322f16(src1HalfTmp, src0FloatTmp, params.repeat, 1, 1, HALF_DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
    pipe_barrier(PIPE_V);
    // 7、Duplicate: 0
    __ubuf__ half *tmpBufferHalf = reinterpret_cast<__ubuf__ half *>(tmpBuffer);
    if (params.maskMode != 0) {
        set_mask_count();
        set_vector_mask(0, static_cast<uint64_t>(params.calcSize));
    }
    vector_dup(tmpBufferHalf, static_cast<half>(0), 1, static_cast<uint16_t>(DEFAULT_BLK_STRIDE), 1,
        DEFAULT_REPEAT_STRIDE, 0);
    if (params.maskMode != 0) {
        set_mask_norm();
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    }
    pipe_barrier(PIPE_V);
    // 8、Max(castRes, DupRes)
    if constexpr (isSetMask) {
        SetAddDeqReluMaskCal(params);
    }
    if (params.maskMode == 0) {
        vmax(dst, tmpBufferHalf, src1HalfTmp, params.repeat, params.dstBlkStride, DEFAULT_BLK_STRIDE,
            DEFAULT_BLK_STRIDE, params.dstRepStride, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
    } else {
        vmax(dst, tmpBufferHalf, src1HalfTmp, params.repeat, params.dstBlkStride, DEFAULT_BLK_STRIDE,
            DEFAULT_BLK_STRIDE, params.dstRepStride, HALF_DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE);
    }
    pipe_barrier(PIPE_V);
}

__aicore__ inline void GetAddDeqReluParamCal(AddDeqReluParams &params, uint8_t repeatTimes)
{
    if (params.needTmpSize <= TMP_UB_SIZE / sizeof(int32_t)) {
        params.calcSize = params.needTmpSize;
        if (params.maskMode != 0) {
            params.repeat = repeatTimes;
        }
    } else {
        params.calcSize = TMP_UB_SIZE / sizeof(int32_t);
        if (params.maskMode != 0) {
            params.repeat = params.calcSize / B32_DATA_NUM_PER_REPEAT;
        }
    }
    if (params.maskMode == 0) {
        params.repeat = repeatTimes;
    }
    params.mainBlock = params.needTmpSize / params.calcSize;
    params.tailSize = params.needTmpSize % params.calcSize;
    if (params.maskMode == 0) {
        params.src0Offset = params.calcSize;
        params.src1Offset = params.calcSize;
        params.dstOffset = params.calcSize;
    } else {
        params.src0Offset = params.repeat * params.src0RepStride * B32_DATA_NUM_PER_BLOCK;
        params.src1Offset = params.repeat * params.src1RepStride * B32_DATA_NUM_PER_BLOCK;
        params.dstOffset = params.repeat * params.dstRepStride * B16_DATA_NUM_PER_BLOCK;
    }
    params.tailSrc0Offset = params.mainBlock * params.src0Offset;
    params.tailSrc1Offset = params.mainBlock * params.src1Offset;
    params.tailDstOffset = params.mainBlock * params.dstOffset;
}

__aicore__ inline void AddDeqReluImpl(__ubuf__ half *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1,
    const int32_t &calCount)
{
    if ASCEND_IS_AIV {
        AddDeqReluParams params;
        params.needTmpSize = calCount;
        GetAddDeqReluParamCal(params, 1);
        __ubuf__ int32_t *tmpBuffer = AscendCUtils::GetTemporaryBufferAddr<int32_t>(TMP_UB_OFFSET, params.calcSize);
        set_mask_count();
        set_vector_mask(0, static_cast<uint64_t>(params.calcSize));
        for (int i = 0; i < params.mainBlock; i++) {
            AddDeqReluComput<false>(dst + i * params.dstOffset, src0 + i * params.src0Offset,
                src1 + i * params.src1Offset, tmpBuffer, params);
        }
        if (params.tailSize != 0) {
            params.calcSize = params.tailSize;
            set_vector_mask(0, static_cast<uint64_t>(params.calcSize));
            AddDeqReluComput<false>(dst + params.tailDstOffset, src0 + params.tailSrc0Offset,
                src1 + params.tailSrc1Offset, tmpBuffer, params);
        }
        set_mask_norm();
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    }
}

// AddDeqRelu::Level 0
template <bool isSetMask = true>
__aicore__ inline void AddDeqReluImpl(__ubuf__ half *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1,
    const uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    if ASCEND_IS_AIV {
        AddDeqReluParams params;
        params.maskMode = ADDDEQRELU_MASK_MODE_ONE;
        params.mask1 = mask;
        params.needTmpSize = repeatTimes * DEFAULT_BLOCK_SIZE / sizeof(int32_t);
        // repeatPramas
        params.dstBlkStride = repeatParams.dstBlkStride;
        params.src0BlkStride = repeatParams.src0BlkStride;
        params.src1BlkStride = repeatParams.src1BlkStride;
        params.dstRepStride = repeatParams.dstRepStride;
        params.src0RepStride = repeatParams.src0RepStride;
        params.src1RepStride = repeatParams.src1RepStride;
        GetAddDeqReluParamCal(params, repeatTimes);
        __ubuf__ int32_t *tmpBuffer = AscendCUtils::GetTemporaryBufferAddr<int32_t>(TMP_UB_OFFSET, params.calcSize);
        if constexpr (isSetMask) {
            AscendCUtils::SetMask<int32_t>(mask);
        }
        for (int i = 0; i < params.mainBlock; i++) {
            AddDeqReluComput<isSetMask>(dst + i * params.dstOffset, src0 + i * params.src0Offset,
                src1 + i * params.src1Offset, tmpBuffer, params);
        }
        if (params.tailSize != 0) {
            if constexpr (isSetMask) {
                AscendCUtils::SetMask<int32_t>(mask);
            }
            params.repeat = repeatTimes - params.repeat * params.mainBlock;
            AddDeqReluComput<isSetMask>(dst + params.tailDstOffset, src0 + params.tailSrc0Offset,
                src1 + params.tailSrc1Offset, tmpBuffer, params);
        }
    }
}

template <bool isSetMask = true>
__aicore__ inline void AddDeqReluImpl(__ubuf__ half *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1,
    const uint64_t mask[], const uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    if ASCEND_IS_AIV {
        AddDeqReluParams params;
        params.maskMode = ADDDEQRELU_MASK_MODE_TWO;
        params.mask2[0] = mask[0];
        params.mask2[1] = mask[1];
        params.needTmpSize = repeatTimes * DEFAULT_BLOCK_SIZE / sizeof(int32_t);
        // repeatPramas
        params.dstBlkStride = repeatParams.dstBlkStride;
        params.src0BlkStride = repeatParams.src0BlkStride;
        params.src1BlkStride = repeatParams.src1BlkStride;
        params.dstRepStride = repeatParams.dstRepStride;
        params.src0RepStride = repeatParams.src0RepStride;
        params.src1RepStride = repeatParams.src1RepStride;

        GetAddDeqReluParamCal(params, repeatTimes);
        __ubuf__ int32_t *tmpBuffer = AscendCUtils::GetTemporaryBufferAddr<int32_t>(TMP_UB_OFFSET, params.calcSize);
        if constexpr (isSetMask) {
            AscendCUtils::SetMask<int32_t>(mask[1], mask[0]);
        }
        for (int i = 0; i < params.mainBlock; i++) {
            AddDeqReluComput<isSetMask>(dst + i * params.dstOffset, src0 + i * params.src0Offset,
                src1 + i * params.src1Offset, tmpBuffer, params);
        }
        if (params.tailSize != 0) {
            if constexpr (isSetMask) {
                AscendCUtils::SetMask<int32_t>(mask[1], mask[0]);
            }
            params.repeat = repeatTimes - params.repeat * params.mainBlock;
            AddDeqReluComput<isSetMask>(dst + params.tailDstOffset, src0 + params.tailSrc0Offset,
                src1 + params.tailSrc1Offset, tmpBuffer, params);
        }
    }
}
/* **************************************************************************************************
 * FusedMulAdd                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void FusedMulAddIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1,
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, half, float>(), "Failed to check dtype in FusedMulAdd, current api support dtype "
        "combination is src and dst both: half / float.");
    vmadd(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

// FusedMulAdd::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void FusedMulAddImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
        FusedMulAddIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void FusedMulAddImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask);
        FusedMulAddIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}

// FusedMulAdd::Level 2
template <typename T>
__aicore__ inline void FusedMulAddImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    if ASCEND_IS_AIV {
        ASCENDC_ASSERT((SupportType<T, half, float>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in "
            "FusedMulAdd, current api support dtype combination is src and dst both: half / float.");});
        set_mask_count();
        set_vector_mask(0, calCount);
        vmadd(dst, src0, src1, 1,
            DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE,
            DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
        set_mask_norm();
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    }
}

/* **************************************************************************************************
 * MulAddDst                                             *
 * ************************************************************************************************* */
template <typename T, typename U>
__aicore__ inline void MulAddDstIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1, uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    static_assert(SupportType<Tuple<T, U>, Tuple<half, half>, Tuple<float, float>, Tuple<float, half>>(), "Failed to "
        "check dtype in MulAddDst, current api support dtype combination is src: half, dst: half / float; src: float, "
        "dst: float.");
    vmla(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

// MulAddDst::Level 0
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void MulAddDstImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
        MulAddDstIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void MulAddDstImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask);
        MulAddDstIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}

// MulAddDst::Level 2
template <typename T, typename U>
__aicore__ inline void MulAddDstImpl(__ubuf__ T* dst, __ubuf__ U* src0, __ubuf__ U* src1, const int32_t& calCount)
{
    if ASCEND_IS_AIV {
        ASCENDC_ASSERT((SupportType<Tuple<T, U>, Tuple<half, half>, Tuple<float, float>, Tuple<float, half>>()),
            {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in MulAddDst, current api support dtype combination is "
            "src: half, dst: half / float; src: float, dst: float.");});
        set_mask_count();
        set_vector_mask(0, calCount);
        if constexpr (sizeof(T) == sizeof(U)) {
            vmla(dst, src0, src1, 1,
                DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE,
                DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
        } else {
            vmla(dst, src0, src1, 1,
                DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE,
                DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE);
        }
        set_mask_norm();
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    }
}

/* **************************************************************************************************
 * FusedMulAddRelu                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void FusedMulAddReluIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1,
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, half, float>(), "Failed to check dtype in FusedMulAddRelu, current api support dtype "
        "combination is src and dst both: half / float.");
    vmaddrelu(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

// FusedMulAddRelu::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void FusedMulAddReluImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1,
    const uint64_t mask[], const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
        FusedMulAddReluIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void FusedMulAddReluImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1,
    const uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask);
        FusedMulAddReluIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}

// FusedMulAddRelu::Level 2
template <typename T>
__aicore__ inline void FusedMulAddReluImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    if ASCEND_IS_AIV {
        ASCENDC_ASSERT((SupportType<T, half, float>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in "
            "FusedMulAddRelu, current api support dtype combination is src and dst both: half / float.");});
        set_mask_count();
        set_vector_mask(0, calCount);
        vmaddrelu(dst, src0, src1, 1, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_REPEAT_STRIDE,
            DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
        set_mask_norm();
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    }
}

/* **************************************************************************************************
 * SubRelu                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void SubReluIntrinsicsImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    static_assert(SupportType<T, int16_t, half, float>(), "Failed to check dtype in SubRelu, current api support dtype "
        "combination is src and dst both: int16_t / half / float.");
    vsubrelu(dst, src0, src1, repeatTimes, repeatParams.dstBlkStride, repeatParams.src0BlkStride,
        repeatParams.src1BlkStride, repeatParams.dstRepStride, repeatParams.src0RepStride, repeatParams.src1RepStride);
}

// SubRelu::Level 2
template <typename T>
__aicore__ inline void SubReluImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& calCount)
{
    if ASCEND_IS_AIV {
        ASCENDC_ASSERT((SupportType<T, int16_t, half, float>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in "
            "SubRelu, current api support dtype combination is src and dst both: int16_t / half / float.");});
        set_mask_count();
        set_vector_mask(0, calCount);
        vsubrelu(dst, src0, src1, 1, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE,
            DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
        set_mask_norm();
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    }
}

// SubRelu::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void SubReluImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask[],
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask[1], mask[0]);
        SubReluIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}

template <typename T, bool isSetMask = true>
__aicore__ inline void SubReluImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const uint64_t mask,
    const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    if ASCEND_IS_AIV {
        AscendCUtils::SetMask<T, isSetMask>(mask);
        SubReluIntrinsicsImpl(dst, src0, src1, repeatTimes, repeatParams);
    }
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_BINARY_IMPL_H