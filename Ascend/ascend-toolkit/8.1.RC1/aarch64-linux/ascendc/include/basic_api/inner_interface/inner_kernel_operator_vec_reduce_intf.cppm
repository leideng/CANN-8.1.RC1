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
 * \file inner_kernel_operator_vec_reduce_intf.cppm
 * \brief
 */
#ifndef ASCENDC_MODULE_INNER_OPERATOR_VEC_REDUCE_INTERFACE_H
#define ASCENDC_MODULE_INNER_OPERATOR_VEC_REDUCE_INTERFACE_H
#include "kernel_tensor.h"

#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_vec_reduce_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_vec_reduce_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_vec_reduce_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_vec_reduce_impl.h"
#endif

#include "kernel_check.h"

namespace AscendC {
#pragma begin_pipe(V)
/* *************** BlockReduceMax /BlockReduceMin /BlockReduceSum PairReduceSum ********************* */
/*
 * @ingroup BlockReduceSum
 * @brief Sum all elements in each block
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] repeat repeat times
 * @param [in] mask[]/maskcount mask array/count
 * @param [in] dstRepStride dst repeat stride
 * @param [in] srcBlkStride src block stride
 * @param [in] srcRepStride src repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void BlockReduceSum(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t repeat, const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in "
        "BlockReduceSum, current api support dtype combination is src and dst both: half / float");});
    ASCENDC_CHECK_VALUE_RANGE(repeat, 0, 255, "repeat", "BlockReduceSum");
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecReduceOther(dstLocal, srcLocal, repeat, mask, dstRepStride, srcBlkStride, srcRepStride,
        "BlockReduceSum")) {
        ASCENDC_REPORT_CHECK_ERROR("BlockReduceSum", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    BlockReduceSumImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), repeat,
        mask, dstRepStride, srcBlkStride, srcRepStride);
}

/*
 * @ingroup BlockReduceMax
 * @brief Maximize all elements in each block
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] repeat repeat times
 * @param [in] mask[]/maskcount mask array/count
 * @param [in] dstRepStride dst repeat stride
 * @param [in] srcBlkStride src block stride
 * @param [in] srcRepStride src repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void BlockReduceMax(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t repeat, const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in "
        "BlockReduceMax, current api support dtype combination is src and dst both: half / float");});
    ASCENDC_CHECK_VALUE_RANGE(repeat, 0, 255, "repeat", "BlockReduceMax");
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecReduceOther(dstLocal, srcLocal, repeat, mask, dstRepStride, srcBlkStride, srcRepStride,
        "BlockReduceMax")) {
        ASCENDC_REPORT_CHECK_ERROR("BlockReduceMax", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    BlockReduceMaxImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), repeat,
        mask, dstRepStride, srcBlkStride, srcRepStride);
}

/*
 * @ingroup BlockReduceMin
 * @brief Find the minimum value of all elements in each block
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] repeat repeat times
 * @param [in] mask[]/maskcount mask array/count
 * @param [in] dstRepStride dst repeat stride
 * @param [in] srcBlkStride src block stride
 * @param [in] srcRepStride src repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void BlockReduceMin(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t repeat, const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in "
        "BlockReduceMin, current api support dtype combination is src and dst both: half / float");});
    ASCENDC_CHECK_VALUE_RANGE(repeat, 0, 255, "repeat", "BlockReduceMin");
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecReduceOther(dstLocal, srcLocal, repeat, mask, dstRepStride, srcBlkStride, srcRepStride,
        "BlockReduceMin")) {
        ASCENDC_REPORT_CHECK_ERROR("BlockReduceMin", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    BlockReduceMinImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), repeat,
        mask, dstRepStride, srcBlkStride, srcRepStride);
}

/*
 * @ingroup PairReduceSum
 * @brief Sum of adjacent inner pair (parity) elements
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] repeat repeat times
 * @param [in] mask[]/maskcount mask array/count
 * @param [in] dstRepStride dst repeat stride
 * @param [in] srcBlkStride src block stride
 * @param [in] srcRepStride src repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void PairReduceSum(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t repeat, const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in "
        "PairReduceSum, current api support dtype combination is src and dst both: half / float");});
    ASCENDC_CHECK_VALUE_RANGE(repeat, 0, 255, "repeat", "PairReduceSum");
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecReduceOther(dstLocal, srcLocal, repeat, mask, dstRepStride, srcBlkStride, srcRepStride,
        "PairReduceSum")) {
        ASCENDC_REPORT_CHECK_ERROR("PairReduceSum", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    PairReduceSumImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), repeat,
        mask, dstRepStride, srcBlkStride, srcRepStride);
}

template <typename T, bool isSetMask>
__aicore__ inline void BlockReduceSum(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t repeat, const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in "
        "BlockReduceSum, current api support dtype combination is src and dst both: half / float");});
    ASCENDC_CHECK_VALUE_RANGE(repeat, 0, 255, "repeat", "BlockReduceSum");
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecReduceOther(dstLocal, srcLocal, repeat, mask, dstRepStride, srcBlkStride, srcRepStride,
        "BlockReduceSum")) {
        ASCENDC_REPORT_CHECK_ERROR("BlockReduceSum", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    BlockReduceSumImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), repeat,
        mask, dstRepStride, srcBlkStride, srcRepStride);
}

template <typename T, bool isSetMask>
__aicore__ inline void BlockReduceMax(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t repeat, const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in "
        "BlockReduceMax, current api support dtype combination is src and dst both: half / float");});
    ASCENDC_CHECK_VALUE_RANGE(repeat, 0, 255, "repeat", "BlockReduceMax");
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecReduceOther(dstLocal, srcLocal, repeat, mask, dstRepStride, srcBlkStride, srcRepStride,
        "BlockReduceMax")) {
        ASCENDC_REPORT_CHECK_ERROR("BlockReduceMax", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    BlockReduceMaxImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), repeat,
        mask, dstRepStride, srcBlkStride, srcRepStride);
}

template <typename T, bool isSetMask>
__aicore__ inline void BlockReduceMin(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t repeat, const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in "
        "BlockReduceMin, current api support dtype combination is src and dst both: half / float");});
    ASCENDC_CHECK_VALUE_RANGE(repeat, 0, 255, "repeat", "BlockReduceMin");
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecReduceOther(dstLocal, srcLocal, repeat, mask, dstRepStride, srcBlkStride, srcRepStride,
        "BlockReduceMin")) {
        ASCENDC_REPORT_CHECK_ERROR("BlockReduceMin", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    BlockReduceMinImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), repeat,
        mask, dstRepStride, srcBlkStride, srcRepStride);
}

template <typename T, bool isSetMask>
__aicore__ inline void PairReduceSum(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t repeat, const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in "
        "PairReduceSum, current api support dtype combination is src and dst both: half / float");});
    ASCENDC_CHECK_VALUE_RANGE(repeat, 0, 255, "repeat", "PairReduceSum");
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecReduceOther(dstLocal, srcLocal, repeat, mask, dstRepStride, srcBlkStride, srcRepStride,
        "PairReduceSum")) {
        ASCENDC_REPORT_CHECK_ERROR("PairReduceSum", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    PairReduceSumImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), repeat,
        mask, dstRepStride, srcBlkStride, srcRepStride);
}

template <typename T, bool isSetMask>
__aicore__ inline void RepeatReduceSum(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t repeat, const int32_t elemsInOneRepeat, const int32_t dstBlkStride, const int32_t srcBlkStride,
    const int32_t dstRepStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in "
        "RepeatReduceSum, current api support dtype combination is src and dst both: half / float");});
    ASCENDC_CHECK_VALUE_RANGE(repeat, 0, 255, "repeat", "RepeatReduceSum");
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecReduceOther(dstLocal, srcLocal, repeat, elemsInOneRepeat, dstRepStride, srcBlkStride, srcRepStride,
        "RepeatReduceSum")) {
        ASCENDC_REPORT_CHECK_ERROR("RepeatReduceSum", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    RepeatReduceSumImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), repeat,
        elemsInOneRepeat, dstBlkStride, srcBlkStride, dstRepStride, srcRepStride);
}

/* **************************************** Whole Reduce Interface ****************************************** */
/*
 * @ingroup WholeReduceSum
 * @brief Sum of all effective elements in each repeat
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] repeat repeat times
 * @param [in] mask[]/maskcount mask array/count
 * @param [in] dstRepStride dst repeat stride
 * @param [in] srcBlkStride src block stride
 * @param [in] srcRepStride src repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void WholeReduceSum(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const uint64_t mask[], const int32_t repeatTimes, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in "
        "WholeReduceSum, current api support dtype combination is src and dst both: half / float");});
    ASCENDC_CHECK_VALUE_RANGE(repeatTimes, 0, 255, "repeatTimes", "WholeReduceSum");
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecReduceOther(dstLocal, srcLocal, repeatTimes, mask, dstRepStride, srcBlkStride, srcRepStride,
        "WholeReduceSum")) {
        ASCENDC_REPORT_CHECK_ERROR("WholeReduceSum", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    WholeReduceSumImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), mask,
        repeatTimes, dstRepStride, srcBlkStride, srcRepStride);
}

/*
 * @ingroup WholeReduceMax
 * @brief Index of the maximum value of all elements in each repeat
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] repeat repeat times
 * @param [in] mask[]/maskcount mask array/count
 * @param [in] dstRepStride dst repeat stride
 * @param [in] srcBlkStride src block stride
 * @param [in] srcRepStride src repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void WholeReduceMax(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const uint64_t mask[], const int32_t repeatTimes, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride, ReduceOrder order)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in "
        "WholeReduceMax, current api support dtype combination is src and dst both: half / float");});
    ASCENDC_CHECK_VALUE_RANGE(repeatTimes, 0, 255, "repeatTimes", "WholeReduceMax");
#if __CCE_AICORE__ >= 220
    ASCENDC_CHECK_VALUE_RANGE((int)order, 0, 3, "order", "WholeReduceMax");
#elif __CCE_AICORE__ <= 200
    ASCENDC_CHECK_VALUE_RANGE((int)order, 0, 1, "order", "WholeReduceMax");
#endif
#if ASCENDC_CPU_DEBUG && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecReduceOtherWhl(dstLocal, srcLocal, repeatTimes, mask, dstRepStride, srcBlkStride, srcRepStride,
        order, "WholeReduceMax")) {
        ASCENDC_REPORT_CHECK_ERROR("WholeReduceMax", KernelFuncType::MASK_BIT_MODE);
    }
#endif
#if ASCENDC_CPU_DEBUG && __CCE_AICORE__ == 300
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecReduceOther(dstLocal, srcLocal, repeatTimes, mask, dstRepStride, srcBlkStride, srcRepStride,
        "WholeReduceMax")) {
        ASCENDC_REPORT_CHECK_ERROR("WholeReduceMax", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    WholeReduceMaxImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), mask,
        repeatTimes, dstRepStride, srcBlkStride, srcRepStride, order);
}

/*
 * @ingroup WholeReduceMin
 * @brief Index of the minimum value of all elements in each repeat
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] repeat repeat times
 * @param [in] mask[]/maskcount mask array/count
 * @param [in] dstRepStride dst repeat stride
 * @param [in] srcBlkStride src block stride
 * @param [in] srcRepStride src repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void WholeReduceMin(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const uint64_t mask[], const int32_t repeatTimes, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride, ReduceOrder order)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in "
        "WholeReduceMin, current api support dtype combination is src and dst both: half / float");});
    ASCENDC_CHECK_VALUE_RANGE(repeatTimes, 0, 255, "repeatTimes", "WholeReduceMin");
#if __CCE_AICORE__ >= 220
    ASCENDC_CHECK_VALUE_RANGE((int)order, 0, 3, "order", "WholeReduceMin");
#elif __CCE_AICORE__ <= 200
    ASCENDC_CHECK_VALUE_RANGE((int)order, 0, 1, "order", "WholeReduceMin");
#endif
#if ASCENDC_CPU_DEBUG && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecReduceOtherWhl(dstLocal, srcLocal, repeatTimes, mask, dstRepStride, srcBlkStride, srcRepStride,
        order, "WholeReduceMin")) {
        ASCENDC_REPORT_CHECK_ERROR("WholeReduceMin", KernelFuncType::MASK_BIT_MODE);
    }
#endif
#if ASCENDC_CPU_DEBUG && __CCE_AICORE__ == 300
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecReduceOther(dstLocal, srcLocal, repeatTimes, mask, dstRepStride, srcBlkStride, srcRepStride,
        "WholeReduceMax")) {
        ASCENDC_REPORT_CHECK_ERROR("WholeReduceMax", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    WholeReduceMinImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(),
        mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride, order);
}

template <typename T, bool isSetMask>
__aicore__ inline void WholeReduceSum(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t mask, const int32_t repeatTimes, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in "
        "WholeReduceSum, current api support dtype combination is src and dst both: half / float");});
    ASCENDC_CHECK_VALUE_RANGE(repeatTimes, 0, 255, "repeatTimes", "WholeReduceSum");
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecReduceOther(dstLocal, srcLocal, repeatTimes, mask, dstRepStride, srcBlkStride, srcRepStride,
        "WholeReduceSum")) {
        ASCENDC_REPORT_CHECK_ERROR("WholeReduceSum", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    WholeReduceSumImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(),
        mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride);
}
template <typename T, bool isSetMask>
__aicore__ inline void WholeReduceMax(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t mask, const int32_t repeatTimes, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride, ReduceOrder order)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in "
        "WholeReduceMax, current api support dtype combination is src and dst both: half / float");});
    ASCENDC_CHECK_VALUE_RANGE(repeatTimes, 0, 255, "repeatTimes", "WholeReduceMax");
#if __CCE_AICORE__ == 220
    ASCENDC_CHECK_VALUE_RANGE((int)order, 0, 3, "order", "WholeReduceMax");
#elif __CCE_AICORE__ == 200
    ASCENDC_CHECK_VALUE_RANGE((int)order, 0, 1, "order", "WholeReduceMax");
#endif
#if ASCENDC_CPU_DEBUG && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecReduceOtherWhl(dstLocal, srcLocal, repeatTimes, mask, dstRepStride, srcBlkStride, srcRepStride,
        order, "WholeReduceMax")) {
        ASCENDC_REPORT_CHECK_ERROR("WholeReduceMax", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
#if ASCENDC_CPU_DEBUG && __CCE_AICORE__ == 300
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecReduceOther(dstLocal, srcLocal, repeatTimes, mask, dstRepStride, srcBlkStride, srcRepStride,
        "WholeReduceMax")) {
        ASCENDC_REPORT_CHECK_ERROR("WholeReduceMax", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    WholeReduceMaxImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(),
        mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride, order);
}
template <typename T, bool isSetMask>
__aicore__ inline void WholeReduceMin(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t mask, const int32_t repeatTimes, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride, ReduceOrder order)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in "
        "WholeReduceMin, current api support dtype combination is src and dst both: half / float");});
    ASCENDC_CHECK_VALUE_RANGE(repeatTimes, 0, 255, "repeatTimes", "WholeReduceMin");
#if __CCE_AICORE__ >= 220
    ASCENDC_CHECK_VALUE_RANGE((int)order, 0, 3, "order", "WholeReduceMin");
#elif __CCE_AICORE__ <= 200
    ASCENDC_CHECK_VALUE_RANGE((int)order, 0, 1, "order", "WholeReduceMin");
#endif
#if ASCENDC_CPU_DEBUG && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecReduceOtherWhl(dstLocal, srcLocal, repeatTimes, mask, dstRepStride, srcBlkStride, srcRepStride,
        order, "WholeReduceMin")) {
        ASCENDC_REPORT_CHECK_ERROR("WholeReduceMin", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
#if ASCENDC_CPU_DEBUG && __CCE_AICORE__ == 300
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecReduceOther(dstLocal, srcLocal, repeatTimes, mask, dstRepStride, srcBlkStride, srcRepStride,
        "WholeReduceMax")) {
        ASCENDC_REPORT_CHECK_ERROR("WholeReduceMax", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    WholeReduceMinImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(),
        mask, repeatTimes, dstRepStride, srcBlkStride, srcRepStride, order);
}

/* **************************************** Reduce Interface ****************************************** */
/*
 * @ingroup ReduceMax Level 0
 * @brief Index of the maximum value of all input elements
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] workLocal LocalTensor to store the intermediate results
 * @param [in] repeat repeat times
 * @param [in] mask[]/maskcount mask array/count
 * @param [in] srcRepStride src repeat stride
 * @param [in] calIndex Specify whether to get the index with the highest value
 */
template <typename T>
__aicore__ inline void ReduceMax(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<T>& workLocal, const int32_t mask, const int32_t repeatTimes, const int32_t srcRepStride,
    bool calIndex)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in ReduceMax, "
        "current api support dtype combination is src and dst both: half / float");});
#if ASCENDC_CPU_DEBUG
    if (!CheckFunVecReduce(dstLocal, srcLocal, workLocal, repeatTimes, mask, calIndex, srcRepStride, "ReduceMax")) {
        ASCENDC_REPORT_CHECK_ERROR("ReduceMax", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    ReduceRepeatParams params(mask, repeatTimes, DEFAULT_REDUCE_DST_REP_SRIDE, DEFAULT_BLK_STRIDE, srcRepStride);

    ReduceImpl<T>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(),
        (__ubuf__ T*)workLocal.GetPhyAddr(), params, calIndex, ReduceMode::REDUCE_MAX);
}

/*
 * @ingroup ReduceMin
 * @brief Index of the minimum value of all input elements
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] workLocal LocalTensor to store the intermediate results
 * @param [in] repeat repeat times
 * @param [in] mask[]/maskcount mask array/count
 * @param [in] srcRepStride src repeat stride
 * @param [in] calIndex Specify whether to get the index with the highest value
 */
template <typename T>
__aicore__ inline void ReduceMin(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<T>& workLocal, const int32_t mask, const int32_t repeatTimes, const int32_t srcRepStride,
    bool calIndex)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in ReduceMin, "
        "current api support dtype combination is src and dst both: half / float");});
#if ASCENDC_CPU_DEBUG
    if (!CheckFunVecReduce(dstLocal, srcLocal, workLocal, repeatTimes, mask, calIndex, srcRepStride, "ReduceMin")) {
        ASCENDC_REPORT_CHECK_ERROR("ReduceMin", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    struct ReduceRepeatParams params(mask, repeatTimes, DEFAULT_REDUCE_DST_REP_SRIDE, DEFAULT_BLK_STRIDE, srcRepStride);

    ReduceImpl<T>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(),
        (__ubuf__ T*)workLocal.GetPhyAddr(), params, calIndex, ReduceMode::REDUCE_MIN);
}

/*
 * @ingroup ReduceSum
 * @brief sum all input elements
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] workLocal LocalTensor to store the intermediate results
 * @param [in] repeat repeat times
 * @param [in] mask[]/maskcount mask array/count
 * @param [in] srcRepStride src repeat stride
 */
template <typename T>
__aicore__ inline void ReduceSum(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<T>& workLocal, const int32_t mask, const int32_t repeatTimes, const int32_t srcRepStride)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in ReduceSum, "
        "current api support dtype combination is src and dst both: half / float");});
#if ASCENDC_CPU_DEBUG
    if (!CheckFunVecReduce(dstLocal, srcLocal, workLocal, repeatTimes, mask, srcRepStride, "ReduceSum")) {
        ASCENDC_REPORT_CHECK_ERROR("ReduceSum", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    ReduceRepeatParams params(mask, repeatTimes, DEFAULT_REDUCE_DST_REP_SRIDE, DEFAULT_BLK_STRIDE, srcRepStride);

    ReduceImpl<T>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(),
        (__ubuf__ T*)workLocal.GetPhyAddr(), params, 0, ReduceMode::REDUCE_SUM);
}

template <typename T>
__aicore__ inline void ReduceMax(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<T>& workLocal, const uint64_t mask[], const int32_t repeatTimes, const int32_t srcRepStride,
    bool calIndex)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in ReduceMax, "
        "current api support dtype combination is src and dst both: half / float");});
#if ASCENDC_CPU_DEBUG
    if (!CheckFunVecReduce(dstLocal, srcLocal, workLocal, repeatTimes, mask, calIndex, srcRepStride, "ReduceMax")) {
        ASCENDC_REPORT_CHECK_ERROR("ReduceMax", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    struct ReduceRepeatParams params(mask, repeatTimes, DEFAULT_REDUCE_DST_REP_SRIDE, DEFAULT_BLK_STRIDE, srcRepStride);

    ReduceImpl<T>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(),
        (__ubuf__ T*)workLocal.GetPhyAddr(), params, calIndex, ReduceMode::REDUCE_MAX);
}
template <typename T>
__aicore__ inline void ReduceMin(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<T>& workLocal, const uint64_t mask[], const int32_t repeatTimes, const int32_t srcRepStride,
    bool calIndex)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in ReduceMin, "
        "current api support dtype combination is src and dst both: half / float");});
#if ASCENDC_CPU_DEBUG
    if (!CheckFunVecReduce(dstLocal, srcLocal, workLocal, repeatTimes, mask, calIndex, srcRepStride, "ReduceMin")) {
        ASCENDC_REPORT_CHECK_ERROR("ReduceMin", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    struct ReduceRepeatParams params(mask, repeatTimes, DEFAULT_REDUCE_DST_REP_SRIDE, DEFAULT_BLK_STRIDE, srcRepStride);

    ReduceImpl<T>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(),
        (__ubuf__ T*)workLocal.GetPhyAddr(), params, calIndex, ReduceMode::REDUCE_MIN);
}
template <typename T>
__aicore__ inline void ReduceSum(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<T>& workLocal, const uint64_t mask[], const int32_t repeatTimes, const int32_t srcRepStride)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in ReduceSum, "
        "current api support dtype combination is src and dst both: half / float");});
#if ASCENDC_CPU_DEBUG
    if (!CheckFunVecReduce(dstLocal, srcLocal, workLocal, repeatTimes, mask, srcRepStride, "ReduceSum")) {
        ASCENDC_REPORT_CHECK_ERROR("ReduceSum", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    struct ReduceRepeatParams params(mask, repeatTimes, DEFAULT_REDUCE_DST_REP_SRIDE, DEFAULT_BLK_STRIDE, srcRepStride);

    ReduceImpl<T>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(),
        (__ubuf__ T*)workLocal.GetPhyAddr(), params, 0, ReduceMode::REDUCE_SUM);
}

/*
 * @ingroup ReduceMin Level 2
 * @brief Index of the minimum value of all input elements
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] workLocal LocalTensor to store the intermediate results
 * @param [in] count Number of data involved in calculation
 * @param [in] calIndex Specify whether to get the index with the highest value
 */
template <typename T>
__aicore__ inline void ReduceMin(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<T>& workLocal, const int32_t count, bool calIndex)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in ReduceMin, "
        "current api support dtype combination is src and dst both: half / float");});
    ASCENDC_CHECK_VALUE_RANGE(count, 1, TOTAL_UB_SIZE / sizeof(T), "count", "ReduceMin");
    int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T); // fp16=128 , fp32=64
    int32_t repeatTimes = count / elementNumPerRep;
    int32_t tailCount = count % elementNumPerRep; // tailCount  <= 128/64 repeat=1
    int32_t bodyCount = elementNumPerRep;

    if (repeatTimes == 0) { // if count < elementNumPerRep ,repeatTimes will be 0
        repeatTimes = 1;
        bodyCount = count;
        tailCount = 0;
    }
#if ASCENDC_CPU_DEBUG
    if (!CheckFunVecReduce(dstLocal, srcLocal, workLocal, repeatTimes, count, calIndex, "ReduceMin")) {
        ASCENDC_REPORT_CHECK_ERROR("ReduceMin", KernelFuncType::NONE_MODE);
    }
#endif
    struct ReduceRepeatParams params(bodyCount, repeatTimes, DEFAULT_REDUCE_DST_REP_SRIDE, DEFAULT_BLK_STRIDE,
        DEFAULT_REPEAT_STRIDE);
    ReduceImpl<T>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(),
        (__ubuf__ T*)workLocal.GetPhyAddr(), params, calIndex, ReduceMode::REDUCE_MIN);

    if (tailCount != 0) {
        ReduceTailCompute(dstLocal, srcLocal, workLocal, count, calIndex, ReduceMode::REDUCE_MIN);
    }
}

/*
 * @ingroup ReduceMax Level 2
 * @brief Index of the maximum value of all input elements
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] workLocal LocalTensor to store the intermediate results
 * @param [in] count Number of data involved in calculation
 * @param [in] calIndex Specify whether to get the index with the highest value
 */
template <typename T>
__aicore__ inline void ReduceMax(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<T>& workLocal, const int32_t count, bool calIndex)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in ReduceMax, "
        "current api support dtype combination is src and dst both: half / float");});
    ASCENDC_CHECK_VALUE_RANGE(count, 1, TOTAL_UB_SIZE / sizeof(T), "count", "ReduceMax");
    int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T); // fp16=128 , fp32=64
    int32_t repeatTimes = count / elementNumPerRep;
    int32_t tailCount = count % elementNumPerRep; // tailCount  <= 128/64 repeat=1
    int32_t bodyCount = elementNumPerRep;

    if (repeatTimes == 0) { // if count < elementNumPerRep ,repeatTimes will be 0
        repeatTimes = 1;
        bodyCount = count;
        tailCount = 0;
    }
#if ASCENDC_CPU_DEBUG
    if (!CheckFunVecReduce(dstLocal, srcLocal, workLocal, repeatTimes, count, calIndex, "ReduceMax")) {
        ASCENDC_REPORT_CHECK_ERROR("ReduceMax", KernelFuncType::NONE_MODE);
    }
#endif

    struct ReduceRepeatParams params(bodyCount, repeatTimes, DEFAULT_REDUCE_DST_REP_SRIDE, DEFAULT_BLK_STRIDE,
        DEFAULT_REPEAT_STRIDE);
    ReduceImpl<T>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(),
        (__ubuf__ T*)workLocal.GetPhyAddr(), params, calIndex, ReduceMode::REDUCE_MAX);

    if (tailCount != 0) {
        ReduceTailCompute(dstLocal, srcLocal, workLocal, count, calIndex, ReduceMode::REDUCE_MAX);
    }
}

/*
 * @ingroup ReduceSum Level 2
 * @brief sum all input elements
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] workLocal LocalTensor to store the intermediate results
 * @param [in] count Number of data involved in calculation
 */
template <typename T, bool isSetMask>
__aicore__ inline void ReduceSum(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<T>& workLocal, const int32_t count)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in ReduceSum, "
        "current api support dtype combination is src and dst both: half / float");});
    ASCENDC_CHECK_VALUE_RANGE(count, 1, TOTAL_UB_SIZE / sizeof(T), "count", "ReduceSum");
#if __CCE_AICORE__ == 220
    ReduceSumImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), count);
#else
    int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T); // fp16=128 , fp32=64
    int32_t repeatTimes = count / elementNumPerRep;
    int32_t tailCount = count % elementNumPerRep; // tailCount  <= 128/64 repeat=1
    int32_t bodyCount = elementNumPerRep;

    if (repeatTimes == 0) { // if count < elementNumPerRep ,repeatTimes will be 0
        repeatTimes = 1;
        bodyCount = count;
        tailCount = 0;
    }
#if ASCENDC_CPU_DEBUG
    if (!CheckFunVecReduce(dstLocal, srcLocal, workLocal, count, repeatTimes, "ReduceSum")) {
        ASCENDC_REPORT_CHECK_ERROR("ReduceSum", KernelFuncType::NONE_MODE);
    }
#endif

    struct ReduceRepeatParams params(bodyCount, repeatTimes, DEFAULT_REDUCE_DST_REP_SRIDE, DEFAULT_BLK_STRIDE,
        DEFAULT_REPEAT_STRIDE);
    ReduceImpl<T>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(),
        (__ubuf__ T*)workLocal.GetPhyAddr(), params, 0, ReduceMode::REDUCE_SUM);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    T bodySumValue = dstLocal.GetValue(0);

    if (tailCount != 0) {
        struct ReduceRepeatParams tailParams(tailCount, 1, DEFAULT_REDUCE_DST_REP_SRIDE, DEFAULT_BLK_STRIDE,
            DEFAULT_REPEAT_STRIDE);

        ReduceImpl<T>((__ubuf__ T*)dstLocal.GetPhyAddr(),
            (__ubuf__ T*)srcLocal.GetPhyAddr(elementNumPerRep * repeatTimes), (__ubuf__ T*)workLocal.GetPhyAddr(),
            tailParams, 0, ReduceMode::REDUCE_SUM);
        event_t eventIdVToS1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS1);
        WaitFlag<HardEvent::V_S>(eventIdVToS1);
        T tailSumValue = dstLocal.GetValue(0);

        workLocal.SetValue(0, bodySumValue);
        workLocal.SetValue(1, tailSumValue); // bodyresult tailresult vcadd again
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        struct ReduceRepeatParams newParams(2, 1, DEFAULT_REDUCE_DST_REP_SRIDE, DEFAULT_BLK_STRIDE,
            DEFAULT_REPEAT_STRIDE);

        ReduceImpl<T>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)workLocal.GetPhyAddr(),
            (__ubuf__ T*)workLocal.GetPhyAddr(), newParams, 0, ReduceMode::REDUCE_SUM);
    }
#endif
}
#pragma end_pipe
template <typename T>
__aicore__ inline void GetReduceMaxMinCount(uint32_t &maxMinValue, uint32_t &maxMinIndex)
{
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        return;
    }
#endif
    GetReduceMaxMinCountImpl<T>(maxMinValue, maxMinIndex);
}

template <typename T>
__aicore__ inline void GetReduceMaxMinCount(uint32_t &maxMinValue)
{
    GetReduceMaxMinCountImpl<T>(maxMinValue);
}

__aicore__ inline int64_t GetAccVal()
{
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        return 0;
    }
#endif
    return get_acc_val();
}

template <typename T>
__aicore__ inline __inout_pipe__(S) void GetReduceMaxMinCount(T &maxMinValue, T &maxMinIndex)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in "
        "GetReduceMaxMinCount, current api support dtype combination is maxMinValue and maxMinIndex both: half / "
        "float");});
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        return;
    }
#endif
    GetReduceMaxMinCountImpl<T>(maxMinValue, maxMinIndex);
}

template <typename T>
__aicore__ inline __inout_pipe__(S) void GetReduceMaxMinCount(T &maxMinValue)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in "
        "GetReduceMaxMinCount, current api support dtype combination is maxMinValue: half / float");});
    GetReduceMaxMinCountImpl<T>(maxMinValue);
}

template <typename T>
__aicore__ inline __inout_pipe__(S) T GetAccVal()
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in GetAccVal, "
        "current api support dtype combination is half / float");});
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        return 0;
    }
#endif
    return GetAccValImpl<T>();
}
} // namespace AscendC
#endif // ASCENDC_MODULE_INNER_OPERATOR_VEC_REDUCE_INTERFACE_H
