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
 * \file kernel_operator_vec_reduce_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_REDUCE_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_VEC_REDUCE_INTERFACE_H
#include "kernel_tensor.h"

#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_vec_reduce_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_vec_reduce_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_vec_reduce_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_vec_reduce_impl.h"
#elif defined(__DAV_M310__)
#include "dav_m310/kernel_operator_vec_reduce_impl.h"
#endif

#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif

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
template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceSum(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t repeat, const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride);

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
template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceMax(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t repeat, const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride);

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
template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceMin(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t repeat, const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride);

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
template <typename T, bool isSetMask = true>
__aicore__ inline void PairReduceSum(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t repeat, const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride);

template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceSum(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t repeat, const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride);

template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceMax(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t repeat, const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride);

template <typename T, bool isSetMask = true>
__aicore__ inline void BlockReduceMin(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t repeat, const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride);

template <typename T, bool isSetMask = true>
__aicore__ inline void PairReduceSum(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t repeat, const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride);

template <typename T, bool isSetMask = true>
__aicore__ inline void RepeatReduceSum(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t repeat, const int32_t elemsInOneRepeat, const int32_t dstBlkStride, const int32_t srcBlkStride,
    const int32_t dstRepStride, const int32_t srcRepStride);

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
template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceSum(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const uint64_t mask[], const int32_t repeatTimes, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride);

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
template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMax(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const uint64_t mask[], const int32_t repeatTimes, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride, ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX);

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
template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMin(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const uint64_t mask[], const int32_t repeatTimes, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride, ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX);

template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceSum(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t mask, const int32_t repeatTimes, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride);
template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMax(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t mask, const int32_t repeatTimes, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride, ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX);
template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMin(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t mask, const int32_t repeatTimes, const int32_t dstRepStride, const int32_t srcBlkStride,
    const int32_t srcRepStride, ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX);

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
    bool calIndex = 0);

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
    bool calIndex = 0);

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
    const LocalTensor<T>& workLocal, const int32_t mask, const int32_t repeatTimes, const int32_t srcRepStride);

template <typename T>
__aicore__ inline void ReduceMax(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<T>& workLocal, const uint64_t mask[], const int32_t repeatTimes, const int32_t srcRepStride,
    bool calIndex = 0);
template <typename T>
__aicore__ inline void ReduceMin(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<T>& workLocal, const uint64_t mask[], const int32_t repeatTimes, const int32_t srcRepStride,
    bool calIndex = 0);
template <typename T>
__aicore__ inline void ReduceSum(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<T>& workLocal, const uint64_t mask[], const int32_t repeatTimes, const int32_t srcRepStride);

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
    const LocalTensor<T>& workLocal, const int32_t count, bool calIndex = 0);

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
    const LocalTensor<T>& workLocal, const int32_t count, bool calIndex = 0);

/*
 * @ingroup ReduceSum Level 2
 * @brief sum all input elements
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] workLocal LocalTensor to store the intermediate results
 * @param [in] count Number of data involved in calculation
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void ReduceSum(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<T>& workLocal, const int32_t count);
#pragma end_pipe
template <typename T>
__aicore__ inline __inout_pipe__(S) void GetReduceMaxMinCount(T &maxMinValue, T &maxMinIndex);

template <typename T>
__aicore__ inline __inout_pipe__(S) void GetReduceMaxMinCount(T &maxMinValue);

template <typename T>
__aicore__ inline __inout_pipe__(S) T GetAccVal();
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_REDUCE_INTERFACE_H
