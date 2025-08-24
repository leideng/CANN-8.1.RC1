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
 * \file kernel_operator_vec_binary_scalar_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_BINARY_SCALAR_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_VEC_BINARY_SCALAR_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_struct_unary.h"

#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif
#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_vec_binary_scalar_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_vec_binary_scalar_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_vec_binary_scalar_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_vec_binary_scalar_impl.h"
#elif defined(__DAV_M310__)
#include "dav_m310/kernel_operator_vec_binary_scalar_impl.h"
#endif
#pragma begin_pipe(V)
namespace AscendC {
/* **************************************************************************************************
 * Adds                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Adds Level 0
 * @brief dst[i] = src[i] + sacalar
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Adds(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true,
    typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Adds(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void Adds(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true,
    typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Adds(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup Adds Level 2
 * @brief dst = src[i] + sacalar
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Adds(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    const int32_t& calCount);

template <typename T, typename U, bool isSetMask = true,
    typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Adds(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    const int32_t& calCount);

/* **************************************************************************************************
 * Muls                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Muls Level 0
 * @brief dst[i] = src[i] * sacalar
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Muls(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true,
    typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Muls(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void Muls(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true,
    typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Muls(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup Muls Level 2
 * @brief dst = src[i] * sacalar
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Muls(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    const int32_t& calCount);

template <typename T, typename U, bool isSetMask = true,
    typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Muls(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    const int32_t& calCount);

/* **************************************************************************************************
 * Maxs                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Maxs Level 0
 * @brief dst[i] = src[i] > sacalar ? src[0] : scalar
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Maxs(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true,
    typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Maxs(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void Maxs(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true,
    typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Maxs(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup Maxs Level 2
 * @brief dst = src[i] > sacalar ? src[0] : scalar
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Maxs(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    const int32_t& calCount);

template <typename T, typename U, bool isSetMask = true,
    typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Maxs(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    const int32_t& calCount);

/* **************************************************************************************************
 * Mins                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Mins Level 0
 * @brief dst[i] = src[i] < sacalar ? src[0] : scalar
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Mins(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true,
    typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Mins(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void Mins(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true,
    typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Mins(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup Mins Level 2
 * @brief dst = src[i] < sacalar ? src[0] : scalar
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Mins(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    const int32_t& calCount);

template <typename T, typename U, bool isSetMask = true,
    typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Mins(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    const int32_t& calCount);

/* **************************************************************************************************
 * ShiftLeft                                             *
 * ************************************************************************************************* */
/*
 * @ingroup ShiftLeft Level 0
 * @brief dst[i] = src[i] << sacalar
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftLeft(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true,
    typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void ShiftLeft(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftLeft(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true,
    typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void ShiftLeft(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup Shiftleft Level 2
 * @brief dst = src[i] << sacalar
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftLeft(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    const int32_t& calCount);

template <typename T, typename U, bool isSetMask = true,
    typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void ShiftLeft(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    const int32_t& calCount);

/* **************************************************************************************************
 * ShiftRight                                             *
 * ************************************************************************************************* */
/*
 * @ingroup ShiftRight Level 0
 * @brief dst[i] = src[i] >> sacalar
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftRight(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams, bool roundEn = false);

template <typename T, typename U, bool isSetMask = true,
    typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void ShiftRight(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams, bool roundEn);

template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftRight(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams, bool roundEn = false);

template <typename T, typename U, bool isSetMask = true,
    typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void ShiftRight(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams, bool roundEn);

/*
 * @ingroup ShiftRight Level 2
 * @brief dst = src[i] >> sacalar
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftRight(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    const int32_t& calCount);

template <typename T, typename U, bool isSetMask = true,
    typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void ShiftRight(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    const int32_t& calCount);

/* **************************************************************************************************
 * LeakyRelu                                             *
 * ************************************************************************************************* */
/*
 * @ingroup LeakyRelu Level 0
 * @brief dst[i] = src[i] < 0 ? (scalar * src[i]) : src[i]
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void LeakyRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true,
    typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void LeakyRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void LeakyRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true,
    typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void LeakyRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup LeakyRelu Level 2
 * @brief dst = src[i] < 0 ? (scalar * src[i]) : src[i]
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void LeakyRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    const int32_t& calCount);

template <typename T, typename U, bool isSetMask = true,
    typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void LeakyRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    const int32_t& calCount);
} // namespace AscendC
#pragma end_pipe
#endif // ASCENDC_MODULE_OPERATOR_VEC_BINARY_SCALAR_INTERFACE_H
