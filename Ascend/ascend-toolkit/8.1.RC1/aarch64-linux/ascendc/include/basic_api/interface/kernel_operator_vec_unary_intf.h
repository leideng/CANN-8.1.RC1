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
 * \file kernel_operator_vec_unary_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_UNARY_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_VEC_UNARY_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_struct_unary.h"

#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif
#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_vec_unary_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_vec_unary_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_vec_unary_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_vec_unary_impl.h"
#elif defined(__DAV_M310__)
#include "dav_m310/kernel_operator_vec_unary_impl.h"
#endif

#pragma begin_pipe(V)
namespace AscendC {
/* **************************************************************************************************
 * Unary                                              *
 * ************************************************************************************************* */

/* **************************************** Relu ****************************************** */
/*
 * @ingroup Relu Level 0
 * @brief dst[i] = (src[i] < 0) ? 0 : src[i]
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Relu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);
template <typename T, bool isSetMask = true>
__aicore__ inline void Relu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup Relu Level 2
 * @brief dst[i] = (src[i] < 0) ? 0 : src[i]
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Relu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const int32_t& calCount);

/* **************************************** Exp ****************************************** */
/*
 * @ingroup Exp Level 0
 * @brief dst[i] = exp(src[i])
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Exp(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);
template <typename T, bool isSetMask = true>
__aicore__ inline void Exp(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup Exp Level 2
 * @brief dst[i] = exp(src[i])
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Exp(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const int32_t& calCount);

/* **************************************** Ln ****************************************** */
/*
 * @ingroup Ln Level 0
 * @brief dst[i] = Ln(src[i])
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Ln(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);
template <typename T, bool isSetMask = true>
__aicore__ inline void Ln(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup Ln Level 2
 * @brief dst[i] = Ln(src[i])
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Ln(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const int32_t& calCount);

/* **************************************** Abs ****************************************** */
/*
 * @ingroup Abs Level 0
 * @brief dst[i] = abs(src[i])
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Abs(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);
template <typename T, bool isSetMask = true>
__aicore__ inline void Abs(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup Abs Level 2
 * @brief dst[i] = abs(src[i])
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Abs(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const int32_t& calCount);

/* **************************************** Reciprocal ****************************************** */
/*
 * @ingroup Rec Level 0
 * @brief dst[i] = 1/src[i]
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Reciprocal(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);
template <typename T, bool isSetMask = true>
__aicore__ inline void Reciprocal(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup Rec Level 2
 * @brief dst[i] = 1/src[i]
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Reciprocal(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t& calCount);

/* **************************************** Rsqrt ****************************************** */
/*
 * @ingroup Rsqrt Level 0
 * @brief dst[i] = 1/sqrt(src[i])
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Rsqrt(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);
template <typename T, bool isSetMask = true>
__aicore__ inline void Rsqrt(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup Rsqrt Level 2
 * @brief dst[i] = 1/sqrt(src[i])
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Rsqrt(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const int32_t& calCount);

/* **************************************** Sqrt ****************************************** */
/*
 * @ingroup Sqrt Level 0
 * @brief dst[i] = src[i]^(0.5)
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Sqrt(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);
template <typename T, bool isSetMask = true>
__aicore__ inline void Sqrt(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup Sqrt Level 2
 * @brief dst[i] = src[i]^(0.5)
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Sqrt(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const int32_t& calCount);

/* **************************************** Not ****************************************** */
/*
 * @ingroup Not Level 0
 * @brief dst[i] = ~src[i]
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Not(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);
template <typename T, bool isSetMask = true>
__aicore__ inline void Not(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams);

/*
 * @ingroup Not Level 2
 * @brief dst[i] = ~src[i]
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Not(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const int32_t& calCount);
} // namespace AscendC
#pragma end_pipe
#endif // ASCENDC_MODULE_OPERATOR_VEC_UNARY_INTERFACE_H
