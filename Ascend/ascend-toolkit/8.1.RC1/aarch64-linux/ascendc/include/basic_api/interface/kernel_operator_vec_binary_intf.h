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
 * \file kernel_operator_vec_binary_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_BINARY_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_VEC_BINARY_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_struct_binary.h"

#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif

#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_vec_binary_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_vec_binary_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_vec_binary_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_vec_binary_impl.h"
#elif defined(__DAV_M310__)
#include "dav_m310/kernel_operator_vec_binary_impl.h"
#endif
#pragma begin_pipe(V)
namespace AscendC {
/* **************************************************************************************************
 * Add                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Add Level 0
 * @brief dst = src0 + src1
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Add(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                           const LocalTensor<T>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
                           const BinaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void Add(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                           const LocalTensor<T>& src1Local, uint64_t mask, const uint8_t repeatTimes,
                           const BinaryRepeatParams& repeatParams);

/*
 * @ingroup Add Level 2
 * @brief dst = src0 + src1
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Add(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                           const LocalTensor<T>& src1Local, const int32_t& calCount);

/* **************************************************************************************************
 * Sub                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Sub Level 0
 * @brief dst = src0 - src1
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Sub(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                           const LocalTensor<T>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
                           const BinaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void Sub(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                           const LocalTensor<T>& src1Local, uint64_t mask, const uint8_t repeatTimes,
                           const BinaryRepeatParams& repeatParams);

/*
 * @ingroup Sub Level 2
 * @brief dst = src0 - src1
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Sub(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                           const LocalTensor<T>& src1Local, const int32_t& calCount);

/* **************************************************************************************************
 * Mul                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Mul Level 0
 * @brief dst = src0 * src1
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Mul(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                           const LocalTensor<T>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
                           const BinaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void Mul(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                           const LocalTensor<T>& src1Local, uint64_t mask, const uint8_t repeatTimes,
                           const BinaryRepeatParams& repeatParams);

/*
 * @ingroup Mul Level 2
 * @brief dst = src0 * src1
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Mul(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                           const LocalTensor<T>& src1Local, const int32_t& calCount);

/* **************************************************************************************************
 * Div                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Div Level 0
 * @brief dst = src0 / src1
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Div(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                           const LocalTensor<T>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
                           const BinaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void Div(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                           const LocalTensor<T>& src1Local, uint64_t mask, const uint8_t repeatTimes,
                           const BinaryRepeatParams& repeatParams);

/*
 * @ingroup Div Level 2
 * @brief dst = src0 / src1
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Div(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                           const LocalTensor<T>& src1Local, const int32_t& calCount);

/* **************************************************************************************************
 * MulAddDst                                             *
 * ************************************************************************************************* */
/*
 * @ingroup MulAddDst Level 0
 * @brief dst = src0 * src1 + dst
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void MulAddDst(const LocalTensor<T>& dstLocal, const LocalTensor<U>& src0Local,
                                 const LocalTensor<U>& src1Local, const uint64_t mask[], const uint8_t repeatTimes,
                                 const BinaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void MulAddDst(const LocalTensor<T>& dstLocal, const LocalTensor<U>& src0Local,
                                 const LocalTensor<U>& src1Local, uint64_t mask, const uint8_t repeatTimes,
                                 const BinaryRepeatParams& repeatParams);

/*
 * @ingroup MulAddDst Level 2
 * @brief dst = src0 * src1 + dst
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T, typename U>
__aicore__ inline void MulAddDst(const LocalTensor<T>& dstLocal, const LocalTensor<U>& src0Local,
                                 const LocalTensor<U>& src1Local, const int32_t& calCount);

/* **************************************************************************************************
 * Max                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Max Level 0
 * @brief dst = src0 > src1 ? src0 : src1
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Max(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                           const LocalTensor<T>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
                           const BinaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void Max(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                           const LocalTensor<T>& src1Local, uint64_t mask, const uint8_t repeatTimes,
                           const BinaryRepeatParams& repeatParams);

/*
 * @ingroup Max Level 2
 * @brief dst = src0 > src1 ? src0 : src1
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Max(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                           const LocalTensor<T>& src1Local, const int32_t& calCount);

/* **************************************************************************************************
 * Min                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Min Level 0
 * @brief dst = src0 > src1 ? src1 : src0
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Min(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                           const LocalTensor<T>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
                           const BinaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void Min(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                           const LocalTensor<T>& src1Local, uint64_t mask, const uint8_t repeatTimes,
                           const BinaryRepeatParams& repeatParams);

/*
 * @ingroup Min Level 2
 * @brief dst = src0 > src1 ? src1 : src0
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Min(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                           const LocalTensor<T>& src1Local, const int32_t& calCount);

/* **************************************************************************************************
 * And                                             *
 * ************************************************************************************************* */
/*
 * @ingroup And Level 0
 * @brief dst = src0 & src1
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void And(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                           const LocalTensor<T>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
                           const BinaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void And(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                           const LocalTensor<T>& src1Local, uint64_t mask, const uint8_t repeatTimes,
                           const BinaryRepeatParams& repeatParams);

/*
 * @ingroup And Level 2
 * @brief dst = src0 & src1
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void And(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                           const LocalTensor<T>& src1Local, const int32_t& calCount);

/* **************************************************************************************************
 * Or                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Or Level 0
 * @brief dst = src0 | src1
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void Or(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                          const LocalTensor<T>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
                          const BinaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void Or(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                          const LocalTensor<T>& src1Local, uint64_t mask, const uint8_t repeatTimes,
                          const BinaryRepeatParams& repeatParams);

/*
 * @ingroup Or Level 2
 * @brief dst = src0 | src1
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Or(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                          const LocalTensor<T>& src1Local, const int32_t& calCount);

/* **************************************************************************************************
 * AddRelu                                             *
 * ************************************************************************************************* */
/*
 * @ingroup AddRelu Level 0
 * @brief dst = Relu(src0 + src1)
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void AddRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                               const LocalTensor<T>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
                               const BinaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void AddRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                               const LocalTensor<T>& src1Local, uint64_t mask, const uint8_t repeatTimes,
                               const BinaryRepeatParams& repeatParams);

/*
 * @ingroup AddRelu Level 2
 * @brief dst = Relu(src0 + src1)
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void AddRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                               const LocalTensor<T>& src1Local, const int32_t& calCount);

/* **************************************************************************************************
 * AddDeqRelu                                             *
 * ************************************************************************************************* */
/*
 * @ingroup AddDeqRelu Level 0
 * @brief dst = DeqRelu(src0 + src1)
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
template <bool isSetMask = true>
__aicore__ inline void AddDeqRelu(const LocalTensor<half>& dstLocal, const LocalTensor<int32_t>& src0Local,
                                  const LocalTensor<int32_t>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
                                  const BinaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void AddDeqRelu(const LocalTensor<T>& dstLocal, const LocalTensor<U>& src0Local,
                                  const LocalTensor<U>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
                                  const BinaryRepeatParams& repeatParams);

template <bool isSetMask = true>
__aicore__ inline void AddDeqRelu(const LocalTensor<half>& dstLocal, const LocalTensor<int32_t>& src0Local,
                                  const LocalTensor<int32_t>& src1Local, uint64_t mask, const uint8_t repeatTimes,
                                  const BinaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void AddDeqRelu(const LocalTensor<T>& dstLocal, const LocalTensor<U>& src0Local,
                                  const LocalTensor<U>& src1Local, uint64_t mask, const uint8_t repeatTimes,
                                  const BinaryRepeatParams& repeatParams);
/*
 * @ingroup AddDeqRelu Level 2
 * @brief dst = DeqRelu(src0 + src1)
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
__aicore__ inline void AddDeqRelu(const LocalTensor<half>& dstLocal, const LocalTensor<int32_t>& src0Local,
                                  const LocalTensor<int32_t>& src1Local, const int32_t& calCount);

template <typename T, typename U>
__aicore__ inline void AddDeqRelu(const LocalTensor<T>& dstLocal, const LocalTensor<U>& src0Local,
                                  const LocalTensor<U>& src1Local, const int32_t& calCount);

/* **************************************************************************************************
 * FusedMulAdd                                             *
 * ************************************************************************************************* */
/*
 * @ingroup FusedMulAdd Level 0
 * @brief dst = src0 * dst + src1
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void FusedMulAdd(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                                   const LocalTensor<T>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
                                   const BinaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void FusedMulAdd(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                                   const LocalTensor<T>& src1Local, uint64_t mask, const uint8_t repeatTimes,
                                   const BinaryRepeatParams& repeatParams);

/*
 * @ingroup FusedMulAdd Level 2
 * @brief dst = src0 * dst + src1
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void FusedMulAdd(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                                   const LocalTensor<T>& src1Local, const int32_t& calCount);

/* **************************************************************************************************
 * FusedMulAddRelu                                             *
 * ************************************************************************************************* */
/*
 * @ingroup FusedMulAddRelu Level 0
 * @brief dst = relu(src0 * dst + src1)
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void FusedMulAddRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                                       const LocalTensor<T>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
                                       const BinaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void FusedMulAddRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                                       const LocalTensor<T>& src1Local, uint64_t mask, const uint8_t repeatTimes,
                                       const BinaryRepeatParams& repeatParams);

/*
 * @ingroup FusedMulAddRelu Level 2
 * @brief dst = relu(src0 * dst + src1)
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void FusedMulAddRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                                       const LocalTensor<T>& src1Local, const int32_t& calCount);

/* **************************************************************************************************
 * SubRelu                                             *
 * ************************************************************************************************* */
/*
 * @ingroup SubRelu Level 0
 * @brief dst = Relu(src0 - src1)
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask = true>
__aicore__ inline void SubRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                               const LocalTensor<T>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
                               const BinaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void SubRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                               const LocalTensor<T>& src1Local, uint64_t mask, const uint8_t repeatTimes,
                               const BinaryRepeatParams& repeatParams);

template <typename T>
__aicore__ inline void SubRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
                               const LocalTensor<T>& src1Local, const int32_t& calCount);
}  // namespace AscendC
#pragma end_pipe
#endif  // ASCENDC_MODULE_OPERATOR_VEC_BINARY_INTERFACE_H