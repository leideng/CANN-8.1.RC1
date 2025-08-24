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
 * \file kernel_operator_vec_cmpsel_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_CMPSEL_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_VEC_CMPSEL_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_struct_binary.h"
#include "kernel_struct_unary.h"

#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif
#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_vec_cmpsel_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_vec_cmpsel_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_vec_cmpsel_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_vec_cmpsel_impl.h"
#elif defined(__DAV_M310__)
#include "dav_m310/kernel_operator_vec_cmpsel_impl.h"
#endif

#pragma begin_pipe(V)
namespace AscendC {
/* **************************************************************************************************
 * Compare                                           *
 * ************************************************************************************************* */
/*
 * @ingroup Compare Level 0
 * @brief Compare the size of two tensors one by one. If true, the corresponding bit is 1, otherwise it is 0
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] cmpMode compare mode
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
__aicore__ inline void Compare(const LocalTensor<U>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, CMPMODE cmpMode, const uint64_t mask[], uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void Compare(const LocalTensor<U>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, CMPMODE cmpMode, const uint64_t mask, uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void Compare(const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, CMPMODE cmpMode, const uint64_t mask[],
    const BinaryRepeatParams& repeatParams);

template <typename T, bool isSetMask = true>
__aicore__ inline void Compare(const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, CMPMODE cmpMode, const uint64_t mask,
    const BinaryRepeatParams& repeatParams);

/*
 * @ingroup Compare Level 2
 * @brief Compare the size of two tensors one by one. If true, the corresponding bit is 1, otherwise it is 0
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] cmpMode compare mode
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T, typename U>
__aicore__ inline void Compare(const LocalTensor<U>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, CMPMODE cmpMode, uint32_t calCount);

template <typename T>
__aicore__ inline void GetCmpMask(const LocalTensor<T>& dst);

template <typename T>
__aicore__ inline void SetCmpMask(const LocalTensor<T>& src);

/* **************************************************************************************************
 * Compare                                           *
 * ************************************************************************************************* */
/*
 * @ingroup Compare Level 0
 * @brief Compare the size of a tensor and a scalar one by one. If true, the corresponding bit is 1, otherwise it is 0
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Scalar input Scalar
 * @param [in] cmpMode compare mode
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.srcBlkStride src0 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.srcRepStride src0 repeat stride
 */
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void CompareScalar(const LocalTensor<U>& dstLocal, const LocalTensor<T>& src0Local,
    const T src1Scalar, CMPMODE cmpMode, const uint64_t mask[], uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams);

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void CompareScalar(const LocalTensor<U>& dstLocal, const LocalTensor<T>& src0Local,
    const T src1Scalar, CMPMODE cmpMode, const uint64_t mask, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams);

/*
 * @ingroup CompareScalar Level 2
 * @brief CompareScalar the size of two tensors one by one. If true, the corresponding bit is 1, otherwise it is 0
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Scalar input Scalar
 * @param [in] cmpMode compare mode
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T, typename U>
__aicore__ inline void CompareScalar(const LocalTensor<U>& dstLocal, const LocalTensor<T>& src0Local,
    const T src1Scalar, CMPMODE cmpMode, uint32_t calCount);

/* **************************************************************************************************
 * Select                                            *
 * ************************************************************************************************* */
// T must be half or Float
// U must be uint8_t

// ================================
/*
 * @ingroup Select Level 0
 * @brief Select element according to the bit value of sel
 * @param [out] dstLocal output LocalTensor
 * @param [in] selMask select mask LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] selMode select mode
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
// select mode: 0/1/2
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void Select(const LocalTensor<T>& dstLocal, const LocalTensor<U>& selMask,
    const LocalTensor<T>& src0Local, const LocalTensor<T>& src1Local, SELMODE selMode, uint64_t mask[],
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams);

// select mode: 0/1/2
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void Select(const LocalTensor<T>& dstLocal, const LocalTensor<U>& selMask,
    const LocalTensor<T>& src0Local, const LocalTensor<T>& src1Local, SELMODE selMode, uint64_t mask,
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams);

template <typename T, SELMODE selMode>
__aicore__ inline void Select(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    SelectCal<T, selMode>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)src0Local.GetPhyAddr(),
        (__ubuf__ T*)src1Local.GetPhyAddr(), repeatTimes, repeatParams);
}

template <typename T, typename U>
__aicore__ inline void Select(const LocalTensor<T>& dstLocal, const LocalTensor<U>& selMask,
    const LocalTensor<T>& src0Local, uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    SelectCal<T, U>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ U*)selMask.GetPhyAddr(),
        (__ubuf__ T*)src0Local.GetPhyAddr(), repeatTimes, repeatParams);
}

/*
 * @ingroup Select Level 2
 * @brief Select element according to the bit value of sel
 * @param [out] dstLocal output LocalTensor
 * @param [in] selMask select mask LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] selMode select mode
 * @param [in] calcount number Number of data involved in calculation
 */
// select mode: 0/1/2
template <typename T, typename U>
__aicore__ inline void Select(const LocalTensor<T>& dstLocal, const LocalTensor<U>& selMask,
    const LocalTensor<T>& src0Local, const LocalTensor<T>& src1Local, SELMODE selMode, uint32_t calCount);

// ================================
/*
 * @ingroup Select Level 0
 * @brief Select element according to the bit value of sel
 * @param [out] dstLocal output LocalTensor
 * @param [in] selMask select mask LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input number
 * @param [in] selMode select mode
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] repeatParams.dstBlkStride dst block stride
 * @param [in] repeatParams.src0BlkStride src0 block stride
 * @param [in] repeatParams.src1BlkStride src1 block stride
 * @param [in] repeatParams.dstRepStride dst repeat stride
 * @param [in] repeatParams.src0RepStride src0 repeat stride
 * @param [in] repeatParams.src1RepStride src1 repeat stride
 */
// select mode: 1
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void Select(const LocalTensor<T>& dstLocal, const LocalTensor<U>& selMask,
    const LocalTensor<T>& src0Local, T src1Local, SELMODE selMode, uint64_t mask[], uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams);

// select mode: 1
template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void Select(const LocalTensor<T>& dstLocal, const LocalTensor<U>& selMask,
    const LocalTensor<T>& src0Local, T src1Local, SELMODE selMode, uint64_t mask, uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams);

// select mode: 1
/*
 * @ingroup Select Level 2
 * @brief Select element according to the bit value of sel
 * @param [out] dstLocal output LocalTensor
 * @param [in] selMask select mask LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input number
 * @param [in] selMode select mode
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T, typename U>
__aicore__ inline void Select(const LocalTensor<T>& dstLocal, const LocalTensor<U>& selMask,
    const LocalTensor<T>& src0Local, T src1Local, SELMODE selMode, uint32_t calCount);
} // namespace AscendC
#pragma end_pipe
#endif // ASCENDC_MODULE_OPERATOR_VEC_CMPSEL_INTERFACE_H
