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
 * \file inner_kernel_operator_vec_binary_scalar_intf.cppm
 * \brief
 */
#ifndef ASCENDC_MODULE_INNER_OPERATOR_VEC_BINARY_SCALAR_INTERFACE_H
#define ASCENDC_MODULE_INNER_OPERATOR_VEC_BINARY_SCALAR_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_check.h"
#include "kernel_struct_unary.h"

#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_vec_binary_scalar_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_vec_binary_scalar_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_vec_binary_scalar_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_vec_binary_scalar_impl.h"
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
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.srcBlkStride src block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void Adds(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "Adds")) {
        ASCENDC_REPORT_CHECK_ERROR("Adds", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    AddsImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), scalarValue, mask,
        repeatTimes, repeatParams);
}

template <typename T, typename U, bool isSetMask, typename std::enable_if<IsSameType< PrimT<T>, U>::value, bool>::type>
__aicore__ inline void Adds(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "Adds")) {
        ASCENDC_REPORT_CHECK_ERROR("Adds", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    AddsImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        scalarValue, mask, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask>
__aicore__ inline void Adds(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "Adds")) {
        ASCENDC_REPORT_CHECK_ERROR("Adds", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    AddsImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), scalarValue, mask,
        repeatTimes, repeatParams);
}

template <typename T, typename U, bool isSetMask, typename std::enable_if<IsSameType< PrimT<T>, U>::value, bool>::type>
__aicore__ inline void Adds(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "Adds")) {
        ASCENDC_REPORT_CHECK_ERROR("Adds", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    AddsImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        scalarValue, mask, repeatTimes, repeatParams);
}

/*
 * @ingroup Adds Level 2
 * @brief dst = src[i] + sacalar
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T, bool isSetMask>
__aicore__ inline void Adds(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    const int32_t& calCount)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, calCount, "Adds")) {
        ASCENDC_REPORT_CHECK_ERROR("Adds", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    AddsImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), scalarValue,
        calCount);
}

template <typename T, typename U, bool isSetMask, typename std::enable_if<IsSameType< PrimT<T>, U>::value, bool>::type>
__aicore__ inline void Adds(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    const int32_t& calCount)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, calCount, "Adds")) {
        ASCENDC_REPORT_CHECK_ERROR("Adds", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    AddsImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        scalarValue, calCount);
}

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
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.srcBlkStride src block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void Muls(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "Muls")) {
        ASCENDC_REPORT_CHECK_ERROR("Muls", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    MulsImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), scalarValue, mask,
        repeatTimes, repeatParams);
}

template <typename T, typename U, bool isSetMask, typename std::enable_if<IsSameType< PrimT<T>, U>::value, bool>::type>
__aicore__ inline void Muls(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "Muls")) {
        ASCENDC_REPORT_CHECK_ERROR("Muls", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    MulsImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        scalarValue, mask, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask>
__aicore__ inline void Muls(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "Muls")) {
        ASCENDC_REPORT_CHECK_ERROR("Muls", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    MulsImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), scalarValue, mask,
        repeatTimes, repeatParams);
}

template <typename T, typename U, bool isSetMask, typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type>
__aicore__ inline void Muls(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "Muls")) {
        ASCENDC_REPORT_CHECK_ERROR("Muls", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    MulsImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), scalarValue, mask,
        repeatTimes, repeatParams);
}

/*
 * @ingroup Muls Level 2
 * @brief dst = src[i] * sacalar
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T, bool isSetMask>
__aicore__ inline void Muls(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    const int32_t& calCount)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, calCount, "Muls")) {
        ASCENDC_REPORT_CHECK_ERROR("Muls", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    MulsImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), scalarValue,
        calCount);
}

template <typename T, typename U, bool isSetMask, typename std::enable_if<IsSameType< PrimT<T>, U>::value, bool>::type>
__aicore__ inline void Muls(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    const int32_t& calCount)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, calCount, "Muls")) {
        ASCENDC_REPORT_CHECK_ERROR("Muls", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    MulsImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        scalarValue, calCount);
}

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
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.srcBlkStride src block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void Maxs(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "Maxs")) {
        ASCENDC_REPORT_CHECK_ERROR("Maxs", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    MaxsImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), scalarValue, mask,
        repeatTimes, repeatParams);
}

template <typename T, typename U, bool isSetMask, typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type>
__aicore__ inline void Maxs(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "Maxs")) {
        ASCENDC_REPORT_CHECK_ERROR("Maxs", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    MaxsImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        scalarValue, mask, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask>
__aicore__ inline void Maxs(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "Maxs")) {
        ASCENDC_REPORT_CHECK_ERROR("Maxs", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    MaxsImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), scalarValue, mask,
        repeatTimes, repeatParams);
}

template <typename T, typename U, bool isSetMask, typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type>
__aicore__ inline void Maxs(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "Maxs")) {
        ASCENDC_REPORT_CHECK_ERROR("Maxs", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    MaxsImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        scalarValue, mask, repeatTimes, repeatParams);
}

/*
 * @ingroup Maxs Level 2
 * @brief dst = src[i] > sacalar ? src[0] : scalar
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T, bool isSetMask>
__aicore__ inline void Maxs(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    const int32_t& calCount)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, calCount, "Maxs")) {
        ASCENDC_REPORT_CHECK_ERROR("Maxs", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    MaxsImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), scalarValue,
        calCount);
}

template <typename T, typename U, bool isSetMask, typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type>
__aicore__ inline void Maxs(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    const int32_t& calCount)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, calCount, "Maxs")) {
        ASCENDC_REPORT_CHECK_ERROR("Maxs", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    MaxsImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        scalarValue, calCount);
}

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
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.srcBlkStride src block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void Mins(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "Mins")) {
        ASCENDC_REPORT_CHECK_ERROR("Mins", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    MinsImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), scalarValue, mask,
        repeatTimes, repeatParams);
}

template <typename T, typename U, bool isSetMask, typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type>
__aicore__ inline void Mins(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "Mins")) {
        ASCENDC_REPORT_CHECK_ERROR("Mins", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    MinsImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        scalarValue, mask, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask>
__aicore__ inline void Mins(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "Mins")) {
        ASCENDC_REPORT_CHECK_ERROR("Mins", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    MinsImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), scalarValue, mask,
        repeatTimes, repeatParams);
}

template <typename T, typename U, bool isSetMask, typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type>
__aicore__ inline void Mins(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "Mins")) {
        ASCENDC_REPORT_CHECK_ERROR("Mins", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    MinsImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        scalarValue, mask, repeatTimes, repeatParams);
}

/*
 * @ingroup Mins Level 2
 * @brief dst = src[i] < sacalar ? src[0] : scalar
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T, bool isSetMask>
__aicore__ inline void Mins(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    const int32_t& calCount)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, calCount, "Mins")) {
        ASCENDC_REPORT_CHECK_ERROR("Mins", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    MinsImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), scalarValue,
        calCount);
}

template <typename T, typename U, bool isSetMask, typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type>
__aicore__ inline void Mins(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    const int32_t& calCount)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, calCount, "Mins")) {
        ASCENDC_REPORT_CHECK_ERROR("Mins", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    MinsImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        scalarValue, calCount);
}

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
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.srcBlkStride src block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void ShiftLeft(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "ShiftLeft")) {
        ASCENDC_REPORT_CHECK_ERROR("ShiftLeft", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    ShiftLeftImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), scalarValue,
        mask, repeatTimes, repeatParams);
}

template <typename T, typename U, bool isSetMask, typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type >
__aicore__ inline void ShiftLeft(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "ShiftLeft")) {
        ASCENDC_REPORT_CHECK_ERROR("ShiftLeft", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    ShiftLeftImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(),
        (__ubuf__ PrimType*)srcLocal.GetPhyAddr(), scalarValue, mask, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask>
__aicore__ inline void ShiftLeft(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "ShiftLeft")) {
        ASCENDC_REPORT_CHECK_ERROR("ShiftLeft", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    ShiftLeftImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), scalarValue,
        mask, repeatTimes, repeatParams);
}

template <typename T, typename U, bool isSetMask, typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type>
__aicore__ inline void ShiftLeft(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "ShiftLeft")) {
        ASCENDC_REPORT_CHECK_ERROR("ShiftLeft", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    ShiftLeftImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(),
        (__ubuf__ PrimType*)srcLocal.GetPhyAddr(), scalarValue, mask, repeatTimes, repeatParams);
}

/*
 * @ingroup Shiftleft Level 2
 * @brief dst = src[i] << sacalar
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T, bool isSetMask>
__aicore__ inline void ShiftLeft(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    const int32_t& calCount)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, calCount, "ShiftLeft")) {
        ASCENDC_REPORT_CHECK_ERROR("ShiftLeft", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    ShiftLeftImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), scalarValue,
        calCount);
}

template <typename T, typename U, bool isSetMask, typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type>
__aicore__ inline void ShiftLeft(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    const int32_t& calCount)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, calCount, "ShiftLeft")) {
        ASCENDC_REPORT_CHECK_ERROR("ShiftLeft", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    ShiftLeftImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(),
        (__ubuf__ PrimType*)srcLocal.GetPhyAddr(), scalarValue, calCount);
}

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
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.srcBlkStride src block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void ShiftRight(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams, bool roundEn)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "ShiftRight")) {
        ASCENDC_REPORT_CHECK_ERROR("ShiftRight", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    ShiftRightImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), scalarValue,
        mask, repeatTimes, repeatParams, roundEn);
}

template <typename T, typename U, bool isSetMask, typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type>
__aicore__ inline void ShiftRight(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams, bool roundEn)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "ShiftRight")) {
        ASCENDC_REPORT_CHECK_ERROR("ShiftRight", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    ShiftRightImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(),
        (__ubuf__ PrimType*)srcLocal.GetPhyAddr(), scalarValue, mask, repeatTimes, repeatParams, roundEn);
}

template <typename T, bool isSetMask>
__aicore__ inline void ShiftRight(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams, bool roundEn)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "ShiftRight")) {
        ASCENDC_REPORT_CHECK_ERROR("ShiftRight", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    ShiftRightImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), scalarValue,
        mask, repeatTimes, repeatParams, roundEn);
}

template <typename T, typename U, bool isSetMask, typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type>
__aicore__ inline void ShiftRight(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams, bool roundEn)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "ShiftRight")) {
        ASCENDC_REPORT_CHECK_ERROR("ShiftRight", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    ShiftRightImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(),
        (__ubuf__ PrimType*)srcLocal.GetPhyAddr(), scalarValue, mask, repeatTimes, repeatParams, roundEn);
}

/*
 * @ingroup ShiftRight Level 2
 * @brief dst = src[i] >> sacalar
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T, bool isSetMask>
__aicore__ inline void ShiftRight(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    const int32_t& calCount)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, calCount, "ShiftRight")) {
        ASCENDC_REPORT_CHECK_ERROR("ShiftRight", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    ShiftRightImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), scalarValue,
        calCount);
}

template <typename T, typename U, bool isSetMask, typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type>
__aicore__ inline void ShiftRight(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    const int32_t& calCount)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, calCount, "ShiftRight")) {
        ASCENDC_REPORT_CHECK_ERROR("ShiftRight", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    ShiftRightImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(),
        (__ubuf__ PrimType*)srcLocal.GetPhyAddr(), scalarValue, calCount);
}

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
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.srcBlkStride src block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void LeakyRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "LeakyRelu")) {
        ASCENDC_REPORT_CHECK_ERROR("LeakyRelu", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    LeakyReluImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), scalarValue,
        mask, repeatTimes, repeatParams);
}

template < typename T, typename U, bool isSetMask, typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type >
__aicore__ inline void LeakyRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "LeakyRelu")) {
        ASCENDC_REPORT_CHECK_ERROR("LeakyRelu", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    LeakyReluImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(),
        (__ubuf__ PrimType*)srcLocal.GetPhyAddr(), scalarValue, mask, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask>
__aicore__ inline void LeakyRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "LeakyRelu")) {
        ASCENDC_REPORT_CHECK_ERROR("LeakyRelu", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    LeakyReluImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), scalarValue,
        mask, repeatTimes, repeatParams);
}

template <typename T, typename U, bool isSetMask, typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type>
__aicore__ inline void LeakyRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, mask, repeatTimes, repeatParams, "LeakyRelu")) {
        ASCENDC_REPORT_CHECK_ERROR("LeakyRelu", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    LeakyReluImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(),
        (__ubuf__ PrimType*)srcLocal.GetPhyAddr(), scalarValue, mask, repeatTimes, repeatParams);
}

/*
 * @ingroup LeakyRelu Level 2
 * @brief dst = src[i] < 0 ? (scalar * src[i]) : src[i]
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] scalar input scalar number
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T, bool isSetMask>
__aicore__ inline void LeakyRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue,
    const int32_t& calCount)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, calCount, "LeakyRelu")) {
        ASCENDC_REPORT_CHECK_ERROR("LeakyRelu", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    LeakyReluImpl<T, isSetMask>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), scalarValue,
        calCount);
}

template < typename T, typename U, bool isSetMask, typename std::enable_if<IsSameType<PrimT<T>, U>::value, bool>::type >
__aicore__ inline void LeakyRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const U& scalarValue,
    const int32_t& calCount)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, scalarValue, calCount, "LeakyRelu")) {
        ASCENDC_REPORT_CHECK_ERROR("LeakyRelu", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    LeakyReluImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(),
        (__ubuf__ PrimType*)srcLocal.GetPhyAddr(), scalarValue, calCount);
}
} // namespace AscendC
#pragma end_pipe
#endif // ASCENDC_MODULE_INNER_OPERATOR_VEC_BINARY_SCALAR_INTERFACE_H
