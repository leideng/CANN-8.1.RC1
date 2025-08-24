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
 * \file inner_kernel_operator_vec_unary_intf.cppm
 * \brief
 */
#ifndef ASCENDC_MODULE_INNER_OPERATOR_VEC_UNARY_INTERFACE_H
#define ASCENDC_MODULE_INNER_OPERATOR_VEC_UNARY_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_check.h"
#include "kernel_struct_unary.h"

#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_vec_unary_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_vec_unary_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_vec_unary_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_vec_unary_impl.h"
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
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.srcBlkStride src block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void Relu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, static_cast<PrimType>(0), mask, repeatTimes, repeatParams,
        "Relu")) {
        ASCENDC_REPORT_CHECK_ERROR("Relu", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    ReluImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        mask, repeatTimes, repeatParams);
}
template <typename T, bool isSetMask>
__aicore__ inline void Relu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, static_cast<PrimType>(0), mask, repeatTimes, repeatParams,
        "Relu")) {
        ASCENDC_REPORT_CHECK_ERROR("Relu", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    ReluImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        mask, repeatTimes, repeatParams);
}

/*
 * @ingroup Relu Level 2
 * @brief dst[i] = (src[i] < 0) ? 0 : src[i]
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Relu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const int32_t& calCount)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, static_cast<PrimType>(0), calCount, "Relu")) {
        ASCENDC_REPORT_CHECK_ERROR("Relu", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    ReluImpl((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(), calCount);
}

/* **************************************** Exp ****************************************** */
/*
 * @ingroup Exp Level 0
 * @brief dst[i] = exp(src[i])
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.srcBlkStride src block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void Exp(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, static_cast<PrimType>(0), mask, repeatTimes, repeatParams,
        "Exp")) {
        ASCENDC_REPORT_CHECK_ERROR("Exp", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    ExpImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        mask, repeatTimes, repeatParams);
}
template <typename T, bool isSetMask>
__aicore__ inline void Exp(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, static_cast<PrimType>(0), mask, repeatTimes, repeatParams,
        "Exp")) {
        ASCENDC_REPORT_CHECK_ERROR("Exp", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    ExpImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        mask, repeatTimes, repeatParams);
}

/*
 * @ingroup Exp Level 2
 * @brief dst[i] = exp(src[i])
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Exp(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const int32_t& calCount)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, static_cast<PrimType>(0), calCount, "Exp")) {
        ASCENDC_REPORT_CHECK_ERROR("Exp", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    ExpImpl((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(), calCount);
}

/* **************************************** Ln ****************************************** */
/*
 * @ingroup Ln Level 0
 * @brief dst[i] = Ln(src[i])
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.srcBlkStride src block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void Ln(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, static_cast<PrimType>(0), mask, repeatTimes, repeatParams, "Ln")) {
        ASCENDC_REPORT_CHECK_ERROR("Ln", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    LnImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        mask, repeatTimes, repeatParams);
}
template <typename T, bool isSetMask>
__aicore__ inline void Ln(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, static_cast<PrimType>(0), mask, repeatTimes, repeatParams, "Ln")) {
        ASCENDC_REPORT_CHECK_ERROR("Ln", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    LnImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        mask, repeatTimes, repeatParams);
}

/*
 * @ingroup Ln Level 2
 * @brief dst[i] = Ln(src[i])
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Ln(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const int32_t& calCount)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, static_cast<PrimType>(0), calCount, "Ln")) {
        ASCENDC_REPORT_CHECK_ERROR("Ln", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    LnImpl((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(), calCount);
}

/* **************************************** Abs ****************************************** */
/*
 * @ingroup Abs Level 0
 * @brief dst[i] = abs(src[i])
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.srcBlkStride src block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void Abs(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, static_cast<PrimType>(0), mask, repeatTimes, repeatParams,
        "Abs")) {
        ASCENDC_REPORT_CHECK_ERROR("Abs", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    AbsImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        mask, repeatTimes, repeatParams);
}
template <typename T, bool isSetMask>
__aicore__ inline void Abs(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, static_cast<PrimType>(0), mask, repeatTimes, repeatParams,
        "Abs")) {
        ASCENDC_REPORT_CHECK_ERROR("Abs", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    AbsImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        mask, repeatTimes, repeatParams);
}

/*
 * @ingroup Abs Level 2
 * @brief dst[i] = abs(src[i])
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Abs(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const int32_t& calCount)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, static_cast<PrimType>(0), calCount, "Abs")) {
        ASCENDC_REPORT_CHECK_ERROR("Abs", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    AbsImpl((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(), calCount);
}

/* **************************************** Reciprocal ****************************************** */
/*
 * @ingroup Rec Level 0
 * @brief dst[i] = 1/src[i]
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.srcBlkStride src block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void Reciprocal(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, static_cast<PrimType>(0), mask, repeatTimes, repeatParams,
        "Reciprocal")) {
        ASCENDC_REPORT_CHECK_ERROR("Reciprocal", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    ReciprocalImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(),
        (__ubuf__ PrimType*)srcLocal.GetPhyAddr(), mask, repeatTimes, repeatParams);
}
template <typename T, bool isSetMask>
__aicore__ inline void Reciprocal(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, static_cast<PrimType>(0), mask, repeatTimes, repeatParams,
        "Reciprocal")) {
        ASCENDC_REPORT_CHECK_ERROR("Reciprocal", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    ReciprocalImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(),
        (__ubuf__ PrimType*)srcLocal.GetPhyAddr(), mask, repeatTimes, repeatParams);
}

/*
 * @ingroup Rec Level 2
 * @brief dst[i] = 1/src[i]
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Reciprocal(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t& calCount)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, static_cast<PrimType>(0), calCount, "Reciprocal")) {
        ASCENDC_REPORT_CHECK_ERROR("Reciprocal", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    ReciprocalImpl((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(), calCount);
}

/* **************************************** Rsqrt ****************************************** */
/*
 * @ingroup Rsqrt Level 0
 * @brief dst[i] = 1/sqrt(src[i])
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.srcBlkStride src block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void Rsqrt(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, static_cast<PrimType>(0), mask, repeatTimes, repeatParams,
        "Rsqrt")) {
        ASCENDC_REPORT_CHECK_ERROR("Rsqrt", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    RsqrtImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        mask, repeatTimes, repeatParams);
}
template <typename T, bool isSetMask>
__aicore__ inline void Rsqrt(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, static_cast<PrimType>(0), mask, repeatTimes, repeatParams,
        "Rsqrt")) {
        ASCENDC_REPORT_CHECK_ERROR("Rsqrt", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    RsqrtImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        mask, repeatTimes, repeatParams);
}

/*
 * @ingroup Rsqrt Level 2
 * @brief dst[i] = 1/sqrt(src[i])
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Rsqrt(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const int32_t& calCount)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, static_cast<PrimType>(0), calCount, "Rsqrt")) {
        ASCENDC_REPORT_CHECK_ERROR("Rsqrt", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    RsqrtImpl((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(), calCount);
}

/* **************************************** Sqrt ****************************************** */
/*
 * @ingroup Sqrt Level 0
 * @brief dst[i] = src[i]^(0.5)
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.srcBlkStride src block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void Sqrt(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, static_cast<PrimType>(0), mask, repeatTimes, repeatParams,
        "Sqrt")) {
        ASCENDC_REPORT_CHECK_ERROR("Sqrt", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    SqrtImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        mask, repeatTimes, repeatParams);
}
template <typename T, bool isSetMask>
__aicore__ inline void Sqrt(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, static_cast<PrimType>(0), mask, repeatTimes, repeatParams,
        "Sqrt")) {
        ASCENDC_REPORT_CHECK_ERROR("Sqrt", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    SqrtImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        mask, repeatTimes, repeatParams);
}

/*
 * @ingroup Sqrt Level 2
 * @brief dst[i] = src[i]^(0.5)
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Sqrt(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const int32_t& calCount)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, static_cast<PrimType>(0), calCount, "Sqrt")) {
        ASCENDC_REPORT_CHECK_ERROR("Sqrt", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    SqrtImpl((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(), calCount);
}

/* **************************************** Not ****************************************** */
/*
 * @ingroup Not Level 0
 * @brief dst[i] = ~src[i]
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.srcBlkStride src block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void Not(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask[],
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, static_cast<PrimType>(0), mask, repeatTimes, repeatParams,
        "Not")) {
        ASCENDC_REPORT_CHECK_ERROR("Not", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    NotImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        mask, repeatTimes, repeatParams);
}
template <typename T, bool isSetMask>
__aicore__ inline void Not(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, uint64_t mask,
    const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, static_cast<PrimType>(0), mask, repeatTimes, repeatParams,
        "Not")) {
        ASCENDC_REPORT_CHECK_ERROR("Not", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    NotImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(),
        mask, repeatTimes, repeatParams);
}

/*
 * @ingroup Not Level 2
 * @brief dst[i] = ~src[i]
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void Not(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const int32_t& calCount)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    if (!CheckFunVecBinaryScalar(dstLocal, srcLocal, static_cast<PrimType>(0), calCount, "Not")) {
        ASCENDC_REPORT_CHECK_ERROR("Not", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    NotImpl((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)srcLocal.GetPhyAddr(), calCount);
}
} // namespace AscendC
#pragma end_pipe
#endif // ASCENDC_MODULE_INNER_OPERATOR_VEC_UNARY_INTERFACE_H
