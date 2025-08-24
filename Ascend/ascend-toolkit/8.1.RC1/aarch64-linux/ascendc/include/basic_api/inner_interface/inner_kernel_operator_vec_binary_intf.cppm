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
 * \file inner_kernel_operator_vec_binary_intf.cppm
 * \brief
 */
#ifndef ASCENDC_MODULE_INNER_OPERATOR_VEC_BINARY_INTERFACE_H
#define ASCENDC_MODULE_INNER_OPERATOR_VEC_BINARY_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_check.h"
#include "kernel_struct_binary.h"

#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_vec_binary_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_vec_binary_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_vec_binary_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_vec_binary_impl.h"
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
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.src0BlkStride src0 block stride
 * @param [in] intriParams.src1BlkStride src1 block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src0 repeat stride
 * @param [in] intriParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void Add(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "Add")) {
        ASCENDC_REPORT_CHECK_ERROR("Add", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    AddImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimType*)src1Local.GetPhyAddr(), mask, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask>
__aicore__ inline void Add(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "Add")) {
        ASCENDC_REPORT_CHECK_ERROR("Add", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    AddImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimType*)src1Local.GetPhyAddr(), mask, repeatTimes, repeatParams);
}

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
    const LocalTensor<T>& src1Local, const int32_t& calCount)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, calCount, "Add")) {
        ASCENDC_REPORT_CHECK_ERROR("Add", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    AddImpl((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimType*)src1Local.GetPhyAddr(), calCount);
}

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
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.src0BlkStride src0 block stride
 * @param [in] intriParams.src1BlkStride src1 block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src0 repeat stride
 * @param [in] intriParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void Sub(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "Sub")) {
        ASCENDC_REPORT_CHECK_ERROR("Sub", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    SubImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimType*)src1Local.GetPhyAddr(), mask, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask>
__aicore__ inline void Sub(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "Sub")) {
        ASCENDC_REPORT_CHECK_ERROR("Sub", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    SubImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimType*)src1Local.GetPhyAddr(), mask, repeatTimes, repeatParams);
}

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
    const LocalTensor<T>& src1Local, const int32_t& calCount)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, calCount, "Sub")) {
        ASCENDC_REPORT_CHECK_ERROR("Sub", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    SubImpl((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimType*)src1Local.GetPhyAddr(), calCount);
}

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
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.src0BlkStride src0 block stride
 * @param [in] intriParams.src1BlkStride src1 block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src0 repeat stride
 * @param [in] intriParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void Mul(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "Mul")) {
        ASCENDC_REPORT_CHECK_ERROR("Mul", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    MulImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimType*)src1Local.GetPhyAddr(), mask, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask>
__aicore__ inline void Mul(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "Mul")) {
        ASCENDC_REPORT_CHECK_ERROR("Mul", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    MulImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimType*)src1Local.GetPhyAddr(), mask, repeatTimes, repeatParams);
}

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
    const LocalTensor<T>& src1Local, const int32_t& calCount)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, calCount, "Mul")) {
        ASCENDC_REPORT_CHECK_ERROR("Mul", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    MulImpl((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimType*)src1Local.GetPhyAddr(), calCount);
}

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
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.src0BlkStride src0 block stride
 * @param [in] intriParams.src1BlkStride src1 block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src0 repeat stride
 * @param [in] intriParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void Div(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "Div")) {
        ASCENDC_REPORT_CHECK_ERROR("Div", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    DivImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimType*)src1Local.GetPhyAddr(), mask, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask>
__aicore__ inline void Div(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "Div")) {
        ASCENDC_REPORT_CHECK_ERROR("Div", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    DivImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimType*)src1Local.GetPhyAddr(), mask, repeatTimes, repeatParams);
}

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
    const LocalTensor<T>& src1Local, const int32_t& calCount)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, calCount, "Div")) {
        ASCENDC_REPORT_CHECK_ERROR("Div", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    DivImpl((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimType*)src1Local.GetPhyAddr(), calCount);
}

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
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.src0BlkStride src0 block stride
 * @param [in] intriParams.src1BlkStride src1 block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src0 repeat stride
 * @param [in] intriParams.src1RepStride src1 repeat stride
 */
template <typename T, typename U, bool isSetMask>
__aicore__ inline void MulAddDst(const LocalTensor<T>& dstLocal, const LocalTensor<U>& src0Local,
    const LocalTensor<U>& src1Local, const uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    using PrimDstType = PrimT<T>;
    using PrimSrcType = PrimT<U>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinaryDiffType(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "MulAddDst")) {
        ASCENDC_REPORT_CHECK_ERROR("MulAddDst", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    MulAddDstImpl<PrimDstType, PrimSrcType, isSetMask>((__ubuf__ PrimDstType*)dstLocal.GetPhyAddr(),
        (__ubuf__ PrimSrcType*)src0Local.GetPhyAddr(), (__ubuf__ PrimSrcType*)src1Local.GetPhyAddr(), mask, repeatTimes,
        repeatParams);
}

template <typename T, typename U, bool isSetMask>
__aicore__ inline void MulAddDst(const LocalTensor<T>& dstLocal, const LocalTensor<U>& src0Local,
    const LocalTensor<U>& src1Local, uint64_t mask, const uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    using PrimDstType = PrimT<T>;
    using PrimSrcType = PrimT<U>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinaryDiffType(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "MulAddDst")) {
        ASCENDC_REPORT_CHECK_ERROR("MulAddDst", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    MulAddDstImpl<PrimDstType, PrimSrcType, isSetMask>((__ubuf__ PrimDstType*)dstLocal.GetPhyAddr(),
        (__ubuf__ PrimSrcType*)src0Local.GetPhyAddr(), (__ubuf__ PrimSrcType*)src1Local.GetPhyAddr(), mask, repeatTimes,
        repeatParams);
}

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
    const LocalTensor<U>& src1Local, const int32_t& calCount)
{
    using PrimDstType = PrimT<T>;
    using PrimSrcType = PrimT<U>;
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncVecBinaryDiffType(dstLocal, src0Local, src1Local, calCount, "MulAddDst")) {
        ASCENDC_REPORT_CHECK_ERROR("MulAddDst", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    MulAddDstImpl((__ubuf__ PrimDstType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimSrcType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimSrcType*)src1Local.GetPhyAddr(), calCount);
}

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
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.src0BlkStride src0 block stride
 * @param [in] intriParams.src1BlkStride src1 block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src0 repeat stride
 * @param [in] intriParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void Max(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "Max")) {
        ASCENDC_REPORT_CHECK_ERROR("Max", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    MaxImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimType*)src1Local.GetPhyAddr(), mask, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask>
__aicore__ inline void Max(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "Max")) {
        ASCENDC_REPORT_CHECK_ERROR("Max", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    MaxImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimType*)src1Local.GetPhyAddr(), mask, repeatTimes, repeatParams);
}

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
    const LocalTensor<T>& src1Local, const int32_t& calCount)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, calCount, "Max")) {
        ASCENDC_REPORT_CHECK_ERROR("Max", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    MaxImpl((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimType*)src1Local.GetPhyAddr(), calCount);
}

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
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.src0BlkStride src0 block stride
 * @param [in] intriParams.src1BlkStride src1 block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src0 repeat stride
 * @param [in] intriParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void Min(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "Min")) {
        ASCENDC_REPORT_CHECK_ERROR("Min", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    MinImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimType*)src1Local.GetPhyAddr(), mask, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask>
__aicore__ inline void Min(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "Min")) {
        ASCENDC_REPORT_CHECK_ERROR("Min", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    MinImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimType*)src1Local.GetPhyAddr(), mask, repeatTimes, repeatParams);
}

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
    const LocalTensor<T>& src1Local, const int32_t& calCount)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, calCount, "Min")) {
        ASCENDC_REPORT_CHECK_ERROR("Min", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    MinImpl((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimType*)src1Local.GetPhyAddr(), calCount);
}

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
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.src0BlkStride src0 block stride
 * @param [in] intriParams.src1BlkStride src1 block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src0 repeat stride
 * @param [in] intriParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void And(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "And")) {
        ASCENDC_REPORT_CHECK_ERROR("And", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    AndImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimType*)src1Local.GetPhyAddr(), mask, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask>
__aicore__ inline void And(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "And")) {
        ASCENDC_REPORT_CHECK_ERROR("And", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    AndImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimType*)src1Local.GetPhyAddr(), mask, repeatTimes, repeatParams);
}

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
    const LocalTensor<T>& src1Local, const int32_t& calCount)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, calCount, "And")) {
        ASCENDC_REPORT_CHECK_ERROR("And", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    AndImpl((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimType*)src1Local.GetPhyAddr(), calCount);
}

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
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.src0BlkStride src0 block stride
 * @param [in] intriParams.src1BlkStride src1 block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src0 repeat stride
 * @param [in] intriParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void Or(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "Or")) {
        ASCENDC_REPORT_CHECK_ERROR("Or", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    OrImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimType*)src1Local.GetPhyAddr(), mask, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask>
__aicore__ inline void Or(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "Or")) {
        ASCENDC_REPORT_CHECK_ERROR("Or", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    OrImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimType*)src1Local.GetPhyAddr(), mask, repeatTimes, repeatParams);
}

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
    const LocalTensor<T>& src1Local, const int32_t& calCount)
{
    using PrimType = PrimT<T>;
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, calCount, "Or")) {
        ASCENDC_REPORT_CHECK_ERROR("Or", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    OrImpl((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimType*)src1Local.GetPhyAddr(), calCount);
}

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
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.src0BlkStride src0 block stride
 * @param [in] intriParams.src1BlkStride src1 block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src0 repeat stride
 * @param [in] intriParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void AddRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        return;
    }
#endif
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "AddRelu")) {
        ASCENDC_REPORT_CHECK_ERROR("AddRelu", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    AddReluImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(),
        (__ubuf__ PrimType*)src0Local.GetPhyAddr(), (__ubuf__ PrimType*)src1Local.GetPhyAddr(), mask, repeatTimes,
        repeatParams);
}

template <typename T, bool isSetMask>
__aicore__ inline void AddRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        return;
    }
#endif
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "AddRelu")) {
        ASCENDC_REPORT_CHECK_ERROR("AddRelu", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    AddReluImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(),
        (__ubuf__ PrimType*)src0Local.GetPhyAddr(), (__ubuf__ PrimType*)src1Local.GetPhyAddr(), mask, repeatTimes,
        repeatParams);
}

template <typename T>
__aicore__ inline void AddRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, const int32_t& calCount)
{
    using PrimType = PrimT<T>;
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        return;
    }
#endif
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, calCount, "AddRelu")) {
        ASCENDC_REPORT_CHECK_ERROR("AddRelu", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    AddReluImpl((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimType*)src1Local.GetPhyAddr(), calCount);
}

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
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.src0BlkStride src0 block stride
 * @param [in] intriParams.src1BlkStride src1 block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src0 repeat stride
 * @param [in] intriParams.src1RepStride src1 repeat stride
 */
template <bool isSetMask>
__aicore__ inline void AddDeqRelu(const LocalTensor<half>& dstLocal, const LocalTensor<int32_t>& src0Local,
    const LocalTensor<int32_t>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinaryDiffType(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "AddDeqRelu")) {
        ASCENDC_REPORT_CHECK_ERROR("AddDeqRelu", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    AddDeqReluImpl<isSetMask>((__ubuf__ half*)dstLocal.GetPhyAddr(), (__ubuf__ int32_t*)src0Local.GetPhyAddr(),
        (__ubuf__ int32_t*)src1Local.GetPhyAddr(), mask, repeatTimes, repeatParams);
}

template <typename T, typename U, bool isSetMask>
__aicore__ inline void AddDeqRelu(const LocalTensor<T>& dstLocal, const LocalTensor<U>& src0Local,
    const LocalTensor<U>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    using PrimDstType = PrimT<T>;
    using PrimSrcType = PrimT<U>;
    static_assert((IsSameType<PrimDstType, half>::value && IsSameType<PrimSrcType, int32_t>::value) &&
        "Failed to check dtype in AddDeqRelu, current api support dtype combination is src: int32_t, dst: half.");
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinaryDiffType(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "AddDeqRelu")) {
        ASCENDC_REPORT_CHECK_ERROR("AddDeqRelu", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    AddDeqReluImpl<isSetMask>((__ubuf__ PrimDstType*)dstLocal.GetPhyAddr(),
        (__ubuf__ PrimSrcType*)src0Local.GetPhyAddr(), (__ubuf__ PrimSrcType*)src1Local.GetPhyAddr(), mask, repeatTimes,
        repeatParams);
}

template <bool isSetMask>
__aicore__ inline void AddDeqRelu(const LocalTensor<half> &dstLocal, const LocalTensor<int32_t> &src0Local,
    const LocalTensor<int32_t> &src1Local, uint64_t mask, const uint8_t repeatTimes,
    const BinaryRepeatParams &repeatParams)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinaryDiffType(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "AddDeqRelu")) {
        ASCENDC_REPORT_CHECK_ERROR("AddDeqRelu", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    AddDeqReluImpl<isSetMask>((__ubuf__ half*)dstLocal.GetPhyAddr(), (__ubuf__ int32_t*)src0Local.GetPhyAddr(),
        (__ubuf__ int32_t*)src1Local.GetPhyAddr(), mask, repeatTimes, repeatParams);
}

template <typename T, typename U, bool isSetMask>
__aicore__ inline void AddDeqRelu(const LocalTensor<T> &dstLocal, const LocalTensor<U> &src0Local,
    const LocalTensor<U> &src1Local, uint64_t mask, const uint8_t repeatTimes,
    const BinaryRepeatParams &repeatParams)
{
    using PrimDstType = PrimT<T>;
    using PrimSrcType = PrimT<U>;
    static_assert((IsSameType<PrimDstType, half>::value && IsSameType<PrimSrcType, int32_t>::value) &&
        "Failed to check dtype in AddDeqRelu, current api support dtype combination is src: int32_t, dst: half.");
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinaryDiffType(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "AddDeqRelu")) {
        ASCENDC_REPORT_CHECK_ERROR("AddDeqRelu", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    AddDeqReluImpl<isSetMask>((__ubuf__ PrimDstType*)dstLocal.GetPhyAddr(),
        (__ubuf__ PrimSrcType*)src0Local.GetPhyAddr(), (__ubuf__ PrimSrcType*)src1Local.GetPhyAddr(), mask, repeatTimes,
        repeatParams);
}

/*
 * @ingroup AddDeqRelu Level 2
 * @brief dst = DeqRelu(src0 + src1)
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
__aicore__ inline void AddDeqRelu(const LocalTensor<half>& dstLocal, const LocalTensor<int32_t>& src0Local,
    const LocalTensor<int32_t>& src1Local, const int32_t& calCount)
{
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncVecBinaryDiffType(dstLocal, src0Local, src1Local, calCount, "AddDeqRelu")) {
        ASCENDC_REPORT_CHECK_ERROR("AddDeqRelu", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    AddDeqReluImpl((__ubuf__ half *)dstLocal.GetPhyAddr(), (__ubuf__ int32_t *)src0Local.GetPhyAddr(),
        (__ubuf__ int32_t *)src1Local.GetPhyAddr(), calCount);
}

template <typename T, typename U>
__aicore__ inline void AddDeqRelu(const LocalTensor<T>& dstLocal, const LocalTensor<U>& src0Local,
    const LocalTensor<U>& src1Local, const int32_t& calCount)
{
    using PrimDstType = PrimT<T>;
    using PrimSrcType = PrimT<U>;
    static_assert((IsSameType<PrimDstType, half>::value && IsSameType<PrimSrcType, int32_t>::value) &&
        "Failed to check dtype in AddDeqRelu, current api support dtype combination is src: int32_t, dst: half.");
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncVecBinaryDiffType(dstLocal, src0Local, src1Local, calCount, "AddDeqRelu")) {
        ASCENDC_REPORT_CHECK_ERROR("AddDeqRelu", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    AddDeqReluImpl((__ubuf__ PrimDstType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimSrcType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimSrcType*)src1Local.GetPhyAddr(), calCount);
}

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
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.src0BlkStride src0 block stride
 * @param [in] intriParams.src1BlkStride src1 block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src0 repeat stride
 * @param [in] intriParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void FusedMulAdd(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        return;
    }
#endif
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "FusedMulAdd")) {
        ASCENDC_REPORT_CHECK_ERROR("FusedMulAdd", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    FusedMulAddImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(),
        (__ubuf__ PrimType*)src0Local.GetPhyAddr(), (__ubuf__ PrimType*)src1Local.GetPhyAddr(), mask, repeatTimes,
        repeatParams);
}

template <typename T, bool isSetMask>
__aicore__ inline void FusedMulAdd(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        return;
    }
#endif
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "FusedMulAdd")) {
        ASCENDC_REPORT_CHECK_ERROR("FusedMulAdd", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    FusedMulAddImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(),
        (__ubuf__ PrimType*)src0Local.GetPhyAddr(), (__ubuf__ PrimType*)src1Local.GetPhyAddr(), mask, repeatTimes,
        repeatParams);
}

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
    const LocalTensor<T>& src1Local, const int32_t& calCount)
{
    using PrimType = PrimT<T>;
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        return;
    }
#endif
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, calCount, "FusedMulAdd")) {
        ASCENDC_REPORT_CHECK_ERROR("FusedMulAdd", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    FusedMulAddImpl((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimType*)src1Local.GetPhyAddr(), calCount);
}

/* **************************************************************************************************
 * FusedMulAddRelu                                             *
 * ************************************************************************************************* */
/*
 * @ingroup FusedMulAddRelu Level 0
 * @brief dst = src0 * dst + src1
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.src0BlkStride src0 block stride
 * @param [in] intriParams.src1BlkStride src1 block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src0 repeat stride
 * @param [in] intriParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void FusedMulAddRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        return;
    }
#endif
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "FusedMulAddRelu")) {
        ASCENDC_REPORT_CHECK_ERROR("FusedMulAddRelu", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    FusedMulAddReluImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(),
        (__ubuf__ PrimType*)src0Local.GetPhyAddr(), (__ubuf__ PrimType*)src1Local.GetPhyAddr(), mask, repeatTimes,
        repeatParams);
}

template <typename T, bool isSetMask>
__aicore__ inline void FusedMulAddRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        return;
    }
#endif
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "FusedMulAddRelu")) {
        ASCENDC_REPORT_CHECK_ERROR("FusedMulAddRelu", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    FusedMulAddReluImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(),
        (__ubuf__ PrimType*)src0Local.GetPhyAddr(), (__ubuf__ PrimType*)src1Local.GetPhyAddr(), mask, repeatTimes,
        repeatParams);
}

/*
 * @ingroup FusedMulAddRelu Level 2
 * @brief dst = src0 * dst + src1
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T>
__aicore__ inline void FusedMulAddRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, const int32_t& calCount)
{
    using PrimType = PrimT<T>;
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        return;
    }
#endif
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, calCount, "FusedMulAddRelu")) {
        ASCENDC_REPORT_CHECK_ERROR("FusedMulAddRelu", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    FusedMulAddReluImpl((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimType*)src1Local.GetPhyAddr(), calCount);
}

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
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.src0BlkStride src0 block stride
 * @param [in] intriParams.src1BlkStride src1 block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src0 repeat stride
 * @param [in] intriParams.src1RepStride src1 repeat stride
 */
template <typename T, bool isSetMask>
__aicore__ inline void SubRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        return;
    }
#endif
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "SubRelu")) {
        ASCENDC_REPORT_CHECK_ERROR("SubRelu", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    SubReluImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(),
        (__ubuf__ PrimType*)src0Local.GetPhyAddr(), (__ubuf__ PrimType*)src1Local.GetPhyAddr(), mask, repeatTimes,
        repeatParams);
}

template <typename T, bool isSetMask>
__aicore__ inline void SubRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
    using PrimType = PrimT<T>;
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        return;
    }
#endif
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "SubRelu")) {
        ASCENDC_REPORT_CHECK_ERROR("SubRelu", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    SubReluImpl<PrimType, isSetMask>((__ubuf__ PrimType*)dstLocal.GetPhyAddr(),
        (__ubuf__ PrimType*)src0Local.GetPhyAddr(), (__ubuf__ PrimType*)src1Local.GetPhyAddr(), mask, repeatTimes,
        repeatParams);
}

template <typename T>
__aicore__ inline void SubRelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, const int32_t& calCount)
{
    using PrimType = PrimT<T>;
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        return;
    }
#endif
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncVecBinary(dstLocal, src0Local, src1Local, calCount, "SubRelu")) {
        ASCENDC_REPORT_CHECK_ERROR("SubRelu", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    SubReluImpl((__ubuf__ PrimType*)dstLocal.GetPhyAddr(), (__ubuf__ PrimType*)src0Local.GetPhyAddr(),
        (__ubuf__ PrimType*)src1Local.GetPhyAddr(), calCount);
}
} // namespace AscendC
#pragma end_pipe
#endif // ASCENDC_MODULE_INNER_OPERATOR_VEC_BINARY_INTERFACE_H