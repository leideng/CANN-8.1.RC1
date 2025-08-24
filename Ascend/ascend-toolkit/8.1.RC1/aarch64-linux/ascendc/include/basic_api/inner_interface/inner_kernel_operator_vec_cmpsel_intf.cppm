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
 * \file inner_kernel_operator_vec_cmpsel_intf.cppm
 * \brief
 */
#ifndef ASCENDC_MODULE_INNER_OPERATOR_VEC_CMPSEL_INTERFACE_H
#define ASCENDC_MODULE_INNER_OPERATOR_VEC_CMPSEL_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_check.h"
#include "kernel_struct_binary.h"
#include "kernel_struct_unary.h"

#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_vec_cmpsel_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_vec_cmpsel_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_vec_cmpsel_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_vec_cmpsel_impl.h"
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
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.src0BlkStride src0 block stride
 * @param [in] intriParams.src1BlkStride src1 block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src0 repeat stride
 * @param [in] intriParams.src1RepStride src1 repeat stride
 */
template <typename T, typename U, bool isSetMask>
__aicore__ inline void Compare(const LocalTensor<U>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, CMPMODE cmpMode, const uint64_t mask[], uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
#if __CCE_AICORE__ == 200
    ASCENDC_ASSERT((SupportType<T, half, float>() && SupportType<U, int8_t, uint8_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Compare, current api support dtype combination is src: "
        "half / float, dst: int8_t / uint8_t.");});
#elif __CCE_AICORE__ == 220
    ASCENDC_ASSERT((SupportType<T, half, float, int32_t>() && SupportType<U, int8_t, uint8_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Compare, current api support dtype combination is src: "
        "half / float / int32_t, dst: int8_t / uint8_t.");});
#endif
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinaryCmp(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "Compare")) {
        ASCENDC_REPORT_CHECK_ERROR("Compare", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    VcmpvImpl<T, U, isSetMask>((__ubuf__ U*)dstLocal.GetPhyAddr(), (__ubuf__ T*)src0Local.GetPhyAddr(),
        (__ubuf__ T*)src1Local.GetPhyAddr(), cmpMode, mask, repeatTimes, repeatParams);
}

template <typename T, typename U, bool isSetMask>
__aicore__ inline void Compare(const LocalTensor<U>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<T>& src1Local, CMPMODE cmpMode, const uint64_t mask, uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
#if __CCE_AICORE__ == 200
    ASCENDC_ASSERT((SupportType<T, half, float>() && SupportType<U, int8_t, uint8_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Compare, current api support dtype combination is src: "
        "half / float, dst: int8_t / uint8_t.");});
#elif __CCE_AICORE__ == 220
    ASCENDC_ASSERT((SupportType<T, half, float, int32_t>() && SupportType<U, int8_t, uint8_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Compare, current api support dtype combination is src: "
        "half / float / int32_t, dst: int8_t / uint8_t.");});
#endif
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinaryCmp(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "Compare")) {
        ASCENDC_REPORT_CHECK_ERROR("Compare", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    VcmpvImpl<T, U, isSetMask>((__ubuf__ U*)dstLocal.GetPhyAddr(), (__ubuf__ T*)src0Local.GetPhyAddr(),
        (__ubuf__ T*)src1Local.GetPhyAddr(), cmpMode, mask, repeatTimes, repeatParams);
}

template <typename T, bool isSetMask>
__aicore__ inline void Compare(const LocalTensor<T>& src0Local, const LocalTensor<T>& src1Local, CMPMODE cmpMode,
    const uint64_t mask[], const BinaryRepeatParams& repeatParams)
{
#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 220
    ASCENDC_ASSERT((SupportType<T, half, float>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Compare, "
        "current api support dtype combination is src: half / float.");});
#endif
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinaryCmpRgt(src0Local, src1Local, mask, repeatParams, "Compare")) {
        ASCENDC_REPORT_CHECK_ERROR("Compare", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    VcmpImpl<T, isSetMask>((__ubuf__ T*)src0Local.GetPhyAddr(),
        (__ubuf__ T*)src1Local.GetPhyAddr(), cmpMode, mask, repeatParams);
}

template <typename T, bool isSetMask>
__aicore__ inline void Compare(const LocalTensor<T>& src0Local, const LocalTensor<T>& src1Local, CMPMODE cmpMode,
    const uint64_t mask, const BinaryRepeatParams& repeatParams)
{
#if __CCE_AICORE__ == 200 || __CCE_AICORE__ == 220
    ASCENDC_ASSERT((SupportType<T, half, float>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Compare, "
        "current api support dtype combination is src: half / float.");});
#endif
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinaryCmpRgt(src0Local, src1Local, mask, repeatParams, "Compare")) {
        ASCENDC_REPORT_CHECK_ERROR("Compare", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    VcmpImpl<T, isSetMask>((__ubuf__ T*)src0Local.GetPhyAddr(), (__ubuf__ T*)src1Local.GetPhyAddr(), cmpMode, mask,
        repeatParams);
}

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
    const LocalTensor<T>& src1Local, CMPMODE cmpMode, uint32_t calCount)
{
#if __CCE_AICORE__ == 200
    ASCENDC_ASSERT((SupportType<T, half, float>() && SupportType<U, int8_t, uint8_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Compare, current api support dtype combination is src: "
        "half / float, dst: int8_t / uint8_t.");});
#elif __CCE_AICORE__ == 220
    ASCENDC_ASSERT((SupportType<T, half, float, int32_t>() && SupportType<U, int8_t, uint8_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Compare, current api support dtype combination is src: "
        "half / float / int32_t, dst: int8_t / uint8_t.");});
#endif
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncVecBinaryCmp(dstLocal, src0Local, src1Local, calCount, "Compare")) {
        ASCENDC_REPORT_CHECK_ERROR("Compare", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    VcmpvImpl((__ubuf__ U*)dstLocal.GetPhyAddr(), (__ubuf__ T*)src0Local.GetPhyAddr(),
        (__ubuf__ T*)src1Local.GetPhyAddr(), cmpMode, calCount);
}

template <typename T>
__aicore__ inline void GetCmpMask(const LocalTensor<T>& dst)
{
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        return;
    }
#endif
#if ASCENDC_CPU_DEBUG
    constexpr uint64_t ALIGN_16B = 16;
    CheckTensorAlign<T>(dst, ALIGN_16B, "dst", "GetCmpMask");
    CheckTensorPos<T>(dst, Hardware::UB, "dst", "VECIN / VECCALC / VECOUT", "GetCmpMask");
#endif
    GetCmpMaskImpl((__ubuf__ T*)dst.GetPhyAddr());
}

template <typename T>
__aicore__ inline void SetCmpMask(const LocalTensor<T>& src)
{
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        return;
    }
#endif
#if ASCENDC_CPU_DEBUG
    constexpr uint64_t ALIGN_16B = 16;
    CheckTensorAlign<T>(src, ALIGN_16B, "src", "SetCmpMask");
    CheckTensorPos<T>(src, Hardware::UB, "src", "VECIN / VECCALC / VECOUT", "SetCmpMask");
#endif
    SetCmpMaskImpl((__ubuf__ T*)src.GetPhyAddr());

}

/* **************************************************************************************************
 * CompareScalar                                           *
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
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.srcBlkStride src0 block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.srcRepStride src0 repeat stride
 */
template <typename T, typename U, bool isSetMask>
__aicore__ inline void CompareScalar(const LocalTensor<U>& dstLocal, const LocalTensor<T>& src0Local,
    const T src1Scalar, CMPMODE cmpMode, const uint64_t mask[], uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
#if __CCE_AICORE__ == 200
    ASCENDC_ASSERT((SupportType<T, half, float>() && SupportType<U, uint8_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in CompareScalar, current api support dtype combination is "
        "src0: half / float, dst: uint8_t.");});
#elif __CCE_AICORE__ == 220
    ASCENDC_ASSERT((SupportType<T, half, float, int32_t>() && SupportType<U, uint8_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in CompareScalar, current api support dtype combination is "
        "src0: half / float / int32_t, dst: uint8_t.");});
#endif
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncVecBinaryScalarCmp(dstLocal, src0Local, src1Scalar, ONE_REPEAT_BYTE_SIZE / sizeof(T), repeatTimes,
        repeatParams, "CompareScalar")) {
        ASCENDC_REPORT_CHECK_ERROR("CompareScalar", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    VcmpvsImpl<T, U, isSetMask>((__ubuf__ U*)dstLocal.GetPhyAddr(), (__ubuf__ T*)src0Local.GetPhyAddr(), src1Scalar,
        cmpMode, mask, repeatTimes, repeatParams);
}

template <typename T, typename U, bool isSetMask>
__aicore__ inline void CompareScalar(const LocalTensor<U>& dstLocal, const LocalTensor<T>& src0Local,
    const T src1Scalar, CMPMODE cmpMode, const uint64_t mask, uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
#if __CCE_AICORE__ == 200
    ASCENDC_ASSERT((SupportType<T, half, float>() && SupportType<U, uint8_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in CompareScalar, current api support dtype combination is "
        "src0: half / float, dst: uint8_t.");});
#elif __CCE_AICORE__ == 220
    ASCENDC_ASSERT((SupportType<T, half, float, int32_t>() && SupportType<U, uint8_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in CompareScalar, current api support dtype combination is "
        "src0: half / float / int32_t, dst: uint8_t.");});
#endif
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncVecBinaryScalarCmp(dstLocal, src0Local, src1Scalar, ONE_REPEAT_BYTE_SIZE / sizeof(T), repeatTimes,
        repeatParams, "CompareScalar")) {
        ASCENDC_REPORT_CHECK_ERROR("CompareScalar", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    VcmpvsImpl<T, U, isSetMask>((__ubuf__ U*)dstLocal.GetPhyAddr(), (__ubuf__ T*)src0Local.GetPhyAddr(), src1Scalar,
        cmpMode, mask, repeatTimes, repeatParams);
}

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
    const T src1Scalar, CMPMODE cmpMode, uint32_t calCount)
{
#if __CCE_AICORE__ == 200
    ASCENDC_ASSERT((SupportType<T, half, float>() && SupportType<U, uint8_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in CompareScalar, current api support dtype combination is "
        "src0: half / float, dst: uint8_t.");});
#elif __CCE_AICORE__ == 220
    ASCENDC_ASSERT((SupportType<T, half, float, int32_t>() && SupportType<U, uint8_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in CompareScalar, current api support dtype combination is "
        "src0: half / float / int32_t, dst: uint8_t.");});
#endif
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncVecBinaryScalarCmp(dstLocal, src0Local, src1Scalar, calCount, "CompareScalar")) {
        ASCENDC_REPORT_CHECK_ERROR("CompareScalar", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    VcmpvsImpl((__ubuf__ U*)dstLocal.GetPhyAddr(), (__ubuf__ T*)src0Local.GetPhyAddr(), src1Scalar, cmpMode, calCount);
}

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
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.src0BlkStride src0 block stride
 * @param [in] intriParams.src1BlkStride src1 block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src0 repeat stride
 * @param [in] intriParams.src1RepStride src1 repeat stride
 */
// select mode: 0/1/2
template <typename T, typename U, bool isSetMask>
__aicore__ inline void Select(const LocalTensor<T>& dstLocal, const LocalTensor<U>& selMask,
    const LocalTensor<T>& src0Local, const LocalTensor<T>& src1Local, SELMODE selMode, uint64_t mask[],
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncSelectVec(dstLocal, selMask, src0Local, src1Local, mask, repeatTimes, repeatParams, "Select")) {
        ASCENDC_REPORT_CHECK_ERROR("Select", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    ASCENDC_ASSERT((SupportType<T, half, float>() && SupportType<U, uint8_t, uint16_t, uint32_t, uint64_t>()), {
        KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Select, current api support dtype combination is dst, "
        "src0Local and src1Local are both: half / float, selMask:  uint8_t / uint16_t / uint32_t / uint64_t");});
    VselImpl((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ U*)selMask.GetPhyAddr(), (__ubuf__ T*)src0Local.GetPhyAddr(),
        (__ubuf__ T*)src1Local.GetPhyAddr(), selMode, mask, repeatTimes, repeatParams);
}

// select mode: 0/1/2
template <typename T, typename U, bool isSetMask>
__aicore__ inline void Select(const LocalTensor<T>& dstLocal, const LocalTensor<U>& selMask,
    const LocalTensor<T>& src0Local, const LocalTensor<T>& src1Local, SELMODE selMode, uint64_t mask,
    uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncSelectVec(dstLocal, selMask, src0Local, src1Local, mask, repeatTimes, repeatParams, "Select")) {
        ASCENDC_REPORT_CHECK_ERROR("Select", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    ASCENDC_ASSERT((SupportType<T, half, float>() && SupportType<U, uint8_t, uint16_t, uint32_t, uint64_t>()), {
        KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Select, current api support dtype combination is dst, "
        "src0Local and src1Local are both: half / float, selMask:  uint8_t / uint16_t / uint32_t / uint64_t");});
    VselImpl((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ U*)selMask.GetPhyAddr(), (__ubuf__ T*)src0Local.GetPhyAddr(),
        (__ubuf__ T*)src1Local.GetPhyAddr(), selMode, mask, repeatTimes, repeatParams);
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
    const LocalTensor<T>& src0Local, const LocalTensor<T>& src1Local, SELMODE selMode, uint32_t calCount)
{
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncSelectVec(dstLocal, selMask, src0Local, src1Local, (int32_t)calCount, "Select")) {
        ASCENDC_REPORT_CHECK_ERROR("Select", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    ASCENDC_ASSERT((SupportType<T, half, float>() && SupportType<U, uint8_t, uint16_t, uint32_t, uint64_t>()), {
        KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Select, current api support dtype combination is dst, "
        "src0Local and src1Local are both: half / float, selMask: uint8_t / uint16_t / uint32_t / uint64_t");});
    VselImpl((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ U*)selMask.GetPhyAddr(), (__ubuf__ T*)src0Local.GetPhyAddr(),
        (__ubuf__ T*)src1Local.GetPhyAddr(), selMode, calCount);
}

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
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.src0BlkStride src0 block stride
 * @param [in] intriParams.src1BlkStride src1 block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src0 repeat stride
 * @param [in] intriParams.src1RepStride src1 repeat stride
 */
// select mode: 1
template <typename T, typename U, bool isSetMask>
__aicore__ inline void Select(const LocalTensor<T>& dstLocal, const LocalTensor<U>& selMask,
    const LocalTensor<T>& src0Local, T src1Local, SELMODE selMode, uint64_t mask[], uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "Select")) {
        ASCENDC_REPORT_CHECK_ERROR("Select", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    CheckTensorPos<U>(selMask, Hardware::UB, "selMask", "VECIN / VECCALC / VECOUT", "Select");
    ASCENDC_ASSERT((SupportType<T, half, float>() && SupportType<U, uint8_t, uint16_t, uint32_t, uint64_t>()),
        { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Select, current api support dtype combination is dst, "
        "src0Local and src1Local are both: half / float, selMask:  uint8_t / uint16_t / uint32_t / uint64_t");});
    VselImpl((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ U*)selMask.GetPhyAddr(), (__ubuf__ T*)src0Local.GetPhyAddr(),
        src1Local, selMode, mask, repeatTimes, repeatParams);
}

// select mode: 1
template <typename T, typename U, bool isSetMask>
__aicore__ inline void Select(const LocalTensor<T>& dstLocal, const LocalTensor<U>& selMask,
    const LocalTensor<T>& src0Local, T src1Local, SELMODE selMode, uint64_t mask, uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalar(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "Select")) {
        ASCENDC_REPORT_CHECK_ERROR("Select", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    CheckTensorPos<U>(selMask, Hardware::UB, "selMask", "VECIN / VECCALC / VECOUT", "Select");
    ASCENDC_ASSERT((SupportType<T, half, float>() && SupportType<U, uint8_t, uint16_t, uint32_t, uint64_t>()),
        { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Select, current api support dtype combination is dst, "
        "src0Local and src1Local are both: half / float, selMask:  uint8_t / uint16_t / uint32_t / uint64_t");});
    VselImpl((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ U*)selMask.GetPhyAddr(), (__ubuf__ T*)src0Local.GetPhyAddr(),
        src1Local, selMode, mask, repeatTimes, repeatParams);
}

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
    const LocalTensor<T>& src0Local, T src1Local, SELMODE selMode, uint32_t calCount)
{
#if ASCENDC_CPU_DEBUG
    if (!CheckFunVecBinaryScalar(dstLocal, src0Local, src1Local, (int32_t)calCount, "Select")) {
        ASCENDC_REPORT_CHECK_ERROR("Select", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    CheckTensorPos<U>(selMask, Hardware::UB, "selMask", "VECIN / VECCALC / VECOUT", "Select");
    ASCENDC_ASSERT((SupportType<T, half, float>() && SupportType<U, uint8_t, uint16_t, uint32_t, uint64_t>()),
        { KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Select, current api support dtype combination is dst, "
        "src0Local and src1Local are both: half / float, selMask:  uint8_t / uint16_t / uint32_t / uint64_t");});
    VselImpl((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ U*)selMask.GetPhyAddr(), (__ubuf__ T*)src0Local.GetPhyAddr(),
        src1Local, selMode, calCount);
}
} // namespace AscendC
#pragma end_pipe
#endif // ASCENDC_MODULE_INNER_OPERATOR_VEC_CMPSEL_INTERFACE_H
