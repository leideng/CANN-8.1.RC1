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
 * \file inner_kernel_operator_vec_vconv_intf.cppm
 * \brief
 */
#ifndef ASCENDC_MODULE_INNER_OPERATOR_VEC_VCONV_INTERFACE_H
#define ASCENDC_MODULE_INNER_OPERATOR_VEC_VCONV_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_check.h"
#include "kernel_struct_binary.h"
#include "kernel_struct_unary.h"
#include "kernel_struct_vdeq.h"

#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_vec_vconv_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_vec_vconv_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_vec_vconv_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_vec_vconv_impl.h"
#endif

namespace AscendC {
#pragma begin_pipe(V)
/* **************************************************************************************************
 * Cast                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Cast Level 0
 * @brief Precision conversion
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] round_mode round mode
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.srcBlkStride src block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src repeat stride
 */
// Cast::Level 0 - mask bit mode
template <typename T1, typename T2, bool isSetMask>
__aicore__ inline void Cast(const LocalTensor<T1>& dstLocal, const LocalTensor<T2>& srcLocal,
    const RoundMode& round_mode, const uint64_t mask[], const uint8_t repeatTimes,
    const UnaryRepeatParams& repeatParams)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if constexpr (IsSameType<T1, int4b_t>::value) {
        Int4Setter::Instance().SetDstInt4();
    } else if constexpr (IsSameType<T2, int4b_t>::value) {
        Int4Setter::Instance().SetSrcInt4();
    }
    if (!CheckFunVecBinaryScalarDiffType(dstLocal, srcLocal, static_cast<T2>(0), mask, repeatTimes, repeatParams,
        "Cast")) {
        ASCENDC_REPORT_CHECK_ERROR("Cast", KernelFuncType::MASK_BIT_MODE);
    }
    Int4Setter::Instance().ResetDstSrcInt4();
#endif
    CastImpl<T1, T2, isSetMask>((__ubuf__ T1*)dstLocal.GetPhyAddr(), (__ubuf__ T2*)srcLocal.GetPhyAddr(), round_mode,
        mask, repeatTimes, repeatParams);
}

// Cast::Level 0 - mask count mode
template <typename T1, typename T2, bool isSetMask>
__aicore__ inline void Cast(const LocalTensor<T1>& dstLocal, const LocalTensor<T2>& srcLocal,
    const RoundMode& round_mode, const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if constexpr (IsSameType<T1, int4b_t>::value) {
        Int4Setter::Instance().SetDstInt4();
    } else if constexpr (IsSameType<T2, int4b_t>::value) {
        Int4Setter::Instance().SetSrcInt4();
    }
    if (!CheckFunVecBinaryScalarDiffType(dstLocal, srcLocal, static_cast<T2>(0), mask, repeatTimes, repeatParams,
        "Cast")) {
        ASCENDC_REPORT_CHECK_ERROR("Cast", KernelFuncType::MASK_COUNT_MODE);
    }
    Int4Setter::Instance().ResetDstSrcInt4();
#endif
    CastImpl<T1, T2, isSetMask>((__ubuf__ T1*)dstLocal.GetPhyAddr(), (__ubuf__ T2*)srcLocal.GetPhyAddr(), round_mode,
        mask, repeatTimes, repeatParams);
}

/*
 * @ingroup Cast Level 2
 * @brief dst[i] = Precision conversion
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] round_mode round mode
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T1, typename T2>
__aicore__ inline void Cast(const LocalTensor<T1>& dstLocal, const LocalTensor<T2>& srcLocal,
    const RoundMode& round_mode, const uint32_t calCount)
{
#if ASCENDC_CPU_DEBUG
    if constexpr (IsSameType<T1, int4b_t>::value) {
        Int4Setter::Instance().SetDstInt4();
    } else if constexpr (IsSameType<T2, int4b_t>::value) {
        Int4Setter::Instance().SetSrcInt4();
    }
    if (!CheckFunVecBinaryScalarDiffType(dstLocal, srcLocal, static_cast<T2>(0), calCount, "Cast")) {
        ASCENDC_REPORT_CHECK_ERROR("Cast", KernelFuncType::CALCOUNT_MODE);
    }
    Int4Setter::Instance().ResetDstSrcInt4();
#endif
    if constexpr (!(IsSameType<T1, int4b_t>::value) && !(IsSameType<T2, int4b_t>::value)) {
        ASCENDC_DEBUG_ASSERT(CheckCastOverlappingHigh(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())), sizeof(T1), sizeof(T2),
            calCount), "Failed to pass Cast calcount mode check.");
    }
    CastImpl((__ubuf__ T1*)dstLocal.GetPhyAddr(), (__ubuf__ T2*)srcLocal.GetPhyAddr(), round_mode, calCount);
}

/*
 * @ingroup CastDeq Level 0
 * @brief Dequant from int16_t to uint8_t/int8_t
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] repeatTimes repeat times
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.srcBlkStride src block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.srcRepStride src repeat stride
 */
template <typename T1, typename T2, bool isSetMask, bool isVecDeq, bool halfBlock>
__aicore__ inline void CastDeq(const LocalTensor<T1>& dstLocal, const LocalTensor<T2>& srcLocal,
    const uint64_t mask[], uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalarDiffType(dstLocal.template ReinterpretCast<half>(), srcLocal, static_cast<T2>(0), mask,
        repeatTimes, repeatParams, "CastDeq")) {
        ASCENDC_REPORT_CHECK_ERROR("CastDeq", KernelFuncType::MASK_BIT_MODE);
    }

    ASCENDC_ASSERT((SupportType<T1, int8_t, uint8_t>() && SupportType<T2, int16_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in CastDeq, current api support dtype combination is dst: "
        "int8_t / uint8_t, src: int16_t");});
#endif
    CastDeqImpl<T1, T2, isSetMask, isVecDeq, halfBlock>((__ubuf__ T1*)dstLocal.GetPhyAddr(),
        (__ubuf__ T2*)srcLocal.GetPhyAddr(), mask, repeatTimes, repeatParams);
}
template <typename T1, typename T2, bool isSetMask, bool isVecDeq, bool halfBlock>
__aicore__ inline void CastDeq(const LocalTensor<T1>& dstLocal, const LocalTensor<T2>& srcLocal,
    const int32_t mask, uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)
{
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFunVecBinaryScalarDiffType(dstLocal.template ReinterpretCast<half>(), srcLocal, static_cast<T2>(0), mask,
        repeatTimes, repeatParams, "CastDeq")) {
        ASCENDC_REPORT_CHECK_ERROR("CastDeq", KernelFuncType::MASK_COUNT_MODE);
    }
    ASCENDC_ASSERT((SupportType<T1, int8_t, uint8_t>() && SupportType<T2, int16_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in CastDeq, current api support dtype combination is dst: "
        "int8_t / uint8_t, src: int16_t");});
#endif
    CastDeqImpl<T1, T2, isSetMask, isVecDeq, halfBlock>((__ubuf__ T1*)dstLocal.GetPhyAddr(),
        (__ubuf__ T2*)srcLocal.GetPhyAddr(), mask, repeatTimes, repeatParams);
}

/*
 * @ingroup CastDeq Level 2
 * @brief Dequant from int16_t to uint8_t/int8_t
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] repeatTimes repeat times
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.srcBlkStride src block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.srcRepStride src repeat stride
 */
template <typename T1, typename T2, bool isVecDeq, bool halfBlock>
__aicore__ inline void CastDeq(const LocalTensor<T1>& dstLocal, const LocalTensor<T2>& srcLocal,
    const uint32_t calCount)
{
#if ASCENDC_CPU_DEBUG
    if (!CheckFunVecBinaryScalarDiffType(dstLocal.template ReinterpretCast<half>(), srcLocal, static_cast<T2>(0),
        calCount, "CastDeq")) {
        ASCENDC_REPORT_CHECK_ERROR("CastDeq", KernelFuncType::CALCOUNT_MODE);
    }
    ASCENDC_ASSERT((SupportType<T1, int8_t, uint8_t>() && SupportType<T2, int16_t>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in CastDeq, current api support dtype combination is dst: "
        "int8_t / uint8_t, src: int16_t");});
#endif
    CastDeqImpl<T1, T2, isVecDeq, halfBlock>((__ubuf__ T1*)dstLocal.GetPhyAddr(), (__ubuf__ T2*)srcLocal.GetPhyAddr(),
        calCount);
}

/* **************************************************************************************************
 * AddReluCast                                             *
 * ************************************************************************************************* */
/*
 * @ingroup AddReluCast Level 0
 * @brief Precision conversion
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.src0BlkStride src block stride
 * @param [in] intriParams.src1BlkStride src block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src repeat stride
 * @param [in] intriParams.src1RepStride src repeat stride
 */
// AddReluCast::Level 0 - mask count mode
template <typename T1, typename T2, bool isSetMask>
__aicore__ inline void AddReluCast(const LocalTensor<T1>& dstLocal, const LocalTensor<T2>& src0Local,
    const LocalTensor<T2>& src1Local, uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        return;
    }
#endif
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinaryCmp(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "AddReluCast")) {
        ASCENDC_REPORT_CHECK_ERROR("AddReluCast", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    AddReluCastImpl<T1, T2, isSetMask>((__ubuf__ T1*)dstLocal.GetPhyAddr(), (__ubuf__ T2*)src0Local.GetPhyAddr(),
        (__ubuf__ T2*)src1Local.GetPhyAddr(), mask, repeatTimes, repeatParams);
}

// AddReluCast::Level 0 - mask bit mode
template <typename T1, typename T2, bool isSetMask>
__aicore__ inline void AddReluCast(const LocalTensor<T1>& dstLocal, const LocalTensor<T2>& src0Local,
    const LocalTensor<T2>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        return;
    }
#endif
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinaryCmp(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "AddReluCast")) {
        ASCENDC_REPORT_CHECK_ERROR("AddReluCast", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    AddReluCastImpl<T1, T2, isSetMask>((__ubuf__ T1*)dstLocal.GetPhyAddr(), (__ubuf__ T2*)src0Local.GetPhyAddr(),
        (__ubuf__ T2*)src1Local.GetPhyAddr(), mask, repeatTimes, repeatParams);
}

/*
 * @ingroup AddReluCast Level 2
 * @brief dst[i] = Precision conversion
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T1, typename T2>
__aicore__ inline void AddReluCast(const LocalTensor<T1>& dstLocal, const LocalTensor<T2>& src0Local,
    const LocalTensor<T2>& src1Local, const uint32_t calCount)
{
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        return;
    }
#endif
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncVecBinaryCmp(dstLocal, src0Local, src1Local, calCount, "AddReluCast")) {
        ASCENDC_REPORT_CHECK_ERROR("AddReluCast", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    AddReluCastImpl((__ubuf__ T1*)dstLocal.GetPhyAddr(), (__ubuf__ T2*)src0Local.GetPhyAddr(),
        (__ubuf__ T2*)src1Local.GetPhyAddr(), calCount);
}

/* **************************************************************************************************
 * SubReluCast                                             *
 * ************************************************************************************************* */
/*
 * @ingroup SubReluCast Level 0
 * @brief Precision conversion
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] mask[]/mask mask array/count
 * @param [in] repeatTimes repeat times
 * @param [in] intriParams.dstBlkStride dst block stride
 * @param [in] intriParams.src0BlkStride src block stride
 * @param [in] intriParams.src1BlkStride src block stride
 * @param [in] intriParams.dstRepStride dst repeat stride
 * @param [in] intriParams.src0RepStride src repeat stride
 * @param [in] intriParams.src1RepStride src repeat stride
 */
// AddReluCast::Level 0 - mask count mode
template <typename T1, typename T2, bool isSetMask>
__aicore__ inline void SubReluCast(const LocalTensor<T1>& dstLocal, const LocalTensor<T2>& src0Local,
    const LocalTensor<T2>& src1Local, uint64_t mask, const uint8_t repeatTimes, const BinaryRepeatParams& repeatParams)
{
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        return;
    }
#endif
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinaryCmp(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "SubReluCast")) {
        ASCENDC_REPORT_CHECK_ERROR("SubReluCast", KernelFuncType::MASK_COUNT_MODE);
    }
#endif
    SubReluCastImpl<T1, T2, isSetMask>((__ubuf__ T1*)dstLocal.GetPhyAddr(), (__ubuf__ T2*)src0Local.GetPhyAddr(),
        (__ubuf__ T2*)src1Local.GetPhyAddr(), mask, repeatTimes, repeatParams);
}

// SubReluCast::Level 0 - mask bit mode
template <typename T1, typename T2, bool isSetMask>
__aicore__ inline void SubReluCast(const LocalTensor<T1>& dstLocal, const LocalTensor<T2>& src0Local,
    const LocalTensor<T2>& src1Local, uint64_t mask[], const uint8_t repeatTimes,
    const BinaryRepeatParams& repeatParams)
{
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        return;
    }
#endif
#if ASCENDC_CPU_DEBUG
    MaskSetter::Instance().SetMask(isSetMask);
    if (!CheckFuncVecBinaryCmp(dstLocal, src0Local, src1Local, mask, repeatTimes, repeatParams, "SubReluCast")) {
        ASCENDC_REPORT_CHECK_ERROR("SubReluCast", KernelFuncType::MASK_BIT_MODE);
    }
#endif
    SubReluCastImpl<T1, T2, isSetMask>((__ubuf__ T1*)dstLocal.GetPhyAddr(), (__ubuf__ T2*)src0Local.GetPhyAddr(),
        (__ubuf__ T2*)src1Local.GetPhyAddr(), mask, repeatTimes, repeatParams);
}

/*
 * @ingroup SubReluCast Level 2
 * @brief dst[i] = Precision conversion
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] calcount number Number of data involved in calculation
 */
template <typename T1, typename T2>
__aicore__ inline void SubReluCast(const LocalTensor<T1>& dstLocal, const LocalTensor<T2>& src0Local,
    const LocalTensor<T2>& src1Local, const uint32_t calCount)
{
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        return;
    }
#endif
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncVecBinaryCmp(dstLocal, src0Local, src1Local, calCount, "SubReluCast")) {
        ASCENDC_REPORT_CHECK_ERROR("SubReluCast", KernelFuncType::CALCOUNT_MODE);
    }
#endif
    SubReluCastImpl((__ubuf__ T1*)dstLocal.GetPhyAddr(), (__ubuf__ T2*)src0Local.GetPhyAddr(),
        (__ubuf__ T2*)src1Local.GetPhyAddr(), calCount);
}

#pragma end_pipe
__aicore__ inline void SetDeqScale(half scale)
{
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        return;
    }
#endif
#if ASCENDC_CPU_DEBUG
    AscendC::g_isVdeq = false;
#endif
    SetDeqScaleImpl(scale);
}

__aicore__ inline void SetDeqScale(float scale, int16_t offset, bool signMode)
{
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        return;
    }
#endif
#if ASCENDC_CPU_DEBUG
    AscendC::g_isVdeq = false;
#endif
    SetDeqScaleImpl(scale, offset, signMode);
}

template <typename T>
__aicore__ inline void SetDeqScale(const LocalTensor<T>& vdeqTensor, const VdeqInfo& vdeqInfo)
{
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        return;
    }
#endif
#if ASCENDC_CPU_DEBUG
    AscendC::g_isVdeq = true;
    ASCENDC_ASSERT((SupportType<T, uint64_t>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in SetDeqScale, "
        "current api support dtype combination is vdeq: uint64_t");});
    CheckTensorAlign<T>(vdeqTensor, ONE_BLK_SIZE, "vdeqTensor", "SetDeqScale");
    CheckTensorPos<T>(vdeqTensor, Hardware::UB, "vdeqTensor", "VECIN / VECCALC / VECOUT", "SetDeqScale");
#endif
    SetDeqScaleImpl<T>(vdeqTensor, vdeqInfo);
}

template <bool castMode>
__aicore__ inline void SetCastOverflowMode()
{
    SetCastOverflowModeImpl<castMode>();
}
} // namespace AscendC
#endif // ASCENDC_MODULE_INNER_OPERATOR_VEC_VCONV_INTERFACE_H
