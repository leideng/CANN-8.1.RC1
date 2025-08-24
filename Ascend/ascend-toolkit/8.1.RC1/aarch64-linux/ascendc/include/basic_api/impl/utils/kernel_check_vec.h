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
 * \file kernel_check_vec.h
 * \brief
 */
#ifndef ASCENDC_MODULE_CHECK_VEC_H
#define ASCENDC_MODULE_CHECK_VEC_H

#if ASCENDC_CPU_DEBUG
#include "tikcpp_check_util.h"
#include "kernel_common.h"
#include "kernel_struct_unary.h"
#include "kernel_struct_mm.h"

namespace AscendC {
template <typename T, typename U>
bool CheckVectorPadding(const LocalTensor<T>& dstLocal, const LocalTensor<U>& srcLocal, const uint8_t padMode,
    const bool padSide, const uint64_t mask[], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams,
    const char* intriName)
{
    check::VectorPaddingApiParams chkParams {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        repeatTimes, static_cast<uint16_t>(repeatParams.dstBlkStride), static_cast<uint16_t>(repeatParams.srcBlkStride),
        static_cast<uint16_t>(repeatParams.dstRepStride), static_cast<uint16_t>(repeatParams.srcRepStride),
        static_cast<uint32_t>(sizeof(T)), static_cast<uint32_t>(sizeof(U)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)), static_cast<uint64_t>(srcLocal.GetSize() * sizeof(U)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()),
        padMode, padSide};
    return CheckVectorPaddingForMaskArray(chkParams, mask, intriName);
}

template <typename T, typename U>
bool CheckVectorPadding(const LocalTensor<T>& dstLocal, const LocalTensor<U>& srcLocal, const uint8_t padMode,
    const bool padSide, const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams,
    const char* intriName)
{
    check::VectorPaddingApiParams chkParams {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        repeatTimes, static_cast<uint16_t>(repeatParams.dstBlkStride), static_cast<uint16_t>(repeatParams.srcBlkStride),
        static_cast<uint16_t>(repeatParams.dstRepStride), static_cast<uint16_t>(repeatParams.srcRepStride),
        static_cast<uint32_t>(sizeof(T)), static_cast<uint32_t>(sizeof(U)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)), static_cast<uint64_t>(srcLocal.GetSize() * sizeof(U)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()),
        padMode, padSide};
    return CheckVectorPadding(chkParams, mask, intriName);
}

template <typename T, typename U>
bool CheckVectorPadding(const LocalTensor<T>& dstLocal, const LocalTensor<U>& srcLocal, const uint8_t padMode,
    const bool padSide, const uint32_t calCount, const char* intriName)
{
    check::VectorPaddingApiParams chkParams {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(T)), static_cast<uint32_t>(sizeof(U)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)), static_cast<uint64_t>(srcLocal.GetSize() * sizeof(U)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()),
        calCount, padMode, padSide};
    return CheckVectorPadding(chkParams, intriName);
}

template <typename T>
bool CheckFuncLoadDataTranspose(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const LoadData2dTransposeParams &loadDataParams, const char *intriName)
{
#if defined(__DAV_M310__)
    constexpr bool dtypeMatch =
        IsSameType<PrimT<T>, int8_t>::value || IsSameType<PrimT<T>, uint8_t>::value ||
        IsSameType<PrimT<T>, half>::value;
    ASSERT(dtypeMatch && "LoadData2dTransposeParams without dtype of u8/s8/fp16 is not supported on current device");
    return dtypeMatch;
#endif
    return true;
}

template <typename T>
bool CheckFuncLoadDataTranspose(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal,
    const LoadData2dTransposeParamsV2 &loadDataParams, const char *intriName)
{
#if defined(__DAV_M310__)
    bool scopeMatch = (GetPhyType(static_cast<TPosition>(dstLocal.GetPosition())) == Hardware::L0B &&
                       GetPhyType(static_cast<TPosition>(srcLocal.GetPosition())) == Hardware::L1);
    ASSERT(scopeMatch && "LoadDataWithTranspose without B1->B2 is not supported on current device");
    constexpr bool dtypeMatch =
        IsSameType<PrimT<T>, int4b_t>::value || sizeof(PrimT<T>) == sizeof(int8_t) || sizeof(PrimT<T>) == sizeof(half);
    ASSERT(dtypeMatch && "LoadDataWithTranspose is not supported on current device");
    return scopeMatch && dtypeMatch ;
#else
    ASSERT(false && "Current version don't support LoadDataWithTranspose using LoadData2dTransposeParamsV2");
    return false;
#endif
}

template <typename dst_T, typename src0_T, typename src1_T, typename bias_T>
bool CheckMmadParams(const LocalTensor<dst_T>& dstLocal, const LocalTensor<src0_T>& fmLocal,
    const LocalTensor<src1_T>& filterLocal, const LocalTensor<bias_T>& biasLocal, const MmadParams& mmadParams,
    const char* intriName)
{
    check::MmadApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(fmLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(filterLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(biasLocal.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(PrimT<dst_T>)),
        static_cast<uint32_t>(sizeof(PrimT<src0_T>)),
        static_cast<uint32_t>(sizeof(PrimT<src1_T>)),
        static_cast<uint32_t>(sizeof(PrimT<bias_T>)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(PrimT<dst_T>)),
        static_cast<uint64_t>(fmLocal.GetSize() * sizeof(PrimT<src0_T>)),
        static_cast<uint64_t>(filterLocal.GetSize() * sizeof(PrimT<src1_T>)),
        static_cast<uint64_t>(biasLocal.GetSize() * sizeof(PrimT<bias_T>)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(fmLocal.GetPosition()),
        static_cast<uint8_t>(filterLocal.GetPosition()),
        static_cast<uint8_t>(biasLocal.GetPosition()),
        mmadParams.m,
        mmadParams.n,
        mmadParams.k,
        mmadParams.isBias,
        mmadParams.fmOffset,
        mmadParams.enSsparse,
        mmadParams.enWinogradA,
        mmadParams.enWinogradB };
    return CheckFuncMmadImpl(chkParams, intriName);
}
template <typename dst_T, typename src0_T, typename src1_T>
bool CheckMmadParams(const LocalTensor<dst_T>& dstLocal, const LocalTensor<src0_T>& fmLocal,
    const LocalTensor<src1_T>& filterLocal, const MmadParams& mmadParams, const char* intriName)
{
    check::MmadApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(fmLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(filterLocal.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(PrimT<dst_T>)),
        static_cast<uint32_t>(sizeof(PrimT<src0_T>)),
        static_cast<uint32_t>(sizeof(PrimT<src1_T>)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(PrimT<dst_T>)),
        static_cast<uint64_t>(fmLocal.GetSize() * sizeof(PrimT<src0_T>)),
        static_cast<uint64_t>(filterLocal.GetSize() * sizeof(PrimT<src1_T>)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(fmLocal.GetPosition()),
        static_cast<uint8_t>(filterLocal.GetPosition()),
        mmadParams.m,
        mmadParams.n,
        mmadParams.k,
        mmadParams.isBias,
        mmadParams.fmOffset,
        mmadParams.enSsparse,
        mmadParams.enWinogradA,
        mmadParams.enWinogradB };
    return CheckFuncMmadImpl(chkParams, intriName);
}

template <typename T, typename U>
bool CheckFuncBroadCastToMM(const LocalTensor<T>& dstLocal, const LocalTensor<U>& srcLocal, const int32_t blockCount,
    const uint8_t blockLen, const uint8_t srcGap, const uint8_t dstGap, const char* intriName)
{
    check::VecBroadCastToMMApiParams chkParams { static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(PrimT<T>)),
        static_cast<uint32_t>(sizeof(PrimT<U>)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(PrimT<T>)),
        static_cast<uint64_t>(srcLocal.GetSize() * sizeof(PrimT<U>)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()),
        static_cast<uint32_t>(blockCount),
        static_cast<uint8_t>(blockLen),
        static_cast<uint8_t>(srcGap),
        static_cast<uint8_t>(dstGap) };
    return CheckFuncBroadCastToMMImpl(chkParams, intriName);
}

template <typename T>
bool CheckFunVecReduceOther(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const int32_t repeatTimes,
    const int32_t maskCount, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const char* intriName)
{
    check::VecReduceApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        repeatTimes,
        static_cast<uint16_t>(dstRepStride),
        static_cast<uint16_t>(srcBlkStride),
        static_cast<uint16_t>(srcRepStride),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(srcLocal.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()) };
    return CheckFunReduceOtherImpl(chkParams, maskCount, intriName);
}

template <typename T>
bool CheckFunVecReduceOther(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const int32_t repeatTimes,
    const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    const char* intriName)
{
    check::VecReduceApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        repeatTimes,
        static_cast<uint16_t>(dstRepStride),
        static_cast<uint16_t>(srcBlkStride),
        static_cast<uint16_t>(srcRepStride),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(srcLocal.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()) };
    return CheckFunReduceOtherImplForMaskArray(chkParams, mask, intriName);
}

template <typename T>
bool CheckFunVecReduceOtherWhl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const int32_t repeatTimes,
    const int32_t maskCount, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    ReduceOrder order, const char* intriName)
{
    check::VecReduceWhlApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        repeatTimes,
        static_cast<uint16_t>(dstRepStride),
        static_cast<uint16_t>(srcBlkStride),
        static_cast<uint16_t>(srcRepStride),
        order,
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(srcLocal.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()) };
    return CheckFunReduceOtherWhlImpl(chkParams, maskCount, intriName);
}

template <typename T>
bool CheckFunVecReduceOtherWhl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const int32_t repeatTimes,
    const uint64_t mask[], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride,
    ReduceOrder order, const char* intriName)
{
    check::VecReduceWhlApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        repeatTimes,
        static_cast<uint16_t>(dstRepStride),
        static_cast<uint16_t>(srcBlkStride),
        static_cast<uint16_t>(srcRepStride),
        order,
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(srcLocal.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()) };
    return CheckFunReduceOtherWhlImplForMaskArray(chkParams, mask, intriName);
}

template <typename T>
bool CheckFunVecReduce(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const LocalTensor<T>& workLocal,
    const int32_t repeatTimes, const int32_t mask, bool calIndex, const int32_t srcRepStride, const char* intriName)
{
    // max or min level0
    check::VecReduceApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(workLocal.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        repeatTimes,
        calIndex,
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(srcLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(workLocal.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()),
        static_cast<uint8_t>(workLocal.GetPosition()),
        static_cast<uint16_t>(srcRepStride) };
    return CheckFunReduceImpl(chkParams, mask, intriName);
}

template <typename T>
bool CheckFunVecReduce(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const LocalTensor<T>& workLocal,
    const int32_t repeatTimes, const uint64_t mask[], bool calIndex, const int32_t srcRepStride, const char* intriName)
{
    // max or min level0
    check::VecReduceApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(workLocal.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        repeatTimes,
        calIndex,
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(srcLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(workLocal.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()),
        static_cast<uint8_t>(workLocal.GetPosition()),
        static_cast<uint16_t>(srcRepStride) };
    return CheckFunReduceImplForMaskArray(chkParams, mask, intriName);
}

template <typename T>
bool CheckFunVecReduce(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const LocalTensor<T>& workLocal,
    const int32_t repeatTimes, const int32_t mask, const int32_t srcRepStride, const char* intriName)
{
    check::VecReduceApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(workLocal.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        repeatTimes,
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(srcLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(workLocal.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()),
        static_cast<uint8_t>(workLocal.GetPosition()),
        static_cast<uint16_t>(srcRepStride) };
    return CheckFunReduceImpl(chkParams, mask, intriName);
}

template <typename T>
bool CheckFunVecReduce(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const LocalTensor<T>& workLocal,
    const int32_t repeatTimes, const uint64_t mask[], const int32_t srcRepStride, const char* intriName)
{
    // sum level0
    check::VecReduceApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(workLocal.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        repeatTimes,
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(srcLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(workLocal.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()),
        static_cast<uint8_t>(workLocal.GetPosition()),
        static_cast<uint16_t>(srcRepStride) };
    return CheckFunReduceImplForMaskArray(chkParams, mask, intriName);
}

template <typename T>
bool CheckFunVecReduce(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const LocalTensor<T>& workLocal,
    int32_t repeatTimes, const int32_t count, bool calIndex, const char* intriName)
{
    // max or min level2
    check::VecReduceApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(workLocal.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        repeatTimes,
        static_cast<uint32_t>(count),
        calIndex,
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(srcLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(workLocal.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()),
        static_cast<uint8_t>(workLocal.GetPosition()) };
    return CheckFunReduceImpl(chkParams, intriName);
}

template <typename T>
bool CheckFunVecReduce(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const LocalTensor<T>& workLocal,
    const int32_t count, int32_t repeatTimes, const char* intriName)
{
    // sum level 2
    check::VecReduceApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(workLocal.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        repeatTimes,
        static_cast<uint32_t>(count),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(srcLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(workLocal.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()),
        static_cast<uint8_t>(workLocal.GetPosition()) };
    return CheckFunReduceImpl(chkParams, intriName);
}

template <typename T, typename U>
bool CheckFunScatter(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<U>& dstOffsetLocal, const uint32_t dstBaseAddr, const uint64_t mask[],
    const uint8_t repeatTimes, const uint16_t srcRepStride, const char* intriName)
{
    check::VecScatterApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstOffsetLocal.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(U)),
        dstBaseAddr,
        repeatTimes,
        static_cast<uint16_t>(srcRepStride),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(srcLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(dstOffsetLocal.GetSize() * sizeof(U)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()),
        static_cast<uint8_t>(dstOffsetLocal.GetPosition()) };
    return CheckFunScatterImplForMaskArray(chkParams, mask, intriName);
}

template <typename T, typename U>
bool CheckFunScatter(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<U>& dstOffsetLocal, const uint32_t dstBaseAddr, const uint64_t mask,
    const uint8_t repeatTimes, const uint16_t srcRepStride, const char* intriName)
{
    check::VecScatterApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstOffsetLocal.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(U)),
        dstBaseAddr,
        repeatTimes,
        static_cast<uint16_t>(srcRepStride),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(srcLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(dstOffsetLocal.GetSize() * sizeof(U)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()),
        static_cast<uint8_t>(dstOffsetLocal.GetPosition()) };
    return CheckFunScatterImpl(chkParams, mask, intriName);
}

template <typename T, typename U>
bool CheckFunScatter(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const LocalTensor<U>& dstOffsetLocal, const uint32_t dstBaseAddr,
    const uint32_t count, const char* intriName)
{
    check::VecScatterApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstOffsetLocal.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(U)),
        dstBaseAddr,
        static_cast<uint32_t>(count),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(srcLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(dstOffsetLocal.GetSize() * sizeof(U)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()),
        static_cast<uint8_t>(dstOffsetLocal.GetPosition()) };
    return CheckFunScatterImpl(chkParams, intriName);
}
} // namespace AscendC
#endif

#endif