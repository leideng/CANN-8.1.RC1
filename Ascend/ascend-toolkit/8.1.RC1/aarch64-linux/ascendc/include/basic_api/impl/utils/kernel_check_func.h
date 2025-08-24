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
 * \file kernel_check_func.h
 * \brief
 */
#ifndef ASCENDC_MODULE_CHECK_FUNC_H
#define ASCENDC_MODULE_CHECK_FUNC_H

#if ASCENDC_CPU_DEBUG
#include "tikcpp_check_util.h"
#include "kernel_common.h"
#include "kernel_struct_brcb.h"
#include "kernel_struct_gather.h"
#include "kernel_struct_transpose.h"
#include "kernel_struct_proposal.h"

namespace AscendC {
template <typename T>
bool CheckFunDup(const LocalTensor<T>& dstLocal, const uint64_t mask, const uint8_t repeatTimes,
    const uint16_t dstBlockStride, const uint8_t dstRepeatStride, const char* intriName)
{
    check::VecDupApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        repeatTimes,
        static_cast<uint16_t>(dstBlockStride),
        static_cast<uint16_t>(dstRepeatStride),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()) };
    return CheckFunDupImpl(chkParams, mask, intriName);
}

template <typename T>
bool CheckFunDup(const LocalTensor<T>& dstLocal, const uint64_t mask[], const uint8_t repeatTimes,
    const uint16_t dstBlockStride, const uint8_t dstRepeatStride, const char* intriName)
{
    check::VecDupApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        repeatTimes,
        static_cast<uint16_t>(dstBlockStride),
        static_cast<uint16_t>(dstRepeatStride),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()) };
    return CheckFunDupImplForMaskArray(chkParams, mask, intriName);
}

template <typename T> bool CheckFunDup(const LocalTensor<T>& dstLocal, const int32_t& calCount, const char* intriName)
{
    check::VecDupApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint32_t>(sizeof(T)), static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint32_t>(calCount) };
    return CheckFunDupImpl(chkParams, intriName);
}

template <typename T>
bool CheckFunBcB(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local, const uint8_t repeatTimes,
    const BrcbRepeatParams& repeatParams, const char* intriName)
{
    check::VecBroadCastApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0Local.GetPhyAddr())),
        repeatTimes,
        static_cast<uint16_t>(repeatParams.dstBlkStride),
        static_cast<uint16_t>(repeatParams.dstRepStride),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(src0Local.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(src0Local.GetPosition()) };
    return CheckFunBcBImpl(chkParams, sizeof(T), intriName);
}

template <typename T, typename U>
bool CheckFuncGatherb(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local, const LocalTensor<U>& offsetLocal,
    const uint8_t repeatTimes, const GatherRepeatParams& repeatParams, const char* intriName)
{
    check::VecGatherApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0Local.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(offsetLocal.GetPhyAddr())),
        repeatTimes,
        static_cast<uint16_t>(repeatParams.dstBlkStride),
        static_cast<uint16_t>(repeatParams.dstRepStride),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(U)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(src0Local.GetSize() * sizeof(T)),
        static_cast<uint64_t>(offsetLocal.GetSize() * sizeof(U)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(src0Local.GetPosition()),
        static_cast<uint8_t>(offsetLocal.GetPosition()) };
    return CheckFuncGatherbImpl(chkParams, sizeof(T), intriName);
}

template <typename T, typename U>
bool CheckFuncGather(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local, const LocalTensor<U>& offsetLocal,
    const uint32_t srcBaseAddr, const uint64_t mask, const uint8_t repeatTimes, const uint16_t dstRepStride,
    const char* intriName)
{
    check::VecGatherApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0Local.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(offsetLocal.GetPhyAddr())),
        srcBaseAddr,
        repeatTimes,
        static_cast<uint16_t>(1),
        dstRepStride,
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(U)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(src0Local.GetSize() * sizeof(T)),
        static_cast<uint64_t>(offsetLocal.GetSize() * sizeof(U)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(src0Local.GetPosition()),
        static_cast<uint8_t>(offsetLocal.GetPosition()) };
    return CheckFuncGatherImpl(chkParams, mask, intriName);
}

template <typename T, typename U>
bool CheckFuncGather(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local, const LocalTensor<U>& offsetLocal,
    const uint32_t srcBaseAddr, const uint64_t mask[], const uint8_t repeatTimes, const uint16_t dstRepStride,
    const char* intriName)
{
    check::VecGatherApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0Local.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(offsetLocal.GetPhyAddr())),
        srcBaseAddr,
        repeatTimes,
        static_cast<uint16_t>(1),
        dstRepStride,
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(U)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(src0Local.GetSize() * sizeof(T)),
        static_cast<uint64_t>(offsetLocal.GetSize() * sizeof(U)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(src0Local.GetPosition()),
        static_cast<uint8_t>(offsetLocal.GetPosition()) };
    return CheckFuncGatherImpl(chkParams, mask, intriName);
}

template <typename T, typename U>
bool CheckFuncGather(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local, const LocalTensor<U>& offsetLocal,
    const uint32_t srcBaseAddr, const uint32_t calCount, const char* intriName)
{
    check::VecGatherApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0Local.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(offsetLocal.GetPhyAddr())),
        srcBaseAddr,
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(U)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(src0Local.GetSize() * sizeof(T)),
        static_cast<uint64_t>(offsetLocal.GetSize() * sizeof(U)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(src0Local.GetPosition()),
        static_cast<uint8_t>(offsetLocal.GetPosition()),
        calCount };
    return CheckFuncGatherImpl(chkParams, intriName);
}

template <typename T>
bool CheckFuncCreateVecIndex(const LocalTensor<T>& dstLocal, const uint64_t mask, const uint8_t repeatTimes,
    const uint16_t dstBlkStride, const uint16_t dstRepStride, const char* intriName)
{
    check::VecCreateVecIndexApiParams chkParams {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        repeatTimes,
        dstBlkStride,
        dstRepStride,
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint32_t>(0) };
    return CheckFuncCreateVecIndexImpl(chkParams, mask, intriName);
}

template <typename T>
bool CheckFuncCreateVecIndex(const LocalTensor<T>& dstLocal, const uint64_t mask[], const uint8_t repeatTimes,
    const uint16_t dstBlkStride, const uint16_t dstRepStride, const char* intriName)
{
    check::VecCreateVecIndexApiParams chkParams {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        repeatTimes,
        dstBlkStride,
        dstRepStride,
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint32_t>(0)};
    return CheckFuncCreateVecIndexImpl(chkParams, mask, intriName);
}

template <typename T>
bool CheckFuncCreateVecIndex(const LocalTensor<T>& dstLocal, const uint32_t calCount, const char* intriName)
{
    check::VecCreateVecIndexApiParams chkParams {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint8_t>(1),
        static_cast<uint16_t>(1),
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        calCount};
    return CheckFuncCreateVecIndexImpl(chkParams, intriName);
}

template <typename T>
bool CheckFuncInitConstValue(const LocalTensor<T>& dstLocal, const uint16_t repeatTimes, const uint16_t blockNum,
    const uint16_t dstGap, const char* intriName)
{
    check::CubeInitConstValueApiParams chkParams {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        repeatTimes,
        blockNum,
        dstGap,
        static_cast<uint32_t>(sizeof(PrimT<T>)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(PrimT<T>)),
        static_cast<uint8_t>(dstLocal.GetPosition()) };
    return CheckFuncInitConstValueImpl(chkParams, intriName);
}

template <typename T, typename U>
bool CheckFuncBilinearInterpolation(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<U>& src0OffsetLocal, const LocalTensor<T>& src1Local, const uint64_t mask[],
    const uint8_t hRepeat, const bool repeatMode, const uint16_t dstBlkStride, const uint16_t vROffset,
    const uint8_t vRepeat, const char* intriName)
{
    check::VecBilinearInterpolationApiParams chkParams {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0Local.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0OffsetLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src1Local.GetPhyAddr())),
        hRepeat,
        repeatMode,
        dstBlkStride,
        vROffset,
        vRepeat,
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(U)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(src0Local.GetSize() * sizeof(T)),
        static_cast<uint64_t>(src0OffsetLocal.GetSize() * sizeof(U)),
        static_cast<uint64_t>(src1Local.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(src0Local.GetPosition()),
        static_cast<uint8_t>(src0OffsetLocal.GetPosition()),
        static_cast<uint8_t>(src1Local.GetPosition()) };
    return CheckFuncBilinearInterpolationImpl(chkParams, mask, intriName);
}

template <typename T, typename U>
bool CheckFuncBilinearInterpolation(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<U>& src0OffsetLocal, const LocalTensor<T>& src1Local, const uint64_t mask, const uint8_t hRepeat,
    const bool repeatMode, const uint16_t dstBlkStride, const uint16_t vROffset, const uint8_t vRepeat,
    const char* intriName)
{
    check::VecBilinearInterpolationApiParams chkParams {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0Local.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0OffsetLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src1Local.GetPhyAddr())),
        hRepeat,
        repeatMode,
        dstBlkStride,
        vROffset,
        vRepeat,
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(U)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(src0Local.GetSize() * sizeof(T)),
        static_cast<uint64_t>(src0OffsetLocal.GetSize() * sizeof(U)),
        static_cast<uint64_t>(src1Local.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(src0Local.GetPosition()),
        static_cast<uint8_t>(src0OffsetLocal.GetPosition()),
        static_cast<uint8_t>(src1Local.GetPosition()) };
    return CheckFuncBilinearInterpolationImpl(chkParams, mask, intriName);
}

template <typename T>
bool CheckFunTranspose(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local, const char* intriName)
{
    const uint16_t defualtStride = 16;
    check::VecTransposeApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0Local.GetPhyAddr())),
        1,
        defualtStride,
        defualtStride,
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(src0Local.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(src0Local.GetPosition()) };
    return CheckFunTransposeImpl(chkParams, intriName);
}

template <typename T, typename U>
bool CheckFunTranspose(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<U> &sharedTmpBuffer, const TransposeParamsExt &transposeParams, const char* intriName)
{
    const uint16_t defualtStride = 16;
    check::VecTransposeApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0Local.GetPhyAddr())),
        1,
        defualtStride,
        defualtStride,
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(src0Local.GetSize() * sizeof(T)),
        static_cast<uint64_t>(sharedTmpBuffer.GetSize() * sizeof(U)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(src0Local.GetPosition()),
        transposeParams.nSize, transposeParams.cSize, transposeParams.hSize, transposeParams.wSize,
        transposeParams.transposeType};
    return CheckFunTransposeImpl(chkParams, intriName);
}

template <typename T, typename U>
bool CheckFunTransDataTo5HD(const LocalTensor<U> &dstLocal, const LocalTensor<U> src0Local,
    const TransDataTo5HDParams& nchwconvParams, const char* intriName)
{
    check::VecTransposeApiParams chkParams {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0Local.GetPhyAddr())),
        nchwconvParams.repeatTimes,
        nchwconvParams.dstRepStride,
        nchwconvParams.srcRepStride,
        static_cast<uint32_t>(sizeof(U)),
        static_cast<uint32_t>(sizeof(U)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(U)),
        static_cast<uint64_t>(src0Local.GetSize() * sizeof(U)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(src0Local.GetPosition()) };
    return CheckFunTransposeImpl(chkParams, intriName);
}

template <typename T>
bool CheckFunTransDataTo5HD(const LocalTensor<T> (&dstLocal)[16], const LocalTensor<T> (&src0Local)[16],
    const TransDataTo5HDParams& nchwconvParams, const char* intriName)
{
    const int8_t dataNum = 16;
    for (int8_t i = 0; i < dataNum; i++) {
        check::VecTransposeApiParams chkParams {
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal[i].GetPhyAddr())),
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0Local[i].GetPhyAddr())),
            nchwconvParams.repeatTimes,
            nchwconvParams.dstRepStride,
            nchwconvParams.srcRepStride,
            static_cast<uint32_t>(sizeof(T)),
            static_cast<uint32_t>(sizeof(T)),
            static_cast<uint64_t>(dstLocal[i].GetSize() * sizeof(T)),
            static_cast<uint64_t>(src0Local[i].GetSize() * sizeof(T)),
            static_cast<uint8_t>(dstLocal[i].GetPosition()),
            static_cast<uint8_t>(src0Local[i].GetPosition()),
            i};
        if (!CheckFunTransposeImpl(chkParams, intriName)) {
            return false;
        }
    }
    return true;
}

template <typename T>
bool CheckFunProposal(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const int32_t repeatTimes,
    const char* intriName)
{
    check::VecProposalApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.GetPhyAddr())),
        static_cast<uint8_t>(repeatTimes),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(srcLocal.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.GetPosition()) };
    return CheckFunProposalImpl(chkParams, intriName);
}

template <typename T, typename U>
bool CheckFunProposal(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local, const LocalTensor<U>& src1Local,
    const int32_t repeatTimes, const char* intriName)
{
    check::VecProposalApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0Local.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src1Local.GetPhyAddr())),
        static_cast<uint8_t>(repeatTimes),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(U)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(src0Local.GetSize() * sizeof(T)),
        static_cast<uint64_t>(src1Local.GetSize() * sizeof(U)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(src0Local.GetPosition()),
        static_cast<uint8_t>(src1Local.GetPosition()) };
    return CheckFunProposalImpl(chkParams, intriName);
}

template <typename T>
bool CheckFunProposal(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local, const LocalTensor<T>& src1Local,
    const int32_t repeatTimes, const char* intriName)
{
    check::VecProposalApiParams chkParams { static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src0Local.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src1Local.GetPhyAddr())),
        static_cast<uint8_t>(repeatTimes),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(src0Local.GetSize() * sizeof(T)),
        static_cast<uint64_t>(src1Local.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(src0Local.GetPosition()),
        static_cast<uint8_t>(src1Local.GetPosition()) };
    return CheckFunProposalImpl(chkParams, intriName);
}

template <typename T>
bool CheckFunProposal(const LocalTensor<T>& dstLocal, const MrgSortSrcList<T>& srcLocal, const MrgSort4Info& params,
    const char* intriName)
{
    uint64_t dstAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr()));
    uint64_t src1Addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.src1.GetPhyAddr()));
    uint64_t src2Addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.src2.GetPhyAddr()));
    uint64_t src3Addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.src3.GetPhyAddr()));
    uint64_t src4Addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcLocal.src4.GetPhyAddr()));
    bool isContinuous = ((src1Addr + params.elementLengths[0] * sizeof(T)) == src2Addr) &&
        ((src2Addr + params.elementLengths[1] * sizeof(T)) == src3Addr) &&
        ((src3Addr + params.elementLengths[2] * sizeof(T)) == src4Addr);
    check::VecProposalApiParams chkParams {dstAddr, src1Addr, static_cast<uint8_t>(params.repeatTimes),
        static_cast<uint32_t>(sizeof(T)), static_cast<uint32_t>(sizeof(T)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(srcLocal.src1.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(srcLocal.src1.GetPosition()),
        params.validBit, params.elementLengths, 1, params.ifExhaustedSuspension, isContinuous};
    bool res = CheckFunProposalImpl(chkParams, intriName);               // src1 res
    chkParams.src0Addr = src2Addr;
    chkParams.src0Size = srcLocal.src2.GetSize() * sizeof(T);
    chkParams.src0LogicPos = static_cast<uint8_t>(srcLocal.src2.GetPosition());
    chkParams.src0Pos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(srcLocal.src2.GetPosition())));
    chkParams.srcIndex = 2;                                              // 2 means src2
    res = res && CheckFunProposalImpl(chkParams, intriName);             // src2 res
    if (params.validBit == 7 || params.validBit == 15) {                 // 7, 15 means need calculate src3 res
        chkParams.src0Addr = src3Addr;
        chkParams.src0Size = srcLocal.src3.GetSize() * sizeof(T);
        chkParams.src0LogicPos = static_cast<uint8_t>(srcLocal.src3.GetPosition());
        chkParams.src0Pos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(srcLocal.src3.GetPosition())));
        chkParams.srcIndex = 3;                                          // 3 means src3
        res = res && CheckFunProposalImpl(chkParams, intriName);         // src3 res
    }
    if (params.validBit == 15) {                                         // 15 means need calculate src4 res
        chkParams.src0Addr = src4Addr;
        chkParams.src0Size = srcLocal.src4.GetSize() * sizeof(T);
        chkParams.src0LogicPos = static_cast<uint8_t>(srcLocal.src4.GetPosition());
        chkParams.src0Pos = static_cast<uint8_t>(GetPhyType(static_cast<TPosition>(srcLocal.src4.GetPosition())));
        chkParams.srcIndex = 4;                                          // 4 means src4
        res = res && CheckFunProposalImpl(chkParams, intriName);         // src4 res
    }
    return res;
}

template <typename T, typename U, bool isFullSort>
bool CheckFuncSort(const LocalTensor<T>& dstLocal, const LocalTensor<T>& concatLocal, const LocalTensor<U>& indexLocal,
    const LocalTensor<T>& tmpLocal, const int32_t repeatTimes, const char* intriName)
{
    check::SortApiParams chkParams {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(concatLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(indexLocal.GetPhyAddr())),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tmpLocal.GetPhyAddr())),
        static_cast<uint8_t>(repeatTimes),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint32_t>(sizeof(U)),
        static_cast<uint32_t>(sizeof(T)),
        static_cast<uint64_t>(dstLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(concatLocal.GetSize() * sizeof(T)),
        static_cast<uint64_t>(indexLocal.GetSize() * sizeof(U)),
        static_cast<uint64_t>(tmpLocal.GetSize() * sizeof(T)),
        static_cast<uint8_t>(dstLocal.GetPosition()),
        static_cast<uint8_t>(concatLocal.GetPosition()),
        static_cast<uint8_t>(indexLocal.GetPosition()),
        static_cast<uint8_t>(tmpLocal.GetPosition()),
        isFullSort};
    return CheckSortImpl(chkParams, intriName);
}
} // namespace AscendC
#endif

#endif