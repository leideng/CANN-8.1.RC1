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
 * \file kernel_operator_proposal_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_PROPOSAL_IMPL_H
#define ASCENDC_MODULE_OPERATOR_PROPOSAL_IMPL_H
#include "kernel_operator_proposal_base_impl.h"
#include "kernel_struct_proposal.h"

namespace AscendC {
constexpr uint32_t singleSortElementCountV220 = 32;
constexpr uint32_t singleSortElementCountV200 = 16;
constexpr uint32_t regionProposalDataSize = 8;

template <typename T>
[[deprecated("NOTICE: MrgSort4 is not deprecated. Currently, MrgSort4 is an unsupported API on current device."
             "Please check your code!")]]
__aicore__ inline void Vmrgsort4Cal(__ubuf__ T* dstLocal, __ubuf__ T* addrArray[MRG_SORT_ELEMENT_LEN], uint64_t config)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "MrgSort4");
}

template <typename T>
[[deprecated("NOTICE: RpSort16 is not deprecated. Currently, RpSort16 is an unsupported API on current device."
             "Please check your code!")]]
__aicore__ inline void VbitsortCal(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const ProposalIntriParams& intriParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "RpSort16");
}

template <typename T>
__aicore__ inline void VbitsortCal(__ubuf__ T* dstLocal, __ubuf__ T* src0Local, __ubuf__ uint32_t* src1Local,
    const ProposalIntriParams& intriParams)
{
    if ASCEND_IS_AIV {
        vbitsort(dstLocal, src0Local, src1Local, intriParams.repeat);
    }
}

template <typename T>
__aicore__ inline void Vmrgsort4Cal(__ubuf__ T* dstLocal, __ubuf__ T* addrArray[MRG_SORT_ELEMENT_LEN], uint64_t src1,
    uint64_t config)
{
    if ASCEND_IS_AIV {
        vmrgsort4(dstLocal, addrArray, src1, config);
    }
}

template <typename T>
[[deprecated(
    "NOTICE: ProposalConcat is not deprecated. Currently, ProposalConcat is an unsupported API on current device."
    "Please check your code!")]]
__aicore__ inline void VconcatCal(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const ProposalIntriParams& intriParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "ProposalConcat");
}

template <typename T>
[[deprecated(
    "NOTICE: ProposalExtract is not deprecated. Currently, ProposalExtract is an unsupported API on current device."
    "Please check your code!")]]
__aicore__ inline void VextractCal(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const ProposalIntriParams& intriParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "ProposalExtract");
}

template <typename T>
__aicore__ inline void MrgSortCal(const LocalTensor<T> &dstLocal, const MrgSortSrcList<T> &sortList,
    const uint16_t elementCountList[4], uint32_t sortedNum[4], uint16_t validBit, const int32_t repeatTimes)
{
    MrgSort4Info mrgSortInfo(elementCountList, false, validBit, (uint16_t)repeatTimes);
#if ASCENDC_CPU_DEBUG
    if (!CheckFunProposal(dstLocal, sortList, mrgSortInfo, "MrgSort")) {
        ASCENDC_REPORT_CHECK_ERROR("MrgSort", KernelFuncType::NONE_MODE);
    }
#endif
    uint64_t config = 0;
    config |= (mrgSortInfo.repeatTimes & 0xFF);
    config |= (uint64_t(mrgSortInfo.validBit & 0xF) << 8);
    config |= (uint64_t(mrgSortInfo.ifExhaustedSuspension & 0x1) << 12);

    uint64_t src1 = 0;
    src1 |= (uint64_t(mrgSortInfo.elementLengths[0] & 0xFFFF));
    src1 |= (uint64_t(mrgSortInfo.elementLengths[1] & 0xFFFF) << 16);
    src1 |= (uint64_t(mrgSortInfo.elementLengths[2] & 0xFFFF) << 32);
    src1 |= (uint64_t(mrgSortInfo.elementLengths[3] & 0xFFFF) << 48);

    __ubuf__ T *addrArray[MRG_SORT_ELEMENT_LEN] = {(__ubuf__ T *)sortList.src1.GetPhyAddr(),
        (__ubuf__ T *)sortList.src2.GetPhyAddr(),
        (__ubuf__ T *)sortList.src3.GetPhyAddr(),
        (__ubuf__ T *)sortList.src4.GetPhyAddr()};

    Vmrgsort4Cal((__ubuf__ T *)dstLocal.GetPhyAddr(), addrArray, src1, config);
}

__aicore__ inline void GetMrgSortResultImpl(
    uint16_t &mrgSortList1, uint16_t &mrgSortList2, uint16_t &mrgSortList3, uint16_t &mrgSortList4)
{
    int64_t mrgSortResult = get_vms4_sr();
    constexpr uint64_t resMask = 0xFFFF;
    // VMS4_SR[15:0], number of finished region proposals in list0
    mrgSortList1 = static_cast<uint64_t>(mrgSortResult) & resMask;
    constexpr uint64_t sortList2Bit = 16;
    // VMS4_SR[31:16], number of finished region proposals in list1
    mrgSortList2 = (static_cast<uint64_t>(mrgSortResult) >> sortList2Bit) & resMask;
    constexpr uint64_t sortList3Bit = 32;
    // VMS4_SR[47:32], number of finished region proposals in list2
    mrgSortList3 = (static_cast<uint64_t>(mrgSortResult) >> sortList3Bit) & resMask;
    constexpr uint64_t sortList4Bit = 48;
    // VMS4_SR[63:48], number of finished region proposals in list3
    mrgSortList4 = (static_cast<uint64_t>(mrgSortResult) >> sortList4Bit) & resMask;
}

template <typename T>
__aicore__ inline void FullSortInnerLoop(const LocalTensor<T> &dstLocal, const LocalTensor<T> &tmpLocal,
    const uint32_t baseOffset, const uint16_t singleMergeTmpElementCount, const int32_t mergeTmpRepeatTimes)
{
    if (mergeTmpRepeatTimes <= 0) {
        return;
    }
    MrgSortSrcList sortList =
        MrgSortSrcList(tmpLocal[0], tmpLocal[baseOffset], tmpLocal[2 * baseOffset], tmpLocal[3 * baseOffset]);
    const uint16_t elementCountList[MRG_SORT_ELEMENT_LEN] = {singleMergeTmpElementCount, singleMergeTmpElementCount,
        singleMergeTmpElementCount, singleMergeTmpElementCount};
    uint32_t sortedNum[MRG_SORT_ELEMENT_LEN];
    MrgSortCal<T>(dstLocal, sortList, elementCountList, sortedNum, 0b1111, mergeTmpRepeatTimes);
}

template <typename T>
__aicore__ inline void FullSortInnerLoopTail(const LocalTensor<T> &dstLocal, const LocalTensor<T> &tmpLocal,
    const uint32_t baseOffset, const uint16_t singleMergeTmpElementCount, const uint32_t elementCountTail,
    const int32_t mergeTmpRepeatTimes, int32_t mergeTmpTailQueNum)
{
    if (mergeTmpTailQueNum <= 0) {
        return;
    }
    uint16_t validBitTail;
    uint16_t elementCountListTail[MRG_SORT_ELEMENT_LEN] = {singleMergeTmpElementCount, singleMergeTmpElementCount,
        singleMergeTmpElementCount, singleMergeTmpElementCount};
    uint32_t offset1Tail, offset2Tail, offset3Tail;
    uint32_t offset0Tail = MRG_SORT_ELEMENT_LEN * baseOffset * mergeTmpRepeatTimes;

    ComSortInnerLoopTail(offset0Tail, offset1Tail, offset2Tail, offset3Tail, validBitTail, elementCountListTail,
        baseOffset, elementCountTail, mergeTmpTailQueNum);
    if (mergeTmpTailQueNum > 1) {
        MrgSortSrcList sortListTail = MrgSortSrcList(tmpLocal[offset0Tail], tmpLocal[offset1Tail],
            tmpLocal[offset2Tail], tmpLocal[offset3Tail]);
        uint32_t sortedNumTail[MRG_SORT_ELEMENT_LEN];
        MrgSortCal<T>(dstLocal[offset0Tail], sortListTail, elementCountListTail, sortedNumTail,
            validBitTail, 1);
    } else {
        if constexpr (IsSameType<T, half>::value) {
            DataCopy(dstLocal[offset0Tail], tmpLocal[offset0Tail], elementCountTail * 4);
        } else {
            DataCopy(dstLocal[offset0Tail], tmpLocal[offset0Tail], elementCountTail * 2);
        }
    }
}

__aicore__ inline uint32_t GetFullSortInnerLoopTimes(const int32_t repeatTimes)
{
    uint32_t loopi = 0;
    int32_t queNum = repeatTimes;
    while (queNum > 1) {
        queNum = Ceil(queNum, MRG_SORT_ELEMENT_LEN);
        loopi++;
    }
    return loopi;
}

template <typename T>
__aicore__ inline void DoFullSort(const LocalTensor<T> &dstLocal, const LocalTensor<T> &concatLocal,
    const LocalTensor<uint32_t> &indexLocal, LocalTensor<T> &tmpLocal, const int32_t repeatTimes)
{
    uint32_t elementCount = concatLocal.GetSize();
    uint32_t singleMergeElementCount = singleSortElementCountV220;
    uint32_t loopi = GetFullSortInnerLoopTimes(repeatTimes);
    uint16_t singleMergeTmpElementCount = singleMergeElementCount;
    uint32_t srcLocalElementCount = repeatTimes * singleMergeElementCount;
    uint32_t dstLocalElementCount = srcLocalElementCount * regionProposalDataSize / sizeof(T);
    int32_t mergeTmpTotalQueNum = repeatTimes;
    int32_t mergeTmpTailQueNum = repeatTimes % MRG_SORT_ELEMENT_LEN;
    int32_t mergeTmpQueNum = mergeTmpTotalQueNum - mergeTmpTailQueNum;
    int32_t mergeTmpRepeatTimes = repeatTimes / MRG_SORT_ELEMENT_LEN;
    DataCopy(tmpLocal, dstLocal, dstLocalElementCount);
    PipeBarrier<PIPE_V>();
    for (int i = 0; i < loopi; i++) {
        uint32_t baseOffset;
        baseOffset = singleMergeTmpElementCount * regionProposalDataSize / sizeof(T);
        FullSortInnerLoop(dstLocal, tmpLocal, baseOffset, singleMergeTmpElementCount, mergeTmpRepeatTimes);
        PipeBarrier<PIPE_V>();
        uint16_t elementCountTail = srcLocalElementCount % singleMergeTmpElementCount ?
            srcLocalElementCount % singleMergeTmpElementCount :
            singleMergeTmpElementCount;
        FullSortInnerLoopTail(dstLocal, tmpLocal, baseOffset, singleMergeTmpElementCount, elementCountTail,
            mergeTmpRepeatTimes, mergeTmpTailQueNum);
        PipeBarrier<PIPE_V>();
        DataCopy(tmpLocal, dstLocal, dstLocalElementCount);
        PipeBarrier<PIPE_V>();
        singleMergeTmpElementCount *= MRG_SORT_ELEMENT_LEN;
        mergeTmpTotalQueNum = mergeTmpTotalQueNum % MRG_SORT_ELEMENT_LEN ?
            mergeTmpTotalQueNum / MRG_SORT_ELEMENT_LEN + 1 :
            mergeTmpTotalQueNum / MRG_SORT_ELEMENT_LEN;
        mergeTmpTailQueNum = mergeTmpTotalQueNum % MRG_SORT_ELEMENT_LEN;
        if (mergeTmpTailQueNum == 0 && elementCountTail != singleMergeTmpElementCount) {
            mergeTmpTailQueNum = MRG_SORT_ELEMENT_LEN;
        }
        mergeTmpQueNum = mergeTmpTotalQueNum - mergeTmpTailQueNum;
        mergeTmpRepeatTimes = mergeTmpQueNum / MRG_SORT_ELEMENT_LEN;
    }
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_PROPOSAL_IMPL_H
