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
 * \file inner_kernel_operator_proposal_intf.cppm
 * \brief
 */
#ifndef ASCENDC_MODULE_INNER_OPERATOR_PROPOSAL_INTERFACE_H
#define ASCENDC_MODULE_INNER_OPERATOR_PROPOSAL_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_struct_proposal.h"

#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_proposal_impl.h"
#include "dav_c100/kernel_operator_vec_gather_mask_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_proposal_impl.h"
#include "dav_m200/kernel_operator_vec_gather_mask_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_proposal_impl.h"
#include "dav_c220/kernel_operator_vec_gather_mask_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_proposal_impl.h"
#include "dav_m300/kernel_operator_vec_gather_mask_impl.h"
#endif

#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif
namespace AscendC {
// for src is fp32, index store in label
// for src is fp16, index store in label + y1, and using GatherMask do extract
constexpr int32_t REGION_PROPOSAL_LABEL_POSITION = 5;
constexpr int32_t REGION_PROPOSAL_Y1_POSITION = 1;
constexpr uint8_t GATHER_MASK_MODE_FOR_INDEX_EVEN = 1;
constexpr uint8_t GATHER_MASK_MODE_FOR_INDEX_ODD = 2;
// gahter mask mode 4 is 00100010: fetch 2nd and 6th elems for each 8 elems
constexpr uint8_t GATHER_MASK_MODE_FOR_EXTRACT_INDEX = 4;
constexpr int32_t REGION_PROPOSAL_SCORE_POSITION = 4;

#pragma begin_pipe(V)
/* **************************************** MrgSort4 ****************************************** */
/*
 * @ingroup MrgSort4
 * @brief Arrange and merge up to four arranged potential queues into one queue
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor list
 * @param [in] filterLocal input LocalTensor
 * @param [in] Params.elementLengths length of proposal list
 * @param [in] Params.ifExhaustedSuspension judge whether to stop after a queue is exhausted
 * @param [in] Params.validBit judge value is valid or not
 * @param [in] Params.repeatTimes repeat times
 */
template <typename T>
__aicore__ inline void MrgSort4(const LocalTensor<T>& dstLocal, const MrgSortSrcList<T>& srcLocal,
    const MrgSort4Info& params)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in MrgSort4, current api support dtype combination is "
        "src and dst both: half / float");});
    for (int8_t i = 0; i < MRG_SORT_ELEMENT_LEN; ++i) {
        ASCENDC_CHECK_VALUE_RANGE(params.elementLengths[i], 0, 4095, "elementLengths", "MrgSort4");
    }
    ASCENDC_ASSERT((params.validBit == 3 || params.validBit == 7 || params.validBit == 15),
        { KERNEL_LOG(KERNEL_ERROR, "Failed to check validBit value in MrgSort4, its valid value is 3 / 7 / 15"); });
#if ASCENDC_CPU_DEBUG
    if (!CheckFunProposal(dstLocal, srcLocal, params, "MrgSort4")) {
        ASCENDC_REPORT_CHECK_ERROR("MrgSort4", KernelFuncType::NONE_MODE);
    }
#endif
    uint64_t config = 0;
    config |= (params.repeatTimes & 0xFF);
    config |= (uint64_t(params.elementLengths[0] & 0xFFF) << 8);
    config |= (uint64_t(params.elementLengths[1] & 0xFFF) << 20);
    config |= (uint64_t(params.elementLengths[2] & 0xFFF) << 32);
    config |= (uint64_t(params.elementLengths[3] & 0xFFF) << 44);
    config |= (uint64_t(params.ifExhaustedSuspension & 0x1) << 59);
    config |= (uint64_t(params.validBit & 0xF) << 60);

    __ubuf__ T *addrArray[MRG_SORT_ELEMENT_LEN] = {(__ubuf__ T *)srcLocal.src1.GetPhyAddr(),
        (__ubuf__ T *)srcLocal.src2.GetPhyAddr(),
        (__ubuf__ T *)srcLocal.src3.GetPhyAddr(),
        (__ubuf__ T *)srcLocal.src4.GetPhyAddr()};
    Vmrgsort4Cal((__ubuf__ T*)dstLocal.GetPhyAddr(), addrArray, config);
}

/* **************************************** RpSort16 ****************************************** */
/*
 * @ingroup RpSort16
 * @brief Sort them according to the score field in the Region Proposals
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] repeatTimes repeat times
 */
template <typename T>
__aicore__ inline void RpSort16(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t repeatTimes)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in RpSort16, current api support dtype combination is "
        "src and dst both: half / float");});
    ASCENDC_CHECK_VALUE_RANGE(repeatTimes, 0, 255, "repeatTimes", "RpSort16");
#if ASCENDC_CPU_DEBUG
    if (!CheckFunProposal(dstLocal, srcLocal, repeatTimes, "RpSort16")) {
        ASCENDC_REPORT_CHECK_ERROR("RpSort16", KernelFuncType::NONE_MODE);
    }
#endif
    struct ProposalIntriParams repeatParams;
    repeatParams.repeat = repeatTimes;
    VbitsortCal((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), repeatParams);
}

/* **************************************** MrgSort ****************************************** */
/*
 * @ingroup MrgSort
 * @brief Arrange and merge up to four arranged potential queues into one queue
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor list
 * @param [in] filterLocal input LocalTensor
 * @param [in] Params.elementLengths length of proposal list
 * @param [in] Params.ifExhaustedSuspension judge whether to stop after a queue is exhausted
 * @param [in] Params.validBit judge value is valid or not
 * @param [in] Params.repeatTimes repeat times
 */
template <typename T>
__aicore__ inline void MrgSort(const LocalTensor<T>& dstLocal, const MrgSortSrcList<T>& srcLocal,
    const MrgSort4Info& params)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in MrgSort, current api support dtype combination is "
        "src and dst both: half / float");});
    for (int8_t i = 0; i < MRG_SORT_ELEMENT_LEN; ++i) {
        ASCENDC_CHECK_VALUE_RANGE(params.elementLengths[i], 0, 4095, "elementLengths", "MrgSort");
    }
    ASCENDC_ASSERT((params.validBit == 3 || params.validBit == 7 || params.validBit == 15),
        { KERNEL_LOG(KERNEL_ERROR, "Failed to check validBit value in MrgSort, its valid value is 3 / 7 / 15"); });
#if ASCENDC_CPU_DEBUG
    if (!CheckFunProposal(dstLocal, srcLocal, params, "MrgSort")) {
        ASCENDC_REPORT_CHECK_ERROR("MrgSort", KernelFuncType::NONE_MODE);
    }
#endif
    uint64_t config = 0;
    config |= (params.repeatTimes & 0xFF);                          // Xt[7:0]: repeat time
    config |= (uint64_t(params.validBit & 0xF) << 8);               // Xt[11:8]: 4-bit mask signal
    config |= (uint64_t(params.ifExhaustedSuspension & 0x1) << 12); // Xt[12]: 1-enable input list exhausted suspension

    uint64_t src1 = 0;
    src1 |= (uint64_t(params.elementLengths[0] & 0xFFFF));
    src1 |= (uint64_t(params.elementLengths[1] & 0xFFFF) << 16);
    src1 |= (uint64_t(params.elementLengths[2] & 0xFFFF) << 32);
    src1 |= (uint64_t(params.elementLengths[3] & 0xFFFF) << 48);

    __ubuf__ T *addrArray[MRG_SORT_ELEMENT_LEN] = {(__ubuf__ T *)srcLocal.src1.GetPhyAddr(),
        (__ubuf__ T *)srcLocal.src2.GetPhyAddr(),
        (__ubuf__ T *)srcLocal.src3.GetPhyAddr(),
        (__ubuf__ T *)srcLocal.src4.GetPhyAddr()};

    Vmrgsort4Cal((__ubuf__ T*)dstLocal.GetPhyAddr(), addrArray, src1, config);
}

/* **************************************** Sort32 ****************************************** */
/*
 * @ingroup Sort32
 * @brief Sort 32 elements
 * @param [out] dstLocal output LocalTensor
 * @param [in] src0Local input LocalTensor
 * @param [in] src1Local input LocalTensor
 * @param [in] repeatTimes repeat times
 */
template <typename T>
__aicore__ inline void Sort32(const LocalTensor<T>& dstLocal, const LocalTensor<T>& src0Local,
    const LocalTensor<uint32_t>& src1Local, const int32_t repeatTimes)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Sort32, current api support dtype combination is "
        "src and dst both: half / float");});
    ASCENDC_CHECK_VALUE_RANGE(repeatTimes, 0, 255, "repeatTimes", "Sort32");
#if ASCENDC_CPU_DEBUG
    if (!CheckFunProposal(dstLocal, src0Local, src1Local, repeatTimes, "Sort32")) {
        ASCENDC_REPORT_CHECK_ERROR("Sort32", KernelFuncType::NONE_MODE);
    }
#endif
    struct ProposalIntriParams repeatParams;
    repeatParams.repeat = repeatTimes;
    VbitsortCal((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)src0Local.GetPhyAddr(),
        (__ubuf__ uint32_t*)src1Local.GetPhyAddr(), repeatParams);
}


/* **************************************** ProposalConcat ****************************************** */
/*
 * @ingroup ProposalConcat
 * @brief Combine continuous elements into corresponding positions in the Region Proposal
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] repeatTimes repeat times
 * @param [in] modeNumbe Position parameter
 */
template <typename T>
__aicore__ inline void ProposalConcat(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t repeatTimes, const int32_t modeNumber)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in ProposalConcat,"
        " current api support dtype combination is src and dst both: half / float");});
    ASCENDC_CHECK_VALUE_RANGE(repeatTimes, 0, 255, "repeatTimes", "ProposalConcat");
    ASCENDC_CHECK_VALUE_RANGE(modeNumber, 0, 5, "modeNumber", "ProposalConcat");
#if ASCENDC_CPU_DEBUG
    if (!CheckFunProposal(dstLocal, srcLocal, repeatTimes, "ProposalConcat")) {
        ASCENDC_REPORT_CHECK_ERROR("ProposalConcat", KernelFuncType::NONE_MODE);
    }
#endif
    struct ProposalIntriParams repeatParams;
    repeatParams.repeat = repeatTimes;
    repeatParams.modeNumber = modeNumber;
    VconcatCal((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), repeatParams);
}

/* **************************************** ProposalExtract ****************************************** */
/*
 * @ingroup ProposalExtract
 * @brief ProposalExtract and rearrange the individual elements in the corresponding position from the Region Proposals
 * @param [out] dstLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] repeatTimes repeat times
 * @param [in] modeNumbe Position parameter
 */
template <typename T>
__aicore__ inline void ProposalExtract(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t repeatTimes, const int32_t modeNumber)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in "
        "ProposalExtract, current api support dtype combination is src and dst both: half / float");});
    ASCENDC_CHECK_VALUE_RANGE(repeatTimes, 0, 255, "repeatTimes", "ProposalExtract");
    ASCENDC_CHECK_VALUE_RANGE(modeNumber, 0, 5, "modeNumber", "ProposalExtract");
#if ASCENDC_CPU_DEBUG
    if (!CheckFunProposal(dstLocal, srcLocal, repeatTimes, "ProposalExtract")) {
        ASCENDC_REPORT_CHECK_ERROR("ProposalExtract", KernelFuncType::NONE_MODE);
    }
#endif
    struct ProposalIntriParams repeatParams;
    repeatParams.repeat = repeatTimes;
    repeatParams.modeNumber = modeNumber;
    VextractCal((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)srcLocal.GetPhyAddr(), repeatParams);
}

/* **************************************** Concat ****************************************** */
/*
 * @ingroup Concat
 * @brief Combine continuous elements into corresponding positions
 * @param [out] concatLocal output LocalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] tmpLocal tmp buffer
 * @param [in] repeatTimes repeat times
 */
template <typename T>
__aicore__ inline void Concat(LocalTensor<T> &concatLocal, const LocalTensor<T> &srcLocal,
    const LocalTensor<T> &tmpLocal, const int32_t repeatTimes)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Concat, "
        "current api support dtype combination is src and dst both: half / float");});
    ASCENDC_CHECK_VALUE_RANGE(repeatTimes, 0, 255, "repeatTimes", "Concat");
#if __CCE_AICORE__ >= 220
    concatLocal = srcLocal;
#elif __CCE_AICORE__ <= 200
    ProposalConcat(tmpLocal, srcLocal, repeatTimes, REGION_PROPOSAL_SCORE_POSITION);
    concatLocal = tmpLocal;
#endif
#if ASCENDC_CPU_DEBUG
    if (!CheckFunProposal(concatLocal, srcLocal, tmpLocal, repeatTimes, "Concat")) {
        ASCENDC_REPORT_CHECK_ERROR("Concat", KernelFuncType::NONE_MODE);
    }
#endif
}

/* **************************************** Extract ****************************************** */
/*
 * @ingroup Extract
 * @brief Extract and rearrange the individual elements in the corresponding position
 * @param [out] dstValueLocal output LocalTensor
 * @param [in] dstIndexLocal output LocalTensor
 * @param [in] sortedLocal input LocalTensor
 * @param [in] repeatTimes repeat times
 */
template <typename T>
__aicore__ inline void Extract(const LocalTensor<T> &dstValueLocal, const LocalTensor<uint32_t> &dstIndexLocal,
    const LocalTensor<T> &sortedLocal, const int32_t repeatTimes)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Extract, "
        "current api support dtype combination is src and dst both: half / float");});
    ASCENDC_CHECK_VALUE_RANGE(repeatTimes, 0, 255, "repeatTimes", "Extract");
#if ASCENDC_CPU_DEBUG
    if (!CheckFunProposal(dstValueLocal, sortedLocal, dstIndexLocal, repeatTimes, "Extract")) {
        ASCENDC_REPORT_CHECK_ERROR("Extract", KernelFuncType::NONE_MODE);
    }
#endif
#if __CCE_AICORE__ >= 220
    uint64_t rsvdCnt;
    if constexpr (IsSameType<T, half>::value) {
        constexpr uint8_t GATHER_MASK_PATTERN_3 = 3;
        constexpr uint8_t GATHER_MASK_PATTERN_2 = 2;
        GatherMaskCal((__ubuf__ T *)dstValueLocal.GetPhyAddr(), (__ubuf__ T *)sortedLocal.GetPhyAddr(),
            GATHER_MASK_PATTERN_3, false, (uint32_t)0, { 1, (uint16_t)repeatTimes, DEFAULT_REPEAT_STRIDE, 0 }, rsvdCnt);
        PipeBarrier<PIPE_V>();
        GatherMaskCal((__ubuf__ uint32_t *)dstIndexLocal.GetPhyAddr(), (__ubuf__ uint32_t *)sortedLocal.GetPhyAddr(),
            GATHER_MASK_PATTERN_2, false, (uint32_t)0, { 1, (uint16_t)(repeatTimes * 2), 8, 0 }, rsvdCnt);
    } else {
        constexpr uint8_t GATHER_MASK_PATTERN_1 = 1;
        constexpr uint8_t GATHER_MASK_PATTERN_2 = 2;
        GatherMaskCal((__ubuf__ T *)dstValueLocal.GetPhyAddr(), (__ubuf__ T *)sortedLocal.GetPhyAddr(),
            GATHER_MASK_PATTERN_1, false, (uint32_t)0, { 1, (uint16_t)repeatTimes, DEFAULT_REPEAT_STRIDE, 0 }, rsvdCnt);
        PipeBarrier<PIPE_V>();
        GatherMaskCal((__ubuf__ uint32_t *)dstIndexLocal.GetPhyAddr(), (__ubuf__ uint32_t *)sortedLocal.GetPhyAddr(),
            GATHER_MASK_PATTERN_2, false, (uint32_t)0, { 1, (uint16_t)repeatTimes, 8, 0 }, rsvdCnt);
    }

#elif __CCE_AICORE__ <= 200
    ProposalExtract(dstValueLocal, sortedLocal, repeatTimes, REGION_PROPOSAL_SCORE_POSITION);
    if (dstIndexLocal.GetSize() != 0) {
        PipeBarrier<PIPE_V>();
        if constexpr (IsSameType<T, half>::value) {
            uint64_t rsvdCnt;
            GatherMaskCal((__ubuf__ T *)dstIndexLocal.GetPhyAddr(), (__ubuf__ T *)sortedLocal.GetPhyAddr(),
                GATHER_MASK_MODE_FOR_EXTRACT_INDEX, false, (uint32_t)0,
                {1, (uint16_t)repeatTimes, DEFAULT_REPEAT_STRIDE, 0}, rsvdCnt);
        } else {
            ProposalExtract(dstIndexLocal.ReinterpretCast<T>(), sortedLocal, repeatTimes,
                            REGION_PROPOSAL_LABEL_POSITION);
        }
    }
#endif
}

/* **************************************** MrgSort ****************************************** */
/*
 * @ingroup MrgSort
 * @brief Arrange and merge up to four arranged potential queues into one queue
 * @param [out] dstLocal output LocalTensor
 * @param [in] sortList input LocalTensor list
 * @param [in] elementCountList input LocalTensor list length
 * @param [in] sortedNum output sorted numbers
 * @param [in] validBit input valid bit
 * @param [in] repeatTimes repeat times
 */
template <typename T, bool isExhaustedSuspension>
__aicore__ inline void MrgSort(const LocalTensor<T> &dstLocal, const MrgSortSrcList<T> &sortList,
    const uint16_t elementCountList[4], uint32_t sortedNum[4], uint16_t validBit, const int32_t repeatTimes)
{
    if ASCEND_IS_AIC {
        return;
    }
    ASCENDC_ASSERT((SupportType<T, half, float>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in MrgSort, current api support dtype combination is "
        "src and dst both: half / float");});
    MrgSort4Info mrgSortInfo(elementCountList, isExhaustedSuspension, validBit, (uint16_t)repeatTimes);
#if __CCE_AICORE__ >= 220
    MrgSort(dstLocal, sortList, mrgSortInfo);
#elif __CCE_AICORE__ <= 200
    MrgSort4(dstLocal, sortList, mrgSortInfo);
#endif
    if (isExhaustedSuspension) {
#if __CCE_AICORE__ == 220
        constexpr uint32_t validBitMask = 0xFFFF;
        constexpr uint32_t shiftBase = 16;     // register is 16 bit per num
#elif __CCE_AICORE__ == 200
        constexpr uint32_t validBitMask = 0x1FFF;
        constexpr uint32_t shiftBase = 13;     // register is 13 bit per num
#else
        constexpr uint32_t validBitMask = 0;
        constexpr uint32_t shiftBase = 0;     // not support
#endif
        auto res = get_vms4_sr();
        sortedNum[0] = res & validBitMask;
        sortedNum[1] = (res >> shiftBase) & validBitMask;
        sortedNum[2] = (res >> (2 * shiftBase)) & validBitMask;
        sortedNum[3] = (res >> (3 * shiftBase)) & validBitMask;
    }
}

/* **************************************** Sort ****************************************** */
/*
 * @ingroup Sort
 * @brief Sort them according to the value
 * @param [out] dstLocal output LocalTensor
 * @param [in] concatLocal input LocalTensor
 * @param [in] indexLocal input LocalTensor
 * @param [in] tmpLocal tmp buffer
 * @param [in] repeatTimes repeat times
 */
template <typename T, bool isFullSort>
__aicore__ inline void Sort(const LocalTensor<T> &dstLocal, const LocalTensor<T> &concatLocal,
    const LocalTensor<uint32_t> &indexLocal, LocalTensor<T> &tmpLocal, const int32_t repeatTimes)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in Sort, current "
        "api support dtype combination is src and dst both: half / float");});
    ASCENDC_CHECK_VALUE_RANGE(repeatTimes, 0, 255, "repeatTimes", "Sort");
#if ASCENDC_CPU_DEBUG
    if (!CheckFuncSort<T, uint32_t, isFullSort>(dstLocal, concatLocal, indexLocal, tmpLocal, repeatTimes, "Sort")) {
        ASCENDC_REPORT_CHECK_ERROR("Sort", KernelFuncType::NONE_MODE);
    }
#endif
#if __CCE_AICORE__ >= 220
    Sort32(dstLocal, concatLocal, indexLocal, repeatTimes);
#elif __CCE_AICORE__ <= 200
    if (indexLocal.GetSize() != 0) {
        if constexpr (IsSameType<T, half>::value) {
            uint64_t rsvdCnt = 0;
            // sort process 16-elem each repeat, while gatherMask process 64-elem(uint32_t) each repeat
            // repeat time for gather mask is 1/4 of sort's repeat time
            // align repeat time to 64-elem
            constexpr uint16_t SORT_ELEM_PER_REPEAT = 16;
            constexpr uint16_t GATHER_ELEM_PER_REPEAT = 64;
            const uint16_t gatherRepTimes = (repeatTimes * SORT_ELEM_PER_REPEAT + GATHER_ELEM_PER_REPEAT - 1) /
                GATHER_ELEM_PER_REPEAT;
            GatherMaskCal((__ubuf__ T *)dstLocal.GetPhyAddr(), (__ubuf__ T *)indexLocal.GetPhyAddr(),
                          GATHER_MASK_MODE_FOR_INDEX_EVEN, false, (uint32_t)0,
                          {1, gatherRepTimes, DEFAULT_REPEAT_STRIDE, 0}, rsvdCnt);
            PipeBarrier<PIPE_V>();
            ProposalConcat(concatLocal, dstLocal, repeatTimes, REGION_PROPOSAL_Y1_POSITION);
            PipeBarrier<PIPE_V>();
            GatherMaskCal((__ubuf__ T *)dstLocal.GetPhyAddr(), (__ubuf__ T *)indexLocal.GetPhyAddr(),
                         GATHER_MASK_MODE_FOR_INDEX_ODD, false, (uint32_t)0,
                         {1, gatherRepTimes, DEFAULT_REPEAT_STRIDE, 0}, rsvdCnt);
            PipeBarrier<PIPE_V>();
            ProposalConcat(concatLocal, dstLocal, repeatTimes, REGION_PROPOSAL_LABEL_POSITION);
        } else {
            ProposalConcat(concatLocal, indexLocal.ReinterpretCast<T>(), (uint16_t)repeatTimes,
                           REGION_PROPOSAL_LABEL_POSITION);
        }
        PipeBarrier<PIPE_V>();
    }
    RpSort16(dstLocal, concatLocal, repeatTimes);
#endif
    if constexpr (isFullSort) {
        PipeBarrier<PIPE_V>();
        DoFullSort(dstLocal, concatLocal, indexLocal, tmpLocal, repeatTimes);
    }
}

constexpr uint32_t halfSortedDataSize = 4;
constexpr uint32_t floatSortedDataSize = 2;
/* **************************************** GetSortOffset ****************************************** */
/*
 * @ingroup GetSortOffset
 * @brief get sort offset in the sorted struct
 * @param [in] elemOffset element number offer
 */
template <typename T>
__aicore__ inline uint32_t GetSortOffset(const uint32_t elemOffset)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in GetSortOffset, current api support dtype combination is "
        "half / float");});
#if __CCE_AICORE__>= 220
    if constexpr (IsSameType<T, half>::value) {
        return elemOffset * halfSortedDataSize;
    } else {
        return elemOffset * floatSortedDataSize;
    }
#elif __CCE_AICORE__ <= 200
    return elemOffset * regionProposalDataSize;
#endif
}

/* **************************************** GetSortLen ****************************************** */
/*
 * @ingroup GetSortLen
 * @brief get sort length in the sorted struct
 * @param [in] elemOffset element number ocountffer
 */
template <typename T>
__aicore__ inline uint32_t GetSortLen(const uint32_t elemCount)
{
    ASCENDC_ASSERT((SupportType<T, half, float>()),
        {KERNEL_LOG(KERNEL_ERROR, "Failed to check dtype in GetSortLen, current api support dtype combination is "
        "half / float");});
#if __CCE_AICORE__>= 220
    if constexpr (IsSameType<T, half>::value) {
        return elemCount * halfSortedDataSize;
    } else {
        return elemCount * floatSortedDataSize;
    }
#elif __CCE_AICORE__ <= 200
    return elemCount * regionProposalDataSize;
#endif
}
#pragma end_pipe
__aicore__ inline __inout_pipe__(S) void GetMrgSortResult(
    uint16_t &mrgSortList1, uint16_t &mrgSortList2, uint16_t &mrgSortList3, uint16_t &mrgSortList4)
{
#if __CCE_AICORE__ == 220
    if (g_coreType == AIC) {
        return;
    }
#endif
    GetMrgSortResultImpl(mrgSortList1, mrgSortList2, mrgSortList3, mrgSortList4);
}
} // namespace AscendC
#endif // ASCENDC_MODULE_INNER_OPERATOR_PROPOSAL_INTERFACE_H
