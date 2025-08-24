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
 * \file kernel_operator_proposal_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_PROPOSAL_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_PROPOSAL_INTERFACE_H
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
#elif defined(__DAV_M310__)
#include "dav_m310/kernel_operator_proposal_impl.h"
#include "dav_m310/kernel_operator_vec_gather_mask_impl.h"
#endif

#if ASCENDC_CPU_DEBUG
#include "kernel_check.h"
#endif
namespace AscendC {
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
    const MrgSort4Info& params);

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
    const int32_t repeatTimes);

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
    const MrgSort4Info& params);

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
    const LocalTensor<uint32_t>& src1Local, const int32_t repeatTimes);


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
    const int32_t repeatTimes, const int32_t modeNumber);

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
    const int32_t repeatTimes, const int32_t modeNumber);

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
    const LocalTensor<T> &tmpLocal, const int32_t repeatTimes);

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
    const LocalTensor<T> &sortedLocal, const int32_t repeatTimes);

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
template <typename T, bool isExhaustedSuspension = false>
__aicore__ inline void MrgSort(const LocalTensor<T> &dstLocal, const MrgSortSrcList<T> &sortList,
    const uint16_t elementCountList[4], uint32_t sortedNum[4], uint16_t validBit, const int32_t repeatTimes);

/* ***************************************** Sort ****************************************** */
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
    const LocalTensor<uint32_t> &indexLocal, LocalTensor<T> &tmpLocal, const int32_t repeatTimes);

/* **************************************** GetSortOffset ****************************************** */
/*
 * @ingroup GetSortOffset
 * @brief get sort offset in the sorted struct
 * @param [in] elemOffset element number offer
 */
template <typename T>
__aicore__ inline uint32_t GetSortOffset(const uint32_t elemOffset);

/* **************************************** GetSortLen ****************************************** */
/*
 * @ingroup GetSortLen
 * @brief get sort length in the sorted struct
 * @param [in] elemOffset element number ocountffer
 */
template <typename T>
__aicore__ inline uint32_t GetSortLen(const uint32_t elemCount);
#pragma end_pipe
__aicore__ inline __inout_pipe__(S) void GetMrgSortResult(
    uint16_t &mrgSortList1, uint16_t &mrgSortList2, uint16_t &mrgSortList3, uint16_t &mrgSortList4);
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_PROPOSAL_INTERFACE_H
