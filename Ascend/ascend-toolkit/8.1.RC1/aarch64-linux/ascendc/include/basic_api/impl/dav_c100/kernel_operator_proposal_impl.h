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

namespace AscendC {
constexpr uint32_t regionProposalDataSize = 8;

template <typename T>
__aicore__ inline void Vmrgsort4Cal(__ubuf__ T* dstLocal, __ubuf__ T* addrArray[MRG_SORT_ELEMENT_LEN], uint64_t config)
{
    vmrgsort4(dstLocal, addrArray, config);
}

template <typename T>
__aicore__ inline void VbitsortCal(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const ProposalIntriParams& intriParams)
{
    vbitsort(dstLocal, srcLocal, intriParams.repeat);
}

template <typename T>
__aicore__ inline void VbitsortCal(__ubuf__ T* dstLocal, __ubuf__ T* src0Local, __ubuf__ uint32_t* src1Local,
    const ProposalIntriParams& intriParams)
{
    vbitsort(dstLocal, src0Local, src1Local, intriParams.repeat);
}

template <typename T>
__aicore__ inline void Vmrgsort4Cal(__ubuf__ T* dstLocal, __ubuf__ T* addrArray[MRG_SORT_ELEMENT_LEN], uint64_t src1,
    uint64_t config)
{
    vmrgsort4(dstLocal, addrArray, src1, config);
}

template <typename T>
__aicore__ inline void VconcatCal(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const ProposalIntriParams& intriParams)
{
    vconcat(dstLocal, srcLocal, intriParams.repeat, intriParams.modeNumber);
}

template <typename T>
__aicore__ inline void VextractCal(__ubuf__ T* dstLocal, __ubuf__ T* srcLocal, const ProposalIntriParams& intriParams)
{
    vextract(dstLocal, srcLocal, intriParams.repeat, intriParams.modeNumber);
}

template <typename T>
__aicore__ inline void DoFullSort(const LocalTensor<T> &dstLocal, const LocalTensor<T> &concatLocal,
    const LocalTensor<uint32_t> &indexLocal, LocalTensor<T> &tmpLocal, const int32_t repeatTimes)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "Sort");
}

__aicore__ inline void GetMrgSortResultImpl(
    uint16_t &mrgSortList1, uint16_t &mrgSortList2, uint16_t &mrgSortList3, uint16_t &mrgSortList4)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "GetMrgSortResult");
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_PROPOSAL_IMPL_H
