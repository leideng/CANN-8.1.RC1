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
 * \file inner_kernel_operator_fixpipe_intf.cppm
 * \brief
 */
#ifndef ASCENDC_MODULE_INNER_OPERATOR_FIXPIPE_INTERFACE_H
#define ASCENDC_MODULE_INNER_OPERATOR_FIXPIPE_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_check.h"
#include "kernel_struct_fixpipe.h"

#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_fixpipe_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_fixpipe_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_fixpipe_v2_impl.h"
#include "dav_c220/kernel_operator_fixpipe_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_fixpipe_v2_impl.h"
#include "dav_m300/kernel_operator_fixpipe_impl.h"
#endif

namespace AscendC {
/* **************************************************************************************************
 * Fixpipe                                             *
 * ************************************************************************************************* */
/*
 * @ingroup Fixpipe
 * @brief After calculation, process the results
 * @param [out] dstGlobal output GlobalTensor
 * @param [in] srcLocal input LocalTensor
 * @param [in] intriParams.cburstNum number of burst
 * @param [in] intriParams.burstLen burst length
 * @param [in] intriParams.srcStride src block stride
 * @param [in] intriParams.dstStride dst block stride
 * @param [in] intriParams.biasParams contains isBias flag and bias LocalTensor
 * @param [in] intriParams.quantParams contains quant mode and quant params
 * @param [in] intriParams.reluEn indicates whether to enable the relu function
 * @param [in] intriParams.nz2ndParams contains the input params for enable the nz2nd function
 */

template <typename T>
__aicore__ inline void SetFixPipeConfig(const LocalTensor<T> &reluPre, const LocalTensor<T> &quantPre,
    bool isUnitFlag)
{
    SetFixPipeConfigImpl<T>(reluPre, quantPre, isUnitFlag);
}

template <typename T, bool setRelu>
__aicore__ inline void SetFixPipeConfig(const LocalTensor<T> &preTensor, bool isUnitFlag)
{
    SetFixPipeConfigImpl<T, setRelu>(preTensor, isUnitFlag);
}

__aicore__ inline void SetFixpipeNz2ndFlag(uint16_t ndNum, uint16_t srcNdStride, uint16_t dstNdStride)
{
    SetFixpipeNz2ndFlagImpl(ndNum, srcNdStride, dstNdStride);
}

__aicore__ inline void SetFixpipePreQuantFlag(uint64_t config)
{
    SetFixpipePreQuantFlagImpl(config);
}

__aicore__ inline void SetFixPipeClipRelu(uint64_t config)
{
    SetFixPipeClipReluImpl(config);
}

template <typename T>
__aicore__ inline void SetFixPipeAddr(const LocalTensor<T> &eleWiseTensor, uint16_t c0ChStride)
{
    SetFixPipeAddrImpl(eleWiseTensor, c0ChStride);
}
// L0C -> L1 for v220
template <typename DstT, typename SrcT, const FixpipeConfig &config>
__aicore__ inline void Fixpipe(const LocalTensor<DstT> &dstLocal, const LocalTensor<SrcT> &srcLocal,
    const FixpipeParamsV220 &intriParams)
{
    CheckTensorPos<SrcT>(srcLocal, Hardware::L0C, "srcLocal", "CO1", "Fixpipe");
    ASCENDC_CHECK_TPOSITION((GetPhyType((TPosition)dstLocal.GetPosition()) == Hardware::L1) ||
        (GetPhyType((TPosition)dstLocal.GetPosition()) == Hardware::UB), "dstLocal", "A1", "Fixpipe",
        ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(dstLocal.GetPosition())));

    if ((GetPhyType((TPosition)dstLocal.GetPosition()) == Hardware::L1)) {
        FixpipeL0C2L1Impl<PrimT<DstT>, PrimT<SrcT>, config>((__cbuf__ PrimT<DstT>*)dstLocal.GetPhyAddr(),
            (__cc__ PrimT<SrcT>*)srcLocal.GetPhyAddr(), intriParams);
    } else if ((GetPhyType((TPosition)dstLocal.GetPosition()) == Hardware::UB)) {
        FixpipeL0C2UBImpl<PrimT<DstT>, PrimT<SrcT>, config>((__ubuf__ PrimT<DstT>*)dstLocal.GetPhyAddr(),
            (__cc__ PrimT<SrcT>*)srcLocal.GetPhyAddr(), intriParams);
    }
}

// L0C->L1 deq tensor quant
template <typename DstT, typename SrcT, const FixpipeConfig& config, typename BufT,
    typename std::enable_if<IsSameType<PrimT<BufT>, uint64_t>::value, bool>::type>
__aicore__ inline void Fixpipe(const LocalTensor<DstT>& dstLocal, const LocalTensor<SrcT>& srcLocal,
    const LocalTensor<BufT>& cbufWorkspace, const FixpipeParamsV220& intriParams)
{
    CheckTensorPos<SrcT>(srcLocal, Hardware::L0C, "srcLocal", "CO1", "Fixpipe");
    ASCENDC_CHECK_TPOSITION((GetPhyType((TPosition)dstLocal.GetPosition()) == Hardware::L1) ||
        (GetPhyType((TPosition)dstLocal.GetPosition()) == Hardware::UB), "dstLocal", "A1", "Fixpipe",
        ConstDefiner::Instance().logicNameMap.at(static_cast<uint8_t>(srcLocal.GetPosition())));
    CheckTensorPos<BufT>(cbufWorkspace, Hardware::L1, "cbufWorkspace", "A1", "Fixpipe");
    ASCENDC_ASSERT((intriParams.quantPre == QuantMode_t::VDEQF16 || intriParams.quantPre == QuantMode_t::VQF322B8_PRE ||
        intriParams.quantPre == QuantMode_t::VREQ8), {KERNEL_LOG(KERNEL_ERROR, "Failed to check quantPre value in "
        "Fixpipe, when cbufWorkspace is given, supported values are VDEQF16 / VQF322B8_PRE / VREQ8");});
    if ((GetPhyType((TPosition)dstLocal.GetPosition()) == Hardware::L1)) {
        FixpipeL0C2L1Impl<PrimT<DstT>, PrimT<SrcT>, config>((__cbuf__ PrimT<DstT>*)dstLocal.GetPhyAddr(),
            (__cc__ PrimT<SrcT>*)srcLocal.GetPhyAddr(), (__cbuf__ uint64_t*)cbufWorkspace.GetPhyAddr(), intriParams);
    } else if ((GetPhyType((TPosition)dstLocal.GetPosition()) == Hardware::UB)) {
        FixpipeL0C2UBImpl<PrimT<DstT>, PrimT<SrcT>, config>((__ubuf__ PrimT<DstT>*)dstLocal.GetPhyAddr(),
            (__cc__ PrimT<SrcT>*)srcLocal.GetPhyAddr(), (__cbuf__ uint64_t*)cbufWorkspace.GetPhyAddr(), intriParams);
    }
}

// L0C->GM
template <typename DstT, typename SrcT, const FixpipeConfig &config>
__aicore__ inline void Fixpipe(const GlobalTensor<DstT> &dstGlobal, const LocalTensor<SrcT> &srcLocal,
    const FixpipeParamsV220 &intriParams)
{
#ifdef ASCENDC_CPU_DEBUG
    bool isUsedProcessLock = false;
    if (g_isAtomic == true) {
        ProcessLock::GetProcessLock()->Write();
        isUsedProcessLock = true;
    }
#endif  // ASCENDC_CPU_DEBUG
    CheckTensorPos<SrcT>(srcLocal, Hardware::L0C, "srcLocal", "CO1", "Fixpipe");
    FixpipeL0C2GMImpl<PrimT<DstT>, PrimT<SrcT>, config>((__gm__ PrimT<DstT>*)dstGlobal.GetPhyAddr(),
        (__cc__ PrimT<SrcT>*)srcLocal.GetPhyAddr(),
        intriParams);
#ifdef ASCENDC_CPU_DEBUG
    if (isUsedProcessLock == true) {
        isUsedProcessLock = false;
        ProcessLock::GetProcessLock()->Unlock();
    }
#endif  // ASCENDC_CPU_DEBUG
}

// L0C->GM deq tensor quant
template <typename DstT, typename SrcT, const FixpipeConfig &config, typename BufT,
    typename std::enable_if<IsSameType<PrimT<BufT>, uint64_t>::value, bool>::type>
__aicore__ inline void Fixpipe(const GlobalTensor<DstT> &dstGlobal, const LocalTensor<SrcT> &srcLocal,
    const LocalTensor<BufT> &cbufWorkspace, const FixpipeParamsV220 &intriParams)
{
    CheckTensorPos<SrcT>(srcLocal, Hardware::L0C, "srcLocal", "CO1", "Fixpipe");
    CheckTensorPos<BufT>(cbufWorkspace, Hardware::L1, "cbufWorkspace", "A1", "Fixpipe");
    ASCENDC_ASSERT((intriParams.quantPre == QuantMode_t::VDEQF16 || intriParams.quantPre == QuantMode_t::VQF322B8_PRE ||
        intriParams.quantPre == QuantMode_t::VREQ8), {KERNEL_LOG(KERNEL_ERROR, "Failed to check quantPre value in "
        "Fixpipe, when cbufWorkspace is given, supported values are VDEQF16 / VQF322B8_PRE / VREQ8");});
    FixpipeL0C2GMImpl<PrimT<DstT>, PrimT<SrcT>, config>((__gm__ PrimT<DstT>*)dstGlobal.GetPhyAddr(),
        (__cc__ PrimT<SrcT>*)srcLocal.GetPhyAddr(),
        (__cbuf__ uint64_t*)cbufWorkspace.GetPhyAddr(), intriParams);
}
} // namespace AscendC
#endif // ASCENDC_MODULE_INNER_OPERATOR_FIXPIPE_INTERFACE_H
