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
 * \file kernel_operator_fixpipe_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_FIXPIPE_IMPL_H
#define ASCENDC_MODULE_OPERATOR_FIXPIPE_IMPL_H
#include "kernel_struct_fixpipe.h"

namespace AscendC {

template <typename T>
__aicore__ inline void SetFixPipeConfigImpl(
    const LocalTensor<T> &reluPre, const LocalTensor<T> &quantPre, bool isUnitFlag = false)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "SetFixPipeConfig");
}

template <typename T, bool setRelu = false>
__aicore__ inline void SetFixPipeConfigImpl(const LocalTensor<T> &preTensor, bool isUnitFlag = false)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "SetFixPipeConfig");
}

__aicore__ inline void SetFixpipeNz2ndFlagImpl(uint16_t ndNum, uint16_t srcNdStride, uint16_t dstNdStride)
{
    (void)(ndNum);
    (void)(srcNdStride);
    (void)(dstNdStride);
    ASCENDC_REPORT_NOT_SUPPORT(false, "SetFixpipeNz2ndFlag");
}

__aicore__ inline void SetFixpipePreQuantFlagImpl(uint64_t config)
{
    (void)(config);
    ASCENDC_REPORT_NOT_SUPPORT(false, "SetFixpipePreQuantFlag");
}

__aicore__ inline void SetFixPipeClipReluImpl(uint64_t config)
{
    (void)(config);
    ASCENDC_REPORT_NOT_SUPPORT(false, "SetFixPipeClipRelu");
}

template <typename T>
__aicore__ inline void SetFixPipeAddrImpl(const LocalTensor<T> &eleWiseTensor, uint16_t c0ChStride)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "SetFixPipeAddr");
}
/* **************************************************************************************************
 * Fixpipe                                             *
 * ************************************************************************************************* */
template <typename DstT, typename SrcT>
[[deprecated("NOTICE: Fixpipe is not deprecated. Currently, Fixpipe is an unsupported API on current device. "
             "Please check your code!")]]
__aicore__ inline void FixpipeL0C2GMImpl(__gm__ DstT* dst, __cc__ SrcT* src, const FixpipeParams<SrcT>& intriParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "Fixpipe from CO1 to GM");
}

template <typename DstT, typename SrcT>
[[deprecated("NOTICE: Fixpipe is not deprecated. Currently, Fixpipe is an unsupported API on current device. "
             "Please check your code!")]]
__aicore__ inline void FixpipeL0C2L1Impl(__cbuf__ DstT* dst, __cc__ SrcT* src, const FixpipeParams<SrcT>& intriParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "Fixpipe from CO1 to C1");
}

// L0C->L1
template <typename DstT, typename SrcT, typename ParamT>
[[deprecated("NOTICE: Fixpipe is not deprecated. Currently, Fixpipe is an unsupported API on current device. "
             "Please check your code!")]]
__aicore__ inline void Fixpipe(const LocalTensor<DstT>& dstLocal, const LocalTensor<SrcT>& srcLocal,
    const FixpipeParams<ParamT>& intriParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "Fixpipe from CO1 to C1");
}
// L0C->L1 deq tensor quant
template <typename DstT, typename SrcT, typename BufT, typename ParamT>
[[deprecated("NOTICE: Fixpipe is not deprecated. Currently, Fixpipe is an unsupported API on current device."
             "Please check your code!")]]
__aicore__ inline void Fixpipe(const LocalTensor<DstT>& dstLocal, const LocalTensor<SrcT>& srcLocal,
    const LocalTensor<BufT>& cbufWorkspace, const FixpipeParams<ParamT>& intriParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "Fixpipe from CO1 to C1");
}

// L0C->GM
template <typename DstT, typename SrcT, typename ParamT>
[[deprecated("NOTICE: Fixpipe is not deprecated. Currently, Fixpipe is an unsupported API on current device. "
             "Please check your code!")]]
__aicore__ inline void Fixpipe(const GlobalTensor<DstT>& dstGlobal, const LocalTensor<SrcT>& srcLocal,
    const FixpipeParams<ParamT>& intriParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "Fixpipe from CO1 to GM");
}

// L0C->GM deq tensor quant
template <typename DstT, typename SrcT, typename BufT, typename ParamT>
[[deprecated("NOTICE: Fixpipe is not deprecated. Currently, Fixpipe is an unsupported API on current device. "
             "Please check your code!")]]
__aicore__ inline void Fixpipe(const GlobalTensor<DstT> &dstGlobal, const LocalTensor<SrcT> &srcLocal,
    const LocalTensor<BufT> &cbufWorkspace, const FixpipeParams<ParamT> &intriParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "Fixpipe from CO1 to GM");
}

template <typename DstT, typename SrcT, const FixpipeConfig& config>
__aicore__ inline void FixpipeL0C2UBImpl(__ubuf__ DstT *dst, __cc__ SrcT *src, const FixpipeParamsV220 &intriParams)
{
    ASCENDC_DEBUG_ASSERT(false, "Fixpipe doesn't support L0C to UB on current device\n");
}

template <typename DstT, typename SrcT, const FixpipeConfig &config>
__aicore__ inline void FixpipeL0C2UBImpl(
    __ubuf__ DstT *dst, __cc__ SrcT *src, __cbuf__ uint64_t *cbufWorkspace, const FixpipeParamsV220 &intriParams)
{
    ASCENDC_DEBUG_ASSERT(false, "Fixpipe doesn't support L0C to UB on current device\n");
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_FIXPIPE_IMPL_H
