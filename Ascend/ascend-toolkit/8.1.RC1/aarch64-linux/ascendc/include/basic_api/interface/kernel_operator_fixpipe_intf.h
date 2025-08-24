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
 * \file kernel_operator_fixpipe_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_FIXPIPE_INTERFACE_H
#define ASCENDC_MODULE_OPERATOR_FIXPIPE_INTERFACE_H
#include "kernel_tensor.h"
#include "kernel_struct_fixpipe.h"

#if __CCE_AICORE__ == 100
#include "dav_c100/kernel_operator_fixpipe_impl.h"
#elif __CCE_AICORE__ == 200
#include "dav_m200/kernel_operator_fixpipe_impl.h"
#elif __CCE_AICORE__ == 220
#include "dav_c220/kernel_operator_fixpipe_impl.h"
#include "dav_c220/kernel_operator_fixpipe_v2_impl.h"
#elif __CCE_AICORE__ == 300
#include "dav_m300/kernel_operator_fixpipe_impl.h"
#include "dav_m300/kernel_operator_fixpipe_v2_impl.h"
#elif defined(__DAV_M310__)
#include "dav_m310/kernel_operator_fixpipe_impl.h"
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
    bool isUnitFlag = false);

template <typename T, bool setRelu = false>
__aicore__ inline void SetFixPipeConfig(const LocalTensor<T> &preTensor, bool isUnitFlag = false);

__aicore__ inline void SetFixpipeNz2ndFlag(uint16_t ndNum, uint16_t srcNdStride, uint16_t dstNdStride);

__aicore__ inline void SetFixpipePreQuantFlag(uint64_t config);

__aicore__ inline void SetFixPipeClipRelu(uint64_t config);

template <typename T>
__aicore__ inline void SetFixPipeAddr(const LocalTensor<T> &eleWiseTensor, uint16_t c0ChStride);

#if __CCE_AICORE__ == 220
// L0C->L1
template <typename DstT, typename SrcT, const FixpipeConfig& config = CFG_ROW_MAJOR>
__aicore__ inline void Fixpipe(const LocalTensor<DstT>& dstLocal, const LocalTensor<SrcT>& srcLocal,
    const FixpipeParamsV220& intriParams);

// L0C->L1 deq tensor quant
template <typename DstT, typename SrcT, const FixpipeConfig& config = CFG_ROW_MAJOR, typename BufT = uint64_t,
    typename std::enable_if<IsSameType<PrimT<BufT>, uint64_t>::value, bool>::type = true>
__aicore__ inline void Fixpipe(const LocalTensor<DstT>& dstLocal, const LocalTensor<SrcT>& srcLocal,
    const LocalTensor<BufT>& cbufWorkspace, const FixpipeParamsV220& intriParams);

// L0C->GM
template <typename DstT, typename SrcT, const FixpipeConfig& config = CFG_ROW_MAJOR>
__aicore__ inline void Fixpipe(const GlobalTensor<DstT>& dstGlobal, const LocalTensor<SrcT>& srcLocal,
    const FixpipeParamsV220& intriParams);

// L0C->GM deq tensor quant
template <typename DstT, typename SrcT, const FixpipeConfig& config = CFG_ROW_MAJOR, typename BufT = uint64_t,
    typename std::enable_if<IsSameType<PrimT<BufT>, uint64_t>::value, bool>::type = true>
__aicore__ inline void Fixpipe(const GlobalTensor<DstT>& dstGlobal, const LocalTensor<SrcT>& srcLocal,
    const LocalTensor<BufT>& cbufWorkspace, const FixpipeParamsV220& intriParams);
#elif __CCE_AICORE__ == 300
// L0C->L1/UB
template <typename DstT, typename SrcT, const FixpipeConfig& config = CFG_ROW_MAJOR>
__aicore__ inline void Fixpipe(const LocalTensor<DstT>& dstLocal, const LocalTensor<SrcT>& srcLocal,
    const FixpipeParamsM300& intriParams);

// L0C->L1/UB deq tensor quant
template <typename DstT, typename SrcT, const FixpipeConfig& config = CFG_ROW_MAJOR, typename BufT = uint64_t,
    typename std::enable_if<IsSameType<PrimT<BufT>, uint64_t>::value, bool>::type = true>
__aicore__ inline void Fixpipe(const LocalTensor<DstT>& dstLocal, const LocalTensor<SrcT>& srcLocal,
    const LocalTensor<BufT>& cbufWorkspace, const FixpipeParamsM300& intriParams);

// L0C->GM
template <typename DstT, typename SrcT, const FixpipeConfig& config = CFG_ROW_MAJOR>
__aicore__ inline void Fixpipe(const GlobalTensor<DstT>& dstGlobal, const LocalTensor<SrcT>& srcLocal,
    const FixpipeParamsM300& intriParams);

// L0C->GM deq tensor quant
template <typename DstT, typename SrcT, const FixpipeConfig& config = CFG_ROW_MAJOR, typename BufT = uint64_t,
    typename std::enable_if<IsSameType<PrimT<BufT>, uint64_t>::value, bool>::type = true>
__aicore__ inline void Fixpipe(const GlobalTensor<DstT>& dstGlobal, const LocalTensor<SrcT>& srcLocal,
    const LocalTensor<BufT>& cbufWorkspace, const FixpipeParamsM300& intriParams);
#elif defined(__DAV_M310__)
// L0C->L1/UB
template <typename DstT, typename SrcT, const FixpipeConfig& config = CFG_ROW_MAJOR>
__aicore__ inline void Fixpipe(const LocalTensor<DstT>& dstLocal, const LocalTensor<SrcT>& srcLocal,
    const FixpipeParamsM310& intriParams);

// L0C->L1/UB deq tensor quant
template <typename DstT, typename SrcT, const FixpipeConfig& config = CFG_ROW_MAJOR, typename BufT = uint64_t,
    typename std::enable_if<IsSameType<PrimT<BufT>, uint64_t>::value, bool>::type = true>
__aicore__ inline void Fixpipe(const LocalTensor<DstT>& dstLocal, const LocalTensor<SrcT>& srcLocal,
    const LocalTensor<BufT>& cbufWorkspace, const FixpipeParamsM310& intriParams);

// L0C->GM
template <typename DstT, typename SrcT, const FixpipeConfig& config = CFG_ROW_MAJOR>
__aicore__ inline void Fixpipe(const GlobalTensor<DstT>& dstGlobal, const LocalTensor<SrcT>& srcLocal,
    const FixpipeParamsM310& intriParams);

// L0C->GM deq tensor quant
template <typename DstT, typename SrcT, const FixpipeConfig& config = CFG_ROW_MAJOR, typename BufT = uint64_t,
    typename std::enable_if<IsSameType<PrimT<BufT>, uint64_t>::value, bool>::type = true>
__aicore__ inline void Fixpipe(const GlobalTensor<DstT>& dstGlobal, const LocalTensor<SrcT>& srcLocal,
    const LocalTensor<BufT>& cbufWorkspace, const FixpipeParamsM310& intriParams);
#endif
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_FIXPIPE_INTERFACE_H
