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

#include "kernel_operator_set_spr_impl.h"
#include "kernel_check.h"
namespace AscendC {
/* **************************************************************************************************
 * SPR                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void SetFixPipeConfigImpl(const LocalTensor<T> &reluPre, const LocalTensor<T> &quantPre,
    bool isUnitFlag = false)
{
    if ASCEND_IS_AIC {
        CheckTensorPos<T>(reluPre, Hardware::FIXBUF, "reluPre", "C2PIPE2GM", "SetFixPipeConfig");
        CheckTensorPos<T>(quantPre, Hardware::FIXBUF, "quantPre", "C2PIPE2GM", "SetFixPipeConfig");
        uint64_t config = 0;
        config = config | ((uint64_t)reluPre.GetPhyAddr() >> 6);         // in unit of 64B, FPC[7:0], ReluPreAddr
        config = config | (((uint64_t)quantPre.GetPhyAddr() >> 7) << 8); // in unit of 128B, FPC[15:8], QuantPreAddr.
        config = config | ((uint64_t)isUnitFlag << 63);                  // FPC[63], UnitFlag.
        set_fpc(config);
    }
}

template <typename T, bool setRelu = false>
__aicore__ inline void SetFixPipeConfigImpl(const LocalTensor<T> &preTensor, bool isUnitFlag = false)
{
    if ASCEND_IS_AIC {
        CheckTensorPos<T>(preTensor, Hardware::FIXBUF, "preTensor", "C2PIPE2GM", "SetFixPipeConfig");
        uint64_t config = 0;
        if constexpr (setRelu) {
            config = config | ((uint64_t)preTensor.GetPhyAddr() >> 6);        // in unit of 64B, FPC[7:0], ReluPreAddr.
        } else {
            config = config | (((uint64_t)preTensor.GetPhyAddr() >> 7) << 8); // in unit of 128B,FPC[15:8], QuantPreAddr
        }
        config = config | ((uint64_t)isUnitFlag << 63);                       // FPC[63], UnitFlag.
        set_fpc(config);
    }
}

__aicore__ inline void SetFixpipeNz2ndFlagImpl(uint16_t ndNum, uint16_t srcNdStride, uint16_t dstNdStride)
{
    if ASCEND_IS_AIC {
        ASCENDC_CHECK_VALUE_RANGE(srcNdStride, 0, VALUE_512, "srcNdStride", "SetFixpipeNz2ndFlag");
        uint64_t config = 0;
        config = config | ((uint64_t)ndNum);             // ND_PARA[15:0], nd number.
        config = config | ((uint64_t)srcNdStride << 16); // ND_PARA[31:16], src nd stride.
        config = config | ((uint64_t)dstNdStride << 32); // ND_PARA[47:32], dst nd stride.
        set_nd_para(config);
    }
}

__aicore__ inline void SetFixpipePreQuantFlagImpl(uint64_t config)
{
    if ASCEND_IS_AIC {
        set_quant_pre(config);
    }
}

/* **************************************************************************************************
 * Fixpipe                                             *
 * ************************************************************************************************* */

// tiling params
struct FixpipeTiling {
    uint16_t nIterNum = 0;
    uint16_t nSize = 0;
    bool isDb = false;
    uint16_t tailNSize = 0;
};

// fixpipe tiling calculating
__aicore__ inline FixpipeTiling GenFixpipeTiling(uint16_t n)
{
    FixpipeTiling tiling;
    // deqTensor/reluTensor in FB valid num is 256
    uint16_t maxDeqNums = 256;
    if (n <= maxDeqNums) {
        tiling.nIterNum = 1;
        tiling.nSize = n;
        tiling.isDb = false;
        tiling.tailNSize = 0;
    } else {
        tiling.isDb = true;
        uint16_t dbMaxDeqNums = maxDeqNums / 2;
        tiling.nIterNum = n / dbMaxDeqNums;
        tiling.nSize = dbMaxDeqNums;
        tiling.tailNSize = n % dbMaxDeqNums;
    }
    return tiling;
}

template <typename SrcT> struct FixpipeInfoParams {
    __aicore__ inline FixpipeInfoParams() {}

    __aicore__ inline FixpipeInfoParams(const FixpipeParams<SrcT>& intriParams, const uint8_t dstByteSize)
    {
        dstTypeSize = dstByteSize;
        srcTypeSize = B32_BYTE_SIZE;
        howo = (intriParams.burstLen * ONE_BLK_SIZE / srcTypeSize) / BLOCK_CUBE;
        roundHowo = DivCeil(howo, BLOCK_CUBE) * BLOCK_CUBE;
        fracLen = BLOCK_CUBE;
        c0 = fracLen;

        // for 910Pro
        // burst is defined as consective ceil(M/16) 16X16 fractals,
        // and burst length is defined as M*16*sizeof(data_type)
        n = intriParams.cburstNum * BLOCK_CUBE;
        m = howo;

        // original src_stride unit is 256 elements, it's the gap
        // new src_busrt_gap unit is C0_size, for example, src dtype is b32, gap unit is 16*4, it's the stride
        srcStride = intriParams.srcStride * BLOCK_CUBE + roundHowo;

        // original dst_stride unit is 32B, it's the gap, new dst_stride it's the stride
        // note: input burst_len is calculated by src dtype, if src dtype is different with dst dtype, need to
        // re-caculate burst_len for dst
        if (intriParams.nz2ndParams.nz2ndEn) {
            // If NZ2ND is enabled, it is the dst_D value in unit of element. Loop2_dst_stride
            dstStride = intriParams.dstStride;
            // If NZ2ND is enabled, n size could be unaligned
            ASCENDC_ASSERT((intriParams.nz2ndParams.originalNSize != 0), {
                KERNEL_LOG(KERNEL_ERROR, "If NZ2ND is enabled, originalNSize should be set.");
            });
            n = intriParams.nz2ndParams.originalNSize;
        } else {
            // If NZ2ND is disabled, it is the destination stride between the start addresses of different bursts in
            // unit of 32B, Loop1_dst_stride
            dstStride = intriParams.dstStride + intriParams.burstLen * dstTypeSize / srcTypeSize;
        }

        sid = 0;
        quantPre = intriParams.quantParams.quantPre;
        reluEn = intriParams.reluEn;
        nz2ndEn = intriParams.nz2ndParams.nz2ndEn;
        ndNum = intriParams.nz2ndParams.ndNum;
        srcNdStride = intriParams.nz2ndParams.srcNdStride;
        dstNdStride = intriParams.nz2ndParams.dstNdStride;

        // quant
        if (intriParams.quantParams.quantPre == QuantMode_t::DEQF16 ||
            intriParams.quantParams.quantPre == QuantMode_t::QF322B8_PRE ||
            intriParams.quantParams.quantPre == QuantMode_t::REQ8) {
            deqScalar = intriParams.quantParams.deqScalar;
        }

        unitFlag = intriParams.unitFlag;
    }

    // basic params
    uint8_t dstTypeSize = 0;
    uint8_t srcTypeSize = 0;
    uint16_t howo = 0;
    uint16_t roundHowo = 0;
    uint8_t fracLen = 0;
    uint8_t c0 = 0;
    uint16_t n = 0;
    uint16_t m = 0;
    uint16_t srcStride = 0;
    uint32_t dstStride = 0;
    uint16_t burstLen = 0;
    uint8_t sid = 0;
    bool channelSplit = false;
    uint8_t unitFlag = 0;

    // quant param
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    __cbuf__ uint64_t* cbufWorkspace;
    uint64_t deqScalar = 0;
    // relu param
    bool reluEn = false;
    // nz2nd param
    bool nz2ndEn = false;
    uint16_t ndNum = 1;
    uint16_t srcNdStride = 0;
    uint16_t dstNdStride = 0;
    // fixpipe tiling
    FixpipeTiling tiling;
};

// notice: in 910b soc, fixpipe doesn't support float->float and int32->int32
template <typename T>
__aicore__ inline void FixpipeL0C2L1Impl(__cbuf__ T *dst, __cc__ T *src, FixpipeInfoParams<T> &fixpipeInfo)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "Fixpipe from CO1 to A1 / B1 with src and dst both float / int32_t");
}

template <typename DstT, typename SrcT>
__aicore__ inline void FixpipeL0C2L1Impl(__cbuf__ DstT* dst, __cc__ SrcT* src, FixpipeInfoParams<SrcT>& fixpipeInfo)
{
    ASCENDC_REPORT_NOT_SUPPORT((!fixpipeInfo.nz2ndEn), "Fixpipe from CO1 to A1 / B1 with nz2ndEn = true");
    /*
    make code for vector quant mode:
    1. generate tiling
    2. copy deq tensor from gm to fb0 (gm -> l1 -> fb0)
    3. code gen: move data from l0c to l1
    */
    if (fixpipeInfo.quantPre == QuantMode_t::VDEQF16 || fixpipeInfo.quantPre == QuantMode_t::VQF322B8_PRE ||
        fixpipeInfo.quantPre == QuantMode_t::VREQ8) {
        fixpipeInfo.tiling = GenFixpipeTiling(fixpipeInfo.n);
        for (uint16_t i = 0; i < fixpipeInfo.tiling.nIterNum; ++i) {
            FixpipeL0C2L1ImplN(dst, src, fixpipeInfo, fixpipeInfo.tiling.nSize, i);
        }
        // deal with the tail, it also need copy deq/relu tensor from L1 to fb0
        if (fixpipeInfo.tiling.tailNSize > 0) {
            FixpipeL0C2L1ImplN(dst, src, fixpipeInfo, fixpipeInfo.tiling.tailNSize, fixpipeInfo.tiling.nIterNum);
        }
        return;
    }
    /*
    make code for scalar quant mode:
    1. copy deq scalar float immediate
    2. code gen: move data from l0c to l1
    */
    if (fixpipeInfo.quantPre == QuantMode_t::DEQF16 || fixpipeInfo.quantPre == QuantMode_t::QF322B8_PRE ||
        fixpipeInfo.quantPre == QuantMode_t::REQ8) {
        // deq factor of uint64 bits describe: bits[31:13] is deq value of fp32,
        SetQuantPreImpl(fixpipeInfo.deqScalar);
    }
    PipeBarrier<PIPE_FIX>();
    // LOC -> L1
    FixpipeL0cToL1(dst, src, fixpipeInfo, fixpipeInfo.n);
}

template <typename DstT, typename SrcT>
__aicore__ inline void FixpipeL0C2GMImpl(__gm__ DstT* dst, __cc__ SrcT* src, FixpipeInfoParams<SrcT>& fixpipeInfo)
{
    if (fixpipeInfo.nz2ndEn) {
        uint64_t ndPara = static_cast<uint64_t>(fixpipeInfo.dstNdStride) << 32; // ND_PARA[47:32]
        ndPara |= static_cast<uint64_t>(fixpipeInfo.srcNdStride) << 16;         // ND_PARA[31:16]
        ndPara |= static_cast<uint64_t>(fixpipeInfo.ndNum);                     // ND_PARA[15:0]
        SetNdParaImpl(ndPara);
    }
    /*
    make code for vector quant mode:
    1. generate tiling
    2. copy deq tensor from gm to fb0 (gm -> l1 -> fb0)
    3. code gen: move data from l0c to gm
    */
    if (fixpipeInfo.quantPre == QuantMode_t::VDEQF16 || fixpipeInfo.quantPre == QuantMode_t::VQF322B8_PRE ||
        fixpipeInfo.quantPre == QuantMode_t::VREQ8) {
        fixpipeInfo.tiling = GenFixpipeTiling(fixpipeInfo.n);
        for (uint16_t i = 0; i < fixpipeInfo.tiling.nIterNum; ++i) {
            FixpipeL0C2GMImplN(dst, src, fixpipeInfo, fixpipeInfo.tiling.nSize, i);
        }
        // deal with the tail, it also need copy deq/relu tensor from L1 to fb0
        if (fixpipeInfo.tiling.tailNSize > 0) {
            FixpipeL0C2GMImplN(dst, src, fixpipeInfo, fixpipeInfo.tiling.tailNSize, fixpipeInfo.tiling.nIterNum);
        }
        return;
    }

    /*
    make code for scalar quant mode:
    1. copy deq scalar float immediate
    2. code gen: move data from l0c to gm
    */
    if (fixpipeInfo.quantPre == QuantMode_t::DEQF16 || fixpipeInfo.quantPre == QuantMode_t::QF322B8_PRE ||
        fixpipeInfo.quantPre == QuantMode_t::REQ8) {
        SetQuantPreImpl(fixpipeInfo.deqScalar);
    }
    PipeBarrier<PIPE_FIX>();
    // LOC -> GM
    FixpipeL0cToOut(dst, src, fixpipeInfo, fixpipeInfo.n);
}

template <typename DstT, typename SrcT>
__aicore__ inline void FixpipeL0C2L1ImplN(__cbuf__ DstT* dst, __cc__ SrcT* src,
    const FixpipeInfoParams<SrcT>& fixpipeInfo, uint16_t calNSize, uint16_t nIterIndex)
{
    // mov deq tensor from L1 to FB
    CopyDeqTensorToFbuf(fixpipeInfo, calNSize, nIterIndex);
    PipeBarrier<PIPE_FIX>();
    // L0C->L1
    FixpipeL0cToL1(dst, src, fixpipeInfo, calNSize, nIterIndex);
}

template <typename DstT, typename SrcT>
__aicore__ inline void FixpipeL0C2GMImplN(__gm__ DstT* dst, __cc__ SrcT* src,
    const FixpipeInfoParams<SrcT>& fixpipeInfo, uint16_t calNSize, uint16_t nIterIndex)
{
    // mov deq tensor from L1 to FB
    CopyDeqTensorToFbuf(fixpipeInfo, calNSize, nIterIndex);
    PipeBarrier<PIPE_FIX>();
    // L0C->GM
    FixpipeL0cToOut(dst, src, fixpipeInfo, calNSize, nIterIndex);
}

// contains loop info and cal n size for each loop
// move data L0C->L1
template <typename DstT, typename SrcT>
__aicore__ inline void FixpipeL0cToL1(__cbuf__ DstT* dst, __cc__ SrcT* src,
    const FixpipeInfoParams<SrcT>& fixpipeInfo, uint16_t calNSize, uint16_t nIterIndex = 0)
{
    if ASCEND_IS_AIV {
        return;
    }
    uint16_t cburstNum = fixpipeInfo.tiling.nSize / 16;
    uint32_t srcOffset = cburstNum * nIterIndex * fixpipeInfo.srcStride * fixpipeInfo.c0;
    uint32_t dstOffset = 0;
    if (fixpipeInfo.nz2ndEn) {
        dstOffset = nIterIndex * fixpipeInfo.tiling.nSize;
    } else {
        dstOffset = cburstNum * nIterIndex * fixpipeInfo.dstStride * 32 / sizeof(DstT);
    }

    // LOC -> L1 only n direction need tiling, m no need tiling
    // 910b soc, dst_stride in unit of 32B, input dst_stride in unit of 32B.
    return copy_matrix_cc_to_cbuf((__cbuf__ DstT*)(dst + dstOffset), (__cc__ SrcT*)(src + srcOffset), fixpipeInfo.sid,
        calNSize, fixpipeInfo.m, fixpipeInfo.dstStride, fixpipeInfo.srcStride, fixpipeInfo.unitFlag,
        fixpipeInfo.quantPre, static_cast<uint8_t>(fixpipeInfo.reluEn), fixpipeInfo.channelSplit, fixpipeInfo.nz2ndEn);
}

template <typename SrcT>
__aicore__ inline uint64_t GetGMLen(const FixpipeInfoParams<SrcT>& fixpipeInfo,
                                    const uint16_t& calNSize, const uint16_t& dstEleSize)
{
    constexpr uint16_t dstStrideUnit = 32;
    constexpr uint16_t fractalNsize = 16;
    uint64_t cburstNum = calNSize / fractalNsize;
    uint64_t gmLen = (cburstNum - 1) * fixpipeInfo.dstStride * dstStrideUnit +
        fixpipeInfo.m * fractalNsize * dstEleSize;
    if (fixpipeInfo.nz2ndEn) {
        // dstStride is dst_D
        gmLen = (static_cast<uint64_t>(fixpipeInfo.ndNum) - 1) * dstEleSize * fixpipeInfo.dstNdStride +
            (fixpipeInfo.m - 1) * fixpipeInfo.dstStride * dstEleSize + cburstNum * fractalNsize * dstEleSize;
    }
    return gmLen;
}

// contains loop info and cal n size for each loop
// move data L0C->GM
template <typename DstT, typename SrcT>
__aicore__ inline void FixpipeL0cToOut(__gm__ DstT* dst, __cc__ SrcT* src,
    const FixpipeInfoParams<SrcT>& fixpipeInfo, uint16_t calNSize, uint16_t nIterIndex = 0)
{
    if ASCEND_IS_AIV {
        return;
    }
    uint16_t cburstNum = fixpipeInfo.tiling.nSize / 16;
    uint32_t srcOffset = cburstNum * nIterIndex * fixpipeInfo.srcStride * fixpipeInfo.c0;
    uint32_t dstOffset = 0;
    if (fixpipeInfo.nz2ndEn) {
        dstOffset = nIterIndex * fixpipeInfo.tiling.nSize;
    } else {
        dstOffset = cburstNum * nIterIndex * fixpipeInfo.dstStride * 32 / sizeof(DstT);
    }
    if constexpr (g_gm_overflow_check) {
        bool isSrc = false;
        uint16_t dstEleSize = sizeof(DstT);
        uint64_t gmLen = GetGMLen(fixpipeInfo, calNSize, dstEleSize);
        AscendCUtils::CheckGmMemOverflow((__gm__ DstT*)(dst + dstOffset), isSrc, gmLen);
    }
    // LOC -> GM only n direction need tiling, m no need tiling
    // 910b soc, dst_stride in unit of 32B, input dst_stride in unit of 32B.
    return copy_matrix_cc_to_gm((__gm__ DstT*)(dst + dstOffset), (__cc__ SrcT*)(src + srcOffset), fixpipeInfo.sid,
        calNSize, fixpipeInfo.m, fixpipeInfo.dstStride, fixpipeInfo.srcStride, fixpipeInfo.unitFlag,
        fixpipeInfo.quantPre, static_cast<uint8_t>(fixpipeInfo.reluEn), fixpipeInfo.channelSplit, fixpipeInfo.nz2ndEn);
}

template <typename SrcT>
__aicore__ inline void CopyDeqTensorToFbuf(const FixpipeInfoParams<SrcT>& fixpipeInfo, uint16_t calNSize,
    uint16_t nIterIndex)
{
    if ASCEND_IS_AIV {
        return;
    }
    uint16_t deqDataSize = DivCeil(calNSize * sizeof(uint64_t), 128) * 128;
    __fbuf__ uint64_t* deqTensorTempBuf =
        AscendCUtils::GetTemporaryFbBufferAddr<uint64_t>(0, deqDataSize / sizeof(uint64_t));
    uint32_t deqValueOffset = nIterIndex * fixpipeInfo.tiling.nSize;
    // L1 -> FB
    uint16_t fbufBurstLen = deqDataSize / 128; // copy from cbuf to fbuf, burst_len unit is 128Bytes
    copy_cbuf_to_fbuf(deqTensorTempBuf, fixpipeInfo.cbufWorkspace + deqValueOffset, 1, fbufBurstLen, 0, 0);
    // FPC of fixpipe buffer for Quant_PRE is FPC[15:8], unit is 128Bytes
    uint64_t deqTensorAddr = (((uint64_t)deqTensorTempBuf) >> (uint64_t)7) << 8;
    set_fpc(deqTensorAddr);
    AscendCUtils::FreeTemporaryFbBuffer<uint64_t>(deqTensorTempBuf);
}
// L0C->L1
template <typename DstT, typename SrcT, typename ParamT = PrimT<SrcT>,
    typename std::enable_if<IsSameType<PrimT<SrcT>, ParamT>::value, bool>::type = true>
__aicore__ inline void Fixpipe(const LocalTensor<DstT>& dstLocal, const LocalTensor<SrcT>& srcLocal,
    const FixpipeParams<ParamT>& intriParams)
{
    FixpipeInfoParams<PrimT<SrcT>> fixpipeInfo(intriParams, sizeof(PrimT<DstT>));
    FixpipeL0C2L1Impl((__cbuf__ PrimT<DstT>*)dstLocal.GetPhyAddr(),
        (__cc__ PrimT<SrcT>*)srcLocal.GetPhyAddr(), fixpipeInfo);
}
// L0C->L1 deq tensor quant
template <typename DstT, typename SrcT, typename BufT, typename ParamT = PrimT<SrcT>,
    typename std::enable_if<IsSameType<PrimT<SrcT>, ParamT>::value, bool>::type = true>
__aicore__ inline void Fixpipe(const LocalTensor<DstT>& dstLocal, const LocalTensor<SrcT>& srcLocal,
    const LocalTensor<BufT>& cbufWorkspace, const FixpipeParams<ParamT>& intriParams)
{
    FixpipeInfoParams<PrimT<SrcT>> fixpipeInfo(intriParams, sizeof(PrimT<DstT>));
    fixpipeInfo.cbufWorkspace = (__cbuf__ uint64_t*)cbufWorkspace.GetPhyAddr();
    FixpipeL0C2L1Impl((__cbuf__ PrimT<DstT>*)dstLocal.GetPhyAddr(),
        (__cc__ PrimT<SrcT>*)srcLocal.GetPhyAddr(), fixpipeInfo);
}

// L0C->GM
template <typename DstT, typename SrcT, typename ParamT = PrimT<SrcT>,
    typename std::enable_if<IsSameType<PrimT<SrcT>, ParamT>::value, bool>::type = true>
__aicore__ inline void Fixpipe(const GlobalTensor<DstT>& dstGlobal, const LocalTensor<SrcT>& srcLocal,
    const FixpipeParams<ParamT>& intriParams)
{
#ifdef ASCENDC_CPU_DEBUG
    bool isUsedProcessLock = false;
    if (g_isAtomic == true) {
        ProcessLock::GetProcessLock()->Write();
        isUsedProcessLock = true;
    }
#endif // ASCENDC_CPU_DEBUG
    FixpipeInfoParams<PrimT<SrcT>> fixpipeInfo(intriParams, sizeof(PrimT<DstT>));

    FixpipeL0C2GMImpl((__gm__ PrimT<DstT>*)dstGlobal.GetPhyAddr(),
        (__cc__ PrimT<SrcT>*)srcLocal.GetPhyAddr(), fixpipeInfo);
#ifdef ASCENDC_CPU_DEBUG
    if (isUsedProcessLock == true) {
        isUsedProcessLock = false;
        ProcessLock::GetProcessLock()->Unlock();
    }
#endif // ASCENDC_CPU_DEBUG
}

// L0C->GM deq tensor quant
template <typename DstT, typename SrcT, typename BufT, typename ParamT = PrimT<SrcT>,
    typename std::enable_if<IsSameType<PrimT<SrcT>, ParamT>::value, bool>::type = true>
__aicore__ inline void Fixpipe(const GlobalTensor<DstT> &dstGlobal, const LocalTensor<SrcT> &srcLocal,
    const LocalTensor<BufT> &cbufWorkspace, const FixpipeParams<ParamT> &intriParams)
{
    FixpipeInfoParams<PrimT<SrcT>> fixpipeInfo(intriParams, sizeof(PrimT<DstT>));
    fixpipeInfo.cbufWorkspace = (__cbuf__ uint64_t *)cbufWorkspace.GetPhyAddr();
    FixpipeL0C2GMImpl((__gm__ PrimT<DstT>*)dstGlobal.GetPhyAddr(),
        (__cc__ PrimT<SrcT>*)srcLocal.GetPhyAddr(), fixpipeInfo);
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_FIXPIPE_IMPL_H
