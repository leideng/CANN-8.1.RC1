/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
/* **************************************************************************************************
 * SPR                                             *
 * ************************************************************************************************* */
template <typename T>
__aicore__ inline void SetFixPipeConfigImpl(const LocalTensor<T> &reluPre, const LocalTensor<T> &quantPre,
    bool isUnitFlag = false)
{
    uint64_t config = 0;
    config = config | ((uint64_t)reluPre.GetPhyAddr() >> 6);         // align with 64bit, FPC[7:0], ReluPreAddr
    config = config | (((uint64_t)quantPre.GetPhyAddr() >> 7) << 8); // align with 128bit, FPC[15:8], QuantPreAddr.
    config = config | ((uint64_t)isUnitFlag << 63);                  // FPC[63], UnitFlag.
    set_fpc(config);
}

template <typename T, bool setRelu = false>
__aicore__ inline void SetFixPipeConfigImpl(const LocalTensor<T> &preTensor, bool isUnitFlag = false)
{
    uint64_t config = 0;
    if constexpr (setRelu) {
        config = config | ((uint64_t)preTensor.GetPhyAddr() >> 6); // align with 64bit, FPC[7:0], ReluPreAddr.
    } else {
        config =
            config | (((uint64_t)preTensor.GetPhyAddr() >> 7) << 8); // align with 128bit, FPC[15:8], QuantPreAddr.
    }
    config = config | ((uint64_t)isUnitFlag << 63); // FPC[63], UnitFlag.
    set_fpc(config);
}

__aicore__ inline void SetFixpipeNz2ndFlagImpl(uint16_t ndNum, uint16_t srcNdStride, uint16_t dstNdStride)
{
    // ND_PARA[63:32], dst nd stride in unit of element
    // ND_PARA[31:16], src nd stride in uint of C0_SIZE
    // ND_PARA[15:0], nd number.
    uint64_t config = (static_cast<uint64_t>(dstNdStride) << 32) | (static_cast<uint64_t>(srcNdStride) << 16) |
                        (static_cast<uint64_t>(ndNum));
    set_loop3_para(config);
}

__aicore__ inline void SetFixpipePreQuantFlagImpl(uint64_t config)
{
    set_quant_pre(config);
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
const uint32_t L0C_SRC_ALIGN = 16 * sizeof(float);       // src must align with 16 elements, each of them is F32 / S32

__aicore__ inline bool IsVectorQuantMode(QuantMode_t quantPre)
{
    return (quantPre == QuantMode_t::VDEQF16 || quantPre == QuantMode_t::VQF322B8_PRE ||
            quantPre == QuantMode_t::VREQ8);
}

__aicore__ inline bool IsScalarQuantMode(QuantMode_t quantPre)
{
    return (quantPre == QuantMode_t::DEQF16 || quantPre == QuantMode_t::QF322B8_PRE || quantPre == QuantMode_t::REQ8);
}

__aicore__ inline void SetLoop3Para(const FixpipeParamsM310& intriParams)
{
    ASCENDC_ASSERT((intriParams.ndNum > 0), { KERNEL_LOG(KERNEL_ERROR, "ndNum must be larger than 0"); });
    uint64_t loop3Para = static_cast<uint64_t>(intriParams.dstNdStride) << 32;  // LOOP3_PARA[63:32]
    loop3Para |= static_cast<uint64_t>(intriParams.srcNdStride) << 16;  // LOOP3_PARA[31:16]
    loop3Para |= static_cast<uint64_t>(intriParams.ndNum);              // LOOP3_PARA[15:0]
    set_loop3_para(loop3Para);
}

template <typename DstT, typename SrcT, const FixpipeConfig& config>
__aicore__ inline void CheckCommonFixpipeParam(__cc__ SrcT *src, const FixpipeParamsM310 &params)
{
#if ASCENDC_CPU_DEBUG
    ASCENDC_CHECK_TENSOR_PTR_ALIGN(src, TPosition::CO1, L0C_SRC_ALIGN, "srcLocal", "Fixpipe");
    if (params.isChannelSplit) {
        ASCENDC_ASSERT((params.nSize <= UINT12_MAX && params.nSize >=1 && params.nSize % 8 == 0),
            {KERNEL_LOG(KERNEL_ERROR,"Failed to check nSize value in Fixpipe, when isChannelSplit is true, its valid "
            "range is 1 ~ 4095 and must be divisible by 8, current value is %u", params.nSize); });
    } else if (config.format == CO2Layout::ROW_MAJOR) {
        ASCENDC_CHECK_VALUE_RANGE(params.nSize, 1, UINT12_MAX, "nSize",
            "Fixpipe when isChannelSplit is false and format is NZ2ND");
    } else {
        ASCENDC_ASSERT((params.nSize <= UINT12_MAX && params.nSize >=1 && params.nSize % 16 == 0),
            {KERNEL_LOG(KERNEL_ERROR, "Failed to check nSize value in Fixpipe, when isChannelSplit is false and format "
            "is NZ, its valid range is 1 ~ 4095 and must be divisible by 16, current value is %u", params.nSize); });
    }
    if constexpr(config.format == CO2Layout::ROW_MAJOR) {
        ASCENDC_CHECK_VALUE_RANGE(params.mSize, 1, 8192, "mSize", "Fixpipe when format is ROW_MAJOR");
    } else {
        ASCENDC_CHECK_VALUE_RANGE(params.mSize, 1, UINT16_MAX, "mSize", "Fixpipe when format is NZ");
    }

    ASCENDC_ASSERT((params.dstStride != 0), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dstStride value in Fixpipe, "\
        "its valid range is 1 ~ 4294967295, current value is %u", params.dstStride);});
    if (params.ndNum > 1) {
        ASCENDC_CHECK_VALUE_RANGE(params.srcNdStride, 1, VALUE_512, "srcNdStride", "Fixpipe when ndNum is > 1");
        ASCENDC_CHECK_VALUE_RANGE(params.dstNdStride, 1, UINT16_MAX, "dstNdStride", "Fixpipe when ndNum is > 1");
    }

    if constexpr(IsSameType<SrcT, float>::value && SupportType<DstT, int8_t, uint8_t>()) {
        ASCENDC_ASSERT((params.quantPre == QuantMode_t::QF322B8_PRE || params.quantPre == QuantMode_t::VQF322B8_PRE),
            {KERNEL_LOG(KERNEL_ERROR, "Failed to check quantPre value in Fixpipe, when src is float, dst is "
            "int8_t / uint8_t, supported values are QF322B8_PRE and VQF322B8_PRE");});
    } else if constexpr(IsSameType<SrcT, float>::value && IsSameType<DstT, half>::value) {
        ASCENDC_ASSERT((params.quantPre == QuantMode_t::F322F16), {KERNEL_LOG(KERNEL_ERROR, "Failed to check quantPre "\
        "value in Fixpipe, when src is float, dst is half, supported value is F322F16");});
    } else if constexpr(IsSameType<SrcT, int32_t>::value && SupportType<DstT, int8_t, uint8_t>()) {
        ASCENDC_ASSERT((params.quantPre == QuantMode_t::REQ8 || params.quantPre == QuantMode_t::VREQ8),
            {KERNEL_LOG(KERNEL_ERROR, "Failed to check quantPre value in Fixpipe, when src is int32_t, dst is "\
            "int8_t / uint8_t, supported values are REQ8 and VREQ8");});
    } else if constexpr(IsSameType<SrcT, int32_t>::value && IsSameType<DstT, half>::value) {
        ASCENDC_ASSERT((params.quantPre == QuantMode_t::DEQF16 || params.quantPre == QuantMode_t::VDEQF16),
            {KERNEL_LOG(KERNEL_ERROR, "Failed to check quantPre value in Fixpipe, when src is int32_t, dst is half, "\
            "supported values are DEQF16 and VDEQF16");});
    }
#endif
}

template <typename DstT, typename SrcT, const FixpipeConfig& config>
__aicore__ inline void CheckFixpipeL0C2UBParam(__ubuf__ DstT *dst, __cc__ SrcT *src, const FixpipeParamsM310 &params)
{
    CheckCommonFixpipeParam<DstT, SrcT, config>(src, params);
    ASCENDC_CHECK_TENSOR_PTR_ALIGN(
        dst, TPosition::VECCALC, ONE_BLK_SIZE, "dstLocal", "Fixpipe when dst position is VECIN/VECCALC/VECOUT");
    static_assert((SupportType<Tuple<SrcT, DstT>, Tuple<float, int8_t>, Tuple<float, uint8_t>, Tuple<float, half>,
        Tuple<float, float>, Tuple<int32_t, int8_t>, Tuple<int32_t, uint8_t>,
        Tuple<int32_t, half>, Tuple<int32_t, int32_t>>()), "Failed to check dtype in Fixpipe, when src position is "\
        "CO1 and dst position is GM, support dtype combinations are src: float, dst: int8_t / uint8_t / half "\
        "/ float; src: int32_t, dst: int8_t / uint8_t / half / int32_t");
    if constexpr(IsSameType<SrcT, float>::value && IsSameType<DstT, float>::value) {
        ASCENDC_ASSERT((params.quantPre == QuantMode_t::NoQuant), {KERNEL_LOG(KERNEL_ERROR, "Failed to check quantPre "\
            "value in Fixpipe, when src is float, dst is float, supported value is NoQuant");});
    } else if constexpr(IsSameType<SrcT, int32_t>::value && IsSameType<DstT, int32_t>::value) {
        ASCENDC_ASSERT((params.quantPre == QuantMode_t::NoQuant), {KERNEL_LOG(KERNEL_ERROR, "Failed to check quantPre "\
            "value in Fixpipe, when src is int32_t, dst is int32_t, supported value is NoQuant");});
    }
    if (params.isChannelSplit) {
        ASCENDC_DEBUG_ASSERT((IsSameType<SrcT, float>::value && IsSameType<DstT, float>::value), "Failed to check "\
            "isChannelSplit value in Fixpipe, isChannelSplit can be set true only when src and dst are both float \n");
        ASCENDC_DEBUG_ASSERT((config.format != CO2Layout::ROW_MAJOR), "Failed to check format value in Fixpipe, "\
            "when isChannelSplit is set true, format must be set as NZ \n");
    }
}

template <typename DstT, typename SrcT, const FixpipeConfig& config>
__aicore__ inline void CheckFixpipeL0C2L1Param(__cbuf__ DstT *dst, __cc__ SrcT *src, const FixpipeParamsM310 &params)
{
    CheckCommonFixpipeParam<DstT, SrcT, config>(src, params);
#if ASCENDC_CPU_DEBUG
    uint64_t dstAbsAddr = dst - (DstT*)(GetTPipePtr()->GetBaseAddr(int8_t(TPosition::A1)));
    ASCENDC_ASSERT((dstAbsAddr % ONE_BLK_SIZE == 0), {KERNEL_LOG(KERNEL_ERROR, "Failed to check dstLocal start "\
        "address alignment in Fixpipe, when dst position is A1, its start address must align with 32B");});
#endif
    ASCENDC_DEBUG_ASSERT((!(params.isChannelSplit)), "Failed to check isChannelSplit in Fixpipe, when src position is "\
        "CO1 and dst position is A1, isChannelSplit must be set as false \n");

    static_assert((SupportType<Tuple<SrcT, DstT>, Tuple<float, int8_t>, Tuple<float, uint8_t>, Tuple<float, half>,
        Tuple<float, float>, Tuple<int32_t, int32_t>, Tuple<int32_t, int8_t>, Tuple<int32_t, uint8_t>, Tuple<int32_t, half>>()),
        "Failed to check dtype in Fixpipe, when src position is CO1 and dst position is A1, "\
        "support dtype combinations are src: float, dst: int8_t / uint8_t / half; src: int32_t, dst: "\
        "int8_t / uint8_t / half");
}

template <typename DstT, typename SrcT, const FixpipeConfig& config>
__aicore__ inline void CheckFixpipeL0C2GMParam(__gm__ DstT *dst, __cc__ SrcT *src, const FixpipeParamsM310 &params)
{
    CheckCommonFixpipeParam<DstT, SrcT, config>(src, params);

    static_assert((SupportType<Tuple<SrcT, DstT>, Tuple<float, int8_t>, Tuple<float, uint8_t>, Tuple<float, half>,
        Tuple<float, float>, Tuple<int32_t, int8_t>, Tuple<int32_t, uint8_t>, Tuple<int32_t, half>, Tuple<int32_t, int32_t>>()),
        "Failed to check dtype in Fixpipe, when src position is CO1 "\
        "and dst position is GM, support dtype combinations are src: float, dst: int8_t / uint8_t / half "\
        "/ float; src: int32_t, dst: int8_t / uint8_t / half / int32_t");
    if constexpr(IsSameType<SrcT, float>::value && IsSameType<DstT, float>::value) {
        ASCENDC_ASSERT((params.quantPre == QuantMode_t::NoQuant), {KERNEL_LOG(KERNEL_ERROR, "Failed to check quantPre "\
            "value in Fixpipe, when src is float, dst is float, supported value is NoQuant");});
    } else if constexpr(IsSameType<SrcT, int32_t>::value && IsSameType<DstT, int32_t>::value) {
        ASCENDC_ASSERT((params.quantPre == QuantMode_t::NoQuant), {KERNEL_LOG(KERNEL_ERROR, "Failed to check quantPre "\
            "value in Fixpipe, when src is int32_t, dst is int32_t, supported value is NoQuant");});
    }
    if (params.isChannelSplit) {
        ASCENDC_DEBUG_ASSERT((IsSameType<SrcT, float>::value && IsSameType<DstT, float>::value), "Failed to check "\
            "isChannelSplit value in Fixpipe, isChannelSplit can be set true only when src and dst are both float \n");
        ASCENDC_DEBUG_ASSERT((config.format != CO2Layout::ROW_MAJOR), "Failed to check format value in Fixpipe, "\
            "when isChannelSplit is set true, format must be set as NZ \n");
    }
}

// adaptation due to unit differences;
template <typename DstT, typename SrcT, const FixpipeConfig& config>
__aicore__ inline FixpipeParamsM310 AdaptFixpipeParams(const FixpipeParamsM310 &intriParams)
{
    FixpipeParamsM310 params(intriParams);
    if constexpr (config.format == CO2Layout::NZ) {
        params.dstStride = intriParams.dstStride * ONE_BLK_SIZE / sizeof(DstT);
    }

    if constexpr(config.format == CO2Layout::ROW_MAJOR) {
        constexpr uint16_t ORI_UNIT = 1024;
        constexpr uint16_t multiples = ORI_UNIT / (BLOCK_CUBE * sizeof(SrcT));
        params.srcNdStride = params.srcNdStride * multiples;
    }
    return params;
}

// tiling params
struct FixpipeTilingM310 {
    uint16_t nIterNum = 0;
    uint16_t nSize = 0;
    bool isDb = false;
    uint16_t tailNSize = 0;
};

// fixpipe tiling calculating
__aicore__ inline FixpipeTilingM310 GenFixpipeTilingM310(uint16_t n)
{
    FixpipeTilingM310 tiling;
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

__aicore__ inline void CopyDeqTensorToFbuf(
    __cbuf__ uint64_t *cbufWorkspace, const FixpipeTilingM310 &fixpipeTiling, uint16_t calNSize, uint16_t nIterIndex)
{
    uint16_t deqDataSize = DivCeil(calNSize * sizeof(uint64_t), 128) * 128;
    __fbuf__ uint64_t *deqTensorTempBuf =
        AscendCUtils::GetTemporaryFbBufferAddr<uint64_t>(0, deqDataSize / sizeof(uint64_t));
    uint32_t deqValueOffset = nIterIndex * fixpipeTiling.nSize;
    // L1 -> FB
    constexpr uint16_t fbufBurstLenUnit = 64;
    uint16_t fbufBurstLen = deqDataSize / fbufBurstLenUnit;  // copy from cbuf to fbuf, burst_len unit is 64Bytes
    copy_cbuf_to_fbuf(deqTensorTempBuf, cbufWorkspace + deqValueOffset, 1, fbufBurstLen, 0, 0);
    // FPC of fixpipe buffer for Quant_PRE is FPC[15:8], unit is 128Bytes
    uint64_t deqTensorAddr = (((uint64_t)deqTensorTempBuf) >> (uint64_t)7) << 8;
    set_fpc(deqTensorAddr);
    AscendCUtils::FreeTemporaryFbBuffer<uint64_t>(deqTensorTempBuf);
}

// contains loop info and cal n size for each loop
// move data L0C->UB
template <typename DstT, typename SrcT, const FixpipeConfig &config>
__aicore__ inline void FixpipeL0cToUB(__ubuf__ DstT *dst, __cc__ SrcT *src, const FixpipeParamsM310 &intriParams,
    const FixpipeTilingM310 &fixpipeTiling, uint16_t calNSize, uint16_t nIterIndex = 0)
{
    uint16_t cburstNum = fixpipeTiling.nSize / 16;
    uint32_t srcOffset = cburstNum * nIterIndex * intriParams.srcStride * BLOCK_CUBE;
    uint32_t dstOffset = 0;
    bool nz2ndEn = false;
    if constexpr (config.format == CO2Layout::ROW_MAJOR) {
        dstOffset = nIterIndex * fixpipeTiling.nSize;
        nz2ndEn = true;
    } else {
        dstOffset = cburstNum * nIterIndex * intriParams.dstStride * 32 / sizeof(DstT);
    }
    // LOC -> UB only n direction need tiling, m no need tiling
    return copy_matrix_cc_to_ub((__ubuf__ DstT*)(dst + dstOffset), (__cc__ SrcT*)(src + srcOffset), 0,
        calNSize, intriParams.mSize, intriParams.dstStride, intriParams.srcStride, 0, intriParams.unitFlag,
        intriParams.quantPre, static_cast<uint8_t>(intriParams.reluEn),
        intriParams.isChannelSplit, nz2ndEn, false);
}

template <typename DstT, typename SrcT, const FixpipeConfig& config>
__aicore__ inline void FixpipeL0C2UBImpl(__ubuf__ DstT *dst, __cc__ SrcT *src, const FixpipeParamsM310 &intriParams)
{
    const FixpipeParamsM310& params = AdaptFixpipeParams<DstT, SrcT, config>(intriParams);
    CheckFixpipeL0C2UBParam<DstT, SrcT, config>(dst, src, params);
    if constexpr (config.format == CO2Layout::ROW_MAJOR) {
        SetLoop3Para(params);
    }
    /*
    make code for scalar quant mode:
    1. copy deq scalar u64 immediate
    2. code gen: move data from l0c to ub
    */
    if (IsScalarQuantMode(params.quantPre)) {
        set_quant_pre(params.deqScalar);
    }
    PipeBarrier<PIPE_FIX>();
    // LOC -> UB
    FixpipeTilingM310 fixpipeTiling;
    FixpipeL0cToUB<DstT, SrcT, config>(dst, src, params, fixpipeTiling, params.nSize, 0);
}

template <typename DstT, typename SrcT, const FixpipeConfig &config>
__aicore__ inline void FixpipeL0C2UBImplN(__ubuf__ DstT *dst, __cc__ SrcT *src, __cbuf__ uint64_t *cbufWorkspace,
    const FixpipeParamsM310 &intriParams, const FixpipeTilingM310 &fixpipeTiling, uint16_t calNSize,
    uint16_t nIterIndex)
{
    // mov deq tensor from L1 to FB
    CopyDeqTensorToFbuf(cbufWorkspace, fixpipeTiling, calNSize, nIterIndex);
    PipeBarrier<PIPE_FIX>();
    // L0C->UB
    FixpipeL0cToUB<DstT, SrcT, config>(dst, src, intriParams, fixpipeTiling, calNSize, nIterIndex);
}

template <typename DstT, typename SrcT, const FixpipeConfig &config>
__aicore__ inline void FixpipeL0C2UBImpl(
    __ubuf__ DstT *dst, __cc__ SrcT *src, __cbuf__ uint64_t *cbufWorkspace, const FixpipeParamsM310 &intriParams)
{
    const FixpipeParamsM310& params = AdaptFixpipeParams<DstT, SrcT, config>(intriParams);
    CheckFixpipeL0C2UBParam<DstT, SrcT, config>(dst, src, params);
    if constexpr (config.format == CO2Layout::ROW_MAJOR) {
        SetLoop3Para(params);
    }
    /*
    make code for vector quant mode:
    1. generate tiling
    2. copy deq tensor from l1 to fb0
    3. code gen: move data from l0c to ub
    */
    FixpipeTilingM310 fixpipeTiling = GenFixpipeTilingM310(params.nSize);
    if (IsVectorQuantMode(params.quantPre)) {
        for (uint16_t i = 0; i < fixpipeTiling.nIterNum; ++i) {
            FixpipeL0C2UBImplN<DstT, SrcT, config>(
                dst, src, cbufWorkspace, params, fixpipeTiling, fixpipeTiling.nSize, i);
        }
        // deal with the tail, it also need copy deq/relu tensor from L1 to fb0
        if (fixpipeTiling.tailNSize > 0) {
            FixpipeL0C2UBImplN<DstT, SrcT, config>(
                dst, src, cbufWorkspace, params, fixpipeTiling, fixpipeTiling.tailNSize, fixpipeTiling.nIterNum);
        }
        return;
    }
}

// contains loop info and cal n size for each loop
// move data L0C->L1
template <typename DstT, typename SrcT, const FixpipeConfig &config>
__aicore__ inline void FixpipeL0cToL1(__cbuf__ DstT *dst, __cc__ SrcT *src, const FixpipeParamsM310 &intriParams,
    const FixpipeTilingM310 &fixpipeTiling, uint16_t calNSize, uint16_t nIterIndex = 0)
{
    uint16_t cburstNum = fixpipeTiling.nSize / 16;
    uint32_t srcOffset = cburstNum * nIterIndex * intriParams.srcStride * BLOCK_CUBE;

    uint32_t dstOffset = 0;
    bool nz2ndEn = false;
    if constexpr (config.format == CO2Layout::ROW_MAJOR) {
        dstOffset = nIterIndex * fixpipeTiling.nSize;
        nz2ndEn = true;
    } else {
        dstOffset = cburstNum * nIterIndex * intriParams.dstStride * 32 / sizeof(DstT);
    }
    // LOC -> L1 only n direction need tiling, m no need tiling
    return copy_matrix_cc_to_cbuf((__cbuf__ DstT*)(dst + dstOffset), (__cc__ SrcT*)(src + srcOffset), 0,
        calNSize, intriParams.mSize, intriParams.dstStride, intriParams.srcStride, 0, intriParams.unitFlag,
        intriParams.quantPre, static_cast<uint8_t>(intriParams.reluEn),
        false, nz2ndEn, false);
}


template <typename DstT, typename SrcT, const FixpipeConfig& config>
__aicore__ inline void FixpipeL0C2L1Impl(__cbuf__ DstT *dst, __cc__ SrcT *src, const FixpipeParamsM310 &intriParams)
{
    const FixpipeParamsM310& params = AdaptFixpipeParams<DstT, SrcT, config>(intriParams);
    CheckFixpipeL0C2L1Param<DstT, SrcT, config>(dst, src, intriParams);
    if constexpr (config.format == CO2Layout::ROW_MAJOR) {
        SetLoop3Para(params);
    }
    /*
    make code for scalar quant mode:
    1. copy deq scalar u64 immediate
    2. code gen: move data from l0c to l1
    */
    if (IsScalarQuantMode(params.quantPre)) {
        // deq factor of uint64 bits describe: bits[31:13] is deq value of fp32,
        set_quant_pre(params.deqScalar);
    }
    PipeBarrier<PIPE_FIX>();
    FixpipeTilingM310 fixpipeTiling;
    // LOC -> L1
    FixpipeL0cToL1<DstT, SrcT, config>(dst, src, params, fixpipeTiling, params.nSize);
}

template <typename DstT, typename SrcT, const FixpipeConfig &config>
__aicore__ inline void FixpipeL0C2L1ImplN(__cbuf__ DstT *dst, __cc__ SrcT *src, __cbuf__ uint64_t *cbufWorkspace,
    const FixpipeParamsM310 &intriParams, const FixpipeTilingM310 &fixpipeTiling, uint16_t calNSize,
    uint16_t nIterIndex)
{
    // mov deq tensor from L1 to FB
    CopyDeqTensorToFbuf(cbufWorkspace, fixpipeTiling, calNSize, nIterIndex);
    PipeBarrier<PIPE_FIX>();
    // L0C->L1
    FixpipeL0cToL1<DstT, SrcT, config>(dst, src, intriParams, fixpipeTiling, calNSize, nIterIndex);
}

template <typename DstT, typename SrcT, const FixpipeConfig &config>
__aicore__ inline void FixpipeL0C2L1Impl(
    __cbuf__ DstT *dst, __cc__ SrcT *src, __cbuf__ uint64_t *cbufWorkspace, const FixpipeParamsM310 &intriParams)
{
    const FixpipeParamsM310& params = AdaptFixpipeParams<DstT, SrcT, config>(intriParams);
    CheckFixpipeL0C2L1Param<DstT, SrcT, config>(dst, src, params);
    if constexpr (config.format == CO2Layout::ROW_MAJOR) {
        SetLoop3Para(params);
    }
    /*
    make code for vector quant mode:
    1. generate tiling
    2. copy deq tensor from l1 to fb0
    3. code gen: move data from l0c to l1
    */
    FixpipeTilingM310 fixpipeTiling = GenFixpipeTilingM310(params.nSize);
    if (IsVectorQuantMode(params.quantPre)) {
        for (uint16_t i = 0; i < fixpipeTiling.nIterNum; ++i) {
            FixpipeL0C2L1ImplN<DstT, SrcT, config>(
                dst, src, cbufWorkspace, params, fixpipeTiling, fixpipeTiling.nSize, i);
        }
        // deal with the tail, it also need copy deq/relu tensor from L1 to fb0
        if (fixpipeTiling.tailNSize > 0) {
            FixpipeL0C2L1ImplN<DstT, SrcT, config>(
                dst, src, cbufWorkspace, params, fixpipeTiling, fixpipeTiling.tailNSize, fixpipeTiling.nIterNum);
        }
        return;
    }
}

template <typename DstT>
__aicore__ inline uint64_t GetGMLength(
    const FixpipeParamsM310 &intriParams, const uint16_t &calNSize, const bool &nz2ndEn)
{
    const uint16_t dstEleSize = sizeof(DstT);
    constexpr uint16_t dstStrideUnit = 32;
    constexpr uint16_t fractalNsize = 16;
    uint64_t cburstNum = calNSize / fractalNsize;
    uint64_t gmLen;
    gmLen =
        (cburstNum - 1) * intriParams.dstStride * dstStrideUnit + intriParams.mSize * fractalNsize * dstEleSize;
    if (nz2ndEn) {
        gmLen = (static_cast<uint64_t>(intriParams.ndNum) - 1) * dstEleSize * intriParams.dstNdStride +
                (intriParams.mSize - 1) * intriParams.dstStride * dstEleSize + cburstNum * fractalNsize * dstEleSize;
    }
    return gmLen;
}

// contains loop info and cal n size for each loop
// move data L0C->GM
template <typename DstT, typename SrcT, const FixpipeConfig &config>
__aicore__ inline void FixpipeL0cToOut(__gm__ DstT *dst, __cc__ SrcT *src, const FixpipeParamsM310 &intriParams,
    const FixpipeTilingM310 &fixpipeTiling, uint16_t calNSize, uint16_t nIterIndex = 0)
{
    uint16_t cburstNum = fixpipeTiling.nSize / 16;
    uint32_t srcOffset = cburstNum * nIterIndex * intriParams.srcStride * BLOCK_CUBE;
    uint32_t dstOffset = 0;
    bool nz2ndEn = false;
    if constexpr (config.format == CO2Layout::ROW_MAJOR) {
        dstOffset = nIterIndex * fixpipeTiling.nSize;
        nz2ndEn = true;
    } else {
        dstOffset = cburstNum * nIterIndex * intriParams.dstStride * 32 / sizeof(DstT);
    }

    if constexpr (g_gm_overflow_check) {
        uint64_t gmLen = GetGMLength<DstT>(intriParams, calNSize, nz2ndEn);
        AscendCUtils::CheckGmMemOverflow((__gm__ DstT *)(dst + dstOffset), false, gmLen);  // isSrc is false
    }
    return copy_matrix_cc_to_gm((__gm__ DstT*)(dst + dstOffset), (__cc__ SrcT*)(src + srcOffset), 0,
        calNSize, intriParams.mSize, intriParams.dstStride, intriParams.srcStride, 0, intriParams.unitFlag,
        intriParams.quantPre, static_cast<uint8_t>(intriParams.reluEn),
        intriParams.isChannelSplit, nz2ndEn, false);
}

template <typename DstT, typename SrcT, const FixpipeConfig& config>
__aicore__ inline void FixpipeL0C2GMImpl(__gm__ DstT *dst, __cc__ SrcT *src, const FixpipeParamsM310 &intriParams)
{
    const FixpipeParamsM310& params = AdaptFixpipeParams<DstT, SrcT, config>(intriParams);
    CheckFixpipeL0C2GMParam<DstT, SrcT, config>(dst, src, params);
    if constexpr (config.format == CO2Layout::ROW_MAJOR) {
        SetLoop3Para(params);
    }
    /*
    make code for scalar quant mode:
    1. copy deq scalar u64 immediate
    2. code gen: move data from l0c to gm
    */
    if (IsScalarQuantMode(params.quantPre)) {
        set_quant_pre(params.deqScalar);
    }
    PipeBarrier<PIPE_FIX>();
    // LOC -> GM
    FixpipeTilingM310 fixpipeTiling;
    FixpipeL0cToOut<DstT, SrcT, config>(dst, src, params, fixpipeTiling, params.nSize);
}

template <typename DstT, typename SrcT, const FixpipeConfig &config>
__aicore__ inline void FixpipeL0C2GMImplN(__gm__ DstT *dst, __cc__ SrcT *src, __cbuf__ uint64_t *cbufWorkspace,
    const FixpipeParamsM310 &intriParams, const FixpipeTilingM310 &fixpipeTiling, uint16_t calNSize,
    uint16_t nIterIndex)
{
    // mov deq tensor from L1 to FB
    CopyDeqTensorToFbuf(cbufWorkspace, fixpipeTiling, calNSize, nIterIndex);
    PipeBarrier<PIPE_FIX>();
    // L0C->GM
    FixpipeL0cToOut<DstT, SrcT, config>(dst, src, intriParams, fixpipeTiling, calNSize, nIterIndex);
}

template <typename DstT, typename SrcT, const FixpipeConfig &config>
__aicore__ inline void FixpipeL0C2GMImpl(
    __gm__ DstT *dst, __cc__ SrcT *src, __cbuf__ uint64_t *cbufWorkspace, const FixpipeParamsM310 &intriParams)
{
    const FixpipeParamsM310& params = AdaptFixpipeParams<DstT, SrcT, config>(intriParams);
    CheckFixpipeL0C2GMParam<DstT, SrcT, config>(dst, src, params);
    if constexpr (config.format == CO2Layout::ROW_MAJOR) {
        SetLoop3Para(params);
    }
    /*
    make code for vector quant mode:
    1. generate tiling
    2. copy deq tensor from gm to fb0 (l1 -> fb0)
    3. code gen: move data from l0c to gm
    */
    FixpipeTilingM310 fixpipeTiling = GenFixpipeTilingM310(params.nSize);
    if (IsVectorQuantMode(params.quantPre)) {
        for (uint16_t i = 0; i < fixpipeTiling.nIterNum; ++i) {
            FixpipeL0C2GMImplN<DstT, SrcT, config>(
                dst, src, cbufWorkspace, params, fixpipeTiling, fixpipeTiling.nSize, i);
        }
        // deal with the tail, it also need copy deq/relu tensor from L1 to fb0
        if (fixpipeTiling.tailNSize > 0) {
            FixpipeL0C2GMImplN<DstT, SrcT, config>(dst, src, cbufWorkspace, params,
                fixpipeTiling, fixpipeTiling.tailNSize, fixpipeTiling.nIterNum);
        }
        return;
    }
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_FIXPIPE_V2_IMPL_H
