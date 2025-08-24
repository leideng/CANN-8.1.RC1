/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2025. All rights reserved.
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
 * \file detect_mat_mul.h
 * \brief
 */
#ifndef __OP_KERNEL_MATMUL_STRESS_DETECT_H__
#define __OP_KERNEL_MATMUL_STRESS_DETECT_H__
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "mat_mul_nd2nz_stress_detect.h"

namespace MatmulStressDetect {

using namespace AscendC;
using namespace matmul;
#if defined(__CCE_KT_TEST__)
using namespace std;
#endif

const uint64_t NUM_TWO = 2;
const uint64_t NUM_THREE = 3;
const uint64_t NUM_FOUR = 4;
const uint64_t DATA_SIZE_FP16 = 2;
const uint64_t DATA_SIZE_FP32 = 4;
const uint64_t BANK_SIZE = 256;
const uint64_t ALIGN_BYTE = 256;
const uint64_t ALIGN_128_BYTE = 128;
const uint64_t MAX_BLOCK_NUM = 100;
const uint64_t DEFAULT_BLOCK_LEN = 8;
const uint64_t BLOCK_SIZE = 16;


const uint32_t ROW_FIRST = 1;
const uint32_t COL_FIRST = 2;
const uint32_t SELECT_COL_ROW_FIRST_MULTI = 5;
const uint32_t SELECT_COL_ROW_FIRST_MULTI_L2_CACHE = 2;
const uint32_t CONTROL_DB = 1;
const uint32_t MAX_UINT16 = 65535;
const uint32_t ALL_L2_CACHE_ENABLE = 1;
const uint32_t A_L2_DISABLE = 2;
const uint32_t B_L2_DISABLE = 4;
const uint32_t BIAS_L2_DISABLE = 8;
const uint32_t C_L2_DISABLE = 16;
const uint32_t UBSIZE_910BC = 192 * 1024;
const uint32_t BLOCK_COUNT_MAX = 4095;

// 512 byte
const uint32_t MM_ALIGN_SIZE = 512;
const uint32_t VNCHW_CONV_ADDR_LIST_SIZE = 16;
const uint64_t AIV_SYNC_AIC_FLAG = 4;
const uint64_t AIC_SYNC_AIV_FLAG = 6;
const uint64_t ND2NZ_AIV_SYNC_AIC_FLAG = 8;

// set isVecND2Nz
constexpr MatmulConfig MM_CFG_VEC_ND2NZ = GetMDLConfig(false, false, 0, true);
// set enUnitFlag
constexpr MatmulConfig MM_CFG_NO_PRELOAD = GetMDLConfig(false, false, 0, false, false, false, true);
// set doMTE2Preload
constexpr MatmulConfig MM_CFG_PRELOAD = GetMDLConfig(false, false, 1);

enum ND2NZ_SELECT {
    ONLY_A = 1,
    ONLY_B = 2,
    BOTH_AB = 3
};


#if defined(__CCE_KT_TEST__)
#define SET_G_CORE_TYPE_IS_AIV thread_local int g_coreType = 2
#define SET_G_CORE_TYPE_IS_AIC thread_local int g_coreType = 1
#else
#define SET_G_CORE_TYPE_IS_AIV
#define SET_G_CORE_TYPE_IS_AIC
#endif

template <HardEvent event>
__aicore__ inline void TPipeSetWaitFlag() {
    auto eventID = GetTPipePtr()->FetchEventID(event);
    SetFlag<event>(eventID);
    WaitFlag<event>(eventID);
}

template <class T>
__aicore__ inline void GetSizeC0(uint64_t &c0Size) {
    if (sizeof(T) == sizeof(float)) {
        c0Size = 8;
    } else if (sizeof(T) == sizeof(int8_t)) {
        c0Size = 32;
    } else {
        c0Size = 16;
    }
}

template <class A_T, class B_T, class C_T, class BiasT>
__aicore__ inline void SetL2CacheEnable(const L2cacheUseInfo& l2EnableInfo,
    GlobalTensor<A_T> &aGlobal, GlobalTensor<B_T> &bGlobal,
    GlobalTensor<C_T> &cGlobal, GlobalTensor<BiasT> &biasGlobal)
{
    if ((l2EnableInfo.l2CacheFlag & ALL_L2_CACHE_ENABLE) == 0) {
        if (l2EnableInfo.l2CacheFlag & C_L2_DISABLE) {
            cGlobal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
    }
}

/**
 * if b is 0, return a
 */
__aicore__ inline uint64_t MMV3DivCeil(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

/**
 * if b is 0, return 0
 */
__aicore__ inline uint64_t MMV3CeilAlign(uint64_t a, uint64_t b)
{
    return MMV3DivCeil(a, b) * b;
}

/**
 * if b is 0, return a
 */
__aicore__ inline uint64_t MMV3DivFloor(uint64_t a, uint64_t b)
{
    return b == 0 ? a : a / b;
}

/**
 * if b is 0, return 0
 */
__aicore__ inline uint64_t MMV3FloorAlign(uint64_t a, uint64_t b)
{
    return b == 0 ? 0 : a / b * b;
}

#if defined(__DAV_C310__)
template <class T>
__aicore__ inline void CopyGmToUbufAlign(__ubuf__ void* dst, __gm__ void* src, uint8_t sid, uint16_t nBurst,
    uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum, uint32_t srcGap, uint32_t dstGap)
{
    bool constantPaddingCtl = true;
    uint8_t l2CacheCtl = 0;
    uint32_t dstStride = lenBurst + dstGap * 32 + (leftPaddingNum + rightPaddingNum) * sizeof(T); // 32 is blocksize
    copy_gm_to_ubuf_align_v2((__ubuf__ T*)dst, (__gm__ T*)src, sid, nBurst, lenBurst, leftPaddingNum, rightPaddingNum,
                             constantPaddingCtl, l2CacheCtl, (lenBurst + srcGap), dstStride);
}

template <typename T>
__aicore__ inline void CopyUbufToGmAlign(__gm__ void* dst, __ubuf__ void* src, uint8_t sid, uint16_t nBurst,
    uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum, uint32_t srcGap, uint32_t dstGap)
{
    uint8_t l2CacheCtl = 0;
    copy_ubuf_to_gm_align_v2((__gm__ T*)dst, (__ubuf__ T*)src, sid, nBurst, lenBurst, l2CacheCtl,
                             (lenBurst + dstGap), (lenBurst + srcGap * 32)); // 32 is blocksize
}
#else
template <class T>
__aicore__ inline void CopyGmToUbufAlign(__ubuf__ void* dst, __gm__ void* src, uint8_t sid, uint16_t nBurst,
    uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum, uint32_t srcGap, uint32_t dstGap)
{
    if constexpr (sizeof(T) == 1) {
        copy_gm_to_ubuf_align_b8(dst, src, sid, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
    } else if constexpr (sizeof(T) == DATA_SIZE_FP16) {
        copy_gm_to_ubuf_align_b16(dst, src, sid, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
    } else if constexpr (sizeof(T) == DATA_SIZE_FP32) {
        copy_gm_to_ubuf_align_b32(dst, src, sid, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
    } else {
        ASSERT(false);
    }
}

template <typename T>
__aicore__ inline void CopyUbufToGmAlign(__gm__ void* dst, __ubuf__ void* src, uint8_t sid, uint16_t nBurst,
    uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum, uint32_t srcGap, uint32_t dstGap)
{
    if constexpr (sizeof(T) == 1) {
        copy_ubuf_to_gm_align_b8(dst, src, sid, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
    } else if constexpr (sizeof(T) == DATA_SIZE_FP16) {
        copy_ubuf_to_gm_align_b16(dst, src, sid, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
    } else if constexpr (sizeof(T) == DATA_SIZE_FP32) {
        copy_ubuf_to_gm_align_b32(dst, src, sid, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
    } else {
        ASSERT(false);
    }
}
#endif

template <typename T>
__aicore__ inline void CopyCast(const LocalTensor<float>& ubSrc, const LocalTensor<T>& ubDst, __gm__ float* src,
    __gm__ T* dst, uint64_t offset, uint16_t nBurst, uint16_t lenBurst, uint32_t gap, uint8_t pingpongEventId)
{
    CopyGmToUbufAlign<float>((__ubuf__ float*)ubSrc.GetPhyAddr(), (__gm__ float*)src + offset, 0, nBurst,
        lenBurst * sizeof(float), 0, 0, gap * sizeof(float), 0);
    SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(pingpongEventId));
    WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(pingpongEventId));
    Cast(ubDst, ubSrc, RoundMode::CAST_ROUND, nBurst * lenBurst);
    SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(pingpongEventId));
    WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(pingpongEventId));
    CopyUbufToGmAlign<T>((__gm__ T *)dst + offset, (__ubuf__ T*)ubDst.GetPhyAddr(), 0, nBurst,
        lenBurst * sizeof(T), 0, 0, 0, gap * sizeof(T));
}

#define DBCAST
#ifdef DBCAST
// v220
template <typename T>
__aicore__ inline void Cast32to16V220(__gm__ T *dst, __gm__ float *src, uint64_t size, uint32_t nCoreUse,
    uint32_t n, TBuf<TPosition::VECCALC> &tmpBuf)
{
    uint16_t dataSize = static_cast<uint16_t>(TOTAL_UB_SIZE / NUM_TWO / sizeof(float));
    uint16_t dataSize1 = static_cast<uint16_t>(TOTAL_UB_SIZE / NUM_TWO / sizeof(T));
    LocalTensor<T> ubDstPing = tmpBuf.Get<T>();
    LocalTensor<float> ubSrcPing = ubDstPing.template ReinterpretCast<float>();
    LocalTensor<T> ubDstPong = ubDstPing[dataSize1];
    LocalTensor<float> ubSrcPong = ubSrcPing[dataSize];

    LocalTensor<T> ubDst = ubDstPing;
    LocalTensor<float> ubSrc = ubSrcPing;

    GlobalTensor<float> gmSrc;
    GlobalTensor<T> gmDst;
    gmSrc.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(src), size);
    gmDst.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dst), size);

    uint8_t pingpongEventId = 0;

    if (nCoreUse >= dataSize) { // 操作的最小一行数据量超ub时，对于每一行数据循环做cast处理
        uint64_t mRepeat = size / nCoreUse;
        uint16_t nBurst = static_cast<uint16_t>(nCoreUse / dataSize);
        uint16_t tail = static_cast<uint16_t>(nCoreUse % dataSize);

        for (uint64_t i = 0; i < mRepeat; ++i) {
            SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
            SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(1));
            for (uint16_t j = 0; j < nBurst; ++j) {
                if ((j & CONTROL_DB) == 0) {
                    pingpongEventId = 0;
                    ubDst = ubDstPing;
                    ubSrc = ubSrcPing;
                } else {
                    pingpongEventId = 1;
                    ubDst = ubDstPong;
                    ubSrc = ubSrcPong;
                }
                WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(pingpongEventId));
                CopyCast<T>(ubSrc, ubDst, src, dst, i * n + j * dataSize, 1, dataSize, 0, pingpongEventId);
                SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(pingpongEventId));
            }
            WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
            WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(1));
            if (tail > 0) {
                CopyCast<T>(ubSrc, ubDst, src, dst, i * n + nBurst * dataSize, 1, tail, 0, pingpongEventId);
            }
        }
        return;
    }

    uint16_t nBurst = static_cast<uint16_t>(dataSize / nCoreUse);
    uint64_t repeat = size / (nBurst * nCoreUse);
    uint16_t tail = static_cast<uint16_t>(size % (nBurst * nCoreUse));

    SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
    SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(1));
    for (uint64_t i = 0; i < repeat; ++i) {
        if ((i & CONTROL_DB) == 0) {
            pingpongEventId = 0;
            ubDst = ubDstPing;
            ubSrc = ubSrcPing;
        } else {
            pingpongEventId = 1;
            ubDst = ubDstPong;
            ubSrc = ubSrcPong;
        }
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(pingpongEventId));
        CopyCast<T>(ubSrc, ubDst, src, dst, i * nBurst * n, nBurst, static_cast<uint16_t>(nCoreUse),
            n - nCoreUse, pingpongEventId);
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(pingpongEventId));
    }
    WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
    WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(1));
    if ((repeat & CONTROL_DB) == 0) {
        pingpongEventId = 0;
        ubDst = ubDstPing;
        ubSrc = ubSrcPing;
    } else {
        pingpongEventId = 1;
        ubDst = ubDstPong;
        ubSrc = ubSrcPong;
    }
    if (tail > 0) {
        uint16_t tailNBurst = static_cast<uint16_t>(tail / nCoreUse);
        CopyCast<T>(ubSrc, ubDst, src, dst, repeat * nBurst * n, tailNBurst, static_cast<uint16_t>(nCoreUse),
            n - nCoreUse, pingpongEventId);
    }
    return;
}

template <typename T>
__aicore__ inline void UnAlignedCast32to16V220(__gm__ T *dst, __gm__ float *src, uint32_t offset, uint32_t size,
    TBuf<TPosition::VECCALC> &tmpBuf)
{
    uint32_t dataSize = TOTAL_UB_SIZE / NUM_TWO / sizeof(float);
    uint32_t dataSize1 = TOTAL_UB_SIZE / NUM_TWO / sizeof(T);
    LocalTensor<T> ubDstPing = tmpBuf.Get<T>();
    LocalTensor<float> ubSrcPing = ubDstPing.template ReinterpretCast<float>();
    LocalTensor<T> ubDstPong = ubDstPing[dataSize1];
    LocalTensor<float> ubSrcPong = ubSrcPing[dataSize];

    LocalTensor<T> ubDst = ubDstPing;
    LocalTensor<float> ubSrc = ubSrcPing;

    GlobalTensor<float> gmSrc;
    GlobalTensor<T> gmDst;
    gmSrc.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(src), size);
    gmDst.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dst), size);

    uint32_t repeat = size / dataSize;
    uint32_t tail = size % dataSize;

    uint8_t pingpongEventId = 0;

    SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
    SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(1));

    for (uint32_t i = 0; i < repeat; ++i) {
        if ((i & CONTROL_DB) == 0) {
            pingpongEventId = 0;
            ubDst = ubDstPing;
            ubSrc = ubSrcPing;
        } else {
            pingpongEventId = 1;
            ubDst = ubDstPong;
            ubSrc = ubSrcPong;
        }
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(pingpongEventId));
        CopyCast<T>(ubSrc, ubDst, src, dst, i * dataSize, 1, dataSize, 0, pingpongEventId);
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(pingpongEventId));
    }
    WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(0));
    WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(1));
    if ((repeat & CONTROL_DB) == 0) {
        pingpongEventId = 0;
        ubDst = ubDstPing;
        ubSrc = ubSrcPing;
    } else {
        pingpongEventId = 1;
        ubDst = ubDstPong;
        ubSrc = ubSrcPong;
    }
    if (tail > 0) {
        CopyCast<T>(ubSrc, ubDst, src, dst, repeat * dataSize, 1, tail, 0, pingpongEventId);
    }
    return;
}

#endif

template <class T>
__aicore__ inline void CopyPadNd2Nz(const GlobalTensor<T>& dstGlobal, const GlobalTensor<T>& srcGlobal, uint32_t baseH,
    uint32_t baseW, uint32_t orgHeight, uint32_t orgWidth, LocalTensor<T> ubLocal1, LocalTensor<T> ubLocal2,
    uint8_t padH, uint8_t padW)
{
    uint64_t c0Size;
    GetSizeC0<T>(c0Size);

    uint32_t width = baseW - padW;
    uint32_t height = baseH - padH;

    // Process gm->ub
    uint16_t blockLen = width * sizeof(T);
    uint32_t srcStride = (orgWidth - width) * sizeof(T);
    uint32_t numIter = height / BLOCK_COUNT_MAX;
    for (uint32_t i = 0; i < numIter; i++) {
        DataCopyPad(ubLocal1[BLOCK_COUNT_MAX * i * baseW],
            srcGlobal[static_cast<uint64_t>(BLOCK_COUNT_MAX) * i * orgWidth],
            {BLOCK_COUNT_MAX, blockLen, srcStride, 0, 0}, {false, 0, padW, 0});
    }
    uint16_t blockCountTail = height % BLOCK_COUNT_MAX;

    if (blockCountTail) {
        DataCopyPad(ubLocal1[BLOCK_COUNT_MAX * numIter * baseW],
            srcGlobal[static_cast<uint64_t>(BLOCK_COUNT_MAX) * numIter * orgWidth],
            {blockCountTail, blockLen, srcStride, 0, 0}, {false, 0, padW, 0});
    }

    SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));
    WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(0));

    // padding
    if (padH) {
        Duplicate(ubLocal1[height * baseW], (T)0, padH * baseW);
        PipeBarrier<PIPE_V>();
    }

    // Process ub->ub
    uint32_t nRepeat = SINGLE_COPY_SIZE / sizeof(T);
    uint16_t nRowBlock = baseW / c0Size;
    uint32_t numIterI = baseH / REPEAT_TIMES_MAX;
    uint32_t heightTail = baseH % REPEAT_TIMES_MAX;
    uint32_t numIterJ = baseW / nRepeat;
    uint32_t widthTail = baseW % nRepeat;

    for (uint32_t i = 0; i < numIterI; i++) {
        for (uint32_t j = 0; j < numIterJ; j++) {
            Copy(ubLocal2[baseH * nRepeat * j + i * REPEAT_TIMES_MAX * c0Size],
                ubLocal1[nRepeat * j + i * REPEAT_TIMES_MAX * baseW], nRepeat, REPEAT_TIMES_MAX,
                {static_cast<uint16_t>(baseH), 1, 1, nRowBlock});
        }
        if (widthTail) {
            Copy(ubLocal2[baseH * nRepeat * numIterJ + i * REPEAT_TIMES_MAX * c0Size],
                ubLocal1[nRepeat * numIterJ + i * REPEAT_TIMES_MAX * baseW], widthTail, REPEAT_TIMES_MAX,
                {static_cast<uint16_t>(baseH), 1, 1, nRowBlock});
        }
    }
    for (uint32_t j = 0; j < numIterJ; j++) {
        Copy(ubLocal2[baseH * nRepeat * j + numIterI * REPEAT_TIMES_MAX * c0Size],
            ubLocal1[nRepeat * j + numIterI * REPEAT_TIMES_MAX * baseW], nRepeat, heightTail,
            {static_cast<uint16_t>(baseH), 1, 1, nRowBlock});
    }
    if (widthTail) {
        Copy(ubLocal2[baseH * nRepeat * numIterJ + numIterI * REPEAT_TIMES_MAX * c0Size],
            ubLocal1[nRepeat * numIterJ + numIterI * REPEAT_TIMES_MAX * baseW], widthTail, heightTail,
            {static_cast<uint16_t>(baseH), 1, 1, nRowBlock});
    }

    SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));
    WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(0));

    uint64_t orgHeightRound = MMV3DivCeil(orgHeight, ALIGNED_H) * ALIGNED_H;
    // Process ub->gm
    if (orgHeightRound - baseH <= UINT16_MAX) {
        DataCopy(dstGlobal, ubLocal2, {nRowBlock, static_cast<uint16_t>(baseH), 0, uint16_t(orgHeightRound - baseH)});
    } else {
        for (uint16_t i = 0; i < nRowBlock; i++) {
            DataCopy(dstGlobal[orgHeightRound * c0Size * i], ubLocal2[baseH * c0Size * i],
                {1, static_cast<uint16_t>(baseH), 0, 0});
        }
    }
}

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ > 200)
template <>
__aicore__ inline void CopyPadNd2Nz<bfloat16_t>(const GlobalTensor<bfloat16_t>& dstGlobal, const GlobalTensor<bfloat16_t>& srcGlobal,
    uint32_t baseH, uint32_t baseW, uint32_t orgHeight, uint32_t orgWidth, LocalTensor<bfloat16_t> ubLocal1,
    LocalTensor<bfloat16_t> ubLocal2, uint8_t padH, uint8_t padW)
{
    GlobalTensor<half> dstGlobalTrans;
    GlobalTensor<half> srcGlobalTrans;
    dstGlobalTrans.SetGlobalBuffer((__gm__ half*)dstGlobal.GetPhyAddr(0));
    srcGlobalTrans.SetGlobalBuffer((__gm__ half*)srcGlobal.GetPhyAddr(0));
    CopyPadNd2Nz<half>(dstGlobalTrans, srcGlobalTrans, baseH, baseW, orgHeight, orgWidth, ubLocal1.ReinterpretCast<half>(),
        ubLocal2.ReinterpretCast<half>(), padH, padW);
}
#endif


template <typename T>
__aicore__ inline void TransDataTo5HDIn(const LocalTensor<T> dstTensor, const LocalTensor<T> srcTensor,
                                        uint64_t splitWidth, uint64_t NDimDataPerRound) {
    // Initialize the TransDataTo5HDParams for the first vnchw_conv
    // [single * 8 * 16 * k] --> [single * 16 * k * 8]
    // 拦截 Repeat = 1 的特殊分支
    TransDataTo5HDParams params;
    params.repeatTimes = splitWidth;
    // Has to process 16 elements
    params.dstRepStride = 16;
    // Has to process 16 elements, each C0 has 8 elements so src rep is 2
    params.srcRepStride = 2;
    uint64_t dstLocalList[VNCHW_CONV_ADDR_LIST_SIZE];
    uint64_t srcLocalList[VNCHW_CONV_ADDR_LIST_SIZE];
    uint64_t C0;
    GetSizeC0<T>(C0);
    if constexpr (sizeof(T) == sizeof(float)) {
        // Input src has to process 16 elements and repeat [splitWidth] times
        uint64_t srcOffset = 16 * splitWidth;
        // Dst's next addr is in the next 8 Block.
        uint64_t dstOffset = 8 * C0;
        uint64_t loopOffset = 128 * splitWidth;
        for (uint64_t outerLoop = 0; outerLoop < MMV3DivCeil(NDimDataPerRound, C0); outerLoop++) {
            for (uint32_t i = 0; i < VNCHW_CONV_ADDR_LIST_SIZE / 2; i++) {
                srcLocalList[i] = (uint64_t)srcTensor[outerLoop * loopOffset + i * srcOffset].GetPhyAddr();
                srcLocalList[i + 8] = (uint64_t)srcTensor[outerLoop * loopOffset + i * srcOffset + C0].GetPhyAddr();
            }
            for (uint32_t j = 0; j < VNCHW_CONV_ADDR_LIST_SIZE / 2; j++) {
                dstLocalList[2 * j] = (uint64_t)dstTensor[outerLoop * loopOffset + j * C0].GetPhyAddr();
                dstLocalList[2 * j + 1] = (uint64_t)dstTensor[outerLoop * loopOffset + j * C0 + dstOffset].GetPhyAddr();
            }
            TransDataTo5HD<T>(dstLocalList, srcLocalList, params);
        }
    }
}

template <typename T>
__aicore__ inline void CopyAndDuplicate(const LocalTensor<T> dstTensor, const LocalTensor<T> srcTensor,
                                        uint64_t splitWidth, uint64_t logicRepeatTimes, uint64_t innerBlockSize){
    uint64_t C0;
    GetSizeC0<T>(C0);
    // This func can only be used in Float dtype.
    const uint64_t REPEAT_MAX = 255UL;
    uint64_t mask = min(splitWidth * 8UL, 8UL * 8UL);
    // params{dstBlkStride, srcBlkStride, dstRepStride, srcRepStride} -> // 初始化vector的搬运参数
    CopyRepeatParams params{1, 1, static_cast<uint16_t>(innerBlockSize), static_cast<uint16_t>(splitWidth)};
    // The following binary number means the first 2 * splitWidth elements are skipped and the following
    // (64 - 2 * splitWidth) elements are padded zero.
    // can only process 63 bit data. when 4 * 2 * splitWidth equals 64, the res will be zero.
    uint64_t maskFp32 = 4 * 2 * splitWidth == 64 ? 0 : 0xffffffffffffffff << (4 * 2 * splitWidth);
    uint64_t duplicatedMask[2] = {maskFp32, 0};
    uint64_t roundTimes = MMV3DivCeil(logicRepeatTimes, REPEAT_MAX); // 255 meas uint8Max
    uint8_t repeatTimes = static_cast<uint8_t>(min(logicRepeatTimes, REPEAT_MAX));

    uint8_t repeatTail = repeatTimes;
    if (roundTimes > 1) {
        repeatTail = static_cast<uint8_t>(logicRepeatTimes - (roundTimes - 1) * REPEAT_MAX);
    }
    for (uint64_t roundIdx = 0; roundIdx < roundTimes; roundIdx++) {
        // 8 means the single loop process N dim data multi.
        if (roundIdx == roundTimes - 1) {
            repeatTimes = repeatTail;
        }
        uint64_t srcOffset = splitWidth * C0 * REPEAT_MAX * roundIdx;
        // In Float Mode each Copy process 8 * C0 elements.
        uint64_t dstOffset = 8 * C0 * REPEAT_MAX * roundIdx;
        Copy(dstTensor[dstOffset], srcTensor[srcOffset], mask, repeatTimes, params);
        PipeBarrier<PIPE_V>();
        // params{dstLocal, value, mask, repeat = repeatTimes, dstBlockStride = 1, dstRepStride = 8}
        // 初始化vector的搬运参数
        Duplicate(dstTensor[dstOffset], 0.0f, duplicatedMask, repeatTimes, 1, 8);
        // [single * 16 * k * 8] --> [k1, single * 16 * 8 * 8]
    }
}

template <typename T>
__aicore__ inline void TransDataTo5HDOut(const LocalTensor<T> dstTensor, const LocalTensor<T> srcTensor,
                                         uint64_t NDimDataPerRound) {
    // Initialize the TransDataTo5HDParams for the second vnchw_conv
    TransDataTo5HDParams params;
    params.repeatTimes = 8;
    params.dstRepStride = 2;
    params.srcRepStride = 16;
    uint64_t dstLocalList[VNCHW_CONV_ADDR_LIST_SIZE];
    uint64_t srcLocalList[VNCHW_CONV_ADDR_LIST_SIZE];
    uint64_t c0Size;
    GetSizeC0<T>(c0Size);
    if constexpr (sizeof(T) == sizeof(float)) {
        // VNCHW_CONV in Float
        uint64_t C0 = 8;
        // Dst for next dst Addr is 16 * 8 elements away.
        uint64_t dstOffset = 16 * 8;
        // Dst addr for next transData5HD is 128 * 8 elements aways
        uint64_t loopOffset = 128 * 8;
        for (uint64_t outerLoop = 0; outerLoop < MMV3DivCeil(NDimDataPerRound, C0); outerLoop++) {
            for (uint32_t i = 0; i < VNCHW_CONV_ADDR_LIST_SIZE; i++) {
                srcLocalList[i] = (uint64_t)srcTensor[outerLoop * loopOffset + i * C0].GetPhyAddr();
            }
            for (uint32_t j = 0; j < VNCHW_CONV_ADDR_LIST_SIZE / 2; j++) { // Process 2 addr each time.
                dstLocalList[2 * j] = (uint64_t)dstTensor[outerLoop * loopOffset + j * dstOffset].GetPhyAddr();
                dstLocalList[2 * j + 1] = (uint64_t)dstTensor[outerLoop * loopOffset + j * dstOffset + C0].GetPhyAddr();
            }
            TransDataTo5HD<T>(dstLocalList, srcLocalList, params);
        }
    }
}

template <typename T>
__aicore__ inline void SingleRoundUsingVnchwConv(const GlobalTensor<T> &workspace, const GlobalTensor<T> &src,
                                                 uint64_t splitWidth, uint64_t NDimSingleCoreData,
                                                 uint64_t innerBlockSize, uint64_t outerBlockSize,
                                                 TBuf<TPosition::VECCALC> &tmpBuf) {
    // Implement NzFsion using vnchw-conv
    // UB is devided into 2 Sections. Each Section contains 48 KB. Could be change into 96KB
    uint64_t dataSize = TOTAL_UB_SIZE / NUM_TWO / sizeof(T);
    LocalTensor<T> ubPingBuf0 = tmpBuf.Get<T>();
    LocalTensor<T> ubPingBuf1 = ubPingBuf0[dataSize];
    uint64_t nDimSingleCoreFracData = MMV3DivCeil(NDimSingleCoreData, outerBlockSize);
    uint64_t dataCopyCalCount = MMV3CeilAlign(NDimSingleCoreData * splitWidth, innerBlockSize);
    uint64_t nzOutCalCount = nDimSingleCoreFracData * outerBlockSize * MMV3CeilAlign(splitWidth, innerBlockSize);
    uint8_t pingPongEventId = 0;
    auto bufA = ubPingBuf0;
    auto bufB = ubPingBuf1;
    DataCopy(bufA, src, dataCopyCalCount);  // MoveIn()
    // [m, k] --> [single * 8 * 16 * k]
    SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(pingPongEventId));
    WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(pingPongEventId));
    TransDataTo5HDIn<T>(bufB, bufA, splitWidth, nDimSingleCoreFracData);
    // [single * 8 * 16 * k] --> [single * 16 * k * 8]
    PipeBarrier<PIPE_V>();
    uint64_t logicRepeatTimes = MMV3DivCeil(nDimSingleCoreFracData, innerBlockSize) * outerBlockSize;
    CopyAndDuplicate<T>(bufA, bufB, splitWidth, logicRepeatTimes, innerBlockSize);
    PipeBarrier<PIPE_V>();
    TransDataTo5HDOut<T>(bufB, bufA, nDimSingleCoreFracData);
    SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(pingPongEventId));
    WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(pingPongEventId));
    DataCopy(workspace, bufB, nzOutCalCount);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void MatrixtoNZWithVnchwConv(GM_ADDR workspace, GM_ADDR src, const TCubeTiling &cfg, bool isAMatrix,
                                               bool isTranspose, TBuf<TPosition::VECCALC> &tmpBuf) {
  if ASCEND_IS_AIV {
    uint64_t c0Size;
    GetSizeC0<T>(c0Size);
    uint64_t splitWidth = isAMatrix ? cfg.Ka : cfg.N;
    uint64_t nDimension = isAMatrix ? cfg.M : cfg.Kb;
    uint64_t innerBlockSize = c0Size;
    uint64_t outerBlockSize = BLOCK_SIZE;
    uint64_t usedCoreNum = cfg.usedCoreNum * GetTaskRation();  // 使用最大的核数
    if (isTranspose) {
      splitWidth = isAMatrix ? cfg.M : cfg.Kb;
      nDimension = isAMatrix ? cfg.Ka : cfg.N;
    }
    uint64_t dataSize = TOTAL_UB_SIZE / NUM_TWO / sizeof(T);
    uint64_t nDimBasicBlockSize =
        MMV3CeilAlign(dataSize / innerBlockSize - innerBlockSize * outerBlockSize, innerBlockSize * outerBlockSize);
    uint64_t nDimDataPerCore = nDimBasicBlockSize;
    uint64_t totalRound = MMV3DivCeil(nDimension, nDimBasicBlockSize);
    uint64_t nDimTail = nDimension - (totalRound - 1) * nDimBasicBlockSize;
    uint64_t roundPerCore = totalRound / usedCoreNum;
    uint64_t extraRoundCoreNum = totalRound % usedCoreNum;
    uint64_t roundIdx = GetBlockIdx() * roundPerCore + extraRoundCoreNum;  // 哪一个roundIdx开始
    if (GetBlockIdx() < extraRoundCoreNum) {
      roundPerCore = roundPerCore + 1;
      roundIdx = GetBlockIdx() * roundPerCore;
    }
    uint64_t fractalPerNum = MMV3DivCeil(splitWidth, innerBlockSize);
    uint64_t NDimFractalNum = MMV3DivCeil(nDimension, outerBlockSize);
    // Main Block Size
    uint64_t workSpaceSize =
        NDimFractalNum * outerBlockSize * fractalPerNum * innerBlockSize;                // N-single-Align * D-Align
    uint64_t srcSize = nDimension * splitWidth;                                          // N-single * D
    uint64_t eachBlockWorkSpaceSize = nDimDataPerCore * fractalPerNum * innerBlockSize;  // N-single * D-Align
    GlobalTensor<T> vnchwDstTensor;
    GlobalTensor<T> vnchwSrcTensor;
    vnchwDstTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace), workSpaceSize);
    vnchwSrcTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(src), srcSize);
    // 当前该函数仅允许N维度16对齐才可使用
    for (uint64_t idxCnt = 0; idxCnt < roundPerCore; idxCnt++) {
      size_t srcGmIdx = (roundIdx + idxCnt) * nDimDataPerCore * splitWidth;
      size_t dstGmIdx = (roundIdx + idxCnt) * eachBlockWorkSpaceSize;
      if ((roundIdx + idxCnt) == (totalRound - 1)) {
        nDimDataPerCore = nDimTail;
      }
      SingleRoundUsingVnchwConv<T>(vnchwDstTensor[dstGmIdx], vnchwSrcTensor[srcGmIdx], splitWidth, nDimDataPerCore,
                                   innerBlockSize, outerBlockSize, tmpBuf);
    }
  }
}

template <class T>
__aicore__ inline void MatrixtoNZ(uint64_t oriN, uint64_t oriD, uint64_t nValue, uint64_t dValue, uint32_t baseN,
    uint32_t baseD, uint32_t usedCoreNum, GlobalTensor<T>& tempSrcGlobal, GlobalTensor<T>& tempDstGlobal,
    TBuf<TPosition::VECCALC>& tmpBuf)
{
    if ASCEND_IS_AIV {
        uint32_t vBlockIndex = GetBlockIdx();
        uint64_t c0Size;
        GetSizeC0<T>(c0Size);
        LocalTensor<T> tempUb = tmpBuf.Get<T>();
        LocalTensor<T> transBuf = tempUb[(TOTAL_UB_SIZE / NUM_TWO) / sizeof(T)];
        uint64_t nCnt = MMV3DivCeil(oriN, baseN);
        uint64_t dCnt = MMV3DivCeil(dValue, baseD);
        uint64_t nBaseTil = nValue - (nCnt - 1) * baseN;   // n方向上的baseN尾块
        uint64_t dBaseTail = dValue - (dCnt - 1) * baseD;  // m方向上的baseM尾块
        uint64_t totalCnt = nCnt * dCnt;
        uint32_t round = (totalCnt + usedCoreNum - 1) / usedCoreNum;  // 每一个core最大做base块计算的次数
        uint32_t realRound = 0;                                       // 单核做多少次base块计算
        uint32_t preCoreNum = totalCnt % usedCoreNum;  // 从0core开始有多少个core会多做一次base块
        uint32_t preTotalBlock = 0;
        uint32_t index = 0;  // 当前block_idx的起始基本块Index，这个idex是按照先循环D，再循环N的次序
        if (preCoreNum == 0) {
            preCoreNum = usedCoreNum;
        }
        // ND
        if (vBlockIndex < preCoreNum) {
            index = vBlockIndex * round;
            // 前面preCoreNum个core会多做一次
            realRound = round;
        } else {
            index = vBlockIndex * (round - 1) + preCoreNum;
            // 后面的core会少做一次
            realRound = round - 1;
        }
        uint32_t nCalcLen = 0;
        uint32_t dCalcLen = 0;
        uint32_t padN = 0;
        uint32_t padD = 0;
        uint32_t nIndx = 0;
        uint32_t dIndx = 0;
        uint32_t lastD = oriD % baseD;
        for (uint32_t j = 0; j < realRound; ++j) {
            if (index < totalCnt) {
                if ((index + 1) % (nCnt * dCnt) == 0) {
                    // 最后一块是尾块
                    nCalcLen = nBaseTil;
                    dCalcLen = dBaseTail;
                } else if ((index + 1) % (nCnt * dCnt) > (nCnt - 1) * dCnt) {
                    // n方向尾块
                    nCalcLen = nBaseTil;
                    dCalcLen = baseD;
                } else if ((index + 1) % dCnt == 0) {
                    // d方向尾块
                    nCalcLen = baseN;
                    dCalcLen = dBaseTail;
                } else {
                    // 对齐整块
                    nCalcLen = baseN;
                    dCalcLen = baseD;
                }
            }
            // calc pad_value
            nIndx = index / dCnt;
            dIndx = index % dCnt;
            padN = (nIndx == nCnt - 1) ? nValue - oriN : 0;
            padD = (dIndx == dCnt - 1) ? dValue - oriD : 0;  // will be used ???
            auto srcGmIdx = (dIndx * baseD + nIndx * baseN * oriD);
            auto dstGmIdx = (dIndx * nValue * baseD + nIndx * baseN * c0Size);
            CopyPadNd2Nz(tempDstGlobal[dstGmIdx], tempSrcGlobal[srcGmIdx], nCalcLen, dCalcLen, oriN, oriD, tempUb,
                transBuf, padN, padD);
            event_t eventMTE3MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2);
            index += 1;
        }
    }
}

template <class T>
__aicore__ inline bool NeedToUseVnchwConv(uint64_t oriN, uint64_t oriD) {
    // Only support Float Mode now.
    uint64_t c0Size;
    GetSizeC0<T>(c0Size);
    // 72368 is a experiment threshold.
    if (oriN >= 72368 && oriD <= c0Size && oriD > 1) {
        return true;
    }
    return false;
}

template <class T>
__aicore__ inline void MatrixAtoNZV2(GM_ADDR workspace, GM_ADDR src, const TCubeTiling &cfg, bool isTransposeA,
    TBuf<TPosition::VECCALC>& tmpBuf, uint32_t baseAN, uint32_t baseAD, uint32_t batch = 1) {
    uint64_t c0Size;
    GetSizeC0<T>(c0Size);
    uint32_t usedCoreNum = cfg.usedCoreNum * GetTaskRation();  // 使用最大的核数
    uint64_t alignedMSize = 0;
    uint64_t alignedKSize = 0;
    alignedMSize = isTransposeA ? MMV3DivCeil(cfg.M, c0Size) * c0Size
                                : MMV3DivCeil(cfg.M, ALIGNED_H) * ALIGNED_H;  // N轴转换成分型
    alignedKSize = isTransposeA ? MMV3DivCeil(cfg.Ka, ALIGNED_H) * ALIGNED_H
                                : MMV3DivCeil(cfg.Ka, c0Size) * c0Size;  // K轴转换成分型
    uint64_t oriN = isTransposeA ? cfg.Ka : cfg.M;
    uint64_t oriD = isTransposeA ? cfg.M : cfg.Ka;
    uint64_t nValue = isTransposeA ? alignedKSize : alignedMSize;
    uint64_t dValue = isTransposeA ? alignedMSize : alignedKSize;
    if constexpr (sizeof(T) == sizeof(float)) {
        if (batch == 1 && NeedToUseVnchwConv<T>(oriN, oriD)) {
            MatrixtoNZWithVnchwConv<T>(workspace, src, cfg, true, isTransposeA, tmpBuf);
            return;
        }
    }
    GlobalTensor<T> tempSrcGlobal;
    GlobalTensor<T> tempDstGlobal;
    tempDstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace), alignedMSize * alignedKSize);
    tempSrcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src), oriD * oriN);
#if defined(__DAV_C220_VEC__)
    if (batch > 1) {
        Nd2nzVnchw(tempDstGlobal, tempSrcGlobal, oriN, oriD, batch, tmpBuf, usedCoreNum);
    } else {
        MatrixtoNZ(oriN, oriD, nValue, dValue, baseAN, baseAD, usedCoreNum, tempSrcGlobal, tempDstGlobal, tmpBuf);
    }
#else
    MatrixtoNZ(oriN, oriD, nValue, dValue, baseAN, baseAD, usedCoreNum, tempSrcGlobal, tempDstGlobal, tmpBuf);
#endif
}

template <class T>
__aicore__ inline void MatrixBtoNZV2(GM_ADDR workspace, GM_ADDR src, const TCubeTiling &cfg, bool isTransposeB,
                                     TBuf<TPosition::VECCALC> &tmpBuf, uint32_t baseBN, uint32_t baseBD, uint32_t batch = 1) {
    uint64_t c0Size;
    GetSizeC0<T>(c0Size);
    uint32_t usedCoreNum = cfg.usedCoreNum * GetTaskRation();  // 使用最大的核数
    uint64_t alignedNSize = 0;
    uint64_t alignedKSize = 0;
    alignedNSize = isTransposeB ? MMV3DivCeil(cfg.N, ALIGNED_H) * ALIGNED_H : MMV3DivCeil(cfg.N, c0Size) * c0Size;  // N轴转换成分型
    alignedKSize = isTransposeB ? MMV3DivCeil(cfg.Kb, c0Size) * c0Size : MMV3DivCeil(cfg.Kb, ALIGNED_H) * ALIGNED_H;       // K轴转换成分型
    uint64_t oriN = isTransposeB ? cfg.N : cfg.Kb;
    uint64_t oriD = isTransposeB ? cfg.Kb : cfg.N;
    uint64_t nValue = isTransposeB ? alignedNSize : alignedKSize;
    uint64_t dValue = isTransposeB ? alignedKSize : alignedNSize;
    if constexpr (sizeof(T) == sizeof(float)) {
        if (batch == 1 && NeedToUseVnchwConv<T>(oriN, oriD)) {
            MatrixtoNZWithVnchwConv<T>(workspace, src, cfg, false, isTransposeB, tmpBuf);
            return;
        }
    }
    GlobalTensor<T> tempSrcGlobal1;
    GlobalTensor<T> tempDstGlobal1;
    tempDstGlobal1.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace), alignedNSize * alignedKSize);
    tempSrcGlobal1.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src), oriD * oriN);
#if defined(__DAV_C220_VEC__)
    if (batch > 1) {
        Nd2nzVnchw(tempDstGlobal1, tempSrcGlobal1, oriN, oriD, batch, tmpBuf, usedCoreNum);
    } else {
        MatrixtoNZ(oriN, oriD, nValue, dValue, baseBN, baseBD, usedCoreNum, tempSrcGlobal1, tempDstGlobal1, tmpBuf);
    }
#else
    MatrixtoNZ(oriN, oriD, nValue, dValue, baseBN, baseBD, usedCoreNum, tempSrcGlobal1, tempDstGlobal1, tmpBuf);
#endif
}


template <typename T1, typename T2>
__aicore__ inline void CopyRemovePad(const GlobalTensor<T2>& outputGlobal, const GlobalTensor<T1>& inputGlobal,
    const LocalTensor<T1>& srcUb, const LocalTensor<T2>& castDstUb, uint32_t nBurst, uint32_t inputWidth,
    uint32_t outputWidth)
{
    CopyGmToUbufAlign<T1>((__ubuf__ void*)srcUb.GetPhyAddr(), (__gm__ void*)inputGlobal.GetPhyAddr(), 0,
                          static_cast<uint16_t>(nBurst), inputWidth * sizeof(T1), 0, 0, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
    wait_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
    Cast(castDstUb, srcUb, RoundMode::CAST_ROUND, nBurst * inputWidth);
    set_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
    wait_flag(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
    uint32_t srcGap = (inputWidth - outputWidth) * sizeof(T2) / 32; // 32 is blocksize
    CopyUbufToGmAlign<T2>((__gm__ void*)outputGlobal.GetPhyAddr(), (__ubuf__ void*)castDstUb.GetPhyAddr(), 0,
        static_cast<uint16_t>(nBurst), outputWidth * sizeof(T2), 0, 0, srcGap, 0);
}

template <typename T1, typename T2>
__aicore__ inline void RemovePaddingImpl(GlobalTensor<T2> outputGlobal, GlobalTensor<T1> inputGlobal,
    uint32_t height, uint32_t width, uint32_t outputWidth, TBuf<TPosition::VECCALC> &tmpBuf)
{
    LocalTensor<T1> srcUb = tmpBuf.Get<T1>();
    LocalTensor<T2> castDstUb = srcUb.template ReinterpretCast<T2>();

    uint32_t nBurst = TOTAL_UB_SIZE / (width * sizeof(T1));
    if (nBurst == 0) {
        uint32_t maxWidthLen = TOTAL_UB_SIZE / sizeof(T1);
        uint32_t castTimes = width / maxWidthLen;
        uint32_t tailWidth = width - castTimes * maxWidthLen;
        uint32_t tailOutWidth = outputWidth - castTimes * maxWidthLen;
        for (uint32_t mIndex = 0; mIndex < height; ++mIndex) {
            set_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(0));
            for (uint32_t index = 0; index < castTimes; ++index) {
                wait_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(0));
                UnAlignedCast32to16V220((__gm__ T2 *)(outputGlobal[index * maxWidthLen + mIndex * outputWidth].GetPhyAddr()),
                               (__gm__ float *)(inputGlobal[index * maxWidthLen + mIndex * width].GetPhyAddr()),
                               0, maxWidthLen, tmpBuf);
                set_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(0));
            }
            wait_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(0));
            if (tailWidth != 0) {
                CopyRemovePad(outputGlobal[castTimes * maxWidthLen + mIndex * outputWidth],
                              inputGlobal[castTimes * maxWidthLen + mIndex * width], srcUb, castDstUb,
                              1, tailWidth, tailOutWidth);
            }
        }
        return;
    }
    uint32_t nBurstTimes = height / nBurst;
    uint32_t nBurstTail = height - nBurstTimes * nBurst;
    set_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(0));
    for (uint32_t i = 0; i < nBurstTimes; ++i) {
        wait_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(0));
        CopyRemovePad(outputGlobal[i * nBurst * outputWidth], inputGlobal[i * nBurst * width], srcUb, castDstUb,
                      nBurst, width, outputWidth);
        set_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(0));
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(0));
    if (nBurstTail > 0) {
        CopyRemovePad(outputGlobal[nBurstTimes * nBurst * outputWidth], inputGlobal[nBurstTimes * nBurst * width],
                      srcUb, castDstUb, nBurstTail, width, outputWidth);
    }
}

template <class T>
__aicore__ inline void SplitKVectorProcess(LocalTensor<float> ubSrc1, LocalTensor<float> ubSrc2,
                                           GlobalTensor<float> gmSrc, LocalTensor<T> ubDst, GlobalTensor<T> gmDst,
                                           uint64_t vIndex, uint64_t mIndex, uint64_t currentLoop,
                                           uint64_t dataSizeToMove, uint64_t dataSize,
                                           uint64_t coreSize, uint64_t singleSize, uint64_t totalSize)
{
    CopyGmToUbufAlign<float>((__ubuf__ float *)ubSrc1.GetPhyAddr(),
                             (__gm__ float *)gmSrc[currentLoop * dataSize + vIndex * coreSize].GetPhyAddr(),
                             0, 1, dataSizeToMove * sizeof(float), 0, 0, 0, 0);

    for (uint64_t j = 1; j < totalSize / singleSize; ++j) {
        CopyGmToUbufAlign<float>((__ubuf__ float *)ubSrc2.GetPhyAddr(), (__gm__ float *)gmSrc[currentLoop *
                                 dataSize + vIndex * coreSize + j * singleSize * NUM_TWO].GetPhyAddr(),
                                 0, 1, dataSizeToMove * sizeof(float), 0, 0, 0, 0);
        // MTE2 to V, enable pingpong
        TPipeSetWaitFlag<HardEvent::MTE2_V>();
        Add(ubSrc1, ubSrc1, ubSrc2, dataSizeToMove);
        TPipeSetWaitFlag<HardEvent::V_MTE2>();
    }
    PipeBarrier<PIPE_V>();
    if constexpr (sizeof(T) == sizeof(half)) {
        Cast(ubDst, ubSrc1, RoundMode::CAST_ROUND, dataSizeToMove);
    }
    TPipeSetWaitFlag<HardEvent::V_MTE3>();
    if constexpr (sizeof(T) == sizeof(half)) {
        CopyUbufToGmAlign<T>((__gm__ T *)gmDst[currentLoop * dataSize + vIndex * coreSize +
                                mIndex * singleSize].GetPhyAddr(), (__ubuf__ T *)ubDst.GetPhyAddr(), 0, 1,
                                dataSizeToMove * sizeof(T), 0, 0, 0, 0);
    } else if constexpr (sizeof(T) == sizeof(float)) {
        CopyUbufToGmAlign<T>((__gm__ T *)gmDst[currentLoop * dataSize + vIndex * coreSize +
                                mIndex * singleSize].GetPhyAddr(), (__ubuf__ T *)ubSrc1.GetPhyAddr(), 0, 1,
                                dataSizeToMove * sizeof(T), 0, 0, 0, 0);
    }
}

template <class C_TYPE>
__aicore__ inline void ReduceKInUb(GM_ADDR cGM, GM_ADDR mmGM, uint64_t coreSize, uint64_t singleSize,
                                   uint64_t totalSize, uint64_t outSize, uint64_t mCnt,
                                   TBuf<TPosition::VECCALC> &tmpBuf)
{
    using T = typename C_TYPE::T;

    uint64_t dataSize = TOTAL_UB_SIZE / NUM_FOUR / sizeof(float);
    uint64_t dataSize1 = TOTAL_UB_SIZE / NUM_FOUR / sizeof(T);
    LocalTensor<T> ubDstPing = tmpBuf.Get<T>();
    LocalTensor<float> ubSrcPing1 = ubDstPing.template ReinterpretCast<float>();
    LocalTensor<float> ubSrcPing2 = ubSrcPing1[dataSize];

    LocalTensor<T> ubDstPong = ubDstPing[dataSize1 * NUM_TWO];
    LocalTensor<float> ubSrcPong1 = ubSrcPing1[dataSize * NUM_TWO];
    LocalTensor<float> ubSrcPong2 = ubSrcPong1[dataSize];

    LocalTensor<T> ubDst = ubDstPing;
    LocalTensor<float> ubSrc1 = ubSrcPing1;
    LocalTensor<float> ubSrc2 = ubSrcPing2;

    GlobalTensor<T> gmDst;
    GlobalTensor<float> gmSrcPing;
    GlobalTensor<float> gmSrcPong;
    GlobalTensor<float> gmSrc = gmSrcPing;

    gmDst.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(cGM), outSize);
    gmSrcPing.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mmGM), totalSize);
    gmSrcPong = gmSrcPing[singleSize];

    uint64_t vIndex = GetBlockIdx();
    uint8_t pingpongEventId = 0;
    uint8_t pingpongEventIdWS = 0;
    for (uint64_t mIndex = 0; mIndex < mCnt; ++mIndex) {
        if ((mIndex & CONTROL_DB) == 0) {
            pingpongEventIdWS = 0;
            gmSrc = gmSrcPing;
        } else {
            pingpongEventIdWS = 1;
            gmSrc = gmSrcPong;
        }
#if defined(__DAV_C310__)
        WaitEvent<PIPE_S>(AIC_SYNC_AIV_FLAG + pingpongEventIdWS);
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        WaitEvent(AIC_SYNC_AIV_FLAG + pingpongEventIdWS);
#endif
        SyncAll();
        PipeBarrier<PIPE_ALL>();
        if (vIndex * coreSize + mIndex * singleSize >= outSize) {
            if (mIndex < mCnt - 1) {
                NotifyEvent<PIPE_MTE3>(AIV_SYNC_AIC_FLAG + pingpongEventIdWS);
                PipeBarrier<PIPE_ALL>();
            }
            continue;
        }
        uint64_t actualSize = min(coreSize, outSize - (vIndex * coreSize + mIndex * singleSize));
        uint64_t repeat = actualSize / dataSize;
        uint64_t tail = actualSize % dataSize;

        // initialize flag, in order to match the relationship in the vector core loop
        auto eventMTE3toMTE2Zero = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
        SetFlag<HardEvent::MTE3_MTE2>(eventMTE3toMTE2Zero);
        auto eventMTE3toMTE2One = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
        SetFlag<HardEvent::MTE3_MTE2>(eventMTE3toMTE2One);

        for (uint64_t i = 0; i < repeat; ++i) {
            if ((i & CONTROL_DB) == 0) {
                pingpongEventId = eventMTE3toMTE2Zero;
                ubDst = ubDstPing;
                ubSrc1 = ubSrcPing1;
                ubSrc2 = ubSrcPing2;
            } else {
                pingpongEventId = eventMTE3toMTE2One;
                ubDst = ubDstPong;
                ubSrc1 = ubSrcPong1;
                ubSrc2 = ubSrcPong2;
            }
            WaitFlag<HardEvent::MTE3_MTE2>(pingpongEventId);
            SplitKVectorProcess<T>(ubSrc1, ubSrc2, gmSrc, ubDst, gmDst, vIndex, mIndex, i, dataSize, dataSize, coreSize,
                                   singleSize, totalSize);
            SetFlag<HardEvent::MTE3_MTE2>(pingpongEventId);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3toMTE2Zero);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3toMTE2One);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventMTE3toMTE2Zero);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventMTE3toMTE2One);
        if ((repeat & CONTROL_DB) == 0) {
            ubDst = ubDstPing;
            ubSrc1 = ubSrcPing1;
            ubSrc2 = ubSrcPing2;
        } else {
            ubDst = ubDstPong;
            ubSrc1 = ubSrcPong1;
            ubSrc2 = ubSrcPong2;
        }
        if (tail > 0) {
            SplitKVectorProcess<T>(ubSrc1, ubSrc2, gmSrc, ubDst, gmDst, vIndex, mIndex, repeat, tail, dataSize,
                                   coreSize, singleSize, totalSize);
        }
        if (mIndex < mCnt - 1) {
            NotifyEvent<PIPE_MTE3>(AIV_SYNC_AIC_FLAG + pingpongEventIdWS);
            PipeBarrier<PIPE_ALL>();
        }
    }
    return;
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatMulMultiCoreSplitKDivide(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR mmOffsetGM,
                                                   uint64_t singleSize, bool isTransposeA, bool isTransposeB,
                                                   bool isHf32, TPipe *que, const TCubeTiling& tiling, bool isBias)
{
    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BIAS_T = typename BIAS_TYPE::T;

    SetAtomicNone();
    if (GetBlockIdx() >= tiling.usedCoreNum) {
        return;
    }
    GlobalTensor<A_T> aGlobal;
    GlobalTensor<B_T> bGlobal;
    GlobalTensor<C_T> cGlobalPing;
    GlobalTensor<C_T> cGlobalPong;
    GlobalTensor<C_T> cGlobal = cGlobalPing;
    GlobalTensor<BIAS_T> biasGlobal;

    uint64_t c0Size = 8;
    GetSizeC0<A_T>(c0Size);
    uint64_t alignedOriM = MMV3CeilAlign(tiling.M, ALIGNED_H);
    uint64_t alignedOriN = MMV3CeilAlign(tiling.N, c0Size);
    uint64_t alignedKaSize = MMV3CeilAlign(tiling.Ka, c0Size);
    uint64_t alignedKbSize = MMV3CeilAlign(tiling.Kb, ALIGNED_H);

    // A B矩阵都是对齐矩阵
    if (isTransposeA) {
        alignedOriM = MMV3CeilAlign(tiling.M, c0Size);
        alignedKaSize = MMV3CeilAlign(tiling.Ka, ALIGNED_H);
    }
    if (isTransposeB) {
        alignedOriN = MMV3CeilAlign(tiling.N, ALIGNED_H);
        alignedKbSize = MMV3CeilAlign(tiling.Kb, c0Size);
    }

    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM), tiling.Kb * tiling.N);
    cGlobalPing.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(mmOffsetGM), singleSize * NUM_TWO);
    cGlobalPong = cGlobalPing[singleSize];
    if (isBias) {
        biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BIAS_T *>(biasGM), tiling.N);
    }

    MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG_PRELOAD> mm;
    mm.SetSubBlockIdx(0);
    PRELOAD(4); // preload commands
    mm.Init(&tiling, que);
    uint64_t offsetA = 0;
    uint64_t offsetB = 0;
    uint64_t offsetC = 0;

    if constexpr (A_TYPE::format == CubeFormat::NZ && B_TYPE::format == CubeFormat::NZ) {
        mm.SetOrgShape(alignedOriM, alignedOriN, alignedKaSize, alignedKbSize, tiling.N);
    } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
        mm.SetOrgShape(alignedOriM, tiling.N, alignedKaSize, tiling.Kb, tiling.N);
    } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
        mm.SetOrgShape(tiling.M, alignedOriN, tiling.Ka, alignedKbSize, tiling.N);
    } else {
        mm.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb, tiling.N);
    }

    uint64_t mCnt = MMV3DivCeil(tiling.M, tiling.singleCoreM);
    uint64_t kCnt = MMV3DivCeil(tiling.Ka, tiling.singleCoreK);
    uint64_t mCoreTail = tiling.M - (mCnt - 1) * tiling.singleCoreM;
    uint64_t kCoreTail = tiling.Ka - (kCnt - 1) * tiling.singleCoreK;
    uint64_t preCoreNum = kCnt % tiling.usedCoreNum;
    if (preCoreNum == 0) {
        preCoreNum = tiling.usedCoreNum;
    }
    uint64_t round = MMV3DivCeil(kCnt, tiling.usedCoreNum);
    uint64_t index = block_idx * round;
    uint64_t realRound = round;
    if (block_idx >= preCoreNum) {
        index = block_idx * (round - 1) + preCoreNum;
        realRound = round - 1;
    }
    uint64_t mCoreUse = 0;
    uint64_t nCoreUse = 0;
    uint64_t kCoreUse = 0;
    uint64_t nTileOffset = 0;
    uint64_t mOffset = 0;
    uint64_t kOffset = 0;
    uint8_t pingpongEventId = 0;
    nCoreUse = tiling.singleCoreN;
    mm.SetHF32(false, 0);
    if (isHf32) {
        mm.SetHF32(true, 1);
    }
    for (uint64_t mIndex = 0; mIndex < mCnt; ++mIndex) {
        if ((mIndex & CONTROL_DB) == 0) {
            pingpongEventId = 0;
            cGlobal = cGlobalPing;
        } else {
            pingpongEventId = 1;
            cGlobal = cGlobalPong;
        }
        if (mIndex > 1) {
#if defined(__DAV_C310__)
            WaitEvent<PIPE_S>(AIV_SYNC_AIC_FLAG + pingpongEventId);
            WaitEvent<PIPE_S>(AIV_SYNC_AIC_FLAG + pingpongEventId + FLAG_ID_MAX);
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
            WaitEvent(AIV_SYNC_AIC_FLAG + pingpongEventId);
#endif
            SyncAll();
            PipeBarrier<PIPE_ALL>();
        }
        mOffset = mIndex * tiling.singleCoreM;
        mCoreUse = tiling.singleCoreM;
        if (mIndex == (mCnt - 1)) {
            mCoreUse = mCoreTail;
        }

        for (uint64_t kIndex = index; kIndex < (index + realRound); ++kIndex) {
            kOffset = kIndex * tiling.singleCoreK;
            kCoreUse = tiling.singleCoreK;
            if (kIndex == (kCnt - 1)) {
                kCoreUse = kCoreTail;
            }
            mm.SetSingleShape(mCoreUse, nCoreUse, kCoreUse);
            if constexpr (A_TYPE::format == CubeFormat::ND) {
                offsetA = isTransposeA ? (mOffset + kOffset * tiling.M) : (mOffset * tiling.Ka + kOffset);
            } else {
                offsetA = isTransposeA ? (mOffset * alignedKaSize + kOffset * c0Size) : (mOffset * c0Size + kOffset *
                                                                                         alignedOriM);
            }
            if constexpr (B_TYPE::format == CubeFormat::ND) {
                offsetB = isTransposeB ? kOffset : (kOffset * tiling.N);
            } else {
                offsetB = isTransposeB ? (kOffset * alignedOriN) : (kOffset * c0Size);
            }
            offsetC = 0; // mmOffsetGM已做偏移，因此输出offsetC无需另作偏移
            mm.SetTensorA(aGlobal[offsetA], isTransposeA);
            mm.SetTensorB(bGlobal[offsetB], isTransposeB);
            // set bias
            if (isBias && kIndex == 0) {
                mm.SetBias(biasGlobal[0]); // set bias at the first k loop
            } else {
                mm.ClearBias(); // clear bias tag in the following loop
            }
            mm.IterateAll(cGlobal[offsetC], kIndex != index);
        }
#if defined(__DAV_C310__)
        NotifyEvent<PIPE_FIX>(AIC_SYNC_AIV_FLAG + pingpongEventId + FLAG_ID_MAX);
        NotifyEvent<PIPE_FIX>(AIC_SYNC_AIV_FLAG + pingpongEventId);
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        NotifyEvent<PIPE_FIX>(AIC_SYNC_AIV_FLAG + pingpongEventId);
#endif
        PipeBarrier<PIPE_ALL>();
    }
    PipeBarrier<PIPE_ALL>();
    SetAtomicNone();
    mm.SetHF32(false, 0);
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatMulKernelDeterministicSplitK(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM,
                                                       const MatmulTilingData& matmulTilingData, GM_ADDR workspaceGM)
{
    const TCubeTiling& tiling = matmulTilingData.matmulTiling;
    TPipe que;

    uint64_t singleSize = static_cast<uint64_t>(tiling.singleCoreM) * static_cast<uint64_t>(tiling.N);
    GM_ADDR mmGM = workspaceGM;

    if ASCEND_IS_AIV {
        // step3: k reduce and cast in UB
        if (GetBlockIdx() >= (tiling.usedCoreNum * NUM_TWO)) {
            return;
        }
        uint64_t totalSize = singleSize * static_cast<uint64_t>(tiling.usedCoreNum);
        uint64_t outSize = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.N);
        uint64_t coreSize = MMV3DivCeil(singleSize, static_cast<uint64_t>(tiling.usedCoreNum) * NUM_TWO);
        uint64_t mCnt = MMV3DivCeil(tiling.M, tiling.singleCoreM);
        TBuf<TPosition::VECCALC> tmpBuf;
        que.InitBuffer(tmpBuf, TOTAL_UB_SIZE);
        ReduceKInUb<C_TYPE>(cGM, mmGM, coreSize, singleSize, totalSize, outSize, mCnt, tmpBuf);
        PipeBarrier<PIPE_ALL>();
        return;
    }

    if ASCEND_IS_AIC {
        if (GetBlockIdx() >= tiling.usedCoreNum) {
#if defined(__DAV_C310__)
            NotifyEvent<PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
            NotifyEvent<PIPE_FIX>(AIC_SYNC_AIV_FLAG);
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
            NotifyEvent<PIPE_FIX>(AIC_SYNC_AIV_FLAG);
#endif
            PipeBarrier<PIPE_ALL>();
            return;
        }
        GM_ADDR mmOffsetGM = reinterpret_cast<GM_ADDR>(mmGM + GetBlockIdx() * singleSize * NUM_TWO * sizeof(float));
        using cType = MatmulType<C_TYPE::pos, C_TYPE::format, float, C_TYPE::isTrans>;
        MatMulMultiCoreSplitKDivide<A_TYPE, B_TYPE, cType, BIAS_TYPE>(aGM, bGM, biasGM, mmOffsetGM, singleSize,
                                                                      matmulTilingData.matmulRunInfo.transA,
                                                                      matmulTilingData.matmulRunInfo.transB,
                                                                      matmulTilingData.matmulRunInfo.isHf32,
                                                                      &que, tiling, tiling.isBias);
        return;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatMulUnAlignedKernelDeterministicSplitK(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM,
                                                                const MatmulTilingData& matmulTilingData,
                                                                GM_ADDR workspaceGM)
{
    const TCubeTiling& tiling = matmulTilingData.matmulTiling;
    using A_T = typename A_TYPE::T;
    uint64_t c0Size = 8; // initial c0size as fp32's c0size
    GetSizeC0<A_T>(c0Size);
    uint64_t alignedOriM = MMV3CeilAlign(tiling.M, ALIGNED_H);
    uint64_t alignedOriN = MMV3CeilAlign(tiling.N, c0Size);
    uint64_t alignedKaSize = MMV3CeilAlign(tiling.Ka, c0Size);
    uint64_t alignedKbSize = MMV3CeilAlign(tiling.Kb, ALIGNED_H);

    TPipe que;

    uint64_t singleSize = static_cast<uint64_t>(tiling.singleCoreM) * static_cast<uint64_t>(tiling.N);
    GM_ADDR mmGM = workspaceGM;
    GM_ADDR mmOffsetGM = reinterpret_cast<GM_ADDR>(mmGM + GetBlockIdx() * singleSize * NUM_TWO * sizeof(float));

    if (matmulTilingData.matmulRunInfo.transA) {
        alignedOriM = MMV3CeilAlign(tiling.M, c0Size);
        alignedKaSize = MMV3CeilAlign(tiling.Ka, ALIGNED_H);
    }
    if (matmulTilingData.matmulRunInfo.transB) {
        alignedOriN = MMV3CeilAlign(tiling.N, ALIGNED_H);
        alignedKbSize = MMV3CeilAlign(tiling.Kb, c0Size);
    }
    GM_ADDR alignedworkspaceGM = reinterpret_cast<GM_ADDR>(mmGM +
                                 tiling.usedCoreNum * singleSize * NUM_TWO * sizeof(float)); // NUM_TWO for DB
    if ASCEND_IS_AIV {
        if (GetBlockIdx() >= (tiling.usedCoreNum * NUM_TWO)) {
            NotifyEvent<PIPE_MTE3>(ND2NZ_AIV_SYNC_AIC_FLAG);
            PipeBarrier<PIPE_ALL>();
            return;
        }
        uint64_t totalSize = singleSize * static_cast<uint64_t>(tiling.usedCoreNum);
        uint64_t outSize = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.N);
        uint64_t coreSize = MMV3DivCeil(singleSize, static_cast<uint64_t>(tiling.usedCoreNum) * NUM_TWO);
        uint64_t mCnt = MMV3DivCeil(static_cast<uint64_t>(tiling.M), static_cast<uint64_t>(tiling.singleCoreM));
        TBuf<TPosition::VECCALC> tmpBuf;
        que.InitBuffer(tmpBuf, TOTAL_UB_SIZE);

        PipeBarrier<PIPE_ALL>();
        // ND2NZ
        GM_ADDR workspaceGMInUsing = alignedworkspaceGM;
        if (matmulTilingData.matmulRunInfo.nd2nzA) {
            MatrixAtoNZV2<typename A_TYPE::T>(workspaceGMInUsing, aGM, tiling, matmulTilingData.matmulRunInfo.transA,
                                              tmpBuf, matmulTilingData.baseAN, matmulTilingData.baseAD);
            workspaceGMInUsing = reinterpret_cast<GM_ADDR>(workspaceGMInUsing +
                                                           alignedOriM * alignedKaSize * sizeof(A_T));
        }
        event_t eventMte3Mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2);
        if (matmulTilingData.matmulRunInfo.nd2nzB) {
            MatrixBtoNZV2<typename B_TYPE::T>(workspaceGMInUsing, bGM, tiling, matmulTilingData.matmulRunInfo.transB,
                                              tmpBuf, matmulTilingData.baseBN, matmulTilingData.baseBD);
        }

        SyncAll();
        NotifyEvent<PIPE_MTE3>(ND2NZ_AIV_SYNC_AIC_FLAG);
        PipeBarrier<PIPE_ALL>();

        ReduceKInUb<C_TYPE>(cGM, mmGM, coreSize, singleSize, totalSize, outSize, mCnt, tmpBuf);
        PipeBarrier<PIPE_ALL>();
        return;
    }

    if ASCEND_IS_AIC {
        WaitEvent(ND2NZ_AIV_SYNC_AIC_FLAG);
        if (GetBlockIdx() >= tiling.usedCoreNum) {
#if defined(__DAV_C310__)
            NotifyEvent<PIPE_FIX>(AIC_SYNC_AIV_FLAG + FLAG_ID_MAX);
            NotifyEvent<PIPE_FIX>(AIC_SYNC_AIV_FLAG);
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
            NotifyEvent<PIPE_FIX>(AIC_SYNC_AIV_FLAG);
#endif
            PipeBarrier<PIPE_ALL>();
            return;
        }
        using cType = MatmulType<C_TYPE::pos, C_TYPE::format, float, C_TYPE::isTrans>;
        if (matmulTilingData.matmulRunInfo.nd2nzA && !matmulTilingData.matmulRunInfo.nd2nzB) {
            using aType = MatmulType<A_TYPE::pos, CubeFormat::NZ, typename A_TYPE::T, A_TYPE::isTrans>;
            MatMulMultiCoreSplitKDivide<aType, B_TYPE, cType, BIAS_TYPE>(alignedworkspaceGM, bGM, biasGM, mmOffsetGM,
                                                                         singleSize,
                                                                         matmulTilingData.matmulRunInfo.transA,
                                                                         matmulTilingData.matmulRunInfo.transB,
                                                                         matmulTilingData.matmulRunInfo.isHf32,
                                                                         &que, tiling, tiling.isBias);
        } else if (!matmulTilingData.matmulRunInfo.nd2nzA && matmulTilingData.matmulRunInfo.nd2nzB) {
            using bType = MatmulType<B_TYPE::pos, CubeFormat::NZ, typename B_TYPE::T, B_TYPE::isTrans>;
            MatMulMultiCoreSplitKDivide<A_TYPE, bType, cType, BIAS_TYPE>(aGM, alignedworkspaceGM,
                                                                         biasGM, mmOffsetGM, singleSize,
                                                                         matmulTilingData.matmulRunInfo.transA,
                                                                         matmulTilingData.matmulRunInfo.transB,
                                                                         matmulTilingData.matmulRunInfo.isHf32,
                                                                         &que, tiling, tiling.isBias);
        } else if (matmulTilingData.matmulRunInfo.nd2nzA && matmulTilingData.matmulRunInfo.nd2nzB) {
            using aType = MatmulType<A_TYPE::pos, CubeFormat::NZ, typename A_TYPE::T, A_TYPE::isTrans>;
            using bType = MatmulType<B_TYPE::pos, CubeFormat::NZ, typename B_TYPE::T, B_TYPE::isTrans>;
            MatMulMultiCoreSplitKDivide<aType, bType, cType, BIAS_TYPE>(alignedworkspaceGM, alignedworkspaceGM +
                                                                        alignedOriM * alignedKaSize * sizeof(A_T),
                                                                        biasGM, mmOffsetGM, singleSize,
                                                                        matmulTilingData.matmulRunInfo.transA,
                                                                        matmulTilingData.matmulRunInfo.transB,
                                                                        matmulTilingData.matmulRunInfo.isHf32,
                                                                        &que, tiling, tiling.isBias);
        }
        return;
    }
}
}
#endif // __OP_KERNEL_MATMUL_STRESS_DETECT_H__
