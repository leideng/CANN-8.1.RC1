/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file incre_flash_attention_preload_mla.h
 * \brief
 */

#ifndef INCRE_FLASH_ATTENTION_PRELOAD_MLA
#define INCRE_FLASH_ATTENTION_PRELOAD_MLA

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "ifa_public_define.h"
#include "ifa_service_matmul_kv_nd.h"
#include "ifa_service_matmul_kv_nz.h"

using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

#define USE_SERVICE
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

struct TaskContext {
    uint32_t bidx;
    uint32_t gidx;
    uint32_t s1idx;
    uint32_t s2idx;
    uint32_t s2loops;
    uint32_t s2SizeTail;
    uint32_t s1Size;
    uint32_t s2Size;
    uint32_t isFirstLoop;
    static constexpr uint32_t nidx = 0;
};

template <typename IFAT> class IncreFlashAttentionAttenPreloadMla {
public:
    __aicore__ inline IncreFlashAttentionAttenPreloadMla(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *pseShift, __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths,
                                __gm__ uint8_t *blockTable, __gm__ uint8_t *kvPaddingSize,
                                __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                                __gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
                                const IncreFlashAttentionTilingDataMla *__restrict tiling, __gm__ uint8_t *gmTiling,
                                TPipe *tPipe, bool isPrefix = false);
    __aicore__ inline void InitQuant(__gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2,
                                     __gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2,
                                     __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
                                     __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset,
                                     __gm__ uint8_t *valueAntiquantScale, __gm__ uint8_t *valueAntiquantOffset,
                                     __gm__ uint8_t *workspace);

    __aicore__ inline void Process();

    // 中间计算数据类型为float，高精度模式
    using T = float;

    using Q_T = typename IFAT::queryType;
    using KV_T = typename IFAT::kvType;
    using OUT_T = typename IFAT::outputType;
    using ORIGIN_T = typename IFAT::orginalType;
    static constexpr bool PAGE_ATTENTION = IFAT::pageAttention;
    static constexpr bool FLASH_DECODE = IFAT::flashDecode;
    static constexpr LAYOUT LAYOUT_T = IFAT::layout;
    static constexpr LAYOUT KV_LAYOUT_T = IFAT::kvLayout;

    using MM_OUT_T = T;

protected:
    const IncreFlashAttentionTilingDataMla *__restrict tilingData = nullptr;
    TPipe *pipe = nullptr;

    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<Q_T>  qRopeGm;
    GlobalTensor<KV_T> kRopeGm;

    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<int32_t> blockTableGm;

    // atten mask
    GlobalTensor<bool> attenMaskBoolGm;
    GlobalTensor<uint64_t> actualSeqLengthsGm;

    // workspace
    GlobalTensor<MM_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<MM_OUT_T> mm2ResGm;
    GlobalTensor<T> vec2ResGm;
    GlobalTensor<T> accumOutGm; // no
    GlobalTensor<T> lseSumFdGm; // no
    GlobalTensor<T> lseMaxFdGm; // no

    // queue
    TQue<QuePosition::VECIN, 1> inputQue1;   // 32K, inque
    TQue<QuePosition::VECIN, 1> inputQue2;   // 16K, inque
    TQue<QuePosition::VECOUT, 1> outputQue1; // 32K, outque
    TQue<QuePosition::VECOUT, 1> outputQue2; // 8K, outque

    // 临时tbuf
    TBuf<> tmpBuff1; // 32K
    TBuf<> tmpBuff2; // 32K
    TBuf<> softmaxMaxBuff; // PRE_LOAD_NUM_MLA * 2K
    TBuf<> softmaxExpBuff; // PRE_LOAD_NUM_MLA * 2K
    TBuf<> softmaxSumBuff; // PRE_LOAD_NUM_MLA * 2K

    TBuf<> softmaxMaxDefaultBuff;     // 2K
    TBuf<> softmaxSumDefaultBuff;     // 2K

    LocalTensor<T> softmaxMaxUb;
    LocalTensor<T> softmaxSumUb;
    LocalTensor<T> softmaxExpUb;

    LocalTensor<T> softmaxMaxDefaultUb;
    LocalTensor<T> softmaxSumDefaultUb;

    static constexpr uint64_t SYNC_MODE2 = 2;
    static constexpr uint64_t SYNC_V0_C1_FLAG = 6;
    static constexpr uint64_t SYNC_C1_V1_FLAG = 7;
    static constexpr uint64_t SYNC_V1_C2_FLAG = 8;
    static constexpr uint64_t SYNC_C2_V2_FLAG = 9;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T);
    static constexpr uint32_t REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(T);
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = BUFFER_SIZE_BYTE_32K / sizeof(T);
    static constexpr uint32_t ADDRESS_ALIGN_NUM = 512 / sizeof(KV_T);
    static constexpr uint32_t ADDRESS_ALIGN_NUM_THRESHLOD = 128 / sizeof(KV_T);
    static constexpr T SOFTMAX_MIN_NUM = -2e38;
    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    static constexpr uint64_t kvHeadNum = 1ULL;
    static constexpr uint64_t headDim = 512ULL;
    static constexpr uint64_t headDimAlign = 512ULL;
    static constexpr uint64_t headDimRope = 64ULL;
    static constexpr bool batchContinuous = true;
    static constexpr uint32_t n2Idx = 0U;

    // for workspace pingpong
    const uint32_t dbWorkspaceRatio = PRE_LOAD_NUM_MLA;

    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    __gm__ uint8_t *key_ = nullptr;
    __gm__ uint8_t *value_ = nullptr;

    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;

    // tilingdata
    uint64_t singleProcessSInnerSize = 0U;
    uint32_t sInnerLoopTimes = 0U;
    uint64_t singleProcessSInnerSizeTail = 0U;
    uint32_t usedCoreNum = 0U;
    uint32_t bIdx = 0U;
    uint32_t s1Idx = 0U;  // for flash-decode

    uint32_t mmResUbSize = 0U;
    uint32_t bmm2ResUbSize = 0U;

    uint64_t batchSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t gSize = 0ULL;

    uint64_t actS1Size = 1ULL;
    uint64_t s1SplitSize = 0ULL;
    uint64_t gSizeSub = 0ULL;
    uint64_t gSizeTail = 0ULL;
    uint64_t s1SizeSub = 0ULL;
    uint64_t s1SizeTail = 0ULL;
    uint64_t gOuter = 1ULL;
    uint64_t s1Outer = 1ULL;
    uint64_t gIdx = 0ULL;

    uint64_t mSizeVector = 0ULL;
    uint64_t mSizeVStart = 0ULL;
    uint64_t kvSeqSize = 0ULL;
    uint64_t qSeqSize = 1ULL;

    // pageAttention
    uint32_t kvCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint64_t s2BatchBaseOffset = 0;

    // attention mask
    bool attenMaskFlag = false;
    uint32_t attenMaskSizeAlign = 0U;

    // offset
    uint64_t tensorACoreOffset = 0ULL;
    uint64_t tensorBCoreOffset = 0ULL;
    uint64_t tensorARopeCoreOffset = 0ULL;
    uint64_t tensorBRopeCoreOffset = 0ULL;
    uint64_t tensorBOffset = 0ULL;
    uint64_t attenOutOffset = 0ULL;
    uint64_t attenMaskOffset = 0ULL;
    uint64_t attenMaskCoreOffset = 0ULL;
    uint64_t attenMaskSize = 0ULL;

    // splitKV
    uint32_t splitKVNum = 0U;
    uint32_t s2Idx = 0U;
    uint32_t s2IdxFD = 0U;
    uint64_t sInnerLoopSize = 0ULL;
    uint32_t actualCombineLoopSize = 0U;
    uint64_t combineLseOffset = 0ULL;
    uint64_t combineAccumOutOffset = 0ULL;

    uint64_t curActualSeqLen = 0ULL;
    uint64_t actualSingleProcessSInnerSize = 0ULL;
    uint64_t actualSingleProcessSInnerSizeAlign = 0ULL;
    uint32_t beforeBlockSplitBn2Nums = 0U;
    uint32_t bn2LoopTimes = 0U;

    uint32_t actualLenDims = 0U;
    uint32_t gMax = 128U;
    // 记录当前轮的bIdx nIdx s2Idx actualLen
    uint32_t bn2IdxInCurCore = 0;
    using MatmulServiceNz = IfaMatmulKvNz<IFAT>;
    using MatmulServiceNd = IfaMatmulKvNd<IFAT>;
    using MatmulServiceType = typename AscendC::Conditional<(KV_LAYOUT_T == LAYOUT::NZ), MatmulServiceNz, MatmulServiceNd>::type;
    MatmulServiceType matmulService;

    __aicore__ inline void InitValueGm(uint32_t bIdx);
    __aicore__ inline void InitKeyGm(uint32_t bIdx);
    __aicore__ inline void CalcParams(uint32_t loop, ExtraInfoMla &info, TaskContext &task);

    __aicore__ inline void ComputeMm1(const ExtraInfoMla &info);
    __aicore__ inline void ProcessVec1L(const ExtraInfoMla &info);
    __aicore__ inline void ComputeMm2(const ExtraInfoMla &info);
    __aicore__ inline void ProcessVec2L(const ExtraInfoMla &info);

    bool curActSeqLenIsZero = false;

    template <typename T> __aicore__ inline T Align(T num, T rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
    }

    template <typename T> __aicore__ inline size_t BlockAlign(size_t s)
    {
        if constexpr (IsSameType<T, int4b_t>::value) {
            return (s + 63) / 64 * 64;
        }
        size_t n = (32 / sizeof(KV_T));
        return (s + n - 1) / n * n;
    }

    __aicore__ inline void InitTilingData();
    __aicore__ inline void InitCalcParamsEach();
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengths);
    __aicore__ inline void GetActualSeqLen(uint32_t bIdx, uint32_t s1Idx = 0);
    __aicore__ inline void UpdateInnerLoopCond();
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx);

    __aicore__ inline void GetBN2Gid(const uint32_t bn2gIdx);

    __aicore__ inline void AttenMaskCopyIn(uint64_t offset, uint32_t dealRowCount, uint32_t actualColumnCount);
    __aicore__ inline void AttenMaskCopyIn(const ExtraInfoMla& info);

    __aicore__ inline void FlashDecodeCompute();

    __aicore__ inline void DealBmm1ResBaseBlock(const ExtraInfoMla &info, uint32_t startRow, uint32_t dealRowCount,
                                                uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void ProcessVec1Inner(const ExtraInfoMla &info);

    __aicore__ inline void DealBmm2ResBaseBlock(const ExtraInfoMla &info, uint32_t startRow, uint32_t dealRowCount,
                                                uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void ProcessVec2Inner(const ExtraInfoMla &info);

    __aicore__ inline void SoftmaxFlashV2Compute(const ExtraInfoMla &info, LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
                                                 uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                 uint32_t actualColumnCount);
    __aicore__ inline void ElewiseCompute(const ExtraInfoMla &info, LocalTensor<T> &mmResUb, TBuf<> &tmpBuf, uint32_t startRow,
                                          uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2DataCopyOutFD(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2DataCopyOutTrans(const ExtraInfoMla& info, LocalTensor<OUT_T> &attenOutUb, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2ResCopyOut(const ExtraInfoMla &info, LocalTensor<T> &bmm2ResUb, uint32_t startRow, uint32_t dealRowCount,
                                           uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2CastAndCopyOut(const ExtraInfoMla &info, LocalTensor<T> &bmm2ResUb, uint32_t startRow, uint32_t dealRowCount,
                                              uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void DealInvalidRows(const ExtraInfoMla &info, LocalTensor<OUT_T> &attenOutUb, uint32_t startRow,
                                           uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void CombineSplitKVRes();
    __aicore__ inline void CopyAccumOutIn(uint32_t splitKVIndex, uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void CopyLseIn(uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const ExtraInfoMla &info, LocalTensor<T> &softmaxMaxUb, LocalTensor<T> &softmaxSumUb);
    __aicore__ inline void Bmm2FDDataCopyOut(const ExtraInfoMla &info, LocalTensor<T> &bmm2ResUb, uint32_t startRow, uint32_t dealRowCount,
                                             uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void ComputeScaleValue(LocalTensor<T> &lseSum, LocalTensor<T> &lseMax, uint32_t startRow,
                                             uint32_t dealRowCount);
    __aicore__ inline void ReduceFinalRes(LocalTensor<T> &dst, LocalTensor<T> &lseLocal, uint32_t startRow,
                                          uint32_t dealRowCount);
    __aicore__ inline void CopyFinalResOut(LocalTensor<T> &accumOutLocal, uint32_t startRow, uint32_t dealRowCount);

    __aicore__ inline void InitAllZeroOutput(uint32_t bIdx, uint32_t n2Idx);
    __aicore__ inline uint64_t SeqLenFromTensorList(uint32_t bIdx);

    __aicore__ inline void CopyFixedUbToGm(const GlobalTensor<T> &dst, const LocalTensor<T> &src, size_t size);
};

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::InitTilingData()
{
    singleProcessSInnerSize = tilingData->increFlashAttentionSingleCoreParams.singleProcessSInnerSize;
    usedCoreNum = tilingData->increFlashAttentionSingleCoreParams.usedCoreNum;
    splitKVNum = tilingData->splitKVParams.s2;
    sInnerLoopSize = tilingData->splitKVParams.sInnerLoopSize;

    mmResUbSize = tilingData->increFlashAttentionSingleCoreTensorSize.mmResUbSize;
    bmm2ResUbSize = tilingData->increFlashAttentionSingleCoreTensorSize.bmm2ResUbSize;

    batchSize = tilingData->baseParams.batchSize;
    qHeadNum = gSize = tilingData->baseParams.nNumOfQInOneGroup;

    gSizeSub = tilingData->increFlashAttentionSingleCoreParams.groupSplitSize; // 切块大小Gi
    gOuter = (gSize + gSizeSub - 1) / gSizeSub;
    gSizeTail = gSize - (gOuter - 1) * gSizeSub;

    kvSeqSize = tilingData->baseParams.seqSize;
    qSeqSize = tilingData->baseParams.qSeqSize;

    s1SizeSub = tilingData->increFlashAttentionSingleCoreParams.s1SplitSize; // 切块大小Si
    s1Outer = (qSeqSize + s1SizeSub - 1) / s1SizeSub;
    s1SizeTail = qSeqSize - (s1Outer - 1) * s1SizeSub;

    attenMaskFlag = (tilingData->baseParams.attenMaskFlag != 0) ? true : false;
    attenMaskSize = tilingData->baseParams.attenMaskSize;

    maxBlockNumPerBatch = tilingData->baseParams.maxBlockNumPerBatch;
    kvCacheBlockSize = tilingData->baseParams.blockSize;
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        // queue
        pipe->InitBuffer(inputQue1, 1, BUFFER_SIZE_BYTE_32K);
        pipe->InitBuffer(inputQue2, 1, BUFFER_SIZE_BYTE_16K);
        pipe->InitBuffer(outputQue1, 1, BUFFER_SIZE_BYTE_32K);
        pipe->InitBuffer(outputQue2, 1, BUFFER_SIZE_BYTE_8K);

        // tmpBuff
        pipe->InitBuffer(tmpBuff1, BUFFER_SIZE_BYTE_32K);
        pipe->InitBuffer(tmpBuff2, BUFFER_SIZE_BYTE_32K);

#ifdef IFA_SOFTMAX_WITHOUT_BRC
        // M_MAX=512, 512 * sizeof(T) * N_Buffer
        pipe->InitBuffer(softmaxMaxBuff, BUFFER_SIZE_BYTE_2K * PRE_LOAD_NUM_MLA);
        pipe->InitBuffer(softmaxExpBuff, BUFFER_SIZE_BYTE_2K * PRE_LOAD_NUM_MLA);
        pipe->InitBuffer(softmaxSumBuff, BUFFER_SIZE_BYTE_2K * PRE_LOAD_NUM_MLA);

        pipe->InitBuffer(softmaxMaxDefaultBuff, BUFFER_SIZE_BYTE_2K);
        pipe->InitBuffer(softmaxSumDefaultBuff, BUFFER_SIZE_BYTE_2K);
#else
        pipe->InitBuffer(softmaxMaxBuff, BUFFER_SIZE_BYTE_2K * PRE_LOAD_NUM_MLA);
        pipe->InitBuffer(softmaxExpBuff, BUFFER_SIZE_BYTE_2K * PRE_LOAD_NUM_MLA);
        pipe->InitBuffer(softmaxSumBuff, BUFFER_SIZE_BYTE_2K * PRE_LOAD_NUM_MLA);

        pipe->InitBuffer(softmaxMaxDefaultBuff, BUFFER_SIZE_BYTE_2K);
        pipe->InitBuffer(softmaxSumDefaultBuff, BUFFER_SIZE_BYTE_2K);
#endif
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengths)
{
    actualLenDims = tilingData->baseParams.actualLenDims;
    if (actualLenDims != 0) {
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengths, actualLenDims);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::InitAllZeroOutput(uint32_t bIdx, uint32_t n2Idx)
{
    uint64_t copySize = qSeqSize * gSize * headDim;
    uint64_t outputOffset = 0;
    if constexpr (LAYOUT_T == LAYOUT::BNSD) {
        outputOffset = bIdx * qHeadNum * qSeqSize * headDim + n2Idx * gSize * qSeqSize * headDim +
                       gIdx * gSizeSub * qSeqSize * headDim;
    } else {
        outputOffset = bIdx * qSeqSize * qHeadNum * headDim + s1Idx * s1SizeSub * qHeadNum * headDim +
                       n2Idx * gSize * headDim;
    }
    matmul::InitOutput<OUT_T>(attentionOutGm[outputOffset], copySize, 0);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::GetActualSeqLen(uint32_t bIdx, uint32_t s1Idx)
{
    if (actualLenDims == 0) {
        curActualSeqLen = kvSeqSize;
        if (!batchContinuous) {
            curActualSeqLen = SeqLenFromTensorList(bIdx);
        }
    } else if (actualLenDims == 1) {
        curActualSeqLen = actualSeqLengthsGm.GetValue(0);
    } else {
        curActualSeqLen = actualSeqLengthsGm.GetValue(bIdx);
    }

    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
        actS1Size = (s1Idx == s1Outer - 1) ? s1SizeTail : s1SizeSub;
    } else {
        actS1Size = qSeqSize;
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::GetBN2Gid(const uint32_t bn2gIdx)
{
    if constexpr(FLASH_DECODE) { // TODO
        bIdx = aiCoreIdx / (kvHeadNum * splitKVNum);
        // n2Idx = (aiCoreIdx / splitKVNum) % kvHeadNum;
        s2IdxFD = aiCoreIdx % splitKVNum;
    } else if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
        uint32_t bs1n2 = beforeBlockSplitBn2Nums + bn2gIdx;
        uint32_t s1n2 = bs1n2 % (kvHeadNum * s1Outer);
        bIdx = bs1n2 / (kvHeadNum * s1Outer);
        s1Idx = s1n2 / kvHeadNum;
    } else {
       uint32_t bn2g = beforeBlockSplitBn2Nums + bn2gIdx;
       uint32_t n2g = bn2g % (kvHeadNum * gOuter);
       bIdx = bn2g / (kvHeadNum * gOuter);
       gIdx = n2g % gOuter;
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx)
{
    if ASCEND_IS_AIV {
        InitAllZeroOutput(bIdx, n2Idx);
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::UpdateInnerLoopCond()
{
    if ((curActualSeqLen == 0) || (actS1Size == 0)) {
        curActSeqLenIsZero = true;
        return;
    }
    curActSeqLenIsZero = false;

    int32_t remainSinnerSize = (int32_t)curActualSeqLen;
    int32_t computeSinnerSize = (int32_t)curActualSeqLen;
    if constexpr (FLASH_DECODE) {
        remainSinnerSize = (int32_t)curActualSeqLen - sInnerLoopSize * s2IdxFD;
        computeSinnerSize = remainSinnerSize >= sInnerLoopSize ? sInnerLoopSize : remainSinnerSize;
        if (aiCoreIdx >= batchSize * kvHeadNum * splitKVNum) {
            remainSinnerSize = 0;
        }
    }
    if (remainSinnerSize > 0) {
        if (computeSinnerSize <= singleProcessSInnerSize) {
            singleProcessSInnerSizeTail = computeSinnerSize;
            sInnerLoopTimes = 1;
        } else {
            sInnerLoopTimes = (computeSinnerSize + singleProcessSInnerSize - 1) / singleProcessSInnerSize;
            singleProcessSInnerSizeTail = computeSinnerSize - (sInnerLoopTimes - 1) * singleProcessSInnerSize;
        }
    } else {
        sInnerLoopTimes = 0;
    }
}

template <typename IFAT>
__aicore__ inline uint64_t IncreFlashAttentionAttenPreloadMla<IFAT>::SeqLenFromTensorList(uint32_t bIndex)
{
    uint64_t dimInfo[4]; // this mem is used to set shapeinfo, BSH(3) or BNSD(4)
    AscendC::TensorDesc<__gm__ uint8_t> keyTensorDesc;
    ListTensorDesc keyListTensorDesc((__gm__ void *)keyPtr);
    keyTensorDesc.SetShapeAddr(&dimInfo[0]);
    keyListTensorDesc.GetDesc(keyTensorDesc, bIndex);
    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
        return keyTensorDesc.GetShape(1); // BSH, idx of s is 1
    } else {
        return keyTensorDesc.GetShape(2); // BNSD, idx of s is 2
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *kvPaddingSize,
    __gm__ uint8_t *queryRope,
    __gm__ uint8_t *keyRope,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
    const IncreFlashAttentionTilingDataMla *__restrict tiling, __gm__ uint8_t *gmTiling, TPipe *tPipe, bool isPrefix)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    // init tiling data
    tilingData = tiling;

    InitTilingData();
    // 初始化计算参数
    InitCalcParamsEach();

    pipe = tPipe;
    keyPtr = key;
    valuePtr = value;

    actualSingleProcessSInnerSize = 0ULL;
    actualSingleProcessSInnerSizeAlign = 0ULL;

    // init global buffer
    queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
    qRopeGm.SetGlobalBuffer((__gm__ Q_T *)queryRope);
    kRopeGm.SetGlobalBuffer((__gm__ KV_T *)keyRope);

    attentionOutGm.SetGlobalBuffer((__gm__ OUT_T *)attentionOut);
    // batch连续时,只需要初始化一次;不连续时,需要在使用时根据batchIdx初始化
    if (batchContinuous) {
        InitKeyGm(0);
        InitValueGm(0);
    }

    if (pipe != nullptr) {
        InitBuffers();
    }

    if (attenMaskFlag) {
        attenMaskBoolGm.SetGlobalBuffer((__gm__ bool *)attenMask);
    }

    InitActualSeqLen(actualSeqLengths);

    if constexpr (PAGE_ATTENTION) {
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    }

    if ASCEND_IS_AIV {
        softmaxMaxUb = softmaxMaxBuff.Get<T>();
        softmaxSumUb = softmaxSumBuff.Get<T>();
        softmaxExpUb = softmaxExpBuff.Get<T>();

        softmaxMaxDefaultUb = softmaxMaxDefaultBuff.Get<T>();
        softmaxSumDefaultUb = softmaxSumDefaultBuff.Get<T>();
    }

    uint64_t offset = 0;

    mm1ResGm.SetGlobalBuffer(
            (__gm__ MM_OUT_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * mmResUbSize * sizeof(MM_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * mmResUbSize * sizeof(MM_OUT_T);

    vec1ResGm.SetGlobalBuffer(
        (__gm__ KV_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * mmResUbSize * sizeof(KV_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * mmResUbSize * sizeof(KV_T);

    mm2ResGm.SetGlobalBuffer(
            (__gm__ MM_OUT_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * bmm2ResUbSize * sizeof(MM_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * bmm2ResUbSize * sizeof(MM_OUT_T);

    vec2ResGm.SetGlobalBuffer(
            (__gm__ T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio* bmm2ResUbSize * sizeof(T)));
    offset += GetBlockNum() * dbWorkspaceRatio * bmm2ResUbSize * sizeof(T);

    if constexpr (FLASH_DECODE) {
        accumOutGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
        offset = offset + tilingData->splitKVParams.accumOutSize * sizeof(float);
        lseSumFdGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
        lseMaxFdGm.SetGlobalBuffer((__gm__ float *)(workspace + offset) + tilingData->splitKVParams.logSumExpSize / 2);
        offset = offset + tilingData->splitKVParams.logSumExpSize * sizeof(float);
    }

    if ASCEND_IS_AIC {
        matmulService.InitParams(qHeadNum, kvHeadNum, headDim, headDimRope, qSeqSize, mmResUbSize, bmm2ResUbSize);
        matmulService.InitMm1GlobalTensor(queryGm, qRopeGm, keyGm, kRopeGm, mm1ResGm);
        matmulService.InitMm2GlobalTensor(vec1ResGm, valueGm, mm2ResGm, attentionOutGm);
        matmulService.InitPageAttentionInfo(blockTableGm, kvCacheBlockSize, maxBlockNumPerBatch);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::InitKeyGm(uint32_t bIdx)
{
    ListTensorDesc keyListTensorDesc((__gm__ void *)keyPtr);
    key_ = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);

    keyGm.SetGlobalBuffer((__gm__ KV_T *)key_);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::InitValueGm(uint32_t bIdx)
{
    ListTensorDesc valueListTensorDesc((__gm__ void *)valuePtr);
    value_ = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);

    valueGm.SetGlobalBuffer((__gm__ KV_T *)value_);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::InitQuant(
    __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
    __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
    __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale,
    __gm__ uint8_t *valueAntiquantOffset, __gm__ uint8_t *workspace)
{

}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::InitCalcParamsEach()
{
    // 这里是编译器优化写法，定义一个局部数组变量coreSidxEnd(存在栈上)，使用copy_data_align64接口
    // 可以只从ub中拷贝tiling中coreSidxEnd的内容到栈上，而非将整个increFlashAttentionCoreParams
    // 内容拷贝到栈，减少拷贝时间。
    if constexpr (FLASH_DECODE) {
        bn2LoopTimes = 1U;
    } else {
#ifdef ASCENDC_CPU_DEBUG
        const uint32_t *coreSidxEnd = tilingData->increFlashAttentionCoreParams.coreSidxEnd;
#else
        uint32_t coreSidxEnd[ARRAY_SIZE(tilingData->increFlashAttentionCoreParams.coreSidxEnd)];
        copy_data_align64((uint8_t *)coreSidxEnd,
                            (uint8_t *)(tilingData->increFlashAttentionCoreParams.coreSidxEnd), sizeof(coreSidxEnd));
#endif
        bn2LoopTimes = coreSidxEnd[aiCoreIdx + 1] - coreSidxEnd[aiCoreIdx];
        beforeBlockSplitBn2Nums = coreSidxEnd[aiCoreIdx];
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::AttenMaskCopyIn(uint64_t offset,
                                                                                     uint32_t dealRowCount,
                                                                                     uint32_t actualColumnCount)
{
    LocalTensor<bool> maskUb = inputQue2.AllocTensor<bool>();
    attenMaskSizeAlign = Align(actualColumnCount, 32U);
    maskUb.SetSize(dealRowCount * attenMaskSizeAlign);
#if (__CCE_AICORE__ > 200)
    if (actualColumnCount % 32 == 0) {
        DataCopy(maskUb, attenMaskBoolGm[offset], attenMaskSizeAlign);
    } else {
        uint32_t typeElementSize = BYTE_BLOCK / sizeof(bool);
        DataCopyExtParams intriParams;
        intriParams.blockLen = actualColumnCount * sizeof(bool);
        intriParams.blockCount = 1;
        intriParams.dstStride = (attenMaskSizeAlign - actualColumnCount) / typeElementSize;
        intriParams.srcStride = 0;
        DataCopyPadExtParams<bool> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = (attenMaskSizeAlign - actualColumnCount) % typeElementSize;
        padParams.paddingValue = 0;
        DataCopyPad(maskUb, attenMaskBoolGm[offset], intriParams, padParams);
    }
#else
    DataCopy(maskUb, attenMaskBoolGm[offset], attenMaskSizeAlign);
#endif
    inputQue2.template EnQue(maskUb);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::AttenMaskCopyIn(const ExtraInfoMla& info)
{
    #define ATTENMASK_STRIDE  2048U
    uint32_t offset;
    int32_t delta =
        info.s1Idx * s1SizeSub - info.s2Idx * singleProcessSInnerSize + info.s2Size - qSeqSize; // s1idx = 0
    if (delta < 0) {
        offset = (-delta) < (int32_t)info.s1Size ? (-delta) : info.s1Size; // min (-delta, s1Size)
    } else  {
        offset = (delta < (int32_t)singleProcessSInnerSize ? delta : singleProcessSInnerSize) *
                ATTENMASK_STRIDE; // min(delta, s2inner)
    }

    attenMaskSizeAlign = Align(info.actualSingleProcessSInnerSize, 32U);

    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = info.s1Size;
    dataCopyParams.blockLen = attenMaskSizeAlign * sizeof(bool) /32;
    dataCopyParams.srcStride = (ATTENMASK_STRIDE - attenMaskSizeAlign) * sizeof(bool) / 32;
    dataCopyParams.dstStride = 0;

    LocalTensor<bool> maskUb = inputQue2.AllocTensor<bool>();
    DataCopy(maskUb, attenMaskBoolGm[offset], dataCopyParams);

    inputQue2.template EnQue(maskUb);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::CopyLseIn(uint32_t startRow, uint32_t dealRowCount)
{
    LocalTensor<T> lseMax = inputQue1.AllocTensor<T>();
    LocalTensor<T> lseSum = inputQue2.AllocTensor<T>();

    combineLseOffset = (bIdx * kvHeadNum * splitKVNum + n2Idx * splitKVNum) * gSize + startRow;

    for (uint32_t i = 0; i < actualCombineLoopSize; i++) {
        if ((dealRowCount % FP32_ONE_BLOCK_SIZE ) == 0) { // 32B对齐                
            DataCopy(lseSum[i * dealRowCount], lseSumFdGm[combineLseOffset + i * qSeqSize * gSize], dealRowCount);
            DataCopy(lseMax[i * dealRowCount], lseMaxFdGm[combineLseOffset + i * qSeqSize * gSize], dealRowCount);
        } else {
            DataCopyParams copyInParams;
            DataCopyPadParams copyInPadParams;
            copyInParams.blockCount = 1;
            copyInParams.blockLen = dealRowCount * BYTE_BLOCK / FP32_ONE_BLOCK_SIZE;  // 元素32位，每个连续传输快长度单位为Byte
            copyInParams.srcStride = 0;
            copyInParams.dstStride = 0;

            copyInPadParams.isPad = true;
            copyInPadParams.leftPadding = 0;
            copyInPadParams.rightPadding = FP32_ONE_BLOCK_SIZE - (dealRowCount % FP32_ONE_BLOCK_SIZE); // 补充至32字节对齐，单位是元素个数
            copyInPadParams.paddingValue = 0;
            DataCopyPad(lseSum[i * dealRowCount], lseSumFdGm[combineLseOffset + i * qSeqSize * gSize], copyInParams, copyInPadParams);
            DataCopyPad(lseMax[i * dealRowCount], lseMaxFdGm[combineLseOffset + i * qSeqSize * gSize], copyInParams, copyInPadParams);
        }
    }
    inputQue2.EnQue(lseSum);
    inputQue1.EnQue(lseMax);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::CopyAccumOutIn(uint32_t splitKVIndex,
                                                                                    uint32_t startRow,
                                                                                    uint32_t dealRowCount)
{
    LocalTensor<T> accumOutLocal = inputQue1.AllocTensor<T>();

    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<T> copyInPadParams;
    copyInParams.blockCount = dealRowCount;
    copyInParams.blockLen = headDim * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = (headDimAlign - headDim) / BLOCK_ELEMENT_NUM;

    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = (headDimAlign - headDim) % BLOCK_ELEMENT_NUM;
    copyInPadParams.paddingValue = 0;

    combineAccumOutOffset =
        (bIdx * kvHeadNum * splitKVNum + n2Idx * splitKVNum + splitKVIndex) * gSize * headDim + startRow * headDim;
    DataCopyPad(accumOutLocal, accumOutGm[combineAccumOutOffset], copyInParams, copyInPadParams);
    inputQue1.EnQue(accumOutLocal);
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::ComputeScaleValue(LocalTensor<T> &lseSum, LocalTensor<T> &lseMax,
                                                                uint32_t startRow, uint32_t dealRowCount)
{
    LocalTensor<T> lseMaxUb = softmaxMaxUb[0];
    LocalTensor<T> lseSumUb = softmaxSumUb[0];
    LocalTensor<T> lseExpUb = tmpBuff1.Get<T>();

    // lseLocal的shape为[actualCombineLoopSize, dealRowCount * FP32_ONE_BLOCK_SIZE]
    Duplicate(lseMaxUb, -FLOAT_MAX, dealRowCount);
    Duplicate(lseSumUb, FLOAT_ZERO, dealRowCount);
    pipe_barrier(PIPE_V);
    for (uint32_t i = 0; i < actualCombineLoopSize; ++i) {
        Max(lseMaxUb, lseMaxUb, lseMax[i * dealRowCount], dealRowCount);
        pipe_barrier(PIPE_V);
    }
    for (uint32_t i = 0; i < actualCombineLoopSize; ++i) {
        Sub(lseExpUb[i * dealRowCount], lseMax[i * dealRowCount], lseMaxUb,
            dealRowCount);
    }
    pipe_barrier(PIPE_V);
    Exp(lseExpUb, lseExpUb, actualCombineLoopSize * dealRowCount);
    pipe_barrier(PIPE_V);

    Mul(lseSum, lseSum, lseExpUb, actualCombineLoopSize * dealRowCount);
    pipe_barrier(PIPE_V);

    for (uint32_t i = 0; i < actualCombineLoopSize; ++i) {
        Add(lseSumUb, lseSumUb, lseSum[i * dealRowCount], dealRowCount);
        pipe_barrier(PIPE_V);
    }

    for (uint32_t i = 0; i < actualCombineLoopSize; ++i) {
        Div(lseSum[i * dealRowCount], lseSum[i * dealRowCount], lseSumUb, dealRowCount);
    }
    pipe_barrier(PIPE_V);
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::ReduceFinalRes(LocalTensor<T> &dst, LocalTensor<T> &lseLocal,
                                                             uint32_t startRow, uint32_t dealRowCount)
{
    BinaryRepeatParams repeatParams;
    repeatParams.src0RepStride = 1;
    repeatParams.src0BlkStride = 0;
    repeatParams.src1RepStride = headDimAlign / FP32_ONE_BLOCK_SIZE;
    repeatParams.dstRepStride = headDimAlign / FP32_ONE_BLOCK_SIZE;
    int32_t dtypeMask = 256 / sizeof(float);
    int32_t mulLoop = headDimAlign / dtypeMask;
    int32_t mulRemain = headDimAlign % dtypeMask;

    LocalTensor<T> lseLocalBrcb = tmpBuff2.Get<T>();
    Brcb(lseLocalBrcb, lseLocal, (dealRowCount * actualCombineLoopSize + FP32_ONE_BLOCK_SIZE - 1) / FP32_ONE_BLOCK_SIZE, {1, FP32_ONE_BLOCK_SIZE});

    // 第一次，mul结果直接放到dst里
    CopyAccumOutIn(0, startRow, dealRowCount);
    LocalTensor<T> accumOutLocal = inputQue1.DeQue<T>();
    for (int i = 0; i < mulLoop; i++) {
        Mul(dst[i * dtypeMask], lseLocalBrcb, accumOutLocal[i * dtypeMask], dtypeMask, dealRowCount, repeatParams);
    }
    if (mulRemain > 0) {
        Mul(dst[mulLoop * dtypeMask], lseLocalBrcb, accumOutLocal[mulLoop * dtypeMask], mulRemain, dealRowCount,
            repeatParams);
    }
    pipe_barrier(PIPE_V);
    inputQue1.FreeTensor(accumOutLocal);

    for (uint32_t j = 1; j < actualCombineLoopSize; ++j) {
        CopyAccumOutIn(j, startRow, dealRowCount);
        LocalTensor<T> accumOutLocal = inputQue1.DeQue<T>();
        for (int i = 0; i < mulLoop; i++) {
            Mul(accumOutLocal[i * dtypeMask], lseLocalBrcb[j * dealRowCount * FP32_ONE_BLOCK_SIZE],
                accumOutLocal[i * dtypeMask], dtypeMask, dealRowCount, repeatParams);
        }
        if (mulRemain > 0) {
            Mul(accumOutLocal[mulLoop * dtypeMask], lseLocalBrcb[j * dealRowCount * FP32_ONE_BLOCK_SIZE],
                accumOutLocal[mulLoop * dtypeMask], mulRemain, dealRowCount, repeatParams);
        }
        pipe_barrier(PIPE_V);
        Add(dst, dst, accumOutLocal, dealRowCount * headDimAlign);
        pipe_barrier(PIPE_V);
        // pipe_barrier(PIPI_V)与inputQue1.FreeTensor之间没有关系，这里的PIPE_V是为了让Add和接下来的VEC指令隔开
        inputQue1.FreeTensor(accumOutLocal);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::CopyFinalResOut(LocalTensor<T> &accumOutLocal,
                                                                                     uint32_t startRow,
                                                                                     uint32_t dealRowCount)
{
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputQue1.AllocTensor<OUT_T>();
    uint32_t shapeArray[] = {dealRowCount, (uint32_t)headDim};
    tmpBmm2ResCastTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
        Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_RINT, dealRowCount * headDimAlign);
    } else {
        Cast(tmpBmm2ResCastTensor, accumOutLocal, AscendC::RoundMode::CAST_ROUND, dealRowCount * headDimAlign);
    }

    outputQue1.EnQue(tmpBmm2ResCastTensor);
    outputQue1.DeQue<OUT_T>();
    Bmm2DataCopyOutFD(attenOutOffset, tmpBmm2ResCastTensor, startRow, dealRowCount, headDimAlign, headDim);
    outputQue1.FreeTensor(tmpBmm2ResCastTensor);
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::CombineSplitKVRes()
{
    if (curActualSeqLen != 0) {
        uint32_t gSplitSizeLse = BUFFER_SIZE_BYTE_16K / (BYTE_BLOCK * splitKVNum);
        uint32_t gSplitSizeAccumOut = BASE_BLOCK_MAX_ELEMENT_NUM / headDimAlign;
        // 取两者较小的，用来切g，保证ub够用
        uint32_t gSplitSize = (gSplitSizeLse < gSplitSizeAccumOut) ? gSplitSizeLse : gSplitSizeAccumOut;
        gSplitSize = (gSplitSize > gSize) ? gSize : gSplitSize; // 最大为gSize
        uint32_t loopCount = (gSize + gSplitSize - 1) / gSplitSize;
        uint32_t tailSplitSize = gSize - (loopCount - 1) * gSplitSize;

        // 尾块与非尾块都使用这些ub，减少处理次数
        for (uint32_t i = 0, actualGSplitSize = gSplitSize; i < loopCount; i++) {
            uint32_t startRow = i * gSplitSize;
            if ((i + 1) == loopCount) {
                actualGSplitSize = tailSplitSize;
            }
            CopyLseIn(startRow, actualGSplitSize);
            LocalTensor<T> lseSum = inputQue2.DeQue<T>();
            LocalTensor<T> lseMax = inputQue1.DeQue<T>();
            ComputeScaleValue(lseSum, lseMax, startRow, actualGSplitSize);
            inputQue1.FreeTensor(lseMax);

            uint32_t gSplitBmm2UbSize = headDimAlign * actualGSplitSize;
            LocalTensor<T> tmp1 = tmpBuff1.Get<T>(gSplitBmm2UbSize);
            ReduceFinalRes(tmp1, lseSum, startRow, actualGSplitSize);
            inputQue2.FreeTensor(lseSum);

            CopyFinalResOut(tmp1, startRow, actualGSplitSize);           
        }
    }
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::FlashDecodeCompute()
{
    bIdx = tmpBlockIdx / kvHeadNum;
    attenOutOffset = bIdx * kvHeadNum * gSize * headDim + n2Idx * gSize * headDim;
    mSizeVStart = 0;
    if (tmpBlockIdx >= batchSize * kvHeadNum) {
        return;
    }

    if (actualLenDims == 0) {
        curActualSeqLen = kvSeqSize;
        if (!batchContinuous) {
            curActualSeqLen = SeqLenFromTensorList(bIdx);
        }
    } else if (actualLenDims == 1) {
        curActualSeqLen = actualSeqLengthsGm.GetValue(0);
    } else {
        curActualSeqLen = actualSeqLengthsGm.GetValue(bIdx);
    }

    actualCombineLoopSize = (curActualSeqLen + sInnerLoopSize - 1) / sInnerLoopSize;

    CombineSplitKVRes();
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::ComputeLogSumExpAndCopyToGm(const ExtraInfoMla &info, LocalTensor<T> &softmaxSumUb,
                                                                          LocalTensor<T> &softmaxMaxUb)
{
    size_t size = mSizeVector;
    size_t offset = info.bIdx * kvHeadNum * splitKVNum * gSize + info.n2Idx * splitKVNum * gSize +
                    s2IdxFD * gSize + mSizeVStart;
    // lseSumFdGm: batchQ * kvHeadNum * splitKVNum * gSize * FP32_ONE_BLOCK_SIZE
    CopyFixedUbToGm(lseSumFdGm[offset], softmaxSumUb, size);
    CopyFixedUbToGm(lseMaxFdGm[offset], softmaxMaxUb, size);
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::ElewiseCompute(const ExtraInfoMla &info, LocalTensor<T> &mmResUb, TBuf<> &tmpBuf, uint32_t startRow,
                                                             uint32_t dealRowCount, uint32_t columnCount,
                                                             uint32_t actualColumnCount)
{
    Muls(mmResUb, mmResUb, static_cast<T>(tilingData->baseParams.scaleValue), dealRowCount * columnCount);
    pipe_barrier(PIPE_V);

    // attenMask
    if (attenMaskFlag == 1) {
        LocalTensor<bool> attenMaskUb;
        if (qSeqSize == 1) {
            AttenMaskCopyIn(info.attenMaskOffset, dealRowCount, actualColumnCount);
            attenMaskUb = inputQue2.DeQue<bool>();
            for (int i = 1; i < dealRowCount; i++) {
                DataCopy(attenMaskUb[i * attenMaskSizeAlign], attenMaskUb, attenMaskSizeAlign);
            }
        } else {
            if constexpr (LAYOUT_T == LAYOUT::BNSD) {
                AttenMaskCopyIn(info);
                attenMaskUb = inputQue2.DeQue<bool>();
                LocalTensor<bool> attenMaskUbDst = attenMaskUb[BUFFER_SIZE_BYTE_16K / 2];

                uint32_t headS1Count = 0;
                uint32_t headS1Start = (mSizeVStart + startRow) % info.s1Size;
                if (headS1Start + dealRowCount > info.s1Size) {
                    headS1Count = info.s1Size - headS1Start;
                } else {
                    headS1Count = dealRowCount;
                }

                // head
                DataCopy(attenMaskUbDst, attenMaskUb[headS1Start * attenMaskSizeAlign], headS1Count * attenMaskSizeAlign);
                // mid
                uint32_t reminRowCount = dealRowCount - headS1Count;
                uint32_t midGCount = reminRowCount / info.s1Size;
                uint32_t tailS1Size = reminRowCount % info.s1Size;
                for (uint32_t i = 0; i < midGCount; i++) {
                    DataCopy(attenMaskUbDst[(headS1Count + i * info.s1Size) * attenMaskSizeAlign],
                        attenMaskUb, info.s1Size * attenMaskSizeAlign);
                }
                // tail
                if (tailS1Size > 0) {
                    DataCopy(attenMaskUbDst[(headS1Count + midGCount * info.s1Size) * attenMaskSizeAlign],
                        attenMaskUb, tailS1Size * attenMaskSizeAlign);
                }
                attenMaskUb = attenMaskUbDst;
            } else { // BSH/BSND
                uint32_t s1StartIdx = (mSizeVStart + startRow) / info.gSize;
                uint32_t s1EndIdx = (mSizeVStart + startRow + dealRowCount - 1) / info.gSize;
                uint32_t s1Count = s1EndIdx - s1StartIdx + 1;

                #define ATTENMASK_STRIDE  2048U

                uint32_t offset;
                uint32_t actualSeqQ = qSeqSize;
                int32_t delta = (info.s1Idx * s1SizeSub + s1StartIdx) - info.s2Idx * singleProcessSInnerSize +
                                (info.s2Size - actualSeqQ);
                if (delta < 0) {
                    offset = (-delta) < (int32_t)s1Count ? (-delta) : s1Count;
                } else  {
                    offset = (delta < (int32_t)singleProcessSInnerSize ? delta : singleProcessSInnerSize) *
                            ATTENMASK_STRIDE;
                }

                attenMaskSizeAlign = Align(info.actualSingleProcessSInnerSize, 32U);

                DataCopyParams dataCopyParams;
                dataCopyParams.blockCount = s1Count;
                dataCopyParams.blockLen = attenMaskSizeAlign * sizeof(bool) /32;
                dataCopyParams.srcStride = (ATTENMASK_STRIDE - attenMaskSizeAlign) * sizeof(bool) / 32;
                dataCopyParams.dstStride = 0;

                attenMaskUb = inputQue2.AllocTensor<bool>();
                DataCopy(attenMaskUb, attenMaskBoolGm[offset], dataCopyParams);

                inputQue2.template EnQue(attenMaskUb);

                attenMaskUb = inputQue2.DeQue<bool>();
                LocalTensor<int16_t> mask16 = attenMaskUb.template ReinterpretCast<int16_t>();
                LocalTensor<int16_t> attenMaskUbDst = mask16[BUFFER_SIZE_BYTE_16K / 4];

                uint32_t headGCount = 0;
                uint32_t firstGIdx = (mSizeVStart + startRow) % info.gSize;
                if (s1Count > 1) {
                    headGCount = info.gSize - firstGIdx;
                } else {
                    headGCount = dealRowCount;
                }

                uint32_t dstMaskOffset = 0;
                uint32_t srcMaskBaseOffset = 0;
                // head
                SetMaskCount();
                SetVectorMask<int16_t, MaskMode::COUNTER>(attenMaskSizeAlign / 2);
                Copy<int16_t, false>(attenMaskUbDst[dstMaskOffset], mask16[srcMaskBaseOffset],
                                     AscendC::MASK_PLACEHOLDER, headGCount,
                                     {1, 1, static_cast<uint16_t>(attenMaskSizeAlign / 32), 0});
                dstMaskOffset += headGCount * attenMaskSizeAlign / 2;
                srcMaskBaseOffset += attenMaskSizeAlign / 2;
                // mid
                uint32_t reminRowCount = dealRowCount - headGCount;
                uint32_t midS1Count = reminRowCount / info.gSize;
                uint32_t tailGSize = reminRowCount % info.gSize;
                for (uint32_t midIdx = 0; midIdx < midS1Count; midIdx++) {
                    Copy<int16_t, false>(attenMaskUbDst[dstMaskOffset], mask16[srcMaskBaseOffset],
                                         AscendC::MASK_PLACEHOLDER, info.gSize,
                                         {1, 1, static_cast<uint16_t>(attenMaskSizeAlign / 32), 0});
                    dstMaskOffset += info.gSize * attenMaskSizeAlign / 2;
                    srcMaskBaseOffset += attenMaskSizeAlign / 2;
                }
                // tail
                if (tailGSize > 0) {
                    Copy<int16_t, false>(attenMaskUbDst[dstMaskOffset], mask16[srcMaskBaseOffset],
                                        AscendC::MASK_PLACEHOLDER, tailGSize,
                                        {1, 1, static_cast<uint16_t>(attenMaskSizeAlign / 32), 0});
                }
                SetMaskNorm();
                ResetMask();
                attenMaskUb = attenMaskUbDst.template ReinterpretCast<bool>();
            }
        }

        pipe_barrier(PIPE_V);

        LocalTensor<uint8_t> ubWorkSpace = tmpBuf.Get<uint8_t>();
        SelectWithBytesMaskShapeInfo selectWithBytesMaskShapeInfo;
        selectWithBytesMaskShapeInfo.firstAxis = dealRowCount;
        selectWithBytesMaskShapeInfo.srcLastAxis = columnCount;
        selectWithBytesMaskShapeInfo.maskLastAxis = attenMaskSizeAlign;
        attenMaskUb.SetSize(dealRowCount * attenMaskSizeAlign); // Select接口要求mask size与参数匹配
        mmResUb.SetSize(dealRowCount * columnCount);            // Select接口要求src size与参数匹配
        SelectWithBytesMask(mmResUb, mmResUb, BOOL_ATTEN_MASK_SCALAR_VALUE, attenMaskUb, ubWorkSpace,
                            selectWithBytesMaskShapeInfo);
        mmResUb.SetSize(BUFFER_SIZE_BYTE_32K / sizeof(T)); // mmResUb Size复原,mask不用复原,与原来一致
        inputQue2.FreeTensor(attenMaskUb);

        pipe_barrier(PIPE_V);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::SoftmaxFlashV2Compute(const ExtraInfoMla &info,
    LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t actualColumnCount)
{
    SoftMaxShapeInfo srcShape{dealRowCount, columnCount, dealRowCount, actualColumnCount};
    SoftMaxTiling newTiling =
        SoftMaxFlashV2TilingFunc(srcShape, sizeof(T), sizeof(T), softmaxTmpUb.GetSize(), true, false);

    LocalTensor<T> inSumTensor;
    LocalTensor<T> inMaxTensor;
#ifdef IFA_SOFTMAX_WITHOUT_BRC
    uint32_t baseOffset = startRow;
#else
    uint32_t baseOffset = startRow * BLOCK_ELEMENT_NUM;
#endif
    uint32_t outIdx = info.loop % (PRE_LOAD_NUM_MLA);
    uint32_t softmaxOutOffset = outIdx * BUFFER_SIZE_BYTE_2K / sizeof(T) + baseOffset;
    if (info.isFirstSInnerLoop) {
        inMaxTensor = softmaxMaxDefaultUb;
        inSumTensor = softmaxSumDefaultUb;
    } else {
        uint32_t inIdx = (info.loop -1) % (PRE_LOAD_NUM_MLA);
        inMaxTensor = softmaxMaxUb[inIdx * BUFFER_SIZE_BYTE_2K / sizeof(T) + baseOffset];
        inSumTensor = softmaxSumUb[inIdx * BUFFER_SIZE_BYTE_2K / sizeof(T) + baseOffset];
    }

#ifdef IFA_SOFTMAX_WITHOUT_BRC
    SoftmaxFlashV2<T, true, true, false, false, IFA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC>(
        mmResUb, softmaxSumUb[softmaxOutOffset],
        softmaxMaxUb[softmaxOutOffset], mmResUb, softmaxExpUb[softmaxOutOffset],
        inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
#else
    SoftmaxFlashV2<T, true, true, false, false, IFA_SOFTMAX_FLASHV2_CFG>(
        mmResUb, softmaxSumUb[softmaxOutOffset],
        softmaxMaxUb[softmaxOutOffset], mmResUb, softmaxExpUb[softmaxOutOffset],
        inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
#endif
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::Bmm2FDDataCopyOut(const ExtraInfoMla &info, LocalTensor<T> &attenOutUb, uint32_t startRow,
                                                                uint32_t dealRowCount, uint32_t columnCount,
                                                                uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(T));
    dataCopyParams.dstStride = 0;

    LocalTensor<T> tmp = outputQue1.AllocTensor<T>();
    DataCopy(tmp, attenOutUb, columnCount * dealRowCount);
    outputQue1.EnQue(tmp);
    outputQue1.DeQue<T>();

    size_t base = (info.bIdx * qHeadNum + info.n2Idx * gSize) * splitKVNum * headDim;
    // accumOutGm: batchQ * kvHeadNum * gSize * kvSplitPart_ * headDimAlign_
    DataCopyPad(accumOutGm[base + s2IdxFD * gSize * actualColumnCount +
                           startRow * actualColumnCount + mSizeVStart * actualColumnCount], tmp, dataCopyParams);
    outputQue1.FreeTensor(tmp);
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::Bmm2DataCopyOutFD(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb,
                                                            uint32_t startRow, uint32_t dealRowCount,
                                                            uint32_t columnCount, uint32_t actualColumnCount)
{
    Bmm2DataCopyOut(attenOutOffset, attenOutUb, startRow, dealRowCount, columnCount, actualColumnCount);
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[attenOutOffset + (mSizeVStart + startRow) * actualColumnCount], attenOutUb, dataCopyParams);
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::Bmm2DataCopyOutTrans(const ExtraInfoMla &info, LocalTensor<OUT_T> &attenOutUb,
                                                               uint32_t startRow, uint32_t dealRowCount,
                                                               uint32_t columnCount, uint32_t actualColumnCount)
{
       DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[info.attenOutOffset + (mSizeVStart + startRow) * actualColumnCount], attenOutUb, dataCopyParams);
    return;
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::DealInvalidRows(const ExtraInfoMla &info, LocalTensor<OUT_T> &attenOutUb,
                                                          uint32_t startRow, uint32_t dealRowCount,
                                                          uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t s1Tok = qSeqSize - info.s2Size;

    if constexpr (LAYOUT_T == LAYOUT::BNSD) {
        uint32_t s1 = (mSizeVStart + startRow) % info.s1Size;
        for (uint32_t i = 0; i < dealRowCount;) {
            if (s1 < s1Tok) {
                uint32_t s1Num = s1Tok - s1;
                if (i + s1Num > dealRowCount) {
                    s1Num = dealRowCount - i;
                }
                Duplicate(attenOutUb[i * columnCount], static_cast<OUT_T>(FLOAT_ZERO), columnCount * s1Num);
            }
            i += info.s1Size - s1;
            s1 = 0;
        }
        return;
    }

    // BSH
    uint32_t s1 = info.s1Idx * s1SizeSub + (mSizeVStart + startRow) / info.gSize;
    uint32_t gIdx = (mSizeVStart + startRow) % info.gSize;
    for (uint32_t i = 0; i < dealRowCount;) {
        if (s1 < s1Tok) {
            uint32_t gNum = info.gSize - gIdx;
            if (i + gNum > dealRowCount) {
                gNum = dealRowCount - i;
            }
            Duplicate(attenOutUb[i * columnCount], static_cast<OUT_T>(FLOAT_ZERO), columnCount * gNum);
            i += gNum;
            s1++;
            gIdx = 0;
            continue;
        }
        break;
    }
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::Bmm2ResCopyOut(const ExtraInfoMla &info, LocalTensor<T> &bmm2ResUb, uint32_t startRow,
                                                         uint32_t dealRowCount, uint32_t columnCount,
                                                         uint32_t actualColumnCount)
{
    if constexpr (FLASH_DECODE) {
        Bmm2FDDataCopyOut(info, bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
    } else {
        Bmm2CastAndCopyOut(info, bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
    }
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::Bmm2CastAndCopyOut(const ExtraInfoMla &info, LocalTensor<T> &bmm2ResUb, uint32_t startRow,
                                                             uint32_t dealRowCount, uint32_t columnCount,
                                                             uint32_t actualColumnCount)
{ 
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputQue1.AllocTensor<OUT_T>();
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_RINT, dealRowCount * columnCount);
    } else {
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_ROUND, dealRowCount * columnCount);
    }

    if (attenMaskFlag && (qSeqSize > info.s2Size)) {
        pipe_barrier(PIPE_V);
        DealInvalidRows(info, tmpBmm2ResCastTensor, startRow, dealRowCount, columnCount, actualColumnCount);
    }

    outputQue1.EnQue(tmpBmm2ResCastTensor);
    outputQue1.DeQue<OUT_T>();
    Bmm2DataCopyOutTrans(info, tmpBmm2ResCastTensor, startRow, dealRowCount, columnCount, actualColumnCount);
    outputQue1.FreeTensor(tmpBmm2ResCastTensor);
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::DealBmm1ResBaseBlock(const ExtraInfoMla &info, uint32_t startRow,
                                                                   uint32_t dealRowCount, uint32_t columnCount,
                                                                   uint32_t actualColumnCount)
{
    uint32_t computeSize = dealRowCount * columnCount;
    LocalTensor<T> mmResUb = tmpBuff1.Get<T>();

    size_t batchBase = 0;

    uint64_t inOutGmOffset = (info.loop % PRE_LOAD_NUM_MLA) * mmResUbSize + (mSizeVStart + startRow) * columnCount;

    LocalTensor<MM_OUT_T> tmpMmResUb = inputQue1.AllocTensor<MM_OUT_T>();

    DataCopy(tmpMmResUb, mm1ResGm[inOutGmOffset + batchBase], computeSize);


    inputQue1.EnQue(tmpMmResUb);
    inputQue1.DeQue<MM_OUT_T>();
    DataCopy(mmResUb, tmpMmResUb, computeSize);
    inputQue1.FreeTensor(tmpMmResUb);
    pipe_barrier(PIPE_V);
    ElewiseCompute(info, mmResUb, tmpBuff2, startRow, dealRowCount, columnCount, actualColumnCount);

    LocalTensor<T> tmpAFloorUb = tmpBuff2.Get<T>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();
    SoftmaxFlashV2Compute(info, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount, actualColumnCount);
    pipe_barrier(PIPE_V);

    LocalTensor<KV_T> tmpMMResCastTensor = outputQue1.AllocTensor<KV_T>();
    Cast(tmpMMResCastTensor, mmResUb, AscendC::RoundMode::CAST_ROUND, computeSize);

    outputQue1.EnQue(tmpMMResCastTensor);
    outputQue1.DeQue<KV_T>();

    DataCopy(vec1ResGm[inOutGmOffset + batchBase], tmpMMResCastTensor, computeSize);
    outputQue1.FreeTensor(tmpMMResCastTensor);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::ProcessVec1Inner(const ExtraInfoMla &info)
{
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / info.actualSingleProcessSInnerSizeAlign;
#ifdef IFA_SOFTMAX_WITHOUT_BRC
    // 1. 向下8对齐是因为UB操作至少32B
    // 2. info.actualSingleProcessSInnerSizeAlign最大512, mSplitSize可以确保最小为16
    mSplitSize = mSplitSize / 8 * 8;
#endif
    if (mSplitSize > mSizeVector) {
        mSplitSize = mSizeVector;
    }
    uint32_t loopCount = (mSizeVector + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mSizeVector - (loopCount - 1) * mSplitSize;

    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm1ResBaseBlock(info, i * mSplitSize, dealSize, info.actualSingleProcessSInnerSizeAlign,
                             info.actualSingleProcessSInnerSize);
    }

    if (info.s2Idx == info.curSInnerLoopTimes - 1) {
        if constexpr (FLASH_DECODE) {
            uint32_t outIdx = info.loop % (PRE_LOAD_NUM_MLA);
            auto sumTensor = softmaxSumUb[outIdx * BUFFER_SIZE_BYTE_2K / sizeof(T)];
            auto maxTensor = softmaxMaxUb[outIdx * BUFFER_SIZE_BYTE_2K / sizeof(T)];
            ComputeLogSumExpAndCopyToGm(info, sumTensor, maxTensor);
            return;
        }
    }
}

template <typename IFAT>
__aicore__ inline void
IncreFlashAttentionAttenPreloadMla<IFAT>::DealBmm2ResBaseBlock(const ExtraInfoMla &info, uint32_t startRow,
                                                                   uint32_t dealRowCount, uint32_t columnCount,
                                                                   uint32_t actualColumnCount)
{
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    uint32_t baseOffset = startRow * BLOCK_ELEMENT_NUM;
    LocalTensor<T> bmm2ResUb = tmpBuff1.Get<T>();
    bmm2ResUb.SetSize(vec2ComputeSize);

    size_t batchBase = 0;

    uint64_t inOutBaseOffset = (mSizeVStart + startRow) * columnCount;
    uint64_t srcGmOffset = (info.loop % PRE_LOAD_NUM_MLA) * bmm2ResUbSize + inOutBaseOffset;
    LocalTensor<MM_OUT_T> tmpBmm2ResUb = inputQue1.AllocTensor<MM_OUT_T>();
    DataCopy(tmpBmm2ResUb, mm2ResGm[srcGmOffset + batchBase], vec2ComputeSize);
    inputQue1.EnQue(tmpBmm2ResUb);
    inputQue1.DeQue<MM_OUT_T>();
    DataCopy(bmm2ResUb, tmpBmm2ResUb, vec2ComputeSize);
    inputQue1.FreeTensor(tmpBmm2ResUb);

    // 除第一个循环外，均需要更新中间计算结果
    if (!info.isFirstSInnerLoop) {
        event_t eventIdMte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        LocalTensor<T> bmm2ResPreUb = inputQue2.AllocTensor<T>();
        uint64_t vec2ResGmOffset = ((info.loop - 1) % PRE_LOAD_NUM_MLA) * bmm2ResUbSize + inOutBaseOffset;
        DataCopy(bmm2ResPreUb, vec2ResGm[vec2ResGmOffset + batchBase], vec2ComputeSize);
        inputQue2.EnQue(bmm2ResPreUb);

        inputQue2.DeQue<T>();
        pipe_barrier(PIPE_V);
        uint32_t idx = info.loop % (PRE_LOAD_NUM_MLA);
#ifdef IFA_SOFTMAX_WITHOUT_BRC
        LocalTensor<T> tmpExpBrcbResUb = tmpBuff2.Get<T>();
        Brcb(tmpExpBrcbResUb, softmaxExpUb[idx * BUFFER_SIZE_BYTE_2K / sizeof(T) + startRow], (dealRowCount + 7) / 8, {1, 8});
        pipe_barrier(PIPE_V);
        RowMuls(bmm2ResPreUb, bmm2ResPreUb, tmpExpBrcbResUb, dealRowCount, columnCount, actualColumnCount);
#else
        RowMuls(bmm2ResPreUb, bmm2ResPreUb, softmaxExpUb[idx * BUFFER_SIZE_BYTE_2K / sizeof(T) + baseOffset], dealRowCount, columnCount, actualColumnCount);
#endif
        pipe_barrier(PIPE_V);
        Add(bmm2ResUb, bmm2ResUb, bmm2ResPreUb, vec2ComputeSize);
        inputQue2.FreeTensor(bmm2ResPreUb);
    }

    // 最后一次输出计算结果，否则将中间结果暂存至workspace
    if (info.s2Idx + 1 == info.curSInnerLoopTimes) {
        pipe_barrier(PIPE_V);
        uint32_t idx = info.loop % (PRE_LOAD_NUM_MLA);
#ifdef IFA_SOFTMAX_WITHOUT_BRC
        LocalTensor<T> tmpSumBrcbResUb = tmpBuff2.Get<T>();
        Brcb(tmpSumBrcbResUb, softmaxSumUb[idx * BUFFER_SIZE_BYTE_2K / sizeof(T) + startRow], (dealRowCount + 7) / 8, {1, 8});
        pipe_barrier(PIPE_V);
        RowDivs(bmm2ResUb, bmm2ResUb, tmpSumBrcbResUb, dealRowCount, columnCount, actualColumnCount);
#else
        RowDivs(bmm2ResUb, bmm2ResUb, softmaxSumUb[idx * BUFFER_SIZE_BYTE_2K / sizeof(T) + baseOffset], dealRowCount, columnCount, actualColumnCount);
#endif

        pipe_barrier(PIPE_V);
        Bmm2ResCopyOut(info, bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount);
    } else {
        pipe_barrier(PIPE_V);
        LocalTensor<T> tmpBmm2Res = outputQue1.AllocTensor<T>();
        DataCopy(tmpBmm2Res, bmm2ResUb, dealRowCount * columnCount);
        outputQue1.EnQue(tmpBmm2Res);
        outputQue1.DeQue<T>();
        uint64_t vec2ResGmOffset = (info.loop % PRE_LOAD_NUM_MLA) * bmm2ResUbSize + inOutBaseOffset;
        DataCopy(vec2ResGm[vec2ResGmOffset + batchBase], tmpBmm2Res, vec2ComputeSize);

        outputQue1.FreeTensor(tmpBmm2Res);
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::ProcessVec2Inner(const ExtraInfoMla &info)
{
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / headDimAlign;
    if (mSplitSize > mSizeVector) {
        mSplitSize = mSizeVector;
    }
    uint32_t loopCount = (mSizeVector + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mSizeVector - (loopCount - 1) * mSplitSize;

    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        {
            DealBmm2ResBaseBlock(info, i * mSplitSize, dealSize, headDimAlign, headDim);
        }
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::CalcParams(uint32_t loop, ExtraInfoMla &info, TaskContext &task) {
    info.loop = loop;
    info.bIdx = task.bidx;
    info.s1Idx = task.s1idx;
    info.s2Idx = task.s2idx;
    info.curSInnerLoopTimes = task.s2loops;
    info.s1Size = task.s1Size;
    info.s2Size = task.s2Size;
    info.gIdx = task.gidx;
    info.gSize = (task.gidx == gOuter - 1) ? gSizeTail : gSizeSub;

    if ASCEND_IS_AIV {
        info.mSize = info.gSize * info.s1Size;
        info.mSizeV = (info.mSize + 1) / 2;
        info.mSizeVStart = 0;
        if (tmpBlockIdx % 2 == 1) {
            info.mSizeVStart = info.mSizeV;
            info.mSizeV = info.mSize - info.mSizeV;
        }
    }

    if (batchContinuous) {
       info.isChangeBatch = false;
    } else {
       if (loop == 0) {
           info.isChangeBatch = true;
       } else {
           info.isChangeBatch = (task.nidx == 0 && task.s2idx == 0);
       }
    }

    info.isFirstSInnerLoop = (info.s2Idx == 0);
    if (info.isFirstSInnerLoop) {
        bn2IdxInCurCore++;
    }
    info.bn2IdxInCurCore = bn2IdxInCurCore - 1;
    if (info.isFirstSInnerLoop) {
        if constexpr (LAYOUT_T == LAYOUT::BNSD) {
            // B,N2,G,1,D
            tensorACoreOffset = info.bIdx * qHeadNum * qSeqSize * headDim + info.n2Idx * gSize * qSeqSize * headDim +
                                info.gIdx * gSizeSub * qSeqSize * headDim;
            tensorARopeCoreOffset = info.bIdx * qHeadNum * qSeqSize * headDimRope +
                                    info.n2Idx * gSize * qSeqSize * headDimRope + info.gIdx * gSizeSub * qSeqSize * headDimRope;
            // B,N2,S2,D
            tensorBCoreOffset = info.bIdx * kvHeadNum * kvSeqSize * headDim + info.n2Idx * kvSeqSize * headDim;
            tensorBRopeCoreOffset = info.bIdx * kvHeadNum * kvSeqSize * headDimRope + info.n2Idx * kvSeqSize * headDimRope;
            if (!batchContinuous) {
                uint64_t seqSize = SeqLenFromTensorList(info.bIdx);
                tensorBCoreOffset = info.n2Idx * seqSize * headDim;
                tensorBRopeCoreOffset = info.n2Idx * seqSize * headDimRope;
            }

            if constexpr (FLASH_DECODE) {
                tensorBCoreOffset += s2IdxFD * sInnerLoopSize * headDim;
                tensorBRopeCoreOffset += s2IdxFD * sInnerLoopSize * headDimRope;
            }
        } else {
            // B,S,N2,G,D
            tensorACoreOffset = info.bIdx * qSeqSize * qHeadNum * headDim +
                                info.s1Idx * s1SizeSub * qHeadNum * headDim + info.n2Idx * gSize * headDim;
            tensorARopeCoreOffset = info.bIdx * qSeqSize * qHeadNum * headDimRope +
                                    info.s1Idx * s1SizeSub * qHeadNum * headDimRope + info.n2Idx * gSize * headDimRope;
            // B,S2,N2,D
            tensorBCoreOffset = info.bIdx * kvSeqSize * kvHeadNum * headDim + info.n2Idx * headDim;
            tensorBRopeCoreOffset = info.bIdx * kvSeqSize * kvHeadNum * headDimRope + info.n2Idx * headDimRope;

            if (!batchContinuous) {
                tensorBCoreOffset = info.n2Idx * headDim;
                tensorBRopeCoreOffset = info.n2Idx * headDimRope;
            }

            if constexpr (FLASH_DECODE) {
                tensorBCoreOffset += s2IdxFD * sInnerLoopSize * kvHeadNum * headDim;
                tensorBRopeCoreOffset += s2IdxFD * sInnerLoopSize * kvHeadNum * headDimRope;
            }
        }
    }
    info.tensorAOffset = tensorACoreOffset;
    info.tensorARopeOffset = tensorARopeCoreOffset;
    if constexpr (LAYOUT_T == LAYOUT::BNSD) {
        info.tensorBOffset = tensorBCoreOffset + info.s2Idx * singleProcessSInnerSize * headDim;
        info.tensorBRopeOffset = tensorBRopeCoreOffset + info.s2Idx * singleProcessSInnerSize * headDimRope;
    } else {
        info.tensorBOffset = tensorBCoreOffset + info.s2Idx * singleProcessSInnerSize * kvHeadNum * headDim;
        info.tensorBRopeOffset = tensorBRopeCoreOffset + info.s2Idx * singleProcessSInnerSize * kvHeadNum * headDimRope;
    }
    info.attenOutOffset = tensorACoreOffset;
    info.actualSingleProcessSInnerSize = singleProcessSInnerSize;
    if (info.s2Idx == info.curSInnerLoopTimes - 1) {
        info.actualSingleProcessSInnerSize = task.s2SizeTail;
    }

    info.actualSingleProcessSInnerSizeAlign = Align((uint32_t)info.actualSingleProcessSInnerSize, (uint32_t)BYTE_BLOCK);
    
    uint64_t sInnerOffsetDataSize = info.s2Idx * singleProcessSInnerSize;
    if constexpr (FLASH_DECODE) {
        sInnerOffsetDataSize += s2IdxFD * sInnerLoopSize;
    }

    attenMaskCoreOffset = info.bIdx * attenMaskSize;
    info.attenMaskOffset = attenMaskCoreOffset + sInnerOffsetDataSize;

    info.s2BatchOffset = s2BatchBaseOffset + sInnerOffsetDataSize;
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::ProcessVec2L(const ExtraInfoMla &info) {
    mSizeVector = info.mSizeV;
    mSizeVStart = info.mSizeVStart;
    
    if (mSizeVector == 0) {
        return;
    }

    ProcessVec2Inner(info);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::ProcessVec1L(const ExtraInfoMla &info)
{
    mSizeVector = info.mSizeV;
    mSizeVStart = info.mSizeVStart;

    if (mSizeVector == 0) {
        return;
    }
    ProcessVec1Inner(info);
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::ComputeMm1(const ExtraInfoMla &info)
{
    if (info.isChangeBatch) {
        InitKeyGm(info.bIdx);
        matmulService.UpdateKey(keyGm);
    }
    matmulService.ComputeMm1(info);
}

template <typename IFAT> __aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::ComputeMm2(const ExtraInfoMla &info)
{
    if (info.isChangeBatch) {
        InitValueGm(info.bIdx);
        matmulService.UpdateValue(valueGm);
    }
    matmulService.ComputeMm2(info);
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::Process()
{
    if (aiCoreIdx < usedCoreNum) {
        if ASCEND_IS_AIV {
#ifdef IFA_SOFTMAX_WITHOUT_BRC
            Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, BUFFER_SIZE_BYTE_2K / sizeof(T));
            Duplicate(softmaxSumDefaultUb, FLOAT_ZERO, BUFFER_SIZE_BYTE_2K / sizeof(T));
#else
            Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, BUFFER_SIZE_BYTE_2K / sizeof(T));
            Duplicate(softmaxSumDefaultUb, FLOAT_ZERO, BUFFER_SIZE_BYTE_2K / sizeof(T));
#endif
        } else {
            matmulService.AllocEventID();
        }

        ExtraInfoMla extraInfo[PRE_LOAD_NUM_MLA];
        TaskContext taskContext[PRE_LOAD_NUM_MLA];

        uint32_t gloop = 0;
        uint32_t tasks = 0;

        if constexpr (!FLASH_DECODE) {
            for (uint32_t bn2gIdx = bn2LoopTimes; bn2gIdx > 0; bn2gIdx--) {
                GetBN2Gid(bn2gIdx - 1);
                GetActualSeqLen(bIdx, s1Idx);
                if (curActualSeqLen == 0) {
                    DealActSeqLenIsZero(bIdx, n2Idx);
                }
                if (curActualSeqLen != 0) {
                    break;
                }
                bn2LoopTimes--;
            }
        }

        for (uint32_t bn2gIdx = 0; bn2gIdx < bn2LoopTimes; bn2gIdx++) {
            GetBN2Gid(bn2gIdx);
            GetActualSeqLen(bIdx, s1Idx);
            UpdateInnerLoopCond();
            if (curActSeqLenIsZero) {
                DealActSeqLenIsZero(bIdx, n2Idx);
                continue;
            }

            for (uint32_t sInnerLoopIdx = 0; sInnerLoopIdx < sInnerLoopTimes; sInnerLoopIdx++) {
                TaskContext &ctx = taskContext[tasks++];
                ctx.bidx = bIdx;
                // ctx.nidx = n2Idx;
                ctx.s2idx = sInnerLoopIdx;
                ctx.s2loops = sInnerLoopTimes;
                ctx.gidx = gIdx;
                ctx.s2SizeTail = singleProcessSInnerSizeTail;
                ctx.s1Size = actS1Size;
                ctx.s2Size = curActualSeqLen;
                ctx.s1idx = s1Idx;

                bool isLast = (bn2gIdx == bn2LoopTimes - 1) && (sInnerLoopIdx == sInnerLoopTimes - 1);
                if (tasks < PRE_LOAD_NUM_MLA && !isLast) {
                    continue;
                }

                for (uint32_t i = 0; i < tasks; i++) {
                    uint32_t loop = gloop + i;
                    if (loop >= PRE_LOAD_NUM_MLA) {
                        if ASCEND_IS_AIV {
                            CrossCoreWaitFlag(SYNC_C2_V2_FLAG);
                            ProcessVec2L(extraInfo[(loop - PRE_LOAD_NUM_MLA) % PRE_LOAD_NUM_MLA]);
                        }
                    }

                    CalcParams(loop, extraInfo[loop % PRE_LOAD_NUM_MLA], taskContext[loop % PRE_LOAD_NUM_MLA]);

                    if ASCEND_IS_AIC {
                        ComputeMm1(extraInfo[loop % PRE_LOAD_NUM_MLA]);
                        CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C1_V1_FLAG);
                    }
                }

                for (uint32_t i = 0; i < tasks; i++) {
                    uint32_t loop = gloop + i;
                    if ASCEND_IS_AIV {
                        CrossCoreWaitFlag(SYNC_C1_V1_FLAG);
                        ProcessVec1L(extraInfo[loop % PRE_LOAD_NUM_MLA]);
                        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_V1_C2_FLAG);
                    }
                    if ASCEND_IS_AIC {
                        CrossCoreWaitFlag(SYNC_V1_C2_FLAG);
                        ComputeMm2(extraInfo[loop % PRE_LOAD_NUM_MLA]);
                        CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_C2_V2_FLAG);
                    }
                }

                gloop += tasks;
                tasks = 0;
            }
        }

        for (uint32_t j = 0; j < PRE_LOAD_NUM_MLA; j++) {
            uint32_t loop = gloop + j;
            if (loop >= PRE_LOAD_NUM_MLA) {
                if ASCEND_IS_AIV {
                    CrossCoreWaitFlag(SYNC_C2_V2_FLAG);
                    ProcessVec2L(extraInfo[(loop - PRE_LOAD_NUM_MLA) % PRE_LOAD_NUM_MLA]);
                }
            }
        }

        if ASCEND_IS_AIC {
            matmulService.FreeEventID();
        }
    }

    if constexpr (FLASH_DECODE) {
        SyncAll();
        if ASCEND_IS_AIV {
            // 多核同步
            FlashDecodeCompute();
        }
    }
}

template <typename IFAT>
__aicore__ inline void IncreFlashAttentionAttenPreloadMla<IFAT>::CopyFixedUbToGm(const GlobalTensor<T> &dst,
                                                                                     const LocalTensor<T> &src,
                                                                                     size_t size)
{
    LocalTensor<T> tmp = outputQue2.template AllocTensor<T>();
    if (size % FP32_ONE_BLOCK_SIZE == 0) {
        DataCopy(tmp, src, size);
        outputQue2.EnQue(tmp);
        outputQue2.DeQue();
        DataCopy(dst, tmp, size);
    } else {
        DataCopy(tmp, src, (size + FP32_ONE_BLOCK_SIZE - 1) / FP32_ONE_BLOCK_SIZE * FP32_ONE_BLOCK_SIZE);  // 搬运需要32B的整数倍
        outputQue2.EnQue(tmp);
        outputQue2.DeQue();
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = size * BYTE_BLOCK / FP32_ONE_BLOCK_SIZE;  // 元素32位，每个连续传输快长度单位为Byte;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(dst, tmp, dataCopyParams);
    }
    outputQue2.FreeTensor(tmp);
}

#endif // INCRE_FLASH_ATTENTION_PRELOAD_MLA
