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
 * \file ifa_service_matmul_kv_nd.h
 * \brief
 */
#ifndef IFA_SERVICE_MATMUL_KV_ND_H
#define IFA_SERVICE_MATMUL_KV_ND_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "ifa_public_define.h"

template <typename IFAT>
class IfaMatmulKvNd {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;

    using Q_T = typename IFAT::queryType;
    using KV_T = typename IFAT::kvType;
    using OUT_T = typename IFAT::outputType;
    using ORIGIN_T = typename IFAT::orginalType;
    static constexpr bool PAGE_ATTENTION = IFAT::pageAttention;
    static constexpr bool KV_CONTINUOUS = IFAT::kvContinuous;
    static constexpr bool FLASH_DECODE = IFAT::flashDecode;
    static constexpr LAYOUT LAYOUT_T = IFAT::layout;
    static constexpr LAYOUT KV_LAYOUT_T = IFAT::kvLayout;
    static constexpr uint8_t PER_CHANNEL_MODE = 0; // 伪量化: K V per-channel
    static constexpr uint8_t PER_TOKEN_MODE = 1; // 伪量化: K V per-token
    static constexpr uint8_t PER_CHANNEL_TOKEN_MODE = 2; // 伪量化: K per-channel and V per-token
    static constexpr uint8_t ANTIQUANT_MODE = IFAT::antiquantMode;
    static constexpr bool SHARED_PREFIX = IFAT::sharedPrefix;

    static constexpr bool ANTIQUANT = !IsSameType<Q_T, KV_T>::value;
    static constexpr bool KVINT4 = IsSameType<KV_T, int4b_t>::value;
    static constexpr bool QUANT = (IsSameType<Q_T, KV_T>::value && IsSameType<KV_T, int8_t>::value);
    static constexpr bool ANTIQUANT_PER_CHANNEL_TOKEN = (ANTIQUANT && (ANTIQUANT_MODE == PER_CHANNEL_TOKEN_MODE));
    static constexpr bool ANTIQUANT_PER_TOKEN = (ANTIQUANT && (ANTIQUANT_MODE == PER_TOKEN_MODE));
    static constexpr bool ANTIQUANT_PER_CHANNEL = (ANTIQUANT && (ANTIQUANT_MODE == PER_CHANNEL_MODE));
    using ANTIQ_PARAMS_T = typename AscendC::Conditional<ANTIQUANT_PER_TOKEN, T, Q_T>::type;
    // define pse datetype
    using pseShiftType = typename AscendC::Conditional<AscendC::IsSameType<Q_T, int8_t>::value, half, Q_T>::type;
    // 后接量化的条件需要重新审视
    static constexpr bool POST_QUANT = IsSameType<OUT_T, int8_t>::value;
    using MM_OUT_T = typename AscendC::Conditional<(ANTIQUANT || QUANT), int32_t, T>::type;

    __aicore__ inline IfaMatmulKvNd() {};
    __aicore__ inline void InitParams(uint64_t qHeadNum, uint64_t kvHeadNum, uint64_t headDim, uint64_t headDimRope,
                                      uint64_t qSeqSize, uint32_t mmResUbSize, uint32_t bmm2ResUbSize);
    __aicore__ inline void InitMm1GlobalTensor(GlobalTensor<Q_T> queryGm, GlobalTensor<Q_T> qRopeGm,
        GlobalTensor<KV_T> keyGm, GlobalTensor<KV_T> kRopeGm, GlobalTensor<MM_OUT_T> mm1ResGm);
    __aicore__ inline void InitMm2GlobalTensor(GlobalTensor<KV_T> vec1ResGm, GlobalTensor<KV_T> valueGm,
        GlobalTensor<MM_OUT_T> mm2ResGm, GlobalTensor<OUT_T> attentionOutGm);
    __aicore__ inline void InitPageAttentionInfo(GlobalTensor<int32_t> blockTableGm, uint32_t blockSize,
        uint32_t maxBlockNumPerBatch);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void UpdateKey(GlobalTensor<KV_T> keyGm);
    __aicore__ inline void UpdateValue(GlobalTensor<KV_T> valueGm);

    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const ExtraInfoMla &info);
    __aicore__ inline void ComputeMm2(const ExtraInfoMla &info);

protected:
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

    __aicore__ inline void CopyGmToL1(LocalTensor<KV_T> &l1Tensor, GlobalTensor<KV_T> &gmSrcTensor, uint32_t srcN,
                                      uint32_t srcD, uint32_t srcDstride);
    __aicore__ inline void CopyInMm1AToL1(LocalTensor<KV_T>& aL1Tensor, const ExtraInfoMla &info);
    __aicore__ inline void CopyInMm1ARopeToL1(LocalTensor<KV_T>& aL1Tensor, const ExtraInfoMla &info);
    __aicore__ inline void CopyInMm1BToL1(LocalTensor<KV_T>& bL1Tensor, const ExtraInfoMla &info, uint32_t subNid, uint32_t subNSize);
    __aicore__ inline void CopyInMm1BRopeToL1(LocalTensor<KV_T>& bL1Tensor, const ExtraInfoMla &info, uint32_t subNid, uint32_t subNSize);
    __aicore__ inline void CopyInMm1BToL1ForPA(LocalTensor<KV_T>& bL1Tensor, const uint64_t keyGmBaseOffset,
        uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt, uint32_t nActCopyRowCount);
    __aicore__ inline void CopyInMm1BRopeToL1ForPA(LocalTensor<KV_T>& bL1Tensor, const uint64_t keyGmBaseOffset,
        uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt, uint32_t nActCopyRowCount);
    __aicore__ inline void CopyInMm2AToL1(LocalTensor<KV_T>& aL1Tensor, const ExtraInfoMla &info);
    __aicore__ inline void CopyInMm2BToL1(LocalTensor<KV_T>& aL1Tensor, const ExtraInfoMla &info, uint32_t subKid,
        uint32_t kSplitSize, uint32_t subNid, uint32_t nSplitSize, uint32_t subKSize, uint32_t subNSize);
    __aicore__ inline void CopyInMm2BToL1ForPA(LocalTensor<KV_T>& bL1Tensor, const uint64_t valueGmBaseOffset,
        uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt, uint32_t nActCopyRowCount,
        uint32_t copyStartColumnCount, uint32_t copyColumnCount);

    __aicore__ inline void LoadDataMm1A(LocalTensor<KV_T>& aL0Tensor, LocalTensor<KV_T>& aL1Tensor, uint32_t idx, uint32_t kSplitSize, uint32_t mSize, uint32_t kSize);
    __aicore__ inline void LoadDataMm1B(LocalTensor<KV_T>& bL0Tensor, LocalTensor<KV_T>& bL1Tensor, uint32_t idx, uint32_t kSplitSize, uint32_t kSize, uint32_t nSize);
    __aicore__ inline void LoadDataMm2A(LocalTensor<KV_T>& aL0Tensor, LocalTensor<KV_T>& aL1Tensor, uint32_t mSize, uint32_t subKid, uint32_t subKSize);
    __aicore__ inline void LoadDataMm2B(LocalTensor<KV_T>& bL0Tensor, LocalTensor<KV_T>& bL1Tensor, uint32_t idx, uint32_t nSize, uint32_t kSize);

protected:
    // mm1
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<Q_T> qRopeGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> kRopeGm;
    GlobalTensor<MM_OUT_T> mm1ResGm;

    // mm2
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<MM_OUT_T> mm2ResGm;
    GlobalTensor<OUT_T> attentionOutGm;

    // pageAttention
    uint32_t kvCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    // block_table
    GlobalTensor<int32_t> blockTableGm;

    // params
    uint32_t mmResUbSize = 0U;
    uint32_t bmm2ResUbSize = 0U;

    uint64_t qHeadNum = 0ULL;
    static constexpr uint64_t kvHeadNum = 1ULL;
    uint64_t gSize = 0ULL;

    uint64_t qSeqSize = 1ULL;
    static constexpr uint64_t headDim = 512ULL;
    static constexpr uint64_t headDimRope = 64ULL;

private:
    // L1
    static constexpr uint32_t L1_Q_SIZE = (128 * (512 + 64) * sizeof(Q_T));
    static constexpr uint32_t L1_KV_SIZE = (64 * (512 + 64) * sizeof(KV_T));

    // L0
    static constexpr uint32_t L0A_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0B_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0C_PP_SIZE = (64 * 1024);

    // mte2 <> mte1
    static constexpr uint32_t Q_EVENT0 = EVENT_ID2;
    static constexpr uint32_t Q_EVENT1 = EVENT_ID3;
    static constexpr uint32_t KV_EVENT0 = EVENT_ID4;
    static constexpr uint32_t KV_EVENT1 = EVENT_ID5;

    // m <> mte1
    static constexpr uint32_t L0A_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0A_EVENT1 = EVENT_ID4;
    static constexpr uint32_t L0B_EVENT0 = EVENT_ID5;
    static constexpr uint32_t L0B_EVENT1 = EVENT_ID6;

    // fix <> m
    static constexpr uint32_t L0C_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0C_EVENT1 = EVENT_ID4;

    TBuf<TPosition::A1> queryBufL1;
    LocalTensor<Q_T> qL1Tensor;

    TBuf<TPosition::A1> kvBufL1;
    LocalTensor<KV_T> kvL1Tensor;

    LocalTensor<KV_T> pL1Tensor;

    TBuf<TPosition::A2> tmpBufL0A;
    LocalTensor<KV_T> aL0TensorPingPong;

    // L0_B
    TBuf<TPosition::B2> tmpBufL0B;
    LocalTensor<KV_T> bL0TensorPingPong;

    // L0_C
    TBuf<TPosition::CO1> tmpBufL0C;
    LocalTensor<MM_OUT_T> cL0TensorPingPong;

    uint32_t qL1BufIter = 0;
    uint32_t kvL1BufIter = 0;
    uint32_t aL0BufIter = 0;
    uint32_t bL0BufIter = 0;
    uint32_t cL0BufIter = 0;
};

template <typename IFAT>
__aicore__ inline void IfaMatmulKvNd<IFAT>::InitParams(uint64_t qHeadNum, uint64_t kvHeadNum, uint64_t headDim,
                                                       uint64_t headDimRope, uint64_t qSeqSize, uint32_t mmResUbSize,
                                                       uint32_t bmm2ResUbSize)
{
    this->qHeadNum = qHeadNum;
    this->qSeqSize = qSeqSize;
    this->mmResUbSize = mmResUbSize;
    this->bmm2ResUbSize = bmm2ResUbSize;
    this->gSize = qHeadNum / kvHeadNum;
}

template <typename IFAT>
__aicore__ inline void IfaMatmulKvNd<IFAT>::InitMm1GlobalTensor(GlobalTensor<Q_T> queryGm, GlobalTensor<Q_T> qRopeGm,
                                                                GlobalTensor<KV_T> keyGm, GlobalTensor<KV_T> kRopeGm,
                                                                GlobalTensor<MM_OUT_T> mm1ResGm)
{
    // mm1
    this->queryGm = queryGm;
    this->qRopeGm = qRopeGm;
    this->keyGm = keyGm;
    this->kRopeGm = kRopeGm;
    this->mm1ResGm = mm1ResGm;
}

template <typename IFAT>
__aicore__ inline void IfaMatmulKvNd<IFAT>::InitMm2GlobalTensor(
    GlobalTensor<KV_T> vec1ResGm, GlobalTensor<KV_T> valueGm,
    GlobalTensor<MM_OUT_T> mm2ResGm, GlobalTensor<OUT_T> attentionOutGm)
{
    // mm2
    this->vec1ResGm = vec1ResGm;
    this->valueGm = valueGm;
    this->mm2ResGm = mm2ResGm;
    this->attentionOutGm = attentionOutGm;
}

template <typename IFAT>
__aicore__ inline void IfaMatmulKvNd<IFAT>::InitPageAttentionInfo(GlobalTensor<int32_t> blockTableGm,
                                                                  uint32_t blockSize, uint32_t maxBlockNumPerBatch)
{
    this->blockTableGm = blockTableGm;
    this->kvCacheBlockSize = blockSize;
    this->maxBlockNumPerBatch = maxBlockNumPerBatch;
}

template <typename IFAT>
__aicore__ inline void IfaMatmulKvNd<IFAT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(queryBufL1, L1_Q_SIZE * 2); // 128K + 16K
    qL1Tensor = queryBufL1.Get<Q_T>();
    pL1Tensor = queryBufL1.Get<KV_T>();

    pipe->InitBuffer(kvBufL1, L1_KV_SIZE * 2);
    kvL1Tensor = kvBufL1.Get<KV_T>();

    pipe->InitBuffer(tmpBufL0A, L0A_PP_SIZE * 2); // 64K
    aL0TensorPingPong = tmpBufL0A.Get<KV_T>();
    // L0B
    pipe->InitBuffer(tmpBufL0B, L0B_PP_SIZE * 2); // 64K
    bL0TensorPingPong = tmpBufL0B.Get<KV_T>();
    // L0C
    pipe->InitBuffer(tmpBufL0C, L0C_PP_SIZE * 2); // 128K
    cL0TensorPingPong = tmpBufL0C.Get<MM_OUT_T>();
}

template <typename IFAT>
__aicore__ inline void IfaMatmulKvNd<IFAT>::UpdateKey(GlobalTensor<KV_T> keyGm)
{
    this->keyGm = keyGm;
}

template <typename IFAT>
__aicore__ inline void IfaMatmulKvNd<IFAT>::UpdateValue(GlobalTensor<KV_T> valueGm)
{
    this->valueGm = valueGm;
}

template <typename IFAT>
__aicore__ inline void IfaMatmulKvNd<IFAT>::AllocEventID()
{
    SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT1);
    SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT1);

    SetFlag<HardEvent::M_MTE1>(L0A_EVENT0);
    SetFlag<HardEvent::M_MTE1>(L0A_EVENT1);
    SetFlag<HardEvent::M_MTE1>(L0B_EVENT0);
    SetFlag<HardEvent::M_MTE1>(L0B_EVENT1);

    SetFlag<HardEvent::FIX_M>(L0C_EVENT0);
    SetFlag<HardEvent::FIX_M>(L0C_EVENT1);
}

template <typename IFAT>
__aicore__ inline void IfaMatmulKvNd<IFAT>::FreeEventID()
{
    WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT1);
    WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT1);

    WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0);
    WaitFlag<HardEvent::M_MTE1>(L0A_EVENT1);
    WaitFlag<HardEvent::M_MTE1>(L0B_EVENT0);
    WaitFlag<HardEvent::M_MTE1>(L0B_EVENT1);

    WaitFlag<HardEvent::FIX_M>(L0C_EVENT0);
    WaitFlag<HardEvent::FIX_M>(L0C_EVENT1);
}

template <typename IFAT>
__aicore__ inline void IfaMatmulKvNd<IFAT>::CopyGmToL1(LocalTensor<KV_T>& l1Tensor, GlobalTensor<KV_T> &gmSrcTensor,
                                                     uint32_t srcN, uint32_t srcD, uint32_t srcDstride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = srcN; // 行数
    if constexpr (KVINT4) {
        nd2nzPara.dValue = srcD / 2;
        nd2nzPara.srcDValue = srcDstride / 2;
    } else {
        nd2nzPara.dValue = srcD;
        nd2nzPara.srcDValue = srcDstride;
    }
    nd2nzPara.dstNzC0Stride = (srcN + 15) / 16 * 16; // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
}

template <typename IFAT>
__aicore__ inline void IfaMatmulKvNd<IFAT>::CopyInMm1AToL1(LocalTensor<KV_T> &l1Tensor,
                                                                                const ExtraInfoMla &info)
{
    auto srcGm = queryGm[info.tensorAOffset];
    CopyGmToL1(l1Tensor, srcGm, info.gSize * info.s1Size, headDim, headDim);
}

template <typename IFAT>
__aicore__ inline void IfaMatmulKvNd<IFAT>::CopyInMm1ARopeToL1(LocalTensor<KV_T> &l1Tensor,
                                                                                    const ExtraInfoMla &info)
{
    auto srcGm = qRopeGm[info.tensorARopeOffset];
    CopyGmToL1(l1Tensor, srcGm, info.gSize * info.s1Size, headDimRope, headDimRope);
}

template <typename IFAT>
__aicore__ inline void IfaMatmulKvNd<IFAT>::CopyInMm1BToL1(LocalTensor<KV_T>& bL1Tensor,
                                                                                const ExtraInfoMla &info, uint32_t subNid, uint32_t subNSize)
{
    uint64_t dStride = headDim;
    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
        dStride = headDim * kvHeadNum;
    }
    auto srcGm = keyGm[info.tensorBOffset + subNid * 64 * dStride];
    CopyGmToL1(bL1Tensor, srcGm, subNSize, headDim, dStride);
}

template <typename IFAT>
__aicore__ inline void IfaMatmulKvNd<IFAT>::CopyInMm1BRopeToL1(LocalTensor<KV_T>& bL1Tensor,
                                                                                    const ExtraInfoMla &info, uint32_t subNid, uint32_t subNSize)
{
    uint64_t dStride = headDimRope;
    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
        dStride = headDimRope * kvHeadNum;
    }
    auto srcGm = kRopeGm[info.tensorBRopeOffset + 64 * subNid * dStride];
    CopyGmToL1(bL1Tensor, srcGm, subNSize, headDimRope, dStride);
}

template <typename IFAT>
__aicore__ inline void IfaMatmulKvNd<IFAT>::CopyInMm1BToL1ForPA(
    LocalTensor<KV_T>& bL1Tensor, const uint64_t keyGmBaseOffset, uint32_t copyTotalRowCntAlign,
    uint32_t copyStartRowCnt, uint32_t nActCopyRowCount)
{
    if constexpr (KV_LAYOUT_T == LAYOUT::NZ) {
        uint32_t blockElementCnt = 32 / sizeof(KV_T);
        if constexpr (KVINT4) {
            blockElementCnt = 64;
        }

        DataCopyParams intriParams;
        intriParams.blockLen = nActCopyRowCount;
        intriParams.blockCount = headDim / blockElementCnt;
        intriParams.dstStride = copyTotalRowCntAlign - nActCopyRowCount;
        intriParams.srcStride = kvCacheBlockSize - nActCopyRowCount;
        DataCopy(bL1Tensor[copyStartRowCnt * blockElementCnt], keyGm[keyGmBaseOffset], intriParams);
    } else {
        uint64_t dStride = headDim;
        if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
            dStride = headDim * kvHeadNum;
        }

        uint32_t blockElementCnt = 32 / sizeof(KV_T);
        if constexpr (KVINT4) {
            blockElementCnt = 64;
        }

        Nd2NzParams mm1Nd2NzParamsForB;
        mm1Nd2NzParamsForB.ndNum = 1;
        mm1Nd2NzParamsForB.nValue = nActCopyRowCount;
        if constexpr (KVINT4) {
            mm1Nd2NzParamsForB.dValue = headDim / 2;
            mm1Nd2NzParamsForB.srcDValue = dStride / 2;
        } else {
            mm1Nd2NzParamsForB.dValue = headDim;
            mm1Nd2NzParamsForB.srcDValue = dStride;
        }
        mm1Nd2NzParamsForB.dstNzC0Stride = copyTotalRowCntAlign;
        mm1Nd2NzParamsForB.dstNzNStride = 1;
        mm1Nd2NzParamsForB.srcNdMatrixStride = 0;
        mm1Nd2NzParamsForB.dstNzMatrixStride = 0;
        DataCopy(bL1Tensor[copyStartRowCnt * blockElementCnt], keyGm[keyGmBaseOffset], mm1Nd2NzParamsForB);
    }
}

template <typename IFAT>
__aicore__ inline void IfaMatmulKvNd<IFAT>::CopyInMm1BRopeToL1ForPA(
    LocalTensor<KV_T>& bL1Tensor, const uint64_t kRopeGmBaseOffset, uint32_t copyTotalRowCntAlign,
    uint32_t copyStartRowCnt, uint32_t nActCopyRowCount)
{
    if constexpr (KV_LAYOUT_T == LAYOUT::NZ) {
        uint32_t blockElementCnt = 32 / sizeof(KV_T);
        if constexpr (KVINT4) {
            blockElementCnt = 64;
        }

        DataCopyParams intriParams;
        intriParams.blockLen = nActCopyRowCount;
        intriParams.blockCount = headDimRope / blockElementCnt;
        intriParams.dstStride = copyTotalRowCntAlign - nActCopyRowCount;
        intriParams.srcStride = kvCacheBlockSize - nActCopyRowCount;
        DataCopy(bL1Tensor[copyStartRowCnt * blockElementCnt], kRopeGm[kRopeGmBaseOffset], intriParams);
    } else {
        uint64_t dStride = headDimRope;
        if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
            dStride = headDimRope * kvHeadNum;
        }

        uint32_t blockElementCnt = 32 / sizeof(KV_T);
        if constexpr (KVINT4) {
            blockElementCnt = 64;
        }

        Nd2NzParams mm1Nd2NzParamsForB;
        mm1Nd2NzParamsForB.ndNum = 1;
        mm1Nd2NzParamsForB.nValue = nActCopyRowCount;
        if constexpr (KVINT4) {
            mm1Nd2NzParamsForB.dValue = headDimRope / 2;
            mm1Nd2NzParamsForB.srcDValue = dStride / 2;
        } else {
            mm1Nd2NzParamsForB.dValue = headDimRope;
            mm1Nd2NzParamsForB.srcDValue = dStride;
        }
        mm1Nd2NzParamsForB.dstNzC0Stride = copyTotalRowCntAlign;
        mm1Nd2NzParamsForB.dstNzNStride = 1;
        mm1Nd2NzParamsForB.srcNdMatrixStride = 0;
        mm1Nd2NzParamsForB.dstNzMatrixStride = 0;
        DataCopy(bL1Tensor[copyStartRowCnt * blockElementCnt], kRopeGm[kRopeGmBaseOffset], mm1Nd2NzParamsForB);
    }
}

template <typename IFAT>
__aicore__ inline void IfaMatmulKvNd<IFAT>::LoadDataMm1A(LocalTensor<KV_T>& aL0Tensor,
    LocalTensor<KV_T>& aL1Tensor, uint32_t idx, uint32_t kSplitSize, uint32_t mSize, uint32_t kSize)
{
    uint32_t mLoops = mSize / 16;
    LocalTensor<KV_T> srcTensor = aL1Tensor[mSize * kSplitSize * idx];

    for (uint32_t i = 0; i < mLoops; i++) {
        LoadData2DParams loadData2DParams;
        loadData2DParams.startIndex = 0;

        if constexpr (KVINT4) {
            loadData2DParams.repeatTimes = kSize / 64;
        } else {
            loadData2DParams.repeatTimes = kSize / (32 / sizeof(KV_T));
        }
        loadData2DParams.srcStride = mSize / 16;
        loadData2DParams.dstGap = 0;
        loadData2DParams.ifTranspose = false;

        LocalTensor<KV_T> tmpSrcTensor;
        if constexpr (KVINT4) {
            tmpSrcTensor = srcTensor[i * 16 * 64];
        } else {
            tmpSrcTensor = srcTensor[i * 16 * (32 / sizeof(KV_T))];
        }
        LoadData(aL0Tensor[i * 16 * kSize], tmpSrcTensor, loadData2DParams);
    }
}

template <typename IFAT>
__aicore__ inline void IfaMatmulKvNd<IFAT>::LoadDataMm1B(LocalTensor<KV_T> &l0Tensor,
    LocalTensor<KV_T> &l1Tensor, uint32_t idx, uint32_t kSplitSize, uint32_t kSize, uint32_t nSize)
{
    // N 方向全载
    LocalTensor<KV_T> srcTensor = l1Tensor[nSize * kSplitSize * idx];

    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    if constexpr (KVINT4) {
        loadData2DParams.repeatTimes = (nSize + 15) / 16 * kSize / 64;
    } else {
        loadData2DParams.repeatTimes = (nSize + 15) / 16 * kSize / (32 / sizeof(KV_T));
    }

    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(l0Tensor, srcTensor, loadData2DParams);
}

template <typename IFAT>
__aicore__ inline void IfaMatmulKvNd<IFAT>::CopyInMm2AToL1(LocalTensor<KV_T> &aL1Tensor,
                                                                                const ExtraInfoMla &info)
{
    // 全量拷贝
    auto srcGm = vec1ResGm[(info.loop % PRE_LOAD_NUM_MLA) * mmResUbSize];
    CopyGmToL1(aL1Tensor, srcGm, info.gSize * info.s1Size, info.actualSingleProcessSInnerSize,
               info.actualSingleProcessSInnerSizeAlign);
}

template <typename IFAT>
__aicore__ inline void IfaMatmulKvNd<IFAT>::CopyInMm2BToL1(LocalTensor<KV_T> &bL1Tensor, const ExtraInfoMla &info,
    uint32_t subKid, uint32_t kSplitSize, uint32_t subNid, uint32_t nSplitSize, uint32_t subKSize, uint32_t subNSize)
{
    uint64_t dStride = headDim;
    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
        dStride = headDim * kvHeadNum;
    }
    // 拷贝 128 * 256
    auto srcGm = keyGm[info.tensorBOffset + subKid * kSplitSize * dStride + subNid * nSplitSize];
    CopyGmToL1(bL1Tensor, srcGm, subKSize, subNSize, dStride);
}

template <typename IFAT>
__aicore__ inline void IfaMatmulKvNd<IFAT>::CopyInMm2BToL1ForPA(LocalTensor<KV_T>& bL1Tensor,
    const uint64_t valueGmBaseOffset, uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt, uint32_t nActCopyRowCount,
    uint32_t copyStartColumnCount, uint32_t copyColumnCount)
{
    if constexpr (KV_LAYOUT_T == LAYOUT::NZ) {
        // copyStartColumnCount和copyColumnCount都需要blockElementCnt对齐
        uint32_t blockElementCnt = 32 / sizeof(KV_T);
        if constexpr (KVINT4) {
            blockElementCnt = 64;
        }

        DataCopyParams intriParams;
        intriParams.blockLen = nActCopyRowCount;
        intriParams.blockCount = copyColumnCount / blockElementCnt;
        intriParams.dstStride = copyTotalRowCntAlign - nActCopyRowCount;
        intriParams.srcStride = kvCacheBlockSize - nActCopyRowCount;
        DataCopy(bL1Tensor[copyStartRowCnt * blockElementCnt], valueGm[valueGmBaseOffset + copyStartColumnCount * kvCacheBlockSize], intriParams);
    } else {
        uint64_t step = headDim;
        if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
            step = headDim * kvHeadNum;
        }

        uint32_t blockElementCnt = 32 / sizeof(KV_T);
        if constexpr (KVINT4) {
            blockElementCnt = 64;
        }

        Nd2NzParams mm1Nd2NzParamsForB;
        mm1Nd2NzParamsForB.ndNum = 1;
        mm1Nd2NzParamsForB.nValue = nActCopyRowCount;
        if constexpr (KVINT4) {
            mm1Nd2NzParamsForB.dValue = copyColumnCount / 2;
            mm1Nd2NzParamsForB.srcDValue = step / 2;
        } else {
            mm1Nd2NzParamsForB.dValue = copyColumnCount;
            mm1Nd2NzParamsForB.srcDValue = step;
        }
        mm1Nd2NzParamsForB.dstNzC0Stride = copyTotalRowCntAlign;
        mm1Nd2NzParamsForB.dstNzNStride = 1;
        mm1Nd2NzParamsForB.srcNdMatrixStride = 0;
        mm1Nd2NzParamsForB.dstNzMatrixStride = 0;
        DataCopy(bL1Tensor[copyStartRowCnt * blockElementCnt], valueGm[valueGmBaseOffset + copyStartColumnCount], mm1Nd2NzParamsForB);
    }
}


template <typename IFAT>
__aicore__ inline void IfaMatmulKvNd<IFAT>::LoadDataMm2A(LocalTensor<KV_T> &aL0Tensor,
                                                                              LocalTensor<KV_T> &aL1Tensor, uint32_t mSize, uint32_t subKidx, uint32_t subKSize)
{
    // 加载 64 * 128
    uint32_t mLoops = mSize / 16;

    uint32_t kSplitSize = 128;

    LocalTensor<KV_T> aL1SrcTensor = aL1Tensor[mSize * kSplitSize * subKidx];

    for (uint32_t i = 0; i < mLoops; i++) {
        LoadData2DParams loadData2DParams;
        loadData2DParams.startIndex = 0;

        if constexpr (KVINT4) {
            loadData2DParams.repeatTimes = subKSize / 64;
        } else {
            loadData2DParams.repeatTimes = subKSize / (32 / sizeof(KV_T));
        }
        loadData2DParams.srcStride = mSize / 16;
        loadData2DParams.dstGap = 0;
        loadData2DParams.ifTranspose = false;

        LocalTensor<KV_T> tmpSrcTensor;
        if constexpr (KVINT4) {
            tmpSrcTensor = aL1SrcTensor[i * 16 * 64];
        } else {
            tmpSrcTensor = aL1SrcTensor[i * 16 * (32 / sizeof(KV_T))];
        }
        LoadData(aL0Tensor[i * 16 * subKSize], tmpSrcTensor, loadData2DParams);
    }
}

template <typename IFAT>
__aicore__ inline void IfaMatmulKvNd<IFAT>::LoadDataMm2B(LocalTensor<KV_T> &bL0Tensor,
                                                                              LocalTensor<KV_T> &bL1Tensor,
                                                                              uint32_t idx, uint32_t nSize, uint32_t kSize)
{
    // L1 128 * 256; L0 64 * 128
    uint32_t nSplitSize = 128;
    uint32_t kloops = kSize / 16;

    LocalTensor<KV_T> srcTensor = bL1Tensor[kSize * nSplitSize * idx];
    for (uint32_t i = 0; i < kloops; i++) {
        LoadData2DParams loadData2DParams;
        loadData2DParams.startIndex = 0;
        if constexpr (KVINT4) {
            loadData2DParams.repeatTimes = nSize / 64;
        } else {
            loadData2DParams.repeatTimes = nSize / (32 / sizeof(KV_T));
        }

        loadData2DParams.srcStride = kSize / 16;
        loadData2DParams.dstGap = 0;
        loadData2DParams.ifTranspose = true;

        LocalTensor<KV_T> tmpSrcTensor;
        if constexpr (KVINT4) {
            tmpSrcTensor = srcTensor[i * 16 * 64];
        } else {
            tmpSrcTensor = srcTensor[i * 16 * (32 / sizeof(KV_T))];
        }
        LoadData(bL0Tensor[i * 16 * nSize], tmpSrcTensor, loadData2DParams);

    }
}

template <typename IFAT>
__aicore__ inline void IfaMatmulKvNd<IFAT>::ComputeMm1(const ExtraInfoMla &info)
{
    uint32_t mSizeAct = info.gSize * info.s1Size;
    uint32_t mSize = Align(mSizeAct, 16U);
    LocalTensor<Q_T> qTensor = qL1Tensor[(qL1BufIter % 2) * L1_Q_SIZE / sizeof(Q_T)];
    LocalTensor<Q_T> qRopeTensor = qTensor[mSize * BlockAlign<Q_T>(headDim)];

    WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT0 + qL1BufIter % 2);
    CopyInMm1AToL1(qTensor, info);
    CopyInMm1ARopeToL1(qRopeTensor, info);
    SetFlag<HardEvent::MTE2_MTE1>(Q_EVENT0 + qL1BufIter % 2);
    WaitFlag<HardEvent::MTE2_MTE1>(Q_EVENT0 + qL1BufIter % 2);

    const uint32_t nSplitSize = 64; // n方向切分
    uint32_t nloops = (info.actualSingleProcessSInnerSize + nSplitSize - 1) / nSplitSize;
    uint32_t nTail = info.actualSingleProcessSInnerSize - (nloops - 1) * nSplitSize;
    uint32_t subNSize = nSplitSize;
    uint32_t subNSizeAct = nSplitSize;

    for (uint32_t n = 0; n < nloops; n++) {
        if (n == nloops - 1) {
            subNSizeAct = nTail;
            subNSize = Align(nTail, 16U);
        }

        WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + (kvL1BufIter % 2));
        LocalTensor<KV_T> kTensor = kvL1Tensor[(kvL1BufIter % 2) * (L1_KV_SIZE / sizeof(KV_T))];
        LocalTensor<KV_T> kRopeTensor = kTensor[subNSize * BlockAlign<KV_T>(headDim)];

        if constexpr (PAGE_ATTENTION) {
            uint64_t blockTableBaseOffset = info.bIdx * maxBlockNumPerBatch;
            uint32_t curSeqIdx = info.s2BatchOffset + n * nSplitSize;
            uint32_t copyFinishRowCnt = 0;
            while (copyFinishRowCnt < subNSizeAct) {
                uint64_t blockIdOffset = curSeqIdx / kvCacheBlockSize; // 获取blcok table上的索引
                uint64_t reaminRowCnt = curSeqIdx % kvCacheBlockSize;  // 获取在单个块上超出的行数
                uint64_t idInBlockTable =
                    blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset); // 从block table上的获取编号
                // 计算可以拷贝行数
                uint32_t copyRowCnt = kvCacheBlockSize - reaminRowCnt;
                if (copyFinishRowCnt + copyRowCnt > subNSizeAct) {
                    copyRowCnt = subNSizeAct - copyFinishRowCnt;
                }
                uint64_t keyOffset = idInBlockTable * kvCacheBlockSize * headDim * kvHeadNum;
                uint64_t kRopeOffset = idInBlockTable * kvCacheBlockSize * headDimRope * kvHeadNum;
                if constexpr (KV_LAYOUT_T == LAYOUT::NZ) {
                    uint32_t blockElementCnt = 32 / sizeof(KV_T);
                    if constexpr (KVINT4) {
                        blockElementCnt = 64;
                    }
                    keyOffset += (uint64_t)(info.n2Idx * headDim * kvCacheBlockSize) + reaminRowCnt * blockElementCnt;
                    kRopeOffset += (uint64_t)(info.n2Idx * headDimRope * kvCacheBlockSize) + reaminRowCnt * blockElementCnt;
                } else {
                    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
                        keyOffset += (uint64_t)(info.n2Idx * headDim) + reaminRowCnt * headDim * kvHeadNum;
                        kRopeOffset += (uint64_t)(info.n2Idx * headDimRope) + reaminRowCnt * headDimRope * kvHeadNum;
                    } else {
                        keyOffset += (uint64_t)(info.n2Idx * headDim * kvCacheBlockSize) + reaminRowCnt * headDim;
                        kRopeOffset += (uint64_t)(info.n2Idx * headDimRope * kvCacheBlockSize) + reaminRowCnt * headDimRope;
                    }
                }
                CopyInMm1BToL1ForPA(kTensor, keyOffset, subNSize, copyFinishRowCnt, copyRowCnt);
                CopyInMm1BRopeToL1ForPA(kRopeTensor, kRopeOffset, subNSize, copyFinishRowCnt, copyRowCnt);

                // 更新循环变量
                copyFinishRowCnt += copyRowCnt;
                curSeqIdx += copyRowCnt;
            }
        } else {
            CopyInMm1BToL1(kTensor, info, n, subNSizeAct); // 拷贝 64 * 512
            CopyInMm1BRopeToL1(kRopeTensor, info, n, subNSizeAct); // 拷贝 64 * 64
        }
        SetFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + (kvL1BufIter % 2));
        WaitFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + (kvL1BufIter % 2));

        uint32_t kSplitSize = (mSize <= 64) ? 256 : 128;
        uint32_t kSize = BlockAlign<KV_T>(headDim) + BlockAlign<KV_T>(headDimRope);
        uint32_t kLoops = (kSize + kSplitSize - 1) / kSplitSize;
        uint32_t subKSize = kSplitSize;

        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + cL0BufIter % 2);
        LocalTensor cL0Tensor = cL0TensorPingPong[(cL0BufIter % 2) * (L0C_PP_SIZE / sizeof(MM_OUT_T))];

        for (uint32_t k = 0; k < kLoops; k++) {
            if (k + 1 == kLoops) {
                subKSize = kSize - (kLoops - 1) * kSplitSize;
            }

            WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0 + aL0BufIter % 2);
            LocalTensor<KV_T> aL0Tensor = aL0TensorPingPong[(aL0BufIter % 2) * (L0A_PP_SIZE / sizeof(KV_T))];

            LoadDataMm1A(aL0Tensor, qTensor, k, kSplitSize, mSize, subKSize);

            SetFlag<HardEvent::MTE1_M>(L0A_EVENT0 + aL0BufIter % 2);
            WaitFlag<HardEvent::MTE1_M>(L0A_EVENT0 + aL0BufIter % 2);

            WaitFlag<HardEvent::M_MTE1>(L0B_EVENT0 + bL0BufIter % 2);
            LocalTensor<KV_T> bL0Tensor = bL0TensorPingPong[(bL0BufIter % 2) * (L0B_PP_SIZE / sizeof(KV_T))];

            LoadDataMm1B(bL0Tensor, kTensor, k, kSplitSize, subKSize, subNSize);

            SetFlag<HardEvent::MTE1_M>(L0B_EVENT0 + bL0BufIter % 2);
            WaitFlag<HardEvent::MTE1_M>(L0B_EVENT0 + bL0BufIter % 2);

            MmadParams mmadParams;
            mmadParams.m = mSizeAct;
            mmadParams.n = subNSize;
            mmadParams.k = subKSize;
            mmadParams.cmatrixInitVal = (k == 0);
            mmadParams.cmatrixSource = false;

            Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
            PipeBarrier<PIPE_M>();

            SetFlag<HardEvent::M_MTE1>(L0A_EVENT0 + aL0BufIter % 2);
            aL0BufIter++;

            SetFlag<HardEvent::M_MTE1>(L0B_EVENT0 + bL0BufIter % 2);
            bL0BufIter++;
        }

        SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + (kvL1BufIter % 2));
        kvL1BufIter++;

        SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + cL0BufIter % 2);
        WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + cL0BufIter % 2);

        FixpipeParamsV220 fixParams;
        fixParams.nSize = subNSize;
        fixParams.mSize = mSizeAct;
        fixParams.srcStride = mSize;
        fixParams.dstStride = info.actualSingleProcessSInnerSizeAlign; // mm1ResGm两行之间的间隔
        fixParams.ndNum = 1;
        Fixpipe(mm1ResGm[(info.loop % (PRE_LOAD_NUM_MLA)) * mmResUbSize + n * nSplitSize], cL0Tensor, fixParams);

        SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + cL0BufIter % 2);
        cL0BufIter++;
    }

    SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT0 + qL1BufIter % 2);
    qL1BufIter++;
}

template <typename IFAT>
__aicore__ inline void IfaMatmulKvNd<IFAT>::ComputeMm2(const ExtraInfoMla &info)
{
    uint32_t mSizeAct = info.gSize * info.s1Size;
    uint32_t mSize = Align(mSizeAct, 16U);
    uint32_t nSize = BlockAlign<KV_T>(headDim);

    LocalTensor<KV_T> pTensor = pL1Tensor[(qL1BufIter % 2) * L1_Q_SIZE / sizeof(KV_T)];

    WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT0 + qL1BufIter % 2);
    CopyInMm2AToL1(pTensor, info);
    SetFlag<HardEvent::MTE2_MTE1>(Q_EVENT0 + qL1BufIter % 2);
    WaitFlag<HardEvent::MTE2_MTE1>(Q_EVENT0 + qL1BufIter % 2);

    uint32_t nSplitSize = (mSize <= 64) ? 256 : 128;
    uint32_t nLoops = (nSize + nSplitSize - 1) / nSplitSize;
    uint32_t nTail = nSize - (nLoops - 1) * nSplitSize;
    uint32_t subNSize = nSplitSize;

    for (uint32_t n = 0; n < nLoops; n++) {
        if (n == nLoops - 1) {
            subNSize = nTail;
        }

        // 切K
        uint32_t kSplitSize = 128;
        uint32_t kloops = (info.actualSingleProcessSInnerSize + kSplitSize - 1) / kSplitSize;
        uint32_t kTail = info.actualSingleProcessSInnerSize - (kloops - 1) * kSplitSize;
        uint32_t subKSize = kSplitSize;
        uint32_t subKSizeAct = kSplitSize;

        // wait_flag l0c
        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + cL0BufIter % 2);
        LocalTensor<MM_OUT_T> cL0Tensor = cL0TensorPingPong[(cL0BufIter % 2) * (L0C_PP_SIZE / sizeof(MM_OUT_T))];

        for (uint32_t k = 0; k < kloops; k++) { // 128 循环
            if (k == kloops - 1) {
                subKSizeAct = kTail;
                subKSize = Align(kTail, 16U);
            }

            WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + (kvL1BufIter % 2));
            LocalTensor vTensor = kvL1Tensor[(kvL1BufIter % 2) * (L1_KV_SIZE / sizeof(KV_T))];

            if constexpr (PAGE_ATTENTION) {
                uint64_t blockTableBaseOffset = info.bIdx * maxBlockNumPerBatch;
                uint32_t curSeqIdx = info.s2BatchOffset + k * kSplitSize;
                uint32_t copyFinishRowCnt = 0;
                while (copyFinishRowCnt < subKSizeAct) {
                    uint64_t blockIdOffset = curSeqIdx / kvCacheBlockSize; // 获取blcok table上的索引
                    uint64_t reaminRowCnt = curSeqIdx % kvCacheBlockSize;  // 获取在单个块上超出的行数
                    uint64_t idInBlockTable =
                        blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset); // 从block table上的获取编号
                    // 计算可以拷贝行数
                    uint32_t copyRowCnt = kvCacheBlockSize - reaminRowCnt;
                    if (copyFinishRowCnt + copyRowCnt > subKSizeAct) {
                        copyRowCnt = subKSizeAct - copyFinishRowCnt;
                    }
                    uint64_t valueOffset = idInBlockTable * kvCacheBlockSize * headDim * kvHeadNum;
                    if constexpr (KV_LAYOUT_T == LAYOUT::NZ) {
                        uint32_t blockElementCnt = 32 / sizeof(KV_T);
                        if constexpr (KVINT4) {
                            blockElementCnt = 64;
                        }
                        valueOffset += (uint64_t)(info.n2Idx * headDim * kvCacheBlockSize) + reaminRowCnt * blockElementCnt;
                    } else {
                        if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
                            valueOffset += (uint64_t)(info.n2Idx * headDim) + reaminRowCnt * headDim * kvHeadNum;
                        } else {
                            valueOffset += (uint64_t)(info.n2Idx * headDim * kvCacheBlockSize) + reaminRowCnt * headDim;
                        }
                    }
                    CopyInMm2BToL1ForPA(vTensor, valueOffset, subKSize, copyFinishRowCnt, copyRowCnt, n * nSplitSize, subNSize);

                    // 更新循环变量
                    copyFinishRowCnt += copyRowCnt;
                    curSeqIdx += copyRowCnt;
                }
            } else {
                CopyInMm2BToL1(vTensor, info, k, kSplitSize, n, nSplitSize, subKSizeAct, subNSize); // 128 * 256
            }

            SetFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + (kvL1BufIter % 2));
            WaitFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + (kvL1BufIter % 2));

            WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0 + aL0BufIter % 2);
            LocalTensor<KV_T> aL0Tensor = aL0TensorPingPong[(aL0BufIter % 2) * (L0C_PP_SIZE / sizeof(MM_OUT_T))];

            LoadDataMm2A(aL0Tensor, pTensor, mSize, k, subKSize); // 64 * 128

            SetFlag<HardEvent::MTE1_M>(L0A_EVENT0 + aL0BufIter % 2);
            WaitFlag<HardEvent::MTE1_M>(L0A_EVENT0 + aL0BufIter % 2);

            uint32_t nL0Loops = nSplitSize / 128;
            for (uint32_t i = 0; i < nL0Loops; i++) { // nSplitSize(256) 按照 128 切分两份，这里需要处理尾块
                WaitFlag<HardEvent::M_MTE1>(L0B_EVENT0 + bL0BufIter % 2);
                LocalTensor<KV_T> bL0Tensor = bL0TensorPingPong[(bL0BufIter % 2) * (L0C_PP_SIZE / sizeof(MM_OUT_T))];

                LoadDataMm2B(bL0Tensor, vTensor, i, 128, subKSize);

                SetFlag<HardEvent::MTE1_M>(L0B_EVENT0 + bL0BufIter % 2);
                WaitFlag<HardEvent::MTE1_M>(L0B_EVENT0 + bL0BufIter % 2);

                MmadParams mmadParams;
                mmadParams.m = mSize;
                mmadParams.n = 128;
                mmadParams.k = subKSizeAct;
                mmadParams.cmatrixInitVal = (k == 0);
                mmadParams.cmatrixSource = false;

                LocalTensor<MM_OUT_T> destL0C = cL0Tensor[mSize * 128 * i]; // 64 * 128

                Mmad(destL0C, aL0Tensor, bL0Tensor, mmadParams);

                SetFlag<HardEvent::M_MTE1>(L0B_EVENT0 + bL0BufIter % 2);
                bL0BufIter++;
            }

            PipeBarrier<PIPE_M>();

            SetFlag<HardEvent::M_MTE1>(L0A_EVENT0 + aL0BufIter % 2);
            aL0BufIter++;

            SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + (kvL1BufIter % 2));
            kvL1BufIter++;
        }

        SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + cL0BufIter % 2);
        WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + cL0BufIter % 2);

        FixpipeParamsV220 fixParams;
        fixParams.nSize = subNSize; // 实现切片大小
        fixParams.mSize = mSizeAct; // msdIterNum * gSize; // 有效数据不足16行，只需要输出部分行即可
        fixParams.srcStride = mSize; // ((fixParams.mSize + 15) / 16) * 16
        fixParams.dstStride = headDim; // headdimAlign mm2ResGm两行之间的间隔
        fixParams.ndNum = 1;

        Fixpipe(mm2ResGm[(info.loop % (PRE_LOAD_NUM_MLA)) * bmm2ResUbSize + nSplitSize * n], cL0Tensor, fixParams);

        SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + cL0BufIter % 2);
        cL0BufIter++;
    }

    SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT0 + qL1BufIter % 2);
    qL1BufIter++;
}

#endif
