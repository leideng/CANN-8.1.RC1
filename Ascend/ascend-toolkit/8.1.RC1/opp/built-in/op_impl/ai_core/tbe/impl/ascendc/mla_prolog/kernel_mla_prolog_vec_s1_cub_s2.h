/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License. 
 */

/*!
 * \file kernel_mla_prolog_vec_s1_cub_s2.h
 * \brief
 */

#ifndef KERNEL_MLA_PROLOG_VEC_S1_CUB_S2_H
#define KERNEL_MLA_PROLOG_VEC_S1_CUB_S2_H

#include "mla_prolog_comm.h"
#include "service_matmul.h"
#include "service_rms_norm.h"
#include "service_gather_sin_cos.h"
#include "service_rotary_position_embedding.h"
#include "service_scatter_cache.h"
#include "service_dequant.h"

namespace MlaProlog {

constexpr auto GetMmCfgNorm(int sm, int sn, int sk, int bm, int bn, int bk) {
    auto CFG = CFG_NORM;
    CFG.singleCoreM = sm;
    CFG.singleCoreN = sn;
    CFG.singleCoreK = sk;
    CFG.basicM = bm;
    CFG.basicN = bn;
    CFG.basicK = bk;
    CFG.enUnitFlag = false;

    CFG.enableSetBias = false;
    CFG.enableQuantVector = false;
    CFG.enableSetDefineData = false;
    return CFG; 
}

constexpr auto GetMmCfgMdl(int sm, int sn, int sk, int bm, int bn, int bk) {
    auto CFG = CFG_MDL;
    CFG.singleCoreM = sm;
    CFG.singleCoreN = sn;
    CFG.singleCoreK = sk;
    CFG.basicM = bm;
    CFG.basicN = bn;
    CFG.basicK = bk;
    CFG.enUnitFlag = false;

    CFG.enableSetBias = false;
    CFG.enableQuantVector = false;
    CFG.enableSetDefineData = false;
    return CFG;
}

constexpr auto GetMmCfgMdlUniflag(int sm, int sn, int sk, int bm, int bn, int bk) {
    auto CFG = CFG_MDL;
    CFG.singleCoreM = sm;
    CFG.singleCoreN = sn;
    CFG.singleCoreK = sk;
    CFG.basicM = bm;
    CFG.basicN = bn;
    CFG.basicK = bk;
    CFG.enUnitFlag = true;

    CFG.enableSetBias = false;
    CFG.enableQuantVector = false;
    CFG.enableSetDefineData = false;
    return CFG;
}

constexpr static auto GetMMCqCkvKrTiling(const MatmulApiStaticTiling& mmTiling) {
    auto tiling = mmTiling;
    tiling.stepM = 1;
    tiling.stepN = 1;
    tiling.stepKa = 4;
    tiling.stepKb = 4;
    tiling.depthA1 = 8;
    tiling.depthB1 = 8;
    return tiling;
}

constexpr static auto GetMMQcQrTiling(const MatmulApiStaticTiling& mmTiling) {
    auto tiling = mmTiling;
    tiling.stepM = 1;
    tiling.stepN = 1;
    tiling.stepKa = 24;
    tiling.stepKb = 4;
    tiling.depthA1 = 24;
    tiling.depthB1 = 8;
    return tiling;
}

constexpr static auto GetMMQnTiling(const MatmulApiStaticTiling& mmTiling) {
    auto tiling = mmTiling;
    tiling.stepM = 1;
    tiling.stepN = 2;
    tiling.stepKa = 1;
    tiling.stepKb = 1;
    tiling.depthA1 = 1;
    tiling.depthB1 = 4;
    return tiling;
}

template <typename MLAPT>
class MlaPrologVecS1CubS2 {
public:
    __aicore__ inline MlaPrologVecS1CubS2(TPipe* pipe, const MlaPrologTilingData* __restrict tilingData,
                                          const MlaPrologBaseParams* __restrict baseParams)
        : pipe_(pipe), tilingData_(tilingData), baseParams_(baseParams) {}

    __aicore__ inline void Init(__gm__ uint8_t *tokenX, __gm__ uint8_t *weightDq, __gm__ uint8_t *weightUqQr,
                                __gm__ uint8_t *weightUk, __gm__ uint8_t *weightDkvKr,
                                __gm__ uint8_t *rmsnormGammaCq, __gm__ uint8_t *rmsnormGammaCkv,
                                __gm__ uint8_t *ropeSin, __gm__ uint8_t *ropeCos,
                                __gm__ uint8_t *cacheIndex, __gm__ uint8_t *kvCache, __gm__ uint8_t *krCache,
                                __gm__ uint8_t *dequantScaleX, __gm__ uint8_t *dequantScaleWDq,
                                __gm__ uint8_t *deqScaleQcQrW, __gm__ uint8_t * dequantScaleWDkvkr,
                                __gm__ uint8_t *quantScaleCkv, __gm__ uint8_t *quantScaleCkr,__gm__ uint8_t *smoothScaleCq,
                                __gm__ uint8_t *queryOut, __gm__ uint8_t *queryRopeOut, __gm__ uint8_t *workspace);
    __aicore__ inline void Process();
    __aicore__ inline void MMParamInit();
    __aicore__ inline void UpdateStepBatchParams(int64_t curStepBatchSize);
private:

    __aicore__ inline void MatmulCq(int64_t tokenXOffset, int64_t weightDqOffset, int64_t cqResOffset);
    __aicore__ inline void MatmulCkvKr(int64_t tokenXOffset, int64_t weightDkvKrOffset, int64_t ckvKrResOffset);
    __aicore__ inline void MatmulQcQr(int64_t weightUqQrOffset,int64_t qcQrResOffset);
    __aicore__ inline void MatmulQcQrWightPreload(int64_t weightUqQrOffset);
    __aicore__ inline void MatmulQnWightPreload(int64_t weightUkOffset, int64_t mmQnLoopTime);
    __aicore__ inline void MatmulQn(int64_t qcOffset, int64_t weightUkOffset, int64_t qnResOffset, int64_t mmQnLoopTime);
    __aicore__ inline void GetSinCos(int64_t tokenIndex, int64_t curVecToken);
    __aicore__ inline void RmsNormCq(int64_t rmsNormCqOffset, int64_t curVecToken);
    __aicore__ inline void RmsNormRopeScatterCkvKr(int64_t tokenIndex, int64_t rmsNormCkvOffset, int64_t ropeKrOffset, int64_t curVecToken);
    __aicore__ inline void RopeQr(int64_t ropeQrOffset, int64_t ropeQrResOffset, int64_t curVecToken);
    __aicore__ inline void MatmulQnPreDequant(int64_t mmQnPreDequantOffset, int64_t mmQnPreDequantResOffset, int64_t curVecToken);
    __aicore__ inline void ComputeAiVOffset(int64_t &curVecToken, int64_t &curBlockTokenOffset, int64_t &rmsNormCqOffset, int64_t &rmsNormCkvOffset, 
                                        int64_t &ropeKrOffset, int64_t &mmQnPreDequantOffset, int64_t &mmQnPreDequantResOffset, int64_t &ropeQrOffset, 
                                        int64_t &ropeQrResOffset, int64_t batchOffset);

public:
    using mmInputType = typename MLAPT::mmInputType;
    using mmQcQrInputType = typename MLAPT::mmQcQrInputType;
    using mmQnInputType = typename MLAPT::mmQnInputType;
    using mmCqOutputType = typename MLAPT::mmCqOutputType;
    using mmCkvKrOutputType = typename MLAPT::mmCkvKrOutputType;
    using mmQcQrOutputType = typename MLAPT::mmQcQrOutputType;
    using mmQnOutputType = typename MLAPT::mmQnOutputType;
    using rmsNormGammaType = typename MLAPT::rmsNormGammaType;
    using rmsNormComputType = typename MLAPT::rmsNormComputType;
    using rmsNormCqOutputType = typename MLAPT::rmsNormCqOutputType;
    using rmsNormCkvOutputType = typename MLAPT::rmsNormCkvOutputType;
    using ropeSinCosType = typename MLAPT::ropeSinCosType;
    using ropeComputType = typename MLAPT::ropeComputType;
    using ropeOutputType = typename MLAPT::ropeOutputType;
    using cacheType = typename MLAPT::cacheType;

#ifdef USE_MM_API_MLAP
    typedef MatmulType<TPosition::A1, CubeFormat::NZ, mmInputType, true> aCqCkvKrType;
    typedef MatmulType<TPosition::B1, CubeFormat::NZ, mmInputType, true> bCqCkvKrType;
    typedef MatmulType<TPosition::GM, CubeFormat::ND, mmCqOutputType> cCqCkvKrType;
    typedef MatmulType<TPosition::GM, CubeFormat::ND, mmInputType> biasCqCkvKrType;
    constexpr static auto CFGCqCkvKr = GetMmCfgMdl(128, 128, 256, 128, 128, 128);
    constexpr static auto mmCqCkvKrTiling = GetMatmulApiTiling<aCqCkvKrType, bCqCkvKrType, cCqCkvKrType, biasCqCkvKrType>(CFGCqCkvKr);
    constexpr static auto mmCqCkvKrTilingUpdate = GetMMCqCkvKrTiling(mmCqCkvKrTiling);
    using mmCqCkvKrType = MatmulImpl<aCqCkvKrType, bCqCkvKrType, cCqCkvKrType, biasCqCkvKrType, mmCqCkvKrTilingUpdate>; // 各个matmul需要调整配置

    typedef MatmulType<TPosition::A1, CubeFormat::NZ, mmQcQrInputType, true> aQcQrType;
    typedef MatmulType<TPosition::B1, CubeFormat::NZ, mmQcQrInputType, true> bQcQrType;
    typedef MatmulType<TPosition::GM, CubeFormat::ND, mmQcQrOutputType> cQcQrType;
    typedef MatmulType<TPosition::GM, CubeFormat::ND, mmQcQrInputType> biasQcQrType;
    constexpr static auto CFGQcQr = GetMmCfgMdl(128, 1024, 1536, 128, 256, 128);
    constexpr static auto mmQcQrTiling = GetMatmulApiTiling<aQcQrType, bQcQrType, cQcQrType, biasQcQrType>(CFGQcQr);
    constexpr static auto mmQcQrTilingUpdate = GetMMQcQrTiling(mmQcQrTiling);
    using mmQcQrType = MatmulImpl<aQcQrType, bQcQrType, cQcQrType, biasQcQrType, mmQcQrTilingUpdate>;

    typedef MatmulType<TPosition::A1, CubeFormat::NZ, mmQnInputType, true> aQnType;
    typedef MatmulType<TPosition::B1, CubeFormat::NZ, mmQnInputType, true> bQnType;
    typedef MatmulType<TPosition::GM, CubeFormat::ND, mmQnOutputType> cQnType;
    typedef MatmulType<TPosition::GM, CubeFormat::ND, mmQnInputType> biasQnType;
    constexpr static auto CFGQn = GetMmCfgMdlUniflag(128, 512, 128, 128, 128, 128);
    constexpr static auto mmQnTiling = GetMatmulApiTiling<aQnType, bQnType, cQnType, biasQnType>(CFGQn);
    constexpr static auto mmQnTilingUpdate = GetMMQnTiling(mmQnTiling);
    using mmQnType = MatmulImpl<aQnType, bQnType, cQnType, biasQnType, mmQnTilingUpdate>;

    mmCqCkvKrType mmCqCkvKr_;
    mmQcQrType mmQcQr_;
    mmQnType mmQn_;
#else
    MMBufParams<mmInputType, float> mmBuf_{};
#endif

    MMParams mmCqParam_;
    MMParams mmCkvKrParam_;
    MMParams mmQcQrParam_;
    MMParams mmQnParam_;

private:
    TPipe* pipe_;
    const MlaPrologTilingData* __restrict tilingData_;
    const MlaPrologBaseParams* __restrict baseParams_;
    uint32_t blockIdx_ = 0U;
    int64_t vectorRow_ = 1;
    int64_t curVectorBlockNum_;
    int64_t vectorCoreNum_;
    uint32_t curStepVecFrontToken_;
    uint32_t curStepVecFrontListNum_;
    uint32_t curStepVecBackToken_;
    uint32_t curVecTokenMax_;

    // GM
    GlobalTensor<mmInputType> tokenXGm_;
    GlobalTensor<mmInputType> weightDqGm_;
    GlobalTensor<mmQcQrInputType> weightUqQrGm_;
    GlobalTensor<mmQnInputType> weightUkGm_;
    GlobalTensor<mmInputType> weightDkvKrGm_;
    GlobalTensor<rmsNormGammaType> rmsnormGammaCqGm_;
    GlobalTensor<rmsNormGammaType> rmsnormGammaCkvGm_;
    GlobalTensor<ropeSinCosType> ropeSinGm_;
    GlobalTensor<ropeSinCosType> ropeCosGm_;
    GlobalTensor<int64_t> cacheIndexGm_;
    GlobalTensor<cacheType> kvCacheGm_;
    GlobalTensor<cacheType> krCacheGm_;
    GlobalTensor<ropeOutputType> qrOutGm_;

    GlobalTensor<float> smoothScaleCqGm_;
    GlobalTensor<float> deqScaleQcQrW_; // per-channel反量化参数
    GlobalTensor<float> quantScaleCkvGm_;
    GlobalTensor<float> quantScaleCkrGm_;

    GlobalTensor<rmsNormCqOutputType> rmsNormCqResGm_;
    GlobalTensor<mmCqOutputType> mmCqResGm_;
    GlobalTensor<mmCkvKrOutputType> mmCkvKrResGm_;
    GlobalTensor<mmQcQrOutputType> mmQcQrResGm_;
    GlobalTensor<mmQnInputType> mmQcQrResDequantGm_;
    GlobalTensor<mmQnOutputType> mmQnResGm_;  // 直接输出到outGm
    
    float deQuantScaleCq_[8] = {0};  // aiv为2时，每个vector处理token数的最大值

    // UB
    TQue<QuePosition::VECOUT, 1> rmsNormCqOutQueue_;
    TQue<QuePosition::VECOUT, 1> ropeQrOutQueue_;
    TQue<QuePosition::VECOUT, 1> scatterOutputQueue_;   

    TBuf<TPosition::VECCALC> sincosBuffer_;
    TBuf<TPosition::VECCALC> shareBuffer_;

    LocalTensor<ropeComputType> cosLocal_;
    LocalTensor<ropeComputType> sinLocal_;
#ifdef USE_MM_API_MLAP
    TBuf<TPosition::A1> aBufL1_;
    TBuf<TPosition::B1> bBufL1_;
    LocalTensor<mmInputType> aL1Tensor_;
    LocalTensor<mmInputType> bL1Tensor_;
    MMBufParams bufParam_;
#else
    TBuf<TPosition::A1> aBufL1_;
    TBuf<TPosition::A1> bBufL1_;
    TBuf<TPosition::A2> tmpBufL0A_;
    TBuf<TPosition::B2> tmpBufL0B_;
    TBuf<TPosition::CO1> tmpBufL0C_;
#endif
};

template<typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::Init(__gm__ uint8_t *tokenX, __gm__ uint8_t *weightDq,
                                                        __gm__ uint8_t *weightUqQr, __gm__ uint8_t *weightUk,
                                                        __gm__ uint8_t *weightDkvKr, __gm__ uint8_t *rmsnormGammaCq,
                                                        __gm__ uint8_t *rmsnormGammaCkv, __gm__ uint8_t *ropeSin,
                                                        __gm__ uint8_t *ropeCos, __gm__ uint8_t *cacheIndex,
                                                        __gm__ uint8_t *kvCache, __gm__ uint8_t *krCache,
                                                        __gm__ uint8_t *dequantScaleX, __gm__ uint8_t *dequantScaleWDq,
                                                        __gm__ uint8_t *deqScaleQcQrW, __gm__ uint8_t *dequantScaleWDkvkr,
                                                        __gm__ uint8_t *quantScaleCkv, __gm__ uint8_t *quantScaleCkr, __gm__ uint8_t *smoothScaleCq,
                                                        __gm__ uint8_t *queryOut, __gm__ uint8_t *queryRopeOut, __gm__ uint8_t *workspace) {

    blockIdx_ = GetBlockIdx(); // cube:0-23  vec:0-47
    curVectorBlockNum_ = int64_t(baseParams_->stepBatchSize);
    vectorCoreNum_ = int64_t(baseParams_->vectorBlockNum); // aivNum 48
    curVecTokenMax_ = (curVectorBlockNum_ + vectorCoreNum_ - 1) / vectorCoreNum_;

    MMParamInit();

    // GM
    tokenXGm_.SetGlobalBuffer((__gm__ mmInputType *)tokenX);
    weightDqGm_.SetGlobalBuffer((__gm__ mmInputType *)weightDq);   // NZ
    weightUqQrGm_.SetGlobalBuffer((__gm__ mmQcQrInputType *)weightUqQr);   // NZ
    weightUkGm_.SetGlobalBuffer((__gm__ mmQnInputType *)weightUk);
    weightDkvKrGm_.SetGlobalBuffer((__gm__ mmInputType *)weightDkvKr);  // NZ
    rmsnormGammaCqGm_.SetGlobalBuffer((__gm__ rmsNormGammaType *)rmsnormGammaCq);
    rmsnormGammaCkvGm_.SetGlobalBuffer((__gm__ rmsNormGammaType *)rmsnormGammaCkv);
    ropeSinGm_.SetGlobalBuffer((__gm__ ropeSinCosType *)ropeSin);
    ropeCosGm_.SetGlobalBuffer((__gm__ ropeSinCosType *)ropeCos);
    cacheIndexGm_.SetGlobalBuffer((__gm__ int64_t *)cacheIndex);
    kvCacheGm_.SetGlobalBuffer((__gm__ cacheType *)kvCache);
    krCacheGm_.SetGlobalBuffer((__gm__ cacheType *)krCache);
    mmQnResGm_.SetGlobalBuffer((__gm__ mmQnOutputType *)queryOut);
    qrOutGm_.SetGlobalBuffer((__gm__ ropeOutputType *)queryRopeOut);
    if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
        smoothScaleCqGm_.SetGlobalBuffer((__gm__ float *)smoothScaleCq);
        deqScaleQcQrW_.SetGlobalBuffer((__gm__ float *)deqScaleQcQrW);
        quantScaleCkvGm_.SetGlobalBuffer((__gm__ float *)quantScaleCkv);
        quantScaleCkrGm_.SetGlobalBuffer((__gm__ float *)quantScaleCkr);
    }
    // workspace
    int64_t workspaceOffset = 0;
    mmCqResGm_.SetGlobalBuffer((__gm__ mmCqOutputType *)workspace);         // bf16   32 * 1536
    if constexpr (std::is_same<rmsNormCqOutputType, int8_t>::value) {
        workspaceOffset += int64_t(baseParams_->stepBatchSize) * int64_t(baseParams_->headSizeCq) * sizeof(mmCqOutputType);
    }
    rmsNormCqResGm_.SetGlobalBuffer(
        (__gm__ rmsNormCqOutputType *)workspace + workspaceOffset);
    workspaceOffset += int64_t(baseParams_->stepBatchSize) * int64_t(baseParams_->headSizeCq) * sizeof(rmsNormCqOutputType);
    mmCkvKrResGm_.SetGlobalBuffer((__gm__ mmCkvKrOutputType *)(workspace + workspaceOffset));

    workspaceOffset += int64_t(baseParams_->stepBatchSize) *
                       int64_t(baseParams_->headSizeCkv + baseParams_->dimHeadRope) *
                       sizeof(mmCkvKrOutputType);
    mmQcQrResGm_.SetGlobalBuffer((__gm__ mmQcQrOutputType *)(workspace + workspaceOffset));     // bf16   32 * 32 * (128+64)
    if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
        workspaceOffset += int64_t(baseParams_->stepBatchSize) *
                           int64_t(baseParams_->headSizeQc + baseParams_->headSizeQr) * sizeof(mmQcQrOutputType);
    }
    mmQcQrResDequantGm_.SetGlobalBuffer((__gm__ mmQnInputType *)(workspace + workspaceOffset));

    if ASCEND_IS_AIV {
        pipe_->InitBuffer(rmsNormCqOutQueue_, 1, 
            vectorRow_ * baseParams_->headSizeCq * sizeof(rmsNormCqOutputType));   // [1, 1536] bf16
        pipe_->InitBuffer(ropeQrOutQueue_, 1,
            baseParams_->headSizeQr * sizeof(ropeOutputType)); // [128, 64] bf16
        pipe_->InitBuffer(scatterOutputQueue_, 1,
                vectorRow_ * baseParams_->headSizeCkv * sizeof(rmsNormCkvOutputType)); // [1, 512] bf16

        pipe_->InitBuffer(sincosBuffer_, 2 * baseParams_->dimHeadRope * sizeof(ropeComputType) * curVecTokenMax_); // [2, 64] float
        pipe_->InitBuffer(shareBuffer_, baseParams_->ubBufferSize); // 除 sincos 外的 sharebuffer 共享，不会同时使用

        cosLocal_ = sincosBuffer_.Get<ropeComputType>();
        sinLocal_ = cosLocal_[baseParams_->dimHeadRope * curVecTokenMax_];
    } else {
#ifdef USE_MM_API_MLAP
        pipe_->InitBuffer(aBufL1_, L1_A_SIZE * 2);
        pipe_->InitBuffer(bBufL1_, L1_B_SIZE * 2);
        mmCqCkvKr_.SetSubBlockIdx(0);
        mmCqCkvKr_.Init(&tilingData_->bmm1TilingData, pipe_);
        mmQcQr_.SetSubBlockIdx(0);
        mmQcQr_.Init(&tilingData_->bmm3TilingData, pipe_);
        mmQn_.SetSubBlockIdx(0);
        mmQn_.Init(&tilingData_->bmm4TilingData, pipe_);

        SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0);
        SetFlag<HardEvent::MTE1_MTE2>(A_EVENT1);
        SetFlag<HardEvent::MTE1_MTE2>(B_EVENT0);
        SetFlag<HardEvent::MTE1_MTE2>(B_EVENT1);
        aL1Tensor_ = aBufL1_.Get<mmInputType>();
        bL1Tensor_ = bBufL1_.Get<mmInputType>();
        bufParam_.aL1BufAddr = aBufL1_.GetBufferAddr(aL1Tensor_.GetBufferHandle());
        bufParam_.bL1BufAddr = bBufL1_.GetBufferAddr(bL1Tensor_.GetBufferHandle());;
#else
        pipe_->InitBuffer(aBufL1_, L1_A_SIZE * 2);
        pipe_->InitBuffer(bBufL1_, L1_B_SIZE * 2);
        pipe_->InitBuffer(tmpBufL0A_, L0A_PP_SIZE * 2); // 64K
        pipe_->InitBuffer(tmpBufL0B_, L0B_PP_SIZE * 2); // 64K
        pipe_->InitBuffer(tmpBufL0C_, L0C_PP_SIZE * 2); // 128K
        SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0);
        SetFlag<HardEvent::MTE1_MTE2>(A_EVENT1);
        SetFlag<HardEvent::MTE1_MTE2>(B_EVENT0);
        SetFlag<HardEvent::MTE1_MTE2>(B_EVENT1);

        SetFlag<HardEvent::M_MTE1>(L0A_EVENT0);
        SetFlag<HardEvent::M_MTE1>(L0A_EVENT1);
        SetFlag<HardEvent::M_MTE1>(L0B_EVENT0);
        SetFlag<HardEvent::M_MTE1>(L0B_EVENT1);

        SetFlag<HardEvent::FIX_M>(L0C_EVENT0);
        SetFlag<HardEvent::FIX_M>(L0C_EVENT1);

        mmBuf_.aL1Tensor = aBufL1_.Get<mmInputType>();
        mmBuf_.bL1Tensor = bBufL1_.Get<mmInputType>();
        mmBuf_.aL0TensorPingPong = tmpBufL0A_.Get<mmInputType>();
        mmBuf_.bL0TensorPingPong = tmpBufL0B_.Get<mmInputType>();
        mmBuf_.cL0TensorPingPong = tmpBufL0C_.Get<float>();
#endif
    }
}

template<typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MMParamInit() {
#ifndef USE_MM_API_MLAP
    mmBuf_.aL1BufIter = 0;
    mmBuf_.bL1BufIter = 0;
    mmBuf_.aL0BufIter = 0;
    mmBuf_.bL0BufIter = 0;
    mmBuf_.cL0BufIter = 0;

    mmCqParam_.kL1SplitSize = 512;
    mmCqParam_.kSplitSize = 256;

    mmCkvKrParam_.kL1SplitSize = 512;
    mmCkvKrParam_.kSplitSize = 256;

    if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
        mmQcQrParam_.kL1SplitSize = 512;
        mmQcQrParam_.kSplitSize = 128;
    } else {
        mmQcQrParam_.kL1SplitSize = 256;
        mmQcQrParam_.kSplitSize = 64;
    }

    mmQnParam_.nSplitSize = 128;
#endif
    mmCqParam_.m = baseParams_->stepBatchSize; // 32
    if (blockIdx_ == baseParams_->mm1BlockNum - 1) {
        mmCqParam_.n = baseParams_->headSizeCq - baseParams_->mm1SingleCoreN * blockIdx_;
    } else {
        mmCqParam_.n = baseParams_->mm1SingleCoreN; // 1536 / 24 = 64
    }
    mmCqParam_.k = baseParams_->headSizeX; // 7168
    mmCqParam_.needSetOrgShape = 1;
    mmCqParam_.orgM = mmCqParam_.m;
    mmCqParam_.orgN = mmCqParam_.n;
    mmCqParam_.orgKa = mmCqParam_.k;
    mmCqParam_.orgKb = mmCqParam_.k;
    mmCqParam_.orgKc = baseParams_->headSizeCq;  // 1536
    mmCqParam_.orgM = mmCqParam_.m;
    mmCqParam_.baseK = 128;
    mmCqParam_.baseN = 128;
    mmCqParam_.stepK = 4;

    mmCkvKrParam_.m = baseParams_->stepBatchSize; // 32
    if (blockIdx_ == baseParams_->mm2BlockNum - 1) {
        mmCkvKrParam_.n = baseParams_->headSizeCkv + baseParams_->headSizeKr - baseParams_->mm2SingleCoreN * blockIdx_;
    } else {
        mmCkvKrParam_.n = baseParams_->mm2SingleCoreN;
    }
    mmCkvKrParam_.k = baseParams_->headSizeX; // 7168
    mmCkvKrParam_.needSetOrgShape = 1;
    mmCkvKrParam_.orgM = mmCkvKrParam_.m;
    mmCkvKrParam_.orgN = mmCkvKrParam_.n;
    mmCkvKrParam_.orgKa = mmCkvKrParam_.k;
    mmCkvKrParam_.orgKb = mmCkvKrParam_.k;
    mmCkvKrParam_.orgKc = (baseParams_->headSizeCkv + baseParams_->dimHeadRope);  // 576
    mmCkvKrParam_.baseK = 128;
    mmCkvKrParam_.baseN = 128;
    mmCkvKrParam_.stepK = 4;

    mmQcQrParam_.m = baseParams_->stepBatchSize; // 32
    if (blockIdx_ == baseParams_->mm3BlockNum - 1) {
        mmQcQrParam_.n = baseParams_->headSizeQc + baseParams_->headSizeQr - baseParams_->mm3SingleCoreN * blockIdx_;
    } else {
        mmQcQrParam_.n = baseParams_->mm3SingleCoreN;
    }
    mmQcQrParam_.k = baseParams_->headSizeCq; // 1536
    mmQcQrParam_.needSetOrgShape = 1;
    mmQcQrParam_.orgM = mmQcQrParam_.m;
    mmQcQrParam_.orgN = mmQcQrParam_.n;
    mmQcQrParam_.orgKa = mmQcQrParam_.k;
    mmQcQrParam_.orgKb = mmQcQrParam_.k;
    mmQcQrParam_.orgKc = (baseParams_->headSizeQc + baseParams_->headSizeQr); // (128 * 32 + 64 * 32)
    mmQcQrParam_.baseK = (sizeof(mmQcQrInputType) == sizeof(int8_t)) ? 128 : 64;
    mmQcQrParam_.baseN = 256;
    mmQcQrParam_.stepK = 4;

    mmQnParam_.m = baseParams_->stepBatchSize; // 32
    mmQnParam_.n = baseParams_->headSizeCkv; // 512
    mmQnParam_.k = baseParams_->dimHeadSizeQc; // 128, 这里numHeadSize被分核，matmul设置里不体现
    mmQnParam_.needSetOrgShape = 1;
    mmQnParam_.orgM = mmQnParam_.m;
    mmQnParam_.orgN = mmQnParam_.n;
    if constexpr (std::is_same<mmQcQrOutputType, int32_t>::value) {
        mmQnParam_.orgKa = baseParams_->headSizeQc;
    } else {
        mmQnParam_.orgKa = baseParams_->headSizeQc + baseParams_->headSizeQr;
    }
    mmQnParam_.orgKb = baseParams_->dimHeadSizeQc;
    mmQnParam_.orgKc = baseParams_->headSizeCkv * baseParams_->numHeadSize;
}

template<typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::UpdateStepBatchParams(int64_t curStepBatchSize) {
    mmCqParam_.m = curStepBatchSize;
    mmCkvKrParam_.m = curStepBatchSize;
    mmQcQrParam_.m = curStepBatchSize;
    mmQnParam_.m = curStepBatchSize;
    curVectorBlockNum_ = curStepBatchSize;
}

// MlaProg算子计算流程
// token_x ──> MatmulCq ──> RmsNorm(Cq) ──> MatmulQcQr ──> MatmulQn ──> query_out
//        │                                           └──> Rope(Qr) ──> query_rope_out
//        └──> MatmulCkvKr ──> RmsNorm(Ckv) ──> Scatter(Ckv) ──> kv_cache_out
//                        └──> Rope(Kr) ──> Scatter(Kr) ──> kr_cache_out
template<typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::Process() {

    // AIC的offset参数
    int64_t weightDqOffset = int64_t(baseParams_->headSizeX) *
                             baseParams_->mm1SingleCoreN * blockIdx_;  // 7168 * 64 * idx
    int64_t cqResOffset = baseParams_->mm1SingleCoreN * blockIdx_;  // 64 * idx

    int64_t weightDkvKrOffset = int64_t(baseParams_->headSizeX) *
                                baseParams_->mm2SingleCoreN * blockIdx_;  // 7168 * (512 + 64) / 9 * idx = 7168 * 64 * idx
    int64_t ckvKrResOffset = baseParams_->mm2SingleCoreN * blockIdx_;  //  (512 + 64) / 9 * idx  = 64 * idx

    int64_t weightUqQrOffset = int64_t(baseParams_->headSizeCq) *
                               baseParams_->mm3SingleCoreN * blockIdx_;  // 1536 * (32 * 128 + 32 * 64) / 24 * idx = 1536 * 256 * idx
    int64_t qcQrResOffset = baseParams_->mm3SingleCoreN * blockIdx_;  // (32 * 128 + 32 * 64) / 24 * idx = 256 * idx

    int64_t qcOffset;
    int64_t weightUkOffset;
    int64_t qnResOffset;   // BS, N, Dq
    int64_t mmQnLoopTime;
    if (blockIdx_ < baseParams_->mm4BlockNum) {
        if (blockIdx_ == baseParams_->mm4BlockNum - 1) {
            mmQnLoopTime = baseParams_->numHeadSize - baseParams_->mm4SingleCoreBatch * blockIdx_;
        } else {
            mmQnLoopTime = baseParams_->mm4SingleCoreBatch;
        }
        int64_t numHeadOffset = blockIdx_ * baseParams_->mm4SingleCoreBatch;
        if constexpr (std::is_same<mmQcQrOutputType, int32_t>::value) {
            qcOffset = int64_t(baseParams_->dimHeadSizeQc) * numHeadOffset; // 128 * idx
        } else {
            qcOffset = int64_t(baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope) * numHeadOffset; // (128 + 64) * idx
        }
        weightUkOffset = int64_t(baseParams_->dimHeadSizeQc) * int64_t(baseParams_->headSizeCkv) * numHeadOffset;   // (128 * 512) * idx
        qnResOffset = int64_t(baseParams_->headSizeCkv) * numHeadOffset;
    }

    // AIV的offset参数
    int64_t curVecToken = blockIdx_ < curStepVecFrontListNum_ ? curStepVecFrontToken_ : curStepVecBackToken_;
    int64_t curBlockTokenOffset = blockIdx_ < curStepVecFrontListNum_ ? blockIdx_ * curVecToken : blockIdx_ * curVecToken + curStepVecFrontListNum_;

    int64_t rmsNormCqOffset;
    int64_t rmsNormCkvOffset;
    int64_t ropeKrOffset;
    int64_t mmQnPreDequantOffset;
    int64_t mmQnPreDequantResOffset;
    int64_t ropeQrOffset;
    int64_t ropeQrResOffset;
    ComputeAiVOffset(curVecToken, curBlockTokenOffset, rmsNormCqOffset, rmsNormCkvOffset, ropeKrOffset, mmQnPreDequantOffset, mmQnPreDequantResOffset, ropeQrOffset, ropeQrResOffset, 0);
    int64_t bsSize = int64_t(baseParams_->tokenSize);
    // 需要考虑BS合轴的尾块情况
    for (int64_t batchOffset = 0; batchOffset < bsSize; batchOffset += baseParams_->stepBatchSize) {
        if (bsSize - batchOffset < baseParams_->stepBatchSize) {
            UpdateStepBatchParams(bsSize - batchOffset); // 320 - 256
            if ((bsSize - batchOffset) != baseParams_->stepBatchSize) {
                ComputeAiVOffset(curVecToken, curBlockTokenOffset, rmsNormCqOffset, rmsNormCkvOffset, ropeKrOffset, mmQnPreDequantOffset, mmQnPreDequantResOffset, ropeQrOffset, ropeQrResOffset, batchOffset);
            }            
        }
        if ASCEND_IS_AIC {
            int64_t tokenXOffset = batchOffset * baseParams_->headSizeX;
            MatmulCq(tokenXOffset, weightDqOffset, cqResOffset);
            CrossCoreSetFlag<0x2, PIPE_FIX>(0x6);
            MatmulCkvKr(tokenXOffset, weightDkvKrOffset, ckvKrResOffset);
            MatmulQcQrWightPreload(weightUqQrOffset);
            CrossCoreSetFlag<0x2, PIPE_FIX>(0x7);
            // wait RmsNormCq
            // MatmulQcQr依赖RmsNormCq的输出，需要插入CV核间同步
            CrossCoreWaitFlag(0x6);
            MatmulQcQr(weightUqQrOffset, qcQrResOffset);
            MatmulQnWightPreload(weightUkOffset, mmQnLoopTime);
            CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);

            if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
                // CV 核间同步， wait all mmQnPreDequant at vec
                CrossCoreWaitFlag(0x6);
            } else {
                // wait all of MatmulQcQr at diff cube
                // 由于 MatmulQn 和 MatmulQcQr的分核策略不一样，MatmulQn又依赖MatmulQcQr的输出
                // 需要等所有cube核上的MatmulQcQr执行完后才能启动MatmulQn
                CrossCoreSetFlag<0x0, PIPE_FIX>(0x9);
                CrossCoreWaitFlag(0x9);
            }
            MatmulQn(qcOffset, weightUkOffset, qnResOffset, mmQnLoopTime);
            // MatmulQn的结果直接输出到 queryOut, qnOffset需要按Batch轴偏移
            qnResOffset += int64_t(baseParams_->stepBatchSize) * int64_t(baseParams_->headSizeCkv) *
                           int64_t(baseParams_->numHeadSize);
        }
        if ASCEND_IS_AIV {
            int64_t tokenIndex = batchOffset + curBlockTokenOffset;
            GetSinCos(tokenIndex, curVecToken);

            // wait MatmulCq
            CrossCoreWaitFlag(0x6);
            // wait all vector
            CrossCoreSetFlag<0x0, PIPE_MTE3>(0x9);
            CrossCoreWaitFlag(0x9);
            RmsNormCq(rmsNormCqOffset, curVecToken);
            // wait all cube
            // 由于RmsNormCq和MatmulQcQr的分核策略不一样，需要等所有vector上的RmsNormCq执行完成后才能启动MatmulQcQr
            // 需要所有vector核上的RmsNormCq执行完成后，才发起MatmulQcQr的执行
            CrossCoreSetFlag<0x0, PIPE_MTE3>(0x9);
            CrossCoreWaitFlag(0x9);
            // 保障MatmulQcQr等RmsNormCq
            CrossCoreSetFlag<0x2, PIPE_MTE3>(0x6);

            // wait MatmulCkvKr
            CrossCoreWaitFlag(0x7);
            // wait all vector
            CrossCoreSetFlag<0x0, PIPE_MTE3>(0x9);
            CrossCoreWaitFlag(0x9);
            RmsNormRopeScatterCkvKr(tokenIndex, rmsNormCkvOffset, ropeKrOffset, curVecToken);

            // wait MatmulQcQr
            CrossCoreWaitFlag(0x8);
            // wait all vector
            CrossCoreSetFlag<0x0, PIPE_MTE3>(0x9);
            CrossCoreWaitFlag(0x9);
            if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
                MatmulQnPreDequant(mmQnPreDequantOffset, mmQnPreDequantResOffset, curVecToken);
                // 确保所有的 vector 核上的 MatmulQnPreDequant 执行完，再启动MatmulQn
                // wait all vector
                CrossCoreSetFlag<0x0, PIPE_MTE3>(0x9);
                CrossCoreWaitFlag(0x9);
                // 确保 MatmulQn等MatmulQnPreDequant
                CrossCoreSetFlag<0x2, PIPE_MTE3>(0x6);
            }

            RopeQr(ropeQrOffset, ropeQrResOffset, curVecToken);
            ropeQrResOffset += int64_t(baseParams_->stepBatchSize) * int64_t(baseParams_->headSizeQr);
        }
    }
    if ASCEND_IS_AIC {
#ifdef USE_MM_API_MLAP
        WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT0);
        WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT1);
        WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT0);
        WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT1);
#else
        WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT0);
        WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT1);
        WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT0);
        WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT1);

        WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0);
        WaitFlag<HardEvent::M_MTE1>(L0A_EVENT1);
        WaitFlag<HardEvent::M_MTE1>(L0B_EVENT0);
        WaitFlag<HardEvent::M_MTE1>(L0B_EVENT1);

        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0);
        WaitFlag<HardEvent::FIX_M>(L0C_EVENT1);
#endif
    }
}

template<typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::ComputeAiVOffset(int64_t &curVecToken, int64_t &curBlockTokenOffset, int64_t &rmsNormCqOffset, int64_t &rmsNormCkvOffset, 
                                        int64_t &ropeKrOffset, int64_t &mmQnPreDequantOffset, int64_t &mmQnPreDequantResOffset, int64_t &ropeQrOffset, 
                                        int64_t &ropeQrResOffset, int64_t batchOffset) {
    curStepVecBackToken_ = curVectorBlockNum_ / uint32_t(vectorCoreNum_);
    curStepVecFrontListNum_ = curVectorBlockNum_ % uint32_t(vectorCoreNum_);
    curStepVecFrontToken_ = curStepVecFrontListNum_ == 0 ? curStepVecBackToken_ : curStepVecBackToken_ + 1;

    curVecToken = blockIdx_ < curStepVecFrontListNum_ ? curStepVecFrontToken_ : curStepVecBackToken_;
    curBlockTokenOffset = blockIdx_ < curStepVecFrontListNum_ ? blockIdx_ * curVecToken : blockIdx_ * curVecToken + curStepVecFrontListNum_;
    rmsNormCqOffset = baseParams_->headSizeCq * curBlockTokenOffset;  //  1536 * idx
    rmsNormCkvOffset = (baseParams_->headSizeCkv + baseParams_->dimHeadRope) * curBlockTokenOffset;  // (512 + 64) * idx
    ropeKrOffset = baseParams_->headSizeCkv + (baseParams_->headSizeCkv + baseParams_->dimHeadRope) * curBlockTokenOffset;  // 512 + (512 + 64) * idx
    mmQnPreDequantOffset = (baseParams_->headSizeQc + baseParams_->headSizeQr) * curBlockTokenOffset;
    mmQnPreDequantResOffset = baseParams_->headSizeQc * curBlockTokenOffset;
    ropeQrOffset = baseParams_->dimHeadSizeQc + int64_t(baseParams_->numHeadSize) *
                    int64_t(baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope) * curBlockTokenOffset; // 128 + 32 * (128 + 64) * idx
    ropeQrResOffset = int64_t(baseParams_->headSizeQr) * (curBlockTokenOffset + batchOffset); //32 * 64 * idx;    // 按BS合轴切分step
}

template<typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MatmulCq(int64_t tokenXOffset, int64_t weightDqOffset,
                                                            int64_t cqResOffset) {
    if (blockIdx_ >= baseParams_->mm1BlockNum) {
        return;
    }
    // MatmulCq ──> RmsNorm(Cq)
    // [32, 7168] * [7168, 1536] = [32, 1536]
#ifdef USE_MM_API_MLAP
    MatmulImplNormal<mmInputType, mmCqOutputType, mmCqCkvKrType>(mmCqCkvKr_, tokenXGm_[tokenXOffset],
           weightDqGm_[weightDqOffset], mmCqResGm_[cqResOffset], mmCqParam_, &bufParam_, blockIdx_);
#else
    KKMatmul<mmInputType, mmCqOutputType>(tokenXGm_[tokenXOffset],
           weightDqGm_[weightDqOffset], mmCqResGm_[cqResOffset], mmCqParam_, &mmBuf_);
#endif
}

template<typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MatmulCkvKr(int64_t tokenXOffset, int64_t weightDkvKrOffset,
                                                               int64_t ckvKrResOffset) {
    if (blockIdx_ >= baseParams_->mm2BlockNum) {
        return;
    }
    // MatmulCkvKr ──> RmsNorm(Ckv)
    //            └──> Rope(Kr)
    // [32, 7168] * [7168, 512+64] = [32, 576]
#ifdef USE_MM_API_MLAP
    MatmulImplNormal<mmInputType, mmCkvKrOutputType, mmCqCkvKrType>(mmCqCkvKr_, tokenXGm_[tokenXOffset],
        weightDkvKrGm_[weightDkvKrOffset], mmCkvKrResGm_[ckvKrResOffset], mmCkvKrParam_, &bufParam_);
#else
    KKMatmul<mmInputType, mmCkvKrOutputType>(tokenXGm_[tokenXOffset],
        weightDkvKrGm_[weightDkvKrOffset], mmCkvKrResGm_[ckvKrResOffset], mmCkvKrParam_, &mmBuf_);
#endif
}

template<typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MatmulQcQrWightPreload(int64_t weightUqQrOffset) {
    if (blockIdx_ >= baseParams_->mm3BlockNum) {
        return;
    }
    QcQrWeightPreload<rmsNormCqOutputType, mmQcQrOutputType, mmQcQrType>(weightUqQrGm_[weightUqQrOffset], mmQcQrParam_, &bufParam_);
}

template<typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MatmulQcQr(int64_t weightUqQrOffset, int64_t qcQrResOffset) {
    if (blockIdx_ >= baseParams_->mm3BlockNum) {
        return;
    }
    // RmsNorm(Cq) ──> MatmulQcQr ──> MatmulQn
    //                           └──> Rope(Qr)
    // [32, 1536] * [1536, 32*(128+64)] = [32, 32*192]
#ifdef USE_MM_API_MLAP
    constexpr uint32_t mSize = (sizeof(mmQcQrInputType) == sizeof(int8_t)) ? 64 : 32;
    bool isAFullLoad = (mmQcQrParam_.m <= mSize) ? true : false;
    if (isAFullLoad) {
        MatmulImplQcQr<rmsNormCqOutputType, mmQcQrOutputType, mmQcQrType>(mmQcQr_, rmsNormCqResGm_, // 复用mmCqResGm_ workspace
            weightUqQrGm_[weightUqQrOffset], mmQcQrResGm_[qcQrResOffset], mmQcQrParam_, &bufParam_); 
    } else {
        MatmulImplNormalPreload<rmsNormCqOutputType, mmQcQrOutputType, mmQcQrType>(mmQcQr_, rmsNormCqResGm_, // 复用mmCqResGm_ workspace
            weightUqQrGm_[weightUqQrOffset], mmQcQrResGm_[qcQrResOffset], mmQcQrParam_, &bufParam_);
    }
#else
    if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
        KKMatmulQuant<rmsNormCqOutputType, mmQcQrOutputType>(rmsNormCqResGm_, // 复用rmsNormCqResGm_ workspace
            weightUqQrGm_[weightUqQrOffset], mmQcQrResGm_[qcQrResOffset], mmQcQrParam_, &mmBuf_);

    } else {
        KKMatmul<rmsNormCqOutputType, mmQcQrOutputType>(rmsNormCqResGm_, // 复用rmsNormCqResGm_ workspace
            weightUqQrGm_[weightUqQrOffset], mmQcQrResGm_[qcQrResOffset], mmQcQrParam_, &mmBuf_);
    }
#endif
}

template<typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MatmulQnWightPreload(int64_t weightUkOffset, int64_t subLoopTimes) {
    if (blockIdx_ >= baseParams_->mm4BlockNum) {
        return;
    }
    int64_t weightOffset = weightUkOffset;
    for (int32_t i = 0; i < subLoopTimes; ++i) {
        if (i < 2) { // preload pingpong
            QnWeightPreload<mmQnInputType, mmQnOutputType, mmQnType>(weightUkGm_[weightOffset], mmQnParam_, &bufParam_);
            weightOffset += int64_t(baseParams_->dimHeadSizeQc) * int64_t(baseParams_->headSizeCkv);
        }
    }
    if (subLoopTimes == 1) {
        bufParam_.bL1BufIter--;
    }  
}

template<typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MatmulQn(int64_t qcOffset, int64_t weightUkOffset,
                                                            int64_t qnResOffset, int64_t subLoopTimes) {
    if (blockIdx_ >= baseParams_->mm4BlockNum) {
        return;
    }
    // MatmulQcQr ──> MatmulQn ──> query_out
    // [32, 128] * [128, 512] = [32, 512]
    // [32, 2, 128] * [2, 128, 512] = [32, 2, 512]
    for (int64_t i = 0; i < subLoopTimes; i++) {
#ifdef USE_MM_API_MLAP
        if (i < 2) {
            MatmulImplQn<mmQnInputType, mmQnOutputType, mmQnType>(mmQn_, mmQcQrResDequantGm_[qcOffset], // 复用workspace
                weightUkGm_[weightUkOffset], mmQnResGm_[qnResOffset], mmQnParam_, &bufParam_);
        } else {
            MatmulImplABFullLoad<mmQnInputType, mmQnOutputType, mmQnType>(mmQn_, mmQcQrResDequantGm_[qcOffset], // 复用workspace
                weightUkGm_[weightUkOffset], mmQnResGm_[qnResOffset], mmQnParam_, &bufParam_);
        }
#else
        NMatmul<mmQnInputType, mmQnOutputType>(mmQcQrResDequantGm_[qcOffset], // 复用workspace
            weightUkGm_[weightUkOffset], mmQnResGm_[qnResOffset], mmQnParam_, &mmBuf_);
#endif
        if constexpr (std::is_same<mmQcQrOutputType, int32_t>::value) {
            qcOffset += int64_t(baseParams_->dimHeadSizeQc);
        } else {
            qcOffset += int64_t(baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope);
        }
        weightUkOffset += int64_t(baseParams_->dimHeadSizeQc) * int64_t(baseParams_->headSizeCkv);
        qnResOffset += baseParams_->headSizeCkv;  
    }
}

template<typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::GetSinCos(int64_t tokenIndex, int64_t curVecToken) {
    if (blockIdx_ >= curVectorBlockNum_) {
        return;
    }
    LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();
    // [1, 64]
    GatherSinCos<ropeSinCosType, ropeComputType>(ropeCosGm_, ropeSinGm_, tokenIndex, curVecToken,
                 shareTmpUb, vectorRow_, baseParams_->dimHeadRope, cosLocal_, sinLocal_);
}

template<typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::RmsNormCq(int64_t rmsNormCqOffset, int64_t curVecToken) {
    if (blockIdx_ >= curVectorBlockNum_) {
        return;
    }
    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVecToken; curVecTokenIdx++) {
        // MatmulCq ──> RmsNorm(Cq) ──> MatmulQcQr
        float quantScale = 0;
        LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();
        LocalTensor<rmsNormCqOutputType> outputLocal = rmsNormCqOutQueue_.AllocTensor<rmsNormCqOutputType>();
        // [1, 1536]
        if constexpr (std::is_same<rmsNormCqOutputType, int8_t>::value) {
            RmsNormWithPostQuant<mmCqOutputType, rmsNormGammaType, rmsNormComputType, float, rmsNormCqOutputType>(
                mmCqResGm_[rmsNormCqOffset], rmsnormGammaCqGm_, baseParams_->reciprocalCq, baseParams_->epsilonCq, 1.0,
                vectorRow_, baseParams_->headSizeCq, shareTmpUb, outputLocal, smoothScaleCqGm_, quantScaleCkvGm_, deQuantScaleCq_[curVecTokenIdx], true);
        } else {
            RmsNorm<mmCqOutputType, rmsNormGammaType, rmsNormComputType, rmsNormCqOutputType>(
                mmCqResGm_[rmsNormCqOffset], rmsnormGammaCqGm_, baseParams_->reciprocalCq, baseParams_->epsilonCq, quantScale,
                vectorRow_, baseParams_->headSizeCq, shareTmpUb, outputLocal);
        }

        rmsNormCqOutQueue_.EnQue(outputLocal);
        outputLocal = rmsNormCqOutQueue_.DeQue<rmsNormCqOutputType>();
        // RmsNorm(Cq)的结果拷进mmCqResGm_中，用于MatmulQcQr的A矩阵
        DataCopy(rmsNormCqResGm_[rmsNormCqOffset], outputLocal, baseParams_->headSizeCq);
        rmsNormCqOutQueue_.FreeTensor(outputLocal);
        rmsNormCqOffset += baseParams_->headSizeCq;
    }
}

template<typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::RmsNormRopeScatterCkvKr(int64_t tokenIndex, 
                                                                           int64_t rmsNormCkvOffset,
                                                                           int64_t ropeKrOffset,
                                                                           int64_t curVecToken) {
    if (blockIdx_ >= curVectorBlockNum_) {
        return;
    }
    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVecToken; curVecTokenIdx++) {
        LocalTensor<rmsNormCkvOutputType> outputLocal = scatterOutputQueue_.AllocTensor<rmsNormCkvOutputType>();
        // RmsNorm(Ckv)
        // MatmulCkvKr ──> RmsNorm(Ckv) ──> Scatter(Ckv) 由于当前只处理1行数据，不需要stride
        float quantScale = 0;
        LocalTensor<uint8_t> rmsNormShareTmpUb = shareBuffer_.Get<uint8_t>();
        if constexpr (std::is_same<rmsNormCkvOutputType, int8_t>::value) {
            // bf16, bf16, float, bf16, int8
            RmsNormWithPostQuant<mmCkvKrOutputType, rmsNormGammaType, rmsNormComputType, ropeSinCosType, cacheType>(
                mmCkvKrResGm_[rmsNormCkvOffset], rmsnormGammaCkvGm_, baseParams_->reciprocalCkv, baseParams_->epsilonCkv,
                1.0, vectorRow_, baseParams_->headSizeCkv, rmsNormShareTmpUb, outputLocal, smoothScaleCqGm_, quantScaleCkvGm_, deQuantScaleCq_[curVecTokenIdx], false);
        } else {
            RmsNorm<mmCkvKrOutputType, rmsNormGammaType, rmsNormComputType, rmsNormCkvOutputType>(
                mmCkvKrResGm_[rmsNormCkvOffset], rmsnormGammaCkvGm_, baseParams_->reciprocalCkv, baseParams_->epsilonCkv,
                quantScale, vectorRow_, baseParams_->headSizeCkv, rmsNormShareTmpUb, outputLocal);
        }

        scatterOutputQueue_.EnQue(outputLocal);
        outputLocal = scatterOutputQueue_.DeQue<rmsNormCkvOutputType>();
        // Scatter(Ckv)
        // RmsNorm(Ckv) ──> Scatter(Ckv) ──> kv_cache_out
        if constexpr (MLAPT::cacheMode == CACHE_MODE_BNSD) {  // 非PA
            int64_t batchIndex = 0;   // TODO：需要明确非PA场景中 cache_index的含义
            int64_t tokenIndexPerBatch = 0;
            int64_t cacheLength = 0;
            ScatterCache<cacheType>(outputLocal, kvCacheGm_, cacheLength,
                batchIndex, tokenIndexPerBatch, vectorRow_, baseParams_->headSizeCkv);
        } else {  // PA场景， CACHE_MODE_PA_NZ 判断是否为NZ
            int64_t paTokenIndex = cacheIndexGm_(tokenIndex);
            ScatterCache<cacheType, (MLAPT::cacheMode == CACHE_MODE_PA_NZ)>(outputLocal,
                kvCacheGm_, baseParams_->blockSize, paTokenIndex, vectorRow_, baseParams_->headSizeCkv);
        }
        scatterOutputQueue_.FreeTensor(outputLocal);
    
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        outputLocal = scatterOutputQueue_.AllocTensor<rmsNormCkvOutputType>();
        // Rope(Kr)
        // MatmulCkvKr ──> Rope(Ckv) ──> Scatter(Kr) 
        LocalTensor<uint8_t> ropeShareTmpUb = shareBuffer_.Get<uint8_t>();
        int64_t stride = baseParams_->headSizeCkv + baseParams_->headSizeKr; // 512 + 64
        if constexpr (std::is_same<rmsNormCkvOutputType, int8_t>::value) {
            LocalTensor<ropeSinCosType> tmpOut = ropeShareTmpUb.ReinterpretCast<ropeSinCosType>();
            LocalTensor<uint8_t> sharedBuf = ropeShareTmpUb.ReinterpretCast<uint8_t>()[baseParams_->dimHeadRope * sizeof(ropeSinCosType)];
            RotaryPosEmb<mmCkvKrOutputType, ropeComputType, ropeSinCosType>(
                mmCkvKrResGm_[ropeKrOffset], cosLocal_[baseParams_->dimHeadRope * curVecTokenIdx], sinLocal_[baseParams_->dimHeadRope * curVecTokenIdx], stride, vectorRow_, baseParams_->dimHeadRope,
                sharedBuf, tmpOut, deqScaleQcQrW_, deQuantScaleCq_[curVecTokenIdx]);
            RopePostQuantPerChannel(tmpOut, quantScaleCkrGm_, stride, vectorRow_, baseParams_->dimHeadRope,
                                    sharedBuf, outputLocal);
        } else {
            RotaryPosEmb<mmCkvKrOutputType, ropeComputType, rmsNormCkvOutputType>(mmCkvKrResGm_[ropeKrOffset],
                cosLocal_[baseParams_->dimHeadRope * curVecTokenIdx], sinLocal_[baseParams_->dimHeadRope * curVecTokenIdx], stride, vectorRow_, baseParams_->dimHeadRope, ropeShareTmpUb, outputLocal);
        }

        scatterOutputQueue_.EnQue(outputLocal);
        outputLocal = scatterOutputQueue_.DeQue<rmsNormCkvOutputType>();
        // scatter(Kr)
        // Rope(Kr) ──> Scatter(Kr) ──> kr_cache_out
        if constexpr (MLAPT::cacheMode == CACHE_MODE_BNSD) {  // 非PA
            int64_t batchIndex = 0;   // TODO：需要明确非PA场景中 cache_index的含义
            int64_t tokenIndexPerBatch = 0;
            int64_t cacheLength = 0;
            ScatterCache<cacheType>(outputLocal, krCacheGm_, cacheLength,
                batchIndex, tokenIndexPerBatch, vectorRow_, baseParams_->dimHeadRope);
        } else {  // PA场景， CACHE_MODE_PA_NZ 判断是否为NZ
            int64_t paTokenIndex = cacheIndexGm_(tokenIndex);
            ScatterCache<cacheType, (MLAPT::cacheMode == CACHE_MODE_PA_NZ)>(outputLocal,
                krCacheGm_, baseParams_->blockSize, paTokenIndex, vectorRow_, baseParams_->dimHeadRope);
        }
        scatterOutputQueue_.FreeTensor(outputLocal);

        tokenIndex += 1;
        rmsNormCkvOffset += baseParams_->headSizeCkv + baseParams_->dimHeadRope;
        ropeKrOffset += baseParams_->headSizeCkv + baseParams_->dimHeadRope;
    }
}

template<typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::RopeQr(int64_t ropeQrOffset, int64_t ropeQrResOffset, int64_t curVecToken) {
    if (blockIdx_ >= curVectorBlockNum_) {
        return;
    }
    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVecToken; curVecTokenIdx++) {
        // MatmulQcQr ──> Rope(Qr) ──> query_rope_out
        LocalTensor<uint8_t> ropeShareTmpUb = shareBuffer_.Get<uint8_t>();
        LocalTensor<ropeOutputType> outputLocal = ropeQrOutQueue_.AllocTensor<ropeOutputType>();
        int64_t stride = baseParams_->dimHeadRope + baseParams_->dimHeadSizeQc;
        if constexpr (std::is_same<mmQcQrInputType, int8_t>::value) {
            RotaryPosEmb<mmQcQrOutputType, ropeComputType, ropeOutputType>(mmQcQrResGm_[ropeQrOffset],
                cosLocal_[baseParams_->dimHeadRope * curVecTokenIdx], sinLocal_[baseParams_->dimHeadRope * curVecTokenIdx], stride, baseParams_->numHeadSize,
                baseParams_->dimHeadRope, ropeShareTmpUb, outputLocal, deqScaleQcQrW_[baseParams_->dimHeadSizeQc], deQuantScaleCq_[curVecTokenIdx]);
        } else {
            RotaryPosEmb<mmQcQrOutputType, ropeComputType, ropeOutputType>(mmQcQrResGm_[ropeQrOffset],
                cosLocal_[baseParams_->dimHeadRope * curVecTokenIdx], sinLocal_[baseParams_->dimHeadRope * curVecTokenIdx], stride, baseParams_->numHeadSize,
                baseParams_->dimHeadRope, ropeShareTmpUb, outputLocal);
        }

        ropeQrOutQueue_.EnQue(outputLocal);
        outputLocal = ropeQrOutQueue_.DeQue<ropeOutputType>();
        // [32, 64]
        DataCopy(qrOutGm_[ropeQrResOffset], outputLocal, baseParams_->headSizeQr);
        ropeQrOutQueue_.FreeTensor(outputLocal);

        ropeQrOffset += int64_t(baseParams_->numHeadSize) * 
                            int64_t(baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope);
        ropeQrResOffset += int64_t(baseParams_->headSizeQr);
    }
}

template <typename MLAPT>
__aicore__ inline void MlaPrologVecS1CubS2<MLAPT>::MatmulQnPreDequant(int64_t mmQnPreDequantOffset,
                                                                      int64_t mmQnPreDequantResOffset,
                                                                      int64_t curVecToken)
{
    if (blockIdx_ >= curVectorBlockNum_) {
        return;
    }
    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVecToken; curVecTokenIdx++) {
        LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();
        Dequant(mmQcQrResGm_[mmQnPreDequantOffset], deqScaleQcQrW_, deQuantScaleCq_[curVecTokenIdx], baseParams_->stepNumHeadDequant,
            baseParams_->dimHeadSizeQc, baseParams_->numHeadSize, baseParams_->dimHeadRope,
            mmQcQrResDequantGm_[mmQnPreDequantResOffset], shareTmpUb);
        mmQnPreDequantOffset += baseParams_->headSizeQc + baseParams_->headSizeQr;
        mmQnPreDequantResOffset += baseParams_->headSizeQc;
    }
}

} // namespace MlaProlog

#endif // MLA_PROLOG_VEC_S1_CUB_S2_H