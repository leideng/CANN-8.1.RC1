/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_score_grad_s1s2_bn2.h
 * \brief
 */

#ifndef _FLASH_ATTENTION_SCORE_GRAD_S1S2_BN2_H_
#define _FLASH_ATTENTION_SCORE_GRAD_S1S2_BN2_H_

#include "lib/matmul_intf.h"
#include "kernel_operator.h"
#include "pse.h"
#include "dropmask.h"
#include "matmul_modules/policys/fag_custom_matmul_policy_s1s2_bn2.h"

using matmul::Matmul;
using matmul::MatmulType;

struct StaticVParams {
    // 静态VEC参数
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t singleM = 0;
    uint32_t singleMTail = 0;
    uint32_t singleN = 0;
    uint32_t singleNTail = 0;
    uint32_t s1OuterOuterNum = 0;
    uint32_t s2OuterOuterNum = 0;
};

struct StaticSFTParams {
    // 静态SFT参数
    uint32_t singleM = 0;
    uint32_t baseM = 0;
    uint32_t dInner = 0;
    uint32_t dInnerTail = 0;
};

struct StaticCParams {
    // 静态Bmm参数
    uint32_t baseM = 0;
    uint32_t baseN = 0;
};

struct DyncReal {
    // 动态参数
    uint32_t processM = 0;
    uint32_t vectorProcessM = 0;
    uint32_t processN = 0;
    // ping pong
    uint32_t vPingS1Inner = 0;
    uint32_t vPongS1Inner = 0;
    uint32_t vPingS2Inner = 0;
    uint32_t vPongS2Inner = 0;
    uint32_t vPingS2InnerAlign = 0;
    uint32_t vPongS2InnerAlign = 0;
    uint32_t vPingLoopS1 = 0;
    uint32_t vPongLoopS1 = 0;
    uint32_t vPingLoopS2 = 0;
    uint32_t vPongLoopS2 = 0;
    bool vPingPrefixCompressCanSimplify = false;
    bool vPongPrefixCompressCanSimplify = false;
    // SFT
    uint32_t SFTS1Inner = 0;
    uint32_t SFTS1Times = 0;
    uint32_t vS1Times = 0;
    uint32_t vS2Times = 0;
    uint32_t SFTS1InnerTail = 0;
    uint32_t vS1InnerTail = 0;
    uint32_t vS2InnerTail = 0;
    uint32_t loopS1 = 0;
    uint32_t loopS2 = 0;
    uint32_t SFTLoopS1 = 0;
    uint32_t mm1mm2OrgM = 0;
};

struct DyncLoop {
    // 动态循环
    uint32_t s1OuterInnerNum = 0;
    uint32_t s2OuterInnerNum = 0;
};

struct PingPongEmitInsn {
    // ping pong
    uint32_t s1Inner;
    uint32_t s2Inner;
    uint32_t s2InnerAlign;
    uint32_t vLoopS1;
    uint32_t vLoopS2;
    int64_t s1Index;
    int64_t s2Index;
};

struct PostIndices {
    int64_t bIndex = 0;
    int64_t n2Index = 0;
    int64_t gIndex = 0;
};

__aicore__ inline void DataCopyOutLocal(const __gm__ void *gm, const LocalTensor<int8_t> &co1Local,
                                   const void *dataCopyOutParams, const uint64_t tilingPtr, const uint64_t dataPtr)
{
    const DataCopyOutParams *param = reinterpret_cast<const DataCopyOutParams *>(dataCopyOutParams);
    uint64_t dstStride = tilingPtr * 16 / 8 - param->burstLen;
    FixpipeParams<float> fixpipeParams(param->cBurstNum, param->burstLen, param->srcStride,
                                       static_cast<uint32_t>(dstStride));

    if (param->enUnitFlag) {
        fixpipeParams.unitFlag = 3;
    }
    LocalTensor<float> tmpLocal = co1Local.template ReinterpretCast<float>();
    GlobalTensor<float> tmpGm;
    tmpGm.SetGlobalBuffer((__gm__ float *)(gm));
    Fixpipe(tmpGm, tmpLocal, fixpipeParams);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT = 2,
          const CubeFormat MM2_OUT_FORMAT = CubeFormat::ND, const bool POST = false, const bool L1CUSTOM = false>
class FlashAttentionScoreGradS1s2Bn2 {
public:
    __aicore__ inline FlashAttentionScoreGradS1s2Bn2(){};
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR pse_shift,
                                GM_ADDR drop_mask, GM_ADDR padding_mask, GM_ADDR atten_mask, GM_ADDR softmax_max,
                                GM_ADDR softmax_sum, GM_ADDR prefixN, GM_ADDR softmax_in, GM_ADDR actual_seq_qlen,
                                GM_ADDR actual_seq_kvlen, GM_ADDR attention_in, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                GM_ADDR dpse, GM_ADDR workspace,
                                const FlashAttentionScoreGradTilingDataS1s2Bn2 *__restrict ordTilingData,
                                TPipe *pipe_in);
    __aicore__ inline void Process();

    using aType = typename AscendC::Conditional<
        (L1CUSTOM == true), 
        MatmulType<TPosition::GM, CubeFormat::ND, T1, false, LayoutMode::NONE, true>, 
        MatmulType<TPosition::GM, CubeFormat::ND, T1>>::type;

    using aTypeMM34 = typename AscendC::Conditional<
        (L1CUSTOM == true), 
        MatmulType<TPosition::GM, MM_OUT_FORMAT, T1, false, LayoutMode::NONE, true>, 
        MatmulType<TPosition::GM, MM_OUT_FORMAT, T1>>::type;

    using bType = typename AscendC::Conditional<
        (L1CUSTOM == true), 
        MatmulType<TPosition::GM, CubeFormat::ND, T1, false, LayoutMode::NONE, true>, 
        MatmulType<TPosition::GM, CubeFormat::ND, T1>>::type;
    
    using aTypeTranspose = typename AscendC::Conditional<
        (L1CUSTOM == true), 
        MatmulType<TPosition::GM, MM_OUT_FORMAT, T1, true, LayoutMode::NONE, true>, 
        MatmulType<TPosition::GM, MM_OUT_FORMAT, T1, true>>::type;
        
    using bTypeTranspose = typename AscendC::Conditional<
        (L1CUSTOM == true), 
        MatmulType<TPosition::GM, CubeFormat::ND, T1, true, LayoutMode::NONE, true>, 
        MatmulType<TPosition::GM, CubeFormat::ND, T1, true>>::type;

    using aTypeUB = MatmulType<TPosition::VECCALC, CubeFormat::ND, T1>;
    using bTypeUB = MatmulType<TPosition::VECCALC, CubeFormat::ND, T1>;
    using aTypeUBTranspose = MatmulType<TPosition::VECCALC, CubeFormat::ND, T1, true>;
    using bTypeUBTranspose = MatmulType<TPosition::VECCALC, CubeFormat::ND, T1, true>;

    using cType = MatmulType<TPosition::GM, MM2_OUT_FORMAT, float>;
    using cTypeMMNZ = MatmulType<TPosition::GM, CubeFormat::NZ, T2>;
    using cTypeMMAlign = MatmulType<TPosition::GM, CubeFormat::ND_ALIGN, T2>;
    using biasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

    using modeTypemm12L1CustomNz = typename AscendC::Conditional<
        (L1CUSTOM == true), 
        Matmul<aType, bTypeTranspose, cTypeMMNZ, biasType, MM_CFG, MatmulCallBackFunc<nullptr, nullptr, nullptr>, AscendC::Impl::Detail::FagMatmulPolicy>,
        Matmul<aType, bTypeTranspose, cTypeMMNZ, biasType, MM_CFG>>::type;
    using modeTypemm12L1CustomNd = typename AscendC::Conditional<
        (L1CUSTOM == true), 
        Matmul<aType, bTypeTranspose, cTypeMMAlign, biasType, MM_CFG, MatmulCallBackFunc<nullptr, nullptr, nullptr>, AscendC::Impl::Detail::FagMatmulPolicy>,
        Matmul<aType, bTypeTranspose, cTypeMMAlign, biasType, MM_CFG>>::type;
    using modeTypemm12 = typename AscendC::Conditional<
        (MM_OUT_FORMAT == CubeFormat::NZ),
        modeTypemm12L1CustomNz,
        modeTypemm12L1CustomNd>::type;

    modeTypemm12 mm1; //mm12

    using modeTypeDqL1CustomNz = typename AscendC::Conditional<
        (L1CUSTOM == true), 
        Matmul<aTypeMM34, bType, cType, biasType, MM_CFG, MatmulCallBackFunc<DataCopyOutLocal>, AscendC::Impl::Detail::FagMatmulPolicy>,
        Matmul<aTypeMM34, bType, cType, biasType, MM_CFG, MatmulCallBackFunc<DataCopyOutLocal>>>::type;
    using modeTypeDqL1CustomNd = typename AscendC::Conditional<
        (L1CUSTOM == true), 
        Matmul<aTypeMM34, bType, cType, biasType, MM_CFG, MatmulCallBackFunc<nullptr, nullptr, nullptr>, AscendC::Impl::Detail::FagMatmulPolicy>,
        Matmul<aTypeMM34, bType, cType, biasType, MM_CFG>>::type;
    using modeTypeDq = typename AscendC::Conditional<
        (MM_OUT_FORMAT == CubeFormat::NZ && MM2_OUT_FORMAT == CubeFormat::NZ),
        modeTypeDqL1CustomNz,
        modeTypeDqL1CustomNd>::type;

    modeTypeDq mm3_1; //dq

    using modeTypeDvL1CustomNz = typename AscendC::Conditional<
        (L1CUSTOM == true), 
        Matmul<aTypeTranspose, bType, cType, biasType, MM_CFG, MatmulCallBackFunc<DataCopyOutLocal>, AscendC::Impl::Detail::FagMatmulPolicy>,
        Matmul<aTypeTranspose, bType, cType, biasType, MM_CFG, MatmulCallBackFunc<DataCopyOutLocal>>>::type;
    using modeTypeDvL1CustomNd = typename AscendC::Conditional<
        (L1CUSTOM == true), 
        Matmul<aTypeTranspose, bType, cType, biasType, MM_CFG, MatmulCallBackFunc<nullptr, nullptr, nullptr>, AscendC::Impl::Detail::FagMatmulPolicy>,
        Matmul<aTypeTranspose, bType, cType, biasType, MM_CFG>>::type;
    using modeTypeDv = typename AscendC::Conditional<
        (MM2_OUT_FORMAT == CubeFormat::NZ),
        modeTypeDvL1CustomNz,
        modeTypeDvL1CustomNd>::type;

    modeTypeDv mm4; //dv

protected:
    // init
    __aicore__ inline void InitOutputBuffer(GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR dpse);
    __aicore__ inline void InitRequireInputBuffer(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy);
    __aicore__ inline void InitOptionInputBuffer(GM_ADDR pse_shift, GM_ADDR drop_mask, GM_ADDR padding_mask,
                                                 GM_ADDR atten_mask, GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                                 GM_ADDR attention_in);
    __aicore__ inline void InitParams(const FlashAttentionScoreGradTilingDataS1s2Bn2 *__restrict ordTilingData,
                                      GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen, GM_ADDR prefixN);
    __aicore__ inline void InitUB(TPipe *pipe_in);
    __aicore__ inline void InitBmmWorkspace(GM_ADDR workspace);
    __aicore__ inline void InitCastWorkspace(GM_ADDR workspace);
    __aicore__ inline void InitDropWorkspace(GM_ADDR workspace);
    __aicore__ inline void AtomicClean();
    __aicore__ inline void DumpGmZero(GlobalTensor<float> &gm, int64_t num);

    // process
    __aicore__ inline void SendMatmul2(const int64_t m, const int64_t n, const int64_t a_addr, const int64_t b_addr,
                                       const int64_t org_m);
    __aicore__ inline void SendMatmul1(const int64_t m, const int64_t n, const int64_t a_addr, const int64_t b_addr,
                                       const int64_t org_m);
    __aicore__ inline void SendMatmulDV(const uint32_t real_n, const uint32_t align_n, const uint32_t s1_inner,
                                        const int64_t a_in_addr, const int64_t b_in_addr, const int64_t out_addr,
                                        const bool is_sync, const uint8_t kvAtomic);
    __aicore__ inline void SendMatmulDQ(const uint32_t real_n, const uint32_t align_n, const uint32_t s1_inner,
                                        const int64_t a_in_addr, const int64_t b_in_addr, const int64_t out_addr,
                                        const bool is_sync, const uint8_t qAtomic);
    __aicore__ inline void SendMatmulDK(const uint32_t real_n, const uint32_t align_n, const uint32_t s1_inner,
                                        const int64_t a_in_addr, const int64_t b_in_addr, const int64_t out_addr,
                                        const bool is_sync, const uint8_t kvAtomic);
    __aicore__ inline void MTE2_ATMask(LocalTensor<uint8_t> &attenMaskTensor, int64_t &attenMaskOffset,
                                       PingPongEmitInsn &insn);
    __aicore__ inline void MTE2_SFT(LocalTensor<T2> &sumTensor, LocalTensor<T2> &maxTensor, int64_t &sumMaxOffset,
                                    PingPongEmitInsn &insn);
    __aicore__ inline void MTE2_STFGrad(GlobalTensor<T1> &gmTensor, int64_t addr, LocalTensor<T1> &localTensor,
                                        int64_t num, int64_t count);
    __aicore__ inline void DoMaskU8(LocalTensor<T2> &dstTensor, LocalTensor<uint8_t> &attenMaskTensor,
                                    LocalTensor<uint8_t> &helpTensor, PingPongEmitInsn &insn,
                                    const uint8_t maskType = 0);
    __aicore__ inline void DoSimpleSoftMax(LocalTensor<T2> &dstTensor, LocalTensor<float> &sumTensor,
                                           LocalTensor<float> &maxTensor, LocalTensor<uint8_t> &helpTensor,
                                           PingPongEmitInsn &insn);
    __aicore__ inline void DoSoftmaxGrad(LocalTensor<T2> &dstTensor);
    __aicore__ inline void FullGrad(LocalTensor<T2> &dstTensor);
    __aicore__ inline void SplitGrad(LocalTensor<T2> &dstTensor);
    __aicore__ inline void DoSub(LocalTensor<T2> &dstTensor, LocalTensor<T2> &srcTensor, PingPongEmitInsn &insn);
    __aicore__ inline void DoMul(LocalTensor<T2> &dstTensor, LocalTensor<T2> &srcTensor, PingPongEmitInsn &insn);
    __aicore__ inline void MMOffsetTensorA(const int64_t s1_idx, int64_t &a_addr);
    __aicore__ inline void MMOffsetTensorB(const int64_t s2_idx, int64_t &b_addr);
    __aicore__ inline void MMOffsetNzOut(const int64_t s1_idx, const int64_t s2_idx, int64_t &a_addr, int64_t &b_addr);
    __aicore__ inline void CalcCausalAttenMaskOffset(int64_t &attenMaskOffset, const int64_t delta, bool isPingMode);
    __aicore__ inline void CalcBandAttenMaskOffset(int64_t &attenMaskOffsetPre, int64_t &attenMaskOffset,
                                                   const int64_t delta, bool isPingMode);
    __aicore__ inline void CalcPrefixCompressAttenMaskOffset(int64_t &attenMaskOffsetPre, int64_t &attenMaskOffset,
                                                             const int64_t delta, int64_t s2Idx, bool isPingMode);
    __aicore__ inline void CopyInOffsetForSimpleSoftmax(int64_t s1Idx, bool isPingMode);
    __aicore__ inline void CopyInOffset(int64_t s1Idx, int64_t s2Idx, bool isPingMode);
    __aicore__ inline void CastTo32(LocalTensor<T2> &dstTensor, LocalTensor<T1> &srcTensor, uint32_t count);
    __aicore__ inline void CastTo16(LocalTensor<T1> &dstTensor, LocalTensor<T2> &srcTensor, uint32_t count);
    __aicore__ inline void DoMulsScale(LocalTensor<T2> &dstTensor, PingPongEmitInsn &insn);
    __aicore__ inline void CalcSparseIdx(const int64_t bIndex, const int64_t s1Idx, const int64_t s1Size,
                                         int64_t &s2_start_idx, int64_t &s2_end_idx);
    __aicore__ inline void CopyoutWorkspace(const GlobalTensor<T1> &dstGm, const LocalTensor<T1> &srcTensor,
                                            PingPongEmitInsn &insn);
    __aicore__ inline void AssureUsefulDataBySingleN();

    __aicore__ inline void S2Ratio(const int64_t gIdx);
    __aicore__ inline void S1Ratio(int64_t s2_o_o, const int64_t gIdx);
    __aicore__ inline bool CalcUsefulDataByS2();
    __aicore__ inline void VectorByCS1(int64_t mm1Addr, int64_t mm2Addr, int64_t mm3Addr, int64_t mm4Addr,
                                       int64_t realProcessNtmp, int64_t processM);
    __aicore__ inline void VectorByS1S2(int64_t mm1Addr, int64_t mm2Addr, int64_t mm3Addr, int64_t mm4Addr, int64_t processM);
    __aicore__ inline void InnerT2Process(int64_t processM);
    __aicore__ inline void MTE2ForMM2(LocalTensor<T2> &mm2TensorCurr, int64_t &mm2Offset, PingPongEmitInsn &insn);
    __aicore__ inline void MTE2ForMM1(LocalTensor<T2> &mm1TensorCurr, int64_t &mm1Offset, PingPongEmitInsn &insn);
    __aicore__ inline void NZ2ND(LocalTensor<T2> &ndTensor, LocalTensor<T2> &nzTensor, PingPongEmitInsn &insn);
    __aicore__ inline void ND2NZ(LocalTensor<T1> &nzTensor, LocalTensor<T1> &ndTensor, PingPongEmitInsn &insn);
    __aicore__ inline void NZCopyIn(int64_t mmAddr, GlobalTensor<T2> &mmWspGm, LocalTensor<T2> &mmTensorCurr, PingPongEmitInsn &insn, int64_t processM);
    __aicore__ inline void UpdateLoopParams(int64_t i);
    __aicore__ inline void MallocNodes();
    __aicore__ inline void PingClcParams(int64_t mm1Addr, int64_t mm2Addr, int64_t mm3Addr, int64_t mm4Addr, int64_t processM);
    __aicore__ inline void PongClcParams(int64_t mm1Addr, int64_t mm2Addr, int64_t mm3Addr, int64_t mm4Addr, int64_t processM);
    __aicore__ inline void DropOutCopy(LocalTensor<uint8_t> &dropmaskTensor, PingPongEmitInsn &insn);
    __aicore__ inline void SavePostBNIdx();
    __aicore__ inline void PostProcess(bool isLast);
    __aicore__ inline void PostProcessQ();
    __aicore__ inline void PostProcessKV();
    __aicore__ inline void PostProcessND(GlobalTensor<float> &workspaceGm, GlobalTensor<T1> &outGm, const int64_t &type);
    __aicore__ inline void PostCalND(GlobalTensor<float> &workspaceGm, GlobalTensor<T1> &outGm, const uint16_t &procS, 
                                     const int64_t &gmOffset, const int64_t &type);
    __aicore__ inline void PostProcessNZ(GlobalTensor<float> &workspaceGm, GlobalTensor<T1> &outGm, const uint32_t &type);
    __aicore__ inline void PostCalNZ(GlobalTensor<float> &workspaceGm, GlobalTensor<T1> &outGm, const uint64_t &procS,
                                     const uint64_t &SIdx, const uint32_t &type);
    __aicore__ inline uint64_t InitL1Idx(const uint32_t &aIdx, const uint32_t &bIdx, const uint32_t matmulIdx, const bool& reuseA, const bool& reuseB);

protected:
    GlobalTensor<T1> queryGm;
    GlobalTensor<T1> keyGm;
    GlobalTensor<T1> valueGm;
    GlobalTensor<T1> dyGm;

    GlobalTensor<T1> pseShiftGm;
    GlobalTensor<uint8_t> dropMaskGm;
    GlobalTensor<T1> paddingMaskGm;
    GlobalTensor<T1> attenMaskGm;
    GlobalTensor<uint8_t> attenMaskU8Gm;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    GlobalTensor<T1> softmaxInGm;
    GlobalTensor<T1> attentionInGm;

    GM_ADDR prefixN_addr;
    GM_ADDR actual_seq_qlen_addr;
    GM_ADDR actual_seq_kvlen_addr;

    /*
    FP16-HighPerformance: res is workspace
    FP16-HighPrecision:: res is workspace
    BFP16: res is workspace
    FP32: res is dxGm
    */
    GlobalTensor<T1> dqGm;
    GlobalTensor<T1> dkGm;
    GlobalTensor<T1> dvGm;

    GlobalTensor<T2> mm1WorkspaceGm;
    GlobalTensor<T2> mm2WorkspaceGm;
    GlobalTensor<T1> mm3InputWorkspaceGm;
    GlobalTensor<T1> mm4InputWorkspaceGm;

    GlobalTensor<uint8_t> dropoutWorkspaceGm;
    GlobalTensor<float> dqWorkspaceGm;
    GlobalTensor<float> dkWorkspaceGm;
    GlobalTensor<float> dvWorkspaceGm;

    GlobalTensor<int32_t> syncAtomicCleanGlobal;
    GlobalTensor<int32_t> syncCastGlobal;

    GlobalTensor<half> pseAlibiGm;
    __gm__ uint8_t *pseSlope;

    PseInfo pseInfo = {0};

    TPipe *pipe;
    int64_t blockIdx;
    int64_t cubeBlockIdx;
    int64_t subBlockIdx;
    const FlashAttentionScoreGradTilingDataS1s2Bn2 *__restrict tilingData;

    TBuf<> vecQue;                        // 176K + 5K
    TBuf<> softmaxGradOutQue;             // 8K
    LocalTensor<T2> softmaxGradOutTensor; // 8K

    // Common nodes
    LocalTensor<uint8_t> b8Node0;
    LocalTensor<T1> b16Node0;
    LocalTensor<T2> b32Node0;
    // Ping nodes
    LocalTensor<uint8_t> b8Node1;
    LocalTensor<uint8_t> b8Node2;
    LocalTensor<T1> b16Node2;
    LocalTensor<T2> b32Node3;
    LocalTensor<T2> b32Node4;
    LocalTensor<uint8_t> b8Node5;
    LocalTensor<T2> b32Node6;
    LocalTensor<T2> b32PingFuseNode;
    LocalTensor<uint8_t> b8PingFuseNode;
    // Pong nodes
    LocalTensor<uint8_t> b8Node7;
    LocalTensor<uint8_t> b8Node8;
    LocalTensor<T1> b16Node8;
    LocalTensor<T2> b32Node9;
    LocalTensor<T2> b32Node10;
    LocalTensor<uint8_t> b8Node11;
    LocalTensor<T2> b32Node12;
    LocalTensor<T2> b32PongFuseNode;
    LocalTensor<uint8_t> b8PongFuseNode;
    // Post buffer
    LocalTensor<float> postInBuf;
    LocalTensor<T1> postTmpBuf;
    LocalTensor<T1> postOutBuf;
    // sftg buffer
    LocalTensor<T1> dyT1Tensor;                // 0~16K
    LocalTensor<T1> attentionT1Tensor; // 16K~32K
    LocalTensor<T2> dyT2Tensor;        // 32K~64K
    LocalTensor<T2> attentionT2Tensor; // 64K~96K
    LocalTensor<uint8_t> helpTensor;           // 96K~160K

    // core
    int64_t usedCoreNum;
    int64_t formerCoreNum;

    // Shape
    int64_t dimB;
    int64_t dimN2;
    int64_t dimS1;
    int64_t dimS2;
    int64_t dimG;
    int64_t dimD;
    int64_t dimDAlign;
    int64_t dimT_kv{0};
    int64_t dimT_q{0};
    uint32_t attenMaskDimS2;
    int64_t seqS1Current;
    int64_t seqS2Current;
    int64_t seqS1CurrentOffset;
    int64_t seqS2CurrentOffset;
    uint64_t seqS1S2ProductSum{0};
    int64_t bandIdx;
    bool unpadUseLeftUpCasual{0};
    bool unpadUseRightDownCasual{0};
    bool unpadUseBand{0};
    bool unpadEmptyInput{false};

    // attr
    float scaleValue;
    float keepProb;
    uint32_t existAttenMask;
    uint32_t compressMode;
    uint32_t maskDataType;
    uint32_t maskShapeType;
    uint32_t pseShapeType;
    bool dropBitMode = false;

    // ub space
    uint32_t inputBufferLen;
    uint32_t helpBufferLen;

    // workspace (couldn't bigger than 4G)
    uint32_t mm1WorkspaceLen;
    uint32_t mm2WorkspaceLen;
    uint32_t mm3InputWorkspaceLen;
    uint32_t mm4InputWorkspaceLen;
    int64_t dqWorkspaceLen;
    int64_t dkWorkspaceLen;
    int64_t dvWorkspaceLen;
    int64_t dropoutWorkspaceLen;

    // Index
    uint32_t processBNByCore;
    int64_t bIndex;
    int64_t n2Index;
    int64_t s1Index{0};
    int64_t s2Index{0};
    int64_t gIndex{0};
    // 上一轮 Index
    int64_t lastBIndex{0};
    int64_t lastN2Index{0};
    int64_t lastS1Index{0};
    int64_t lastS2Index{0};
    int64_t lastGIndex{0};
    // tmp Index
    int64_t tmpBIndex{0};
    int64_t tmpN2Index{0};
    int64_t tmpS1Index{0};
    int64_t tmpS2Index{0};
    int64_t tmpGIndex{0};

    // 地址相关
    int64_t inputMMLeftMatrixAddr;  // MM1's addr as same as MM2
    int64_t inputMMRighMatrixtAddr; // MM1's addr as same as MM2
    int64_t dyGmAddr;
    int64_t attentionInGmAddr;
    int64_t mm3_4_tensor_g_s1_addr;
    int64_t mm3_4_tensor_1_s2_addr;
    int64_t mm3_4_out_g_s1_addr;
    int64_t mm3_4_out_1_s2_addr;
    int64_t mm3PangInputWspOffset;
    int64_t mm4PangInputWspOffset;

    // sparse相关
    int64_t sparse_s2_start_idx{0};
    int64_t sparse_s2_end_idx{0};
    int64_t preTokens{0};
    int64_t nextTokens{0};
    uint32_t isSparse{0};

    uint32_t usedSingleNBegin{0};
    uint32_t lastUsedSingleNBegin{0};
    uint32_t tmpUsedSingleNBegin{0};
    uint32_t usedSingleNEnd{0};
    uint32_t usedSingleNNum{0};
    uint32_t realProcessN{0};
    uint32_t b16AlignProcessN{0};
    uint32_t alignProcessM{0};
    uint32_t b32AlignProcessN{0};
    int64_t mm1WorkspaceAddr{0};
    int64_t mm2WorkspaceAddr{0};
    int64_t mm3InputWorkspaceAddr{0};
    int64_t mm4InputWorkspaceAddr{0};
    uint32_t currentLoop{0};

    // 记录前一次有效地址
    int64_t lastMM3InputWorkspaceAddr{0};
    int64_t lastMM4InputWorkspaceAddr{0};
    int64_t last_mm3_4_tensor_g_s1_addr{0};
    int64_t last_mm3_4_tensor_1_s2_addr{0};
    int64_t last_mm3_4_out_g_s1_addr{0};
    int64_t last_mm3_4_out_1_s2_addr{0};
    //mm345 nzout
    int64_t s1_addr_nzout{0};
    int64_t s2_addr_nzout{0};
    // 记录前一次变量
    uint32_t lastGIdx{0};
    uint32_t lastS1OO{0};
    uint32_t lastS2OO{0};
    uint32_t lastProcessM{0};
    uint32_t lastRealProcessN{0};
    uint32_t b16LastRealAlignProcessN{0};

    StaticCParams cube;
    StaticVParams vec;
    StaticSFTParams sft;
    DyncReal rp;
    DyncLoop lp;

    bool isLastBN{false};
    bool isLastG{false};
    bool isLastSingleM{false};
    bool isLastSingleN{false};
    bool isFirstSingleM{false};
    bool isFirstSingleN{false};
    bool haveS1Loop{false};
    bool haveS2Loop{false};
    bool enableS1Reuse{false};

    // 寄存已使用的workspaceLen
    int64_t usedWorkspaceLen{0};
    int64_t keySize;

    // OrgShape变化标志位
    uint32_t mm1Scalar{0};
    uint32_t mm2Scalar{0};
    uint32_t mm1ScalarOrgM{0};
    uint32_t mm2ScalarOrgM{0};
    uint32_t mmDQScalar{0};
    uint32_t mmDVScalar{0};

    // For DB
    int64_t pingHighMask{0};
    int64_t pingLowerMask{0};
    int64_t pingVar{1};
    int64_t pongHighMask{0};
    int64_t pongLowerMask{0};
    int64_t pongVar{1};

    uint32_t totalLoops{0};
    uint32_t mmPingpongIdx{0};
    uint32_t pingLoop{0};
    uint32_t pongLoop{1};
    bool isEven{false};
    bool pingOK{false};
    bool pongOK{false};
    bool isPingFirst{false};
    bool isPongFirst{false};
    bool isPingLast{false};
    bool isPongLast{false};
    bool isLeftRightBandCausal{false};
    DropMaskInfo dropMaskInfo = {0};

    int64_t mm1PingAddr{0};
    int64_t mm2PingAddr{0};
    int64_t mm3PingAddr{0};
    int64_t mm4PingAddr{0};
    int64_t mm1PongAddr{0};
    int64_t mm2PongAddr{0};
    int64_t mm3PongAddr{0};
    int64_t mm4PongAddr{0};

    int64_t attenmaskPingPreAddress{0};
    int64_t attenmaskPongPreAddress{0};
    int64_t attenmaskPingAddress{0};
    int64_t attenmaskPongAddress{0};
    int64_t dropmaskPingAddress{0};
    int64_t dropmaskPongAddress{0};
    int64_t softmaxMaxSumPingAddress{0};
    int64_t softmaxMaxSumPongAddress{0};

    // post data
    bool isLastWorkSpaceFinished{false};
    uint32_t postUbBaseSize{0};
    int64_t postCalCnt{0};
    int64_t postCalConsum{0};
    PostIndices lastIndices;
    PostIndices calIndices;
    PostIndices savedIndices0;
    PostIndices savedIndices1;

    PingPongEmitInsn pingEI;
    PingPongEmitInsn pongEI;
    bool isPseInnerGenerate{0};
    float postScaleValue{0};

    // L1 reuse
    uint32_t mm345PingPangIdx{0};
    uint32_t qPingPongIdx{0};
    uint32_t kPingPongIdx{0};
    uint32_t dyPingPongIdx{0};
    uint32_t vPingPongIdx{0};
    uint32_t dqPingPongIdx{0};
    uint32_t dkPingPongIdx{0};
    uint32_t dvPingPongIdx{0};

    uint32_t dqConsum{0};
    uint32_t dkConsum{0};
    uint32_t dvConsum{0};
    uint32_t qStartPingPongIdx{0};


    // const
    constexpr static const uint32_t BNGSD = 0;
    constexpr static const uint32_t SBNGD = 1;
    constexpr static const uint32_t BSNGD = 2;
    constexpr static const uint32_t TND = 3;

    constexpr static uint32_t BAND_MODE = 3;
    constexpr static uint32_t PREFIX_COMPRESS_MODE = 4;
    constexpr static uint32_t LEFT_UP_CAUSAL = 2;
    constexpr static uint32_t BAND = 4;

    constexpr static int64_t BLOCK = 32;
    constexpr static int64_t C0_SIZE = 16;
    constexpr static int64_t STRIDE_LIMIT = 65535;
    constexpr static int64_t VALUE_ZERO = 0;
    constexpr static int64_t B32_BLOCK_NUM = BLOCK / sizeof(int32_t);
    constexpr static int64_t VEC_REPEAT = 8;
    constexpr static int64_t AVOID_BANK_CONFLICT_USE = 1024;

    constexpr static int64_t calcBlockNum = BLOCK / sizeof(T2);
    constexpr static int64_t dataCopyBlockNum = BLOCK / sizeof(T1);

    constexpr static uint32_t PREFIX_COMPRESS_CAUSAL_S_SIZE = 2048;
    constexpr static uint32_t PREFIX_COMPRESS_ALL_MASK_S1_SIZE = 1024;

    constexpr static uint32_t TYPE_Q = 0;
    constexpr static uint32_t TYPE_K = 1;
    constexpr static uint32_t TYPE_V = 2;
    constexpr static uint32_t MAX_STORE_POST_INDEX = 2;

    // attenmask
    enum class AttenMaskCompress {
        Empty = 0,
        PreOnly = 1,
        NextOnly = 2,
        All = 3
    };
    AttenMaskCompress AttenBandPingMode = AttenMaskCompress::All;
    AttenMaskCompress AttenBandPongMode = AttenMaskCompress::All;

    event_t mte2WaitV;
    event_t vWaitMte2;
    event_t mte2WaitMte3;
    event_t vWaitMte3;
    event_t mte3WaitV;
};

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG, LAYOUT,
    MM2_OUT_FORMAT, POST, L1CUSTOM>::Init(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR pse_shift, GM_ADDR drop_mask, GM_ADDR padding_mask,
    GM_ADDR atten_mask, GM_ADDR softmax_max, GM_ADDR softmax_sum, GM_ADDR prefixN, GM_ADDR softmax_in,
    GM_ADDR actual_seq_qlen, GM_ADDR actual_seq_kvlen, GM_ADDR attention_in, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
    GM_ADDR dpse, GM_ADDR workspace, const FlashAttentionScoreGradTilingDataS1s2Bn2 *__restrict ordTilingData,
    TPipe *pipe_in)
{
    InitOutputBuffer(dq, dk, dv, dpse);

    InitRequireInputBuffer(query, key, value, dy);

    InitOptionInputBuffer(pse_shift, drop_mask, padding_mask, atten_mask, softmax_max, softmax_sum, attention_in);

    InitParams(ordTilingData, actual_seq_qlen, actual_seq_kvlen, prefixN);

    if constexpr (DROPOUT_CFG != 0) {
        InitDropWorkspace(workspace);
    }

    InitBmmWorkspace(workspace);

    InitCastWorkspace(workspace);

    if constexpr (!L1CUSTOM) {
        if (isSparse == 1 || unpadEmptyInput) {
            AtomicClean();
        }
    }

    InitUB(pipe_in);

    if constexpr (PSE_CFG != 0) {
        if ((L1CUSTOM ? cubeBlockIdx : blockIdx) < usedCoreNum && 
            (tilingData->opInfo.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
            tilingData->opInfo.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE)) {
            LocalTensor<half> pseHelpBuffer = vecQue.GetWithOffset<half>(16 * 1024 / sizeof(half), 116 * 1024);
            PseInnerAlibiCreate<true>(this->pseAlibiGm, pseHelpBuffer, pseInfo);
            isPseInnerGenerate = true;
        }
    }

    if constexpr (DROPOUT_CFG != 0) {
        if constexpr (LAYOUT != TND) {
            dropMaskInfo.s1Size = dimS1;
            dropMaskInfo.s2Size = dimS2;
        }
        // for compute dropout mask offset
        dropMaskInfo.n2G = dimN2 * dimG;
        dropMaskInfo.gSize = dimG;
        dropMaskInfo.s1BaseSize = 1;
        dropMaskInfo.s2BaseNratioSize = 1;

        // for compute dropout mask
        dropMaskInfo.keepProb = keepProb;
        dropMaskInfo.boolMode = dropBitMode ? false : true;
    }
    MallocNodes();
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG,
                LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::InitRequireInputBuffer(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy)
{
    // 必选输入初始化
    queryGm.SetGlobalBuffer((__gm__ T1 *)query);
    keyGm.SetGlobalBuffer((__gm__ T1 *)key);
    valueGm.SetGlobalBuffer((__gm__ T1 *)value);
    dyGm.SetGlobalBuffer((__gm__ T1 *)dy);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG,
               LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::InitOptionInputBuffer(GM_ADDR pse_shift, GM_ADDR drop_mask,
                                                              GM_ADDR padding_mask, GM_ADDR atten_mask,
                                                              GM_ADDR softmax_max, GM_ADDR softmax_sum,
                                                              GM_ADDR attention_in)
{
    // 可选输入初始化
    pseSlope = pse_shift;
    pseShiftGm.SetGlobalBuffer((__gm__ T1 *)pse_shift);
    dropMaskGm.SetGlobalBuffer((__gm__ uint8_t *)drop_mask);
    paddingMaskGm.SetGlobalBuffer((__gm__ T1 *)padding_mask);
    attenMaskGm.SetGlobalBuffer((__gm__ T1 *)atten_mask);
    attenMaskU8Gm.SetGlobalBuffer((__gm__ uint8_t *)atten_mask);
    softmaxMaxGm.SetGlobalBuffer((__gm__ float *)softmax_max);
    softmaxSumGm.SetGlobalBuffer((__gm__ float *)softmax_sum);
    attentionInGm.SetGlobalBuffer((__gm__ T1 *)attention_in);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG,
                                        DROPOUT_CFG, LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::InitOutputBuffer(GM_ADDR dq, GM_ADDR dk,
                                                                                             GM_ADDR dv, GM_ADDR dpse)
{
    // 输出初始化
    dqGm.SetGlobalBuffer((__gm__ T1*)dq);
    dkGm.SetGlobalBuffer((__gm__ T1*)dk);
    dvGm.SetGlobalBuffer((__gm__ T1*)dv);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG, LAYOUT,
    MM2_OUT_FORMAT, POST, L1CUSTOM>::InitParams(
    const FlashAttentionScoreGradTilingDataS1s2Bn2 *__restrict ordTilingData, GM_ADDR actual_seq_qlen,
    GM_ADDR actual_seq_kvlen, GM_ADDR prefixN)
{
    // op_info
    blockIdx = GetBlockIdx();
    if constexpr (L1CUSTOM) {
        cubeBlockIdx = blockIdx / 2;
        subBlockIdx = blockIdx % 2;
    }
    tilingData = ordTilingData;
    usedCoreNum = tilingData->opInfo.usedCoreNum;
    formerCoreNum = tilingData->opInfo.formerCoreNum;

    dimB = tilingData->opInfo.B;
    dimN2 = tilingData->opInfo.N2;
    dimS1 = tilingData->opInfo.S1;
    dimS2 = tilingData->opInfo.S2;
    dimG = tilingData->opInfo.G;
    dimD = tilingData->opInfo.D;
    dimDAlign = (dimD + C0_SIZE - 1) / C0_SIZE * C0_SIZE;
    attenMaskDimS2 = tilingData->opInfo.attenMaskS2Size;

    actual_seq_qlen_addr = actual_seq_qlen;
    actual_seq_kvlen_addr = actual_seq_kvlen;

    scaleValue = tilingData->opInfo.scaleValue;
    keepProb = tilingData->opInfo.keepProb;
    preTokens = tilingData->opInfo.preTokens;
    nextTokens = tilingData->opInfo.nextTokens;
    isSparse = tilingData->opInfo.isSparse;
    maskShapeType = tilingData->opInfo.maskShapeType;
    maskDataType = tilingData->opInfo.maskDataType;
    pseShapeType = tilingData->opInfo.pseShapeType;
    bandIdx = tilingData->opInfo.bandIdx;
    existAttenMask = tilingData->opInfo.existAttenMask;
    compressMode = tilingData->opInfo.attenMaskCompressMode;
    inputBufferLen = tilingData->opInfo.inputBufferLen;
    helpBufferLen = tilingData->opInfo.helpBufferLen;
    mm1WorkspaceLen = tilingData->opInfo.mm1WorkspaceLen;
    mm2WorkspaceLen = tilingData->opInfo.mm2WorkspaceLen;
    mm3InputWorkspaceLen = tilingData->opInfo.mm3InputWorkspaceLen;
    mm4InputWorkspaceLen = tilingData->opInfo.mm4InputWorkspaceLen;
    dqWorkspaceLen = tilingData->opInfo.dqWorkspaceLen;
    dkWorkspaceLen = tilingData->opInfo.dkWorkspaceLen;
    dvWorkspaceLen = tilingData->opInfo.dvWorkspaceLen;
    dropoutWorkspaceLen = tilingData->opInfo.dropoutWorkspaceLen;
    unpadEmptyInput = static_cast<bool>(tilingData->opInfo.unpadEmptyInput);

    if constexpr (PSE_CFG != 0) {
        pseInfo.s2Size = dimS2;
        pseInfo.s1Size = dimS1;
        pseInfo.gSize = dimG;
        pseInfo.n2G = dimN2 * dimG;
        pseInfo.pseType = tilingData->opInfo.pseType;
        pseInfo.pseShapeType = pseShapeType;
        if (pseShapeType == 2 || pseShapeType == 3 || pseShapeType == 4) {
            pseInfo.pseShapeType = 0;
        } else if (pseShapeType == 5) {
            pseInfo.pseShapeType = 2;
        } else if (pseShapeType == 6) {
            pseInfo.pseShapeType = 3;
        }
        pseInfo.pseAlibiBaseS1 = tilingData->opInfo.pseAlibiBaseS1;
        pseInfo.pseAlibiBaseS2 = tilingData->opInfo.pseAlibiBaseS2;
        pseInfo.qStartIdx = tilingData->opInfo.qStartIdx;
        pseInfo.kvStartIdx = tilingData->opInfo.kvStartIdx;
        pseInfo.pseBSize = (pseShapeType == 2 || pseShapeType == 4) ? 1 : dimB;
        pseInfo.pseEncodeType = (pseShapeType == 3 || pseShapeType == 4) ? 0x11 : 0;
        pseInfo.pseS1Size = 1024;
        pseInfo.pseS2Size = dimS2;
        pseInfo.needCast = false;
    }

    mm3PangInputWspOffset = mm3InputWorkspaceLen / sizeof(T1) / 2;
    mm4PangInputWspOffset = mm4InputWorkspaceLen / sizeof(T1) / 2;

    // 确定Vector相关参数
    vec.baseM = tilingData->splitCoreParams.baseM;
    vec.baseN = tilingData->splitCoreParams.baseN;
    vec.singleM = tilingData->splitCoreParams.singleM;
    vec.singleN = tilingData->splitCoreParams.singleN;
    vec.singleMTail = dimS1 % vec.singleM == 0 ? vec.singleM : dimS1 % vec.singleM;
    vec.singleNTail = dimS2 % vec.singleN == 0 ? vec.singleN : dimS2 % vec.singleN;
    vec.s1OuterOuterNum = tilingData->splitCoreParams.s1OuterOuter;
    vec.s2OuterOuterNum = tilingData->splitCoreParams.s2OuterOuter;

    // 确定sft相关参数
    sft.baseM = tilingData->splitCoreParams.SFTBaseM;
    sft.singleM = tilingData->splitCoreParams.SFTSingleM;
    sft.dInner = tilingData->splitCoreParams.dInner;
    sft.dInnerTail = dimD % sft.dInner == 0 ? sft.dInner : dimD % sft.dInner;

    // 确定matmul相关参数
    cube.baseM = tilingData->mm1TilingData.baseM;
    cube.baseN = tilingData->mm1TilingData.baseN;

    dropBitMode = dimS2 % vec.baseN == 0;
    haveS1Loop = vec.s1OuterOuterNum != 1;
    haveS2Loop = vec.s2OuterOuterNum != 1;
    enableS1Reuse =  vec.s1OuterOuterNum <= 2;

    // 确定压缩类型
    if constexpr (!L1CUSTOM) {
        if (compressMode == 1 || compressMode == 2 || compressMode == BAND_MODE || tilingData->opInfo.sparseMode == 7 ||
            tilingData->opInfo.sparseMode == 8) {
            isLeftRightBandCausal = true;
        }
    }

    prefixN_addr = prefixN;

    if constexpr (LAYOUT == TND) {
        for (int64_t i = 0; i < dimB; i++) {
            int64_t seqS1Len = 0;
            int64_t seqS2Len = 0;
            if (unlikely(i == 0)) {
                seqS1Len = ((__gm__ int64_t *)actual_seq_qlen)[i];
                seqS2Len = ((__gm__ int64_t *)actual_seq_kvlen)[i];
            } else {
                seqS1Len = ((__gm__ int64_t *)actual_seq_qlen)[i] - ((__gm__ int64_t *)actual_seq_qlen)[i - 1];
                seqS2Len = ((__gm__ int64_t *)actual_seq_kvlen)[i] - ((__gm__ int64_t *)actual_seq_kvlen)[i - 1];
            }
            dropBitMode = (dropBitMode && ((int64_t)seqS2Len % vec.baseN == 0));
            dimT_q += (int64_t)seqS1Len;
            dimT_kv += (int64_t)seqS2Len;
        }
    }

    if constexpr (LAYOUT != TND) {
        if ((L1CUSTOM ? cubeBlockIdx : blockIdx) < formerCoreNum) {
            processBNByCore = tilingData->opInfo.formerCoreProcessNNum;
        } else {
            processBNByCore = tilingData->opInfo.remainCoreProcessNNum;
        }
        keySize = dimB * dimN2 * dimS2 * dimD;
    } else {
        processBNByCore = tilingData->tndSplitCoreParams.bN2idxEnds[(L1CUSTOM ? cubeBlockIdx : blockIdx)] -
                          tilingData->tndSplitCoreParams.bN2idxStarts[(L1CUSTOM ? cubeBlockIdx : blockIdx)] + 1;
        keySize = dimT_kv * dimN2 * dimD;
    }
    if(POST){
        postUbBaseSize = tilingData->postTilingData.postMaxDataSize;
        postScaleValue = tilingData->postTilingData.scaleValue;
    }
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG, LAYOUT,
     MM2_OUT_FORMAT, POST, L1CUSTOM>::InitUB(
    TPipe *pipe_in)
{
    pipe = pipe_in;
    pipe->InitBuffer(vecQue, 183 * 1024);
    pipe->InitBuffer(softmaxGradOutQue, sft.singleM * BLOCK); // 8K

    // hold on tensors
    softmaxGradOutTensor = softmaxGradOutQue.Get<T2>();

    if constexpr (POST){
        // post init
        uint64_t FP32UbSize = postUbBaseSize * sizeof(float);
        uint64_t FP16UbSize = postUbBaseSize * sizeof(T1);

        if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ){
            postInBuf = vecQue.GetWithOffset<float>(postUbBaseSize, 0);
            postOutBuf = vecQue.GetWithOffset<T1>(postUbBaseSize, FP32UbSize);
            postTmpBuf = vecQue.GetWithOffset<T1>(postUbBaseSize, FP32UbSize + FP16UbSize);
        }else{
            postInBuf = vecQue.GetWithOffset<float>(postUbBaseSize, 0);
            postOutBuf = vecQue.GetWithOffset<T1>(postUbBaseSize, FP32UbSize);
        }
    }
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG, LAYOUT,
     MM2_OUT_FORMAT, POST, L1CUSTOM>::DumpGmZero(GlobalTensor<float> &gm, int64_t num)
{
    // dump 0 to gm by blockIdx
    int64_t perSize = (num + tilingData->opInfo.castUsedCoreNum - 1) / tilingData->opInfo.castUsedCoreNum;
    int64_t coreNum = (num + perSize - 1) / perSize;
    int64_t tailSize = num - perSize * (coreNum - 1);
    int64_t initSize = perSize;

    if (blockIdx == coreNum - 1) {
        initSize = tailSize;
    }

    if (blockIdx < coreNum) {
        InitOutput<float>(gm[blockIdx * perSize], initSize, 0);
    }
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG,
                                                      DROPOUT_CFG, LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::AtomicClean()
{
    // FP32 clean
    // Input is B16 clean workspace
    // Input is B32 clean output-gm
    // Used All UB before InitUB
    int64_t dqSize, dkvSize;
    if constexpr (LAYOUT != TND) {
        dkvSize = dimB * dimN2 * dimS2 * dimDAlign;
        dqSize = dimB * dimN2 * dimG * dimS1 * dimDAlign;
    } else {
        dkvSize = dimT_kv * dimN2 * dimDAlign;
        dqSize = dimT_q * dimN2 * dimG * dimDAlign;
    }
    dkvSize = (dkvSize + B32_BLOCK_NUM - 1) / B32_BLOCK_NUM * B32_BLOCK_NUM;
    dqSize = (dqSize + B32_BLOCK_NUM - 1) / B32_BLOCK_NUM * B32_BLOCK_NUM;
    if constexpr (sizeof(T1) == sizeof(float)) {
        int64_t dvGmRealSize = dimB * dimN2 * dimS2 * dimD;
        if constexpr (LAYOUT == TND) {
            dvGmRealSize = dimT_kv * dimN2 * dimD;
        }
        DumpGmZero(dvGm, dvGmRealSize);
    } else {
        DumpGmZero(dvWorkspaceGm, dkvSize); // FP32 FUNC:T1 FP16
    }
    DumpGmZero(dqWorkspaceGm, dqSize);
    DumpGmZero(dkWorkspaceGm, dkvSize);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG,
                                             DROPOUT_CFG, LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::InitDropWorkspace(GM_ADDR workspace)
{
    if (dropBitMode) {
        return;
    }

    // dropout op
    auto dropoutAddr = usedWorkspaceLen / sizeof(uint8_t);
    dropoutWorkspaceGm.SetGlobalBuffer((__gm__ uint8_t *)workspace + dropoutAddr);

    usedWorkspaceLen += dropoutWorkspaceLen;
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG,
                                            DROPOUT_CFG, LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::InitBmmWorkspace(GM_ADDR workspace)
{
    // bmm used T2
    // mm1WorkspaceLen should be as same as mm1
    int64_t mm1Addr = 0;
    int64_t mm2Addr = 0;
    if constexpr (L1CUSTOM) {
        mm1Addr = usedWorkspaceLen / sizeof(T2) + cubeBlockIdx * mm1WorkspaceLen / sizeof(T2) * 2;
        mm2Addr = usedWorkspaceLen / sizeof(T2) + usedCoreNum * mm1WorkspaceLen / sizeof(T2) * 2 +
            cubeBlockIdx * mm2WorkspaceLen / sizeof(T2) * 2;
    } else {
        mm1Addr = usedWorkspaceLen / sizeof(T2) + blockIdx * mm1WorkspaceLen / sizeof(T2);
        mm2Addr = usedWorkspaceLen / sizeof(T2) + usedCoreNum * mm1WorkspaceLen / sizeof(T2) +
            blockIdx * mm2WorkspaceLen / sizeof(T2);
    }

    mm1WorkspaceGm.SetGlobalBuffer((__gm__ T2 *)workspace + mm1Addr);
    mm2WorkspaceGm.SetGlobalBuffer((__gm__ T2 *)workspace + mm2Addr);

    int64_t mm4Addr = 0;
    int64_t mm3Addr = 0;
    if constexpr (L1CUSTOM) {
        usedWorkspaceLen += (mm1WorkspaceLen + mm2WorkspaceLen) * usedCoreNum * 2;
        // mm3.1 mm3.2 mm4 used workspace as input of Bmm
        mm4Addr = usedWorkspaceLen / sizeof(T1) + cubeBlockIdx * mm4InputWorkspaceLen / sizeof(T1);
        usedWorkspaceLen += mm4InputWorkspaceLen * usedCoreNum;
        mm3Addr = usedWorkspaceLen / sizeof(T1) + cubeBlockIdx * mm3InputWorkspaceLen / sizeof(T1);
        usedWorkspaceLen += mm3InputWorkspaceLen * usedCoreNum;
    } else {
        usedWorkspaceLen += (mm1WorkspaceLen + mm2WorkspaceLen) * usedCoreNum;
        // mm3.1 mm3.2 mm4 used workspace as input of Bmm
        mm4Addr = usedWorkspaceLen / sizeof(T1) + blockIdx * mm4InputWorkspaceLen / sizeof(T1);
        usedWorkspaceLen += mm4InputWorkspaceLen * usedCoreNum;
        mm3Addr = usedWorkspaceLen / sizeof(T1) + blockIdx * mm3InputWorkspaceLen / sizeof(T1);
        usedWorkspaceLen += mm3InputWorkspaceLen * usedCoreNum;
    }

    mm4InputWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + mm4Addr);
    mm3InputWorkspaceGm.SetGlobalBuffer((__gm__ T1 *)workspace + mm3Addr);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG,
                                            DROPOUT_CFG, LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::InitCastWorkspace(GM_ADDR workspace)
{
    auto dqAddr = usedWorkspaceLen / sizeof(float);
    auto dkAddr = dqAddr + dqWorkspaceLen / sizeof(float);
    auto dvAddr = dkAddr + dkWorkspaceLen / sizeof(float);
    dqWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + dqAddr);
    dkWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + dkAddr);
    dvWorkspaceGm.SetGlobalBuffer((__gm__ float *)workspace + dvAddr);

    usedWorkspaceLen += dqWorkspaceLen + dkWorkspaceLen + dvWorkspaceLen;

    int64_t pseInnerAlibiSize = tilingData->opInfo.pseAlibiBaseS1 *
                                this->tilingData->opInfo.pseAlibiBaseS2 * sizeof(half);
    int64_t pseAlibiOffset =  CeilDiv(pseInnerAlibiSize, 512) * 512;

    uint64_t pseAlibiAddr = dvAddr + dvWorkspaceLen / sizeof(float);
    this->pseAlibiGm.SetGlobalBuffer((__gm__ half*)workspace + pseAlibiAddr * 2 +
                                    blockIdx * pseAlibiOffset / sizeof(half));
}


template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG,
                               LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::CopyoutWorkspace(const GlobalTensor<T1> &dstGm,
                                                         const LocalTensor<T1> &srcTensor, PingPongEmitInsn &insn)
{
    // send data to workspace which used as bmm's input
    // send shape is [rp.vS1Inner, rp.vS2Inner]
    // need consider DB
    DataCopyParams intriParams;
    if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
        intriParams.blockCount = insn.s2InnerAlign / C0_SIZE;
        intriParams.blockLen = insn.s1Inner * C0_SIZE * sizeof(T1);
        intriParams.srcStride = 1;
        intriParams.dstStride = (alignProcessM - insn.s1Inner) * C0_SIZE * sizeof(T1);
    } else {
        if (b16AlignProcessN == insn.s2InnerAlign) {
            intriParams.blockCount = 1;
            intriParams.blockLen = insn.s1Inner * insn.s2InnerAlign * sizeof(T1);
            intriParams.srcStride = 0;
            intriParams.dstStride = 0;
        } else {
            intriParams.blockCount = insn.s1Inner;
            intriParams.blockLen = insn.s2InnerAlign * sizeof(T1);
            intriParams.srcStride = 0;
            intriParams.dstStride = (b16AlignProcessN - insn.s2InnerAlign) * sizeof(T1);
        }
    }
    DataCopyPad(dstGm, srcTensor, intriParams);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline uint64_t
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG,
                               LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::InitL1Idx(const uint32_t &aIdx, const uint32_t &bIdx, 
                                                                                  const uint32_t matmulIdx,
                                                                                  const bool& reuseA, const bool& reuseB)
{    
    uint32_t AIdx = reuseA ? aIdx+5 : aIdx;
    uint32_t BIdx = reuseB ? bIdx+5 : bIdx;

    return AIdx*100 + matmulIdx*10 + BIdx;
}


template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG,
                            LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::SendMatmul2(const int64_t m, const int64_t n, const int64_t a_addr,
                                                    const int64_t b_addr, const int64_t org_m)
{
    if constexpr (L1CUSTOM) {
        bool reuseQ = false;
        bool reuseK = false;

        if(likely(currentLoop != 0)){
            if(isFirstSingleN && isFirstSingleM){
                qPingPongIdx = (qPingPongIdx + 1) % 2;
                kPingPongIdx = (kPingPongIdx + 1) % 2;
                qStartPingPongIdx = qPingPongIdx;
            }else if(isFirstSingleN && !isFirstSingleM){
                reuseK = true;
                qPingPongIdx = (qPingPongIdx + 1) % 2;
            }else if(!isFirstSingleN && isFirstSingleM){
                if(enableS1Reuse){
                    reuseQ = true;
                    qPingPongIdx = qStartPingPongIdx;
                }else{
                    qPingPongIdx = (qPingPongIdx + 1) % 2;
                }
                kPingPongIdx = (kPingPongIdx + 1) % 2;
            }else if(!isFirstSingleN && !isFirstSingleM){
                if(enableS1Reuse){
                    reuseQ = true;
                }
                reuseK = true;
                qPingPongIdx = (qPingPongIdx + 1) % 2;
            }
        }
        mm1.SetSelfDefineData(InitL1Idx(qPingPongIdx, kPingPongIdx, T12_MM_QK, reuseQ, reuseK));
    }

    if (mm2Scalar != n || mm2ScalarOrgM != org_m) {
        mm2Scalar = n;
        mm2ScalarOrgM = org_m;
        mm1.SetOrgShape(org_m, tilingData->mm1TilingData.N, tilingData->mm1TilingData.Ka, tilingData->mm1TilingData.Kb,
                        n);
    }
    mm1.SetTail(m, n, -1);
    mm1.SetTensorA(queryGm[a_addr]);
    mm1.SetTensorB(keyGm[b_addr], true);
    if constexpr (L1CUSTOM) {
        mm1.template IterateAll<false>(mm2WorkspaceGm[mmPingpongIdx * mm2WorkspaceLen / sizeof(T2) + mm2WorkspaceAddr], false, false, true);
    } else {
        mm1.template IterateAll<false>(mm2WorkspaceGm[mm2WorkspaceAddr], false, false, true);
    }

    mm1.End();
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG,
                            LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::SendMatmul1(const int64_t m, const int64_t n, const int64_t a_addr,
                                                    const int64_t b_addr, const int64_t org_m)
{
    if constexpr (L1CUSTOM) {
        bool reuseDy = false;
        if(currentLoop != 0){
            if(!enableS1Reuse || isFirstSingleN){
                dyPingPongIdx = (dyPingPongIdx + 1) % 2;
            }else{
                reuseDy = true;
                dyPingPongIdx = isFirstSingleM ? qStartPingPongIdx : (dyPingPongIdx + 1) % 2;
            }
            vPingPongIdx = (vPingPongIdx + 1) % 2;
        }
        mm1.SetSelfDefineData(InitL1Idx(dyPingPongIdx, vPingPongIdx, T12_MM_DYV, reuseDy, false));
    }

    if (mm1Scalar != n || mm1ScalarOrgM != org_m) {
        mm1Scalar = n;
        mm1ScalarOrgM = org_m;
        mm1.SetOrgShape(org_m, tilingData->mm1TilingData.N, tilingData->mm1TilingData.Ka, tilingData->mm1TilingData.Kb,
                        n);
    }
    mm1.SetTail(m, n, -1);
    mm1.SetTensorA(dyGm[a_addr]);
    mm1.SetTensorB(valueGm[b_addr], true);
    if constexpr (L1CUSTOM) {
        mm1.template IterateAll<false>(mm1WorkspaceGm[mmPingpongIdx * mm1WorkspaceLen / sizeof(T2) + mm1WorkspaceAddr], false, false, true);
    } else {
        mm1.template IterateAll<false>(mm1WorkspaceGm[mm1WorkspaceAddr], false, false, true);
    }

    mm1.End();
}

/*-----------------------------NewMatmulBEGIN---------------------------------------*/
template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG,
                               LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::SendMatmulDV(const uint32_t real_n, const uint32_t align_n,
                                                     const uint32_t s1_inner, const int64_t a_in_addr,
                                                     const int64_t b_in_addr, const int64_t out_addr,
                                                     const bool is_sync, const uint8_t kvAtomic)
{
    /*
    BNGSD:
      A: [rp.vS1Inner, realSingleN]
      B: [B, N1, S1, D]
      C: [rp.vS1Inner, realSingleN] x [B, N1, S1, D] = [B, N2, S2, D]
      For A, m = realSingleN, k = rp.vS1Inner
      For B, n = D, k = S1
      For C, m = S2, n = D
    */
    int64_t s1Size = dimS1;
    if constexpr (L1CUSTOM) {
        bool firstMM345 = currentLoop == 1;
        if(likely(!firstMM345)){
            if(enableS1Reuse && !haveS1Loop && haveS2Loop){
                if(dvConsum == vec.s2OuterOuterNum){
                    dvConsum = 0;
                    dvPingPongIdx = (dvPingPongIdx + 1) % 2;
                }
            }else{
                dvPingPongIdx = (dvPingPongIdx + 1) % 2;
            }
        }
        dvConsum++;
        mm4.SetSelfDefineData(InitL1Idx(mm345PingPangIdx % 2, dvPingPongIdx, T12_MM_DV, false, true));
    }
    if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
        s1Size = alignProcessM;
    }
    if (mmDVScalar != align_n) {
        mmDVScalar = align_n;
        if constexpr (LAYOUT == BNGSD) {
            mm4.SetOrgShape(align_n, dimD, s1Size, dimS1);
            if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
                mm4.SetUserDefInfo(dimS2);
            }
        } else if constexpr (LAYOUT == BSNGD) {
            if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
                mm4.SetOrgShape(align_n, dimD * dimN2 * dimG, s1Size, dimS1, dimD);
                mm4.SetUserDefInfo(dimS2);
            } else {
                mm4.SetOrgShape(align_n, dimD * dimN2 * dimG, s1Size, dimS1, dimD * dimN2);
            }
        } else if constexpr (LAYOUT == TND) {
            if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
                mm4.SetOrgShape(align_n, dimD * dimN2 * dimG, seqS1Current, seqS1Current, dimD);
                mm4.SetUserDefInfo(dimS2);
            } else {
                mm4.SetOrgShape(align_n, dimD * dimN2 * dimG, seqS1Current, seqS1Current, dimD * dimN2);
            }
        } else {
            if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
                mm4.SetOrgShape(align_n, dimD * dimN2 * dimG * dimB, s1Size, dimS1, dimD);
                mm4.SetUserDefInfo(dimS2);
            } else {
                mm4.SetOrgShape(align_n, dimD * dimN2 * dimG * dimB, s1Size, dimS1, dimD * dimN2 * dimB);
            }
        }
    }

    mm4.SetTail(real_n, -1, s1_inner);
    mm4.SetTensorA(mm4InputWorkspaceGm[a_in_addr], true);
    mm4.SetTensorB(dyGm[b_in_addr]);

    if constexpr (POST){
        if constexpr (!IsSameType<T1, float>::value) {
            mm4.template IterateAll<false>(dvWorkspaceGm[out_addr], kvAtomic, false, is_sync, false);
        }else {
            mm4.template IterateAll<false>(dvGm[out_addr], kvAtomic, false, is_sync, false);
        }
    }else{
        if (is_sync) {
            if constexpr (!IsSameType<T1, float>::value) {
                mm4.template IterateAll<true>(dvWorkspaceGm[out_addr], kvAtomic);
            } else {
                mm4.template IterateAll<true>(dvGm[out_addr], kvAtomic);
            }
        } else {
            if constexpr (!IsSameType<T1, float>::value) {
                mm4.template IterateAll<false>(dvWorkspaceGm[out_addr], kvAtomic);
            } else {
                mm4.template IterateAll<false>(dvGm[out_addr], kvAtomic);
            }
        }
    }
    mm4.End();
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG,
                               LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::SendMatmulDK(const uint32_t real_n, const uint32_t align_n,
                                                     const uint32_t s1_inner, const int64_t a_in_addr,
                                                     const int64_t b_in_addr, const int64_t out_addr,
                                                     const bool is_sync, const uint8_t kvAtomic)
{
    if constexpr (L1CUSTOM) {
        bool firstMM345 = currentLoop == 1;
        if(likely(!firstMM345)){
            if(enableS1Reuse && !haveS1Loop && haveS2Loop){
                if(dkConsum == vec.s2OuterOuterNum){
                    dkConsum = 0;
                    dkPingPongIdx = (dkPingPongIdx + 1) % 2;
                }
            }else{
                dkPingPongIdx = (dkPingPongIdx + 1) % 2;
            }
        }
        dkConsum++;
        mm4.SetSelfDefineData(InitL1Idx(mm345PingPangIdx % 2, dkPingPongIdx, T12_MM_DK, false, true));
    }
    mm4.SetTensorA(mm3InputWorkspaceGm[a_in_addr], true);
    mm4.SetTensorB(queryGm[b_in_addr]);

    if constexpr (POST){
        mm4.template IterateAll<false>(dkWorkspaceGm[out_addr], kvAtomic, false, is_sync, false);
    }else{
        if (is_sync) {
            mm4.template IterateAll<true>(dkWorkspaceGm[out_addr], kvAtomic);
        } else {
            mm4.template IterateAll<false>(dkWorkspaceGm[out_addr], kvAtomic);
        }
    }
    mm4.End();
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG,
                               LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::SendMatmulDQ(const uint32_t real_n, const uint32_t align_n,
                                                     const uint32_t s1_inner, const int64_t a_in_addr,
                                                     const int64_t b_in_addr, const int64_t out_addr,
                                                     const bool is_sync, const uint8_t qAtomic)
{
    /*
    BSH:
      A: [rp.vS1Inner, realSingleN]
      B: [B, N2, S2, D]
      C: [rp.vS1Inner, realSingleN] x [B, S2, N2, D] = [B, S1, N1, D]
      For A, m = rp.vS1Inner, k = realSingleN
      For B, n = N2 * D, k = S2
      For C, m = S1, n = N1 * D
    */
    int64_t s1Size = dimS1;
    if constexpr (L1CUSTOM) {
        bool firstMM345 = currentLoop == 1;
        if(likely(!firstMM345)){
            if(haveS1Loop){
                if(dqConsum == vec.s1OuterOuterNum){
                    dqConsum = 0;
                    dqPingPongIdx = (dqPingPongIdx + 1) % 2;
                }
            }else{
                dqPingPongIdx = (dqPingPongIdx + 1) % 2;
            }
        }
        dqConsum++;
        mm3_1.SetSelfDefineData(InitL1Idx(mm345PingPangIdx % 2, dqPingPongIdx, T12_MM_DQ, false, true));
    }
    if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
        s1Size = alignProcessM;
    }
    if (mmDQScalar != align_n) {
        mmDQScalar = align_n;
        if constexpr (MM_OUT_FORMAT == CubeFormat::NZ && MM2_OUT_FORMAT == CubeFormat::NZ) {
            mm3_1.SetUserDefInfo(dimS1);
        }
        if constexpr (LAYOUT == BNGSD) {
            mm3_1.SetOrgShape(s1Size, dimD, align_n, align_n);
        } else if constexpr (LAYOUT == BSNGD) {
            if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
                mm3_1.SetOrgShape(s1Size, dimD * dimN2, align_n, align_n, dimD);
            } else {
                mm3_1.SetOrgShape(s1Size, dimD * dimN2, align_n, align_n, dimD * dimN2 * dimG);
            }

        } else if constexpr (LAYOUT == TND) {
            if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
                mm3_1.SetOrgShape(seqS1Current, dimD * dimN2, align_n, align_n, dimD);
            } else {
                mm3_1.SetOrgShape(seqS1Current, dimD * dimN2, align_n, align_n, dimD * dimN2 * dimG);
            }
        } else {
            if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
                mm3_1.SetOrgShape(s1Size, dimD * dimN2 * dimB, align_n, align_n, dimD);
            } else {
                mm3_1.SetOrgShape(s1Size, dimD * dimN2 * dimB, align_n, align_n, dimD * dimN2 * dimG * dimB);
            }
        }
    }

    mm3_1.SetTail(s1_inner, -1, real_n);
    mm3_1.SetTensorA(mm3InputWorkspaceGm[a_in_addr]);
    mm3_1.SetTensorB(keyGm[b_in_addr]);

    if constexpr (POST){
        mm3_1.template IterateAll<false>(dqWorkspaceGm[out_addr], qAtomic, false, is_sync, false);
    }else{
        if (is_sync) {
            mm3_1.template IterateAll<true>(dqWorkspaceGm[out_addr], qAtomic);
        } else {
            mm3_1.template IterateAll<false>(dqWorkspaceGm[out_addr], qAtomic);
        }
    }
    mm3_1.End();
}

/*-----------------------------NewMatmulEND---------------------------------------*/

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG,
                LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::MTE2_ATMask(LocalTensor<uint8_t> &attenMaskTensor, int64_t &attenMaskOffset,
                                                    PingPongEmitInsn &insn)
{
    /*
    Func: move atten_mask to UB
    maskShapeType: 0 means: [S1, S2]
    maskShapeType: 1 means: [B, 1, 1, S1, S2]
    maskShapeType: 2 means: [B, N2, G, S1, S2]

    maskDataType: 0 means: dtype as same as T1
    maskDataType: 1 means: uint8
    */
    attenMaskTensor.SetSize(insn.s1Inner * insn.s2InnerAlign);
    DataCopyParams params(insn.s1Inner, insn.s2Inner, (attenMaskDimS2 - insn.s2Inner), 0);
    DataCopyPad(attenMaskTensor, attenMaskU8Gm[attenMaskOffset], params, {false, 0, 0, 0});
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG, LAYOUT,
     MM2_OUT_FORMAT, POST, L1CUSTOM>::MTE2_SFT(
    LocalTensor<T2> &sumTensor, LocalTensor<T2> &maxTensor, int64_t &sumMaxOffset, PingPongEmitInsn &insn)
{
    /*
    Func: move softmax_sum(fp32) to UB
    max_shape is fixed that [B, N2, G, S1, 8]
    */
    sumTensor.SetSize(insn.s1Inner * 8);
    maxTensor.SetSize(insn.s1Inner * 8);

    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = insn.s1Inner * 8 * sizeof(float);
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;
    DataCopyPad(sumTensor, softmaxSumGm[sumMaxOffset], intriParams, {false, 0, 0, 0});
    DataCopyPad(maxTensor, softmaxMaxGm[sumMaxOffset], intriParams, {false, 0, 0, 0});
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG,
                               LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::MTE2_STFGrad(GlobalTensor<T1> &gmTensor, int64_t addr,
                                                     LocalTensor<T1> &localTensor, int64_t num, int64_t count)
{
    // dy and attentionIn had same address
    int64_t srcStride = 0;
    int64_t num_align = (num + dataCopyBlockNum - 1) / dataCopyBlockNum * dataCopyBlockNum;
    if constexpr (LAYOUT == BNGSD) {
        srcStride = (dimD - num) * sizeof(T1);
    } else if constexpr (LAYOUT == SBNGD) {
        srcStride = (dimB * dimN2 * dimG * dimD - num) * sizeof(T1);
    } else {
        srcStride = (dimN2 * dimG * dimD - num) * sizeof(T1);
    }

    DataCopyPadExtParams<T1> padParams;
    DataCopyExtParams intriParams;
    if (LAYOUT == BNGSD && srcStride == 0 && num_align == num) {
        intriParams.blockCount = 1;
        intriParams.blockLen = count * num * sizeof(T1);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;
    } else {
        intriParams.blockCount = count;
        intriParams.blockLen = num * sizeof(T1);
        intriParams.srcStride = srcStride;
        intriParams.dstStride = 0;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = num_align - num;
        padParams.paddingValue = 0;
    }
    DataCopyPad(localTensor, gmTensor[addr], intriParams, padParams);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG, LAYOUT,
    MM2_OUT_FORMAT, POST, L1CUSTOM>::CastTo32(
    LocalTensor<T2> &dstTensor, LocalTensor<T1> &srcTensor, uint32_t count)
{
    pipe_barrier(PIPE_V);
    Cast(dstTensor, srcTensor, RoundMode::CAST_NONE, count); // 以前是s1_inner * rp.vS2InnerAlign
    pipe_barrier(PIPE_V);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG, LAYOUT,
    MM2_OUT_FORMAT, POST, L1CUSTOM>::CastTo16(
    LocalTensor<T1> &dstTensor, LocalTensor<T2> &srcTensor, uint32_t count)
{
    pipe_barrier(PIPE_V);
    Cast(dstTensor, srcTensor, RoundMode::CAST_ROUND, count); // 以前是s1_inner * rp.vS2InnerAlign
    pipe_barrier(PIPE_V);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG, LAYOUT,
    MM2_OUT_FORMAT, POST, L1CUSTOM>::DoMaskU8(
    LocalTensor<T2> &dstTensor, LocalTensor<uint8_t> &attenMaskTensor, LocalTensor<uint8_t> &helpTensor,
    PingPongEmitInsn &insn, const uint8_t maskType)
{
    /*
    Func:
    1. 仅支持T1场景，待支持UINT8场景
    2. attenMaskTensor FP16 使用 DataMovePad 完成脏数据清0
    maskDataType: 0 means: dtype as same as T1
    maskDataType: 1 means: uint8
    */
    // uint8
    T2 scalar;
    if constexpr (IsSameType<T2, float>::value) {
        uint32_t tmp = 0xFF7FFFFF;
        scalar = *((float *)&tmp);
    } else {
        uint16_t tmp = 0xFBFF;
        scalar = *((half *)&tmp);
    }

    SelectWithBytesMaskShapeInfo info;
    info.firstAxis = insn.s1Inner;
    info.srcLastAxis = insn.s2InnerAlign;
    info.maskLastAxis = (insn.s2InnerAlign + 31) / 32 * 32;
    dstTensor.SetSize(info.firstAxis * info.srcLastAxis);
    attenMaskTensor.SetSize(info.firstAxis * info.maskLastAxis);
    if (maskType == 0) {
        SelectWithBytesMask(dstTensor, dstTensor, scalar, attenMaskTensor, helpTensor, info);
    } else {
        SelectWithBytesMask(dstTensor, scalar, dstTensor, attenMaskTensor, helpTensor, info);
    }
    pipe_barrier(PIPE_V);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG,
                     LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::DoSimpleSoftMax(LocalTensor<T2> &dstTensor, LocalTensor<float> &sumTensor,
                                                        LocalTensor<float> &maxTensor, LocalTensor<uint8_t> &helpTensor,
                                                        PingPongEmitInsn &insn)
{
    /*
    Func:
    1. sum/std::max shape is [S1, 8] FP32
    2. maxTensor had been stored
    */
    sumTensor.SetSize(insn.s1Inner * 8);
    maxTensor.SetSize(insn.s1Inner * 8);

    // set shape
    uint32_t softmaxShape[2] = {static_cast<uint32_t>(insn.s1Inner), 8};
    sumTensor.SetShapeInfo(ShapeInfo(2, softmaxShape, DataFormat::ND));
    maxTensor.SetShapeInfo(ShapeInfo(2, softmaxShape, DataFormat::ND));

    uint32_t dstSoftShape[2] = {static_cast<uint32_t>(insn.s1Inner), static_cast<uint32_t>(insn.s2InnerAlign)};
    dstTensor.SetShapeInfo(ShapeInfo(2, dstSoftShape, DataFormat::ND));

    // calc reused buffer
    if ((insn.s1Inner % 8 == 0) && (insn.s2InnerAlign % 64 == 0)) {
        SimpleSoftMax<T2, true, true>(dstTensor, sumTensor, maxTensor, dstTensor, helpTensor,
                                      tilingData->softmaxTilingData);
    } else {
        SimpleSoftMax<T2, true, false>(dstTensor, sumTensor, maxTensor, dstTensor, helpTensor,
                                       tilingData->softmaxTilingData);
    }
    pipe_barrier(PIPE_V);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG, LAYOUT,
    MM2_OUT_FORMAT, POST, L1CUSTOM>::FullGrad(
    LocalTensor<T2> &dstTensor)
{
    // 1. D通道不切分场景
    // 2. FP16/BFP16 cast 为FP32计算
    // 3. dstTensor为引用，用来指向最终结果
    // 4. que:softmaxGradOutQue
    // 5. 每次UB仅能处理 [sft.baseM, sft.dInner], 累计处理loop_次
    uint32_t times_ = (rp.SFTS1Inner + sft.baseM - 1) / sft.baseM;
    uint32_t s1Inner_ = sft.baseM;

    uint32_t s1InnerTail_ = (rp.SFTS1Inner % sft.baseM == 0) ? sft.baseM : (rp.SFTS1Inner % sft.baseM);
    uint32_t dAlign_ = (dimD + dataCopyBlockNum - 1) / dataCopyBlockNum * dataCopyBlockNum;

    event_t mte2WaitVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t mte2WaitVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t vWaitMte2Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    event_t vWaitMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());

    for (uint32_t loop = 0; loop < times_; loop++) {
        auto parTensor = dstTensor[loop * sft.baseM * 8];
        parTensor.SetSize(s1Inner_ * calcBlockNum);
        if (loop == times_ - 1 && s1InnerTail_ != 0) {
            s1Inner_ = s1InnerTail_;
        }
        // params
        MMOffsetTensorA(s1Index + loop * sft.baseM, dyGmAddr);

        // operation
        if (loop > 0) {
            wait_flag(PIPE_V, PIPE_MTE2, mte2WaitVPong);
        }
        if constexpr (!IsSameType<T1, float>::value) {
            dyT1Tensor.SetSize(s1Inner_ * dAlign_);
            dyT2Tensor.SetSize(s1Inner_ * dAlign_);
            MTE2_STFGrad(dyGm, dyGmAddr, dyT1Tensor, dimD, s1Inner_);
        } else {
            dyT2Tensor.SetSize(s1Inner_ * dAlign_);
            MTE2_STFGrad(dyGm, dyGmAddr, dyT2Tensor, dimD, s1Inner_);
        }
        set_flag(PIPE_MTE2, PIPE_V, vWaitMte2Pong);
        wait_flag(PIPE_MTE2, PIPE_V, vWaitMte2Pong);

        if constexpr (!IsSameType<T1, float>::value) {
            CastTo32(dyT2Tensor, dyT1Tensor, s1Inner_ * dAlign_);
            if (loop < times_ - 1) {
                set_flag(PIPE_V, PIPE_MTE2, mte2WaitVPong);
            }
        }

        // params
        attentionInGmAddr = dyGmAddr;

        // operation
        if (loop > 0) {
            wait_flag(PIPE_V, PIPE_MTE2, mte2WaitVPing);
        }
        if constexpr (!IsSameType<T1, float>::value) {
            attentionT1Tensor.SetSize(s1Inner_ * dAlign_);
            attentionT2Tensor.SetSize(s1Inner_ * dAlign_);
            MTE2_STFGrad(attentionInGm, attentionInGmAddr, attentionT1Tensor, dimD, s1Inner_);
        } else {
            attentionT2Tensor.SetSize(s1Inner_ * dAlign_);
            MTE2_STFGrad(attentionInGm, attentionInGmAddr, attentionT2Tensor, dimD, s1Inner_);
        }
        set_flag(PIPE_MTE2, PIPE_V, vWaitMte2Ping);
        wait_flag(PIPE_MTE2, PIPE_V, vWaitMte2Ping);
        if constexpr (!IsSameType<T1, float>::value) {
            CastTo32(attentionT2Tensor, attentionT1Tensor, s1Inner_ * dAlign_);
            if (loop < times_ - 1) {
                set_flag(PIPE_V, PIPE_MTE2, mte2WaitVPing);
            }
        } else {
            pipe_barrier(PIPE_V);
        }

        // params
        uint32_t dyShape[2] = {static_cast<uint32_t>(s1Inner_), static_cast<uint32_t>(dAlign_)};
        uint32_t attentionInShape[2] = {static_cast<uint32_t>(s1Inner_), static_cast<uint32_t>(dAlign_)};
        uint32_t softmaxGradShape[2] = {static_cast<uint32_t>(s1Inner_), static_cast<uint32_t>(calcBlockNum)};
        parTensor.SetShapeInfo(ShapeInfo(2, softmaxGradShape, DataFormat::ND));
        dyT2Tensor.SetShapeInfo(ShapeInfo(2, dyShape, DataFormat::ND));
        attentionT2Tensor.SetShapeInfo(ShapeInfo(2, attentionInShape, DataFormat::ND));

        // operation
        if (unlikely((s1Inner_ % 8 == 0) && (dimD % 64 == 0))) {
            SoftmaxGradFront<T2, true>(parTensor, dyT2Tensor, attentionT2Tensor, helpTensor,
                                       tilingData->softmaxGradTilingData);
        } else {
            SoftmaxGradFront<T2, false>(parTensor, dyT2Tensor, attentionT2Tensor, helpTensor,
                                        tilingData->softmaxGradTilingData);
        }

        if constexpr (IsSameType<T1, float>::value) {
            if (loop < times_ - 1) {
                set_flag(PIPE_V, PIPE_MTE2, mte2WaitVPing);
            }

            if (loop < times_ - 1) {
                set_flag(PIPE_V, PIPE_MTE2, mte2WaitVPong);
            }
        }
    }
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(mte2WaitVPing);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(mte2WaitVPong);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(vWaitMte2Ping);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(vWaitMte2Pong);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG, LAYOUT,
    MM2_OUT_FORMAT, POST, L1CUSTOM>::SplitGrad(
    LocalTensor<T2> &dstTensor)
{
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG,
                                DROPOUT_CFG, LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::DoSoftmaxGrad(LocalTensor<T2> &dstTensor)
{
    if (likely(sft.dInner >= dimD)) {
        FullGrad(dstTensor);
    } else {
        SplitGrad(dstTensor);
    }
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG, LAYOUT,
    MM2_OUT_FORMAT, POST, L1CUSTOM>::DoSub(
    LocalTensor<T2> &dstTensor, LocalTensor<T2> &srcTensor, PingPongEmitInsn &insn)
{
    /*
    需要支持：非对齐，bf16/fp16数据类型
    输入：a(S1,S2) b(S1,P)
    输出：out(S1,S2)
    计算逻辑：S1方向repeat，S2方向for
    当前仅支持（B，N2，G，S1，D) fp16, D不切分,S1<=255;
    */
    dstTensor.SetSize(insn.s1Inner * insn.s2InnerAlign);
    int64_t s2_for_time = insn.s2InnerAlign / calcBlockNum / 8;
    int64_t s2_for_remain = insn.s2InnerAlign / calcBlockNum % 8;
    int64_t mask_sub = 8 * calcBlockNum;
    for (int64_t i = 0; i < s2_for_time; i++) {
        Sub(dstTensor[i * mask_sub], dstTensor[i * mask_sub], srcTensor, mask_sub, insn.s1Inner,
            {1, 1, 0, (uint8_t)(insn.s2InnerAlign / calcBlockNum), (uint8_t)(insn.s2InnerAlign / calcBlockNum), 1});
    }
    if (s2_for_remain) {
        Sub(dstTensor[s2_for_time * mask_sub], dstTensor[s2_for_time * mask_sub], srcTensor, s2_for_remain * 8,
            insn.s1Inner,
            {1, 1, 0, (uint8_t)(insn.s2InnerAlign / calcBlockNum), (uint8_t)(insn.s2InnerAlign / calcBlockNum), 1});
    }
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG, LAYOUT,
    MM2_OUT_FORMAT, POST, L1CUSTOM>::DoMul(
    LocalTensor<T2> &dstTensor, LocalTensor<T2> &srcTensor, PingPongEmitInsn &insn)
{
    pipe_barrier(PIPE_V);
    Mul(dstTensor, dstTensor, srcTensor, insn.s1Inner * insn.s2InnerAlign);
    pipe_barrier(PIPE_V);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG,
                                        DROPOUT_CFG, LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::DoMulsScale(LocalTensor<T2> &dstTensor,
                                                                                        PingPongEmitInsn &insn)
{
    pipe_barrier(PIPE_V);
    Muls(dstTensor, dstTensor, scaleValue, insn.s1Inner * insn.s2InnerAlign);
    pipe_barrier(PIPE_V);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG,
                LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::CalcSparseIdx(const int64_t bIndex, const int64_t s1Idx, const int64_t s1Size,
                                                      int64_t &s2_start_idx, int64_t &s2_end_idx)
{
    /*
             s2
       _____________________
       |              \    |
       |               \   |
    s1 |\               \  |
       | \               \ |
       |  \               \|
       |___\_______________|

    ---------------sparse暂只支持s2方向MM配比1:1--------------------

    s2_start_idx 表示基本块上边对应行与左下对角线的交点
    s2_end_idx   表示基本块下边对应行与右上对角线的交点
    如果基本块右上角s2索引 <= s2_start_idx 该基本块无需计算
    如果基本块左下角s2索引 >= s2_end_idx 该基本块无需计算
    */

    int64_t S1 = (LAYOUT == TND) ? seqS1Current : dimS1;
    int64_t S2 = (LAYOUT == TND) ? seqS2Current : dimS2;
    unpadUseLeftUpCasual = tilingData->opInfo.sparseMode == 8 && bIndex != bandIdx;
    unpadUseRightDownCasual = tilingData->opInfo.sparseMode == 7 && bIndex != bandIdx;
    unpadUseBand = (tilingData->opInfo.sparseMode == 7 && bIndex == bandIdx) ||
                   (tilingData->opInfo.sparseMode == 8 && bIndex == bandIdx);
    if (tilingData->opInfo.sparseMode == 3 || tilingData->opInfo.sparseMode == 5 ||
        tilingData->opInfo.sparseMode == 6 || unpadUseRightDownCasual) {
        s2_start_idx = 0;
        int64_t s2IgnoredEndLen = S1 - (s1Idx + s1Size);
        int64_t s2EndLen = 0;
        if (S2 > s2IgnoredEndLen) {
            s2EndLen = S2 - s2IgnoredEndLen;
        } else {
            s2EndLen = 0;
        }
        if (tilingData->opInfo.sparseMode == 5 || tilingData->opInfo.sparseMode == 6) {
            int64_t prefix_n = ((__gm__ int64_t *)prefixN_addr)[bIndex];
            int64_t alignPrefixN = (prefix_n + cube.baseN - 1) / cube.baseN * cube.baseN;
            s2EndLen = s2EndLen >= alignPrefixN ? s2EndLen : alignPrefixN;
        }
        s2_end_idx = s2EndLen;
        s2_end_idx = (s2_end_idx > S2) ? S2 : s2_end_idx;
    } else if (tilingData->opInfo.sparseMode == LEFT_UP_CAUSAL || unpadUseLeftUpCasual) {
        s2_start_idx = 0;
        s2_end_idx = (((s1Idx + s1Size) >= S2)) ? S2 : int64_t(s1Idx + s1Size);
    } else {
        s2_start_idx = (int64_t(s1Idx) <= preTokens) ? 0 : (int64_t(s1Idx) - preTokens);
        // s1Idx + s1Inner表示基本块下边s1索引
        int64_t s2_end =
            ((S2 < nextTokens) || ((s1Idx + s1Size) >= (S2 - nextTokens))) ? S2 : ((s1Idx + s1Size) + nextTokens);
        s2_end_idx = (s2_end < 0) ? 0 : s2_end;
    }
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG,
                                        DROPOUT_CFG, LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::MMOffsetTensorA(const int64_t s1_idx,
                                                                                            int64_t &a_addr)
{
    /*
    BNGSD = 0, // A: B N2 G S1 D
    SBNGD = 1, //  A: S1 B N2 G D
    BSNGD = 2, // A: B S1 N2 G D
    */

    if constexpr (LAYOUT == BSNGD) {
        a_addr = bIndex * (dimS1 * dimN2 * dimG * dimD) + s1_idx * (dimN2 * dimG * dimD) + n2Index * (dimG * dimD) +
                 gIndex * dimD;
        return;
    }
    if constexpr (LAYOUT == SBNGD) {
        a_addr = s1_idx * (dimB * dimN2 * dimG * dimD) + bIndex * (dimN2 * dimG * dimD) + n2Index * (dimG * dimD) +
                 gIndex * dimD;
        return;
    }
    if constexpr (LAYOUT == BNGSD) {
        a_addr = bIndex * (dimN2 * dimG * dimS1 * dimD) + n2Index * (dimG * dimS1 * dimD) + gIndex * (dimS1 * dimD) +
                 s1_idx * dimD;
        return;
    }
    if constexpr (LAYOUT == TND) {
        a_addr = seqS1CurrentOffset * (dimN2 * dimG * dimD) + s1_idx * (dimN2 * dimG * dimD) + n2Index * dimG * dimD +
                 gIndex * dimD;
        return;
    }
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG,
                                        DROPOUT_CFG, LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::MMOffsetTensorB(const int64_t s2_idx,
                                                                                            int64_t &b_addr)
{
    /*
    BNGSD = 0, // B: B N2 1 S2 D
    SBNGD = 1, //  B: S2 B N2 1 D
    BSNGD = 2, // B: B S2 N2 1 D
    */

    if constexpr (LAYOUT == BSNGD) {
        b_addr = bIndex * (dimS2 * dimN2 * dimD) + s2_idx * (dimN2 * dimD) + n2Index * dimD;
        return;
    }
    if constexpr (LAYOUT == SBNGD) {
        b_addr = s2_idx * (dimB * dimN2 * dimD) + bIndex * (dimN2 * dimD) + n2Index * dimD;
        return;
    }
    if constexpr (LAYOUT == BNGSD) {
        b_addr = bIndex * (dimN2 * dimS2 * dimD) + n2Index * (dimS2 * dimD) + s2_idx * dimD;
        return;
    }
    if constexpr (LAYOUT == TND) {
        b_addr = seqS2CurrentOffset * (dimN2 * dimD) + s2_idx * (dimN2 * dimD) + n2Index * dimD;
        return;
    }
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG,
                               LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::MMOffsetNzOut(const int64_t s1_idx, const int64_t s2_idx,
                                                      int64_t &a_addr, int64_t &b_addr)
{
    a_addr = bIndex * (dimN2 * dimG * dimS1 * dimDAlign) + n2Index * (dimG * dimS1 * dimDAlign) +
                    gIndex * (dimS1 * dimDAlign) + s1_idx * C0_SIZE;
    b_addr = bIndex * (dimN2 * dimS2 * dimDAlign) + n2Index * (dimS2 * dimDAlign) +
                    s2_idx * C0_SIZE;
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG,
                    LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::CalcCausalAttenMaskOffset(int64_t &attenMaskOffset, const int64_t delta,
                                                                  bool isPingMode)
{
    int64_t s1Idx = isPingMode ? rp.vPingS1Inner : rp.vPongS1Inner;
    int64_t s2Idx = isPingMode ? rp.vPingS2Inner : rp.vPongS2Inner;
    if (delta == 0) {
        attenMaskOffset = 0;
    } else if (delta < 0) {
        if (-delta > s1Idx) {
            attenMaskOffset = s1Idx;
        } else {
            attenMaskOffset = -delta;
        }
    } else {
        if (delta > s2Idx) {
            attenMaskOffset = s2Idx * attenMaskDimS2;
        } else {
            attenMaskOffset = delta * attenMaskDimS2;
        }
    }
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG,
                LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::CalcBandAttenMaskOffset(int64_t &attenMaskOffsetPre, int64_t &attenMaskOffset,
                                                                const int64_t delta, bool isPingMode)
{
    int64_t final_delta = delta - preTokens - 1;
    CalcCausalAttenMaskOffset(attenMaskOffsetPre, final_delta, isPingMode);
    final_delta = delta + nextTokens;
    CalcCausalAttenMaskOffset(attenMaskOffset, final_delta, isPingMode);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG,
                                LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::CalcPrefixCompressAttenMaskOffset(int64_t &attenMaskOffsetPre,
                                                                          int64_t &attenMaskOffset, const int64_t delta,
                                                                          int64_t s2Idx, bool isPingMode)
{
    bool &prefixCompressCanSimplify =
        isPingMode ? rp.vPingPrefixCompressCanSimplify : rp.vPongPrefixCompressCanSimplify;
    prefixCompressCanSimplify = false;
    uint32_t s1VSize = isPingMode ? rp.vPingS1Inner : rp.vPongS1Inner;
    uint32_t s2VSize = isPingMode ? rp.vPingS2Inner : rp.vPongS2Inner;
    int64_t N = ((__gm__ int64_t *)prefixN_addr)[bIndex];
    int64_t S1 = (LAYOUT == TND) ? seqS1Current : (int64_t)dimS1;
    int64_t S2 = (LAYOUT == TND) ? seqS2Current : (int64_t)dimS2;

    // s1 + N <= s2, equivalent to RightDownCausal
    if (S1 + N <= S2) {
        prefixCompressCanSimplify = true;
        int64_t causal_delta = delta - S1 + S2;
        CalcCausalAttenMaskOffset(attenMaskOffset, causal_delta, isPingMode);
        return;
    }

    int64_t delta1 = delta - S1 + S2;
    int64_t delta2 = N + 1 - (int64_t)s2Idx;

    // Y + n <= N, return mask2 offset directly
    if (delta2 > (int64_t)s2VSize) {
        prefixCompressCanSimplify = true;
        attenMaskOffset = PREFIX_COMPRESS_CAUSAL_S_SIZE * attenMaskDimS2;
        return;
    }

    // other, mask = mask1 & mask2, need calculate two mask offsets
    // mask1 part
    if (delta1 >= 0) {
        attenMaskOffset = (delta1 <= s2VSize) ? delta1 * (int64_t)attenMaskDimS2 : s2VSize * (int64_t)attenMaskDimS2;
    } else {
        attenMaskOffset = (-delta1 <= s1VSize) ? -delta1 : s1VSize;
    }

    // mask2 part
    int64_t offsetStartPos =
        (int64_t)PREFIX_COMPRESS_CAUSAL_S_SIZE * (int64_t)attenMaskDimS2 + (int64_t)PREFIX_COMPRESS_ALL_MASK_S1_SIZE;
    attenMaskOffsetPre = (delta2 > 0) ? (offsetStartPos - delta2 + 1) : offsetStartPos;
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG,
                               LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::CopyInOffsetForSimpleSoftmax(int64_t s1Idx, bool isPingMode)
{
    int64_t softmax_max_sum_in_block_num = BLOCK / sizeof(float);
    int64_t &sumMaxAddr = isPingMode ? softmaxMaxSumPingAddress : softmaxMaxSumPongAddress;
    sumMaxAddr = bIndex * (dimN2 * dimG * dimS1 * softmax_max_sum_in_block_num) +
                 n2Index * (dimG * dimS1 * softmax_max_sum_in_block_num) +
                 gIndex * (dimS1 * softmax_max_sum_in_block_num) + s1Idx * softmax_max_sum_in_block_num;
    if constexpr (LAYOUT == TND) {
        int64_t dimT_q_offset = dimN2 * dimG * seqS1CurrentOffset * softmax_max_sum_in_block_num;
        sumMaxAddr = dimT_q_offset + n2Index * (dimG * seqS1Current * softmax_max_sum_in_block_num) +
                     gIndex * (seqS1Current * softmax_max_sum_in_block_num) + s1Idx * softmax_max_sum_in_block_num;
    }
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG,
                                        DROPOUT_CFG, LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::CopyInOffset(int64_t s1Idx, int64_t s2Idx,
                                                                                         bool isPingMode)
{
    /*
    s1Idx: s1Index, s2Idx: s2Index, isPingMode: ping or pong.
    pseShapeType = 0: 代表PSE的输入是[B,N2,G,S1,S2]
    pseShapeType = 1: 代表PSE的输入是[B,N2,G,1,S2]
    pseShapeType = 2: 代表PSE的输入是[1,N2,G,S1,S2]
    pseShapeType = 3: 代表PSE的输入是[B,N2,G,H,S2]
    pseShapeType = 4: 代表PSE的输入是[1,N2,G,H,S2]

    maskShapeType = 0; 代表attenMask的输入是[S1,S2]
    maskShapeType = 1; 代表attenMask的输入是[B,1,1,S1,S2]
    maskShapeType = 2; 代表attenMask的输入是[B,N,G,S1,S2]

    maskDataType: 0 代表数据类型 与普通Q\K\V输入Type一致（他们要么一起是FP16, 要么一起FP32, 要么一起BFP16）
    maskDataType: 1 代表数据类型是UINT8

    drop_mask: 当前认为已经是bool 1:1对应mm输出
    */
    if ((isPingMode && !pingOK) || (!isPingMode && !pongOK)) {
        return;
    }

    int64_t realS1, realS2;
    if constexpr (LAYOUT != TND) {
        realS1 = dimS1;
        realS2 = dimS2;
    } else {
        realS1 = seqS1Current;
        realS2 = seqS2Current;
    }

    int64_t &attenmaskAddress = isPingMode ? attenmaskPingAddress : attenmaskPongAddress;
    int64_t &dropmaskAddress = isPingMode ? dropmaskPingAddress : dropmaskPongAddress;
    int64_t &attenmaskPreAddress = isPingMode ? attenmaskPingPreAddress : attenmaskPongPreAddress;
    AttenMaskCompress &AttenBandMode = isPingMode ? AttenBandPingMode : AttenBandPongMode;

    int64_t causal_delta = (int64_t)s1Idx - (int64_t)s2Idx;
    if constexpr (!L1CUSTOM) {
        if (isLeftRightBandCausal) {
            int64_t next_delta = causal_delta;
            int64_t pre_delta = causal_delta - INT32_MAX - 1;
            if (compressMode == 1 || unpadUseLeftUpCasual) {
            } else if (compressMode == 2 || unpadUseRightDownCasual) {
                next_delta = causal_delta - realS1 + realS2;
            } else {
                next_delta = causal_delta + nextTokens;
                pre_delta = causal_delta - preTokens - 1;
            }

            uint32_t s1Inner = isPingMode ? rp.vPingS1Inner : rp.vPongS1Inner;
            uint32_t s2Inner = isPingMode ? rp.vPingS2Inner : rp.vPongS2Inner;
            bool NoNext = (next_delta - s2Inner >= 0);
            bool NoPre = (pre_delta + 1 + s1Inner <= 0);

            if (NoNext && NoPre) {
                AttenBandMode = AttenMaskCompress::Empty;
            } else if (NoNext && !NoPre) {
                AttenBandMode = AttenMaskCompress::PreOnly;
            } else if (!NoNext && NoPre) {
                AttenBandMode = AttenMaskCompress::NextOnly;
            } else {
                AttenBandMode = AttenMaskCompress::All;
            }
        }
    }

    if constexpr (!L1CUSTOM) {
        if (compressMode == 1 || unpadUseLeftUpCasual) {
            // causal s1==s2
            CalcCausalAttenMaskOffset(attenmaskAddress, causal_delta, isPingMode);
        } else if (compressMode == 2 || unpadUseRightDownCasual) {
            // causal s1!=s2
            causal_delta = causal_delta - realS1 + realS2;
            CalcCausalAttenMaskOffset(attenmaskAddress, causal_delta, isPingMode);
        } else if (compressMode == BAND_MODE || unpadUseBand) {
            CalcBandAttenMaskOffset(attenmaskPreAddress, attenmaskAddress, causal_delta, isPingMode);
        } else if (compressMode == PREFIX_COMPRESS_MODE) {
            // prefix compress
            CalcPrefixCompressAttenMaskOffset(attenmaskPreAddress, attenmaskAddress, causal_delta, s2Idx, isPingMode);
        } else if (maskShapeType == 2) {
            attenmaskAddress = bIndex * (dimN2 * dimG * dimS1 * dimS2) + n2Index * (dimG * dimS1 * dimS2) +
                            gIndex * dimS1 * dimS2 + s1Idx * dimS2 + s2Idx;
        } else if (maskShapeType == 1) {
            attenmaskAddress = bIndex * (dimS1 * dimS2) + s1Idx * dimS2 + s2Idx;
        } else {
            attenmaskAddress = s1Idx * dimS2 + s2Idx;
        }
    }

    if constexpr (LAYOUT == TND) {
        dropmaskAddress = 0;
        dropmaskAddress += dimN2 * dimG * seqS1S2ProductSum;
        dropmaskAddress += (n2Index * dimG + gIndex) * seqS1Current * seqS2Current + s1Idx * seqS2Current + s2Idx;
    } else {
        dropmaskAddress = bIndex * (dimN2 * dimG * dimS1 * dimS2) + n2Index * (dimG * dimS1 * dimS2) +
                          gIndex * dimS1 * dimS2 + s1Idx * dimS2 + s2Idx;
    }
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG, LAYOUT,
    MM2_OUT_FORMAT, POST, L1CUSTOM>::MTE2ForMM2(
    LocalTensor<T2> &mm2TensorCurr, int64_t &mm2Offset, PingPongEmitInsn &insn)
{
    /*
    Func:
    1. WaitIterateAll: until mm finish.

    */
    if (rp.SFTLoopS1 == 0 && insn.vLoopS1 == 0 && insn.vLoopS2 == 0) {
        mm1.WaitIterateAll();
    }

    DataCopyParams intriParams;
    if (b32AlignProcessN == insn.s2InnerAlign) {
        intriParams.blockCount = 1;
        intriParams.blockLen = insn.s1Inner * insn.s2InnerAlign * sizeof(float);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
    } else {
        intriParams.blockCount = insn.s1Inner;
        intriParams.blockLen = insn.s2Inner * sizeof(float);
        intriParams.srcStride = (b32AlignProcessN - insn.s2Inner) * sizeof(float);
        intriParams.dstStride = (insn.s2InnerAlign - insn.s2Inner) * sizeof(float) / BLOCK;
    }
    if constexpr (L1CUSTOM) {
        DataCopyPad(mm2TensorCurr, mm2WorkspaceGm[mmPingpongIdx * mm2WorkspaceLen / sizeof(T2) + mm2Offset], intriParams, {false, 0, 0, 0});
    } else {
        DataCopyPad(mm2TensorCurr, mm2WorkspaceGm[mm2Offset], intriParams, {false, 0, 0, 0});
    }
    
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG, LAYOUT,
    MM2_OUT_FORMAT, POST, L1CUSTOM>::MTE2ForMM1(
    LocalTensor<T2> &mm1TensorCurr, int64_t &mm1Offset, PingPongEmitInsn &insn)
{
    /*
    Func:
    1. WaitIterateAll: until mm finish.
    */
    if (rp.SFTLoopS1 == 0 && insn.vLoopS1 == 0 && insn.vLoopS2 == 0) {
        mm1.WaitIterateAll();
    }

    DataCopyParams intriParams;
    if (b32AlignProcessN == insn.s2InnerAlign) {
        intriParams.blockCount = 1;
        intriParams.blockLen = insn.s1Inner * insn.s2InnerAlign * sizeof(float);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
    } else {
        intriParams.blockCount = insn.s1Inner;
        intriParams.blockLen = insn.s2Inner * sizeof(float);
        intriParams.srcStride = (b32AlignProcessN - insn.s2Inner) * sizeof(float);
        intriParams.dstStride = (insn.s2InnerAlign - insn.s2Inner) * sizeof(float) / BLOCK;
    }
    if constexpr (L1CUSTOM) {
        DataCopyPad(mm1TensorCurr, mm1WorkspaceGm[mmPingpongIdx * mm1WorkspaceLen / sizeof(T2) + mm1Offset], intriParams, {false, 0, 0, 0});
    } else {
        DataCopyPad(mm1TensorCurr, mm1WorkspaceGm[mm1Offset], intriParams, {false, 0, 0, 0});
    }
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG, LAYOUT,
    MM2_OUT_FORMAT, POST, L1CUSTOM>::NZ2ND(
    LocalTensor<T2> &ndTensor, LocalTensor<T2> &nzTensor, PingPongEmitInsn &insn)
{
    /*
    Func:
    1. Matmul output format is NZ
    2. UB->UB，将NZ分形数据转换为ND，[N, C1，C0] -> [C1，C0, N]
    3. 限制：rp.vS1Inner <= 255, 代码没有写repeat>255的分支
    */
    DataCopy(nzTensor, ndTensor, insn.s2InnerAlign * insn.s1Inner + insn.s2InnerAlign / C0_SIZE * B32_BLOCK_NUM);
    pipe_barrier(PIPE_V);

    CopyRepeatParams nz2ndParams;
    nz2ndParams.srcStride = insn.s1Inner * C0_SIZE / B32_BLOCK_NUM + 1;
    nz2ndParams.dstStride = C0_SIZE / B32_BLOCK_NUM;
    nz2ndParams.srcRepeatSize = C0_SIZE / B32_BLOCK_NUM;
    nz2ndParams.dstRepeatSize = insn.s2InnerAlign / B32_BLOCK_NUM;

    uint16_t c0_repeat = C0_SIZE / B32_BLOCK_NUM;
    uint16_t c1_repeat = insn.s2InnerAlign / C0_SIZE / VEC_REPEAT;
    uint16_t c1_remain = insn.s2InnerAlign / C0_SIZE % VEC_REPEAT;
    uint16_t n_repeat = insn.s1Inner;
    for (uint16_t i = 0; i < c0_repeat; ++i) {
        for (uint16_t j = 0; j < c1_repeat; ++j) {
            Copy(ndTensor[i * B32_BLOCK_NUM + j * VEC_REPEAT * C0_SIZE],
                 nzTensor[i * B32_BLOCK_NUM + j * VEC_REPEAT * (insn.s1Inner * C0_SIZE + B32_BLOCK_NUM)],
                 VEC_REPEAT * B32_BLOCK_NUM, n_repeat, nz2ndParams);
        }
        if (c1_remain > 0) {
            Copy(ndTensor[i * B32_BLOCK_NUM + c1_repeat * VEC_REPEAT * C0_SIZE],
                 nzTensor[i * B32_BLOCK_NUM + c1_repeat * VEC_REPEAT * (insn.s1Inner * C0_SIZE + B32_BLOCK_NUM)],
                 VEC_REPEAT * c1_remain, n_repeat, nz2ndParams);
        }
    }
    pipe_barrier(PIPE_V);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG, LAYOUT,
    MM2_OUT_FORMAT, POST, L1CUSTOM>::ND2NZ(
    LocalTensor<T1> &nzTensor, LocalTensor<T1> &ndTensor, PingPongEmitInsn &insn)
{
    /*
    Func:
    1. Matmul output format is ND
    2. UB->UB，将ND分形数据转换为NZ，[C1，C0, N]->[N, C1，C0]
    3. 限制：rp.vS1Inner <= 255, 代码没有写repeat>255的分支
    */
    CopyRepeatParams nd2nzParams;
    nd2nzParams.dstStride = insn.s1Inner * C0_SIZE / dataCopyBlockNum + 1;
    nd2nzParams.srcStride = C0_SIZE / dataCopyBlockNum;
    nd2nzParams.dstRepeatSize = C0_SIZE / dataCopyBlockNum;
    nd2nzParams.srcRepeatSize = insn.s2InnerAlign / dataCopyBlockNum;

    uint16_t c1_repeat = insn.s2InnerAlign / C0_SIZE / VEC_REPEAT;
    uint16_t c1_remain = insn.s2InnerAlign / C0_SIZE % VEC_REPEAT;

    auto nzTensorTmp = nzTensor.template ReinterpretCast<half>();
    auto ndTensorTmp = ndTensor.template ReinterpretCast<half>();

    for (uint16_t j = 0; j < c1_repeat; ++j) {
        Copy(nzTensorTmp[j * 8 * (insn.s1Inner + 1 ) * C0_SIZE], ndTensorTmp[j * 128], VEC_REPEAT * dataCopyBlockNum,
            insn.s1Inner, nd2nzParams);
    }

    if (c1_remain > 0) {
        Copy(nzTensorTmp[c1_repeat * 8 * (insn.s1Inner + 1) * C0_SIZE], ndTensorTmp[c1_repeat * 128],
             dataCopyBlockNum * c1_remain, insn.s1Inner, nd2nzParams);
    }
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG, LAYOUT,
    MM2_OUT_FORMAT, POST, L1CUSTOM>::NZCopyIn(
    int64_t mmAddr, GlobalTensor<T2> &mmWspGm,
    LocalTensor<T2> &mmTensorCurr, PingPongEmitInsn &insn, int64_t processM)
{
    /*
    Func:
    MM输出NZ数据，数据搬运进UB，当前所取的vec基本块数据在wsp中为非连续，需要间隔搬运
    */
    if (rp.SFTLoopS1 == 0 && insn.vLoopS1 == 0 && insn.vLoopS2 == 0) {
        mm1.WaitIterateAll();
    }
    DataCopyParams intriParams;
    intriParams.blockCount = insn.s2InnerAlign / C0_SIZE;
    intriParams.blockLen = insn.s1Inner * C0_SIZE / B32_BLOCK_NUM;
    if constexpr (L1CUSTOM) {
        intriParams.srcStride = processM * C0_SIZE / B32_BLOCK_NUM - intriParams.blockLen;
    } else {
        intriParams.srcStride = rp.processM * C0_SIZE / B32_BLOCK_NUM - intriParams.blockLen;
    }
    intriParams.dstStride = 1;
    DataCopy(mmTensorCurr, mmWspGm[mmAddr], intriParams);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG,
                                                      DROPOUT_CFG, LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::MallocNodes()
{
    /*
    Nodes of vecGraph is 2.25(1=32KB) in ping(pong) without helpNode.
    Tensors would be malloc from Nodes which named xTensorNode_0, xTensorNode_1, xTensorNode_2.
    Eg:
      ATensorNode_0 and BTensorNode_0 shared same memory.
      ATensorNode_0 and BTensorNode_1 had different memory.
      Ping and Pong used different memory.
      32KB seems like 1 Node.
      8KB seems like 0.25 Node.
    */
    // Common Node [0~32K] only used vector
    b8Node0 = vecQue.GetWithOffset<uint8_t>(33 * 1024, 0);
    b16Node0 = vecQue.GetWithOffset<T1>(33 * 1024 / sizeof(T1), 0);
    b32Node0 = vecQue.GetWithOffset<T2>(33 * 1024 / sizeof(T2), 0);

    // Ping Nodes [32K~104K]
    b8Node1 = vecQue.GetWithOffset<uint8_t>(9 * 1024, 33 * 1024);                  // 0.25 + 1(1来源于NZ)
    b8Node2 = vecQue.GetWithOffset<uint8_t>(9 * 1024, 42 * 1024);                  // 0.25
    b16Node2 = vecQue.GetWithOffset<T1>(17 * 1024 / sizeof(T1), 42 * 1024);        // 0.5 + 1(1来源于NZ)
    b32Node3 = vecQue.GetWithOffset<T2>(4 * 1024 / sizeof(T2), 59 * 1024);         // 0.125
    b32Node4 = vecQue.GetWithOffset<T2>(4 * 1024 / sizeof(T2), 63 * 1024);         // 0.125
    b8Node5 = vecQue.GetWithOffset<uint8_t>(8 * 1024, 67 * 1024);                  // 0.25
    b32Node6 = vecQue.GetWithOffset<T2>(33 * 1024 / sizeof(T2), 75 * 1024);        // 1
    b32PingFuseNode = vecQue.GetWithOffset<T2>(33 * 1024 / sizeof(T2), 33 * 1024); // reused 1234
    b8PingFuseNode = vecQue.GetWithOffset<uint8_t>(33 * 1024, 33 * 1024);          // reused 1234

    // Pong Nodes [104~176K]
    b8Node7 = vecQue.GetWithOffset<uint8_t>(9 * 1024, 108 * 1024); // 0.25 + 1(1来源于NZ)
    b8Node8 = vecQue.GetWithOffset<uint8_t>(9 * 1024, 117 * 1024);
    b16Node8 = vecQue.GetWithOffset<T1>(17 * 1024 / sizeof(T1), 117 * 1024); // 0.5 + 1(1来源于NZ)
    b32Node9 = vecQue.GetWithOffset<T2>(4 * 1024 / sizeof(T2), 134 * 1024);
    b32Node10 = vecQue.GetWithOffset<T2>(4 * 1024 / sizeof(T2), 138 * 1024);
    b8Node11 = vecQue.GetWithOffset<uint8_t>(8 * 1024, 142 * 1024);
    b32Node12 = vecQue.GetWithOffset<T2>(33 * 1024 / sizeof(T2), 150 * 1024);
    b32PongFuseNode = vecQue.GetWithOffset<T2>(33 * 1024 / sizeof(T2), 108 * 1024); // reused 78910
    b8PongFuseNode = vecQue.GetWithOffset<uint8_t>(33 * 1024, 108 * 1024);          // reused 78910

    // sftg
    dyT1Tensor = vecQue.GetWithOffset<T1>(16 * 1024 / sizeof(T1), 0);                // 0~16K
    attentionT1Tensor = vecQue.GetWithOffset<T1>(16 * 1024 / sizeof(T1), 16 * 1024); // 16K~32K
    dyT2Tensor = vecQue.GetWithOffset<T2>(32 * 1024 / sizeof(T2), 32 * 1024);        // 32K~64K
    attentionT2Tensor = vecQue.GetWithOffset<T2>(32 * 1024 / sizeof(T2), 64 * 1024); // 64K~96K
    helpTensor = vecQue.GetWithOffset<uint8_t>(64 * 1024, 96 * 1024);           // 96K~160K
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG,
                    LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::DropOutCopy(LocalTensor<uint8_t> &dropmaskTensor, PingPongEmitInsn &insn)
{
    if constexpr (DROPOUT_CFG != 0) {
        int64_t bSSOffset = bIndex * dimS1 * dimS2;
        int64_t realS2 = dimS2;
        if constexpr (LAYOUT == TND) {
            bSSOffset = seqS1S2ProductSum;
            realS2 = seqS2Current;
            dropMaskInfo.s1Size = seqS1Current;
            dropMaskInfo.s2Size = seqS2Current;
        }
        dropMaskInfo.gOutIdx = gIndex;
        dropMaskInfo.bSSOffset = bSSOffset;
        dropMaskInfo.n2OutIdx = n2Index;
        dropMaskInfo.s1OutIdx = insn.s1Index;
        dropMaskInfo.s2Idx = insn.s2Index;

        // for copy in dropout mask
        dropMaskInfo.s1CopySize = insn.s1Inner;
        dropMaskInfo.s2CopySize = insn.s2Inner;
        dropMaskInfo.s2TotalSize = realS2;

        CopyInDropMask<true>(dropmaskTensor, dropoutWorkspaceGm, dropMaskGm, this->dropMaskInfo);
        if constexpr (IsSameType<T1, float>::value) {
            pipe_barrier(PIPE_ALL);
        }
    }
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG,
                                                      DROPOUT_CFG, LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::InnerT2Process(int64_t processM)
{
    /*
    Func:
    0. Exist Ping and Pong (PP) together.
    1. Structure of PP is MTE2 -> VEC -> MTE3
    2. Consider Set/Wait, Wait/Set and ReusedUB.
    */
    // SubGraph-Left
    // Task of Node1234 is MTE2 VEC
    auto attenmaskTensor = b8Node1;
    auto attenmaskPreTensor = b8Node2;
    auto pseTensor = b16Node2;
    auto sumTensor = b32Node3;
    auto maxTensor = b32Node4;
    // Task of Node5 is MTE2 VEC
    auto dropmaskTensor = b8Node5;
    // Task of Node6 is MTE2 VEC
    auto mm2Tensor = b32Node6;
    // Task of fusedNode is MTE2 VEC
    auto b8FusedTensor = b8PingFuseNode;

    auto attenmaskPreAddr = attenmaskPingPreAddress;
    auto attenmaskAddr = attenmaskPingAddress;
    auto dropmaskAddr = dropmaskPingAddress;
    auto sumMaxAddr = softmaxMaxSumPingAddress;
    auto mm2Addr = mm2PingAddr;
    auto mm4Addr = mm4PingAddr;
    auto emitInsn = pingEI;
    auto lowerMask = pingLowerMask;
    auto highMask = pingHighMask;
    auto isLast = isPingLast;
    auto isFirst = isPingFirst;
    AttenMaskCompress AttenBandMode = AttenBandPingMode;
    bool prefixCompressCanSimplify = rp.vPingPrefixCompressCanSimplify;
    
    event_t mte2WaitVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t mte2WaitVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t vWaitMte2Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    event_t vWaitMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    event_t mte2WaitMte3Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    event_t mte2WaitMte3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    event_t vWaitMte3Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    event_t vWaitMte3Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    event_t mte3WaitVPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    event_t mte3WaitVPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());

    auto eventIdMte2WaitV = mte2WaitVPing;
    auto eventIdVWaitMte2 = vWaitMte2Ping;
    auto eventIdMte2WaitMte3 = mte2WaitMte3Ping;
    auto eventIdVWaitMte3 = vWaitMte3Ping;
    auto eventIdMte3WaitV = mte3WaitVPing;

    if constexpr (PSE_CFG != 0) {
        pseInfo.boIdx = bIndex;
        pseInfo.n2oIdx = n2Index;
        pseInfo.goIdx = gIndex;
        pseInfo.bSSOffset = bIndex * dimS1 * dimS2;
        pseInfo.s2SizeAcc = bIndex * dimS2;
    }

    for (int i = 0; i <= 1; i++) {
        if ((i == 0 && !pingOK) || (i == 1 && !pongOK)) {
            continue;
        }
        if (i == 1) {
            attenmaskTensor = b8Node7;
            attenmaskPreTensor = b8Node8;
            pseTensor = b16Node8;
            sumTensor = b32Node9;
            maxTensor = b32Node10;
            dropmaskTensor = b8Node11;
            mm2Tensor = b32Node12;
            b8FusedTensor = b8PongFuseNode;

            attenmaskPreAddr = attenmaskPongPreAddress;
            attenmaskAddr = attenmaskPongAddress;
            dropmaskAddr = dropmaskPongAddress;
            sumMaxAddr = softmaxMaxSumPongAddress;
            mm2Addr = mm2PongAddr;
            mm4Addr = mm4PongAddr;
            emitInsn = pongEI;

            eventIdMte2WaitV = mte2WaitVPong;
            eventIdVWaitMte2 = vWaitMte2Pong;
            eventIdMte2WaitMte3 = mte2WaitMte3Pong;
            eventIdVWaitMte3 = vWaitMte3Pong;
            eventIdMte3WaitV = mte3WaitVPong;

            lowerMask = pongLowerMask;
            highMask = pongHighMask;
            isLast = isPongLast;
            isFirst = isPongFirst;
            AttenBandMode = AttenBandPongMode;
            prefixCompressCanSimplify = rp.vPongPrefixCompressCanSimplify;
        }

        // [MTE2][PSE][ATMASK][SUM/MAX][DROPOUT][MM2]
        if (!isFirst) {
            wait_flag(PIPE_V, PIPE_MTE2, eventIdMte2WaitV); // x3
        }

        LocalTensor<half> pseTensorHalf = pseTensor.template ReinterpretCast<half>();

        if (IsSameType<T1, float>::value && !isFirst && PSE_CFG == 0) {
            wait_flag(PIPE_MTE3, PIPE_MTE2, eventIdMte2WaitMte3); // x3
        }

        if constexpr (PSE_CFG != 0) {
            if (!isFirst) {
                wait_flag(PIPE_MTE3, PIPE_MTE2, eventIdMte2WaitMte3);
            }
            if constexpr (LAYOUT == TND) {
                pseInfo.bSSOffset = seqS1S2ProductSum;
                pseInfo.s2SizeAcc = seqS2CurrentOffset;
                pseInfo.s1Size = seqS1Current;
                pseInfo.s2Size = seqS2Current;
            }
            pseInfo.vec1S1RealSize = emitInsn.s1Inner;
            pseInfo.s1oIdx = emitInsn.s1Index;
            pseInfo.s2StartIdx = emitInsn.s2Index;
            pseInfo.s1BaseSize = 1;
            pseInfo.s2RealSize = emitInsn.s2Inner;
            pseInfo.s2AlignedSize = emitInsn.s2InnerAlign;
            pseInfo.vec1S1BaseSize = emitInsn.s1Inner;
            pseInfo.align8 = IsSameType<T1, float>::value;

            auto noCastedPseTensor = vecQue.GetWithOffset<T2>(0 / sizeof(T1), 0);
            if (pseInfo.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                pseInfo.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
                PseSlopeCopyIn<T2, true>(noCastedPseTensor, pseTensorHalf, pseSlope, this->pseAlibiGm, this->pseInfo);
            } else {
                if (!IsSameType<T1, float>::value) {
                    pseTensor.SetSize(pseShapeType == 1 ? emitInsn.s2InnerAlign : emitInsn.s1Inner * emitInsn.s2InnerAlign);
                    if constexpr (LAYOUT == TND) {
                        PseCopyIn<T1, T2, LayOutTypeEnum::LAYOUT_TND, true>(noCastedPseTensor, pseTensor, this->pseShiftGm,
                                                                            this->pseInfo);
                    } else {
                        PseCopyIn<T1, T2, LayOutTypeEnum::LAYOUT_BNSD, true>(noCastedPseTensor, pseTensor, this->pseShiftGm,
                                                                            this->pseInfo);
                    }
                }
            }
        }

        if constexpr (ATTEN_MASK_CFG != 0) {
            if (AttenBandMode == AttenMaskCompress::NextOnly || AttenBandMode == AttenMaskCompress::All) {
                MTE2_ATMask(attenmaskTensor, attenmaskAddr, emitInsn);
            } else if (AttenBandMode == AttenMaskCompress::PreOnly) {
                MTE2_ATMask(attenmaskTensor, attenmaskPreAddr, emitInsn);
            }
        }

        DropOutCopy(dropmaskTensor, emitInsn);

        MTE2_SFT(sumTensor, maxTensor, sumMaxAddr, emitInsn);

        if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
            if constexpr (L1CUSTOM) {
                NZCopyIn(mmPingpongIdx * mm2WorkspaceLen / sizeof(T2) + mm2Addr, mm2WorkspaceGm, mm2Tensor, emitInsn, processM);
            } else {
                NZCopyIn(mm2Addr, mm2WorkspaceGm, mm2Tensor, emitInsn, processM);
            }
        } else {
            MTE2ForMM2(mm2Tensor, mm2Addr, emitInsn);
        }

        set_flag(PIPE_MTE2, PIPE_V, eventIdVWaitMte2);
        wait_flag(PIPE_MTE2, PIPE_V, eventIdVWaitMte2);

        // [VEC][NZ2ND][PSE][ATMASK][SFTMAX][DROPOUT]
        if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
            NZ2ND(mm2Tensor, b32Node0, emitInsn);
        }

        if constexpr (PSE_CFG != 0) {
            if (pseInfo.pseType != (uint32_t)PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                DoMulsScale(mm2Tensor, emitInsn);
            }
            if (!(pseInfo.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
                pseInfo.pseType == (uint32_t)PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE)) {
                uint32_t count = pseShapeType == 1 ? emitInsn.s2InnerAlign : emitInsn.s1Inner * emitInsn.s2InnerAlign;
                if constexpr (!IsSameType<T1, float>::value) {
                    CastTo32(b32Node0, pseTensor, count);
                } else {
                    set_flag(PIPE_V, PIPE_MTE2, eventIdMte2WaitV);
                    wait_flag(PIPE_V, PIPE_MTE2, eventIdMte2WaitV);
                    if constexpr (LAYOUT == TND) {
                        PseCopyIn<T1, T2, LayOutTypeEnum::LAYOUT_TND, true>(b32Node0, b32Node0, this->pseShiftGm,
                                                                            this->pseInfo);
                    } else {
                        PseCopyIn<T1, T2, LayOutTypeEnum::LAYOUT_BNSD, true>(b32Node0, b32Node0, this->pseShiftGm,
                                                                            this->pseInfo);
                    }
                    set_flag(PIPE_MTE2, PIPE_V, eventIdVWaitMte2);
                    wait_flag(PIPE_MTE2, PIPE_V, eventIdVWaitMte2);
                }
            } else {
                PseSlopeCast<T2, true>(b32Node0, pseTensorHalf, pseSlope, pseInfo);
            }
            mm2Tensor.SetSize(emitInsn.s1Inner * emitInsn.s2InnerAlign);
            b32Node0.SetSize(emitInsn.s1Inner * emitInsn.s2InnerAlign);
            pipe_barrier(PIPE_V);
            PseCompute<T2, true>(mm2Tensor, b32Node0, this->pseInfo);
            pipe_barrier(PIPE_V);
        }

        if (tilingData->opInfo.pseType == (uint32_t)PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            DoMulsScale(mm2Tensor, emitInsn);
        }

        if (PSE_CFG == 0 && !isFirst) {
            wait_flag(PIPE_MTE3, PIPE_V, eventIdVWaitMte3); // x4
        }

        if constexpr (ATTEN_MASK_CFG != 0) {
            if (compressMode == PREFIX_COMPRESS_MODE) { // prefix compress
                if (prefixCompressCanSimplify == false) {
                    uint32_t s2InnerAlign =
                        (emitInsn.s2Inner + 31) / 32 * 32; // attenmask做pad时会32对齐，故加31/32做ceil
                    int32_t maskNum = emitInsn.s1Inner * s2InnerAlign / 2; // 除2数据量按照uint16类型折半

                    set_flag(PIPE_V, PIPE_MTE2, eventIdMte2WaitV);
                    wait_flag(PIPE_V, PIPE_MTE2, eventIdMte2WaitV);
                    MTE2_ATMask(attenmaskPreTensor, attenmaskPreAddr, emitInsn);

                    set_flag(PIPE_MTE2, PIPE_V, eventIdVWaitMte2);
                    wait_flag(PIPE_MTE2, PIPE_V, eventIdVWaitMte2);
                    auto attenmaskTensorTmp = attenmaskTensor.template ReinterpretCast<uint16_t>();
                    auto attenmaskPreTensorTmp = attenmaskPreTensor.template ReinterpretCast<uint16_t>();
                    And(attenmaskTensorTmp, attenmaskPreTensorTmp, attenmaskTensorTmp, maskNum);
                    pipe_barrier(PIPE_V);
                    attenmaskTensor = attenmaskTensorTmp.template ReinterpretCast<uint8_t>();
                }
            }

            if (AttenBandMode == AttenMaskCompress::NextOnly || AttenBandMode == AttenMaskCompress::All) {
                DoMaskU8(mm2Tensor, attenmaskTensor, b8Node0, emitInsn);
            } else if (AttenBandMode == AttenMaskCompress::PreOnly) {
                DoMaskU8(mm2Tensor, attenmaskTensor, b8Node0, emitInsn, 1);
            }

            if ((compressMode == BAND_MODE || unpadUseBand) && AttenBandMode == AttenMaskCompress::All) {
                set_flag(PIPE_V, PIPE_MTE2, eventIdMte2WaitV);
                wait_flag(PIPE_V, PIPE_MTE2, eventIdMte2WaitV);
                MTE2_ATMask(attenmaskTensor, attenmaskPreAddr, emitInsn);
                event_t vWaitMte2Stage1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                set_flag(PIPE_MTE2, PIPE_V, eventIdVWaitMte2);
                wait_flag(PIPE_MTE2, PIPE_V, eventIdVWaitMte2);
                DoMaskU8(mm2Tensor, attenmaskTensor, b8Node0, emitInsn, 1);
            }
        }

        DoSimpleSoftMax(mm2Tensor, sumTensor, maxTensor, b8Node0, emitInsn);

        if constexpr (!IsSameType<T1, float>::value) {
            if constexpr (DROPOUT_CFG != 0) {
                // for compute dropout mask
                dropMaskInfo.firstAxis = emitInsn.s1Inner;
                dropMaskInfo.lstAxis = emitInsn.s2InnerAlign;
                dropMaskInfo.maskLstAxis = emitInsn.s2Inner;
                ComputeDropMask<T2, true>(b32Node0, mm2Tensor, dropmaskTensor, b8FusedTensor, this->dropMaskInfo);
                pipe_barrier(PIPE_V);

                CastTo16(pseTensor, b32Node0, emitInsn.s1Inner * emitInsn.s2InnerAlign);
            } else {
                CastTo16(pseTensor, mm2Tensor, emitInsn.s1Inner * emitInsn.s2InnerAlign);
            }

            if (lowerMask != 0) {
                uint64_t mask[2] = {static_cast<uint64_t>(lowerMask), static_cast<uint64_t>(highMask)};
                Duplicate<T1>(pseTensor[emitInsn.s2InnerAlign - dataCopyBlockNum], 0, mask, emitInsn.s1Inner, 1,
                            emitInsn.s2InnerAlign / dataCopyBlockNum);
            }

            if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
                pipe_barrier(PIPE_V);
                DataCopy(b16Node0, pseTensor, emitInsn.s1Inner * emitInsn.s2InnerAlign);
                pipe_barrier(PIPE_V);
                ND2NZ(pseTensor, b16Node0, emitInsn);
            }
 
            // [MTE3]
            set_flag(PIPE_V, PIPE_MTE3, eventIdMte3WaitV);
            wait_flag(PIPE_V, PIPE_MTE3, eventIdMte3WaitV);
            CopyoutWorkspace(mm4InputWorkspaceGm[mm4Addr], pseTensor, emitInsn);
            set_flag(PIPE_MTE3, PIPE_MTE2, eventIdMte2WaitMte3); // x1
        } else {
             if constexpr (DROPOUT_CFG != 0) {
                // for compute dropout mask
                dropMaskInfo.firstAxis = emitInsn.s1Inner;
                dropMaskInfo.lstAxis = emitInsn.s2InnerAlign;
                dropMaskInfo.maskLstAxis = emitInsn.s2Inner;
                ComputeDropMask<T2, true>(b32Node0, mm2Tensor, dropmaskTensor, b8FusedTensor, this->dropMaskInfo);
                pipe_barrier(PIPE_ALL);
            }

            if (lowerMask != 0) {
                pipe_barrier(PIPE_V);
                uint64_t mask[2] = {static_cast<uint64_t>(lowerMask), static_cast<uint64_t>(highMask)};
                if constexpr (DROPOUT_CFG != 0) {
                    if (isPseInnerGenerate) {
                        Duplicate<T1>(b32Node0[emitInsn.s2InnerAlign - 16], 0, mask, emitInsn.s1Inner, 1,
                                emitInsn.s2InnerAlign / dataCopyBlockNum);
                    } else {
                        Duplicate<T1>(b32Node0[emitInsn.s2InnerAlign - dataCopyBlockNum], 0, mask, emitInsn.s1Inner, 1,
                            emitInsn.s2InnerAlign / dataCopyBlockNum);
                    }
                } else {
                    if (unlikely(isPseInnerGenerate)) {
                        Duplicate<T1>(mm2Tensor[emitInsn.s2InnerAlign - 16], 0, mask, emitInsn.s1Inner, 1,
                            emitInsn.s2InnerAlign / dataCopyBlockNum);
                    } else {
                        Duplicate<T1>(mm2Tensor[emitInsn.s2InnerAlign - dataCopyBlockNum], 0, mask, emitInsn.s1Inner, 1,
                            emitInsn.s2InnerAlign / dataCopyBlockNum);
                    }
                }
            }

            // [MTE3]
            set_flag(PIPE_V, PIPE_MTE3, eventIdMte3WaitV);
            wait_flag(PIPE_V, PIPE_MTE3, eventIdMte3WaitV);
            if constexpr (DROPOUT_CFG != 0) {
                CopyoutWorkspace(mm4InputWorkspaceGm[mm4Addr], b32Node0, emitInsn);
            } else {
                CopyoutWorkspace(mm4InputWorkspaceGm[mm4Addr], mm2Tensor, emitInsn);
            }

            set_flag(PIPE_MTE3, PIPE_MTE2, eventIdMte2WaitMte3); // x1
        }
    }

    // SubGraph-Right
    auto mm1Tensor = b32PingFuseNode;
    pseTensor = b16Node2;
    mm2Tensor = b32Node6;
    dropmaskTensor = b8Node5;

    auto mm1Addr = mm1PingAddr;
    auto mm3Addr = mm3PingAddr;
    dropmaskAddr = dropmaskPingAddress;
    emitInsn = pingEI;
    lowerMask = pingLowerMask;
    highMask = pingHighMask;
    isLast = isPingLast;
    isFirst = isPingFirst;
    
    eventIdMte2WaitV = mte2WaitVPing;
    eventIdVWaitMte2 = vWaitMte2Ping;
    eventIdMte2WaitMte3 = mte2WaitMte3Ping;
    eventIdVWaitMte3 = vWaitMte3Ping;
    eventIdMte3WaitV = mte3WaitVPing;

    for (int i = 0; i <= 1; i++) {
        if ((i == 0 && !pingOK) || (i == 1 && !pongOK)) {
            continue;
        }
        if (i == 1) {
            mm1Tensor = b32PongFuseNode;
            pseTensor = b16Node8;
            mm2Tensor = b32Node12;
            dropmaskTensor = b8Node11;

            mm1Addr = mm1PongAddr;
            mm3Addr = mm3PongAddr;
            dropmaskAddr = dropmaskPongAddress;
            emitInsn = pongEI;

            eventIdMte2WaitV = mte2WaitVPong;
            eventIdVWaitMte2 = vWaitMte2Pong;
            eventIdMte2WaitMte3 = mte2WaitMte3Pong;
            eventIdVWaitMte3 = vWaitMte3Pong;
            eventIdMte3WaitV = mte3WaitVPong;

            lowerMask = pongLowerMask;
            highMask = pongHighMask;
            isLast = isPongLast;
            isFirst = isPongFirst;
        }

        // [MTE2][MM2]
        wait_flag(PIPE_MTE3, PIPE_MTE2, eventIdMte2WaitMte3); // x1
        if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
            if constexpr (L1CUSTOM) {
                NZCopyIn(mmPingpongIdx * mm1WorkspaceLen / sizeof(T2) + mm1Addr, mm1WorkspaceGm, mm1Tensor, emitInsn, processM);
            } else {
                NZCopyIn(mm1Addr, mm1WorkspaceGm, mm1Tensor, emitInsn, processM);
            }
        } else {
            MTE2ForMM1(mm1Tensor, mm1Addr, emitInsn);
        }

        DropOutCopy(dropmaskTensor, emitInsn);

        // [VEC][NZ2ND][DROPOUT]
        set_flag(PIPE_MTE2, PIPE_V, eventIdVWaitMte2);
        wait_flag(PIPE_MTE2, PIPE_V, eventIdVWaitMte2);
        if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
            NZ2ND(mm1Tensor, b32Node0, emitInsn);
        }

        if constexpr (DROPOUT_CFG != 0) {
            // for compute dropout mask
            dropMaskInfo.firstAxis = emitInsn.s1Inner;
            dropMaskInfo.lstAxis = emitInsn.s2InnerAlign;
            dropMaskInfo.maskLstAxis = emitInsn.s2Inner;
            ComputeDropMask<T2, true>(mm1Tensor, mm1Tensor, dropmaskTensor, b8Node0, this->dropMaskInfo);
            pipe_barrier(PIPE_V);
        }
        auto partGradTensor = softmaxGradOutTensor[emitInsn.vLoopS1 * vec.baseM * 8]; // [rp.SFTS1Inner, 8]
        DoSub(mm1Tensor, partGradTensor, emitInsn);

        DoMul(mm2Tensor, mm1Tensor, emitInsn);

        if constexpr (!IsSameType<T1, float>::value) {
            CastTo16(pseTensor, mm2Tensor, emitInsn.s1Inner * emitInsn.s2InnerAlign);
        }

        if (!isLast) {
            set_flag(PIPE_V, PIPE_MTE2, eventIdMte2WaitV); // x3
        }

        if (lowerMask != 0) {
            uint64_t mask[2] = {static_cast<uint64_t>(lowerMask), static_cast<uint64_t>(highMask)};
            if constexpr (!IsSameType<T1, float>::value) {
                Duplicate<T1>(pseTensor[emitInsn.s2InnerAlign - dataCopyBlockNum], 0, mask, emitInsn.s1Inner, 1,
                          emitInsn.s2InnerAlign / dataCopyBlockNum);
            } else {
                if (isPseInnerGenerate) {
                    Duplicate<T1>(mm2Tensor[emitInsn.s2InnerAlign - 16], 0, mask, emitInsn.s1Inner, 1,
                          emitInsn.s2InnerAlign / dataCopyBlockNum);
                } else {
                    Duplicate<T1>(mm2Tensor[emitInsn.s2InnerAlign - dataCopyBlockNum], 0, mask, emitInsn.s1Inner, 1,
                          emitInsn.s2InnerAlign / dataCopyBlockNum);
                }
            }
        }

        if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
            pipe_barrier(PIPE_V);
            DataCopy(b16Node0, pseTensor, emitInsn.s1Inner * emitInsn.s2InnerAlign);
            pipe_barrier(PIPE_V);
            ND2NZ(pseTensor, b16Node0, emitInsn);
        }

        // [MTE3]
        set_flag(PIPE_V, PIPE_MTE3, eventIdMte3WaitV);
        wait_flag(PIPE_V, PIPE_MTE3, eventIdMte3WaitV);
        if constexpr (!IsSameType<T1, float>::value) {
            CopyoutWorkspace(mm3InputWorkspaceGm[mm3Addr], pseTensor, emitInsn);
        } else {
            CopyoutWorkspace(mm3InputWorkspaceGm[mm3Addr], mm2Tensor, emitInsn);
        }

        if constexpr (!IsSameType<T1, float>::value) {
            if (PSE_CFG != 0 && !isLast) {
                set_flag(PIPE_MTE3, PIPE_MTE2, eventIdMte2WaitMte3); // x2
            }
        } else {
            if (!isLast) {
                set_flag(PIPE_MTE3, PIPE_MTE2, eventIdMte2WaitMte3); // x2
            }
        }

        if (PSE_CFG == 0 && !isLast) {
            set_flag(PIPE_MTE3, PIPE_V, eventIdVWaitMte3); // x4
        }
    }
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(mte2WaitVPing);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(mte2WaitVPong);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(vWaitMte2Ping);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(vWaitMte2Pong);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(mte2WaitMte3Ping);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(mte2WaitMte3Pong);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(vWaitMte3Ping);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(vWaitMte3Pong);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(mte3WaitVPing);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(mte3WaitVPong);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG,
        LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::PingClcParams(int64_t mm1Addr, int64_t mm2Addr, int64_t mm3Addr, int64_t mm4Addr, int64_t processM)
{
    if (!pingOK) {
        return;
    }

    rp.vPingLoopS1 = pingLoop / rp.vS2Times;
    rp.vPingLoopS2 = pingLoop % rp.vS2Times;
    rp.vPingS1Inner = vec.baseM;
    rp.vPingS2Inner = vec.baseN;
    rp.vPingS2InnerAlign = vec.baseN;
    if (rp.vPingLoopS1 == rp.vS1Times - 1) {
        rp.vPingS1Inner = rp.vS1InnerTail;
    }
    if (rp.vPingLoopS2 == rp.vS2Times - 1) {
        rp.vPingS2Inner = rp.vS2InnerTail;
        if (unlikely(isPseInnerGenerate)) {
            rp.vPingS2InnerAlign = (rp.vS2InnerTail + 15) / 16 * 16;
        } else {
            rp.vPingS2InnerAlign = (rp.vS2InnerTail + dataCopyBlockNum - 1) / dataCopyBlockNum * dataCopyBlockNum;
        }
    }

    pingLowerMask = 0;
    if (rp.vPingS2Inner != rp.vPingS2InnerAlign) {
        for (size_t i = 0; i < rp.vPingS2InnerAlign - rp.vPingS2Inner; i++) {
            if constexpr (!IsSameType<T1, float>::value) {
                pingLowerMask = pingLowerMask + (pingVar << (15 - i));
            } else {
                if (isPseInnerGenerate) {
                    pingLowerMask = pingLowerMask + (pingVar << (15 - i));
                } else {
                    pingLowerMask = pingLowerMask + (pingVar << (7 - i));
                }
            }
        }
    }

    isPingFirst = pingLoop == 0;
    isPingLast = isEven ? pingLoop == totalLoops - 2 : pingLoop == totalLoops - 1;

    pingEI.s1Inner = rp.vPingS1Inner;
    pingEI.s2Inner = rp.vPingS2Inner;
    pingEI.s2InnerAlign = rp.vPingS2InnerAlign;
    pingEI.vLoopS1 = rp.vPingLoopS1;
    pingEI.vLoopS2 = rp.vPingLoopS2;
    pingEI.s1Index = s1Index + pingEI.vLoopS1 * vec.baseM;
    pingEI.s2Index =
        static_cast<int64_t>(rp.loopS2) * vec.singleN + usedSingleNBegin * vec.baseN + pingEI.vLoopS2 * vec.baseN;

    if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
        if constexpr (L1CUSTOM) {
            mm1PingAddr = mm1Addr + static_cast<int64_t>(rp.vPingLoopS1) * vec.baseM * C0_SIZE +
                        static_cast<int64_t>(rp.vPingLoopS2) * processM * vec.baseN;
            mm2PingAddr = mm2Addr + static_cast<int64_t>(rp.vPingLoopS1) * vec.baseM * C0_SIZE +
                        static_cast<int64_t>(rp.vPingLoopS2) * processM * vec.baseN;
        } else {
            mm1PingAddr = mm1Addr + static_cast<int64_t>(rp.vPingLoopS1) * vec.baseM * C0_SIZE +
                        static_cast<int64_t>(rp.vPingLoopS2) * rp.processM * vec.baseN;
            mm2PingAddr = mm2Addr + static_cast<int64_t>(rp.vPingLoopS1) * vec.baseM * C0_SIZE +
                        static_cast<int64_t>(rp.vPingLoopS2) * rp.processM * vec.baseN;
        }

    } else {
        mm1PingAddr = mm1Addr + static_cast<int64_t>(rp.vPingLoopS1) * vec.baseM * b32AlignProcessN +
                      static_cast<int64_t>(rp.vPingLoopS2) * vec.baseN;
        mm2PingAddr = mm2Addr + static_cast<int64_t>(rp.vPingLoopS1) * vec.baseM * b32AlignProcessN +
                      static_cast<int64_t>(rp.vPingLoopS2) * vec.baseN;
    }

    if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
        mm3PingAddr = mm3Addr + static_cast<int64_t>(rp.vPingLoopS1) * vec.baseM * C0_SIZE +
                      static_cast<int64_t>(rp.vPingLoopS2) * alignProcessM * vec.baseN;
        mm4PingAddr = mm4Addr + static_cast<int64_t>(rp.vPingLoopS1) * vec.baseM * C0_SIZE +
                      static_cast<int64_t>(rp.vPingLoopS2) * alignProcessM * vec.baseN;
    } else {
        mm3PingAddr = mm3Addr + static_cast<int64_t>(rp.vPingLoopS1) * vec.baseM * b16AlignProcessN +
                    static_cast<int64_t>(rp.vPingLoopS2) * vec.baseN;
        mm4PingAddr = mm4Addr + static_cast<int64_t>(rp.vPingLoopS1) * vec.baseM * b16AlignProcessN +
                    static_cast<int64_t>(rp.vPingLoopS2) * vec.baseN;
    }
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG,
                               LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::PongClcParams(int64_t mm1Addr, int64_t mm2Addr, int64_t mm3Addr,
                                                      int64_t mm4Addr, int64_t processM)
{
    if (!pongOK) {
        return;
    }

    rp.vPongLoopS1 = pongLoop / rp.vS2Times;
    rp.vPongLoopS2 = pongLoop % rp.vS2Times;
    rp.vPongS1Inner = vec.baseM;
    rp.vPongS2Inner = vec.baseN;
    rp.vPongS2InnerAlign = vec.baseN;
    if (rp.vPongLoopS1 == rp.vS1Times - 1) {
        rp.vPongS1Inner = rp.vS1InnerTail;
    }
    if (rp.vPongLoopS2 == rp.vS2Times - 1) {
        rp.vPongS2Inner = rp.vS2InnerTail;
        if (unlikely(isPseInnerGenerate)) {
            rp.vPongS2InnerAlign = (rp.vS2InnerTail + 15) / 16 * 16;
        } else {
            rp.vPongS2InnerAlign = (rp.vS2InnerTail + dataCopyBlockNum - 1) / dataCopyBlockNum * dataCopyBlockNum;
        }
    }

    pongLowerMask = 0;
    if (rp.vPongS2Inner != rp.vPongS2InnerAlign) {
        for (size_t i = 0; i < rp.vPongS2InnerAlign - rp.vPongS2Inner; i++) {
            if constexpr (!IsSameType<T1, float>::value) {
                pongLowerMask = pongLowerMask + (pongVar << (15 - i));
            } else {
                if (isPseInnerGenerate) {
                    pongLowerMask = pongLowerMask + (pongVar << (15 - i));
                } else {
                    pongLowerMask = pongLowerMask + (pongVar << (7 - i));
                }
            }
        }
    }


    isPongFirst = pongLoop == 1;
    isPongLast = isEven ? pongLoop == totalLoops - 1 : pongLoop == totalLoops - 2;

    pongEI.s1Inner = rp.vPongS1Inner;
    pongEI.s2Inner = rp.vPongS2Inner;
    pongEI.s2InnerAlign = rp.vPongS2InnerAlign;
    pongEI.vLoopS1 = rp.vPongLoopS1;
    pongEI.vLoopS2 = rp.vPongLoopS2;
    pongEI.s1Index = s1Index + pongEI.vLoopS1 * vec.baseM;
    pongEI.s2Index = static_cast<int64_t>(rp.loopS2) * vec.singleN +
                     static_cast<int64_t>(usedSingleNBegin) * vec.baseN +
                     static_cast<int64_t>(pongEI.vLoopS2) * vec.baseN;

    if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
        if constexpr (L1CUSTOM) {
            mm1PongAddr = mm1Addr + static_cast<int64_t>(rp.vPongLoopS1) * vec.baseM * C0_SIZE +
                        static_cast<int64_t>(rp.vPongLoopS2) * processM * vec.baseN;
            mm2PongAddr = mm2Addr + static_cast<int64_t>(rp.vPongLoopS1) * vec.baseM * C0_SIZE +
                        static_cast<int64_t>(rp.vPongLoopS2) * processM * vec.baseN;
        } else {
            mm1PongAddr = mm1Addr + static_cast<int64_t>(rp.vPongLoopS1) * vec.baseM * C0_SIZE +
                        static_cast<int64_t>(rp.vPongLoopS2) * rp.processM * vec.baseN;
            mm2PongAddr = mm2Addr + static_cast<int64_t>(rp.vPongLoopS1) * vec.baseM * C0_SIZE +
                        static_cast<int64_t>(rp.vPongLoopS2) * rp.processM * vec.baseN;
        }
    } else {
        mm1PongAddr = mm1Addr + static_cast<int64_t>(rp.vPongLoopS1) * vec.baseM * b32AlignProcessN +
                      static_cast<int64_t>(rp.vPongLoopS2) * vec.baseN;
        mm2PongAddr = mm2Addr + static_cast<int64_t>(rp.vPongLoopS1) * vec.baseM * b32AlignProcessN +
                      static_cast<int64_t>(rp.vPongLoopS2) * vec.baseN;
    }

    if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
        mm3PongAddr = mm3Addr + static_cast<int64_t>(rp.vPongLoopS1) * vec.baseM * C0_SIZE +
                      static_cast<int64_t>(rp.vPongLoopS2) * alignProcessM * vec.baseN;
        mm4PongAddr = mm4Addr + static_cast<int64_t>(rp.vPongLoopS1) * vec.baseM * C0_SIZE +
                      static_cast<int64_t>(rp.vPongLoopS2) * alignProcessM * vec.baseN;
    } else {
        mm3PongAddr = mm3Addr + static_cast<int64_t>(rp.vPongLoopS1) * vec.baseM * b16AlignProcessN +
                    static_cast<int64_t>(rp.vPongLoopS2) * vec.baseN;
        mm4PongAddr = mm4Addr + static_cast<int64_t>(rp.vPongLoopS1) * vec.baseM * b16AlignProcessN +
                    static_cast<int64_t>(rp.vPongLoopS2) * vec.baseN;
    }
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG,
                LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::VectorByCS1(int64_t mm1Addr, int64_t mm2Addr, int64_t mm3Addr,
                int64_t mm4Addr, int64_t realProcessNtmp, int64_t processM)
{
    // Vector follow SFTGrad and SFTMaxSum.
    int32_t halfM = processM / 2;
    rp.vectorProcessM = subBlockIdx ? processM - halfM : halfM;
    rp.SFTS1Inner = sft.singleM;
    if constexpr (L1CUSTOM) {
        rp.SFTS1Times = (rp.vectorProcessM + sft.singleM - 1) / sft.singleM;
        rp.SFTS1InnerTail = rp.vectorProcessM % sft.singleM == 0 ? sft.singleM : rp.vectorProcessM % sft.singleM;
        rp.vS2Times = (realProcessNtmp + vec.baseN - 1) / vec.baseN;
        rp.vS2InnerTail = realProcessNtmp % vec.baseN == 0 ? vec.baseN : realProcessNtmp % vec.baseN;
    } else {
        rp.SFTS1Times = (rp.processM + sft.singleM - 1) / sft.singleM;
        rp.SFTS1InnerTail = rp.processM % sft.singleM == 0 ? sft.singleM : rp.processM % sft.singleM;
        rp.vS2Times = (realProcessN + vec.baseN - 1) / vec.baseN;
        rp.vS2InnerTail = realProcessN % vec.baseN == 0 ? vec.baseN : realProcessN % vec.baseN;
    }

    if constexpr (L1CUSTOM) {
        if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
            mm1Addr += subBlockIdx * halfM * C0_SIZE;
            mm2Addr += subBlockIdx * halfM * C0_SIZE;
            mm3Addr += subBlockIdx * halfM * C0_SIZE;
            mm4Addr += subBlockIdx * halfM * C0_SIZE;
        } else {
            mm1Addr += subBlockIdx * halfM * b32AlignProcessN;
            mm2Addr += subBlockIdx * halfM * b32AlignProcessN;
            mm3Addr += subBlockIdx * halfM * b16AlignProcessN;
            mm4Addr += subBlockIdx * halfM * b16AlignProcessN;
        }
    }

    for (uint32_t SFTLoopS1 = 0; SFTLoopS1 < rp.SFTS1Times; SFTLoopS1++) {
        // 更新S1方向地址, 更新SFT baseM
        if constexpr (L1CUSTOM) {
            s1Index = rp.loopS1 * vec.singleM + SFTLoopS1 * sft.singleM + subBlockIdx * halfM;
        } else {
            s1Index = rp.loopS1 * vec.singleM + SFTLoopS1 * sft.singleM;
        }
        if (SFTLoopS1 == rp.SFTS1Times - 1) {
            rp.SFTS1Inner = rp.SFTS1InnerTail;
        }

        // STFGrad结果外提
        DoSoftmaxGrad(softmaxGradOutTensor); // [rp.SFTS1Inner, 8] FP32

        pipe_barrier(PIPE_ALL);

        rp.SFTLoopS1 = SFTLoopS1;
        VectorByS1S2(mm1Addr, mm2Addr, mm3Addr, mm4Addr, processM);
        if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
            mm1Addr += sft.singleM * C0_SIZE;
            mm2Addr += sft.singleM * C0_SIZE;
        } else {
            mm1Addr += sft.singleM * b32AlignProcessN;
            mm2Addr += sft.singleM * b32AlignProcessN;
        }
        if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
            mm3Addr += sft.singleM * C0_SIZE;
            mm4Addr += sft.singleM * C0_SIZE;
        } else {
            mm3Addr += sft.singleM * b16AlignProcessN;
            mm4Addr += sft.singleM * b16AlignProcessN;
        }
        pipe_barrier(PIPE_ALL);
    }
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG,
            LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::VectorByS1S2(int64_t mm1Addr, int64_t mm2Addr, int64_t mm3Addr, int64_t mm4Addr, int64_t processM)
{
    // Vector follow with Vector-Fused-S1-S2
    rp.vS1Times = (rp.SFTS1Inner + vec.baseM - 1) / vec.baseM;
    rp.vS1InnerTail = rp.SFTS1Inner % vec.baseM == 0 ? vec.baseM : rp.SFTS1Inner % vec.baseM;
    totalLoops = rp.vS1Times * rp.vS2Times;
    isEven = totalLoops % 2 == 0;

    pingLoop = 0;
    pongLoop = 1;
    pingOK = pingLoop < totalLoops;
    pongOK = pongLoop < totalLoops;
    while ((pingOK && !isEven) || (pongOK && isEven)) {
        PingClcParams(mm1Addr, mm2Addr, mm3Addr, mm4Addr, processM);
        PongClcParams(mm1Addr, mm2Addr, mm3Addr, mm4Addr, processM);
        CopyInOffset(pingEI.s1Index, pingEI.s2Index, true);
        CopyInOffset(pongEI.s1Index, pongEI.s2Index, false);
        CopyInOffsetForSimpleSoftmax(pingEI.s1Index, true);
        CopyInOffsetForSimpleSoftmax(pongEI.s1Index, false);
        InnerT2Process(processM);
        pingLoop += 2;
        pongLoop += 2;
        pingOK = pingLoop < totalLoops;
        pongOK = pongLoop < totalLoops;
    }
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG,
                                                      DROPOUT_CFG, LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::AssureUsefulDataBySingleN()
{
    // return should calc data in SingleN
    uint32_t begin = 0;
    uint32_t num = 0;
    uint32_t mark = 0;
    int64_t index = s2Index;
    uint32_t inner = vec.baseN;
    uint32_t innerTail = rp.processN % vec.baseN == 0 ? vec.baseN : rp.processN % vec.baseN;
    for (int64_t s2_o_i = 0; s2_o_i < lp.s2OuterInnerNum; s2_o_i++) {
        if (s2_o_i == lp.s2OuterInnerNum - 1) {
            inner = innerTail;
        }
        bool exclude = (index + inner <= sparse_s2_start_idx) || (index >= sparse_s2_end_idx);
        if (!(isSparse == 1 && exclude)) {
            num++;
            if (mark == 0) {
                begin = s2_o_i;
                mark = 1;
            }
        }
        index += inner;
    }

    usedSingleNNum = num;
    usedSingleNBegin = begin;
    usedSingleNEnd = num == 0 ? begin : begin + num - 1;
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline bool FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG,
                                                      DROPOUT_CFG, LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::CalcUsefulDataByS2()
{
    // 计算S2配比中, 有效的vec.baseN数量(默认S1方向全部有效).
    alignProcessM = (rp.processM + vec.singleM - 1) / vec.singleM * vec.singleM;
    if (isSparse != 1) {
        realProcessN = rp.processN;
        b32AlignProcessN = (realProcessN + calcBlockNum - 1) / calcBlockNum * calcBlockNum;
        if (unlikely(isPseInnerGenerate)) {
            b16AlignProcessN = (realProcessN + 15) / 16 * 16;
        } else {
            b16AlignProcessN = (realProcessN + dataCopyBlockNum - 1) / dataCopyBlockNum * dataCopyBlockNum;
        }
        usedSingleNBegin = 0;
        return true;
    }

    AssureUsefulDataBySingleN();
    if (usedSingleNNum <= 0) {
        return false;
    }

    realProcessN = (usedSingleNNum * vec.baseN) > vec.singleN ? vec.singleN : (usedSingleNNum * vec.baseN);
    bool endIsLastBaseN = usedSingleNEnd == (rp.processN + vec.baseN - 1) / vec.baseN - 1;
    if (isLastSingleN && endIsLastBaseN) {
        if (rp.processN % vec.baseN != 0) {
            realProcessN = usedSingleNNum * vec.baseN - vec.baseN + (rp.processN % vec.baseN);
        }
    }
    b32AlignProcessN = (realProcessN + calcBlockNum - 1) / calcBlockNum * calcBlockNum;
    if (isPseInnerGenerate) {
        b16AlignProcessN = (realProcessN + 15) / 16 * 16;
    } else {
        b16AlignProcessN = (realProcessN + dataCopyBlockNum - 1) / dataCopyBlockNum * dataCopyBlockNum;
    }
    return true;
}

template <typename T1, typename T2, const MatmulConfig& MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
    const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
    const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG, LAYOUT,
    MM2_OUT_FORMAT, POST, L1CUSTOM>::SavePostBNIdx()
{
    if(!isLastWorkSpaceFinished){
        return;
    }

    if(postCalCnt % 2  == 0){
        savedIndices0.bIndex = lastIndices.bIndex;
        savedIndices0.n2Index = lastIndices.n2Index;
        savedIndices0.gIndex = lastIndices.gIndex;
    }else{
        savedIndices1.bIndex = lastIndices.bIndex;
        savedIndices1.n2Index = lastIndices.n2Index;
        savedIndices1.gIndex = lastIndices.gIndex;
    }
    postCalCnt++;
    isLastWorkSpaceFinished = false;
}

template <typename T1, typename T2, const MatmulConfig& MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
    const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
    const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG, LAYOUT,
    MM2_OUT_FORMAT, POST, L1CUSTOM>::PostProcess(bool isLast)
{
    //update BNG index
    if(unlikely(isLast)){
        calIndices.bIndex = bIndex;
        calIndices.n2Index = n2Index;
        calIndices.gIndex = gIndex;
    }else{
        if(postCalConsum % 2 == 0){
            calIndices.bIndex = savedIndices0.bIndex;
            calIndices.n2Index = savedIndices0.n2Index;
            calIndices.gIndex = savedIndices0.gIndex;
        }else {
            calIndices.bIndex = savedIndices1.bIndex;
            calIndices.n2Index = savedIndices1.n2Index;
            calIndices.gIndex = savedIndices1.gIndex;
        }
    }

    vWaitMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    mte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    mte3WaitV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());

    set_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3);
    wait_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3);

    mm3_1.WaitIterateAll();
    PostProcessQ();

    set_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3);
    wait_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3);
    mm4.WaitIterateAll();

    if(calIndices.gIndex == dimG -1){
        PostProcessKV();
        if(likely(!isLast)){
            set_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3);
            wait_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3);
        }
    }
    // update used data
    postCalConsum++;
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(vWaitMte2);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(mte2WaitMte3);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(mte3WaitV);
}

template <typename T1, typename T2, const MatmulConfig& MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
    const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
    const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG,
    DROPOUT_CFG, LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::PostProcessQ()
{
    if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
        PostProcessNZ(dqWorkspaceGm, dqGm, 0);
    }else{
        PostProcessND(dqWorkspaceGm, dqGm, 0);
    }
}

template <typename T1, typename T2, const MatmulConfig& MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
    const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
    const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG,
    DROPOUT_CFG, LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::PostProcessKV()
{
    if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ){
        PostProcessNZ(dkWorkspaceGm, dkGm, 1);
        if constexpr (!AscendC::IsSameType<T1, float>::value){
            set_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3);
            wait_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3);
            PostProcessNZ(dvWorkspaceGm, dvGm, 2);
        }
    }else{
        PostProcessND(dkWorkspaceGm, dkGm, 1);
        if constexpr (!AscendC::IsSameType<T1, float>::value){
            set_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3);
            wait_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3);
            PostProcessND(dvWorkspaceGm, dvGm, 2);
        }
    }
}

template <typename T1, typename T2, const MatmulConfig& MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
    const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
    const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG, LAYOUT,
    MM2_OUT_FORMAT, POST, L1CUSTOM>::PostProcessNZ(GlobalTensor<float> &workspaceGm, GlobalTensor<T1> &outGm, const uint32_t &type)
{
    uint64_t S = type == TYPE_Q ? dimS1 : dimS2;
    uint64_t halfS = 0;
    if constexpr (L1CUSTOM) {
        halfS = type == TYPE_Q ? dimS1 / 2 : dimS2 / 2;
        S = subBlockIdx == 0 ? halfS : S - halfS;
    }


    // tiling S
    uint64_t procS = postUbBaseSize / dimDAlign;
    uint64_t remainS = (S % procS == 0) ? procS : (S % procS);
    uint64_t processLoop = (S + procS - 1) / procS;

    uint64_t SIdx = 0;
    if constexpr (L1CUSTOM) {
        SIdx = subBlockIdx == 0 ? 0 : halfS;
    }

    for (uint64_t i = 0; i < processLoop; i++) {
        if (i == processLoop - 1) {
            procS = remainS;
        }

        if (i > 0) {
            wait_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3);
        }

        PostCalNZ(workspaceGm, outGm, procS, SIdx, type);

        if (i < processLoop - 1) {
            set_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3);
        }
        SIdx += procS;
    }
}

template <typename T1, typename T2, const MatmulConfig& MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
    const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
    const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG, LAYOUT,
                               MM2_OUT_FORMAT, POST, L1CUSTOM>::PostCalNZ(GlobalTensor<float> &workspaceGm,
                                                                GlobalTensor<T1> &outGm, const uint64_t &procS,
                                                                const uint64_t &SIdx, const uint32_t &type)
{
    uint64_t calBIndex = calIndices.bIndex;
    uint64_t calN2Index = calIndices.n2Index;
    uint64_t S = type == TYPE_Q ? dimS1 : dimS2;
    uint64_t G = type == TYPE_Q ? dimG : 1;
    uint64_t gIdx = type == TYPE_Q ? calIndices.gIndex : 0;
    bool needMuls = type == TYPE_V ? false : true;

    uint64_t C0Num = dimDAlign / C0_SIZE;
    uint64_t inGmOffset =
        calBIndex * dimN2 * G * S * dimDAlign + calN2Index * G * S * dimDAlign + gIdx * S * dimDAlign + SIdx * C0_SIZE;
    uint64_t dataSize = dimDAlign * procS;

    DataCopyPadExtParams<float> padExtParams{false, 0, 0, 0};
    DataCopyExtParams extParams;
    extParams.blockCount = C0Num;
    extParams.blockLen = procS * C0_SIZE * sizeof(float);
    extParams.srcStride = (S - procS) * C0_SIZE * sizeof(float);
    extParams.dstStride = 0;

    DataCopyPad(postInBuf, workspaceGm[inGmOffset], extParams, padExtParams);

    set_flag(PIPE_MTE2, PIPE_V, vWaitMte2);
    wait_flag(PIPE_MTE2, PIPE_V, vWaitMte2);

    if constexpr (AscendC::IsSameType<T1, float>::value) {
        if (needMuls) {
            Muls(postTmpBuf, postInBuf, postScaleValue, dataSize);
        }
    } else {
        if (needMuls) {
            Muls(postInBuf, postInBuf, postScaleValue, dataSize);
            pipe_barrier(PIPE_V);
        }
        Cast(postTmpBuf, postInBuf, AscendC::RoundMode::CAST_ROUND, dataSize);
    }

    DataCopyParams repeatParams;
    repeatParams.blockCount = procS;
    repeatParams.blockLen = C0_SIZE * sizeof(T1) / BLOCK;
    repeatParams.srcStride = 0;
    repeatParams.dstStride = C0Num - 1;
    uint64_t inLocalOffset = 0;
    uint64_t outLocalOffset = 0;
    uint64_t loop = C0Num;

    pipe_barrier(PIPE_V);
    for (uint64_t i = 0; i < loop; i++) {
        DataCopy(postOutBuf[outLocalOffset], postTmpBuf[inLocalOffset], repeatParams);
        inLocalOffset += C0_SIZE * procS;
        outLocalOffset += C0_SIZE;
    }

    set_flag(PIPE_V, PIPE_MTE3, mte3WaitV);
    wait_flag(PIPE_V, PIPE_MTE3, mte3WaitV);

    DataCopyExtParams outExt;
    outExt.blockCount = procS;
    outExt.blockLen = dimD * sizeof(T1);
    outExt.srcStride = 0;
    outExt.rsv = 0;
    uint64_t outGmOffset;

    if constexpr (LAYOUT == BSNGD) {
        outExt.dstStride = (dimN2 * G * dimD - dimD) * sizeof(T1);
        outGmOffset = calBIndex * S * dimN2 * G * dimD + SIdx * dimN2 * G * dimD + calN2Index * G * dimD + gIdx * dimD;
    } else if constexpr (LAYOUT == SBNGD) {
        outExt.dstStride = (dimB * dimN2 * G * dimD - dimD) * sizeof(T1);
        outGmOffset =
            SIdx * dimB * dimN2 * G * dimD + calBIndex * dimN2 * G * dimD + calN2Index * G * dimD + gIdx * dimD;
    } else if constexpr (LAYOUT == BNGSD) {
        outExt.dstStride = 0;
        outGmOffset = calBIndex * dimN2 * G * S * dimD + calN2Index * G * S * dimD + gIdx * S * dimD + SIdx * dimD;
    }

    DataCopyPad(outGm[outGmOffset], postOutBuf, outExt);
}

template <typename T1, typename T2, const MatmulConfig& MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
    const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
    const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG,
    DROPOUT_CFG, LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::PostProcessND(GlobalTensor<float> &workspaceGm, GlobalTensor<T1> &outGm, const int64_t &type)
{
    uint64_t S = type == TYPE_Q ? dimS1 : dimS2;
    uint64_t G = type == TYPE_Q ? dimG : 1;
    uint64_t gIdx = type == TYPE_Q ? calIndices.gIndex : 0;

    //tiling S
    uint64_t calBIndex = calIndices.bIndex;
    uint64_t calN2Index = calIndices.n2Index;
    uint64_t gmOffset = 0;

    uint16_t procS = postUbBaseSize / dimDAlign;
    uint64_t remainS = (S % procS == 0) ? procS : (S % procS);
    uint64_t loop = (S + procS - 1) / procS;
    uint64_t loopOffset = 0;
    
    if constexpr (LAYOUT == BSNGD){
        gmOffset = calBIndex * S * dimN2 * G * dimD + calN2Index * G * dimD + gIdx * dimD;
        loopOffset = procS * dimN2 * G * dimD;
    }else if constexpr (LAYOUT == SBNGD) {
        gmOffset = calBIndex * dimN2 * G * dimD  + calN2Index * G * dimD + gIdx * dimD;
        loopOffset = procS * dimB * dimN2 * G * dimD;
    }else if constexpr (LAYOUT == BNGSD){
        gmOffset = calBIndex * dimN2 * G * S * dimD  + calN2Index * G * S * dimD + gIdx * S * dimD;
        loopOffset = procS * dimD;
    }

    for (uint64_t i = 0; i < loop; ++i) {
        if (i == loop - 1) {
            procS = remainS;
        }
        if (i > 0) {
            wait_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3);
        }
        PostCalND(workspaceGm, outGm, procS, gmOffset, type);

        if (i < loop - 1) {
            set_flag(PIPE_MTE3, PIPE_MTE2, mte2WaitMte3);
        }
        gmOffset += loopOffset;
    }
}

template <typename T1, typename T2, const MatmulConfig& MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
    const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
    const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG,
    DROPOUT_CFG, LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::PostCalND(GlobalTensor<float> &workspaceGm, GlobalTensor<T1> &outGm, const uint16_t &procS, 
                                                          const int64_t  &gmOffset , const int64_t &type)
{   
    uint64_t G = type == TYPE_Q ? dimG : 1;
    bool needMuls = type == TYPE_V ? false : true;

    uint32_t inBlockLen = dimD * sizeof(float);
    uint32_t outBlockLen = dimD * sizeof(T1);

    uint32_t srcStride = 0;
    uint32_t dstStride = 0;
    uint64_t dataSize = procS * dimDAlign;
    uint16_t blockCount = procS;
    uint8_t rightPadding = 0;
    bool isPad = false;

    if constexpr (LAYOUT == BSNGD){
        srcStride = (dimN2 * G * dimD - dimD) * sizeof(float);
        dstStride = (dimN2 * G * dimD - dimD) * sizeof(T1);
        rightPadding = dimDAlign - dimD;
        isPad = rightPadding != 0;
    }else if constexpr (LAYOUT == SBNGD) {
        srcStride = (dimB * dimN2 * G * dimD - dimD) * sizeof(float);
        dstStride = (dimB * dimN2 * G * dimD - dimD) * sizeof(T1);
        rightPadding = dimDAlign - dimD;
        isPad = rightPadding != 0;
    }else if constexpr (LAYOUT == BNGSD){
        srcStride = 0;
        dstStride = 0;
        inBlockLen = procS * dimD * sizeof(float);
        outBlockLen = procS * dimD * sizeof(T1);
        blockCount = 1;
    }

    DataCopyPadExtParams<float> padExtParams{ isPad, 0, rightPadding, 0 };
    DataCopyExtParams extParams{ blockCount, inBlockLen, srcStride, 0, 0 };
    DataCopyPad(postInBuf, workspaceGm[gmOffset], extParams, padExtParams);

    set_flag(PIPE_MTE2, PIPE_V, vWaitMte2);
    wait_flag(PIPE_MTE2, PIPE_V, vWaitMte2);

    if constexpr (AscendC::IsSameType<T1, float>::value) {
        if(needMuls){
            Muls(postOutBuf, postInBuf, postScaleValue, dataSize);
        }
    }else {
        if(needMuls){
            Muls(postInBuf, postInBuf, postScaleValue, dataSize);
            pipe_barrier(PIPE_V);
        }
        Cast(postOutBuf, postInBuf, AscendC::RoundMode::CAST_ROUND, dataSize);
    }

    set_flag(PIPE_V, PIPE_MTE3, mte3WaitV);
    wait_flag(PIPE_V, PIPE_MTE3, mte3WaitV);

    DataCopyExtParams outExtParams{ blockCount, outBlockLen, 0, dstStride, 0 };
    DataCopyPad(outGm[gmOffset], postOutBuf, outExtParams);
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG, LAYOUT,
    MM2_OUT_FORMAT, POST, L1CUSTOM>::S1Ratio(int64_t s2_o_o, const int64_t gIdx)
{
    uint32_t kvAtomic = 1;
    uint32_t qAtomic = 1;
    // For S2 To S1
    rp.processM = vec.singleM;
    lp.s1OuterInnerNum = (rp.processM + vec.baseM - 1) / vec.baseM;
    for (uint32_t s1_o_o = 0; s1_o_o < vec.s1OuterOuterNum; s1_o_o++) {
        isFirstSingleM = s1_o_o == 0;
        isLastSingleM = s1_o_o == vec.s1OuterOuterNum - 1;
        if (unlikely(isLastSingleM)) {
            rp.processM = vec.singleMTail;
            lp.s1OuterInnerNum = (rp.processM + vec.baseM - 1) / vec.baseM;
        }

        // 设置S1索引
        s1Index = s1_o_o * vec.singleM;

        // Sparse
        if constexpr (!L1CUSTOM) {
            if (isSparse == 1) {
                CalcSparseIdx(bIndex, s1Index, rp.processM, sparse_s2_start_idx, sparse_s2_end_idx);
                if (sparse_s2_start_idx == 0 && sparse_s2_end_idx == 0) {
                    continue;
                }
            }
        }

        // mm1 mm1 左矩阵起始地址
        MMOffsetTensorA(s1Index, inputMMLeftMatrixAddr);

        // 设置S2索引 + 精细化Sparse: 判断本轮计算是否需要跳过
        s2Index = s2_o_o * vec.singleN;
        if (!CalcUsefulDataByS2()) {
            continue;
        }

        // 初始化本轮地址
        s2Index = s2_o_o * vec.singleN + usedSingleNBegin * vec.baseN;
        MMOffsetTensorB(s2Index, inputMMRighMatrixtAddr);
        mm3_4_tensor_1_s2_addr = inputMMRighMatrixtAddr;
        mm3_4_tensor_g_s1_addr = inputMMLeftMatrixAddr;


        if constexpr (MM2_OUT_FORMAT == CubeFormat::NZ) {
            MMOffsetNzOut(s1Index, s2Index, s1_addr_nzout, s2_addr_nzout);
            mm3_4_out_1_s2_addr = s2_addr_nzout;
            mm3_4_out_g_s1_addr = s1_addr_nzout;
        } else {
            mm3_4_out_1_s2_addr = inputMMRighMatrixtAddr;
            mm3_4_out_g_s1_addr = inputMMLeftMatrixAddr;
        }


        // 发射本轮 mm1 mm1
        if constexpr (MM_OUT_FORMAT == CubeFormat::NZ) {
            rp.mm1mm2OrgM = rp.processM;
        } else {
            rp.mm1mm2OrgM = tilingData->mm1TilingData.M;
        }
        SendMatmul2(rp.processM, realProcessN, inputMMLeftMatrixAddr, inputMMRighMatrixtAddr, rp.mm1mm2OrgM);
        SendMatmul1(rp.processM, realProcessN, inputMMLeftMatrixAddr, inputMMRighMatrixtAddr, rp.mm1mm2OrgM);
        if constexpr (L1CUSTOM) {
            if (currentLoop == 0) {
                mmPingpongIdx = 1 - mmPingpongIdx;
            }
            mm3InputWorkspaceAddr = (currentLoop % 2) * mm3PangInputWspOffset;
            mm4InputWorkspaceAddr = (currentLoop % 2) * mm4PangInputWspOffset;
            // 保存当轮 Index
            tmpBIndex = bIndex;
            tmpN2Index = n2Index;
            tmpGIndex = gIndex;
            tmpS1Index = s1Index;
            tmpS2Index = s2Index;
            tmpUsedSingleNBegin = usedSingleNBegin;
        }

        if (likely(currentLoop > 0)) {
            if constexpr (L1CUSTOM) {
                // vec 使用上一轮 Index
                bIndex = lastBIndex;
                n2Index = lastN2Index;
                gIndex = lastGIndex;
                s1Index = lastS1Index;
                s2Index = lastS2Index;
                usedSingleNBegin = lastUsedSingleNBegin;
                mmPingpongIdx = 1 - mmPingpongIdx;
                // 发射上一轮 vec
                VectorByCS1(mm1WorkspaceAddr, mm2WorkspaceAddr, lastMM3InputWorkspaceAddr, lastMM4InputWorkspaceAddr,
                    lastRealProcessN, lastProcessM);
            }

            // 发射上一轮 mm3 mm4
            kvAtomic = lastGIdx == 0 && lastS1OO == 0 ? 0 : 1; // first GS1, no atomic for kv
            qAtomic = lastS2OO == 0 ? 0 : 1; // first S2, no atomic for q

            if constexpr (L1CUSTOM) {
                SendMatmulDQ(lastRealProcessN, b16LastRealAlignProcessN, lastProcessM, lastMM3InputWorkspaceAddr,
                            last_mm3_4_tensor_1_s2_addr, last_mm3_4_out_g_s1_addr, isLastWorkSpaceFinished, qAtomic);
                SendMatmulDV(lastRealProcessN, b16LastRealAlignProcessN, lastProcessM, lastMM4InputWorkspaceAddr,
                            last_mm3_4_tensor_g_s1_addr, last_mm3_4_out_1_s2_addr, false, kvAtomic);
                SendMatmulDK(lastRealProcessN, b16LastRealAlignProcessN, lastProcessM, lastMM3InputWorkspaceAddr,
                            last_mm3_4_tensor_g_s1_addr, last_mm3_4_out_1_s2_addr, isLastWorkSpaceFinished, kvAtomic);
                mm345PingPangIdx += 1;
            } else {
                SendMatmulDV(lastRealProcessN, b16LastRealAlignProcessN, lastProcessM, lastMM4InputWorkspaceAddr,
                            last_mm3_4_tensor_g_s1_addr, last_mm3_4_out_1_s2_addr, false, kvAtomic);
                SendMatmulDQ(lastRealProcessN, b16LastRealAlignProcessN, lastProcessM, lastMM3InputWorkspaceAddr,
                            last_mm3_4_tensor_1_s2_addr, last_mm3_4_out_g_s1_addr, isLastWorkSpaceFinished, qAtomic);
                SendMatmulDK(lastRealProcessN, b16LastRealAlignProcessN, lastProcessM, lastMM3InputWorkspaceAddr,
                            last_mm3_4_tensor_g_s1_addr, last_mm3_4_out_1_s2_addr, isLastWorkSpaceFinished, kvAtomic);
            }
            if constexpr (POST){
                SavePostBNIdx();
            }
        }

        if constexpr (!L1CUSTOM) {
            // 发射本轮 vec
            mm3InputWorkspaceAddr = (currentLoop % 2) * mm3PangInputWspOffset;
            mm4InputWorkspaceAddr = (currentLoop % 2) * mm4PangInputWspOffset;
            rp.loopS1 = s1_o_o;
            rp.loopS2 = s2_o_o;
            VectorByCS1(mm1WorkspaceAddr, mm2WorkspaceAddr, mm3InputWorkspaceAddr, mm4InputWorkspaceAddr, 0, 0);
        }

        // 备份本轮地址
        last_mm3_4_tensor_g_s1_addr = mm3_4_tensor_g_s1_addr;
        last_mm3_4_tensor_1_s2_addr = mm3_4_tensor_1_s2_addr;
        last_mm3_4_out_g_s1_addr = mm3_4_out_g_s1_addr;
        last_mm3_4_out_1_s2_addr = mm3_4_out_1_s2_addr;
        lastMM3InputWorkspaceAddr = mm3InputWorkspaceAddr;
        lastMM4InputWorkspaceAddr = mm4InputWorkspaceAddr;
        lastRealProcessN = realProcessN;
        b16LastRealAlignProcessN = b16AlignProcessN;
        lastProcessM = rp.processM;
        lastS1OO = s1_o_o;
        lastS2OO = s2_o_o;
        lastGIdx = gIdx;
        if constexpr (L1CUSTOM) {
            rp.loopS1 = s1_o_o;
            rp.loopS2 = s2_o_o;
            // 保存上一轮 Index
            lastBIndex = tmpBIndex;
            lastN2Index = tmpN2Index;
            lastGIndex = tmpGIndex;
            lastS1Index = tmpS1Index;
            lastS2Index = tmpS2Index;
            // 恢复本轮 Index
            bIndex = tmpBIndex;
            n2Index = tmpN2Index;
            gIndex = tmpGIndex;
            s1Index = tmpS1Index;
            s2Index = tmpS2Index;
            lastUsedSingleNBegin = tmpUsedSingleNBegin;
        }
        currentLoop++; // 记录生效的绝对次数
    }

    if constexpr (POST){
        // process data when store full
        if(unlikely(postCalCnt - postCalConsum == MAX_STORE_POST_INDEX)){
            PostProcess(false);
        }

        if(unlikely(isLastSingleM && isLastSingleN)){
            lastIndices.bIndex = bIndex;
            lastIndices.n2Index = n2Index;
            lastIndices.gIndex = gIndex;
            isLastWorkSpaceFinished = true;
        }
    }

    // 当前核的最后一次
    bool is_last = isLastBN && isLastG && isLastSingleM && isLastSingleN && (lastProcessM > 0);
    if (unlikely(is_last)) {
        if constexpr (L1CUSTOM) {
            bIndex = lastBIndex;
            n2Index = lastN2Index;
            gIndex = lastGIndex;
            s1Index = lastS1Index;
            s2Index = lastS2Index;
            usedSingleNBegin = lastUsedSingleNBegin;
            mmPingpongIdx = 1 - mmPingpongIdx;
            VectorByCS1(mm1WorkspaceAddr, mm2WorkspaceAddr, mm3InputWorkspaceAddr, mm4InputWorkspaceAddr,
                lastRealProcessN, lastProcessM);
        }

        kvAtomic = lastGIdx == 0 && lastS1OO == 0 ? 0 : 1; // first GS1, no atomic for kv
        qAtomic = lastS2OO == 0 ? 0 : 1; // first S2, no atomic for q
        if constexpr (L1CUSTOM) {
            SendMatmulDQ(realProcessN, b16AlignProcessN, lastProcessM, mm3InputWorkspaceAddr, mm3_4_tensor_1_s2_addr,
                        mm3_4_out_g_s1_addr, true, qAtomic);
            SendMatmulDV(realProcessN, b16AlignProcessN, lastProcessM, mm4InputWorkspaceAddr, mm3_4_tensor_g_s1_addr,
                        mm3_4_out_1_s2_addr, !POST, kvAtomic);
            SendMatmulDK(realProcessN, b16AlignProcessN, lastProcessM, mm3InputWorkspaceAddr, mm3_4_tensor_g_s1_addr,
                        mm3_4_out_1_s2_addr, true, kvAtomic);
            mm345PingPangIdx += 1;
        } else {
            SendMatmulDV(realProcessN, b16AlignProcessN, lastProcessM, mm4InputWorkspaceAddr, mm3_4_tensor_g_s1_addr,
                        mm3_4_out_1_s2_addr, !POST, kvAtomic);
            SendMatmulDQ(realProcessN, b16AlignProcessN, lastProcessM, mm3InputWorkspaceAddr, mm3_4_tensor_1_s2_addr,
                        mm3_4_out_g_s1_addr, true, qAtomic);
            SendMatmulDK(realProcessN, b16AlignProcessN, lastProcessM, mm3InputWorkspaceAddr, mm3_4_tensor_g_s1_addr,
                        mm3_4_out_1_s2_addr, true, kvAtomic);
        }
        if constexpr (POST){
            // process data not be used
            while(postCalCnt - postCalConsum >= 0){
                PostProcess(!(postCalCnt - postCalConsum > 0));
            }
        }
    }
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG, LAYOUT,
    MM2_OUT_FORMAT, POST, L1CUSTOM>::S2Ratio(const int64_t gIdx)
{
    // Process S2 firstly
    rp.processN = vec.singleN;
    lp.s2OuterInnerNum = (rp.processN + vec.baseN - 1) / vec.baseN;

    for (int64_t s2_o_o = 0; s2_o_o < vec.s2OuterOuterNum; s2_o_o++) {
        isFirstSingleN = s2_o_o == 0;
        isLastSingleN = s2_o_o == vec.s2OuterOuterNum - 1;

        if (unlikely(isLastSingleN)) {
            rp.processN = vec.singleNTail;
            lp.s2OuterInnerNum = (rp.processN + vec.baseN - 1) / vec.baseN;
        }
        S1Ratio(s2_o_o, gIdx);
    }
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG,
                                                      DROPOUT_CFG, LAYOUT, MM2_OUT_FORMAT, POST, L1CUSTOM>::UpdateLoopParams(int64_t i)
{
    if constexpr (L1CUSTOM) {
        bIndex = (tilingData->tndSplitCoreParams.bN2idxStarts[cubeBlockIdx] + i) / dimN2;
    } else {
        bIndex = (tilingData->tndSplitCoreParams.bN2idxStarts[blockIdx] + i) / dimN2;
    }
    seqS1S2ProductSum = 0;
    seqS1CurrentOffset = 0;
    seqS2CurrentOffset = 0;
    if (unlikely(bIndex == 0)) {
        seqS1Current = ((__gm__ int64_t *)actual_seq_qlen_addr)[bIndex];
        seqS2Current = ((__gm__ int64_t *)actual_seq_kvlen_addr)[bIndex];
    } else {
        seqS1Current =
            ((__gm__ int64_t *)actual_seq_qlen_addr)[bIndex] - ((__gm__ int64_t *)actual_seq_qlen_addr)[bIndex - 1];
        seqS2Current =
            ((__gm__ int64_t *)actual_seq_kvlen_addr)[bIndex] - ((__gm__ int64_t *)actual_seq_kvlen_addr)[bIndex - 1];
        seqS1CurrentOffset = ((__gm__ int64_t *)actual_seq_qlen_addr)[bIndex - 1];
        seqS2CurrentOffset = ((__gm__ int64_t *)actual_seq_kvlen_addr)[bIndex - 1];
        for (int64_t i = 0; i < bIndex; i++) {
            int64_t seqS1Len = 0;
            int64_t seqS2Len = 0;
            if (unlikely(i == 0)) {
                seqS1Len = ((__gm__ int64_t *)actual_seq_qlen_addr)[i];
                seqS2Len = ((__gm__ int64_t *)actual_seq_kvlen_addr)[i];
            } else {
                seqS1Len =
                    ((__gm__ int64_t *)actual_seq_qlen_addr)[i] - ((__gm__ int64_t *)actual_seq_qlen_addr)[i - 1];
                seqS2Len =
                    ((__gm__ int64_t *)actual_seq_kvlen_addr)[i] - ((__gm__ int64_t *)actual_seq_kvlen_addr)[i - 1];
            }
            seqS1S2ProductSum += seqS1Len * seqS2Len;
        }
    }
    if constexpr (L1CUSTOM) {
        n2Index = (tilingData->tndSplitCoreParams.bN2idxStarts[cubeBlockIdx] + i) % dimN2;
    } else {
        n2Index = (tilingData->tndSplitCoreParams.bN2idxStarts[blockIdx] + i) % dimN2;
    }
    vec.singleMTail = seqS1Current % vec.singleM == 0 ? vec.singleM : seqS1Current % vec.singleM;
    vec.singleNTail = seqS2Current % vec.singleN == 0 ? vec.singleN : seqS2Current % vec.singleN;
    vec.s2OuterOuterNum = (seqS2Current + vec.singleN - 1) / vec.singleN;
    vec.s1OuterOuterNum = (seqS1Current + vec.singleM - 1) / vec.singleM;
}

template <typename T1, typename T2, const MatmulConfig &MM_CFG, const CubeFormat MM_OUT_FORMAT, const uint64_t PSE_CFG,
          const uint64_t ATTEN_MASK_CFG, const uint64_t DROPOUT_CFG, const uint32_t LAYOUT,
          const CubeFormat MM2_OUT_FORMAT, const bool POST, const bool L1CUSTOM>
__aicore__ inline void
FlashAttentionScoreGradS1s2Bn2<T1, T2, MM_CFG, MM_OUT_FORMAT, PSE_CFG, ATTEN_MASK_CFG, DROPOUT_CFG, LAYOUT,
    MM2_OUT_FORMAT, POST, L1CUSTOM>::Process()
{
    SyncAll();
    // clean
    if ((L1CUSTOM ? cubeBlockIdx : blockIdx) < usedCoreNum) {
        // beginBNByCore: 当前核处理BN份数据的起始索引
        // endBNByCore：当前核处理BN份数据的结束索引
        // vS1Inner, vS2Inner: Vector BaseMN
        // processN, processM: belong mm and vec together.
        for (int64_t i = 0; i < processBNByCore; i++) {
            if constexpr (LAYOUT != TND) {
                bIndex = ((L1CUSTOM ? cubeBlockIdx : blockIdx) + usedCoreNum * i) / dimN2;
                n2Index = ((L1CUSTOM ? cubeBlockIdx : blockIdx) + usedCoreNum * i) % dimN2;
            } else {
                UpdateLoopParams(i);
            }
            isLastBN = i == processBNByCore - 1 ? true : false;
            int64_t realS1, realS2;
            if constexpr (LAYOUT != TND) {
                realS1 = dimS1;
                realS2 = dimS2;
            } else {
                realS1 = seqS1Current;
                realS2 = seqS2Current;
            }
            if (realS1 == 0 || realS2 == 0) {
                continue;
            }
            if (tilingData->opInfo.sparseMode == BAND || tilingData->opInfo.sparseMode == 7 ||
                tilingData->opInfo.sparseMode == 8) {
                preTokens = realS1 - realS2 + tilingData->opInfo.preTokens;
                nextTokens = realS2 - realS1 + tilingData->opInfo.nextTokens;
            }
            for (int64_t gi = 0; gi < dimG; gi++) {
                gIndex = gi;
                isLastG = gi == dimG - 1 ? true : false;
                S2Ratio(gi);
            }
        }
    }
    
    if(!POST){
        SyncAll();
    }
}

#endif // _FLASH_ATTENTION_SCORE_GRAD_S1S2_BN2_H_
