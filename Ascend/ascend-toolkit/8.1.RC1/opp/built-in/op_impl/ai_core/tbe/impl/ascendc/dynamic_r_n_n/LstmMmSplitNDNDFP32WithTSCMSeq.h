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
 * \file LstmMmSplitNDNDFP32WithTSCMSeq.h
 * \brief
 */
#ifndef _ASCENDC_LSTMMMSPLITNDNDFP32WITHTSCMSEQ_H_
#define _ASCENDC_LSTMMMSPLITNDNDFP32WITHTSCMSEQ_H_

#include "dynamic_rnn_common.h"

template <typename T>
class LstmMmSplitNDNDFP32WithTSCMSeq {
 public:
  __aicore__ inline LstmMmSplitNDNDFP32WithTSCMSeq() = default;
  __aicore__ inline void Process();
  __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR weight, GM_ADDR bias, GM_ADDR seqLength, GM_ADDR initH,
                              GM_ADDR initC, GM_ADDR wCi, GM_ADDR wCf, GM_ADDR wCo, GM_ADDR mask, GM_ADDR outputY,
                              GM_ADDR outputH, GM_ADDR outputC, GM_ADDR outputI, GM_ADDR outputJ, GM_ADDR outputF,
                              GM_ADDR outputO, GM_ADDR outputTanhC, DynamicRNNTilingData* rnnTiling, GM_ADDR workspace);
  __aicore__ inline void InitVars();
  __aicore__ inline void InitBuffers(GM_ADDR inputX, GM_ADDR weight, GM_ADDR bias, GM_ADDR seqLength, GM_ADDR initH,
                                     GM_ADDR initC, GM_ADDR wCi, GM_ADDR wCf, GM_ADDR wCo, GM_ADDR mask,
                                     GM_ADDR outputY, GM_ADDR outputH, GM_ADDR outputC, GM_ADDR outputI,
                                     GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO, GM_ADDR outputTanhC,
                                     GM_ADDR workspace);

 protected:
  struct tailSize {
    int64_t tailSingleCoreN;
    int64_t tailSingleCoreM;
    int64_t notTailNCoreCount;
    int64_t notTailMCoreCount;
    int32_t nCoreLoop;
    int32_t mCoreLoop;
    int64_t nCoreIndx;
    int64_t mCoreIndx;
  };

  __aicore__ inline void CalcGMOffset(TCubeTiling& param, TRnnOffsets& offset, tailSize& mmTail, int64_t cNdFormat,
                                      int32_t kSize);
  __aicore__ inline void InitQue();
  __aicore__ inline void ProcessInputMM();
  __aicore__ inline void ProcessHiddenMM(int64_t tIdx);
  __aicore__ inline void ProcessVectorOnce(int64_t tIdx, int64_t mIdx, AscendC::GlobalTensor<T>& mixGm);
  __aicore__ inline void ProcessVector(int64_t tIdx);
  __aicore__ inline void CopyGate(AscendC::LocalTensor<T>& ub, AscendC::GlobalTensor<T>& gm, int64_t mIdx,
                                  int64_t gateOffset);
  __aicore__ inline void CopyWithSigmoid(AscendC::LocalTensor<T>& dstUb, AscendC::GlobalTensor<T>& mixGm, int64_t mIdx,
                                         int64_t gateOffset);
  __aicore__ inline void CopyWithTanh(AscendC::LocalTensor<T>& dstUb, AscendC::GlobalTensor<T>& mixGm, int64_t mIdx,
                                      int64_t gateOffset);
  __aicore__ inline void CopyWithMul(AscendC::LocalTensor<T>& dstUb, AscendC::LocalTensor<T>& other,
                                     AscendC::GlobalTensor<T>& mixGm, int64_t mIdx);
  __aicore__ inline void CopyOutput(AscendC::GlobalTensor<T>& gm, AscendC::LocalTensor<T>& ub, int64_t tIdx,
                                    int64_t mIdx);
  __aicore__ inline void GetCoreIndex(TCubeTiling& param, int32_t& subKIndx, tailSize& mmTail, int32_t kSize);
  __aicore__ inline void CalOutYWithSeq(AscendC::LocalTensor<float>& ubLocal, AscendC::LocalTensor<float>& seqLocal,
                                        int64_t size);
  __aicore__ inline void CalOutHWithSeq(AscendC::LocalTensor<float>& yLocal, AscendC::LocalTensor<float>& hPrevLocal,
                                        AscendC::LocalTensor<float>& dstLocal, AscendC::LocalTensor<float>& seqLocal,
                                        int64_t size);
  __aicore__ inline void CalcNormalOutCWithSeq(AscendC::LocalTensor<float>& cLocal,
                                               AscendC::LocalTensor<float>& cPrevLocal,
                                               AscendC::LocalTensor<float>& seqLocal, int64_t size);
  __aicore__ inline void CopyInLastHC(AscendC::LocalTensor<float>& dstUb, AscendC::GlobalTensor<T>& outGlobal,
                                      int64_t mIdx);
  __aicore__ inline void CopyInSeqLength(AscendC::LocalTensor<float>& dstUb, int64_t mIdx, int64_t tIdx);
  __aicore__ inline int64_t Ceil(int64_t x, int64_t y);

 public:
  AscendC::TPipe pipe;

  // describe Matmul input/output dtype&format
  matmul::Matmul<matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
                 matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
                 matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>,
                 matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>>
      inputMM;

  matmul::Matmul<matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
                 matmul::MatmulType<AscendC::TPosition::TSCM, CubeFormat::NZ, T>,
                 matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>>
      hiddenMM;

 protected:
  // output GlobalTensors
  struct OutputGm {
    __aicore__ inline OutputGm() = default;
    AscendC::GlobalTensor<T> outYGm;
    AscendC::GlobalTensor<T> outHGm;
    AscendC::GlobalTensor<T> outCGm;
    AscendC::GlobalTensor<T> outIGm;
    AscendC::GlobalTensor<T> outJGm;
    AscendC::GlobalTensor<T> outFGm;
    AscendC::GlobalTensor<T> outOGm;
    AscendC::GlobalTensor<T> outTanhCGm;
    AscendC::GlobalTensor<float> workspace;
  };

  // input GlobalTensors
  struct InputGm {
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> weightGm;
    AscendC::GlobalTensor<T> biasGm;
    AscendC::GlobalTensor<T> seqLengthGm;
    AscendC::GlobalTensor<T> initHGm;
    AscendC::GlobalTensor<T> initCGm;
    AscendC::GlobalTensor<T> wciGm;
    AscendC::GlobalTensor<T> wcfGm;
    AscendC::GlobalTensor<T> wcoGm;
    AscendC::GlobalTensor<T> maskGm;
  };

  // Queue
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> qidVecIn;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> qidVecOut;
  AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;

  AscendC::TSCM<AscendC::TPosition::GM> scm;

  // LocalTensor
  AscendC::LocalTensor<float> ubLocal1, ubLocal2, ubLocal3, ubLocal4, ubLocal5, ubLocal6;
  AscendC::LocalTensor<T> scmLocal;

  OutputGm outputGm;
  InputGm inputGm;

  int64_t inputMKAllSize;
  int64_t iOffset;
  int64_t oOffset;
  int64_t jOffset;
  int64_t fOffset;
  int64_t tailSingleCoreN;
  int64_t tailSingleCoreM;
  int64_t notTailNCoreCount;
  int64_t notTailMCoreCount;
  int32_t nCoreLoop;
  int32_t mCoreLoop;
  TRnnOffsets inputOffsets;
  TRnnOffsets hiddenOffsets;

  int64_t allCellSize;
  int64_t oneCellSize;
  tailSize hiddenTail;
  tailSize inputTail;
  int32_t oriSingleCoreN;
  TRnnOffsets oriInputOffsets;
  TRnnOffsets oriHiddenOffsets;

  AscendC::GlobalTensor<int32_t> sync_gm;
  DynamicRNNTilingData* tiling;
  AscendC::LocalTensor<int> sync_buf;

  int32_t blockSize;
  int32_t vectorCoreM;
  int32_t vectorCoreTailM;
  int32_t vectorCoreNum;
  int32_t vectorBaseM;
  int32_t vectorBaseTailM;
  int32_t vectorTailTailM;
  int32_t baseVector;
  int32_t calcSize;
  int32_t calcSizeAlign;
  int32_t blockIdx;
};

#endif
