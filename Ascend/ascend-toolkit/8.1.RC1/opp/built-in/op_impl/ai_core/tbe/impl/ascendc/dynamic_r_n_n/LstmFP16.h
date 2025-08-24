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
 * \file LstmFP16.h
 * \brief
 */
#ifndef _ASCENDC_LSTMFP16_H_
#define _ASCENDC_LSTMFP16_H_

#include "dynamic_rnn_common.h"
__aicore__ inline constexpr auto GetRnnMmConfig()
{
  auto cfg = GetNormalConfig();
  cfg.enableSetOrgShape = false;
  cfg.enableEnd = false;
  cfg.enableGetTensorC = false;
  cfg.enableQuantVector = false;
  cfg.enableSetDefineData = false;
  return cfg;
}
constexpr auto RNN_MM_CFG = GetRnnMmConfig();
template <typename T>
class LstmMmSplitNDNDFP16 {
 public:
  __aicore__ inline LstmMmSplitNDNDFP16() = default;
  __aicore__ inline void Process();
  __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR weight, GM_ADDR bias, GM_ADDR seqLength, GM_ADDR initH,
                        GM_ADDR initC, GM_ADDR wCi, GM_ADDR wCf, GM_ADDR wCo, GM_ADDR mask,
                        GM_ADDR outputY, GM_ADDR outputH, GM_ADDR outputC, GM_ADDR outputI,
                        GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO, GM_ADDR outputTanhC,
                        const DynamicRNNTilingData* __restrict rnnTiling, GM_ADDR workspace);
  __aicore__ inline void InitV2(GM_ADDR inputX, GM_ADDR weightInput, GM_ADDR weightHidden, GM_ADDR bias,
                        GM_ADDR seqLength, GM_ADDR initH, GM_ADDR initC, GM_ADDR wCi, GM_ADDR wCf, GM_ADDR wCo,
                        GM_ADDR mask, GM_ADDR outputY, GM_ADDR outputH, GM_ADDR outputC, GM_ADDR outputI,
                        GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO, GM_ADDR outputTanhC,
                        const DynamicRNNTilingData* __restrict rnnTiling, GM_ADDR workspace);
  __aicore__ inline void InitVars();
  __aicore__ inline void InitBuffers(GM_ADDR inputX, GM_ADDR weight, GM_ADDR bias, GM_ADDR seqLength, GM_ADDR initH,
                        GM_ADDR initC, GM_ADDR wCi, GM_ADDR wCf, GM_ADDR wCo, GM_ADDR mask,
                        GM_ADDR outputY, GM_ADDR outputH, GM_ADDR outputC, GM_ADDR outputI,
                        GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO, GM_ADDR outputTanhC, GM_ADDR workspace);
  __aicore__ inline void InitBuffersV2(GM_ADDR inputX, GM_ADDR weightInput, GM_ADDR weightHidden, GM_ADDR bias,
                        GM_ADDR seqLength, GM_ADDR initH, GM_ADDR initC, GM_ADDR wCi, GM_ADDR wCf,
                        GM_ADDR wCo, GM_ADDR mask, GM_ADDR outputY, GM_ADDR outputH, GM_ADDR outputC, GM_ADDR outputI,
                        GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO, GM_ADDR outputTanhC, GM_ADDR workspace);

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

  __aicore__ inline void CalcGMOffset(TCubeTiling& param, TRnnOffsets& offset, tailSize& mmTail, int32_t kSize);
  __aicore__ inline void InitQue();
  __aicore__ inline void ProcessInputMM();
  __aicore__ inline void ProcessHiddenMM(int64_t tIdx);
  __aicore__ inline void ProcessVectorOnce(int64_t tIdx, int64_t mIdx, int64_t nIdx, AscendC::GlobalTensor<float>& mixGm);
  __aicore__ inline void ProcessVectorInitHC(int64_t mIdx, int64_t nIdx, AscendC::GlobalTensor<float>& mixGm);
  __aicore__ inline void ProcessVector(int64_t tIdx);
  __aicore__ inline void ProcessInitalT();
  __aicore__ inline void CopyInHCSeq(AscendC::LocalTensor<float>& dstUb, AscendC::GlobalTensor<T>& mixGm, int64_t off);
  __aicore__ inline void CopyOutput(AscendC::GlobalTensor<T>& gm, AscendC::LocalTensor<float>& ub, int64_t off);
  __aicore__ inline void GetCoreIndex(TCubeTiling& param, int32_t& subKIndx, tailSize& mmTail, int32_t kSize);
  __aicore__ inline void CalcVecScaler(int64_t tIdx, int64_t mIdx, int64_t nIdx, int64_t& off1, int64_t& off2,
                                       int64_t& off3);
  __aicore__ inline void CopyInFJ(AscendC::LocalTensor<float>& dst, AscendC::GlobalTensor<float>& mixGm, int64_t off);
  __aicore__ inline void CopyInIO(AscendC::LocalTensor<float>& dst, AscendC::GlobalTensor<float>& mixGm, int64_t off);
  __aicore__ inline void CopyInC(AscendC::LocalTensor<T>& dst, AscendC::GlobalTensor<T>& mixGm, const int64_t off);
  __aicore__ inline void AddfSigmoid(AscendC::LocalTensor<float>& dst, AscendC::LocalTensor<float>& src, int64_t off);
  __aicore__ inline void CaliSigmoid(AscendC::LocalTensor<float>& dst, AscendC::LocalTensor<float>& src, int64_t off);
  __aicore__ inline void CaljTanh(AscendC::LocalTensor<float>& dst, AscendC::LocalTensor<float>& src, int64_t off);
  __aicore__ inline void CaloSigmoid(AscendC::LocalTensor<float>& dst, AscendC::LocalTensor<float>& src, int64_t off);
  __aicore__ inline void InitCMulfSigmoid(AscendC::LocalTensor<float>& dst, AscendC::LocalTensor<T>& src1,
                                          AscendC::LocalTensor<float>& src2);
  __aicore__ inline void CalAddTanh(AscendC::LocalTensor<float>& dst, AscendC::LocalTensor<float>& src1,
                                    AscendC::LocalTensor<float>& src2, int64_t off1, int64_t off2);
  __aicore__ inline void CalAddTanht0(AscendC::LocalTensor<float>& dst, AscendC::LocalTensor<float>& src1,
                                      AscendC::LocalTensor<float>& src2, int64_t off1, int64_t off2);
  __aicore__ inline void CopyOutYH(AscendC::LocalTensor<float>& src, int64_t off1, int64_t off2);
  __aicore__ inline void CopyOutYHt0(AscendC::LocalTensor<float>& src, int64_t off);

  __aicore__ inline int64_t Ceil(int64_t x, int64_t y);

 public:
  AscendC::TPipe pipe;

  // describe Matmul input/output dtype&format
  matmul::Matmul<matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
                 matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
                 matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>,
                 matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>, RNN_MM_CFG>
      inputMM;

  matmul::Matmul<matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
                 matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
                 matmul::MatmulType<AscendC::TPosition::VECCALC, CubeFormat::ND, float>,
                 matmul::MatmulType<AscendC::TPosition::VECCALC, CubeFormat::ND, float>,
                 RNN_MM_CFG>
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
    AscendC::GlobalTensor<T> weightInputGm;
    AscendC::GlobalTensor<T> weightHiddenGm;
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
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> qidCIn;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> qidVecIn;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> qidVecIn2;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> qidVecOut;
  AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;

  // LocalTensor
  AscendC::LocalTensor<float> ubLocal1, ubLocal2, ubLocal3, ubLocal4;

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
  const DynamicRNNTilingData* __restrict tiling;
  TCubeTiling inputMMTiling;
  TCubeTiling hiddenMMTiling;
  AscendC::LocalTensor<int> sync_buf;

  int64_t blockSize;
  int64_t calBlockSize;
  int64_t vectorCoreM;
  int64_t vectorTailM;
  int64_t vectorCoreNum;
  int64_t vectorBaseM;
  int64_t vectorBaseTailM;
  int64_t vectorTailTailM;
  int64_t baseVector;
  int64_t calcSize;
  int64_t calcSizeAlign;
  int64_t blockIdx;
  int64_t vectorSplitM;
  int64_t vectorSplitN;
  int64_t vectorTailSplitM;
  int64_t vectorTailN;
  int64_t vectorBaseN;
  int64_t calcM;
  int64_t calcN;
  int64_t coreCalcM;
};

#endif