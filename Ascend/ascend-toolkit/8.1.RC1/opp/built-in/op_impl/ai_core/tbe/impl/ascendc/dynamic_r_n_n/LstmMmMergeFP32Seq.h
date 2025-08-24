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
 * \file LstmMmMergeFP32Seq.h
 * \brief
 */
#ifndef _TIK2_LstmMmMergeFP32SEQ_H_
#define _TIK2_LstmMmMergeFP32SEQ_H_

#include "dynamic_rnn_common.h"

/**
 * Matmul T进/fp32出/ND进/NZ出
 * 两个Matmul合并执行
 **/

template <typename T>
class LstmMmMergeFP32Seq {
 public:
  __aicore__ inline LstmMmMergeFP32Seq() = default;
  __aicore__ inline void Process();
  __aicore__ inline void Init(LstmBean lstmBean, DynamicRNNTilingData* rnnTiling, GM_ADDR workspace);

 protected:
  __aicore__ inline void InitVars();
  __aicore__ inline void InitBuffers(LstmBean lstmBean, GM_ADDR workspace);
  __aicore__ inline void InitInBuffers(LstmBean lstmBean, GM_ADDR workspace);
  __aicore__ inline void InitOutBuffers(LstmBean lstmBean, GM_ADDR workspace);
  __aicore__ inline void CalcGMOffset(TCubeTiling& cubeParam, TRnnOffsets& offset, int32_t kSize);
  __aicore__ inline void ProcessInitalT();
  __aicore__ inline void ProcessNormalT(int64_t startT);

  __aicore__ inline void CalcOneOffset(int64_t mIdx, int64_t nIdx, int64_t tIdx);
  __aicore__ inline void CalcInitBlock(int64_t mIdx, int64_t nIdx);

  __aicore__ inline void CalcNormalBlock(int64_t mIdx, int64_t nIdx, int64_t tIdx);

  __aicore__ inline void MMWithSigmod(AscendC::LocalTensor<float>& ubLocal, int64_t offset);
  __aicore__ inline void MMWithSigmodAndForgetBias(AscendC::LocalTensor<float>& ubLocal, int64_t offset);
  __aicore__ inline void MMWithTanh(AscendC::LocalTensor<float>& ubLocal, int64_t offset);
  __aicore__ inline void CalcMM(AscendC::LocalTensor<float>& ubLocal, int64_t offset);
  __aicore__ inline void CalcMMAdd(AscendC::LocalTensor<float>& ubLocal, int64_t tIdx, int64_t offset);

  __aicore__ inline void MoveOut(AscendC::LocalTensor<float>& ubLocal);
  __aicore__ inline void CopyInInitHC(AscendC::LocalTensor<float>& dstUb, AscendC::GlobalTensor<T>& initGlobal);
  __aicore__ inline void CopyInSeqLength(int64_t tIdx);
  __aicore__ inline void CalcInitOutCWithSeq(AscendC::LocalTensor<float>& ubLocal);
  __aicore__ inline void CalcNormalOutCWithSeq(AscendC::LocalTensor<float>& ubLocal,
                                               AscendC::LocalTensor<float>& ubLocal1,
                                               AscendC::LocalTensor<float>& ubLocal2);
  __aicore__ inline void CalNormalOutHYWithSeq(AscendC::LocalTensor<float>& ubLocal,
                                               AscendC::LocalTensor<float>& ubLocal1,
                                               AscendC::LocalTensor<float>& ubLocal2);
  __aicore__ inline void CalOutHYWithSeq(AscendC::LocalTensor<float>& ubLocal);
  __aicore__ inline void CalOutYWithSeq(AscendC::LocalTensor<float>& ubLocal);
  __aicore__ inline void CopyOut(AscendC::GlobalTensor<T>& outGlobal, int64_t tIdx);
  __aicore__ inline void CopyUB2Out(AscendC::GlobalTensor<T>& outGlobal, AscendC::LocalTensor<float>& ubLocal,
                                    int64_t tIdx);
  __aicore__ inline void CopyUB2OutYH(AscendC::LocalTensor<float>& ubLocal, int64_t tIdx);

 public:
  AscendC::TPipe pipe;

  // describe Matmul input/output dtype&format
  matmul::Matmul<matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
                 matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
                 matmul::MatmulType<AscendC::TPosition::VECCALC, CubeFormat::NZ, float>,
                 matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>>
      inputMM;

  matmul::Matmul<matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
                 matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, T>,
                 matmul::MatmulType<AscendC::TPosition::VECCALC, CubeFormat::NZ, float>>
      hiddenMM;

 protected:
  DynamicRNNTilingData* param;

  // input GlobalTensors
  AscendC::GlobalTensor<T> xGm;
  AscendC::GlobalTensor<T> weightGm;
  AscendC::GlobalTensor<T> weightInputGm;
  AscendC::GlobalTensor<T> weightHiddenGm;
  AscendC::GlobalTensor<T> biasGm;

  AscendC::GlobalTensor<T> seqLengthGm;
  AscendC::GlobalTensor<T> initHGm;
  AscendC::GlobalTensor<T> initCGm;

  // outputs
  AscendC::GlobalTensor<T> outYGm;
  AscendC::GlobalTensor<T> outHGm;
  AscendC::GlobalTensor<T> outCGm;
  AscendC::GlobalTensor<T> outIGm;
  AscendC::GlobalTensor<T> outJGm;
  AscendC::GlobalTensor<T> outFGm;
  AscendC::GlobalTensor<T> outOGm;
  AscendC::GlobalTensor<T> outTanhCGm;
  AscendC::GlobalTensor<int32_t> workspaceGm;

  AscendC::GlobalTensor<T> outTmp;
  AscendC::GlobalTensor<T> outTmp1;
  AscendC::GlobalTensor<T> outCTmp;
  AscendC::GlobalTensor<T> outHOriTmp;

  TRnnOffsets inputOffsets, hiddenOffsets;
  int64_t hiddenAOffset;
  int64_t inputCOffset;

  // Queue
  AscendC::TQue<AscendC::QuePosition::VECIN, DEFAULT_QUEUE_BUFFE_SIZE> inQueue;
  AscendC::TQue<AscendC::QuePosition::VECOUT, DEFAULT_QUEUE_BUFFE_SIZE> outQueue;
  AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;

  // LocalTensor
  AscendC::LocalTensor<float> ubLocal2, ubLocal3, ubLocal4, ubLocal5, ubLocal6, ubLocal7;

  int64_t mLoop;
  int64_t nLoop;
  int64_t bufSize;
  int64_t baseM;
  int64_t baseN;
  int64_t oriBaseM;
  int64_t oriBaseN;
  int64_t inputMKSize;
  int64_t hiddenMKSize;

  int64_t outputMKAllSize;

  int64_t inputMKAllSize;
  int64_t hiddenMKAllSize;
  int64_t inputKNAllSize;
  int64_t hiddenKNAllSize;

  int64_t tailBaseM;
  int64_t tailBaseN;

  int64_t jOffset;
  int64_t fOffset;

  int64_t outCOffset;
};

#endif
