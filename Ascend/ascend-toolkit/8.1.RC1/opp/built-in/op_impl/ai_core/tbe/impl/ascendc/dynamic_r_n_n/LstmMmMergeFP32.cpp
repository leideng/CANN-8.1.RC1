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
 * \file LstmMmMergeFP32.cpp
 * \brief
 */
#include "LstmMmMergeFP32.h"

using namespace AscendC;

template <typename T>
__aicore__ inline void LstmMmMergeFP32<T>::Init(LstmBean lstmBean, DynamicRNNTilingData* rnnTiling, GM_ADDR workspace) {
  param = rnnTiling;
  inputMM.Init(&(param->inputMMParam));
  hiddenMM.Init(&(param->hiddenMMParam));
  InitBuffers(lstmBean, workspace);
  InitVars();
}

template <typename T>
__aicore__ inline void LstmMmMergeFP32<T>::InitBuffers(LstmBean lstmBean, GM_ADDR workspace) {
  CalcGMOffset(param->inputMMParam, inputOffsets, static_cast<int32_t>(param->inputSize));
  CalcGMOffset(param->hiddenMMParam, hiddenOffsets, static_cast<int32_t>(param->hiddenSize));
  bufSize = Ceil(param->inputMMParam.baseM, 16) * 16 * Ceil(param->inputMMParam.baseN, 16) * 16;

  InitInBuffers(lstmBean, workspace);
  InitOutBuffers(lstmBean, workspace);
  // Init Local Tensors
  pipe.InitBuffer(inQueue, 2, bufSize * sizeof(float));
  pipe.InitBuffer(outQueue, 2, bufSize * sizeof(float));

  pipe.InitBuffer(calcBuf, 4 * bufSize * sizeof(float));
  ubLocal2 = calcBuf.Get<float>(4 * bufSize);
  ubLocal3 = ubLocal2[bufSize];
  ubLocal4 = ubLocal3[bufSize];
  ubLocal5 = ubLocal4[bufSize];
}

template <typename T>
__aicore__ inline void LstmMmMergeFP32<T>::InitInBuffers(LstmBean lstmBean, GM_ADDR workspace) {
  GM_ADDR inputX = lstmBean.inputX;
  GM_ADDR weight = lstmBean.weight;
  GM_ADDR bias = lstmBean.bias;
  GM_ADDR seqLength = lstmBean.seqLength;
  GM_ADDR wCi = lstmBean.wCi;
  GM_ADDR wCf = lstmBean.wCf;
  GM_ADDR wCo = lstmBean.wCo;
  GM_ADDR mask = lstmBean.mask;

  xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputX + inputOffsets.AOffset * sizeof(T)),
                      param->timeStep * param->batch * param->inputSize);

  weightInputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight + inputOffsets.BOffset * sizeof(T)),
                                param->inputSize * LSTM_GATE_SIZE * param->hiddenSize);
  weightHiddenGm.SetGlobalBuffer(
      reinterpret_cast<__gm__ T*>(
          weight + (inputOffsets.BOffset + param->inputSize * param->hiddenSize * LSTM_GATE_SIZE) * sizeof(T)),
      param->hiddenSize * LSTM_GATE_SIZE * param->hiddenSize);

  biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias + inputOffsets.BiasOffset * sizeof(T)),
                         LSTM_GATE_SIZE * param->hiddenSize);
  if (param->isSeqLength != 0) {
    seqLengthGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(seqLength + hiddenOffsets.COffset * sizeof(T)),
                                param->timeStep * param->batch * param->hiddenSize);
  }
}

template <typename T>
__aicore__ inline void LstmMmMergeFP32<T>::InitOutBuffers(LstmBean lstmBean, GM_ADDR workspace) {
  GM_ADDR outputY = lstmBean.outputY;
  GM_ADDR outputH = lstmBean.outputH;
  GM_ADDR outputC = lstmBean.outputC;
  GM_ADDR outputI = lstmBean.outputI;
  GM_ADDR outputJ = lstmBean.outputJ;
  GM_ADDR outputF = lstmBean.outputF;
  GM_ADDR outputO = lstmBean.outputO;
  GM_ADDR initH = lstmBean.initH;
  GM_ADDR initC = lstmBean.initC;
  GM_ADDR outputTanhC = lstmBean.outputTanhC;
  outYGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputY + inputOffsets.COffset * sizeof(T)),
                         param->timeStep * param->batch * param->hiddenSize);
  outHGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputH + inputOffsets.COffset * sizeof(T)),
                         param->timeStep * param->batch * param->hiddenSize);
  outCGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputC + inputOffsets.COffset * sizeof(T)),
                         param->timeStep * param->batch * param->hiddenSize);
  if (param->isTraining == 1) {
    outIGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputI + inputOffsets.COffset * sizeof(T)),
                           param->timeStep * param->batch * param->hiddenSize);
    outJGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputJ + inputOffsets.COffset * sizeof(T)),
                           param->timeStep * param->batch * param->hiddenSize);
    outFGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputF + inputOffsets.COffset * sizeof(T)),
                           param->timeStep * param->batch * param->hiddenSize);
    outOGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputO + inputOffsets.COffset * sizeof(T)),
                           param->timeStep * param->batch * param->hiddenSize);
    outTanhCGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputTanhC + inputOffsets.COffset * sizeof(T)),
                               param->timeStep * param->batch * param->hiddenSize);
  }

  hiddenAOffset = hiddenOffsets.AOffset;
  inputCOffset = inputOffsets.COffset;
  outTmp.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputH + hiddenOffsets.AOffset * sizeof(T)),
                         param->timeStep * param->batch * param->hiddenSize);
  outHOriTmp.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputH), param->timeStep * param->batch * param->hiddenSize);

  int64_t outBlockNum = param->usedCoreNum > 48 ? param->usedCoreNum : 48;
  workspaceGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace), 8 * outBlockNum);
  InitOutput(workspaceGm, 8, 0);

  if (param->isInithc != 0) {
    initHGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initH + hiddenOffsets.AOffset * sizeof(T)),
                            param->timeStep * param->batch * param->hiddenSize);
    initCGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initC + hiddenOffsets.COffset * sizeof(T)),
                            param->timeStep * param->batch * param->hiddenSize);
  } else {
    initHGm = outTmp;
    initCGm = outCGm;
  }
}

template <typename T>
__aicore__ inline void LstmMmMergeFP32<T>::InitVars() {
  // 计算常用的偏移变量，提高地址偏移效率
  oriBaseM = param->inputMMParam.baseM;
  oriBaseN = param->inputMMParam.baseN;
  inputMKSize = param->inputMMParam.baseM * param->inputMMParam.singleCoreK;
  hiddenMKSize = param->hiddenMMParam.baseM * param->hiddenMMParam.singleCoreK;

  outputMKAllSize = param->inputMMParam.baseM * param->hiddenSize;
  inputMKAllSize = param->batch * param->inputSize;
  hiddenMKAllSize = param->batch * param->hiddenSize;
  inputKNAllSize = param->inputSize * param->hiddenSize * LSTM_GATE_SIZE;
  hiddenKNAllSize = param->hiddenSize * param->hiddenSize * LSTM_GATE_SIZE;

  mLoop = Ceil(param->inputMMParam.singleCoreM, param->inputMMParam.baseM);
  nLoop = Ceil(param->inputMMParam.singleCoreN, param->inputMMParam.baseN);

  tailBaseM = param->inputMMParam.singleCoreM - (mLoop - 1) * param->inputMMParam.baseM;
  tailBaseN = param->inputMMParam.singleCoreN - (nLoop - 1) * param->inputMMParam.baseN;

  jOffset = param->gateOrder == 0 ? param->hiddenSize : 2 * param->hiddenSize;
  fOffset = param->gateOrder == 0 ? 2 * param->hiddenSize : param->hiddenSize;
}

template <typename T>
__aicore__ inline void LstmMmMergeFP32<T>::CalcGMOffset(TCubeTiling& cubeParam, TRnnOffsets& offset, int32_t kSize) {
  auto temp0 = Ceil(cubeParam.M, cubeParam.singleCoreM);
  auto temp1 = Ceil(cubeParam.N, cubeParam.singleCoreN);
  auto temp2 = Ceil(kSize, cubeParam.singleCoreK);  // 不切K, 应该=1
  if (temp0 == 0) {
    temp0 = 1;
  }
  if (temp2 == 0) {
    temp2 = 1;
  }
  auto divideKcoreNum = param->usedCoreNum / temp2;
  auto subKindx = GetBlockIdx() / divideKcoreNum;  // 缺省为0
  auto nCoreIndx = (GetBlockIdx() % divideKcoreNum) / temp0;
  auto mCoreIndx = (GetBlockIdx() % divideKcoreNum) % temp0;

  // gm nd format
  offset.AOffset = mCoreIndx * kSize * cubeParam.singleCoreM;
  offset.BOffset = nCoreIndx * cubeParam.singleCoreN;
  offset.COffset = mCoreIndx * (cubeParam.N / 4) * cubeParam.singleCoreM + nCoreIndx * cubeParam.singleCoreN;
  offset.BiasOffset = nCoreIndx * cubeParam.singleCoreN;

  // 尾块M
  uint64_t gmUserM = cubeParam.M - mCoreIndx * cubeParam.singleCoreM;
  cubeParam.singleCoreM = gmUserM < cubeParam.singleCoreM ? gmUserM : cubeParam.singleCoreM;
  // 尾块N
  uint64_t gmUserN = cubeParam.N / 4 - nCoreIndx * cubeParam.singleCoreN;
  cubeParam.singleCoreN = gmUserN < cubeParam.singleCoreN ? gmUserN : cubeParam.singleCoreN;
  // 尾块K
  uint64_t gmUserK = kSize - subKindx * cubeParam.singleCoreK;
  cubeParam.singleCoreK = gmUserK < cubeParam.singleCoreK ? gmUserK : cubeParam.singleCoreK;

  cubeParam.baseM = cubeParam.baseM < cubeParam.singleCoreM ? cubeParam.baseM : cubeParam.singleCoreM;
  cubeParam.baseN = cubeParam.baseN < cubeParam.singleCoreN ? cubeParam.baseN : cubeParam.singleCoreN;
}

template <typename T>
__aicore__ inline void LstmMmMergeFP32<T>::Process() {
  /*
   * Matmul Nd进，Nz出，两个Matmul合并计算
   */
  if (param->isInithc == 0) {
    ProcessInitalT();
  }
  ProcessNormalT(param->isInithc == 0 ? 1 : 0);
}

template <typename T>
__aicore__ inline void LstmMmMergeFP32<T>::CalcOneOffset(int64_t mIdx, int64_t nIdx, int64_t tIdx) {
  if (param->direction == 1) {
    inputOffsets.AOffset = mIdx * inputMKSize + (param->timeStep - 1 - tIdx) * inputMKAllSize;
  } else {
    inputOffsets.AOffset = mIdx * inputMKSize + tIdx * inputMKAllSize;
  }
  inputOffsets.BOffset = nIdx * oriBaseN;
  inputOffsets.BiasOffset = inputOffsets.BOffset;
  outCOffset = mIdx * outputMKAllSize + nIdx * oriBaseN;

  if (param->direction == 1) {
    inputOffsets.COffset = outCOffset + (param->timeStep - 1 - tIdx) * hiddenMKAllSize;
  } else {
    inputOffsets.COffset = outCOffset + tIdx * hiddenMKAllSize;
  }

  hiddenOffsets.AOffset = mIdx * hiddenMKSize;
  hiddenOffsets.BOffset = inputKNAllSize + inputOffsets.BOffset;
  hiddenOffsets.BiasOffset = hiddenOffsets.BOffset;
  hiddenOffsets.COffset = inputOffsets.COffset;
}

template <typename T>
__aicore__ inline void LstmMmMergeFP32<T>::ProcessInitalT() {
  /*

                             nLoop
                   |------------------------|---------|
                   |                        |         |
                   |                        |         |
                   |                        |         |
         mLoop     |            A           |    C    |
                   |                        |         |
                   |                        |         |
                   |                        |         |
                   |------------------------|---------|
                   |           B            |    D    |
                   |------------------------|---------|
   */
  // T = 0 场景
  // calc area A
  for (auto mIdx = 0; mIdx < mLoop - 1; mIdx++) {
    for (auto nIdx = 0; nIdx < nLoop - 1; nIdx++) {
      CalcInitBlock(mIdx, nIdx);
    }
  }
  // calc area B
  param->inputMMParam.baseM = tailBaseM;
  inputMM.SetTail(tailBaseM, oriBaseN);  // 重置mm1的左矩阵
  for (auto nIdx = 0; nIdx < nLoop - 1; nIdx++) {
    CalcInitBlock(mLoop - 1, nIdx);
  }
  // calc area C
  param->inputMMParam.baseM = oriBaseM;
  param->inputMMParam.baseN = tailBaseN;
  inputMM.SetTail(oriBaseM, tailBaseN);  // 重置mm1的shape
  for (auto mIdx = 0; mIdx < mLoop - 1; mIdx++) {
    CalcInitBlock(mIdx, nLoop - 1);
  }
  // calc area D
  param->inputMMParam.baseM = tailBaseM;
  inputMM.SetTail(tailBaseM, tailBaseN);  // 重置mm1的shape
  CalcInitBlock(mLoop - 1, nLoop - 1);

  SyncAll();
}

template <typename T>
__aicore__ inline void LstmMmMergeFP32<T>::ProcessNormalT(int64_t startT) {
  // T > 0 场景
  for (auto tIdx = startT; tIdx < param->timeStep; tIdx++) {
    param->inputMMParam.baseM = oriBaseM;
    param->inputMMParam.baseN = oriBaseN;
    param->hiddenMMParam.baseM = oriBaseM;
    param->hiddenMMParam.baseN = oriBaseN;
    inputMM.SetTail(oriBaseM, oriBaseN);
    hiddenMM.SetTail(oriBaseM, oriBaseN);
    // calc area A
    for (auto mIdx = 0; mIdx < mLoop - 1; mIdx++) {
      for (auto nIdx = 0; nIdx < nLoop - 1; nIdx++) {
        CalcNormalBlock(mIdx, nIdx, tIdx);
      }
    }
    // calc area B
    param->inputMMParam.baseM = tailBaseM;
    param->hiddenMMParam.baseM = tailBaseM;
    inputMM.SetTail(tailBaseM, oriBaseN);   // 重置mm1的shape
    hiddenMM.SetTail(tailBaseM, oriBaseN);  // 重置mm2的shape
    for (auto nIdx = 0; nIdx < nLoop - 1; nIdx++) {
      CalcNormalBlock(mLoop - 1, nIdx, tIdx);
    }
    // calc area C
    param->inputMMParam.baseM = oriBaseM;
    param->inputMMParam.baseN = tailBaseN;
    param->hiddenMMParam.baseM = oriBaseM;
    param->hiddenMMParam.baseN = tailBaseN;
    inputMM.SetTail(oriBaseM, tailBaseN);   // 重置mm1的shape
    hiddenMM.SetTail(oriBaseM, tailBaseN);  // 重置mm2的shape
    for (auto mIdx = 0; mIdx < mLoop - 1; mIdx++) {
      CalcNormalBlock(mIdx, nLoop - 1, tIdx);
    }
    // calc area D
    param->inputMMParam.baseM = tailBaseM;
    param->hiddenMMParam.baseM = tailBaseM;
    inputMM.SetTail(tailBaseM, tailBaseN);   // 重置mm1的shape
    hiddenMM.SetTail(tailBaseM, tailBaseN);  // 重置mm2的shape
    CalcNormalBlock(mLoop - 1, nLoop - 1, tIdx);

    SyncAll();

    // 刷新initCGm, initHGm的起始地址, 每次outCGm和outHGm的值从initCGm, initHGm
    if (param->direction == 1) {
      initCGm = outCGm[(param->timeStep - 1 - tIdx) * hiddenMKAllSize];
      initHGm = outHOriTmp[(param->timeStep - 1 - tIdx) * hiddenMKAllSize + hiddenAOffset];
    } else {
      initCGm = outCGm[tIdx * hiddenMKAllSize];
      initHGm = outHOriTmp[tIdx * hiddenMKAllSize + hiddenAOffset];
    }
  }
}

template <typename T>
__aicore__ inline void LstmMmMergeFP32<T>::CalcInitBlock(int64_t mIdx, int64_t nIdx) {
  CalcOneOffset(mIdx, nIdx, 0);

  // stage 1  i
  MMWithSigmod(ubLocal3, 0);
  if (param->isTraining == 1) {
    pipe_barrier(PIPE_V);
    CopyUB2Out(outIGm, ubLocal3, 0);
  }

  // stage 2  j
  pipe_barrier(PIPE_V);
  MMWithTanh(ubLocal4, jOffset);
  if (param->isTraining == 1) {
    pipe_barrier(PIPE_V);
    CopyUB2Out(outJGm, ubLocal4, 0);
  }

  // stage 3
  pipe_barrier(PIPE_V);
  Mul(ubLocal2, ubLocal4, ubLocal3, bufSize);
  if (param->cellClip > 0) {
    pipe_barrier(PIPE_V);
    Mins(ubLocal2, ubLocal2, (float)param->cellClip, bufSize);
  }
  pipe_barrier(PIPE_V);
  CopyUB2Out(outCGm, ubLocal2, 0);

  // stage 4  f
  if (param->forgetBias == 0) {
    pipe_barrier(PIPE_V);
    MMWithSigmod(ubLocal3, fOffset);
  } else {
    pipe_barrier(PIPE_V);
    MMWithSigmodAndForgetBias(ubLocal3, fOffset);
  }
  if (param->isTraining == 1) {
    pipe_barrier(PIPE_V);
    CopyUB2Out(outFGm, ubLocal3, 0);
  }

  // stage 5 o
  pipe_barrier(PIPE_V);
  MMWithSigmod(ubLocal4, 3 * param->hiddenSize);
  if (param->isTraining == 1) {
    pipe_barrier(PIPE_V);
    CopyUB2Out(outOGm, ubLocal4, 0);
  }

  // stage 6
  pipe_barrier(PIPE_V);
  Tanh(ubLocal3, ubLocal2, bufSize);
  if (param->isTraining == 1) {
    pipe_barrier(PIPE_V);
    CopyUB2Out(outTanhCGm, ubLocal3, 0);
  }

  // stage 7
  pipe_barrier(PIPE_V);
  Mul(ubLocal2, ubLocal4, ubLocal3, bufSize);
  pipe_barrier(PIPE_V);
  CopyUB2OutYH(ubLocal2, 0);
}

template <typename T>
__aicore__ inline void LstmMmMergeFP32<T>::CalcNormalBlock(int64_t mIdx, int64_t nIdx, int64_t tIdx) {
  CalcOneOffset(mIdx, nIdx, tIdx);

  // stage 2
  CopyInInitC(ubLocal4);

  // stage 1  --calc forget_gate
  pipe_barrier(PIPE_V);
  CalcMMAdd(ubLocal2, tIdx, fOffset);
  pipe_barrier(PIPE_V);
  Adds(ubLocal2, ubLocal2, (float)param->forgetBias, bufSize);
  pipe_barrier(PIPE_V);
  Sigmoid(ubLocal3, ubLocal2, bufSize);

  // stage 11 --copyout forget_gate
  if (param->isTraining == 1) {
    pipe_barrier(PIPE_V);
    CopyUB2Out(outFGm, ubLocal3, tIdx);
  }

  // stage 3
  pipe_barrier(PIPE_V);
  Mul(ubLocal5, ubLocal4, ubLocal3, bufSize);

  // stage 4 --calc input_gate
  pipe_barrier(PIPE_V);
  CalcMMAdd(ubLocal2, tIdx, 0);
  pipe_barrier(PIPE_V);
  Sigmoid(ubLocal3, ubLocal2, bufSize);

  // stage 12 --copyout input_gate
  if (param->isTraining == 1) {
    pipe_barrier(PIPE_V);
    CopyUB2Out(outIGm, ubLocal3, tIdx);
  }

  // stage 5 --calc current_gate
  pipe_barrier(PIPE_V);
  CalcMMAdd(ubLocal2, tIdx, jOffset);
  pipe_barrier(PIPE_V);
  Tanh(ubLocal4, ubLocal2, bufSize);

  // stage 15 --copyout current_gate
  if (param->isTraining == 1) {
    pipe_barrier(PIPE_V);
    CopyUB2Out(outJGm, ubLocal4, tIdx);
  }

  // stage 6
  pipe_barrier(PIPE_V);
  Mul(ubLocal2, ubLocal4, ubLocal3, bufSize);

  // stage 7  输出outC
  pipe_barrier(PIPE_V);
  Add(ubLocal3, ubLocal5, ubLocal2, bufSize);

  if (param->cellClip > 0) {
    pipe_barrier(PIPE_V);
    Mins(ubLocal3, ubLocal3, (float)param->cellClip, bufSize);
  }

  // stage 13 --copyout outputC
  pipe_barrier(PIPE_V);
  CopyUB2Out(outCGm, ubLocal3, tIdx);

  // stage 8 输出tanhC
  pipe_barrier(PIPE_V);
  Tanh(ubLocal2, ubLocal3, bufSize);

  // stage 14 --copyout tanhc
  if (param->isTraining == 1) {
    pipe_barrier(PIPE_V);
    CopyUB2Out(outTanhCGm, ubLocal2, tIdx);
  }

  // stage 9 --calc output_gate
  pipe_barrier(PIPE_V);
  CalcMMAdd(ubLocal4, tIdx, 3 * param->hiddenSize);
  pipe_barrier(PIPE_V);
  Sigmoid(ubLocal5, ubLocal4, bufSize);

  // stage 16 --copyout output_gate
  if (param->isTraining == 1) {
    pipe_barrier(PIPE_V);
    CopyUB2Out(outOGm, ubLocal5, tIdx);
  }

  // stage 10
  pipe_barrier(PIPE_V);
  Mul(ubLocal3, ubLocal5, ubLocal2, bufSize);

  // stage 17/18 --copyout outputH/Y
  pipe_barrier(PIPE_V);
  CopyUB2OutYH(ubLocal3, tIdx);
}

template <typename T>
__aicore__ inline void LstmMmMergeFP32<T>::MMWithSigmod(LocalTensor<float>& ubLocal, int64_t offset) {
  LocalTensor<float> tmp = outQueue.AllocTensor<float>();
  CalcMM(tmp, offset);
  pipe_barrier(PIPE_V);
  Sigmoid(ubLocal, tmp, bufSize);
  outQueue.FreeTensor(tmp);
}

template <typename T>
__aicore__ inline void LstmMmMergeFP32<T>::MMWithSigmodAndForgetBias(LocalTensor<float>& ubLocal, int64_t offset) {
  LocalTensor<float> tmp = outQueue.AllocTensor<float>();
  CalcMM(tmp, offset);
  Adds(tmp, tmp, (float)param->forgetBias, bufSize);
  Sigmoid(ubLocal, tmp, bufSize);
  outQueue.FreeTensor(tmp);
}

template <typename T>
__aicore__ inline void LstmMmMergeFP32<T>::MMWithTanh(LocalTensor<float>& ubLocal, int64_t offset) {
  LocalTensor<float> tmp = outQueue.AllocTensor<float>();
  CalcMM(tmp, offset);
  pipe_barrier(PIPE_V);
  Tanh(ubLocal, tmp, bufSize);
  outQueue.FreeTensor(tmp);
}

template <typename T>
__aicore__ inline void LstmMmMergeFP32<T>::CalcMM(LocalTensor<float>& ubLocal, int64_t offset) {
  inputMM.SetTensorA(xGm[inputOffsets.AOffset]);
  inputMM.SetTensorB(weightInputGm[inputOffsets.BOffset + offset]);
  inputMM.SetBias(biasGm[inputOffsets.BOffset + offset]);
  if (param->isHF32 == 1) {
    inputMM.SetHF32(true);
  }
  inputMM.Iterate();
  inputMM.GetTensorC(ubLocal, false, true);
  inputMM.End();
}

template <typename T>
__aicore__ inline void LstmMmMergeFP32<T>::CalcMMAdd(LocalTensor<float>& ubLocal, int64_t tIdx, int64_t offset) {
  LocalTensor<float> inputUb = outQueue.AllocTensor<float>();
  LocalTensor<float> hiddenUb = outQueue.AllocTensor<float>();

  offset = inputOffsets.BOffset + offset;

  inputMM.SetTensorA(xGm[inputOffsets.AOffset]);
  inputMM.SetTensorB(weightInputGm[offset]);
  inputMM.SetBias(biasGm[offset]);
  if (param->isHF32 == 1) {
    inputMM.SetHF32(true);
  }
  inputMM.Iterate();
  inputMM.GetTensorC(inputUb, false, true);
  inputMM.End();

  // 第二个mm的输入是从上一轮的输出的outHGlobal中取到的
  hiddenMM.SetTensorA(initHGm[hiddenOffsets.AOffset]);
  hiddenMM.SetTensorB(weightHiddenGm[offset]);
  if (param->isHF32 == 1) {
    hiddenMM.SetHF32(true);
  }
  hiddenMM.Iterate();
  hiddenMM.GetTensorC(hiddenUb, false, true);
  hiddenMM.End();

  Add(ubLocal, inputUb, hiddenUb, bufSize);  // vector操作
  outQueue.FreeTensor(inputUb);
  outQueue.FreeTensor(hiddenUb);
}

template <typename T>
__aicore__ inline void LstmMmMergeFP32<T>::MoveOut(LocalTensor<float>& ubLocal) {
  LocalTensor<T> tmp = outQueue.AllocTensor<T>();
  pipe_barrier(PIPE_V);
  Muls(tmp, ubLocal, (float)1.0, bufSize);
  outQueue.EnQue(tmp);
}

template <typename T>
__aicore__ inline void LstmMmMergeFP32<T>::CopyInInitC(LocalTensor<float>& dstUb) {
  LocalTensor<T> ubLocalIn = inQueue.AllocTensor<T>();
  // 从output_c读取上一轮T循环的输出 按照ND格式读取数据, 要配置一下偏移
  Nd2NzParams transParam = {1,
                            static_cast<uint16_t>(param->inputMMParam.baseM),
                            static_cast<uint16_t>(param->inputMMParam.baseN),
                            0,
                            static_cast<uint16_t>(param->hiddenSize),
                            static_cast<uint16_t>(Ceil(param->inputMMParam.baseM, 16) * 16 * 2),
                            2,
                            0};
  DataCopy(ubLocalIn, initCGm[outCOffset], transParam);

  inQueue.EnQue(ubLocalIn);
  ubLocalIn = inQueue.DeQue<T>();
  pipe_barrier(PIPE_V);
  Muls(dstUb, ubLocalIn, (float)1.0, bufSize);
  inQueue.FreeTensor(ubLocalIn);
}

// 输出的接口
template <typename T>
__aicore__ inline void LstmMmMergeFP32<T>::CopyOut(GlobalTensor<T>& outGlobal, int64_t tIdx) {
  auto tmp = outQueue.DeQue<T>();
  Nz2NdParamsFull transParam = {1,
                                static_cast<uint16_t>(param->inputMMParam.baseM),
                                static_cast<uint16_t>(param->inputMMParam.baseN),
                                1,
                                static_cast<uint16_t>(Ceil(param->inputMMParam.baseM, 16) * 16),
                                static_cast<uint16_t>(param->inputMMParam.N / 4),
                                1};
  DataCopy(outGlobal[inputOffsets.COffset], tmp, transParam);  // 数据输出output_c
  outQueue.FreeTensor(tmp);
}

template <typename T>
__aicore__ inline void LstmMmMergeFP32<T>::CopyUB2Out(GlobalTensor<T>& outGlobal, LocalTensor<float>& ubLocal,
                                                      int64_t tIdx) {
  // 将要输出的数据搬到outque
  MoveOut(ubLocal);
  // 搬到outC的输出地址
  CopyOut(outGlobal, tIdx);
}

template <typename T>
__aicore__ inline void LstmMmMergeFP32<T>::CopyUB2OutYH(LocalTensor<float>& ubLocal, int64_t tIdx) {
  // 将要输出的数据搬到outque
  MoveOut(ubLocal);
  // 搬到y和outH的输出地址
  Nz2NdParamsFull transParam = {1,
                                static_cast<uint16_t>(param->inputMMParam.baseM),
                                static_cast<uint16_t>(param->inputMMParam.baseN),
                                1,
                                static_cast<uint16_t>(Ceil(param->inputMMParam.baseM, 16) * 16),
                                static_cast<uint16_t>(param->inputMMParam.N / 4),
                                1};
  auto tmp = outQueue.DeQue<T>();
  DataCopy(outYGm[inputOffsets.COffset], tmp, transParam);  // 数据输出output_y
  DataCopy(outHGm[inputOffsets.COffset], tmp, transParam);  // 数据输出output_h
  outQueue.FreeTensor(tmp);
}
// -------------- LstmMmMergeFP32 end -----------------