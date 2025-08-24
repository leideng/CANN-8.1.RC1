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
 * \file LstmMmSplitNDNDFP32WithTSCM.cpp
 * \brief
 */
#include "LstmMmSplitNDNDFP32WithTSCM.h"

using namespace AscendC;

template <typename T>
__aicore__ inline int64_t LstmMmSplitNDNDFP32WithTSCM<T>::Ceil(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32WithTSCM<T>::Init(GM_ADDR inputX, GM_ADDR weight, GM_ADDR bias,
                                                            GM_ADDR seqLength, GM_ADDR initH, GM_ADDR initC,
                                                            GM_ADDR wCi, GM_ADDR wCf, GM_ADDR wCo, GM_ADDR mask,
                                                            GM_ADDR outputY, GM_ADDR outputH, GM_ADDR outputC,
                                                            GM_ADDR outputI, GM_ADDR outputJ, GM_ADDR outputF,
                                                            GM_ADDR outputO, GM_ADDR outputTanhC,
                                                            DynamicRNNTilingData* rnnTiling, GM_ADDR workspace) {
  tiling = rnnTiling;
  inputMM.Init(&(tiling->inputMMParam));
  hiddenMM.Init(&(tiling->hiddenMMParam));
  InitBuffers(inputX, weight, bias, seqLength, initH, initC, wCi, wCf, wCo, mask, outputY, outputH, outputC, outputI,
              outputJ, outputF, outputO, outputTanhC, workspace);
  InitVars();
  InitQue();
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32WithTSCM<T>::InitQue() {
  pipe.InitBuffer(qidVecIn, 1, baseVector * sizeof(float));
  pipe.InitBuffer(qidVecOut, 1, baseVector * sizeof(T));
  pipe.InitBuffer(calcBuf, 4 * baseVector * sizeof(float));
  pipe.InitBuffer(scm, 1, tiling->hiddenMMParam.singleCoreK * tiling->hiddenMMParam.singleCoreN * sizeof(T));
  // Init Local Tensors
  ubLocal1 = calcBuf.Get<float>(4 * baseVector);
  ubLocal2 = ubLocal1[baseVector];
  ubLocal3 = ubLocal2[baseVector];
  ubLocal4 = ubLocal3[baseVector];
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32WithTSCM<T>::GetCoreIndex(TCubeTiling& param, int32_t& subKIndx,
                                                                    tailSize& mmTail, int32_t kSize) {
  auto temp0 = Ceil(param.M, param.singleCoreM);
  auto temp1 = Ceil(param.N, param.singleCoreN);
  auto temp2 = Ceil(kSize, param.singleCoreK);  // 不切K, 应该=1
  if (temp0 == 0) {
    temp0 = 1;
  }
  if (temp2 == 0) {
    temp2 = 1;
  }
  auto divideKcoreNum = param.usedCoreNum / temp2;
  mmTail.mCoreIndx = (GetBlockIdx() % divideKcoreNum) % temp0;
  mmTail.nCoreIndx = (GetBlockIdx() % divideKcoreNum) / temp0;
  subKIndx = GetBlockIdx() / divideKcoreNum;  // 缺省为0
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32WithTSCM<T>::CalcGMOffset(TCubeTiling& param, TRnnOffsets& offset,
                                                                    tailSize& mmTail, int64_t cNdFormat,
                                                                    int32_t kSize) {
  int32_t subKIndx;
  GetCoreIndex(param, subKIndx, mmTail, kSize);
  offset.AOffset = mmTail.mCoreIndx * kSize * param.singleCoreM;
  offset.BOffset = mmTail.nCoreIndx * param.singleCoreN;
  offset.BiasOffset = mmTail.nCoreIndx * param.singleCoreN;

  mmTail.nCoreLoop = Ceil(param.N, param.singleCoreN);
  mmTail.tailSingleCoreN = param.N - (mmTail.nCoreLoop - 1) * param.singleCoreN;
  mmTail.notTailNCoreCount = mmTail.nCoreLoop - 1;
  mmTail.mCoreLoop = Ceil(param.M, param.singleCoreM);
  mmTail.tailSingleCoreM = param.M - (mmTail.mCoreLoop - 1) * param.singleCoreM;
  mmTail.notTailMCoreCount = mmTail.mCoreLoop - 1;

  if (cNdFormat == 1) {
    offset.COffset = mmTail.mCoreIndx * param.N * param.singleCoreM + mmTail.nCoreIndx * param.singleCoreN;
  }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32WithTSCM<T>::InitBuffers(
    GM_ADDR inputX, GM_ADDR weight, GM_ADDR bias, GM_ADDR seqLength, GM_ADDR initH, GM_ADDR initC, GM_ADDR wCi,
    GM_ADDR wCf, GM_ADDR wCo, GM_ADDR mask, GM_ADDR outputY, GM_ADDR outputH, GM_ADDR outputC, GM_ADDR outputI,
    GM_ADDR outputJ, GM_ADDR outputF, GM_ADDR outputO, GM_ADDR outputTanhC, GM_ADDR workspace) {
  CalcGMOffset(tiling->hiddenMMParam, hiddenOffsets, hiddenTail, 1, static_cast<int32_t>(tiling->hiddenSize));
  CalcGMOffset(tiling->inputMMParam, inputOffsets, inputTail, 1, static_cast<int32_t>(tiling->inputSize));
  hiddenOffsets.BOffset += tiling->inputSize * LSTM_GATE_SIZE * tiling->hiddenSize;
  inputMKAllSize = tiling->batch * tiling->inputSize;
  oneCellSize = tiling->batch * tiling->hiddenSize;
  allCellSize = oneCellSize * LSTM_GATE_SIZE;
  oriHiddenOffsets = hiddenOffsets;
  oriInputOffsets = inputOffsets;

  inputGm.xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputX),
                              tiling->timeStep * tiling->batch * tiling->inputSize);
  inputGm.weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight),
                                   (tiling->inputSize + tiling->hiddenSize) * LSTM_GATE_SIZE * tiling->hiddenSize);
  inputGm.biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias), LSTM_GATE_SIZE * tiling->hiddenSize);

  if (tiling->isInithc != 0) {
    inputGm.initHGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initH), tiling->batch * tiling->hiddenSize);
    inputGm.initCGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initC), tiling->batch * tiling->hiddenSize);
  }
  outputGm.outYGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputY),
                                  tiling->timeStep * tiling->batch * tiling->hiddenSize);
  outputGm.outHGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputH),
                                  tiling->timeStep * tiling->batch * tiling->hiddenSize);
  outputGm.outCGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputC),
                                  tiling->timeStep * tiling->batch * tiling->hiddenSize);
  if (tiling->isTraining == 1) {
    outputGm.outIGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputI),
                                    tiling->timeStep * tiling->batch * tiling->hiddenSize);
    outputGm.outJGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputJ),
                                    tiling->timeStep * tiling->batch * tiling->hiddenSize);
    outputGm.outFGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputF),
                                    tiling->timeStep * tiling->batch * tiling->hiddenSize);
    outputGm.outOGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputO),
                                    tiling->timeStep * tiling->batch * tiling->hiddenSize);
    outputGm.outTanhCGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputTanhC),
                                        tiling->timeStep * tiling->batch * tiling->hiddenSize);
  }

  outputGm.workspace.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace),
                                     tiling->timeStep * tiling->batch * LSTM_GATE_SIZE * tiling->hiddenSize);

  sync_gm.SetGlobalBuffer(
      reinterpret_cast<__gm__ int32_t*>(
          workspace + (tiling->timeStep * tiling->batch * LSTM_GATE_SIZE * tiling->hiddenSize) * sizeof(float) +
          2 * 1024 * 1024),
      8 * 48);

  InitOutput(sync_gm[GetBlockIdx() * 8], 8, 0);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32WithTSCM<T>::InitVars() {
  int64_t ubSize = 16384;
  int64_t calcMaxSize = ubSize / sizeof(T);
  int64_t blockN = tiling->usedCoreNum;

  int64_t blockCalcSize =
      Ceil(Ceil(tiling->batch * tiling->hiddenSize, blockN), tiling->hiddenSize) * tiling->hiddenSize;
  blockSize = 32 / sizeof(T);
  if (!(blockCalcSize > calcMaxSize)) {
    vectorCoreM = Ceil(Ceil(tiling->batch * tiling->hiddenSize, blockN), tiling->hiddenSize);
    vectorCoreNum = Ceil(tiling->batch, vectorCoreM);
    vectorCoreTailM = ((tiling->batch % vectorCoreM) ? (tiling->batch % vectorCoreM) : vectorCoreM);
    vectorBaseM =
        ((vectorCoreM < (calcMaxSize / tiling->hiddenSize)) ? vectorCoreM : (calcMaxSize / tiling->hiddenSize));

    vectorBaseTailM = ((vectorCoreM % vectorBaseM) ? (vectorCoreM % vectorBaseM) : vectorBaseM);
    vectorTailTailM = ((vectorCoreTailM % vectorBaseM) ? (vectorCoreTailM % vectorBaseM) : vectorBaseM);
    baseVector = Ceil(vectorBaseM * tiling->hiddenSize, blockSize) * blockSize;
  }
  iOffset = 0;
  jOffset = tiling->gateOrder == 0 ? tiling->hiddenSize : 2 * tiling->hiddenSize;
  fOffset = tiling->gateOrder == 0 ? 2 * tiling->hiddenSize : tiling->hiddenSize;
  oOffset = 3 * tiling->hiddenSize;
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32WithTSCM<T>::ProcessInputMM() {
  if (GetBlockIdx() < tiling->inputMMParam.usedCoreNum) {
    inputMM.SetTensorA(inputGm.xGm[inputOffsets.AOffset]);
    inputMM.SetTensorB(inputGm.weightGm[inputOffsets.BOffset]);
    inputMM.SetBias(inputGm.biasGm[inputOffsets.BOffset]);
    if (inputTail.nCoreIndx == inputTail.notTailNCoreCount && inputTail.mCoreIndx == inputTail.notTailMCoreCount) {
      inputMM.SetTail(inputTail.tailSingleCoreM, inputTail.tailSingleCoreN);
    } else if (inputTail.nCoreIndx == inputTail.notTailNCoreCount) {
      inputMM.SetTail(tiling->inputMMParam.singleCoreM, inputTail.tailSingleCoreN);
    } else if (inputTail.mCoreIndx == inputTail.notTailMCoreCount) {
      inputMM.SetTail(inputTail.tailSingleCoreM, tiling->inputMMParam.singleCoreN);
    }
    inputMM.IterateAll(outputGm.workspace[inputOffsets.COffset], false);
  }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32WithTSCM<T>::ProcessHiddenMM(int64_t tIdx) {
  if (GetBlockIdx() < tiling->hiddenMMParam.usedCoreNum) {
    if (tiling->direction == 1) {
      hiddenOffsets.COffset = oriHiddenOffsets.COffset + (tiling->timeStep - 1 - tIdx) * allCellSize;
    } else {
      hiddenOffsets.COffset = oriHiddenOffsets.COffset + tIdx * allCellSize;
    }
    hiddenMM.SetTensorA(inputGm.initHGm[hiddenOffsets.AOffset]);
    hiddenMM.SetTensorB(scmLocal);
    if (hiddenTail.nCoreIndx == hiddenTail.notTailNCoreCount && hiddenTail.mCoreIndx == hiddenTail.notTailMCoreCount) {
      hiddenMM.SetTail(hiddenTail.tailSingleCoreM, hiddenTail.tailSingleCoreN);
    } else if (hiddenTail.nCoreIndx == hiddenTail.notTailNCoreCount) {
      hiddenMM.SetTail(tiling->hiddenMMParam.singleCoreM, hiddenTail.tailSingleCoreN);
    } else if (hiddenTail.mCoreIndx == hiddenTail.notTailMCoreCount) {
      hiddenMM.SetTail(hiddenTail.tailSingleCoreM, tiling->hiddenMMParam.singleCoreN);
    }
    hiddenMM.IterateAll(outputGm.workspace[hiddenOffsets.COffset], true);
  }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32WithTSCM<T>::CopyGate(LocalTensor<T>& ub, GlobalTensor<T>& gm, int64_t mIdx,
                                                                int64_t gateOffset) {
  size_t oneBlockBatch = calcSize / tiling->hiddenSize;
  for (size_t i = 0; i < oneBlockBatch; ++i) {
    DataCopy(ub[i * tiling->hiddenSize],
             gm[gateOffset + blockIdx * vectorCoreM * tiling->hiddenSize * 4 +
                mIdx * vectorBaseM * tiling->hiddenSize * 4 + i * tiling->hiddenSize * 4],
             (tiling->hiddenSize + blockSize - 1) / blockSize * blockSize);
  }
  qidVecIn.EnQue(ub);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32WithTSCM<T>::CopyWithSigmoid(LocalTensor<T>& dstUb, GlobalTensor<T>& mixGm,
                                                                       int64_t mIdx, int64_t gateOffset) {
  LocalTensor<T> ubLocalIn = qidVecIn.AllocTensor<T>();
  CopyGate(ubLocalIn, mixGm, mIdx, gateOffset);
  ubLocalIn = qidVecIn.DeQue<T>();
  Sigmoid(dstUb, ubLocalIn, calcSizeAlign);
  qidVecIn.FreeTensor(ubLocalIn);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32WithTSCM<T>::CopyWithTanh(LocalTensor<T>& dstUb, GlobalTensor<T>& mixGm,
                                                                    int64_t mIdx, int64_t gateOffset) {
  LocalTensor<T> ubLocalIn = qidVecIn.AllocTensor<T>();
  CopyGate(ubLocalIn, mixGm, mIdx, gateOffset);
  ubLocalIn = qidVecIn.DeQue<T>();
  Tanh(dstUb, ubLocalIn, calcSizeAlign);
  qidVecIn.FreeTensor(ubLocalIn);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32WithTSCM<T>::CopyWithMul(LocalTensor<T>& dstUb, LocalTensor<T>& other,
                                                                   GlobalTensor<T>& mixGm, int64_t mIdx) {
  LocalTensor<T> ubLocalIn = qidVecIn.AllocTensor<T>();
  DataCopy(ubLocalIn, mixGm[blockIdx * vectorCoreM * tiling->hiddenSize + mIdx * vectorBaseM * tiling->hiddenSize],
           calcSizeAlign);
  qidVecIn.EnQue(ubLocalIn);
  ubLocalIn = qidVecIn.DeQue<T>();
  Mul(dstUb, ubLocalIn, other, calcSizeAlign);
  qidVecIn.FreeTensor(ubLocalIn);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32WithTSCM<T>::CopyOutput(GlobalTensor<T>& gm, LocalTensor<T>& ub, int64_t tIdx,
                                                                  int64_t mIdx) {
  LocalTensor<T> outLocal = qidVecOut.AllocTensor<T>();
  pipe_barrier(PIPE_V);
  Muls(outLocal, ub, (float)1.0, calcSizeAlign);
  qidVecOut.EnQue(outLocal);
  outLocal = qidVecOut.DeQue<T>();
  int64_t offset;
  if (tiling->direction == 1) {
    offset = (tiling->timeStep - 1 - tIdx) * tiling->batch * tiling->hiddenSize +
             blockIdx * vectorCoreM * tiling->hiddenSize + mIdx * vectorBaseM * tiling->hiddenSize;
  } else {
    offset = tIdx * tiling->batch * tiling->hiddenSize + blockIdx * vectorCoreM * tiling->hiddenSize +
             mIdx * vectorBaseM * tiling->hiddenSize;
  }
  DataCopy(gm[offset], outLocal, calcSize);
  if (calcSize % blockSize != 0) {
    int64_t tailOffset = calcSize - blockSize;
    DataCopy(gm[offset + tailOffset], outLocal[tailOffset], blockSize);
  }
  qidVecOut.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32WithTSCM<T>::ProcessVectorOnce(int64_t tIdx, int64_t mIdx,
                                                                         GlobalTensor<T>& mixGm) {
  blockIdx = GetBlockIdx();

  if (mIdx < Ceil(static_cast<uint32_t>(vectorCoreM), static_cast<uint32_t>(vectorBaseM)) - 1) {
    calcSize = vectorBaseM * tiling->hiddenSize;
  } else if (blockIdx < vectorCoreNum - 1) {
    calcSize = vectorBaseTailM * tiling->hiddenSize;
  } else {
    calcSize = vectorTailTailM * tiling->hiddenSize;
  }
  calcSizeAlign = Ceil(static_cast<uint32_t>(calcSize), static_cast<uint32_t>(blockSize)) * blockSize;

  // f 1 2 3 4 -> [1] 2 3 4
  auto fSigmoid = ubLocal1;
  CopyWithSigmoid(fSigmoid, mixGm, mIdx, fOffset);
  if (tiling->isTraining == 1) {
    CopyOutput(outputGm.outFGm, fSigmoid, tIdx, mIdx);
  }

  // [1] 2 3 4 -> 1 [2] 3 4
  auto cTmp1 = ubLocal2;
  CopyWithMul(cTmp1, fSigmoid, inputGm.initCGm, mIdx);
  // i 1 [2] 3 4 -> [1] [2] 3 4
  auto iSigmoid = ubLocal1;
  CopyWithSigmoid(iSigmoid, mixGm, mIdx, iOffset);
  if (tiling->isTraining == 1) {
    CopyOutput(outputGm.outIGm, iSigmoid, tIdx, mIdx);
  }

  // j [1] [2] 3 4 -> [1] [2] [3] 4
  auto jTanh = ubLocal3;
  CopyWithTanh(jTanh, mixGm, mIdx, jOffset);
  if (tiling->isTraining == 1) {
    CopyOutput(outputGm.outJGm, jTanh, tIdx, mIdx);
  }

  // i * j [1] [2] [3] 4 -> 1 [2] 3 [4]
  auto cTmp2 = ubLocal4;
  Mul(cTmp2, jTanh, iSigmoid, calcSizeAlign);
  pipe_barrier(PIPE_V);

  // i * j + f * c 1 [2] 3 [4] -> [1] 2 3 4
  auto updateC = ubLocal1;
  Add(updateC, cTmp1, cTmp2, calcSizeAlign);
  CopyOutput(outputGm.outCGm, updateC, tIdx, mIdx);

  // tanh(c) 1 [2] 3 4 -> 1 [2] 3 4
  auto cTanh = ubLocal2;
  Tanh(cTanh, updateC, calcSizeAlign);
  if (tiling->isTraining == 1) {
    CopyOutput(outputGm.outTanhCGm, cTanh, tIdx, mIdx);
  }

  // o 1 [2] 3 4 -> [1] [2] 3 4
  auto oSigmoid = ubLocal1;
  CopyWithSigmoid(oSigmoid, mixGm, mIdx, oOffset);
  if (tiling->isTraining == 1) {
    CopyOutput(outputGm.outOGm, oSigmoid, tIdx, mIdx);
  }
  pipe_barrier(PIPE_V);

  // o * Tanh(c) [1] [2] 3 4 -> 1 2 [3] 4
  auto updateH = ubLocal3;
  Mul(updateH, oSigmoid, cTanh, calcSizeAlign);
  CopyOutput(outputGm.outHGm, updateH, tIdx, mIdx);
  CopyOutput(outputGm.outYGm, updateH, tIdx, mIdx);
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32WithTSCM<T>::ProcessVector(int64_t tIdx) {
  auto mCoreIndex = GetBlockIdx();
  if (mCoreIndex < vectorCoreNum) {
    auto coreLoop = Ceil(static_cast<uint32_t>(vectorCoreM), static_cast<uint32_t>(vectorBaseM));
    if (mCoreIndex == vectorCoreNum - 1) {
      coreLoop = Ceil(static_cast<uint32_t>(vectorCoreTailM), static_cast<uint32_t>(vectorBaseM));
    }
    int64_t offset;
    if (tiling->direction == 1) {
      offset = (tiling->timeStep - 1 - tIdx) * allCellSize;
    } else {
      offset = tIdx * allCellSize;
    }
    for (int j = 0; j < coreLoop; ++j) {
      auto mixGm = outputGm.workspace[offset];
      ProcessVectorOnce(tIdx, j, mixGm);
    }
  }
}

template <typename T>
__aicore__ inline void LstmMmSplitNDNDFP32WithTSCM<T>::Process() {
  ProcessInputMM();
  Nd2NzParams transParam = {1,
                            static_cast<uint16_t>(tiling->hiddenMMParam.singleCoreK),
                            static_cast<uint16_t>(tiling->hiddenMMParam.singleCoreN),
                            0,
                            static_cast<uint16_t>(4 * tiling->hiddenSize),
                            static_cast<uint16_t>(Ceil(tiling->hiddenMMParam.singleCoreK, 16) * 16),
                            1,
                            0};
  scmLocal = scm.AllocTensor<T>();
  DataCopy(scmLocal, inputGm.weightGm[hiddenOffsets.BOffset], transParam);
  scm.EnQue(scmLocal);
  scm.DeQue();

  for (auto tIdx = 0; tIdx < tiling->timeStep; tIdx++) {
    SyncAll();

    ProcessHiddenMM(tIdx);

    SyncAll();

    ProcessVector(tIdx);

    if (tiling->direction == 1) {
      inputGm.initCGm = outputGm.outCGm[(tiling->timeStep - 1 - tIdx) * tiling->batch * tiling->hiddenSize];
      inputGm.initHGm = outputGm.outHGm[(tiling->timeStep - 1 - tIdx) * tiling->batch * tiling->hiddenSize];
    } else {
      inputGm.initCGm = outputGm.outCGm[tIdx * tiling->batch * tiling->hiddenSize];
      inputGm.initHGm = outputGm.outHGm[tIdx * tiling->batch * tiling->hiddenSize];
    }
  }
  scm.FreeTensor(scmLocal);
}
