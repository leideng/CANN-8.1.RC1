/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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
 * \file aglu_weight_nz_l1_fullload.h
 * \brief
 */

#ifndef AGLU_WEIGHT_NZ_L1_FULLLOAD_H
#define AGLU_WEIGHT_NZ_L1_FULLLOAD_H

#include "kernel_operator.h"

namespace AGLU {
using namespace AscendC;

constexpr static int AGLU_FORMAT_ND    = 0;
constexpr static int AGLU_FRACTAL_nZ   = 1;
constexpr static int AGLU_FRACTAL_zZ   = 2;
constexpr static int AGLU_FRACTAL_zN   = 3;

constexpr static int SINGLE_WEIGHT     = 1;
constexpr static int DUAL_WEIGHT       = 2;
constexpr static int FRACTAL_SIZE      = 16;
constexpr static int BASIC_BLOCK_SIZE  = 128;

constexpr static int ACT_FUNC_GELU_TANH = 1;
constexpr static int ACT_FUNC_SWISH     = 2;

class AGLUWeightNZL1Fullload {
public:
  __aicore__ inline AGLUWeightNZL1Fullload() {}
  __aicore__ inline void Init(GM_ADDR x,
                              GM_ADDR weight1,
                              GM_ADDR bias1,
                              GM_ADDR weight2,
                              GM_ADDR bias2,
                              GM_ADDR y,
                              GM_ADDR workspace,
                              const AGLUTilingData* __restrict tilingData,
                              TPipe* pipeIn) {
    pipe = pipeIn;
    auto blockIdx = GetBlockIdx();

    // load tilingData
    enableBias = tilingData->enableBias;
    matmulMode = tilingData->matmulMode;
    activateLeft = tilingData->activateLeft;
    activateFunc = tilingData->activateFunc;
    baseM = tilingData->baseM;
    baseK = tilingData->baseK;
    baseN = tilingData->baseN;
    singleCoreM = tilingData->singleCoreM;
    singleCoreK = tilingData->singleCoreK;
    singleCoreN = tilingData->singleCoreN;
    matrixASize = tilingData->matrixASize;
    matrixAOffset = tilingData->matrixAOffset;
    matrixBSize = tilingData->matrixBSize;
    matrixBOffset = tilingData->matrixBOffset;
    matrixCSize = tilingData->matrixCSize;
    matrixCOffset = tilingData->matrixCOffset;
    matmulModeCoeff = matmulMode == DUAL_WEIGHT ? 1 : 2;

    // set global buffer
    xGm.SetGlobalBuffer((__gm__ half*)x + blockIdx * matrixAOffset, matrixASize);
    yGm.SetGlobalBuffer((__gm__ half*)y + blockIdx * matrixCOffset, matrixCSize);
    if (matmulMode == SINGLE_WEIGHT && activateLeft == 1) {
      weight1Gm.SetGlobalBuffer((__gm__ half*)weight1, matrixBSize);
      weight2Gm.SetGlobalBuffer((__gm__ half*)weight1 + matrixBOffset, matrixBSize);
      if (enableBias == 1) {
        bias1Gm.SetGlobalBuffer((__gm__ half*)bias1, singleCoreN);
        bias2Gm.SetGlobalBuffer((__gm__ half*)bias1 + singleCoreN, singleCoreN);
      }
    }
    if (matmulMode == SINGLE_WEIGHT && activateLeft == 0) {
      weight1Gm.SetGlobalBuffer((__gm__ half*)weight1 + matrixBOffset, matrixBSize);
      weight2Gm.SetGlobalBuffer((__gm__ half*)weight1, matrixBSize);
      if (enableBias == 1) {
        bias1Gm.SetGlobalBuffer((__gm__ half*)bias1 + singleCoreN, singleCoreN);
        bias2Gm.SetGlobalBuffer((__gm__ half*)bias1, singleCoreN);
      }
    }
    if (matmulMode == DUAL_WEIGHT && activateLeft == 1) {
      weight1Gm.SetGlobalBuffer((__gm__ half*)weight1, matrixBSize);
      weight2Gm.SetGlobalBuffer((__gm__ half*)weight2, matrixBSize);
      if (enableBias == 1) {
        bias1Gm.SetGlobalBuffer((__gm__ half*)bias1, singleCoreN);
        bias2Gm.SetGlobalBuffer((__gm__ half*)bias2, singleCoreN);
      }
    }
    if (matmulMode == DUAL_WEIGHT && activateLeft == 0) {
      weight1Gm.SetGlobalBuffer((__gm__ half*)weight2, matrixBSize);
      weight2Gm.SetGlobalBuffer((__gm__ half*)weight1, matrixBSize);
      if (enableBias == 1) {
        bias1Gm.SetGlobalBuffer((__gm__ half*)bias2, singleCoreN);
        bias2Gm.SetGlobalBuffer((__gm__ half*)bias1, singleCoreN);
      }
    }

    // pipe init buffer
    // UB
    pipe->InitBuffer(inQueueTensorA, 2, BASIC_BLOCK_SIZE * BASIC_BLOCK_SIZE * sizeof(float));
    pipe->InitBuffer(outQueueTensor, 1, BASIC_BLOCK_SIZE * BASIC_BLOCK_SIZE * sizeof(float));
    // L1
    pipe->InitBuffer(inQueueTensorAL1, 1, singleCoreM * singleCoreK * sizeof(half));
    pipe->InitBuffer(inQueueTensorBL1, 2, baseK * baseN * sizeof(half));
    // L0
    pipe->InitBuffer(inQueueTensorAL0, 2, baseM * baseK * sizeof(half));
    pipe->InitBuffer(inQueueTensorBL0, 2, baseK * baseN * sizeof(half));
    pipe->InitBuffer(inQueueTensorCL0, 2, baseM * baseN * sizeof(float));
  }

  __aicore__ inline void Process() {
    // Load MatrixA to L1
    LocalTensor<half> aL1 = inQueueTensorAL1.template AllocTensor<half>();
    for (uint32_t m = 0; m < singleCoreM; m+=baseM) {
      for (uint32_t k = 0; k < singleCoreK; k+=baseK) {
        LoadNDMatrixAToL1(aL1[m * singleCoreK + k * FRACTAL_SIZE],
                          xGm[m * singleCoreK + k],
                          inQueueTensorA, outQueueTensor,
                          singleCoreM, singleCoreK, baseM, baseK);
      }
    }
    inQueueTensorAL1.EnQue(aL1);
    aL1 = inQueueTensorAL1.template DeQue<half>();

    /****************** Main Loop ******************/
    for (uint32_t n = 0; n < singleCoreN; n += baseN) {

      LocalTensor<float> c1L0 = inQueueTensorCL0.template AllocTensor<float>();
      LocalTensor<half> bL1Ping = inQueueTensorBL1.template AllocTensor<half>();
      LocalTensor<half> bL1Pong = inQueueTensorBL1.template AllocTensor<half>();
      LocalTensor<half> aL0Ping = inQueueTensorAL0.template AllocTensor<half>();
      LocalTensor<half> aL0Pong = inQueueTensorAL0.template AllocTensor<half>();
      LocalTensor<half> bL0Ping = inQueueTensorBL0.template AllocTensor<half>();
      LocalTensor<half> bL0Pong = inQueueTensorBL0.template AllocTensor<half>();
      for (uint32_t k = 0; k < singleCoreK; k += baseK) {
        uint32_t idx = (k / baseK);
        bool isPing = (idx % 2 == 0);
        bool isPong = !isPing;
        bool isFirstPing = (idx == 0);
        bool isFirstPong = (idx == 1);
        bool isLastPing = false;
        bool isLastPong = false;
        if (k + baseK + baseK >= singleCoreK && isPing) { isLastPing = true; }
        if (k + baseK + baseK >= singleCoreK && isPong) { isLastPong = true; }

        MmadParams mmadParams;
        mmadParams.m              = static_cast<uint16_t>(baseM);
        mmadParams.n              = static_cast<uint16_t>(baseN);
        mmadParams.k              = static_cast<uint16_t>(baseK);
        mmadParams.cmatrixInitVal = static_cast<bool>(k == 0);
        mmadParams.cmatrixSource  = static_cast<bool>(false);

        event_t eventIdMte2ToMte1Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>());
        event_t eventIdMte2ToMte1Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>());
        event_t eventIdMte1ToMte2Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>());
        event_t eventIdMte1ToMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>());
        event_t eventIdMte1ToMPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_M>());
        event_t eventIdMte1ToMPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_M>());
        event_t eventIdMToMte1Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>());
        event_t eventIdMToMte1Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>());

        if (!isFirstPing && !isFirstPong) {
          WaitFlag<HardEvent::M_MTE1>(isPing ? eventIdMToMte1Ping : eventIdMToMte1Pong);
        }
        LoadBasicBlock<half, AGLU_FRACTAL_zZ, true>(isPing ? aL0Ping : aL0Pong, aL1[k * FRACTAL_SIZE],
                                                    singleCoreM, singleCoreK, baseM, baseK);
        if (!isFirstPing && !isFirstPong) {
          WaitFlag<HardEvent::MTE1_MTE2>(isPing ? eventIdMte1ToMte2Ping : eventIdMte1ToMte2Pong);
        }
        CopyInBasicBlock<half, AGLU_FRACTAL_nZ>(isPing ? bL1Ping : bL1Pong,
                                                weight1Gm[k * singleCoreN * matmulModeCoeff + n * FRACTAL_SIZE],
                                                singleCoreK, singleCoreN * matmulModeCoeff,
                                                baseK, baseN);
        SetFlag<HardEvent::MTE2_MTE1>(isPing ? eventIdMte2ToMte1Ping : eventIdMte2ToMte1Pong);

        WaitFlag<HardEvent::MTE2_MTE1>(isPing ? eventIdMte2ToMte1Ping : eventIdMte2ToMte1Pong);
        LoadBasicBlock<half, AGLU_FRACTAL_nZ, false>(isPing ? bL0Ping : bL0Pong,
                                                     isPing ? bL1Ping : bL1Pong,
                                                     baseK, baseN,
                                                     baseK, baseN);
        if (!isLastPing && !isLastPong) {
          SetFlag<HardEvent::MTE1_MTE2>(isPing ? eventIdMte1ToMte2Ping : eventIdMte1ToMte2Pong);
        }
        SetFlag<HardEvent::MTE1_M>(isPing ? eventIdMte1ToMPing : eventIdMte1ToMPong);

        WaitFlag<HardEvent::MTE1_M>(isPing ? eventIdMte1ToMPing : eventIdMte1ToMPong);
        Mmad(c1L0, isPing ? aL0Ping : aL0Pong, isPing ? bL0Ping : bL0Pong, mmadParams);
        if (!isLastPing && !isLastPong) {
          SetFlag<HardEvent::M_MTE1>(isPing ? eventIdMToMte1Ping : eventIdMToMte1Pong);
        }

        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1Ping);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1Pong);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2Ping);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2Pong);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_M>(eventIdMte1ToMPing);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_M>(eventIdMte1ToMPong);
        GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(eventIdMToMte1Ping);
        GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(eventIdMToMte1Pong);
      }
      inQueueTensorBL1.FreeTensor(bL1Ping);
      inQueueTensorBL1.FreeTensor(bL1Pong);
      inQueueTensorBL0.FreeTensor(bL0Ping);
      inQueueTensorBL0.FreeTensor(bL0Pong);
      inQueueTensorAL0.FreeTensor(aL0Ping);
      inQueueTensorAL0.FreeTensor(aL0Pong);

        // Load Bias1 from GM to UB
        LocalTensor<float> biasLocal = inQueueTensorA.template AllocTensor<float>();
      if (enableBias == 1) {
        DataCopy(biasLocal.ReinterpretCast<half>()[baseN], bias1Gm[n], baseN);
      }

      // Wait MatMul
      inQueueTensorCL0.EnQue(c1L0);
      c1L0 = inQueueTensorCL0.template DeQue<float>();
      LocalTensor<float> cLocal = inQueueTensorA.template AllocTensor<float>();

        // Cast Bias
        inQueueTensorA.EnQue(biasLocal);
        biasLocal = inQueueTensorA.template DeQue<float>();
      if (enableBias == 1) {
        Cast(biasLocal, biasLocal.ReinterpretCast<half>()[baseN], RoundMode::CAST_NONE, baseN);

        // Load Bias2 from GM to UB
        DataCopy(biasLocal.ReinterpretCast<half>()[3 * baseN], bias2Gm[n], baseN);
      }

      // Copy c1L0 to UB
      CopyL0C2UB<float, float>(cLocal, c1L0, baseM, baseN);
      inQueueTensorCL0.FreeTensor(c1L0);
      pipe_barrier(PIPE_V);

      if (enableBias == 1) {
        // AddBias
        AddBias<float>(cLocal, biasLocal, baseM, baseN);
        pipe_barrier(PIPE_V);
      }

      // Calculate GeLU
      LocalTensor<float> actTensor = outQueueTensor.template AllocTensor<float>();
      if (activateFunc == ACT_FUNC_GELU_TANH) {
        GeluTanh<float>(actTensor, cLocal, baseM * baseN);
      } else if (activateFunc == ACT_FUNC_SWISH) {
        Swish<float>(actTensor, cLocal, baseM * baseN);
      }


      // MatMul
      LocalTensor<float> c2L0 = inQueueTensorCL0.template AllocTensor<float>();
      bL1Ping = inQueueTensorBL1.template AllocTensor<half>();
      bL1Pong = inQueueTensorBL1.template AllocTensor<half>();
      aL0Ping = inQueueTensorAL0.template AllocTensor<half>();
      aL0Pong = inQueueTensorAL0.template AllocTensor<half>();
      bL0Ping = inQueueTensorBL0.template AllocTensor<half>();
      bL0Pong = inQueueTensorBL0.template AllocTensor<half>();
      for (uint32_t k = 0; k < singleCoreK; k += baseK) {
        uint32_t idx = (k / baseK);
        bool isPing = (idx % 2 == 0);
        bool isPong = !isPing;
        bool isFirstPing = (idx == 0);
        bool isFirstPong = (idx == 1);
        bool isLastPing = false;
        bool isLastPong = false;
        if (k + baseK + baseK >= singleCoreK && isPing) { isLastPing = true; }
        if (k + baseK + baseK >= singleCoreK && isPong) { isLastPong = true; }

        MmadParams mmadParams;
        mmadParams.m              = static_cast<uint16_t>(baseM);
        mmadParams.n              = static_cast<uint16_t>(baseN);
        mmadParams.k              = static_cast<uint16_t>(baseK);
        mmadParams.cmatrixInitVal = static_cast<bool>(k == 0);
        mmadParams.cmatrixSource  = static_cast<bool>(false);

        event_t eventIdMte2ToMte1Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>());
        event_t eventIdMte2ToMte1Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>());
        event_t eventIdMte1ToMte2Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>());
        event_t eventIdMte1ToMte2Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>());
        event_t eventIdMte1ToMPing = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_M>());
        event_t eventIdMte1ToMPong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_M>());
        event_t eventIdMToMte1Ping = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>());
        event_t eventIdMToMte1Pong = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>());

        if (!isFirstPing && !isFirstPong) {
          WaitFlag<HardEvent::M_MTE1>(isPing ? eventIdMToMte1Ping : eventIdMToMte1Pong);
        }
        LoadBasicBlock<half, AGLU_FRACTAL_zZ, true>(isPing ? aL0Ping : aL0Pong, aL1[k * FRACTAL_SIZE],
                                                    singleCoreM, singleCoreK, baseM, baseK);
        if (!isFirstPing && !isFirstPong) {
          WaitFlag<HardEvent::MTE1_MTE2>(isPing ? eventIdMte1ToMte2Ping : eventIdMte1ToMte2Pong);
        }
        CopyInBasicBlock<half, AGLU_FRACTAL_nZ>(isPing ? bL1Ping : bL1Pong,
                                                weight2Gm[k * singleCoreN * matmulModeCoeff + n * FRACTAL_SIZE],
                                                singleCoreK, singleCoreN * matmulModeCoeff,
                                                baseK, baseN);
        SetFlag<HardEvent::MTE2_MTE1>(isPing ? eventIdMte2ToMte1Ping : eventIdMte2ToMte1Pong);

        WaitFlag<HardEvent::MTE2_MTE1>(isPing ? eventIdMte2ToMte1Ping : eventIdMte2ToMte1Pong);
        LoadBasicBlock<half, AGLU_FRACTAL_nZ, false>(isPing ? bL0Ping : bL0Pong,
                                                     isPing ? bL1Ping : bL1Pong,
                                                     baseK, baseN,
                                                     baseK, baseN);
        if (!isLastPing && !isLastPong) {
          SetFlag<HardEvent::MTE1_MTE2>(isPing ? eventIdMte1ToMte2Ping : eventIdMte1ToMte2Pong);
        }
        SetFlag<HardEvent::MTE1_M>(isPing ? eventIdMte1ToMPing : eventIdMte1ToMPong);

        WaitFlag<HardEvent::MTE1_M>(isPing ? eventIdMte1ToMPing : eventIdMte1ToMPong);
        Mmad(c2L0, isPing ? aL0Ping : aL0Pong, isPing ? bL0Ping : bL0Pong, mmadParams);
        if (!isLastPing && !isLastPong) {
          SetFlag<HardEvent::M_MTE1>(isPing ? eventIdMToMte1Ping : eventIdMToMte1Pong);
        }

        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1Ping);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1Pong);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2Ping);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(eventIdMte1ToMte2Pong);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_M>(eventIdMte1ToMPing);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_M>(eventIdMte1ToMPong);
        GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(eventIdMToMte1Ping);
        GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(eventIdMToMte1Pong);
      }
      inQueueTensorBL1.FreeTensor(bL1Ping);
      inQueueTensorBL1.FreeTensor(bL1Pong);
      inQueueTensorBL0.FreeTensor(bL0Ping);
      inQueueTensorBL0.FreeTensor(bL0Pong);
      inQueueTensorAL0.FreeTensor(aL0Ping);
      inQueueTensorAL0.FreeTensor(aL0Pong);

      // Wait MatMul
      inQueueTensorCL0.EnQue(c2L0);
      c2L0 = inQueueTensorCL0.template DeQue<float>();

        // Cast Bias
        inQueueTensorA.EnQue(biasLocal);
        biasLocal = inQueueTensorA.template DeQue<float>();
      if (enableBias == 1) {
        Cast(biasLocal[baseN], biasLocal.ReinterpretCast<half>()[3 * baseN], RoundMode::CAST_NONE, baseN);
      }

      // Copy c2L0 to UB
      CopyL0C2UB<float, float>(cLocal, c2L0, baseM, baseN);
      inQueueTensorCL0.FreeTensor(c2L0);
      if (enableBias == 1) {
        pipe_barrier(PIPE_V);

        // Add Bias
        AddBias(cLocal, biasLocal[baseN], baseM, baseN);
      }
        inQueueTensorA.FreeTensor(biasLocal);
        pipe_barrier(PIPE_V);

      // Calculate Mul
      Mul(actTensor, cLocal, actTensor, baseM * baseN);
      inQueueTensorA.FreeTensor(cLocal);

      // Cast back to half
      Cast(actTensor.ReinterpretCast<half>(), actTensor, RoundMode::CAST_NONE, baseM * baseN);
      pipe_barrier(PIPE_V);

      // zN to ND
      TransDatazN2ND<half>(actTensor.ReinterpretCast<half>()[baseM * baseN],
                           actTensor.ReinterpretCast<half>(), baseM, baseN);

      // CopyOut
      outQueueTensor.EnQue(actTensor);
      actTensor = outQueueTensor.template DeQue<float>();
      CopyOutBasicBlock<half, AGLU_FORMAT_ND>(yGm[n],
                                              actTensor.ReinterpretCast<half>()[baseM * baseN],
                                              singleCoreM, singleCoreN,
                                              baseM, baseN);
      outQueueTensor.FreeTensor(actTensor);
    }

    inQueueTensorAL1.FreeTensor(aL1);

  }

private:
  template <typename T>
  __aicore__ inline void GeluTanh(LocalTensor<T> y, LocalTensor<T> x, uint32_t calcCount) {
    constexpr static float scalarOne = 1.0;
    constexpr static float beta = 0.044715;
    constexpr static float alpha = -1.5957691;

    Mul(y, x, x, calcCount);
    Muls(y, y, beta, calcCount);
    Adds(y, y, scalarOne, calcCount);
    Mul(y, x, y, calcCount);
    Muls(y, y, alpha, calcCount);
    Exp(y, y, calcCount);
    Adds(y, y, scalarOne, calcCount);
    Div(y, x, y, calcCount);
  }

  template <typename T>
  __aicore__ inline void Swish(LocalTensor<T> y, LocalTensor<T> x, uint32_t calcCount) {
    Muls(y, x, static_cast<T>(-1.0), calcCount);
    Exp(y, y, calcCount);
    Adds(y, y, static_cast<T>(1.0), calcCount);
    Div(y, x, y, calcCount);
  }

  template <typename T, typename U>
  __aicore__ inline void CopyL0C2UB(LocalTensor<T> yLocal,
                                    LocalTensor<U> xLocal,
                                    uint32_t base_m, uint32_t base_n) {
    constexpr static int FRACTAL_SIZE = 16;
    constexpr static int BLOCK_SIZE   = 32;
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen   = (base_m * base_n) / (FRACTAL_SIZE * FRACTAL_SIZE);
    intriParams.srcStride  = 0;
    intriParams.dstStride  = 0;
    DataCopyEnhancedParams enhancedParams;
    enhancedParams.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(yLocal, xLocal, intriParams, enhancedParams);
  }

  template <typename T>
  __aicore__ inline void AddBias(LocalTensor<T> dstLocal,
                                 LocalTensor<T> biasLocal,
                                 uint32_t base_m, uint32_t base_n) {
    constexpr static int FRACTAL_SIZE = 16;
    constexpr static int BLOCK_SIZE   = 32;
    constexpr static int REPEAT_SIZE  = 8;
    uint64_t mask[2] = { UINT64_MAX, UINT64_MAX };
    if constexpr (std::is_same<T, half>::value) {
      for (uint32_t n = 0; n < base_n; n += FRACTAL_SIZE) {
        Add(dstLocal[base_m * n].template ReinterpretCast<T>(),
            biasLocal[n].template ReinterpretCast<T>(),
            dstLocal[base_m * n].template ReinterpretCast<T>(),
            mask, base_m / REPEAT_SIZE,
            {1, 0, 1, 8, 0, 8});
      }
    }
    if constexpr (std::is_same<T, float>::value) {
      for (uint32_t n = 0; n < base_n; n += FRACTAL_SIZE) {
        Add(dstLocal[base_m * n].template ReinterpretCast<T>(),
            biasLocal[n].template ReinterpretCast<T>(),
            dstLocal[base_m * n].template ReinterpretCast<T>(),
            mask, static_cast<uint8_t>(base_m / REPEAT_SIZE),
            {2, 0, 2, 16, 0, 16});
        Add(dstLocal[base_m * n + REPEAT_SIZE].template ReinterpretCast<T>(),
            biasLocal[n + REPEAT_SIZE].template ReinterpretCast<T>(),
            dstLocal[base_m * n + REPEAT_SIZE].template ReinterpretCast<T>(),
            mask, base_m / REPEAT_SIZE,
            {2, 0, 2, 16, 0, 16});
      }
    }
  }

  template <typename T, uint32_t FORMAT, bool ROW_MAJOR>
  __aicore__ inline void LoadBasicBlock(LocalTensor<T> yLocal, LocalTensor<T> xLocal,
                                        uint32_t m, uint32_t n,
                                        uint32_t base_m, uint32_t base_n) {
    constexpr static int FRACTAL_SIZE = 16;
    constexpr static int BLOCK_SIZE   = 32;
    if constexpr (FORMAT == AGLU_FRACTAL_zZ && ROW_MAJOR == true) {
      LoadData2dParams loadDataParams;
      loadDataParams.startIndex  = static_cast<uint8_t>(0);
      loadDataParams.repeatTimes = static_cast<uint8_t>(base_n / FRACTAL_SIZE);
      loadDataParams.srcStride   = static_cast<uint16_t>(1);
      loadDataParams.dstGap      = static_cast<uint16_t>(0);
      loadDataParams.sid         = static_cast<uint8_t>(0);
      loadDataParams.ifTranspose = static_cast<bool>(0);
      loadDataParams.addrMode    = static_cast<uint8_t>(0);
      for (uint32_t m_ = 0; m_ < base_m; m_+=FRACTAL_SIZE) {
        LoadData(yLocal[m_ * base_n], xLocal[m_ * n], loadDataParams);
      }
    } else if constexpr (FORMAT == AGLU_FRACTAL_nZ && ROW_MAJOR == false) {
      LoadData2dParams loadDataParams;
      loadDataParams.startIndex  = static_cast<uint8_t>(0);
      loadDataParams.repeatTimes = static_cast<uint8_t>(base_m * base_n / FRACTAL_SIZE / FRACTAL_SIZE);
      loadDataParams.srcStride   = static_cast<uint16_t>(1);
      loadDataParams.dstGap      = static_cast<uint16_t>(0);
      loadDataParams.sid         = static_cast<uint8_t>(0);
      loadDataParams.ifTranspose = static_cast<bool>(0);
      loadDataParams.addrMode    = static_cast<uint8_t>(0);
      LoadData(yLocal, xLocal, loadDataParams);
    }
  }

  template <typename T, uint32_t FORMAT>
  __aicore__ inline void CopyInBasicBlock(LocalTensor<T> xLocal, GlobalTensor<T> xGm,
                                          uint32_t m, uint32_t n,
                                          uint32_t base_m, uint32_t base_n) {
    constexpr static int FRACTAL_SIZE = 16;
    constexpr static int BLOCK_SIZE   = 32;
    DataCopyParams intriParams;
    if constexpr (FORMAT == AGLU_FORMAT_ND) {
      intriParams.blockCount = base_m;
      intriParams.blockLen   = base_n * sizeof(T) / BLOCK_SIZE;
      intriParams.srcStride  = (n - base_n) * sizeof(T) / BLOCK_SIZE;
      intriParams.dstStride  = 0;
    } else if constexpr (FORMAT == AGLU_FRACTAL_nZ) {
      intriParams.blockCount = base_m / FRACTAL_SIZE;
      intriParams.blockLen   = base_n * FRACTAL_SIZE * sizeof(T) / BLOCK_SIZE;
      intriParams.srcStride  = (n - base_n) * FRACTAL_SIZE * sizeof(T) / BLOCK_SIZE;
      intriParams.dstStride  = 0;
    }
    DataCopy(xLocal, xGm, intriParams);
  }

  template <typename T, uint32_t FORMAT>
  __aicore__ inline void CopyOutBasicBlock(GlobalTensor<T> xGm, LocalTensor<T> xLocal,
                                           uint32_t m, uint32_t n,
                                           uint32_t base_m, uint32_t base_n) {
    constexpr static int FRACTAL_SIZE = 16;
    constexpr static int BLOCK_SIZE   = 32;
    DataCopyParams intriParams;
    if constexpr (FORMAT == AGLU_FRACTAL_zZ) {
      intriParams.blockCount = base_m / FRACTAL_SIZE;
      intriParams.blockLen   = base_n * FRACTAL_SIZE * sizeof(T) / BLOCK_SIZE;
      intriParams.srcStride  = 0;
      intriParams.dstStride  = (n - base_n) * FRACTAL_SIZE * sizeof(T) / BLOCK_SIZE;
    }
    if constexpr (FORMAT == AGLU_FORMAT_ND) {
      intriParams.blockCount = base_m;
      intriParams.blockLen   = base_n * sizeof(T) / BLOCK_SIZE;
      intriParams.srcStride  = 0;
      intriParams.dstStride  = (n - base_n) * sizeof(T) / BLOCK_SIZE;
    }
    DataCopy(xGm, xLocal, intriParams);
  }

  template <typename T, uint32_t FORMAT>
  __aicore__ inline void CopyOutBasicBlock(LocalTensor<T> xL1, LocalTensor<T> xLocal,
                                           uint32_t m, uint32_t n,
                                           uint32_t base_m, uint32_t base_n) {
    constexpr static int FRACTAL_SIZE = 16;
    constexpr static int BLOCK_SIZE   = 32;
    DataCopyParams intriParams;
    if constexpr (FORMAT == AGLU_FRACTAL_zZ) {
      intriParams.blockCount = base_m / FRACTAL_SIZE;
      intriParams.blockLen   = base_n * FRACTAL_SIZE * sizeof(T) / BLOCK_SIZE;
      intriParams.srcStride  = 0;
      intriParams.dstStride  = (n - base_n) * FRACTAL_SIZE * sizeof(T) / BLOCK_SIZE;
    }
    DataCopy(xL1, xLocal, intriParams);
  }

  template <typename T> __aicore__ inline void TransDataND2zZ(LocalTensor<T> y, LocalTensor<T> x,
                                                              uint32_t base_m, uint32_t base_n) {
    constexpr static int FRACTAL_SIZE = 16;
    constexpr static int BLOCK_SIZE   = 32;
    constexpr static int REPEAT_SIZE  = 8;
    uint64_t mask[2] = { UINT64_MAX, UINT64_MAX };
    if constexpr (std::is_same<T, half>::value) {
      for (uint32_t m = 0; m < base_m; m+=FRACTAL_SIZE) {
        Muls(y[m * base_n].template ReinterpretCast<T>(),
             x[m * base_n].template ReinterpretCast<T>(),
             static_cast<T>(1.0),
             mask, static_cast<uint8_t>(base_n * sizeof(T) / BLOCK_SIZE),
             {static_cast<uint16_t>(1),
              static_cast<uint16_t>(base_n * sizeof(T) / BLOCK_SIZE),
              static_cast<uint8_t>(FRACTAL_SIZE * FRACTAL_SIZE * sizeof(T) / BLOCK_SIZE),
              static_cast<uint8_t>(1)});
        Muls(y[m * base_n + REPEAT_SIZE * FRACTAL_SIZE].template ReinterpretCast<T>(),
             x[m * base_n + REPEAT_SIZE * base_n].template ReinterpretCast<T>(),
             static_cast<T>(1.0),
             mask, static_cast<uint8_t>(base_n * sizeof(T) / BLOCK_SIZE),
             {static_cast<uint16_t>(1),
              static_cast<uint16_t>(base_n * sizeof(T) / BLOCK_SIZE),
              static_cast<uint8_t>(FRACTAL_SIZE * FRACTAL_SIZE * sizeof(T) / BLOCK_SIZE),
              static_cast<uint8_t>(1)});
      }
    }
  }

  template <typename T> __aicore__ inline void TransDatazN2ND(LocalTensor<T> y, LocalTensor<T> x,
                                                              uint32_t base_m, uint32_t base_n) {
    constexpr static int FRACTAL_SIZE = 16;
    constexpr static int BLOCK_SIZE   = 32;
    constexpr static int REPEAT_SIZE  = 8;
    uint64_t mask[2] = { UINT64_MAX, UINT64_MAX };
    for (uint32_t m = 0; m < base_m; m+=FRACTAL_SIZE) {
      Muls(y[m * base_n].template ReinterpretCast<T>(),
           x[m * FRACTAL_SIZE].template ReinterpretCast<T>(),
           static_cast<T>(1.0),
           mask, static_cast<uint8_t>(base_n * sizeof(T) / BLOCK_SIZE),
           {static_cast<uint16_t>(base_n * sizeof(T) / BLOCK_SIZE),
            static_cast<uint16_t>(1),
            static_cast<uint8_t>(1),
            static_cast<uint8_t>(FRACTAL_SIZE * base_m * sizeof(T) / BLOCK_SIZE)});
      Muls(y[(m + REPEAT_SIZE) * base_n].template ReinterpretCast<T>(),
           x[(m + REPEAT_SIZE) * FRACTAL_SIZE].template ReinterpretCast<T>(),
           static_cast<T>(1.0),
           mask, static_cast<uint8_t>(base_n * sizeof(T) / BLOCK_SIZE),
           {static_cast<uint16_t>(base_n * sizeof(T) / BLOCK_SIZE),
            static_cast<uint16_t>(1),
            static_cast<uint8_t>(1),
            static_cast<uint8_t>(FRACTAL_SIZE * base_m * sizeof(T) / BLOCK_SIZE)});
    }
  }

  template <typename T> __aicore__ inline void LoadNDMatrixAToL1(LocalTensor<T> xL1, GlobalTensor<T> xGm,
                                                                 TQue<QuePosition::VECIN, 2> &vecInQ,
                                                                 TQue<QuePosition::VECOUT, 1> &vecOutQ,
                                                                 uint32_t m, uint32_t n,
                                                                 uint32_t base_m, uint32_t base_n) {
    LocalTensor<T> xLocal = vecInQ.template AllocTensor<T>();
    CopyInBasicBlock<T, AGLU_FORMAT_ND>(xLocal, xGm, m, n, base_m, base_n);
    vecInQ.EnQue(xLocal);
    xLocal = vecInQ.template DeQue<T>();
    LocalTensor<T> zzBuf = vecOutQ.template AllocTensor<T>();
    TransDataND2zZ(zzBuf, xLocal, base_m, base_n);
    vecInQ.FreeTensor(xLocal);
    vecOutQ.EnQue(zzBuf);
    zzBuf = vecOutQ.template DeQue<T>();
    CopyOutBasicBlock<T,AGLU_FRACTAL_zZ>(xL1, zzBuf, m, n, base_m, base_n);
    vecOutQ.FreeTensor(zzBuf);
  }

  template <typename T> __aicore__ inline void LoadNDMatrixAToL1(GlobalTensor<T> yGm, GlobalTensor<T> xGm,
                                                                 TQue<QuePosition::VECIN, 2> &vecInQ,
                                                                 TQue<QuePosition::VECOUT, 1> &vecOutQ,
                                                                 uint32_t m, uint32_t n,
                                                                 uint32_t base_m, uint32_t base_n) {
    LocalTensor<T> xLocal = vecInQ.template AllocTensor<T>();
    CopyInBasicBlock<T, AGLU_FORMAT_ND>(xLocal, xGm, m, n, base_m, base_n);
    vecInQ.EnQue(xLocal);
    xLocal = vecInQ.template DeQue<T>();
    LocalTensor<T> zzBuf = vecOutQ.template AllocTensor<T>();
    TransDataND2zZ(zzBuf, xLocal, base_m, base_n);
    vecInQ.FreeTensor(xLocal);
    vecOutQ.EnQue(zzBuf);
    zzBuf = vecOutQ.template DeQue<T>();
    CopyOutBasicBlock<T,AGLU_FRACTAL_zZ>(yGm, zzBuf, m, n, base_m, base_n);
    vecOutQ.FreeTensor(zzBuf);
  }

private:
  TPipe* pipe;

  // Queue
  TQue<QuePosition::VECIN, 2>                              inQueueTensorA;
  TQue<QuePosition::VECIN, 2>                              inQueueTensorBias;
  TQue<QuePosition::VECOUT, 1>                             outQueueTensor;
  TQueBind<QuePosition::VECOUT, QuePosition::A1, 1>        inQueueTensorAL1;
  TQue<QuePosition::B1, 2>                                 inQueueTensorBL1;
  TQue<QuePosition::A2, 2>                                 inQueueTensorAL0;
  TQue<QuePosition::B2, 2>                                 inQueueTensorBL0;
  TQue<QuePosition::CO1, 2>                                inQueueTensorCL0;
  TQueBind<QuePosition::CO2, QuePosition::VECIN, 2>        inQueueTensorC;

  // Global Tensor
  GlobalTensor<half> xGm;
  GlobalTensor<half> weight1Gm;
  GlobalTensor<half> weight2Gm;
  GlobalTensor<half> bias1Gm;
  GlobalTensor<half> bias2Gm;
  GlobalTensor<half> yGm;

  // tiling data
  uint32_t enableBias;
  uint32_t matmulMode;
  uint32_t activateLeft;
  uint32_t activateFunc;
  uint32_t baseM;
  uint32_t baseK;
  uint32_t baseN;
  uint32_t singleCoreM;
  uint32_t singleCoreK;
  uint32_t singleCoreN;
  uint32_t matrixASize;
  uint32_t matrixAOffset;
  uint32_t matrixBSize;
  uint32_t matrixBOffset;
  uint32_t matrixCSize;
  uint32_t matrixCOffset;
  uint32_t matmulModeCoeff;
};

} // namespace AGLU

#endif  // AGLU_WEIGHT_NZ_L1_FULLLOAD_H