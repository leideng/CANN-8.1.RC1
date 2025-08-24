/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
 * \file matmul_layer_norm_reduce_perf.h
 * \brief
 */

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "kernel_event.h"
#include "common.h"
namespace MatmulLayerNormReduceND {
using namespace AscendC;
template <typename T, int32_t bufferNum>
class MatmulLayerNormReducePerf {
 public:
  __aicore__ inline MatmulLayerNormReducePerf(){};
  __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR add, GM_ADDR div, GM_ADDR y, GM_ADDR sum,
                              GM_ADDR squareSum, GM_ADDR workspace, const MatmulLayerNormReduceTilingData* tilingData) {
    tiling = tilingData;
    blockIdx = GetBlockIdx();
    M = tiling->M;
    N = tiling->N;
    K = tiling->K;
    singleCoreM = tiling->singleCoreM;
    singleCoreN = tiling->singleCoreN;
    singleCoreK = tiling->singleCoreK;
    baseM = tiling->baseM;
    baseN = tiling->baseN;
    baseK = tiling->baseK;
    x1Gm.SetGlobalBuffer((__gm__ T*)x1, M * K);
    x2Gm.SetGlobalBuffer((__gm__ T*)x2, N * K);
    biasGm.SetGlobalBuffer((__gm__ T*)bias, N);
    if (div != nullptr) {
      hasDiv = true;
      divGm.SetGlobalBuffer((__gm__ T*)div, M * N);
    }
    if (add != nullptr) {
      hasAdd = true;
      addGm.SetGlobalBuffer((__gm__ T*)add, M * N);
    }
    yGm.SetGlobalBuffer((__gm__ T*)y, M * N);
    sumGm.SetGlobalBuffer((__gm__ float*)sum, M);
    squareSumGm.SetGlobalBuffer((__gm__ float*)squareSum, M);

    pipe.InitBuffer(x1L1Que, 1, singleCoreM * singleCoreK * sizeof(T));
    pipe.InitBuffer(x2L1Que, bufferNum, baseN * baseK * sizeof(T));
    pipe.InitBuffer(x1L0AQue, bufferNum, baseM * baseK * sizeof(T));
    pipe.InitBuffer(x2L0BQue, bufferNum, baseN * baseK * sizeof(T));
    pipe.InitBuffer(yL0CQue, bufferNum, baseM * baseN * sizeof(float));
    pipe.InitBuffer(ubTotalBuf, UB_SIZE);
  }

  __aicore__ inline void Process() {
    ConstructTensor();
    uint32_t x1GmStartOffset = blockIdx * baseM * singleCoreK;
    uint32_t biasGmStartOffset = 0;
    uint32_t divGmStartOffset = 0;
    uint32_t addGmStartOffset = blockIdx * baseM * N;
    uint32_t yGmOffset = blockIdx * baseM * N;
    uint32_t sumGmOffset = blockIdx * baseM;
    uint32_t squareSumOffset = blockIdx * baseM;

    LocalTensor<T> x1L1Buf = x1L1Que.template AllocTensor<T>();
    if (hasDiv) {
      event_t eventIdMTE2ToS = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>());
      DataCopyParams param;
      param.blockCount = 1;
      param.blockLen = 1;
      param.srcStride = 0;
      param.dstStride = 0;
      DataCopy(divInBuf, divGm[divGmStartOffset], param);
      SetFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
      WaitFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
      divScalar = static_cast<T>((float)1.0 / float(divInBuf.GetValue(0)));
      GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_S>(eventIdMTE2ToS);
    }

    float val = 0.0;
    Duplicate(zeroInBuf, val, baseM);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(sumGm[sumGmOffset], zeroInBuf, {1, static_cast<uint16_t>(baseM * sizeof(float) / BLOCK_SIZE), 0, 0});
    DataCopy(squareSumGm[squareSumOffset], zeroInBuf,
             {1, static_cast<uint16_t>(baseM * sizeof(float) / BLOCK_SIZE), 0, 0});

    event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    event_t eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t eventIdMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    LocalTensor<T> addInTensor = yInBuf[baseM * baseN];

    for (int i = 0; i < N / baseN; i++) {
      LocalTensor<float> yL0CBuf = yL0CQue.template AllocTensor<float>();
      LocalTensor<T> bL1Ping = x2L1Que.template AllocTensor<T>();
      LocalTensor<T> bL1Pong = x2L1Que.template AllocTensor<T>();
      LocalTensor<T> aL0Ping = x1L0AQue.template AllocTensor<T>();
      LocalTensor<T> aL0Pong = x1L0AQue.template AllocTensor<T>();
      LocalTensor<T> bL0Ping = x2L0BQue.template AllocTensor<T>();
      LocalTensor<T> bL0Pong = x2L0BQue.template AllocTensor<T>();
      for (int k = 0; k < K; k += baseK) {
        uint32_t idx = (k / baseK);
        bool isPing = (idx % 2 == 0);
        bool isPong = !isPing;
        bool isFirstPing = (idx == 0);
        bool isFirstPong = (idx == 1);
        bool isLastPing = false;
        bool isLastPong = false;
        if (k + baseK + baseK >= singleCoreK && isPing) {
          isLastPing = true;
        }
        if (k + baseK + baseK >= singleCoreK && isPong) {
          isLastPong = true;
        }
        LocalTensor<T> x2L1Buf = isPing ? bL1Ping : bL1Pong;
        LocalTensor<T> x2L0BBuf = isPing ? bL0Ping : bL0Pong;
        LocalTensor<T> x1L0ABuf = isPing ? aL0Ping : aL0Pong;
        LocalTensor<T> x1InBuf = isPing ? x1InPingBuf : x1InPongBuf;
        LocalTensor<T> x1OutBuf = isPing ? x1OutPingBuf : x1OutPongBuf;
        if (i == 0) {
          if (!isFirstPing && !isFirstPong) {
            WaitFlag<HardEvent::MTE3_MTE2>(isPing ? EVENT_ID0 : EVENT_ID1);
          }
          CopyAMatrixBasicBlockGm2L1(x1L1Buf, x1InBuf, x1OutBuf, x1GmStartOffset + k, k * baseM,
                                     isPing ? EVENT_ID0 : EVENT_ID1, isPing ? EVENT_ID0 : EVENT_ID1);
          if (!isLastPing && !isLastPong) {
            SetFlag<HardEvent::MTE3_MTE2>(isPing ? EVENT_ID0 : EVENT_ID1);
          }
          SetFlag<HardEvent::MTE3_MTE1>(isPing ? EVENT_ID0 : EVENT_ID1);
          WaitFlag<HardEvent::MTE3_MTE1>(isPing ? EVENT_ID0 : EVENT_ID1);
        }

        if (!isFirstPing && !isFirstPong) {
          WaitFlag<HardEvent::M_MTE1>(isPing ? EVENT_ID0 : EVENT_ID1);
        }

        CopyAMatrixBasicBlockL12L0A(x1L0ABuf, x1L1Buf, k * baseM, baseM, baseK);
        if (!isFirstPing && !isFirstPong) {
          WaitFlag<HardEvent::MTE1_MTE2>(isPing ? EVENT_ID0 : EVENT_ID1);
        }
        CopyBMatrixBasicBlockGm2L1(x2L1Buf, i * baseN * FRACTAL_M + k * N);
        SetFlag<HardEvent::MTE2_MTE1>(isPing ? EVENT_ID0 : EVENT_ID1);
        WaitFlag<HardEvent::MTE2_MTE1>(isPing ? EVENT_ID0 : EVENT_ID1);
        CopyBMatrixL12L0B(x2L0BBuf, x2L1Buf, 0, baseN, baseK);
        if (!isLastPing && !isLastPong) {
          SetFlag<HardEvent::MTE1_MTE2>(isPing ? EVENT_ID0 : EVENT_ID1);
        }
        SetFlag<HardEvent::MTE1_M>(isPing ? EVENT_ID0 : EVENT_ID1);
        WaitFlag<HardEvent::MTE1_M>(isPing ? EVENT_ID0 : EVENT_ID1);

        CalMatrixInL0C(yL0CBuf, x1L0ABuf, x2L0BBuf, k);
        if (!isLastPing && !isLastPong) {
          SetFlag<HardEvent::M_MTE1>(isPing ? EVENT_ID0 : EVENT_ID1);
        }
      }
      x2L1Que.FreeTensor(bL1Ping);
      x2L1Que.FreeTensor(bL1Pong);
      x1L0AQue.FreeTensor(aL0Ping);
      x1L0AQue.FreeTensor(aL0Pong);
      x2L0BQue.FreeTensor(bL0Ping);
      x2L0BQue.FreeTensor(bL0Pong);
      if (i > 0) {
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
      }
      CopyBias2UB(biasInBuf, biasGmStartOffset + i * baseN);
      yL0CQue.EnQue(yL0CBuf);
      yL0CBuf = yL0CQue.template DeQue<float>();
      pipe_barrier(PIPE_V);
      CopyCMatrixNZ2UB(yInBuf, yL0CBuf);

      yL0CQue.FreeTensor(yL0CBuf);
      SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
      WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
      AddBiasToCMatrixWithNz2Nd(yOutBuf, yInBuf, biasInBuf);

      if (hasAdd) {
        DataCopyParams param;
        param.blockCount = baseM;
        param.blockLen = baseN * sizeof(T) / BLOCK_SIZE;
        param.srcStride = (N - baseN) * sizeof(T) / BLOCK_SIZE;
        param.dstStride = 0;
        DataCopy(addInTensor, addGm[addGmStartOffset + i * baseN], param);
      }

      if (hasDiv) {
        Muls(yOutBuf, yOutBuf, divScalar, baseM * baseN);
      }

      if (hasAdd) {
        SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
        Add(yOutBuf, yOutBuf, addInTensor, baseM * baseN);
      }
      SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
      WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

      CopyCMatrixToGm(yOutBuf, yGmOffset + i * baseN);

      LocalTensor<float> floatDst = yInBuf.template ReinterpretCast<float>();
      Cast(floatDst, yOutBuf, RoundMode::CAST_NONE, baseM * baseN);
      CalculateWholeReduceSum(sumOutBuf, floatDst);

      SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
      WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
      SetAtomicAdd<float>();
      DataCopy(sumGm[sumGmOffset], sumOutBuf, {1, static_cast<uint16_t>(baseM * sizeof(float) / BLOCK_SIZE), 0, 0});
      SetAtomicNone();

      Cast(floatDst, yOutBuf, RoundMode::CAST_NONE, baseM * baseN);
      Mul(floatDst, floatDst, floatDst, baseM * baseN);

      CalculateWholeReduceSum(squareSumOutBuf, floatDst);
      SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
      WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
      SetAtomicAdd<float>();
      DataCopy(squareSumGm[squareSumOffset], squareSumOutBuf,
               {1, static_cast<uint16_t>(baseM * sizeof(float) / BLOCK_SIZE), 0, 0});
      SetAtomicNone();
      if (i != N / baseN - 1) {
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
      }
    }
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMTE2ToV);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMTE3);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMTE2);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);

    x1L1Que.FreeTensor(x1L1Buf);
  }

  __aicore__ inline void CopyCMatrixNZ2UB(LocalTensor<T>& dst, LocalTensor<float>& src) {
    DataCopyParams param;
    param.blockCount = 1;
    param.blockLen = baseN / FRACTAL_N * baseM / FRACTAL_M;
    DataCopyEnhancedParams enhancedParams;
    enhancedParams.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(dst, src, param, enhancedParams);
  }

  __aicore__ inline void CopyBias2UB(LocalTensor<T>& dst, int biasGmOffset) {
    DataCopyParams param;
    param.blockCount = 1;
    param.blockLen = baseN * sizeof(T) / BLOCK_SIZE;
    param.srcStride = 0;
    param.dstStride = 0;
    DataCopy(dst, biasGm[biasGmOffset], param);
  }

  __aicore__ inline void CopyCMatrixToGm(LocalTensor<T> src, int gmOffset) {
    DataCopyParams param;
    param.blockCount = baseM;
    param.blockLen = baseN * sizeof(T) / BLOCK_SIZE;
    param.srcStride = 0;
    param.dstStride = (N - baseN) * sizeof(T) / BLOCK_SIZE;
    DataCopy(yGm[gmOffset], src, param);
  }

  __aicore__ inline void AddBiasToCMatrixWithNz2Nd(LocalTensor<T>& dst, LocalTensor<T>& src, LocalTensor<T>& bias) {
    uint32_t outerFor = baseN / FRACTAL_N;
    uint32_t repeatNum = baseM / REPEAT_MAX_BLOCK_NUM;
    BinaryRepeatParams param;
    param.dstBlkStride = baseN * sizeof(T) / BLOCK_SIZE;
    param.src0BlkStride = 1;
    param.src1BlkStride = 0;
    param.dstRepStride = REPEAT_MAX_BLOCK_NUM * baseN * sizeof(T) / BLOCK_SIZE;
    param.src0RepStride = REPEAT_MAX_BLOCK_NUM;
    param.src1RepStride = 0;
    set_vector_mask(0xffffffffffffffff, 0xffffffffffffffff);
    for (uint32_t i = 0; i < outerFor; i++) {
      vadd((__ubuf__ T*)dst[i * BLOCK_SIZE / sizeof(T)].GetPhyAddr(),
           (__ubuf__ T*)src[i * baseM * FRACTAL_M].GetPhyAddr(),
           (__ubuf__ T*)bias[i * BLOCK_SIZE / sizeof(T)].GetPhyAddr(), repeatNum, baseN * sizeof(T) / BLOCK_SIZE, 1, 0,
           REPEAT_MAX_BLOCK_NUM * baseN * sizeof(T) / BLOCK_SIZE, REPEAT_MAX_BLOCK_NUM, 0);
    }
  }

  __aicore__ inline void CalculateWholeReduceSum(LocalTensor<float>& dst, LocalTensor<float> src) {
    int32_t valCnt = REPEAT_MAX_BLOCK_NUM * BLOCK_SIZE / sizeof(float);
    int32_t outerFor = (baseN - valCnt) / valCnt;
    int32_t repeat = baseM;
    BinaryRepeatParams param;
    param.dstBlkStride = 1;
    param.src0BlkStride = 1;
    param.src1BlkStride = 1;
    param.dstRepStride = baseN * sizeof(float) / BLOCK_SIZE;
    param.src0RepStride = baseN * sizeof(float) / BLOCK_SIZE;
    param.src1RepStride = baseN * sizeof(float) / BLOCK_SIZE;
    for (int32_t i = 0; i < outerFor; i++) {
      uint32_t offset = (i + 1) * valCnt;
      Add(src, src, src[offset], valCnt, repeat, param);
    }

    int32_t validNum = valCnt;
    if (baseN < validNum) {
      validNum = baseN;
    }
    WholeReduceSum(dst, src, valCnt, repeat, 1, 1, baseN * sizeof(float) / BLOCK_SIZE);
  }

  __aicore__ inline void CopyBMatrixL12L0B(LocalTensor<T>& dst, LocalTensor<T>& src, uint32_t startOffset, uint32_t row,
                                           uint32_t col) {
    uint32_t repeat = (row * col) / FRACTAL_N / FRACTAL_M;
    LoadData2dParams param;
    param.startIndex = 0;
    param.repeatTimes = repeat;
    param.srcStride = 1;
    param.sid = 0;
    LoadData(dst, src[startOffset], param);
  }

  __aicore__ inline void CalMatrixInL0C(LocalTensor<float>& dst, LocalTensor<T>& src0, LocalTensor<T>& src1,
                                        uint32_t index) {
    MmadParams param;
    param.m = baseM;
    param.n = baseN;
    param.k = baseK;
    param.isBias = 0;
    param.cmatrixInitVal = (index == 0);
    Mmad(dst, src0, src1, param);
  }

  // 一次只拷贝一个基本块
  __aicore__ inline void CopyBMatrixBasicBlockGm2L1(LocalTensor<T>& x2L1Buf, uint32_t gmOffset) {
    DataCopyParams param;
    param.blockCount = baseK / FRACTAL_M;
    param.blockLen = baseN * FRACTAL_M * sizeof(T) / BLOCK_SIZE;
    param.srcStride = (N - baseN) * FRACTAL_M * sizeof(T) / BLOCK_SIZE;
    param.dstStride = 0;
    DataCopy(x2L1Buf, x2Gm[gmOffset], param);
  }

  __aicore__ inline void CopyAMatrixBasicBlockGm2L1(LocalTensor<T>& x1L1Buf, LocalTensor<T>& x1InBuf,
                                                    LocalTensor<T>& x1OutBuf, uint32_t startOffset,
                                                    uint32_t outputOffset, uint32_t eventIdMTE2V,
                                                    uint32_t eventIdV2MTE3) {
    DataCopyParams mte2Param;
    mte2Param.blockCount = baseM;
    mte2Param.blockLen = static_cast<uint16_t>(baseK * sizeof(T) / BLOCK_SIZE);
    mte2Param.srcStride = (singleCoreK - baseK) * sizeof(T) / BLOCK_SIZE;
    mte2Param.dstStride = 0;
    DataCopy(x1InBuf, x1Gm[startOffset], mte2Param);
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2V);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2V);
    ConvertMatrixNDTozZ(x1OutBuf, x1InBuf, baseM, baseK);
    SetFlag<HardEvent::V_MTE3>(eventIdV2MTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdV2MTE3);
    DataCopyParams mte3Param;
    mte3Param.blockCount = 1;
    mte3Param.blockLen = static_cast<uint16_t>(baseM * baseK * sizeof(T) / BLOCK_SIZE);
    mte3Param.srcStride = 0;
    mte3Param.dstStride = 0;
    DataCopy(x1L1Buf[outputOffset], x1OutBuf, mte3Param);
  }

  __aicore__ inline void CopyAMatrixBasicBlockL12L0A(LocalTensor<T>& dst, LocalTensor<T>& src, uint32_t startOffset,
                                                     uint32_t row, uint32_t col) {
    uint32_t repeat = (row * col) / FRACTAL_N / FRACTAL_M;
    LoadData2dParams param;
    param.startIndex = 0;
    param.repeatTimes = repeat;
    param.srcStride = 1;
    param.sid = 0;
    LoadData(dst, src[startOffset], param);
  }

  __aicore__ inline void ConvertMatrixNDTozZ(LocalTensor<T>& dst, const LocalTensor<T>& src, uint32_t row,
                                             uint32_t col) {
    uint32_t outerFor = row / FRACTAL_M;
    uint32_t repeatNum = col / FRACTAL_N;
    uint32_t throwFor = FRACTAL_M / REPEAT_MAX_BLOCK_NUM;
    T scalar = 1.0;
    set_vector_mask(0xffffffffffffffff, 0xffffffffffffffff);
    for (int32_t i = 0; i < outerFor; i++) {
      for (int j = 0; j < throwFor; j++) {
        vmuls((__ubuf__ T*)dst[i * FRACTAL_M * col + j * REPEAT_MAX_BLOCK_NUM * FRACTAL_M].GetPhyAddr(),
              (__ubuf__ T*)src[i * FRACTAL_M * col + j * REPEAT_MAX_BLOCK_NUM * col].GetPhyAddr(), scalar,
              static_cast<uint8_t>(repeatNum), static_cast<uint16_t>(1),
              static_cast<uint16_t>(col * sizeof(T) / BLOCK_SIZE), static_cast<uint8_t>(FRACTAL_M),
              static_cast<uint8_t>(1));
      }
    }
  }

  __aicore__ inline void ConstructTensor() {
    uint32_t basicBlock = baseM * baseN;
    uint32_t yInBufOffset = 0;
    uint32_t yOutBufOffset = basicBlock * 2;
    uint32_t biasInBufOffset = yOutBufOffset + basicBlock;
    uint32_t divInBufOffset = biasInBufOffset + baseN;
    uint32_t zeroInBufOffset = divInBufOffset + BLOCK_SIZE / sizeof(T);
    uint32_t sumOutBufOffset = zeroInBufOffset + baseM * 2;
    uint32_t squareSumOutBufOffset = sumOutBufOffset + baseM * 2;

    LocalTensor<T> ubBuf = ubTotalBuf.template Get<T>();
    x1InPingBuf = ubBuf[basicBlock * IN_PING_INDEX];
    x1InPongBuf = ubBuf[basicBlock * IN_PONG_INDEX];
    x1OutPingBuf = ubBuf[basicBlock * OUT_PING_INDEX];
    x1OutPongBuf = ubBuf[basicBlock * OUT_PONG_INDEX];

    yInBuf = ubBuf[yInBufOffset];
    yOutBuf = ubBuf[yOutBufOffset];
    biasInBuf = ubBuf[biasInBufOffset];
    divInBuf = ubBuf[divInBufOffset];
    zeroInBuf = ubBuf[zeroInBufOffset].template ReinterpretCast<float>();
    sumOutBuf = ubBuf[sumOutBufOffset].template ReinterpretCast<float>();
    squareSumOutBuf = ubBuf[squareSumOutBufOffset].template ReinterpretCast<float>();
  }

 private:
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t singleCoreM;
  uint32_t singleCoreN;
  uint32_t singleCoreK;
  uint32_t baseM;
  uint32_t baseN;
  uint32_t baseK;
  uint32_t blockIdx;

  bool hasAdd{false};
  bool hasDiv{false};

  GlobalTensor<T> x1Gm;
  GlobalTensor<T> x2Gm;
  GlobalTensor<T> biasGm;
  GlobalTensor<T> addGm;
  GlobalTensor<T> divGm;
  GlobalTensor<T> yGm;
  GlobalTensor<float> sumGm;
  GlobalTensor<float> squareSumGm;

  const MatmulLayerNormReduceTilingData* tiling;
  TPipe pipe;
  LocalTensor<T> x1InPingBuf;
  LocalTensor<T> x1InPongBuf;
  LocalTensor<T> x1OutPingBuf;
  LocalTensor<T> x1OutPongBuf;

  LocalTensor<T> yInBuf;
  LocalTensor<T> yOutBuf;
  LocalTensor<T> biasInBuf;
  LocalTensor<T> divInBuf;
  LocalTensor<float> zeroInBuf;
  LocalTensor<float> sumOutBuf;
  LocalTensor<float> squareSumOutBuf;

  TBuf<> ubTotalBuf;
  T divScalar = 1.0;
  TQueBind<QuePosition::VECOUT, QuePosition::A1, 1> x1L1Que;
  TQue<QuePosition::B1, bufferNum> x2L1Que;
  TQue<QuePosition::A2, bufferNum> x1L0AQue;
  TQue<QuePosition::B2, bufferNum> x2L0BQue;
  TQue<QuePosition::CO1, bufferNum> yL0CQue;
};
}  // namespace MatmulLayerNormReduceND