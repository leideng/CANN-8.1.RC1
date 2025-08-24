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
 * \file ge_glu_v2_fp32_align_last_axis_big_without_pad.h
 * \brief
 */
#ifndef GeGluV2_Float32_LAST_BIG_WITHOUT_PAD_H
#define GeGluV2_Float32_LAST_BIG_WITHOUT_PAD_H

#include "ge_glu_v2_base_310p.h"

namespace GeGluV2 {
using namespace AscendC;

template <typename T>
class GeGluV2Fp32AlignLastAxisBigWithoutPad : public GeGluV2Base310P<T> {
 public:
  __aicore__ inline GeGluV2Fp32AlignLastAxisBigWithoutPad(){};
  __aicore__ inline void Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR gelu, GM_ADDR workspace, const GeGluV2TilingData *tilingData);
  __aicore__ inline void Process();

  constexpr static int32_t bufferNum = 2;
  constexpr static int64_t bufferSize = 8192;

 private:
  __aicore__ inline void CopyInX(const int64_t& idxX, const int64_t& idxY,
                                 const int64_t& length, const int64_t& delta);
  __aicore__ inline void ComputeGelu(const int64_t& ub_num, const int64_t& useTanh);
  __aicore__ inline void CopyOutGelu(const int64_t& idxX, const int64_t& idxY,
                                     const int64_t& length, const int64_t& delta);
  __aicore__ inline void ComputeMul(const int64_t& ub_num);
  __aicore__ inline void CopyOutMul(const int64_t& idxX, const int64_t& idxY,
                                    const int64_t& length, const int64_t& delta);
  __aicore__ inline void ProcessPerCore();
  __aicore__ inline void ProcessLastCore();

 private:
  int64_t actualLoopNum_;
  int64_t tailLoopIndex_;
  TPipe pipe;
  TQue<QuePosition::VECIN, bufferNum> inQueueX1;
  TQue<QuePosition::VECIN, bufferNum> inQueueX2;
  TQue<QuePosition::VECOUT, bufferNum> outQueueGelu;
  TQue<QuePosition::VECOUT, bufferNum> outQueueMul;

  TBuf<QuePosition::VECCALC> resultTempBuf1;
  TBuf<QuePosition::VECCALC> resultTempBuf2;
};

template <typename T>
__aicore__ inline void GeGluV2Fp32AlignLastAxisBigWithoutPad<T>::Init(
  GM_ADDR x, GM_ADDR y, GM_ADDR gelu, GM_ADDR workspace, const GeGluV2TilingData *tilingData) {
  this->BaseInit(x, y, gelu, tilingData, true);
  pipe.InitBuffer(inQueueX1, bufferNum, bufferSize * sizeof(T));
  pipe.InitBuffer(inQueueX2, bufferNum, bufferSize * sizeof(T));
  pipe.InitBuffer(outQueueGelu, 1, bufferSize * sizeof(T));
  pipe.InitBuffer(outQueueMul, 1, bufferSize * sizeof(T));

  pipe.InitBuffer(resultTempBuf1, bufferSize * sizeof(float));
  pipe.InitBuffer(resultTempBuf2, bufferSize * sizeof(float));

  actualLoopNum_ =
        (this->m_tilingData.loopNum + this->m_tilingData.realCoreNum - 1) / this->m_tilingData.realCoreNum;
  tailLoopIndex_ = this->m_tilingData.loopNum % this->m_tilingData.realCoreNum;
}

template <typename T>
__aicore__ inline void GeGluV2Fp32AlignLastAxisBigWithoutPad<T>::Process() {
  if (this->blockIdx >= this->m_tilingData.realCoreNum) {
    return;
  }

  ProcessPerCore();
}

template <typename T>
__aicore__ inline void GeGluV2Fp32AlignLastAxisBigWithoutPad<T>::ProcessPerCore() {
  // process core
  for (int64_t idx = 0; idx < actualLoopNum_; idx++) {
    if (tailLoopIndex_ !=0 && idx == actualLoopNum_ - 1 && this->blockIdx >= tailLoopIndex_) {
      return;
    }
    int64_t z = this->m_tilingData.realCoreNum * idx + this->blockIdx;
    int64_t idxX = 0;
    int64_t idxY = 0;
    int64_t length = 0;
    int64_t alignLength = 0;
    int64_t delta = 0;

    if (this->m_tilingData.tailLoopNum != 0) {
      idxX = z / (this->m_tilingData.group + 1);
      idxY = z % (this->m_tilingData.group + 1);
      length = idxY == this->m_tilingData.group ? this->m_tilingData.tailLoopNum : this->m_tilingData.splitSize;
      alignLength = (length + this->m_tilingData.blockSize - 1) /
                    this->m_tilingData.blockSize * this->m_tilingData.blockSize;
      delta = alignLength - length;
    } else {
      idxX = z / this->m_tilingData.group;
      idxY = z % this->m_tilingData.group;
      length = this->m_tilingData.splitSize;
      alignLength = length;
    }

    CopyInX(idxX, idxY, alignLength / this->m_tilingData.blockSize, delta);
    ComputeGelu(alignLength, this->m_tilingData.approximate);
    CopyOutGelu(idxX, idxY, alignLength / this->m_tilingData.blockSize, delta);
    ComputeMul(alignLength);
    CopyOutMul(idxX, idxY, alignLength / this->m_tilingData.blockSize, delta);
  }
}

template <typename T>
__aicore__ inline void GeGluV2Fp32AlignLastAxisBigWithoutPad <T>::CopyInX(
  const int64_t& idxX, const int64_t& idxY, const int64_t& length, const int64_t& delta) {
  LocalTensor<T> ubX1 = inQueueX1.AllocTensor<T>();
  LocalTensor<T> ubX2 = inQueueX2.AllocTensor<T>();
  this->CopyInXAlignLastBigWithoutPad(idxX, idxY, length, ubX1, ubX2, delta);

  inQueueX1.EnQue(ubX1);
  inQueueX2.EnQue(ubX2);
}

template <typename T>
__aicore__ inline void GeGluV2Fp32AlignLastAxisBigWithoutPad<T>::ComputeGelu(const int64_t& ub_num,
                                                                             const int64_t& useTanh) {
  LocalTensor<T> ubx2_fp32 = inQueueX2.DeQue<T>();
  // after cast to fp32 , input buffer release, to use as tmp buffer wihle do geluv2 compute.
  LocalTensor<T> gelu_out = outQueueGelu.AllocTensor<T>();
  if (useTanh) {
    this->ComputeGeluBase(ubx2_fp32, gelu_out, ub_num);
  } else {
    LocalTensor<float> x1 = resultTempBuf1.Get<float>();
    LocalTensor<float> x_pow = resultTempBuf2.Get<float>();
    this->ComputeGeluErf(ubx2_fp32, gelu_out, x1, x_pow, ub_num);
  }

  inQueueX2.FreeTensor(ubx2_fp32);
  outQueueGelu.EnQue(gelu_out);
}

template <typename T>
__aicore__ inline void GeGluV2Fp32AlignLastAxisBigWithoutPad<T>::ComputeMul(const int64_t& ub_num) {
  LocalTensor<T> ubX1 = inQueueX1.DeQue<T>();
  LocalTensor<T> gelu_out = outQueueGelu.DeQue<T>();
  LocalTensor<T> mul_out = outQueueMul.AllocTensor<T>();
  pipe_barrier(PIPE_V);
  Mul(mul_out, gelu_out, ubX1, ub_num);
  outQueueMul.EnQue(mul_out);

  outQueueGelu.FreeTensor(gelu_out);
  inQueueX1.FreeTensor(ubX1);
}

template <typename T>
__aicore__ inline void GeGluV2Fp32AlignLastAxisBigWithoutPad<T>::CopyOutGelu(
    const int64_t& idxX, const int64_t& idxY, const int64_t& length, const int64_t& delta) {
  LocalTensor<T> outLocalGelu = outQueueGelu.DeQue<T>();
  this->CopyOutGeluBaseLastBigWithoutPad(idxX, idxY, length, outLocalGelu, delta);
  outQueueGelu.EnQue(outLocalGelu);
}

template <typename T>
__aicore__ inline void GeGluV2Fp32AlignLastAxisBigWithoutPad<T>::CopyOutMul(
    const int64_t& idxX, const int64_t& idxY, const int64_t& length, const int64_t& delta) {
  LocalTensor<T> outLocalMul = outQueueMul.DeQue<T>();
  this->CopyOutMulBaseLastBigWithoutPad(idxX, idxY, length, outLocalMul, delta);
  outQueueMul.FreeTensor(outLocalMul);
}
}  // namespace GeGluV2
#endif  // GeGluV2_Float32_LAST_BIG_WITHOUT_PAD_H