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

#ifndef GeGluV2_FLOAT_VREDUCE_ERF_H
#define GeGluV2_FLOAT_VREDUCE_ERF_H

#include "../ge_glu_v2_base.h"


namespace GeGluV2 {
using namespace AscendC;

template <typename T>
class GeGluV2Fp32VReduceErf : public GeGluV2Base<T> {
 public:
  __aicore__ inline GeGluV2Fp32VReduceErf(){};
  __aicore__ inline void Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR gelu, GM_ADDR workspace, const GeGluV2TilingData *tilingData);
  __aicore__ inline void Process();

  constexpr static int32_t bufferNum = 2;
  constexpr static int64_t bufferSize = 4912;

 private:
  __aicore__ inline void CopyInX(const int64_t& index, const int64_t& blockCount);
  __aicore__ inline void ComputeGeluAndMul(const int64_t& ub_num);
  __aicore__ inline void CopyOutGelu(const int64_t& index, const int64_t& ub_num, const int64_t& group);
  __aicore__ inline void CopyOutMul(const int64_t& index, const int64_t& ub_num, const int64_t& group);
  __aicore__ inline void ProcessPerCore();
  __aicore__ inline void ProcessLastCore();

 private:
  TPipe pipe;
  TQue<QuePosition::VECIN, bufferNum> inQueueX;
  TQue<QuePosition::VECOUT, bufferNum> outQueueGelu;
  TQue<QuePosition::VECOUT, bufferNum> outQueueMul;

  TBuf<QuePosition::VECCALC> resultTempBuf1;
  TBuf<QuePosition::VECCALC> resultTempBuf2;
  TBuf<QuePosition::VECCALC> resultTempBuf3;
  TBuf<QuePosition::VECCALC> resultTempBuf4;
};

template <typename T>
__aicore__ inline void GeGluV2Fp32VReduceErf<T>::Init(
  GM_ADDR x, GM_ADDR y, GM_ADDR gelu, GM_ADDR workspace, const GeGluV2TilingData *tilingData) {
  this->BaseInit(x, y, gelu, tilingData, false, true);
  pipe.InitBuffer(inQueueX, bufferNum, 2 * bufferSize * sizeof(T));
  pipe.InitBuffer(outQueueGelu, 1, bufferSize * sizeof(T));
  pipe.InitBuffer(outQueueMul, 1, bufferSize * sizeof(T));

  pipe.InitBuffer(resultTempBuf1, bufferSize * sizeof(T));
  pipe.InitBuffer(resultTempBuf2, bufferSize * sizeof(T));
  pipe.InitBuffer(resultTempBuf3, bufferSize * sizeof(T));
  pipe.InitBuffer(resultTempBuf4, bufferSize * sizeof(T));
}

template <typename T>
__aicore__ inline void GeGluV2Fp32VReduceErf<T>::Process() {
  if (this->blockIdx >= this->m_tilingData.realCoreNum) {
    return;
  }

  if (this->isLastCore) {  // process last core
    ProcessLastCore();
  } else {
    ProcessPerCore();
  }
}

template <typename T>
__aicore__ inline void GeGluV2Fp32VReduceErf<T>::ProcessPerCore() {
  // process core
  for (int64_t idx = 0; idx < this->m_tilingData.loopNum; idx++) {
    CopyInX(idx, this->m_tilingData.group);
    ComputeGeluAndMul(this->group_ub_num);
    CopyOutGelu(idx, this->group_ub_num, this->m_tilingData.group);
    CopyOutMul(idx, this->group_ub_num, this->m_tilingData.group);
  }

  if (this->m_tilingData.nLastTailGroup > 0) {
    CopyInX(this->m_tilingData.loopNum, this->m_tilingData.nLastTailGroup);
    ComputeGeluAndMul(this->nlast_tail_ub_num);
    CopyOutGelu(this->m_tilingData.loopNum, this->nlast_tail_ub_num, this->m_tilingData.nLastTailGroup);
    CopyOutMul(this->m_tilingData.loopNum, this->nlast_tail_ub_num, this->m_tilingData.nLastTailGroup);
  }
}

template <typename T>
__aicore__ inline void GeGluV2Fp32VReduceErf<T>::ProcessLastCore() {
  for (int64_t idx = 0; idx < this->m_tilingData.tailLoopNum; idx++) {
    CopyInX(idx, this->m_tilingData.group);
    ComputeGeluAndMul(this->group_ub_num);
    CopyOutGelu(idx, this->group_ub_num, this->m_tilingData.group);
    CopyOutMul(idx, this->group_ub_num, this->m_tilingData.group);
  }
  if (this->m_tilingData.lastTailGroup > 0) {
    CopyInX(this->m_tilingData.tailLoopNum, this->m_tilingData.lastTailGroup);
    ComputeGeluAndMul(this->last_tail_ub_num);
    CopyOutGelu(this->m_tilingData.tailLoopNum, this->last_tail_ub_num, this->m_tilingData.lastTailGroup);
    CopyOutMul(this->m_tilingData.tailLoopNum, this->last_tail_ub_num, this->m_tilingData.lastTailGroup);
  }
}

template <typename T>
__aicore__ inline void GeGluV2Fp32VReduceErf<T>::CopyInX(const int64_t& index, const int64_t& blockCount) {
  LocalTensor<T> ubX = inQueueX.AllocTensor<T>();
  this->CopyInXVreduce(index, blockCount, ubX);
  inQueueX.EnQue(ubX);
}

template <typename T>
__aicore__ inline void GeGluV2Fp32VReduceErf<T>::ComputeGeluAndMul(const int64_t& ub_num) {
  LocalTensor<T> ubX = inQueueX.DeQue<T>();
  LocalTensor<T> ubx1_fp32 = resultTempBuf3.Get<T>();
  LocalTensor<T> ubx2_fp32 = resultTempBuf4.Get<T>();

  this->ComputeFp32VReduce(ubx1_fp32, ubx2_fp32, ubX, ub_num);
  inQueueX.FreeTensor(ubX);

  // after cast to fp32 , input buffer release, to use as tmp buffer wihle do geluv2 compute.
  LocalTensor<T> tmpBuf1 = resultTempBuf1.Get<T>();
  LocalTensor<T> tmpBuf2 = resultTempBuf2.Get<T>();
  LocalTensor<T> gelu_out = outQueueGelu.AllocTensor<T>();
  this->ComputeGeluErf(ubx2_fp32, gelu_out, tmpBuf1, tmpBuf2, ub_num);

  LocalTensor<T> mul_out = outQueueMul.AllocTensor<T>();
  Mul(mul_out, ubx1_fp32, gelu_out, ub_num);

  outQueueMul.EnQue(mul_out);
  outQueueGelu.EnQue(gelu_out);
}

template <typename T>
__aicore__ inline void GeGluV2Fp32VReduceErf<T>::CopyOutGelu(
    const int64_t& index, const int64_t& ub_num, const int64_t& group) {
  LocalTensor<T> outLocalGelu = outQueueGelu.DeQue<T>();
  this->CopyOutGeluVreduce(index, ub_num, outLocalGelu);
  outQueueGelu.FreeTensor(outLocalGelu);
}

template <typename T>
__aicore__ inline void GeGluV2Fp32VReduceErf<T>::CopyOutMul(
    const int64_t& index, const int64_t& ub_num, const int64_t& group) {
  LocalTensor<T> outLocalMul = outQueueMul.DeQue<T>();
  this->CopyOutMulVreduce(index, ub_num, outLocalMul);
  outQueueMul.FreeTensor(outLocalMul);
}
}  // namespace GeGluV2
#endif  // GeGluV2_FLOAT_VREDUCE_ERF_H