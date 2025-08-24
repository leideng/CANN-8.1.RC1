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

#ifndef GeGluV2_BFLOAT16_LAST_BIG_H
#define GeGluV2_BFLOAT16_LAST_BIG_H

#include "../ge_glu_v2_base.h"

namespace GeGluV2 {
using namespace AscendC;

template <typename T>
class GeGluV2Bf16AlignLastAxisBig : public GeGluV2Base<T> {
 public:
  __aicore__ inline GeGluV2Bf16AlignLastAxisBig(){};
  __aicore__ inline void Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR gelu, GM_ADDR workspace, const GeGluV2TilingData *tilingData);
  __aicore__ inline void Process();

  constexpr static int32_t bufferNum = 2;
  constexpr static int64_t bufferSize = 6144;

 private:
  __aicore__ inline void CopyInX(const int64_t& idx_x, const int64_t& idx_y, const int64_t& length);
  __aicore__ inline void ComputeGeluAndMul(const int64_t& ub_num);
  __aicore__ inline void CopyOutGelu(const int64_t& idx_x, const int64_t& idx_y, const int64_t& length);
  __aicore__ inline void CopyOutMul(const int64_t& idx_x, const int64_t& idx_y, const int64_t& length);
  __aicore__ inline void ProcessPerCore();
  __aicore__ inline void ProcessLastCore();

 private:
  TPipe pipe;
  TQue<QuePosition::VECIN, bufferNum> inQueueX1;
  TQue<QuePosition::VECIN, bufferNum> inQueueX2;
  TQue<QuePosition::VECOUT, bufferNum> outQueueGelu;
  TQue<QuePosition::VECOUT, bufferNum> outQueueMul;

  TBuf<QuePosition::VECCALC> resultTempBuf1;
  TBuf<QuePosition::VECCALC> resultTempBuf2;
};

template <typename T>
__aicore__ inline void GeGluV2Bf16AlignLastAxisBig<T>::Init(
  GM_ADDR x, GM_ADDR y, GM_ADDR gelu, GM_ADDR workspace, const GeGluV2TilingData *tilingData) {
  this->BaseInit(x, y, gelu, tilingData, true);
  pipe.InitBuffer(inQueueX1, bufferNum, bufferSize * sizeof(T));
  pipe.InitBuffer(inQueueX2, bufferNum, bufferSize * sizeof(T));
  pipe.InitBuffer(outQueueGelu, 1, bufferSize * sizeof(T));
  pipe.InitBuffer(outQueueMul, 1, bufferSize * sizeof(T));

  pipe.InitBuffer(resultTempBuf1, bufferSize * sizeof(float));
  pipe.InitBuffer(resultTempBuf2, bufferSize * sizeof(float));
}

template <typename T>
__aicore__ inline void GeGluV2Bf16AlignLastAxisBig<T>::Process() {
  if (this->blockIdx >= this->m_tilingData.realCoreNum) {
    return;
  }

  ProcessPerCore();
}

template <typename T>
__aicore__ inline void GeGluV2Bf16AlignLastAxisBig<T>::ProcessPerCore() {
  // process core
  int64_t actualLoopNum =
    (this->m_tilingData.loopNum + this->m_tilingData.realCoreNum - 1) / this->m_tilingData.realCoreNum;
  int64_t tailLoopIdx = this->m_tilingData.loopNum % this->m_tilingData.realCoreNum;

  for (int64_t idx = 0; idx < actualLoopNum; idx++) {
    if (tailLoopIdx !=0 && idx == actualLoopNum - 1) {
      if (this->blockIdx >= tailLoopIdx) {
        return;
      }
    }
    int64_t z = this->m_tilingData.realCoreNum * idx + this->blockIdx;
    int64_t idx_x = 0;
    int64_t idx_y = 0;
    int64_t length = 0;
    if (this->m_tilingData.tailLoopNum != 0) {
      idx_x = z / (this->m_tilingData.group + 1);
      idx_y = z % (this->m_tilingData.group + 1);
      length = idx_y == this->m_tilingData.group ? this->m_tilingData.tailLoopNum : this->m_tilingData.splitSize;
    } else {
      idx_x = z / this->m_tilingData.group;
      idx_y = z % this->m_tilingData.group;
      length = this->m_tilingData.splitSize;
    }

    CopyInX(idx_x, idx_y, length);
    ComputeGeluAndMul(length);
    CopyOutGelu(idx_x, idx_y, length);
    CopyOutMul(idx_x, idx_y, length);
  }
}

template <typename T>
__aicore__ inline void GeGluV2Bf16AlignLastAxisBig<T>::CopyInX(
  const int64_t& idx_x, const int64_t& idx_y, const int64_t& length) {
  LocalTensor<T> ubX1 = inQueueX1.AllocTensor<T>();
  LocalTensor<T> ubX2 = inQueueX2.AllocTensor<T>();
  this->CopyInXAlignLastBig(idx_x, idx_y, length, ubX1, ubX2);

  inQueueX1.EnQue(ubX1);
  inQueueX2.EnQue(ubX2);
}

template <typename T>
__aicore__ inline void GeGluV2Bf16AlignLastAxisBig<T>::ComputeGeluAndMul(const int64_t& ub_num) {
  LocalTensor<T> ubX2 = inQueueX2.DeQue<T>();
  LocalTensor<float> ubx_fp32 = resultTempBuf2.Get<float>();
  Cast(ubx_fp32, ubX2, RoundMode::CAST_NONE, ub_num);
  inQueueX2.FreeTensor(ubX2);

  // after cast to fp32 , input buffer release, to use as tmp buffer wihle do geluv2 compute.
  LocalTensor<float> gelu_result = resultTempBuf1.Get<float>();
  this->ComputeGeluBase(ubx_fp32, gelu_result, ub_num);

  LocalTensor<T> ubX1 = inQueueX1.DeQue<T>();
  LocalTensor<T> mul_out = outQueueMul.AllocTensor<T>();
  Cast(ubx_fp32, ubX1, RoundMode::CAST_NONE, ub_num);
  inQueueX1.FreeTensor(ubX1);

  Mul(ubx_fp32, ubx_fp32, gelu_result, ub_num);
  Cast(mul_out, ubx_fp32, RoundMode::CAST_RINT, ub_num);
  outQueueMul.EnQue(mul_out);

  LocalTensor<T> gelu_out = outQueueGelu.AllocTensor<T>();
  Cast(gelu_out, gelu_result, RoundMode::CAST_RINT, ub_num);
  outQueueGelu.EnQue(gelu_out);
}

template <typename T>
__aicore__ inline void GeGluV2Bf16AlignLastAxisBig<T>::CopyOutGelu(
  const int64_t& idx_x, const int64_t& idx_y, const int64_t& length) {
  LocalTensor<T> outLocalGelu = outQueueGelu.DeQue<T>();
  this->CopyOutGeluBaseLastBig(idx_x, idx_y, length, outLocalGelu);
  outQueueGelu.FreeTensor(outLocalGelu);
}

template <typename T>
__aicore__ inline void GeGluV2Bf16AlignLastAxisBig<T>::CopyOutMul(
  const int64_t& idx_x, const int64_t& idx_y, const int64_t& length) {
  LocalTensor<T> outLocalMul = outQueueMul.DeQue<T>();
  this->CopyOutMulBaseLastBig(idx_x, idx_y, length, outLocalMul);
  outQueueMul.FreeTensor(outLocalMul);
}
}  // namespace GeGluV2
#endif  // GeGluV2_BFLOAT16_LAST_BIG_H