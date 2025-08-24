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
 * \file swi_glu_bf16.hpp
 * \brief
 */
#ifndef OPP_SWI_GLU_BF16_HPP
#define OPP_SWI_GLU_BF16_HPP
#include "kernel_operator.h"

using namespace AscendC;
template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
class SwigluVectorBF16 {
public:
    __aicore__ inline SwigluVectorBF16() {}
    __aicore__ inline ~SwigluVectorBF16() {}

protected:
    __aicore__ inline void InitUbBuffer(uint64_t tileLength);
    __aicore__ inline void Compute(uint64_t curTileLen);

    float beta = 1.0;
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> inQueueA;
    TQue<QuePosition::VECIN, bufferNum> inQueueB;
    TQue<QuePosition::VECOUT, bufferNum> outQueueC;

    TBuf<TPosition::VECCALC> inputTempBuffer;
    TBuf<TPosition::VECCALC> outputTempBuffer; // a/b复用 // todo tiling中BUFFER的大小刷新

    GlobalTensor<inType> aGm;
    GlobalTensor<inType> bGm;
    GlobalTensor<outType> cGm;
};

  template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
  __aicore__ inline void SwigluVectorBF16<inType, calcType, outType, bufferNum>::InitUbBuffer(uint64_t tileLength) {
      // pipe alloc memory to queue, the unit is Bytes
      pipe.InitBuffer(inQueueA, bufferNum, tileLength * sizeof(inType));
      pipe.InitBuffer(inQueueB, bufferNum, tileLength * sizeof(inType));
      pipe.InitBuffer(outQueueC, bufferNum, tileLength * sizeof(outType));

      pipe.InitBuffer(inputTempBuffer, tileLength * sizeof(calcType));
      pipe.InitBuffer(outputTempBuffer, tileLength * sizeof(calcType));
  }

template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwigluVectorBF16<inType, calcType, outType, bufferNum>::Compute(uint64_t curTileLen)
{
    LocalTensor<inType> aLocal_ = inQueueA.template DeQue<inType>();
    LocalTensor<outType> cLocal_ = outQueueC.template AllocTensor<outType>();

    LocalTensor<calcType> aLocal = inputTempBuffer.Get<calcType>();
    LocalTensor<calcType> cLocal = outputTempBuffer.Get<calcType>();
    Cast(aLocal, aLocal_, RoundMode::CAST_NONE, curTileLen);
    pipe_barrier(PIPE_V);
    inQueueA.template FreeTensor(aLocal_);

    Muls(cLocal, aLocal, beta, curTileLen);
    pipe_barrier(PIPE_V);
    Exp(cLocal, cLocal, curTileLen);
    pipe_barrier(PIPE_V);
    Adds(cLocal, cLocal, calcType(1.0), curTileLen);
    pipe_barrier(PIPE_V);

    Div(cLocal, aLocal, cLocal, curTileLen);
    pipe_barrier(PIPE_V);

    LocalTensor<inType> bLocal_ = inQueueB.template DeQue<inType>();

    LocalTensor<calcType> bLocal = inputTempBuffer.Get<calcType>();
    Cast(bLocal, bLocal_, RoundMode::CAST_NONE, curTileLen);
    pipe_barrier(PIPE_V);
    inQueueB.template FreeTensor(bLocal_);

    Mul(cLocal, cLocal, bLocal, curTileLen);
    pipe_barrier(PIPE_V);

    Cast(cLocal_, cLocal, RoundMode::CAST_RINT, curTileLen);
    pipe_barrier(PIPE_V);
    // enque the output tensor to VECOUT queue
    outQueueC.template EnQue<outType>(cLocal_);
    // free input tensors for reuse
}
#endif  // OPP_SWI_GLU_BF16_HPP
