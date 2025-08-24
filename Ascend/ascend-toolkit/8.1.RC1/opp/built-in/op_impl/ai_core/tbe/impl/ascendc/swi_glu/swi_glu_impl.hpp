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
 * \file swi_glu_impl.hpp
 * \brief
 */
#ifndef OPP_SWI_GLU_IMPL_HPP
#define OPP_SWI_GLU_IMPL_HPP
#include "kernel_operator.h"

using namespace AscendC;
template<typename inType, typename outType, uint16_t bufferNum>
class SwigluVector {
  public:
    __aicore__ inline SwigluVector() {}
    __aicore__ inline ~SwigluVector() {}
  protected:
    __aicore__ inline void InitUbBuffer(uint64_t tileLength);
    __aicore__ inline void Compute(uint64_t curTileLen);
    float beta = 1.0;
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> inQueueA;
    TQue<QuePosition::VECIN, bufferNum> inQueueB;
    TQue<QuePosition::VECOUT, bufferNum> outQueueC;
    TBuf<TPosition::VECCALC> tmpQueue;

    GlobalTensor<inType> aGm;
    GlobalTensor<inType> bGm;
    GlobalTensor<outType> cGm;
};

template<typename inType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwigluVector<inType, outType, bufferNum>::InitUbBuffer(uint64_t tileLength) {
    // pipe alloc memory to queue, the unit is Bytes
    pipe.InitBuffer(inQueueA, bufferNum, tileLength * sizeof(inType));
    pipe.InitBuffer(inQueueB, bufferNum, tileLength * sizeof(inType));
    pipe.InitBuffer(outQueueC, bufferNum, tileLength * sizeof(outType));
    pipe.InitBuffer(tmpQueue, tileLength * sizeof(float));
    LocalTensor<float> tempLocal = tmpQueue.Get<float>();
    Duplicate<float>(tempLocal, (float)(1.0), tileLength);
}

template<typename inType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwigluVector<inType, outType, bufferNum>::Compute(uint64_t curTileLen)
{
    LocalTensor<inType> aLocal = inQueueA.template DeQue<inType>();
    LocalTensor<outType> cLocal = outQueueC.template AllocTensor<outType>();
    pipe_barrier(PIPE_V);
    Muls(cLocal, aLocal, beta, curTileLen);
    pipe_barrier(PIPE_V);
    Exp(cLocal, cLocal, curTileLen);
    pipe_barrier(PIPE_V);
    Adds(cLocal, cLocal, (outType)(1.0), curTileLen);
    pipe_barrier(PIPE_V);

    LocalTensor<float> tempLocal = tmpQueue.Get<float>();
    pipe_barrier(PIPE_V);
    Div(cLocal, tempLocal, cLocal, curTileLen);
    pipe_barrier(PIPE_V);
    Mul(cLocal, cLocal, aLocal, curTileLen);
    pipe_barrier(PIPE_V);
    inQueueA.template FreeTensor(aLocal);

    LocalTensor<inType> bLocal = inQueueB.template DeQue<inType>();
    pipe_barrier(PIPE_V);
    Mul(cLocal, cLocal, bLocal, curTileLen);
    pipe_barrier(PIPE_V);
    inQueueB.template FreeTensor(bLocal);
    // enque the output tensor to VECOUT queue
    outQueueC.template EnQue<outType>(cLocal);
    // free input tensors for reuse
}
#endif // OPP_SWI_GLU_IMPL_HPP
