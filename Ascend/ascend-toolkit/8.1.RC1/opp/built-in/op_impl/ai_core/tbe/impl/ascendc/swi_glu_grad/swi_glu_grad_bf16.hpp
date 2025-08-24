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
 * \file swi_glu_grad_bf16.hpp
 * \brief
 */
#ifndef OPP_SWI_GLU_GRAD_BF16_HPP
#define OPP_SWI_GLU_GRAD_BF16_HPP
#include "kernel_operator.h"

using namespace AscendC;
template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
class SwiGluGradBF16 {
  public:
    __aicore__ inline SwiGluGradBF16() {}

  protected:
    __aicore__ inline void InitUbBuffer(uint64_t tileLength);
    __aicore__ inline void Compute(uint64_t curTileLen);
    __aicore__ inline void ComputeSigLocal(uint64_t curTileLen);
    __aicore__ inline void ComputeGradN(uint64_t curTileLen);

    calcType beta = 1.0;
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> inQueueA;
    TQue<QuePosition::VECIN, bufferNum> inQueueB;
    TQue<QuePosition::VECIN, bufferNum> inQueueL;
    TQue<QuePosition::VECOUT, bufferNum> outQueueM;
    TQue<QuePosition::VECOUT, bufferNum> outQueueN;
    TBuf<TPosition::VECCALC> tmpQueue;
    TBuf<TPosition::VECCALC> sigQueue;
    LocalTensor<calcType> tempLocal;
    LocalTensor<calcType> sigLocal;
    LocalTensor<calcType> aLocal;
    LocalTensor<calcType> nLocal;
    LocalTensor<calcType> lLocal;

    TBuf<TPosition::VECCALC> aTempBuffer;
    TBuf<TPosition::VECCALC> lTempBuffer;
    TBuf<TPosition::VECCALC> outputTempBuffer;

    GlobalTensor<inType> aGm;
    GlobalTensor<inType> bGm;
    GlobalTensor<inType> lGm;
    GlobalTensor<outType> mGm;
    GlobalTensor<outType> nGm;
};

template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwiGluGradBF16<inType, calcType, outType, bufferNum>::InitUbBuffer(uint64_t tileLength)
{
    // pipe alloc memory to queue, the unit is Bytes
    pipe.InitBuffer(inQueueA, bufferNum, tileLength * sizeof(inType));
    pipe.InitBuffer(inQueueB, bufferNum, tileLength * sizeof(inType));
    pipe.InitBuffer(inQueueL, bufferNum, tileLength * sizeof(inType));
    pipe.InitBuffer(outQueueM, bufferNum, tileLength * sizeof(outType)); // The length must be an integer multiple of 32
    pipe.InitBuffer(outQueueN, bufferNum, tileLength * sizeof(outType)); // The length must be an integer multiple of 32

    pipe.InitBuffer(tmpQueue, tileLength * sizeof(calcType));
    pipe.InitBuffer(sigQueue, tileLength * sizeof(calcType));

    pipe.InitBuffer(aTempBuffer, tileLength * sizeof(calcType));
    pipe.InitBuffer(lTempBuffer, tileLength * sizeof(calcType));
    pipe.InitBuffer(outputTempBuffer, tileLength * sizeof(calcType));
    tempLocal = tmpQueue.Get<calcType>();
    sigLocal = sigQueue.Get<calcType>();
    aLocal = aTempBuffer.Get<calcType>(); 
    nLocal = outputTempBuffer.Get<calcType>(); 
    lLocal = lTempBuffer.Get<calcType>();    
}

template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwiGluGradBF16<inType, calcType, outType, bufferNum>::Compute(uint64_t tileLength)
{
    // tbuf::templocaltensor
    // deque input tensors from VECIN queue
    //calc sigLocal
    ComputeSigLocal(tileLength);
    //----------------N
    Mul(nLocal, sigLocal, aLocal, tileLength);
    ComputeGradN(tileLength);

    //----------------M
    Muls(tempLocal, sigLocal, (calcType)(-1.0), tileLength);
    pipe_barrier(PIPE_V);
    Adds(tempLocal, tempLocal, (calcType)(1.0), tileLength);
    pipe_barrier(PIPE_V);

    auto& mLocal = nLocal;                      
    Mul(mLocal, sigLocal, tempLocal, tileLength); 
    pipe_barrier(PIPE_V);
    Mul(mLocal, mLocal, aLocal, tileLength);  
    pipe_barrier(PIPE_V);

    Muls(mLocal, mLocal, -beta, tileLength);
    pipe_barrier(PIPE_V);
    Add(mLocal, mLocal, sigLocal, tileLength);

    LocalTensor<inType> bLocal_ = inQueueB.template DeQue<inType>();
    auto& bLocal = aLocal; 
    Cast(bLocal, bLocal_, RoundMode::CAST_NONE, tileLength);
    pipe_barrier(PIPE_V);

    Mul(mLocal, mLocal, lLocal, tileLength);
    pipe_barrier(PIPE_V);

    LocalTensor<outType> mLocal_ = outQueueM.template AllocTensor<outType>();

    Mul(mLocal, mLocal, bLocal, tileLength);
    inQueueB.template FreeTensor(bLocal_);

    Cast(mLocal_, mLocal, RoundMode::CAST_RINT, tileLength);
    pipe_barrier(PIPE_V);
    outQueueM.template EnQue<outType>(mLocal_);
}

template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwiGluGradBF16<inType, calcType, outType, bufferNum>::ComputeSigLocal(uint64_t tileLength)
{
    // tbuf::templocaltensor
    // deque input tensors from VECIN queue
    //calc sigLocal
    LocalTensor<inType> aLocal_ = inQueueA.template DeQue<inType>(); //input a

    Cast(aLocal, aLocal_, RoundMode::CAST_NONE, tileLength);
    pipe_barrier(PIPE_V);
    inQueueA.template FreeTensor(aLocal_);

    Muls(sigLocal, aLocal, beta, tileLength);
    pipe_barrier(PIPE_V);
    Exp(sigLocal, sigLocal, tileLength);
    pipe_barrier(PIPE_V);
    Adds(sigLocal, sigLocal, (calcType)(1.0), tileLength);
    pipe_barrier(PIPE_V);
    Duplicate<calcType>(tempLocal, (calcType)(1.0), tileLength);
    pipe_barrier(PIPE_V);
    Div(sigLocal, tempLocal, sigLocal, tileLength);
    pipe_barrier(PIPE_V);
}

template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwiGluGradBF16<inType, calcType, outType, bufferNum>::ComputeGradN(uint64_t tileLength)
{
    LocalTensor<inType> lLocal_ = inQueueL.template DeQue<inType>(); // input l
    Cast(lLocal, lLocal_, RoundMode::CAST_NONE, tileLength);
    pipe_barrier(PIPE_V);
    inQueueL.template FreeTensor(lLocal_);

    LocalTensor<outType> nLocal_ = outQueueN.template AllocTensor<outType>(); // lb

    Mul(nLocal, nLocal, lLocal, tileLength);
    pipe_barrier(PIPE_V);

    Cast(nLocal_, nLocal, RoundMode::CAST_RINT, tileLength); // todo nLocal最后使用位置
    pipe_barrier(PIPE_V);
    outQueueN.template EnQue<outType>(nLocal_);
}
#endif // OPP_SWI_GLU_GRAD_BF16_HPP
