/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * \file vector_scheduler.h
 * \brief
 */
#ifndef ASCENDC_OPERATOR_VECTOR_SCHEDULER_H
#define ASCENDC_OPERATOR_VECTOR_SCHEDULER_H

#include "kernel_operator.h"
using namespace AscendC;

constexpr size_t UB_SIZE_BYTE = 248 * 1024;
constexpr size_t ALIGN_SIZE_BYTES = 32;
constexpr size_t BLOCK_SIZE_BYTES = 32;

__aicore__ inline size_t UpAlignN(size_t n, size_t N)
{
    if (N == 0) {
        return 0;
    }
    return (n + N - 1) / N * N;
}

__aicore__ inline size_t DownAlignN(size_t n, size_t N)
{
    if (N == 0) {
        return 0;
    }
    return n / N * N;
}

__aicore__ inline size_t UpAlign32(size_t n)
{
    return UpAlignN(n, ALIGN_SIZE_BYTES);
}

__aicore__ inline size_t DownAlign32(size_t n)
{
    return DownAlignN(n, ALIGN_SIZE_BYTES);
}

class VectorComputer {
public:
    __aicore__ inline VectorComputer() {};

    __aicore__ inline void CalcForAlign32(uint32_t idx, size_t len) {};
};

class VectorScheduler {
public:
    __aicore__ inline VectorScheduler(size_t contentLen, size_t blockDim, size_t bufferNum, float ubVarCount,
                                      size_t sizeofT)
        : blockDim(blockDim), bufferNum(bufferNum), ubVarCount(ubVarCount), sizeofT(sizeofT)
    {
        auto blockIdx = GetBlockIdx();
        this->dataLenPer32B = BLOCK_SIZE_BYTES / this->sizeofT;
        // L1
        this->dataLenPerCore = contentLen / this->blockDim;
        if (this->dataLenPerCore < this->dataLenPer32B) {
            this->dataLenPerCore = blockIdx == 0 ? contentLen : 0;
            this->dataLenTailL1 = 0;
        } else {
            this->dataLenTailL1 = contentLen % this->blockDim;
        }
        // L2
        int maxUbSizePerVar = UB_SIZE_BYTE / ubVarCount / this->bufferNum;
        this->dataBytesPerLoop = DownAlign32(maxUbSizePerVar);
        this->dataLenPerLoop = this->dataBytesPerLoop / this->sizeofT;

        this->dataLen = this->dataLenPerCore;
        if (blockIdx == this->blockDim - 1) {
            this->dataLen += this->dataLenTailL1;
        }
        this->bufferBytesPerVar = this->dataLen > this->dataLenPerLoop ? this->dataBytesPerLoop : UpAlign32(
            this->dataLen * this->sizeofT);
    }

    template<class Computer>
    __aicore__ inline void run(Computer *computer, size_t len)
    {
        if (len <= 0) {
            return;
        }
        size_t loops = len / this->dataLenPerLoop;
        size_t tailLen = len % this->dataLenPerLoop;
        size_t tailLenA32 = DownAlignN(tailLen, this->dataLenPer32B);
        size_t tailLenBackoff = tailLen - tailLenA32;

        uint32_t idx = 0;
        for (size_t i = 0; i < loops; i++) {
            computer->CalcForAlign32(idx, this->dataLenPerLoop);
            idx = idx + this->dataLenPerLoop;
        }
        if (tailLenA32) {
            idx = loops * this->dataLenPerLoop;
            computer->CalcForAlign32(idx, tailLenA32);
        }
        if (tailLenBackoff > 0) {
            idx = len >= this->dataLenPer32B ? len - this->dataLenPer32B : 0;
            computer->CalcForAlign32(idx, this->dataLenPer32B);
        }
    }

public:
    float ubVarCount;
    size_t blockDim;
    size_t bufferNum;
    size_t sizeofT;
    size_t dataLenPer32B;
    // L1
    size_t dataLen;
    size_t dataLenPerCore;
    size_t dataLenTailL1;
    size_t bufferBytesPerVar;
    // L2
    size_t dataLenPerLoop;
    size_t dataBytesPerLoop;
    size_t loopL2;
    size_t dataLenTailL2;
};

#endif // ASCENDC_OPERATOR_VECTOR_SCHEDULER_H