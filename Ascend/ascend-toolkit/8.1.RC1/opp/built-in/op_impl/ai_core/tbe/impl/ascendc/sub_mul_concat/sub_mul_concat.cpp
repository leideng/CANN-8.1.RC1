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
 * \file sub_mul_concat.cpp
 * \brief
 */

#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t ALIGN_32 = 32;
constexpr uint32_t BYTE_32 = 32;
constexpr uint32_t CONCAT_INPUTS_NUM = 4;
constexpr uint32_t OP_INPUTS_NUM = 2;
constexpr uint32_t UB_SIZE = 184 * 1024;

class KernelSubMulConcat {
public:
    __aicore__ inline KernelSubMulConcat() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, SubMulConcatTilingData tiling) {
        uint32_t blockIndex = GetBlockIdx();
        // formerLength、tailLength均已按tileW对齐
        if (blockIndex < tiling.formerNum) {
            this->blockLength = tiling.formerLength;
        } else {
            this->blockLength = tiling.tailLength;
        }
        if (this->blockLength <= 0) {
            this->initSucc = false;
            return;
        }

        uint32_t tileAlien = ALIGN_32 / sizeof(float);
        this->concatLen = tiling.tileW;
        this->concatLenPad = (this->concatLen + tileAlien - 1) / tileAlien * tileAlien;

        if (this->concatLen == this->concatLenPad) {
            uint32_t maxTileLenth = UB_SIZE / BUFFER_NUM / (CONCAT_INPUTS_NUM * 2 + OP_INPUTS_NUM) / sizeof(float) / this->concatLen * this->concatLen; // UB size/2/10/4/tileW*tileW
            if (this->blockLength >= maxTileLenth) {
                this->tileLength = maxTileLenth;
            } else {
                this->tileLength = this->blockLength;
            }
            this->bufferSize =this->tileLength * sizeof(float);
        } else {
            uint32_t maxTileLenth = UB_SIZE / BUFFER_NUM / (CONCAT_INPUTS_NUM * 2 + OP_INPUTS_NUM) / sizeof(float) / this->concatLenPad * this->concatLen;
            uint32_t maxTileLenthPad = UB_SIZE / BUFFER_NUM / (CONCAT_INPUTS_NUM * 2 + OP_INPUTS_NUM) / sizeof(float) / this->concatLenPad * this->concatLenPad;
            if (this->blockLength >= maxTileLenth) {
                this->tileLength = maxTileLenth;
                this->bufferSize = maxTileLenthPad * sizeof(float);
            } else {
                this->tileLength = this->blockLength;
                this->bufferSize = this->tileLength / this->concatLen * this->concatLenPad * sizeof(float);
            }
        }

        this->tileNum = this->blockLength / this->tileLength;
        this->tailLen = this->blockLength - this->tileNum * this->tileLength;

        uint32_t globalOffsetInput = blockIndex < tiling.formerNum ? this->blockLength * blockIndex
            : tiling.formerNum * tiling.formerLength + (blockIndex - tiling.formerNum) * tiling.tailLength;
        uint32_t globalOffsetOutput = globalOffsetInput * CONCAT_INPUTS_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x + globalOffsetInput, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + globalOffsetInput, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ float *)z + globalOffsetOutput, this->blockLength * CONCAT_INPUTS_NUM);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->bufferSize);
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->bufferSize);
        pipe.InitBuffer(outQueue, BUFFER_NUM * 2, this->bufferSize * CONCAT_INPUTS_NUM);  // 2 is because transpose in UB
        this->initSucc = true;
    }

    __aicore__ inline void Process() {
        if (this->concatLen == this->concatLenPad) {
            for (int32_t i = 0; i < this->tileNum; i++) {
                CopyIn(i);
                Compute(i);
                CopyOut(i);
            }

            // process tail data
            if (this->tailLen > 0) {
                CopyIn(this->tileNum);
                Compute(this->tileNum);
                CopyOut(this->tileNum);
            }
        } else {
            for (int32_t i = 0; i < this->tileNum; i++) {
                CopyInPad(i);
                ComputePad(i);
                CopyOutPad(i);
            }
            // process tail data
            if (this->tailLen > 0) {
                CopyInPad(this->tileNum);
                ComputePad(this->tileNum);
                CopyOutPad(this->tileNum);
            }
        }
    }

private:

    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
        uint32_t copyLen = this->tileLength;
        uint32_t gmOffset = this->tileLength;

        if(progress == this->tileNum) {
            copyLen = this->tailLen;
        }

        DataCopy(xLocal, xGm[progress * gmOffset], copyLen);
        DataCopy(yLocal, yGm[progress * gmOffset], copyLen);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

__aicore__ inline void CopyInPad(int32_t progress) {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
        uint32_t copyLen = this->tileLength;
        uint32_t gmOffset = this->tileLength;

        if(progress == this->tileNum) {
            copyLen = this->tailLen;
        }

        uint16_t blockCount = static_cast<uint16_t>(copyLen / this->concatLen);
        DataCopyExtParams copyParams {blockCount, this->concatLen * (uint32_t)sizeof(float), 0, 0, 0};
        DataCopyPadExtParams<float> padParams {true, 0, static_cast<uint8_t>(this->concatLenPad - this->concatLen), 0};
        DataCopyPad(xLocal, xGm[progress * gmOffset], copyParams, padParams);
        DataCopyPad(yLocal, yGm[progress * gmOffset], copyParams, padParams);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

__aicore__ inline void Compute(int32_t progress) {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = inQueueY.DeQue<float>();

        uint32_t computeLen = this->tileLength;

        if (progress == this->tileNum) {
            computeLen = this->tailLen;
        }

        int32_t localOffset0 = 0 * computeLen;
        int32_t localOffset1 = 1 * computeLen;
        int32_t localOffset2 = 2 * computeLen;
        int32_t localOffset3 = 3 * computeLen;
        int32_t zlocalTransOffset0 = 0 * this->concatLen;
        int32_t zlocalTransOffset1 = 1 * this->concatLen;
        int32_t zlocalTransOffset2 = 2 * this->concatLen;
        int32_t zlocalTransOffset3 = 3 * this->concatLen;

        LocalTensor<float> zLocal = outQueue.AllocTensor<float>();
        LocalTensor<float> zLocalA = zLocal[localOffset0];
        LocalTensor<float> zLocalB = zLocal[localOffset1];
        LocalTensor<float> zLocalSub = zLocal[localOffset2];
        LocalTensor<float> zLocalMul = zLocal[localOffset3];

        LocalTensor<float> zLocalTrans = outQueue.AllocTensor<float>();

        DataCopy(zLocalA, xLocal, computeLen);
        DataCopy(zLocalB, yLocal, computeLen);
        Sub(zLocalSub, xLocal, yLocal, computeLen);
        Mul(zLocalMul, xLocal, yLocal, computeLen);

        uint16_t blockCount = static_cast<uint16_t>(computeLen / this->concatLen);
        uint16_t dataCopyBlockLen = static_cast<uint16_t>(this->concatLen * sizeof(float) / BYTE_32);
        uint16_t dataCopyDstStride = static_cast<uint16_t>(((CONCAT_INPUTS_NUM - 1) * this->concatLen * sizeof(float)) / BYTE_32);

        DataCopyParams dataCopyParams {blockCount, dataCopyBlockLen, 0, dataCopyDstStride};
        DataCopy(zLocalTrans[zlocalTransOffset0], zLocalA, dataCopyParams);
        DataCopy(zLocalTrans[zlocalTransOffset1], zLocalB, dataCopyParams);
        DataCopy(zLocalTrans[zlocalTransOffset2], zLocalSub, dataCopyParams);
        DataCopy(zLocalTrans[zlocalTransOffset3], zLocalMul, dataCopyParams);

        outQueue.EnQue<float>(zLocalTrans);
        outQueue.FreeTensor(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
}

__aicore__ inline void ComputePad(int32_t progress) {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = inQueueY.DeQue<float>();

        uint32_t computeLen = this->tileLength;
        if(progress == this->tileNum) {
            computeLen = this->tailLen;
        }

        uint32_t computeLenPad = computeLen / this->concatLen * this->concatLenPad;

        int32_t localOffset0 = 0 * computeLenPad;
        int32_t localOffset1 = 1 * computeLenPad;
        int32_t localOffset2 = 2 * computeLenPad;
        int32_t localOffset3 = 3 * computeLenPad;

        LocalTensor<float> zLocal = outQueue.AllocTensor<float>();
        LocalTensor<float> zLocalA = zLocal[localOffset0];
        LocalTensor<float> zLocalB = zLocal[localOffset1];
        LocalTensor<float> zLocalSub = zLocal[localOffset2];
        LocalTensor<float> zLocalMul = zLocal[localOffset3];

        DataCopy(zLocalA, xLocal, computeLenPad);
        DataCopy(zLocalB, yLocal, computeLenPad);
        Sub(zLocalSub, xLocal, yLocal, computeLenPad);
        Mul(zLocalMul, xLocal, yLocal, computeLenPad);

        outQueue.EnQue<float>(zLocal);

        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
}

__aicore__ inline void CopyOut(int32_t progress) {
    uint32_t copyLen = this->tileLength;
    uint32_t gmOffset = this->tileLength;

    if (progress == this->tileNum) {
        copyLen = this->tailLen;
    }

    LocalTensor<float> zLocalTrans = outQueue.DeQue<float>();
    DataCopy(zGm[progress * gmOffset * CONCAT_INPUTS_NUM], zLocalTrans, copyLen * CONCAT_INPUTS_NUM);

    outQueue.FreeTensor(zLocalTrans);
}

__aicore__ inline void CopyOutPad(int32_t progress) {
    uint32_t copyLen = this->tileLength;
    uint32_t gmOffset = this->tileLength;
    if(progress == this->tileNum) {
            copyLen = this->tailLen;
    }
    uint32_t copyLenPad = copyLen / this->concatLen * this->concatLenPad;

    int32_t localOffset0 = 0 * copyLenPad;
    int32_t localOffset1 = 1 * copyLenPad;
    int32_t localOffset2 = 2 * copyLenPad;
    int32_t localOffset3 = 3 * copyLenPad;

    int32_t zGmOffset0 = 0 * this->concatLen;
    int32_t zGmOffset1 = 1 * this->concatLen;
    int32_t zGmOffset2 = 2 * this->concatLen;
    int32_t zGmOffset3 = 3 * this->concatLen;

    LocalTensor<float> zLocal = outQueue.DeQue<float>();
    LocalTensor<float> zLocalA = zLocal[localOffset0];
    LocalTensor<float> zLocalB = zLocal[localOffset1];
    LocalTensor<float> zLocalSub = zLocal[localOffset2];
    LocalTensor<float> zLocalMul = zLocal[localOffset3];

    uint16_t blockCount = static_cast<uint16_t>(copyLen / this->concatLen);
    DataCopyExtParams copyParams {blockCount, this->concatLen * (uint32_t)sizeof(float), 0, (uint32_t)((CONCAT_INPUTS_NUM - 1) * this->concatLen * sizeof(float)), 0};
    DataCopyPad(zGm[progress * gmOffset * CONCAT_INPUTS_NUM + zGmOffset0], zLocalA, copyParams);
    DataCopyPad(zGm[progress * gmOffset * CONCAT_INPUTS_NUM + zGmOffset1], zLocalB, copyParams);
    DataCopyPad(zGm[progress * gmOffset * CONCAT_INPUTS_NUM + zGmOffset2], zLocalSub, copyParams);
    DataCopyPad(zGm[progress * gmOffset * CONCAT_INPUTS_NUM + zGmOffset3], zLocalMul, copyParams);

    outQueue.FreeTensor(zLocal);
}

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX,inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM * 2> outQueue; // 2 is because transpose in UB
    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    GlobalTensor<float> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t concatLen;
    uint32_t concatLenPad;
    uint32_t tailLen;
    uint32_t bufferSize;
    bool initSucc;
};

extern "C" __global__ __aicore__ void sub_mul_concat(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSubMulConcat op;
    op.Init(x, y, z, tiling_data);
    op.Process();
}