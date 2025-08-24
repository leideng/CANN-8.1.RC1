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
 * \file fake_quant_affine_cachemask_fp32.h
 * \brief
 */
#ifndef _FAKE_QUANT_AFFINE_CACHEMASK_FP32_H_
#define _FAKE_QUANT_AFFINE_CACHEMASK_FP32_H_
#include "fake_quant_affine_cachemask_base.h"

namespace FakeQuantAffineCachemaskN {
using namespace AscendC;
using namespace std;

template <typename yType>
class FakeQuantAffineCachemaskFp32 : public FakeQuantAffineCachemaskBase<yType> {
    public:
        __aicore__ inline FakeQuantAffineCachemaskFp32() {}
        __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR zero_point, GM_ADDR y, GM_ADDR mask,
            const FakeQuantAffineCachemaskTilingData* tilingData) {
            this->BaseMemberDataInit(tilingData);

            xGm.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(x) + this->offset, this->blockLength);
            scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(scale) + this->scaleOffset, this->circleNum);
            zeroGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(zero_point) + this->scaleOffset, this->circleNum);
            yGm.SetGlobalBuffer(reinterpret_cast<__gm__ yType*>(y) + this->offset, this->blockLength);
            maskGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(mask) + this->offset, this->blockLength);

            // pipe alloc memory to queue, the unit is Bytes
            pipe.InitBuffer(inQueueData, BUFFER_NUM, this->tileLength * sizeof(yType));
            pipe.InitBuffer(outQueueOut, BUFFER_NUM, this->tileLength * sizeof(yType));
            pipe.InitBuffer(outQueueMask, BUFFER_NUM, this->tileLength * sizeof(uint8_t));
            pipe.InitBuffer(calcBuf, BUFFER_NUM * this->tileLength * sizeof(yType));
            pipe.InitBuffer(calcInt32Buf, BUFFER_NUM * this->tileLength * sizeof(int32_t));
            pipe.InitBuffer(calcHf16Buf, BUFFER_NUM * this->tileLength * sizeof(half));
            pipe.InitBuffer(maskBuf, BUFFER_NUM * this->tileLength * sizeof(uint8_t));
            pipe.InitBuffer(selectBuf, BUFFER_NUM * this->tileLength * sizeof(yType));
            
            pipe.InitBuffer(quantMinQueueBuf, this->tileLength * sizeof(yType));
            pipe.InitBuffer(quantMaxQueueBuf, this->tileLength * sizeof(yType));
            pipe.InitBuffer(zeroBuf, this->tileLength * sizeof(yType));
            pipe.InitBuffer(infBuf, this->tileLength * sizeof(yType));
            pipe.InitBuffer(oneBuf, this->tileLength * sizeof(yType));
        }

        __aicore__ inline void Process() {
            this->CommonBufferGet(infBuf, zeroBuf, oneBuf, quantMinQueueBuf, quantMaxQueueBuf,
                infTensor, zeroTensor, oneTensor, quantMinTensor, quantMaxTensor, this->tileLength);
            for (uint32_t i = 0; i < this->circleNum; i++) {
                scaleValue = static_cast<float>(scaleGm.GetValue(i));
                zeroPointValue = zeroGm.GetValue(i);

                uint32_t dataOffset = i * this->calcLength;
                for (uint32_t j = 0; j < this->tileNum; j++) {
                    uint32_t calcOffset = dataOffset + j * this->tileLength;
                    repeatTimes = (this->tileLength + this->mask - 1) / this->mask;
                    this->CommonCopyIn(inQueueData, xGm, calcOffset, this->tileLength);
                    pipe_barrier(PIPE_ALL);
                    Compute(this->tileLength);
                    pipe_barrier(PIPE_ALL);
                    this->CommonCopyOut(outQueueOut, outQueueMask, yGm, maskGm, calcOffset, this->tileLength);
                    pipe_barrier(PIPE_ALL);
                }

                if (this->lastTileLength > 0) {
                    uint32_t calcOffset = dataOffset + this->tileNum * this->tileLength;
                    repeatTimes = (this->lastTileLength + this->mask  - 1) / this->mask ;
                    this->CommonCopyIn(inQueueData, xGm, calcOffset, this->lastActulTileLength);
                    pipe_barrier(PIPE_ALL);
                    Compute(this->tileLength);
                    pipe_barrier(PIPE_ALL);
                    this->CommonCopyOut(outQueueOut, outQueueMask, yGm, maskGm, calcOffset, this->lastActulTileLength);
                }
            }
        }

    private:
        __aicore__ inline void Compute(uint32_t calCount) {
            LocalTensor<yType> xLocal = inQueueData.DeQue<yType>();
            LocalTensor<yType> yLocal = outQueueOut.AllocTensor<yType>();
            LocalTensor<uint8_t> maskLocal = outQueueMask.AllocTensor<uint8_t>();

            curTemp = calcBuf.Get<yType>();
            curInt32Temp = calcInt32Buf.Get<int32_t>();
            curHf16Temp = calcHf16Buf.Get<half>();
            selectTemp = selectBuf.Get<yType>();
            maskTemp = maskBuf.Get<uint8_t>();

            // tmp = x / scale + zero_point
            Muls(curTemp, xLocal, static_cast<yType>(1.0f / scaleValue), calCount);
            Cast(curInt32Temp, curTemp, RoundMode::CAST_RINT, calCount);
            Cast(curTemp, curInt32Temp, RoundMode::CAST_NONE, calCount);
            Cast(yLocal, curInt32Temp, RoundMode::CAST_NONE, calCount);
            Adds(curTemp, curTemp, static_cast<yType>(zeroPointValue), calCount);
            Cast(curInt32Temp, curTemp, RoundMode::CAST_RINT, calCount);
            pipe_barrier(PIPE_ALL);

            // maskTemp = (round(curTemp) >= quant_min) & (round(curTemp) <= quant_max)
            Compare(maskTemp, curTemp, quantMinTensor, CMPMODE::GE, calCount);
            BinaryRepeatParams repeatParams = {1, 1, 1, 8, 8, 8};
            Select(selectTemp, maskTemp, oneTensor, zeroTensor, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->mask, repeatTimes, repeatParams);
            Compare(maskTemp, curTemp, quantMaxTensor, CMPMODE::LE, calCount);
            Select(curTemp, maskTemp, oneTensor, zeroTensor, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->mask, repeatTimes, repeatParams);
            // use Mul to replace &
            Mul(curTemp, selectTemp, curTemp, calCount);
            // fp32 -> fp16 -> uint8
            Cast(curHf16Temp, curTemp, RoundMode::CAST_RINT, calCount);
            Cast(maskLocal, curHf16Temp, RoundMode::CAST_RINT, calCount);
            pipe_barrier(PIPE_ALL);

            // output = (Mins(Maxs(tmp, quant_min), quant_max) - zero_point) * scale
            Maxs(curInt32Temp, curInt32Temp, static_cast<int32_t>(this->quantMin), calCount);
            Mins(curInt32Temp, curInt32Temp, static_cast<int32_t>(this->quantMax), calCount);
            Cast(curTemp, curInt32Temp, RoundMode::CAST_ROUND, calCount);
            Compare(maskTemp, xLocal, xLocal, CMPMODE::EQ, calCount);
            Select(curTemp, maskTemp, curTemp, static_cast<yType>(this->quantMin), SELMODE::VSEL_TENSOR_SCALAR_MODE, this->mask, repeatTimes, repeatParams);
            Compare(maskTemp, yLocal, infTensor, CMPMODE::NE, calCount);
            Select(curTemp, maskTemp, curTemp, static_cast<yType>(this->quantMin), SELMODE::VSEL_TENSOR_SCALAR_MODE, this->mask, repeatTimes, repeatParams);
            Adds(curTemp, curTemp, static_cast<yType>(-1 * zeroPointValue), calCount);
            Muls(yLocal, curTemp, static_cast<yType>(scaleValue), calCount);
            pipe_barrier(PIPE_ALL);

            outQueueOut.EnQue<yType>(yLocal);
            outQueueMask.EnQue<uint8_t>(maskLocal);
            inQueueData.FreeTensor(xLocal);
        }

    private:
        TPipe pipe;

        int32_t zeroPointValue = 1;
        float scaleValue = 1.0;
        uint8_t repeatTimes = 0;

        GlobalTensor<yType> xGm, scaleGm, yGm;
        GlobalTensor<int32_t> zeroGm;
        GlobalTensor<uint8_t> maskGm;

        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueData;
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOut, outQueueMask;
        TBuf<QuePosition::VECCALC> calcBuf, calcInt32Buf, calcHf16Buf, maskBuf,
            quantMinQueueBuf, quantMaxQueueBuf, infBuf, zeroBuf, oneBuf, selectBuf;

        LocalTensor<yType> curTemp, infTensor, zeroTensor, oneTensor, quantMinTensor, quantMaxTensor, selectTemp;
        LocalTensor<int32_t> curInt32Temp;
        LocalTensor<half> curHf16Temp;
        LocalTensor<uint8_t> maskTemp;
};
} // FakeQuantAffineCachemaskN
#endif // _FAKE_QUANT_AFFINE_CACHEMASK_FP32_H_