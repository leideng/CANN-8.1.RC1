/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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
 * \file dequant_swiglu_quant_static_base.hpp
 * \brief
 */

#ifndef CANN_DEQUANT_SWIGLU_QUANT_STATIC_BASE_HPP
#define CANN_DEQUANT_SWIGLU_QUANT_STATIC_BASE_HPP
#include "kernel_operator.h"

#define TEMPLATE_DECLARE_STATIC template<typename InType, typename CalcType, typename BiasType, typename OutType, uint16_t bufferNum, uint16_t quantIsOne>
#define TEMPLATE_ARGS_STATIC InType, CalcType, BiasType, OutType, bufferNum, quantIsOne

namespace DequantSwigluQuant {
constexpr uint32_t NUM2 = 2;
using namespace AscendC;

TEMPLATE_DECLARE_STATIC
class DequantSwigluQuantStaticBase {
public:
    __aicore__ inline DequantSwigluQuantStaticBase() {}
    __aicore__ inline ~DequantSwigluQuantStaticBase() {}

    __aicore__ inline void InitCommon(GM_ADDR x_gm, GM_ADDR weight_scale_gm, GM_ADDR activation_scale_gm, GM_ADDR bias_gm,
        GM_ADDR quant_scale_gm, GM_ADDR quant_offset_gm, GM_ADDR y_gm, GM_ADDR scale_gm,
        const SwiGluTilingData* tilingData, TPipe* pipe_) {
        this->blockIdx = GetBlockIdx();
        this->activateLeft = tilingData->activateLeft;
        this->quantScaleIsEmpty = tilingData->quantScaleIsEmpty;
        this->activateScaleIsEmpty = tilingData->activateScaleIsEmpty;
        this->biasIsEmpty = tilingData->biasIsEmpty;
        this->colNum = tilingData->colLen;
        this->rowNum = tilingData->rowLen;
        this->usedCoreNum = tilingData->usedCoreNum;

        this->baseRowLen = tilingData->baseRowLen;
        this->baseColLen = tilingData->baseColLen < this->colNum ? tilingData->baseColLen : this->colNum;
        this->curColNum = this->baseColLen;
        if (this->rowNum < this->usedCoreNum) {
            this->usedCoreNum = this->rowNum;
        }
        int64_t perRoundCnt = this->usedCoreNum == 0 ? 0 : this->rowNum / this->usedCoreNum;
        int64_t remainCnt = this->rowNum - this->usedCoreNum * perRoundCnt;
        this->curCoreRowNum = perRoundCnt;
        if (this->blockIdx < remainCnt) {
            this->curCoreRowNum = perRoundCnt + 1;
            this->inputCopyOffset = this->blockIdx * this->curCoreRowNum;
        } else {
            this->inputCopyOffset = remainCnt * (perRoundCnt + 1) + (this->blockIdx - remainCnt) * perRoundCnt;
        }

        this->xGm.SetGlobalBuffer((__gm__ InType*)x_gm + this->inputCopyOffset * this->colNum * NUM2, this->curCoreRowNum * this->colNum * NUM2);
        this->yGm.SetGlobalBuffer((__gm__ OutType*)y_gm + this->inputCopyOffset * this->colNum, this->curCoreRowNum * this->colNum);
        if (quantScaleIsEmpty == 0) {
            if constexpr(quantIsOne == 0) {
                this->quantOffsetGm.SetGlobalBuffer((__gm__ float*) quant_offset_gm, this->colNum);
                this->quantScaleGm.SetGlobalBuffer((__gm__ float*) quant_scale_gm, this->colNum);
            } else {
                this->quantScaleGm.SetGlobalBuffer((__gm__ float*) quant_scale_gm, 1);
                this->quant_scale = 1 / this->quantScaleGm.GetValue(0);
                this->quantOffsetGm.SetGlobalBuffer((__gm__ float*) quant_offset_gm, 1);
                this->quant_offset = this->quantOffsetGm.GetValue(0);
            }
        }
    }

    __aicore__ inline void InitUbBufferCommon()
    {
        int64_t alignColNum = curColNum == Align(curColNum, sizeof(InType)) ? curColNum : Align(curColNum, sizeof(OutType));
        pipe->InitBuffer(inputTempBufferInt32SD, alignColNum * sizeof(CalcType) * NUM2);
        pipe->InitBuffer(swigluTempBuffer, alignColNum * sizeof(CalcType));
        pipe->InitBuffer(inQueue, bufferNum, alignColNum * sizeof(InType) * NUM2);
        pipe->InitBuffer(outQueue, bufferNum, alignColNum * sizeof(OutType));
        if (quantScaleIsEmpty == 0) {
            if constexpr(quantIsOne == 0) {
                pipe->InitBuffer(inQueueQuant, bufferNum, alignColNum * sizeof(float) * NUM2);
            }
        }
    }

    __aicore__ inline void CopyInWeightAndBias(int64_t offset)
    {
        DataCopyExtParams params = {1, static_cast<uint32_t>(curColNum * sizeof(float)), 0, 0, 0};
        DataCopyExtParams paramsBias = {1, static_cast<uint32_t>(curColNum * sizeof(BiasType)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPadExtParams<BiasType> padParams1{false, 0, 0, 0};

        LocalTensor<float> weightLocal = inQueueWeightScale.template AllocTensor<float>();
        LocalTensor<BiasType> biasTensorLocal;
        if (this->biasIsEmpty == 0) {
            biasTensorLocal = inQueueBias.template AllocTensor<BiasType>();
        }

        if (activateLeft == 0) {
            DataCopyPad(weightLocal, weightScaleGm[offset + colNum], params, padParams);
            DataCopyPad(weightLocal[alignColNum], weightScaleGm[offset], params, padParams);

            if (this->biasIsEmpty == 0) {
                DataCopyPad(biasTensorLocal, biasGm[offset + colNum], paramsBias, padParams1);
                DataCopyPad(biasTensorLocal[alignColNum], biasGm[offset], paramsBias, padParams1);
            }
        } else {
            DataCopyPad(weightLocal, weightScaleGm[offset], params, padParams);
            DataCopyPad(weightLocal[alignColNum], weightScaleGm[offset + colNum], params, padParams);
            if (this->biasIsEmpty == 0) {
                DataCopyPad(biasTensorLocal, biasGm[offset], paramsBias, padParams1);
                DataCopyPad(biasTensorLocal[alignColNum], biasGm[offset + colNum], paramsBias, padParams1);
            }
        }
        if (activateScaleIsEmpty == 0) {
            DataCopyExtParams activateparams = {1, static_cast<uint32_t>(curCoreRowNum * sizeof(float)), 0, 0, 0};
            LocalTensor<float> activateLocal = inQueueActivationScale.template AllocTensor<float>();
            DataCopyPad(activateLocal, activationScaleGm, activateparams, padParams);
            inQueueActivationScale.EnQue(activateLocal);
        }
        inQueueWeightScale.EnQue(weightLocal);
        if (this->biasIsEmpty == 0) {
            inQueueBias.EnQue(biasTensorLocal);
        }
    }

    __aicore__ inline void CopyInQuant(int64_t offset)
    {
        DataCopyExtParams params = {1, static_cast<uint32_t>(curColNum * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        LocalTensor<float> quantLocal = inQueueQuant.template AllocTensor<float>();
        DataCopyPad(quantLocal, quantScaleGm[offset], params, padParams);
        DataCopyPad(quantLocal[alignColNum], quantOffsetGm[offset], params, padParams);
        inQueueQuant.EnQue(quantLocal);
    }

    __aicore__ inline void CopyIn(int64_t offset1, int64_t offset2)
    {
        DataCopyExtParams params = {1, static_cast<uint32_t>(curColNum * sizeof(InType)), 0, 0, 0};
        DataCopyPadExtParams <InType> padParams{false, 0, 0, 0};

        LocalTensor <InType> aLocal = inQueue.template AllocTensor<InType>();
        if (activateLeft == 0) {
            DataCopyPad(aLocal, xGm[offset2], params, padParams);
            DataCopyPad(aLocal[alignColNum], xGm[offset1], params, padParams);
        } else {
            DataCopyPad(aLocal, xGm[offset1], params, padParams);
            DataCopyPad(aLocal[alignColNum], xGm[offset2], params, padParams);
        }
        inQueue.EnQue(aLocal);
    }

    __aicore__ inline void dequant(uint64_t tileLen, uint64_t i)
    {
        LocalTensor <InType> aLocal = this->inQueue.template DeQue<InType>();
        this->inputTmpELocal = this->inputTempBufferInt32SD.template Get<CalcType>();
        if constexpr (std::is_same_v<BiasType, int32_t>) {
            if (this->biasIsEmpty == 0) {
                Add(aLocal, aLocal, this->biasLocal, tileLen);
                pipe_barrier(PIPE_V);
            }
        }

        Cast(this->inputTmpELocal, aLocal, RoundMode::CAST_NONE, tileLen);
        pipe_barrier(PIPE_V);
        this->inQueue.template FreeTensor(aLocal);

        Mul(this->inputTmpELocal, this->inputTmpELocal, this->weightScaleLocal, tileLen);
        pipe_barrier(PIPE_V);

        if (this->activateScaleIsEmpty == 0) {
            float value = this->activateLocal.GetValue(i);
            Muls(this->inputTmpELocal, this->inputTmpELocal, value, tileLen);
            pipe_barrier(PIPE_V);
        }

        if constexpr (std::is_same_v<BiasType, float> || std::is_same_v<BiasType, bfloat16_t> || std::is_same_v<BiasType, half>) {
            if (this->biasIsEmpty == 0) {
                if constexpr (std::is_same_v<BiasType, float>) {
                    Add(this->inputTmpELocal, this->inputTmpELocal, this->biasLocal, tileLen);
                } else {
                    LocalTensor<CalcType> biasFloatLocal = this->inputBiasTempBuffer.template Get<CalcType>();
                    Cast(biasFloatLocal, this->biasLocal, RoundMode::CAST_NONE, tileLen);
                    pipe_barrier(PIPE_V);
                    Add(this->inputTmpELocal, this->inputTmpELocal, biasFloatLocal, tileLen);
                }
                pipe_barrier(PIPE_V);
            }
        }
    }

    __aicore__ inline void processCompute()
    {
        int64_t lastColNum = this->baseColLen;
        int64_t colLoops = 1;
        if (this->baseColLen < this->colNum) {
            colLoops = (this->colNum + this->baseColLen - 1) / this->baseColLen;
            lastColNum = this->colNum - (colLoops - 1) * this->baseColLen;
        }
        for (int64_t colLoop = 0; colLoop < colLoops; colLoop++) {
            if (colLoop == colLoops - 1) {
                this->curColNum = lastColNum;
            }
            bool isAligned = this->curColNum == this->Align(this->curColNum, sizeof(InType));
            this->alignColNum = isAligned ? this->curColNum : this->Align(this->curColNum, sizeof(int8_t));
            this->CopyInWeightAndBias(colLoop * this->baseColLen);
            if (this->biasIsEmpty == 0) {
                this->biasLocal = this->inQueueBias.template DeQue<BiasType>();
            }
            this->weightScaleLocal = this->inQueueWeightScale.template DeQue<float>();
            if (this->activateScaleIsEmpty == 0) {
                this->activateLocal = this->inQueueActivationScale.template DeQue<float>();
            }
            for (int64_t i = 0; i < this->curCoreRowNum; i++) {
                this->CopyIn(i * this->colNum * NUM2 + colLoop * this->baseColLen, i * this->colNum * NUM2 + this->colNum + colLoop * this->baseColLen);
                this->dequant(this->alignColNum * NUM2, i);
                if (i == 0 && quantScaleIsEmpty == 0) {
                    if constexpr(quantIsOne == 0) {
                        this->CopyInQuant(colLoop * this->baseColLen);
                        this->quantLocal = this->inQueueQuant.template DeQue<float>();
                    }
                }
                this->swiglu(this->alignColNum, i);
                this->CopyOut(colLoop, i);
            }
            if (this->biasIsEmpty == 0) {
                this->inQueueBias.FreeTensor(this->biasLocal);
            }
            this->inQueueWeightScale.FreeTensor(this->weightScaleLocal);
            if (quantScaleIsEmpty == 0) {
                if constexpr(quantIsOne == 0) {
                    this->inQueueQuant.FreeTensor(this->quantLocal);
                }
            }
            if (this->activateScaleIsEmpty == 0) {
                this->inQueueActivationScale.template FreeTensor(this->activateLocal);
            }
        }
    }

    __aicore__ inline void swiglu(uint64_t curTileLen, int64_t idx)
    {
        LocalTensor <CalcType> swigluLocal = swigluTempBuffer.Get<CalcType>();
        Muls(swigluLocal, inputTmpELocal, beta, curTileLen);
        pipe_barrier(PIPE_V);
        Exp(swigluLocal, swigluLocal, curTileLen);
        pipe_barrier(PIPE_V);
        Adds(swigluLocal, swigluLocal, CalcType(1.0), curTileLen);
        pipe_barrier(PIPE_V);
        Div(inputTmpELocal, inputTmpELocal, swigluLocal, curTileLen);
        pipe_barrier(PIPE_V);
        Mul(inputTmpELocal[curTileLen], inputTmpELocal, inputTmpELocal[curTileLen], curTileLen);
        pipe_barrier(PIPE_V);
        if (quantScaleIsEmpty == 0) {
            if constexpr(quantIsOne == 0) {
                Div(inputTmpELocal[curTileLen], inputTmpELocal[curTileLen], quantLocal, curTileLen);
                pipe_barrier(PIPE_V);
                Add(inputTmpELocal[curTileLen], inputTmpELocal[curTileLen], quantLocal[curTileLen], curTileLen);
                pipe_barrier(PIPE_V);
            } else {
                Muls(inputTmpELocal[curTileLen], inputTmpELocal[curTileLen], quant_scale, curTileLen);
                pipe_barrier(PIPE_V);
                Adds(inputTmpELocal[curTileLen], inputTmpELocal[curTileLen], quant_offset, curTileLen);
                pipe_barrier(PIPE_V);
            }
        }
        // fp32->int16
        LocalTensor <int16_t> int16Local = swigluTempBuffer.Get<int16_t>();
        Cast(int16Local, inputTmpELocal[curTileLen], RoundMode::CAST_RINT, curTileLen);
        pipe_barrier(PIPE_V);
        // int16-> half
        LocalTensor <half> halfLocal = int16Local.ReinterpretCast<half>();
        Cast(halfLocal, int16Local, RoundMode::CAST_NONE, curTileLen);
        pipe_barrier(PIPE_V);

        LocalTensor <OutType> outLocal = outQueue.template AllocTensor<OutType>();
        // half -> int8
        Cast(outLocal, halfLocal, RoundMode::CAST_NONE, curTileLen);
        outQueue.template EnQue<OutType>(outLocal);
    }

    __aicore__ inline void CopyOut(int64_t colLoop, int64_t idx)
    {
        LocalTensor <OutType> outLocal = outQueue.template DeQue<OutType>();
        DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(curColNum * sizeof(OutType)), 0, 0, 0};
        DataCopyPad(yGm[idx * colNum + colLoop * baseColLen], outLocal, dataCopyParams);
        outQueue.FreeTensor(outLocal);
    }

protected:
    __aicore__ inline int64_t Align(int64_t elementNum, int64_t bytes)
    {
        constexpr int64_t BLOCK_BYTES = 32;
        if (bytes == 0) {
            return 0;
        }
        return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES / bytes;
    }

protected:
    float beta = -1.0;
    float quant_scale = 1;
    float quant_offset = 1;
    TPipe* pipe = nullptr;
    int64_t biasIsEmpty = 0;
    int64_t quantScaleIsEmpty = 0;
    int64_t activateScaleIsEmpty = 0;
    int64_t colNum = 0;
    int64_t rowNum = 0;
    int64_t curCoreRowNum = 0;
    int64_t inputCopyOffset = 0;
    int64_t alignColNum = 0;
    int64_t curColNum = 0;
    int64_t activateLeft = 0;
    int64_t blockIdx = 0;
    int64_t usedCoreNum = 0;
    int64_t baseRowLen = 0;
    int64_t baseColLen = 0;

    GlobalTensor <OutType> yGm;
    GlobalTensor <InType> xGm;
    GlobalTensor<float> weightScaleGm;
    GlobalTensor<float> activationScaleGm;
    GlobalTensor <BiasType> biasGm;
    GlobalTensor<float> quantScaleGm;
    GlobalTensor<float> quantOffsetGm;

    LocalTensor <CalcType> inputTmpELocal;
    LocalTensor<float> weightScaleLocal;
    LocalTensor<BiasType> biasLocal;
    LocalTensor<float> quantLocal;
    LocalTensor<float> activateLocal;

    TQue <QuePosition::VECIN, bufferNum> inQueueWeightScale;
    TQue <QuePosition::VECIN, bufferNum> inQueueActivationScale;
    TQue <QuePosition::VECIN, bufferNum> inQueueBias;
    TQue <QuePosition::VECIN, bufferNum> inQueueQuant;
    TQue <QuePosition::VECIN, bufferNum> inQueue;
    TQue <QuePosition::VECOUT, bufferNum> outQueue;

    TBuf <TPosition::VECCALC> inputTempBufferInt32SD;
    TBuf <TPosition::VECCALC> swigluTempBuffer;
    TBuf <TPosition::VECCALC> inputBiasTempBuffer;
};
}

#endif  // CANN_DEQUANT_SWIGLU_QUANT_STATIC_BASE_HPP
