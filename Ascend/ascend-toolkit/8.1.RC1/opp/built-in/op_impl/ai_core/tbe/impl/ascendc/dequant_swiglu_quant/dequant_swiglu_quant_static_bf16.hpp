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
 * \file dequant_swiglu_quant_static_bf16.hpp
 * \brief
 */

#ifndef CANN_DEQUANT_SWIGLU_QUANT_STATIC_BF16_HPP
#define CANN_DEQUANT_SWIGLU_QUANT_STATIC_BF16_HPP
#include "kernel_operator.h"
#include "dequant_swiglu_quant_static_base.hpp"
namespace DequantSwigluQuant {
using namespace AscendC;

TEMPLATE_DECLARE_STATIC
class DequantSwigluQuantStaticBF16 : public DequantSwigluQuantStaticBase<TEMPLATE_ARGS_STATIC> {
public:
    __aicore__ inline DequantSwigluQuantStaticBF16() {}
    __aicore__ inline ~DequantSwigluQuantStaticBF16() {}
    __aicore__ inline void Init(GM_ADDR x_gm, GM_ADDR weight_scale_gm, GM_ADDR activation_scale_gm, GM_ADDR bias_gm,
        GM_ADDR quant_scale_gm, GM_ADDR quant_offset_gm, GM_ADDR y_gm, GM_ADDR scale_gm,
        const SwiGluTilingData* tilingData, TPipe* pipe_);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void convertFloat(uint64_t curTileLen, uint64_t i);
};

TEMPLATE_DECLARE_STATIC
__aicore__ inline void DequantSwigluQuantStaticBF16<TEMPLATE_ARGS_STATIC>::Init(GM_ADDR x_gm, GM_ADDR weight_scale_gm,
    GM_ADDR activation_scale_gm, GM_ADDR bias_gm,
    GM_ADDR quant_scale_gm, GM_ADDR quant_offset_gm, GM_ADDR y_gm, GM_ADDR scale_gm,
    const SwiGluTilingData* tilingData, TPipe* pipe_)
{
    this->pipe = pipe_;
    this->InitCommon(x_gm, weight_scale_gm, activation_scale_gm, bias_gm, quant_scale_gm, quant_offset_gm, y_gm, scale_gm, tilingData, pipe_);
    this->InitUbBufferCommon();
}

TEMPLATE_DECLARE_STATIC
__aicore__ inline void DequantSwigluQuantStaticBF16<TEMPLATE_ARGS_STATIC>::Process()
{
    if (this->blockIdx >= this->usedCoreNum) {
        return;
    }
    int64_t colLoops = 1;
    int64_t lastColNum = this->baseColLen;
    if (this->baseColLen < this->colNum) {
        colLoops = (this->colNum + this->baseColLen - 1) / this->baseColLen;
        lastColNum = this->colNum - (colLoops - 1) * this->baseColLen;
    }
    for (int64_t colLoop = 0; colLoop < colLoops; colLoop++) {
        if (colLoop == colLoops - 1) {
            this->curColNum = lastColNum;
        }
        bool isOutAligned = this->curColNum == this->Align(this->curColNum, sizeof(InType));
        this->alignColNum = isOutAligned ? this->curColNum : this->Align(this->curColNum, sizeof(OutType));
        for (int64_t i = 0; i < this->curCoreRowNum; i++) {
            this->CopyIn(i * this->colNum * NUM2 + colLoop * this->baseColLen, i * this->colNum * NUM2 + this->colNum + colLoop * this->baseColLen);
            convertFloat(this->alignColNum * NUM2, i);
            if (i == 0 && this->quantScaleIsEmpty == 0) {
                if constexpr(quantIsOne == 0) {
                    this->CopyInQuant(colLoop * this->baseColLen);
                    this->quantLocal = this->inQueueQuant.template DeQue<float>();
                }
            }
            this->swiglu(this->alignColNum, i);
            this->CopyOut(colLoop, i);
        }
        if (this->quantScaleIsEmpty == 0) {
            if constexpr(quantIsOne == 0) {
                this->inQueueQuant.FreeTensor(this->quantLocal);
            }
        }
    }
}

TEMPLATE_DECLARE_STATIC
__aicore__ inline void DequantSwigluQuantStaticBF16<TEMPLATE_ARGS_STATIC>::convertFloat(uint64_t tileLen, uint64_t i)
{
    LocalTensor <InType> aLocal = this->inQueue.template DeQue<InType>();
    this->inputTmpELocal = this->inputTempBufferInt32SD.template Get<CalcType>();

    Cast(this->inputTmpELocal, aLocal, RoundMode::CAST_NONE, tileLen);
    pipe_barrier(PIPE_V);
    this->inQueue.template FreeTensor(aLocal);
}
}

#endif  // CANN_DEQUANT_SWIGLU_QUANT_STATIC_BF16_HPP
