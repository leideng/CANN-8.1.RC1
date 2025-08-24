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
 * \file dequant_swiglu_quant_dynamic_bias_float.hpp
 * \brief
 */

#ifndef CANN_DEQUANT_SWIGLU_QUANT_DYNAMIC_BIAS_FLOAT_HPP
#define CANN_DEQUANT_SWIGLU_QUANT_DYNAMIC_BIAS_FLOAT_HPP

#include "kernel_operator.h"
#include "dequant_swiglu_quant_dynamic_base.hpp"

namespace DequantSwigluQuant {
using namespace AscendC;

TEMPLATE_DECLARE

class DequantSwigluQuantDynamicBiasFloat : public DequantSwigluQuantDynamicBase<TEMPLATE_ARGS> {
public:
    __aicore__ inline DequantSwigluQuantDynamicBiasFloat(){};
    __aicore__ inline ~DequantSwigluQuantDynamicBiasFloat(){};

    __aicore__ inline void Init(GM_ADDR x_gm, GM_ADDR weight_scale_gm, GM_ADDR activation_scale_gm, GM_ADDR bias_gm,
        GM_ADDR quant_scale_gm, GM_ADDR quant_offset_gm, GM_ADDR y_gm, GM_ADDR scale_gm,
        GM_ADDR userspace, const SwiGluTilingData* tilingData, TPipe* pipe_);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitUbBuffer(uint64_t tileLength, uint32_t realRowLen);
};

TEMPLATE_DECLARE
__aicore__ inline void DequantSwigluQuantDynamicBiasFloat<TEMPLATE_ARGS>::Init(
    GM_ADDR x_gm, GM_ADDR weight_scale_gm, GM_ADDR activation_scale_gm, GM_ADDR bias_gm, GM_ADDR quant_scale_gm,
    GM_ADDR quant_offset_gm, GM_ADDR y_gm, GM_ADDR scale_gm, GM_ADDR userspace, const SwiGluTilingData* tilingData,
    TPipe* pipe_)
{
    this->InitCommon(x_gm, weight_scale_gm, activation_scale_gm, bias_gm, quant_scale_gm, quant_offset_gm, y_gm, scale_gm, userspace, tilingData, pipe_);
    this->weightScaleGm.SetGlobalBuffer((__gm__ float*)weight_scale_gm, this->colNum);
    if (this->biasIsEmpty == 0) {
        this->biasGm.SetGlobalBuffer((__gm__ BiasType*)bias_gm, this->colNum);
    }

    if (this->activateScaleIsEmpty == 0) {
        this->activationScaleGm.SetGlobalBuffer((__gm__ float*) activation_scale_gm + this->biasOffset,
            this->numRound);
    }
    
    this->InitUbBufferCommon(this->baseColLen, this->numRound);
    
    InitUbBuffer(this->baseColLen, this->numRound);
}

TEMPLATE_DECLARE
__aicore__ inline void DequantSwigluQuantDynamicBiasFloat<TEMPLATE_ARGS>::Process() {
    this->BaseProcess();
}

TEMPLATE_DECLARE
__aicore__ inline void DequantSwigluQuantDynamicBiasFloat<TEMPLATE_ARGS>::InitUbBuffer(uint64_t tileLength,
    uint32_t realRowLen)
{
    uint64_t alignTileLength = tileLength;
    if (!this->isOut32BAligned) {
        alignTileLength = this->Align(tileLength, sizeof(int8_t));
    }
    if (this->biasIsEmpty == 0) {
        this->pipe->InitBuffer(this->inBiasQueueA, 1, alignTileLength * sizeof(BiasType));
        this->pipe->InitBuffer(this->inBiasQueueB, 1, alignTileLength * sizeof(BiasType));
        if constexpr (std::is_same_v<BiasType, bfloat16_t> || std::is_same_v<BiasType, half>) {
            this->pipe->InitBuffer(this->inputBiasTempBufferA, alignTileLength * sizeof(float));
            this->pipe->InitBuffer(this->inputBiasTempBufferB, alignTileLength * sizeof(float));
            this->biasFloatLocalA = this->inputBiasTempBufferA.template Get<CalcType>();
            this->biasFloatLocalB = this->inputBiasTempBufferB.template Get<CalcType>();
        }
    }

    this->pipe->InitBuffer(this->weightScaleQueueA, 1, alignTileLength * sizeof(float));
    this->pipe->InitBuffer(this->weightScaleQueueB, 1, alignTileLength * sizeof(float));
    if (this->activateScaleIsEmpty == 0) {
        this->pipe->InitBuffer(this->inQueueActivationScale, 1, this->baseRowLen * sizeof(float));
    }
}
}  // namespace DequantSwigluQuant
#endif  // CANN_DEQUANT_SWIGLU_QUANT_DYNAMIC_BIAS_FLOAT_HPP
