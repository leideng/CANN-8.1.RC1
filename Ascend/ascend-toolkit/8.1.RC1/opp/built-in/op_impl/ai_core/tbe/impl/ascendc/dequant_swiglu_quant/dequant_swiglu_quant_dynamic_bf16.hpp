/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
 * \file dequant_swiglu_quant_dynamic_bf16.hpp
 * \brief
 */

#ifndef CANN_DEQUANT_SWIGLU_QUANT_DYNAMIC_BF16_HPP
#define CANN_DEQUANT_SWIGLU_QUANT_DYNAMIC_BF16_HPP

#include "kernel_operator.h"
#include "dequant_swiglu_quant_dynamic_base.hpp"

namespace DequantSwigluQuant {
using namespace AscendC;

TEMPLATE_DECLARE
class DequantSwigluQuantDynamicBF16 : public DequantSwigluQuantDynamicBase<TEMPLATE_ARGS> {
public:
    __aicore__ inline DequantSwigluQuantDynamicBF16(){};
    __aicore__ inline ~DequantSwigluQuantDynamicBF16(){};

    __aicore__ inline void Init(GM_ADDR x_gm, GM_ADDR weight_scale_gm, GM_ADDR activation_scale_gm, GM_ADDR bias_gm,
        GM_ADDR quant_scale_gm, GM_ADDR quant_offset_gm, GM_ADDR y_gm, GM_ADDR scale_gm,
        GM_ADDR userspace, const SwiGluTilingData* tilingData, TPipe* pipe_);
    __aicore__ inline void Process();
};

TEMPLATE_DECLARE
__aicore__ inline void DequantSwigluQuantDynamicBF16<TEMPLATE_ARGS>::Init(
    GM_ADDR x_gm, GM_ADDR weight_scale_gm, GM_ADDR activation_scale_gm, GM_ADDR bias_gm, GM_ADDR quant_scale_gm,
    GM_ADDR quant_offset_gm, GM_ADDR y_gm, GM_ADDR scale_gm, GM_ADDR userspace, const SwiGluTilingData* tilingData,
    TPipe* pipe_) {
    this->InitCommon(x_gm, weight_scale_gm, activation_scale_gm, bias_gm, quant_scale_gm, quant_offset_gm, y_gm, scale_gm, userspace, tilingData, pipe_);

    if (this->numRound < this->baseRowLen) {
        this->baseRowLen = this->numRound;
    }
    this->InitUbBufferCommon(this->baseColLen, this->numRound);
}

TEMPLATE_DECLARE
__aicore__ inline void DequantSwigluQuantDynamicBF16<TEMPLATE_ARGS>::Process() {
    this->BaseProcess();
}
}  // namespace DequantSwigluQuant
#endif  // CANN_DEQUANT_SWIGLU_QUANT_DYNAMIC_BF16_HPP
