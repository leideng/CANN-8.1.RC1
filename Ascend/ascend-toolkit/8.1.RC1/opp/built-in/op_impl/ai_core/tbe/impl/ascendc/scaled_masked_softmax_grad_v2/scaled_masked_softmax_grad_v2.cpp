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
 * \file scaled_masked_softmax_grad_v2.cpp
 * \brief
 */
#include "scaled_masked_softmax_grad_v2_norm_headdim.h"
#include "scaled_masked_softmax_grad_v2_large_headdim.h"

using namespace ScaledMaskedSoftmaxGradV2;

extern "C" __global__ __aicore__ void scaled_masked_softmax_grad_v2(const GM_ADDR yGrad, const GM_ADDR y,
    const GM_ADDR mask, const GM_ADDR xGrad, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(1000)) {
        ScaledMaskedSoftmaxGradV2NormHeadDim<bfloat16_t> op;
        op.Init(yGrad, y, mask, xGrad, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1001)) {
        ScaledMaskedSoftmaxGradV2NormHeadDim<half> op;
        op.Init(yGrad, y, mask, xGrad, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1002)) {
        ScaledMaskedSoftmaxGradV2NormHeadDim<float> op;
        op.Init(yGrad, y, mask, xGrad, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2000)) {
        ScaledMaskedSoftmaxGradV2LargeHeadDim<bfloat16_t> op;
        op.Init(yGrad, y, mask, xGrad, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2001)) {
        ScaledMaskedSoftmaxGradV2LargeHeadDim<half> op;
        op.Init(yGrad, y, mask, xGrad, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2002)) {
        ScaledMaskedSoftmaxGradV2LargeHeadDim<float> op;
        op.Init(yGrad, y, mask, xGrad, tilingData);
        op.Process();
    }
}