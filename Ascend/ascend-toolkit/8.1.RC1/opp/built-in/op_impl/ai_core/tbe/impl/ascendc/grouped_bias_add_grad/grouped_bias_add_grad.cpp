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
 * \file grouped_bias_add_grad.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "grouped_bias_add_grad_base.h"
#include "grouped_bias_add_grad_equal_c.h"
#include "grouped_bias_add_grad_unequal_c.h"
#include "grouped_bias_add_grad_unequal_c_perf.h"

using namespace GroupedBiasAddGradAll;
#define THREE_DIMS_HALF 1000000
#define THREE_DIMS_FLOAT 1000001
#define THREE_DIMS_BFLOAT16 1000002
#define THREE_DIMS_HALF_USEUB 1000100
#define THREE_DIMS_FLOAT_USEUB 1000101
#define THREE_DIMS_BFLOAT16_USEUB 1000102

#define TWO_DIMS_HALF 1000010
#define TWO_DIMS_FLOAT 1000011
#define TWO_DIMS_BFLOAT16 1000012
#define TWO_DIMS_HALF_USEUB 1000110
#define TWO_DIMS_FLOAT_USEUB 1000111
#define TWO_DIMS_BFLOAT16_USEUB 1000112

#define TWO_DIMS_HALF_PERF 1001010
#define TWO_DIMS_FLOAT_PERF 1001011
#define TWO_DIMS_BFLOAT16_PERF 1001012
#define TWO_DIMS_HALF_USEUB_PERF 1001110
#define TWO_DIMS_FLOAT_USEUB_PERF 1001111
#define TWO_DIMS_BFLOAT16_USEUB_PERF 1001112

#define TWO_DIMS_HALF_GRP64 1010010
#define TWO_DIMS_FLOAT_GRP64 1010011
#define TWO_DIMS_BFLOAT16_GRP64 1010012
#define TWO_DIMS_HALF_USEUB_GRP64 1010110
#define TWO_DIMS_FLOAT_USEUB_GRP64 1010111
#define TWO_DIMS_BFLOAT16_USEUB_GRP64 1010112

#define TWO_DIMS_HALF_GRP64_PERF 1011010
#define TWO_DIMS_FLOAT_GRP64_PERF 1011011
#define TWO_DIMS_BFLOAT16_GRP64_PERF 1011012
#define TWO_DIMS_HALF_USEUB_GRP64_PERF 1011110
#define TWO_DIMS_FLOAT_USEUB_GRP64_PERF 1011111
#define TWO_DIMS_BFLOAT16_USEUB_GRP64_PERF 1011112

template <typename T, const uint32_t USE_TYPE>
__aicore__ inline void InvokeTemplateGroupedEqualC(GM_ADDR grad_y,
                                                   GM_ADDR grad_bias, GM_ADDR userWS,
                                                   const GroupedBiasAddGradTilingData& tilingData)
{
    GroupedBiasAddGradEqualC<T, USE_TYPE> op;
    op.Init(grad_y, grad_bias, userWS, tilingData);
    op.Process();
}

template <typename T, typename G, const uint32_t USE_TYPE>
__aicore__ inline void InvokeTemplateGroupedUnequalC(GM_ADDR grad_y, GM_ADDR group_idx,
                                                     GM_ADDR grad_bias, GM_ADDR userWS,
                                                     const GroupedBiasAddGradTilingData& tilingData)
{
    GroupedBiasAddGradUnequalC<T, G, USE_TYPE> op;
    op.Init(grad_y, group_idx, grad_bias, userWS, tilingData);
    op.Process();
}

template <typename T, typename G, const uint32_t USE_TYPE>
__aicore__ inline void InvokeTemplateGroupedUnequalCPerf(GM_ADDR grad_y, GM_ADDR group_idx,
                                                         GM_ADDR grad_bias, GM_ADDR userWS,
                                                         const GroupedBiasAddGradTilingData& tilingData)
{
    GroupedBiasAddGradUnequalCPerf<T, G, USE_TYPE> op;
    op.Init(grad_y, group_idx, grad_bias, userWS, tilingData);
    op.Process();
}

extern "C" __global__ __aicore__ void grouped_bias_add_grad(GM_ADDR grad_y, GM_ADDR group_idx, GM_ADDR grad_bias,
                                                            GM_ADDR workspace, GM_ADDR tiling_data)
{
    if (workspace == nullptr) {
      return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
      return;
    }
    GET_TILING_DATA(tilingData, tiling_data);

    if (TILING_KEY_IS(TWO_DIMS_FLOAT)) {
        InvokeTemplateGroupedUnequalC<float, int32_t, USE_WS>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_HALF)) {
        InvokeTemplateGroupedUnequalC<half, int32_t, USE_WS>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_BFLOAT16)) {
        InvokeTemplateGroupedUnequalC<bfloat16_t, int32_t, USE_WS>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(THREE_DIMS_FLOAT)) {
        InvokeTemplateGroupedEqualC<float, USE_WS>(grad_y, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(THREE_DIMS_HALF)) {
        InvokeTemplateGroupedEqualC<half, USE_WS>(grad_y, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(THREE_DIMS_BFLOAT16)) {
        InvokeTemplateGroupedEqualC<bfloat16_t, USE_WS>(grad_y, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_FLOAT_USEUB)) {
        InvokeTemplateGroupedUnequalC<float, int32_t, USE_UB>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_HALF_USEUB)) {
        InvokeTemplateGroupedUnequalC<half, int32_t, USE_UB>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_BFLOAT16_USEUB)) {
        InvokeTemplateGroupedUnequalC<bfloat16_t, int32_t, USE_UB>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(THREE_DIMS_FLOAT_USEUB)) {
        InvokeTemplateGroupedEqualC<float, USE_UB>(grad_y, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(THREE_DIMS_HALF_USEUB)) {
        InvokeTemplateGroupedEqualC<half, USE_UB>(grad_y, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(THREE_DIMS_BFLOAT16_USEUB)) {
        InvokeTemplateGroupedEqualC<bfloat16_t, USE_UB>(grad_y, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_FLOAT_PERF)) {
        InvokeTemplateGroupedUnequalCPerf<float, int32_t, USE_WS>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_HALF_PERF)) {
        InvokeTemplateGroupedUnequalCPerf<half, int32_t, USE_WS>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_BFLOAT16_PERF)) {
        InvokeTemplateGroupedUnequalCPerf<bfloat16_t, int32_t, USE_WS>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_FLOAT_USEUB_PERF)) {
        InvokeTemplateGroupedUnequalCPerf<float, int32_t, USE_UB>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_HALF_USEUB_PERF)) {
        InvokeTemplateGroupedUnequalCPerf<half, int32_t, USE_UB>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_BFLOAT16_USEUB_PERF)) {
        InvokeTemplateGroupedUnequalCPerf<bfloat16_t, int32_t, USE_UB>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_FLOAT_GRP64)) {
        InvokeTemplateGroupedUnequalC<float, int64_t, USE_WS>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_HALF_GRP64)) {
        InvokeTemplateGroupedUnequalC<half, int64_t, USE_WS>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_BFLOAT16_GRP64)) {
        InvokeTemplateGroupedUnequalC<bfloat16_t, int64_t, USE_WS>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_FLOAT_USEUB_GRP64)) {
        InvokeTemplateGroupedUnequalC<float, int64_t, USE_UB>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_HALF_USEUB_GRP64)) {
        InvokeTemplateGroupedUnequalC<half, int64_t, USE_UB>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_BFLOAT16_USEUB_GRP64)) {
        InvokeTemplateGroupedUnequalC<bfloat16_t, int64_t, USE_UB>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_FLOAT_USEUB_GRP64_PERF)) {
        InvokeTemplateGroupedUnequalCPerf<float, int64_t, USE_UB>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_HALF_USEUB_GRP64_PERF)) {
        InvokeTemplateGroupedUnequalCPerf<half, int64_t, USE_UB>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_BFLOAT16_USEUB_GRP64_PERF)) {
        InvokeTemplateGroupedUnequalCPerf<bfloat16_t, int64_t, USE_UB>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_FLOAT_GRP64_PERF)) {
        InvokeTemplateGroupedUnequalCPerf<float, int64_t, USE_WS>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_HALF_GRP64_PERF)) {
        InvokeTemplateGroupedUnequalCPerf<half, int64_t, USE_WS>(grad_y, group_idx, grad_bias, userWS, tilingData);
    } else if (TILING_KEY_IS(TWO_DIMS_BFLOAT16_GRP64_PERF)) {
        InvokeTemplateGroupedUnequalCPerf<bfloat16_t, int64_t, USE_WS>(grad_y, group_idx, grad_bias, userWS, tilingData);
    }
}