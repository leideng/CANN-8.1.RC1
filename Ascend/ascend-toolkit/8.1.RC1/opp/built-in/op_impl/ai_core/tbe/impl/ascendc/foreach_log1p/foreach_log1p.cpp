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
 * \file foreach_log1p.cpp
 * \brief
 */

#include "kernel_operator.h"

// op kernel building at build_out directory, it's not fully aligned with source code structure
// current op_kernel folder is absent in build_out directory, so the relative path to common has just one layer
#include "../foreach_utils/foreach_implict_output.h"

using namespace AscendC;
using namespace Common::OpKernel;

template <typename T>
__aicore__ void Log1pAdapter(
    const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const int32_t& uValue) {
    T scalarVal = T(1);
    Adds(dstLocal, srcLocal, scalarVal, uValue);
    pipe_barrier(PIPE_V);
    Log<T>(dstLocal, srcLocal);
}

extern "C" __global__ __aicore__ void foreach_log1p(GM_ADDR x,  GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    //foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachImplictOutput<half, half, Log1pAdapter<half>, 2, 1> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachImplictOutput<float, float, Log1pAdapter<float>, 2, 1> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        ForeachImplictOutput<bfloat16_t, float, Log1pAdapter<float>, 2, 1> op;
        op.Init(x, y, userWS, &tilingData);
        op.Process();
    }
}
