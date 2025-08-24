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
 * \file foreach_sub_list.cpp
 * \brief
 */

#include "kernel_operator.h"

// op kernel building at build_out directory, it's not fully aligned with source code structure
// current op_kernel folder is absent in build_out directory, so the relative path to common has just one layer
#include "../foreach_utils/foreach_one_scalar_ternary.h"

using namespace AscendC;
using namespace Common::OpKernel;

constexpr uint8_t BYTE_PER_BLOCK = 32;

template <typename T>
__aicore__ void SubListNormalAdapter(
    const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal1, const LocalTensor<T>& srcLocal2, const T& scalarVal, const int32_t& uValue) {
    Muls(srcLocal2, srcLocal2, scalarVal, uValue);
    pipe_barrier(PIPE_V);
    Sub(dstLocal, srcLocal1, srcLocal2, uValue);
}

template <typename T>
__aicore__ void SubListFloatAdapter(
    const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal1, const LocalTensor<T>& srcLocal2, const T& scalarVal, const int32_t& uValue) {
    Axpy<T, T>(srcLocal1, srcLocal2, -scalarVal, uValue);
    if(dstLocal.GetPhyAddr() != srcLocal1.GetPhyAddr()){
        pipe_barrier(PIPE_V);
        if (uValue * sizeof(T) % BYTE_PER_BLOCK == 0) {
            DataCopy(dstLocal, srcLocal1, uValue);
        } else {
            int32_t dataCountInBlock = BYTE_PER_BLOCK / sizeof(T);
            DataCopy(dstLocal, srcLocal1, (uValue + dataCountInBlock - 1) / dataCountInBlock * dataCountInBlock);
        }
    }
}

extern "C" __global__ __aicore__ void foreach_sub_list(GM_ADDR inputs_1, GM_ADDR inputs_2,
    GM_ADDR alpha, GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling) {
    
    GET_TILING_DATA(tilingData, tiling);

    //foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachOneScalarTernary<half, half, SubListNormalAdapter<half>> op;
        op.Init(inputs_1, inputs_2, alpha, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachOneScalarTernary<float, float, SubListFloatAdapter<float>> op;
        op.Init(inputs_1, inputs_2, alpha, outputs, userWS, &tilingData);
        op.Process();
    } 
    #if __CCE_AICORE__ == 220
    else if (TILING_KEY_IS(3)) {
        ForeachOneScalarTernary<int, int, SubListNormalAdapter<int>> op;
        op.Init(inputs_1, inputs_2, alpha, outputs, userWS, &tilingData);
        op.Process();
    } 
    else if (TILING_KEY_IS(4)) {
        ForeachOneScalarTernary<bfloat16_t, float, SubListFloatAdapter<float>> op;
        op.Init(inputs_1, inputs_2, alpha, outputs, userWS, &tilingData);
        op.Process();
    }
    #endif
}

