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
 * \file foreach_addcmul_list.cpp
 * \brief
 */

#include "kernel_operator.h"

// op kernel building at build_out directory, it's not fully aligned with source code structure
// current op_kernel folder is absent in build_out directory, so the relative path to common has just one layer
#include "../foreach_utils/foreach_one_scalar_quaternary.h"

using namespace AscendC;
using namespace Common::OpKernel;

constexpr uint8_t BYTE_PER_BLOCK = 32;

template <typename T>
__aicore__ void AddcMulListNormalAdapter(
    const LocalTensor<T>& dstLocal, 
    const LocalTensor<T>& tensor1Local,
    const LocalTensor<T>& tensor2Local,
    const LocalTensor<T>& tensor3Local,
    const T& scalarVal,
    const int32_t& uValue) {
    Mul(tensor2Local, tensor2Local, tensor3Local, uValue);
    pipe_barrier(PIPE_V);
    Muls(tensor2Local, tensor2Local, scalarVal, uValue);
    pipe_barrier(PIPE_V);
    Add(dstLocal, tensor1Local, tensor2Local, uValue);
}

template <typename T>
__aicore__ void AddcMulListFloatAdapter(
    const LocalTensor<T>& dstLocal, 
    const LocalTensor<T>& tensor1Local,
    const LocalTensor<T>& tensor2Local,
    const LocalTensor<T>& tensor3Local,
    const T& scalarVal,
    const int32_t& uValue) {
    Mul(tensor2Local, tensor2Local, tensor3Local, uValue);
    pipe_barrier(PIPE_V);
    Axpy<T, T>(tensor1Local, tensor2Local, scalarVal, uValue);
    if(dstLocal.GetPhyAddr() != tensor1Local.GetPhyAddr()){
        pipe_barrier(PIPE_V);
        if (uValue * sizeof(T) % BYTE_PER_BLOCK == 0) {
            DataCopy(dstLocal, tensor1Local, uValue);
        } else {
            int32_t dataCountInBlock = BYTE_PER_BLOCK / sizeof(T);
            DataCopy(dstLocal, tensor1Local, (uValue + dataCountInBlock - 1) / dataCountInBlock * dataCountInBlock);
        }
    }
}

extern "C" __global__ __aicore__ void foreach_addcmul_list(GM_ADDR tensor1, GM_ADDR tensor2, GM_ADDR tensor3, GM_ADDR scalar,
    GM_ADDR outputs, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    //foreach(vector) not need workspace
    GM_ADDR userWS = nullptr;

    if (TILING_KEY_IS(1)) {
        ForeachOneScalarQuaternary<half, half, AddcMulListFloatAdapter, 2, 3> op;
        op.Init(tensor1, tensor2, tensor3, scalar, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ForeachOneScalarQuaternary<float, float, AddcMulListFloatAdapter, 2, 3> op;
        op.Init(tensor1, tensor2, tensor3, scalar, outputs, userWS, &tilingData);
        op.Process();
    }  
    #if __CCE_AICORE__ == 220
    else if (TILING_KEY_IS(3)) {
        ForeachOneScalarQuaternary<int, int, AddcMulListNormalAdapter, 2, 3> op;
        op.Init(tensor1, tensor2, tensor3, scalar, outputs, userWS, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        ForeachOneScalarQuaternary<bfloat16_t, float, AddcMulListFloatAdapter, 2, 3> op;
        op.Init(tensor1, tensor2, tensor3, scalar, outputs, userWS, &tilingData);
        op.Process();
    }
    #endif
}
