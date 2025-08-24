/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
 * \file foreach_binary_op.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "../foreach_common/foreach_op_def.h"
#include "../foreach_common/foreach_functors.h"
#include "../foreach_common/foreach_op_register.h"
#include "../foreach_common/multi_tensor_apply.h"
#include "../foreach_common/foreach_op_compute.h"

using namespace AscendC;
using namespace foreachOp;
using namespace ForeachOpDef;

// List all supported data types
extern "C" __global__ __aicore__ void foreach_binary_op(GM_ADDR input0, GM_ADDR input1, GM_ADDR output,
                                                        GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tilingData, tiling);
  TENSORLIST_FOREACH_BINARY_OP_CALL(ADD_TENSOR_LIST, addOnTensorList, input0, input1, output, float, float, workspace,
                                    tilingData);
  TENSORLIST_FOREACH_BINARY_OP_CALL(ADD_TENSOR_LIST, addOnTensorList, input0, input1, output, int32_t, int32_t,
                                    workspace, tilingData);
  TENSORLIST_FOREACH_BINARY_OP_CALL(ADD_TENSOR_LIST, addOnTensorList, input0, input1, output, half, half, workspace,
                                    tilingData);
}