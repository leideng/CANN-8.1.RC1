/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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
 * \file masked_fill.h
 * \brief
 */
 
#ifndef GE_OP_MASKED_FILL_H
#define GE_OP_MASKED_FILL_H
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Replace the value of X with value according to mask.
* @par Inputs:
* three inputs, including:
*  @li x: A Tensor of dtype is float16 or float32 or int32 or int8.
*  @li mask: A Tensor of dtype float16 or float32 or int32 or int8.
*  @li value: A Tensor or scalar of dtype float16 or float32 or int32 or int8. \n

* @par Outputs:
*  @li y: A tensor. Must be one of the following dtypes:
*   float16, float32, int32, int8.
*/
REG_OP(MaskedFill)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT32}))
    .INPUT(mask, TensorType({DT_BOOL}))
    .INPUT(value, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT32}))
    .OP_END_FACTORY_REG(MaskedFill)
}  // namespace ge

#endif  // GE_OP_MASKED_FILL_H