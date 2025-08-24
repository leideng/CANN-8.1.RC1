/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file adaptive_max_pool_2d.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_ADAPTIVE_MAX_POOL_2D_H_
#define OPS_BUILT_IN_OP_PROTO_INC_ADAPTIVE_MAX_POOL_2D_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
*@brief Applies a 2D adaptive max pooling over an input signal conposed of several input planes. \n
* The output is of size H x W, for any input size. 

* @par Inputs:
* One input, including:
* @li x: A Tensor. Must be one of the following data types:
*     float16, float32, float64. \n

* @par Attributes:
* @li output_size: A required list of 2 ints
*    specifying the size (H,W) of the output tensor. \n

* @par Outputs:
* @li y: A Tensor. Has the same data type as "x" \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator AdaptiveMaxPool2d.
*/
REG_OP(AdaptiveMaxPool2d)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE}))
    .OUTPUT(argmax, TensorType::IndexNumberType())
    .REQUIRED_ATTR(output_size, ListInt)
    .OP_END_FACTORY_REG(AdaptiveMaxPool2d)
}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_ADAPTIVE_MAX_POOL_2D_H_