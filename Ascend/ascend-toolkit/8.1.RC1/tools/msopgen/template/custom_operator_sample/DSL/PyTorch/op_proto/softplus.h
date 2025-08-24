/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file softplus.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_SOFTPLUS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SOFTPLUS_H_

#include "graph/operator_reg.h"

namespace ge {
	
/**
*@brief Computes softplus: log(exp(x) + 1) . \n

*@par Inputs:
* One input:
*x: A Tensor of type bfloat16, float16 or float32. Up to 8D . \n

*@par Outputs:
*y: The activations tensor. Has the same type and format as input "x"

*@par Third-party framework compatibility
* Compatible with the TensorFlow operator Softplus.
*/
REG_OP(Softplus)
    .INPUT(x, TensorType({FloatingDataType, DT_BF16}))
    .OUTPUT(y, TensorType({FloatingDataType, DT_BF16}))
    .OP_END_FACTORY_REG(Softplus)
} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_SOFTPLUS_H_