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
 * \file less.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_LESS_H_
#define OPS_BUILT_IN_OP_PROTO_LESS_H_
#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Returns the truth value of (x1 < x2) element-wise. \n
*when input is int32 and (x2 - x1) > 2**31 or < -2**31
*aicore accuracy is not guaranteed \n

*@par Inputs:
*Two inputs, including:
* @li x1: A Tensor. Must be one of the following types: float16, float32, double, int32,
*     uint8, int16, int8, int64, uint16, uint32, uint64.
* @li x2: A Tensor with the same type as "x1". \n

*@par Outputs:
*y: A Tensor of type bool. \n

*@par Third-party framework compatibility:
* Compatible with TensorFlow operator Less.
*/
REG_OP(Less)
    .INPUT(x1, TensorType::RealNumberType())
    .INPUT(x2, TensorType::RealNumberType())
    .OUTPUT(y, TensorType({DT_BOOL}))
    .OP_END_FACTORY_REG(Less)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_LESS_H_
