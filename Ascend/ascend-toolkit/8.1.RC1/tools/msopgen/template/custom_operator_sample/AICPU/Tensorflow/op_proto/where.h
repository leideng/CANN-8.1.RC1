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
 * \file where.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_WHERE_H_
#define OPS_BUILT_IN_OP_PROTO_INC_WHERE_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
*@brief Returns locations of nonzero / true values in a tensor. \n

*@par Inputs:
*Including:
*x: A Tensor. Must be one of the following types:
DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT16,
DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL. \n

*@par Outputs:
*y: A Tensor of type DT_INT64. \n

*@attention Constraints:
*Where runs on the Ascend AI CPU, which delivers poor performance.\n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Where.
*/

REG_OP(Where)
    .INPUT(x, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT16, \
              DT_UINT16, DT_INT32, DT_UINT32, DT_INT64, DT_UINT64, DT_BOOL}))
    .OUTPUT(y, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(Where)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_WHERE_H_