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
#ifndef GE_OP_FILL_V2_D_H
#define GE_OP_FILL_V2_D_H

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief Fill the value to a tensor has the specified shape.

* @par Attributes:
* @li value: An optional float value. Defaults to 0.0.

* @li dims: A required listInt to specify the shape that the value to fill.

* @par Outputs:
* @li y: A Tensor. Has the shape specify by attr shape, and full of the value specify by attr value.

* @par Third-party framework compatibility
* Compatible with the ONNX operator ConstantOfShape.
*/
REG_OP(FillV2D)
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8, DT_UINT8, DT_INT16, DT_INT32, DT_INT64}))
    .ATTR(value, Float, 0)
    .REQUIRED_ATTR(dims, ListInt)
    .OP_END_FACTORY_REG(FillV2D)
}  // namespace ge

#endif  // GE_OP_FILL_V2_D_H