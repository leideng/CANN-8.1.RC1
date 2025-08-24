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
#ifndef GE_OP_LP_NORM_H
#define GE_OP_LP_NORM_H

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief Computes Lp norm.

* @par Inputs:
* @li x: An ND tensor of type float16, float32. \n
*
* @par Attributes:
* @li p: Int, "inf" or "-inf", default value is 2.
* @li axes: ListInt, {} means all axes will be computed.
* @li keepdim: Bool, default is false.
* @li epsilon: Float, default is 1e-12. \n

* @par Outputs:
* @li y: An ND tensor of type float16, float32. The shape of y is depending
* on axes and keepdim. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator LpNorm.
*/
REG_OP(LpNorm)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(p, Int, 2)
    .ATTR(axes, ListInt, {})
    .ATTR(keepdim, Bool, false)
    .ATTR(epsilon, Float, 1e-12)
    .OP_END_FACTORY_REG(LpNorm)
}  // namespace ge

#endif  // GE_OP_LP_NORM_H