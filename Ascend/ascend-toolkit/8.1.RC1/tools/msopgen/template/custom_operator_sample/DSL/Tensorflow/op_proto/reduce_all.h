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
 * \file reduce_all.h
 * \brief
 */
#ifndef REDUCE_ALL_H
#define REDUCE_ALL_H

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Calculates the "logical sum" of elements of a tensor in a dimension . \n

* @par Inputs:
* One input:
* x: The boolean tensor to reduce . \n

* @par Attributes:
* @li keep_dims: A bool. If true, retains reduced dimensions with length 1.
* @li axis: The dimensions to reduce. If None, reduces all dimensions.
* Must be in the range [- rank (input_sensor), rank (input_sensor)) . \n

* @par Outputs:
* y: The reduced tensor . \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator ReduceAll.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ReduceAll instead.
*/
REG_OP(ReduceAllD)
    .INPUT(x, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BOOL}))
    .REQUIRED_ATTR(axes, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(ReduceAllD)
} //namespace ge
#endif  // REDUCE_ALL_H