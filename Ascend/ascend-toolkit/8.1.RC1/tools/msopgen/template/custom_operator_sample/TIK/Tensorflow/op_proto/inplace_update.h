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
#ifndef GE_OP_INPLACE_UPDATE_H
#define GE_OP_INPLACE_UPDATE_H

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
*@brief Updates specified rows with values in v.
*Computes x[i, :] = v; return x.
*@par Inputs:
*Three inputs, including:
* @li x: A Tensor.
*     TensorType::NumberType().
* @li indices: A vector of type int32.
*     Indices into the left-most dimension of "x".
* @li v: A Tensor of the same type as "x".
*     Same dimension sizes as x except the first dimension,
*     which must be the same as the size of "indices" . \n

*@par Outputs:
*y: A Tensor of the same type as "x".
*   An alias of "x". The content of "y" is undefined if there are duplicates in indices.
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator InplaceUpdate.
*/
REG_OP(InplaceUpdate)
    .INPUT(x, TensorType::BasicType())
    .INPUT(indices, TensorType({DT_INT32}))
    .INPUT(v, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(InplaceUpdate)
}  // namespace ge

#endif  // GE_OP_INPLACE_UPDATE_H