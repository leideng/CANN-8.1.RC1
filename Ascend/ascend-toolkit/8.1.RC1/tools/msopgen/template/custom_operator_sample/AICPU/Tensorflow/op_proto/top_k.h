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
 * \file top_k.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_TOP_K_H_
#define OPS_BUILT_IN_OP_PROTO_TOP_K_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Finds values and indices of the "k" largest elements for the last
* dimension . \n

* @par Inputs:
* Two inputs, including:
* @li x: A 1D or higher tensor of type BasicType, with the last dimension
* at least "k".
* @li k: A 0D Tensor of type int32.
* Number of top elements to look for along the last dimension (along each row
* for matrices) . \n

* @par Attributes:
* @li sorted: Defaults to true.
* If true, the resulting "k" elements will be sorted by the values in descending
* order.
* @li largest:If true the resulting `k` elements will be sorted by the values in descending order.
* @li dim:0-D. Number of top elements to look for along the last dimension (along each row for matrices). \n

* @par Outputs:
* @li values: A Tensor, specifying the sorted data. Has the same type as
* "input".
* @li indices: A Tensor of type int32, specifying the indices of sorted data . \n

* @see TopK()
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator TopKV2.
*/
REG_OP(TopK)
    .INPUT(x, TensorType::RealNumberType())
    .INPUT(k, TensorType({DT_INT32}))
    .OUTPUT(values, TensorType::RealNumberType())
    .OUTPUT(indices, TensorType({DT_INT32}))
    .ATTR(sorted, Bool, true)
    .ATTR(largest, Bool, true)
    .ATTR(dim, Int, -1)
    .OP_END_FACTORY_REG(TopK)
} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_TOP_K_H_
