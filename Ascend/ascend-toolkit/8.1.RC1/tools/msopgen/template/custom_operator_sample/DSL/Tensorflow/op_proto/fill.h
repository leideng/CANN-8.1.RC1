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
 * \file fill.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_FILL_H_
#define OPS_BUILT_IN_OP_PROTO_INC_FILL_H_

#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Creates a tensor filled with a scalar value.
* This operation creates a tensor of shape "dims" and fills it with "value".
*
*@par Inputs:
*@li dims: A 1D tensor of types int32 or int64. Represents the shape of the output tensor . \n

*@li value: A 0D scalar. Specifies the value to fill the returned tensor.
*    Must be one of the following types:
*    float16, float32, double, int32, uint8, int16, int8, complex64, int64,
*    qint8, quint8, qint32, uint16, complex128, uint32, uint64.
*
*@par Outputs:
* y: A tensor. Has the same type as "value".
*
*@par Third-party framework compatibility
*@li Compatible with the TensorFlow operator Fill.
*@li Compatible with the Caffe operator Filler.
*
*/
REG_OP(Fill)
    .INPUT(dims, TensorType::IndexNumberType())
    .INPUT(value, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(Fill)

} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_FILL_H_