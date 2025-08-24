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
 * \file pad.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_PAD_H_
#define OPS_BUILT_IN_OP_PROTO_PAD_H_

#include "graph/operator_reg.h"
namespace ge {
/**
*@brief Pads a tensor . \n

*@par Inputs:
*Two inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, float32, double, int32,
*     uint8, int16, int8, complex64, int64, qint8, quint8, qint32, qint16, quint16, uint16,
*     complex128, uint32, uint64.
* @li paddings: A Tensor of type int32 or int64 . \n

*@par Outputs:
*y: A Tensor of the same type as "x" . \n

*@par Third-party framework compatibility:
* Compatible with TensorFlow operator Pad.
*/
REG_OP(Pad)
    .INPUT(x, TensorType::BasicType())
    .INPUT(paddings, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(Pad)
} // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_PAD_H_