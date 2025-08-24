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
 * \file relu.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_RELU_H_
#define OPS_BUILT_IN_OP_PROTO_INC_RELU_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Computes rectified linear: "max(x, 0)".
*
* @par Inputs:
* x: A tensor. Must be one of the following types: float32, float64, int32, uint8,
*     int16, int8, int64, uint16, float16, qint8.
*
* @par Outputs:
* y: A tensor. Has the same type as "x".
*
* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator Relu.
* @li Compatible with the Caffe operator ReLULayer.
*
*/
REG_OP(Relu)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE,
                          DT_INT8, DT_INT32, DT_INT16, DT_INT64,
                          DT_UINT8, DT_UINT16, DT_QINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE,
                           DT_INT8, DT_INT32, DT_INT16, DT_INT64,
                           DT_UINT8, DT_UINT16, DT_QINT8}))
    .OP_END_FACTORY_REG(Relu)

} // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_RELU_H_