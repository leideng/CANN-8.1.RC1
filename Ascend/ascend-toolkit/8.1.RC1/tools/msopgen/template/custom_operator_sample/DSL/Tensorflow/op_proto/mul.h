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
 * \file mul.h
 * \brief
 */
#ifndef MUL_H
#define MUL_H
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Returns x1 * x2 element-wise.
* "y = x1 * x2"

* @par Inputs:
* @li x1: A Tensor. Must be one of the following types: float16, float32,
* float64, uint8, int8, uint16, int16, int32, int64, complex64, complex128.
* @li x2: A Tensor. Must be one of the following types: float16, float32,
* float64, uint8, int8, uint16, int16, int32, int64, complex64, complex128. \n

* @par Outputs:
* y: A Tensor. Must be one of the following types: float16, float32, float64,
* uint8, int8, uint16, int16, int32, int64, complex64, complex128. \n

* @attention Constraints:
* @li "x1" and "x2" have incompatible shapes or types. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Multiply.
*/
REG_OP(Mul)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DI_UINT16, DT_INT16, DT_INT32, DT_INT64,
                           DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(Mul)
}  // namespace ge
#endif  // MUL_H
