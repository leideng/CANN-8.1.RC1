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
 * \file max_pool.h
 * \brief
 */
#ifndef MAX_POOL_H
#define  MAX_POOL_H

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief Performs max pooling on the input . \n

* @par Inputs:
* One input:
* x: An NC1HWC0 Tensor. Supported type:float16, float32, double, int8, int16,
* int32, int64, uint8, uint16, qint8

* @par Attributes:
* @li ksize: A required list of int8, int16, int32, or int64 values,
* specifying the size of the window for each dimension of the input tensor.
* No default value.
* @li strides: A required list of int8, int16, int32, or int64 values,
* specifying the stride of the sliding window for each dimension of
* the input tensor. No default value.
* @li padding: A required string. No default value.
* @li data_format: An optional string. Defaults to "NHWC" . \n

* @par Outputs:
* y: A Tensor. Has the same type and format as input "x" . \n

* @attention Constraints:
* @li "ksize" is a list that has length 4: ksize[0] = 1 or ksize[3] = 1,
* "ksize[1] * ksize[2] <= 255".
* @li "stride is a list that has length 4: strides[0] = 1 or strides[3] = 1,
* "strides[1] <= 63, strides[0] >= 1, strides[2] <= 63, strides[2] >= 1".
* @li "padding" is either "SAME" or "VALID".


* @par Third-party framework compatibility
* Compatible with the TensorFlow operator MaxPool.
*/
REG_OP(MaxPool)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE, DT_INT8,
                          DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                          DT_UINT16, DT_QINT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT32, DT_DOUBLE, DT_INT8,
                           DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_QINT8}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(padding, String)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(MaxPool)
}  // namespace ge
#endif  // MAX_POOL_H