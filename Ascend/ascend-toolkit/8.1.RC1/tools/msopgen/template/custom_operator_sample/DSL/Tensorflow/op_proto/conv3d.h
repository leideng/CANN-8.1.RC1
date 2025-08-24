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
 * \file conv3d.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_CONV_3D_H_
#define OPS_BUILT_IN_OP_PROTO_INC_CONV_3D_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Computes a 3D convolution given 5D "x" and "filter" tensors.
 *@par Inputs:
 * @li x: A 5D tensor. Must be one of the following types: float16,
 * (Currently does not support int8). The format of x is NCDHW or NDHWC.
 * @li filter: A 5D tensor of the same type as "x".
 * (Currently does not support int8).
 * The format is NCDHW, NDHWC or DHWCN . \n

*@par Optional input:
 * @li bias: An optional 1D tensor of the same type as "x".
 * @li offset_w: An optional 1D tensor for quantized deconvolution. Reserved . \n

*@par Required Attributes:
 * @li strides: A list of 5 integers. Specifies the stride of the sliding window
 * for each dimension of "x".
 * The N and C dimensions must be 1. Has the same format as "x".
 * @li pads: A list of 6 integers.
 * Supports only padding along the D, H and W dimensions in sequence of head,
 * tail, top, bottom, left and right . \n

*@par Attributes:
 * @li groups: Number of blocked connections from input channels to output
 * channels.
 * @li data_format: An optional string from: "NDHWC", "NCDHW".
 * Defaults to "NDHWC". Specify the data format of the input and output data.
 * @li dilations: A list of 5 integers. Specifies the dilation factor for each
 * dimension of "x".
 * The N, C and D dimensions must be 1. Has the same format as "x".
 * @li offset_x: An optional int. Input offset, used for quantized inference.
 * Defaults to 0. Reserved . \n

*@par Outputs:
 *y: A Tensor. Has the same type and data format as "x". \n

*@attention Constraints:
 *The image size after padding is greater than the filter size . \n

*@par Third-party framework compatibility
 * @li Compatible with the TensorFlow operator conv3d.
 * @li Compatible with the Caffe operator Convolution.
*/
REG_OP(Conv3D)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(filter, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NDHWC")
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(Conv3D)
}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_CONV_3D_H_