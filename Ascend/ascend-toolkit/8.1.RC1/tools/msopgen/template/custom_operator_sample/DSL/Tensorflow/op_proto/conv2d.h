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
 * \file conv2d.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_CONV2D_H_
#define OPS_BUILT_IN_OP_PROTO_INC_CONV2D_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Computes a 2D convolution given 4D "x" and "filter" tensors.
*@par Inputs:
*@li x: A 4D tensor of input image. With the format "NHWC", the data is stored
* in the order of: [batch, in_height, in_width, in_channels].
*@li filter: A 4D tensor of learnable filters. Must have the same type as "x".
* With the format "HWCN" , the data is stored in the order of: [filter_height,
* filter_width, in_channels / groups, out_channels].
*@li bias: An optional 1D tensor of additive biases to the filter outputs.
* The data is stored in the order of: [out_channels].
*@li offset_w: Reserved.
*\n
*\n
* The following are the supported data types and data formats:
*@verbatim
    | Tensor    | x       | filter  | bias    | y
    ------------|---------|---------|---------|--------
    | Data Type | float16 | float16 | float16 | float16
    |           |---------|---------|---------|--------
    |           | float32 | float32 | float32 | float32
    |           |---------|---------|---------|--------
    |           | int8    | int8    | int32   | int32
    ------------|---------|---------|---------|--------
    | Format    | NCHW    | NCHW    | ND      | NCHW
    |           | NHWC    | HWCN    |         | NHWC
@endverbatim
* For float32 type, the actual calculation on the chip is based on
* float16. For int8, a dequant or requant operator must be followed.
*\n
*
*@par Attributes:
*@li strides: Required. A list of 4 integers. The stride of the sliding window
* for each dimension of input. The dimension order is determined by the data
* format of "x". The N and C dimensions must be set to 1.
*@li pads: Required. A list of 4 integers. The number of pixels to add to each
* (top, bottom, left, right) side of the input.
*@li dilations: Optional. A list of 4 integers. The dilation factor for each
* dimension of input. The dimension order is determined by the data format of
* "x". The N and C dimensions must be set to 1. Defaults to [1, 1, 1, 1].
*@li groups: Optional. An integer of type int32. The number of blocked
* connections from input channels to output channels. In_channels and
* out_channels must both be divisible by "groups". Defaults to 1.
*@li offset_x: Optional. An integer of type int32. The negative offset added
* to the input image for int8 type. Ensure that the output is within the
* effective range. Defaults to 0.
*@li data_format: Reserved.
*\n
*\n
* The following value range restrictions must be met:
*@verbatim
    | Name             | Field    | Scope
    -------------------|----------|--------------
    | Input Image Size | H        | [1, 100000]
    |                  | W        | [1, 4096]
    -------------------|----------|--------------
    | Filter Size      | H        | [1, 255]
    |                  | W        | [1, 255]
    -------------------|----------|--------------
    | Stride           | H        | [1, 63]
    |                  | W        | [1, 63]
    -------------------|----------|--------------
    | Padding          | Top      | [0, 255]
    |                  | Bottom   | [0, 255]
    |                  | Left     | [0, 255]
    |                  | Right    | [0, 255]
    -------------------|----------|--------------
    | Dilation         | H        | [1, 255]
    |                  | W        | [1, 255]
    -------------------|----------|--------------
    | Offset_x         |          | [-128, 127]

@endverbatim
* The W dimension of the input image supports cases exceeding 4096, but it may
* cause compilation errors.
*\n
*
*@par Outputs:
*@li y: A 4D Tensor of output feature map. Has the same type as "x". With the
* format "NHWC", the data is stored in the order of: [batch, out_height,
* out_width, out_channels].
*\n
*     out_height = (in_height + pad_top + pad_bottom -
*                   (dilation_h * (filter_height - 1) + 1))
*                  / stride_h + 1
*\n
*     out_width = (in_width + pad_left + pad_right -
*                  (dilation_w * (filter_width - 1) + 1))
*                 / stride_w + 1
*\n
*
*@par Quantization supported or not
*@li Yes
*
*@par Third-party framework compatibility
*@li Compatible with the TensorFlow operator "conv2d".
*@li Compatible with the Caffe operator 2D "Convolution".
*/
REG_OP(Conv2D)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NHWC")
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(Conv2D)
}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_CONV2D_H_
