/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
 * \file nn_quantize.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_QUANTIZE_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_QUANTIZE_H_
#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Transfer quant param from float32 to uint64.

* @par Inputs:
* @li scale: A quantization parameter tensor. Must be one of the following types: float32.
             The format support ND. The shape is 1D (t,), with t equal to 1 or n, or 2D(1, n),
             where n is the same as that of x2 in the matmul calculation.
* @li offset: A optional quantization parameter tensor. Must be one of the following types: float32.
              The format support ND. The shape is 1D (t,), with t equal to 1 or n, or 2D(1, n),
              where n is the same as that of x2 in the matmul calculation. \n


* @par Outputs:
* @li y: output tensor. Must be one of the following types: uint64. The format support ND.
         The shape is 1D (t,), with t equal to 1 or n, or 2D(1, n),
         where n is the same as that of x2 in the matmul calculation. \n

* @attention Constraints:
  1. The passed scale, out cannot be a null pointer.
  2. The format, dtype and shape of scale, offset, out must be supported.
*/
REG_OP(TransQuantParamV2)
    .INPUT(scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_UINT64}))
    .OP_END_FACTORY_REG(TransQuantParamV2)

/**
*@brief Fake-quantize the data of 'x' tensor with scale, zero_point, quant_min and quant_max. \n

*@par Inputs:
*Three inputs, including:
*@li x: A Tensor. Must be one of the following types: float16, float32.
*@li scale: A Tensor of type float32 or float16. Has the same type and format as "x".
*@li zero_point: A Tensor of type int32, float16 or float32.\n

*@par Attributes:
*@li axis: An required attribute of type int64.
*@li quant_min: An required attribute of type int64.
*@li quant_max: An required attribute of type int64.\n

*@par Outputs:
*y: A Tensor of type float32 or float16.
*mask: A Tensor of type bool. \n

*@par Third-party framework compatibility
* Compatible with Pytorch operator FakeQuantAffineCachemask.
*/

REG_OP(FakeQuantAffineCachemask)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(scale, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(zero_point, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(mask, TensorType({DT_BOOL}))
    .REQUIRED_ATTR(axis, Int)
    .REQUIRED_ATTR(quant_min, Int)
    .REQUIRED_ATTR(quant_max, Int)
    .OP_END_FACTORY_REG(FakeQuantAffineCachemask)

/**
 * @brief Dynamic Quant. Performs pre-token symmetric dynamic quantization on input tensors.
 * @par Inputs:
 * @li x: A Tensor. Type is:DT_FLOAT16 or DT_BF16. For Atlas A2 Training Series Product/Atlas
 * 800I A2 Inference Product/A200I A2 Box Heterogeneous Component and Atlas A3 Training Series Product/Atlas A3 Inference Series Product.
 * Whose shape must be greater than 1. The data format support ND.
 * @li smooth_scales: An optional Tensor. Shape is the last dimension of x.
 * The data type can be FLOAT16 or BFLOAT16.
 * @li group_index: An optional Tensor. Specifying the index of group. 1-D with shape
 * [E, ], the first dim of scale shape is same as the first dim of scale shape.
 * Must be one of the following types: int32. The format support ND.
 * @par Attributes:
 * dst_type: An optional attribute of type int. Declare the output dtype.
 * Support DT_INT4, DT_INT8. Defaults to DT_INT8.
 * @par Outputs:
 * @li y: A Tensor. Quantized output tensor, Shape is same as input x.
 * The format support ND. Type is DT_INT8 or DT_INT4.
 * @li scale: A Tensor. Scale used for quantization. 
 * Type is DT_FLOAT32. The format support ND.
 */
REG_OP(DynamicQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(smooth_scales, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(group_index, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT4}))
    .OUTPUT(scale, TensorType({DT_FLOAT}))
    .ATTR(dst_type, Int, DT_INT8)
    .OP_END_FACTORY_REG(DynamicQuant)

/**
 * @brief Dynamic Quant V2. Performs pre-token asymmetric dynamic quantization on input tensors.
 * @par Inputs:
 * @li x: A Tensor. Type is:DT_FLOAT16 or DT_BF16. For Atlas A2 Training Series Product/Atlas 800I A2
 * Inference Product/A200I A2 Box Heterogeneous Component and Atlas A3 Training Series Product/Atlas A3 Inference Series Product.
 * Whose shape must be greater than 1. The data format support ND.
 * @li smooth_scales: An optional Tensor. Shape is the last dimension of x.
 * The data type can be FLOAT16 or BFLOAT16.
 * The data type must be the same as that of x. The data format support ND.
 * @li group_index: An optional Tensor. Specifying the index of group. 1-D with shape
 * [E, ], the first dim of scale shape is same as the first dim of scale shape.
 * Must be one of the following types: int32. The format support ND.
 * @par Attributes:
 * dst_type: An optional attribute of type int. Declare the output dtype.
 * Support DT_INT4, DT_INT8. Defaults to DT_INT8.
 * @par Outputs:
 * @li y: A Tensor. Quantized output tensor, Shape is same as input x.
 * The format support ND. Type is DT_INT8 or DT_INT4.
 * @li scale: A Tensor. Scale used for quantization. 
 * Type is DT_FLOAT32. The format support ND.
 * @li offset: A Tensor. Offset used for quantization. 
 * Type is DT_FLOAT32. The format support ND.
 */
REG_OP(DynamicQuantV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(smooth_scales, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(group_index, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT4}))
    .OUTPUT(scale, TensorType({DT_FLOAT}))
    .OUTPUT(offset, TensorType({DT_FLOAT}))
    .ATTR(dst_type, Int, DT_INT8)
    .OP_END_FACTORY_REG(DynamicQuantV2)

/**
 * @brief Quantize feature map by group.
 * @par Inputs:
 * @li x: A Tensor. 2-D with shape [S, H]. Must be one of the following types:
 * float32, float16, bfloat16. The format support ND.
 * @li scale: A Tensor. Specifying the quantitation scale of x. 2-D with shape
 * [E, H], the second dim of scale shape is same as the second dim of x shape.
 * Must be one of the following types: float32, float16, bfloat16.
 * The format support ND.
 * @li group_index: A Tensor. Specifying the index of group. 1-D with shape
 * [E, ], the first dim of scale shape is same as the first dim of scale shape.
 * Must be one of the following types: int32, int64. The format support ND.
 * @li offset: A Tensor. Optional. Specifying the quantitation offset of x. 1-D
 * with shape [1, ] or 0-D with shape []. Must be one of the following types:
 * float32, float16, bfloat16. The dtype of offset should be same as scale.
 * The format support ND.
 * @par Outputs:
 * y: A 2-D Tensor. Shape is same as input x. The format support ND.
 * Must be one of the following types: int4, int8.
 * @par Attributes:
 * dst_type: An optional attribute of type int. Declare the output dtype.
 * Support DT_INT4, DT_INT8. Defaults to DT_INT8.
 * @attention Constraints:
 * @li If output y data type is INT4, the last dim of y shape should be
 * an even number.
 * @li Input group_index value should be in the range of [0, S] and be an
 * non-decreasing sequence. The last value of input group_index must be the
 * same as the first dim of x shape.
 */
REG_OP(GroupQuant)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(scale, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(group_index, TensorType({DT_INT32, DT_INT64}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_INT4, DT_INT8}))
    .ATTR(dst_type, Int, DT_INT8)
    .OP_END_FACTORY_REG(GroupQuant)

/**
 * @brief Flatness matters for LLM quantization. \n
 * 
 * @par Inputs:
 * @li x: A tensor. The original data input. 3-D with shape [K, M, N]. Must be one of the following types:
 * float16, bfloat16. The format support ND.
 * @li kronecker_p1: A tensor. Input calculation matrix 1. 2-D with shape [M, M]. The value of M is same as input "x".
 * Must be one of the following types: float16, bfloat16. Has the same type as input "x".
 * The format support ND.
 * @li kronecker_p2: A tensor. Input calculation matrix 2. 2-D with shape [N, N]. The value of N is same as input "x".
 * Must be one of the following types: float16, bfloat16. Has the same type as input "x".
 * The format support ND. \n
 * 
 * @par Outputs:
 * @li out: A 3-D tensor of type int4. Output result data. Shape is same as input "x". The format support ND.
 * @li quant_scale: A tensor of type float32. Output quantization factor. 1-D with shape [K].
 * The value of K is same as input "x". The format support ND. \n

 * @par Attributes:
 * clip_ratio: An optional float. Used to control the quantization cropping ratio. Defaults to 1. \n
 */
REG_OP(FlatQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(kronecker_p1, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(kronecker_p2, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(out, TensorType({DT_INT4}))
    .OUTPUT(quant_scale, TensorType({DT_FLOAT}))
    .ATTR(clip_ratio, Float, 1)
    .OP_END_FACTORY_REG(FlatQuant)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_QUANTIZE_H_
