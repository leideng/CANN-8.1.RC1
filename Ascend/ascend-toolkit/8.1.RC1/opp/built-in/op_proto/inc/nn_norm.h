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
 * \file nn_norm.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_H_

#include "graph/operator_reg.h"
namespace ge {
/**
* @brief Performs group normalization . \n

* @par Inputs:
* Three inputs
* @li x: A ND Tensor of type float16 or float32, with format NCHW for 4D.
* @li gamma: A Tensor of type float16 or float32. Must be 1D. Specifies the scaling factor.
* @li beta: A Tensor of type float16 or float32. Must be 1D. Specifies the offset.

* @par Attributes:
* @li num_groups: An required int32, specifying the number of group.
* @li eps: An optional float32, specifying the small value added to
variance to avoid dividing by zero. Defaults to "0.0001".
* @li data_format: An optional string, specifying the format of "x".
Defaults to "NHWC".
* @li is_training: An optional bool, specifying if the operation is used for
training or inference. Defaults to "True" .

* @par Outputs:
* Three outputs
* @li y: A ND Tensor of type float16 or float32 for the normalized "x",
with format NCHW for 4D.
* @li mean: A Tensor of type float16 or float32. Must be 1D. Specifies the mean of "x".
* @li variance: A Tensor of type float16 or float32. Must be 1D. Specifies the variance of "x".

* @attention Constraints:
* @li For Atlas 200/300/500 Inference Product, only support NCHW which can be trans to 5HD. 
* @li the value range of the inputs should be constrained between -10000 and 10000.

* @par Third-party framework compatibility
* @li Compatible with the PyTorch operator GroupNorm.

*/
REG_OP(GroupNorm)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(beta, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(variance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(data_format, String, "NHWC")
    .ATTR(eps, Float, 0.0001f)
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(GroupNorm)

/**
* @brief Performs group normalization . \n

* @par Inputs:
* Three inputs
* @li x: A ND Tensor of type float16 or float32, with format NCHW for 4D.
* @li gamma: A Tensor of type float16 or float32. Must be 1D. Specifies the scaling factor.
* @li beta: A Tensor of type float16 or float32. Must be 1D. Specifies the offset. \n

* @par Attributes:
* @li num_groups: An required int32, specifying the number of group.
* @li eps: An optional float32, specifying the small value added to
variance to avoid dividing by zero. Defaults to "0.0001".
* @li data_format: An optional string, specifying the format of "x".
Defaults to "NHWC".
* @li is_training: An optional bool, specifying if the operation is used for
training or inference. Defaults to "True" . \n

* @par Outputs:
* Three outputs
* @li y: A ND Tensor of type float16 or float32 for the normalized "x",
with format NCHW for 4D.
* @li mean: A Tensor of type float16 or float32. Must be 1D. Specifies the mean of "x".
* @li rstd: A Tensor of type float16 or float32. Must be 1D. Specifies the rstd of "x". \n

* @par Third-party framework compatibility
* @li Compatible with the PyTorch operator GroupNorm.

*/
REG_OP(GroupNormV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(beta, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(rstd, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(data_format, String, "NHWC")
    .ATTR(eps, Float, 0.0001f)
    .ATTR(is_training, Bool, true)
    .OP_END_FACTORY_REG(GroupNormV2)

/**
 * @brief backward operator for group normalization. \n
 * @par Inputs:
 * Five input, including:
 * @li dy: A Tensor. Group grad. Datatype support float32, float16, bfloat16. Format support ND. 
 * @li mean: A Tensor. Mean of each group. Datatype support float32, float16, bfloat16. Format support ND.
 * @li rstd: A Tensor. Reciprocal standard deviation of each group. Datatype support float32, float16, bfloat16. Format support ND.
 * @li x: A Tensor. Specifies the offset. Datatype support float32, float16, bfloat16. Format support ND.
 * @li gamma: A Tensor. Specifies the scaling factor. Datatype support float32, float16, bfloat16. Format support ND.

 * @par Attributes:
 * @li num_groups: Int.Number specifying the number of group.
 * @li data_format: An optional String, Defaults to NCHW. 
 * @li dx_is_require: An optional bool, controls whether to return x.grad. Defaults to true. 
 * @li dgamma_is_require: An optional bool, controls whether to return weight.grad. Defaults to true.
 * @li dbeta_is_require: An optional bool, controls whether to return beta.grad. Defaults to true.

 * @par Outputs:
 * Three output, including:
 * @li dx: A Tensor. x factor grad. Datatype is the same as the input Datatype. Format support ND.
 * @li dgamma: A Tensor. scale factor grad. Datatype is the same as the input Datatype. Format support ND.
 * @li dbeta: A Tensor. offset factor grad. Datatype is the same as the input Datatype. Format support ND.
 * @par Third-party framework compatibility
 * @li Compatible with the PyTorch operator GroupNorm.
 */

REG_OP(GroupNormGrad)
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(rstd, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(dx, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(dgamma, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(dbeta, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(data_format, String, "NCHW")
    .ATTR(dx_is_require, Bool, true)
    .ATTR(dgamma_is_require, Bool, true)
    .ATTR(dbeta_is_require, Bool, true)
    .OP_END_FACTORY_REG(GroupNormGrad)

/**
* @brief Performs group normalization and silu. 

* @par Inputs:
* Three inputs
* @li x: A ND Tensor of type bfloat16/float16/float32.
* @li gamma: A Tensor of type bfloat16/float16/float32.
* Must be 1D. Specifies the scaling factor.
* @li beta: A Tensor of type bfloat16/float16/float32.
* Must be 1D. Specifies the offset. 

* @par Attributes:
* @li num_groups: An required int32/int64, specifying the number of group.
* @li eps: An optional float32, specifying the small value added to the 
* denominator for numerical stability. Defaults to "0.00001".
* @li activate_silu: An optional bool.  Defaults to "true". 

* @par Outputs:
* Three outputs
* @li y: A ND Tensor of type bfloat16/float16/float32 for the normalized "x".
* @li mean: A Tensor of type bfloat16/float16/float32.
* Must be 1D. Specifies the mean of "x".
* @li rstd: A Tensor of type bfloat16/float16/float32.
* Must be 1D. Specifies the rstd of "x". 

* @par Third-party framework compatibility
* @li Compatible with the PyTorch operator GroupNorm and Silu.

*/
REG_OP(GroupNormSilu)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(gamma, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(beta, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(mean, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(rstd, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(eps, Float, 0.00001f)
    .ATTR(activate_silu, Bool, true)
    .OP_END_FACTORY_REG(GroupNormSilu)

/**
* @brief Performs group normalization and swish. \n

* @par Inputs:
* Three inputs
* @li x: A ND Tensor of type bfloat16/float16/float32.
* @li gamma: A Tensor of type bfloat16/float16/float32.
* Must be 1D. Specifies the scaling factor.
* @li beta: A Tensor of type bfloat16/float16/float32.
* Must be 1D. Specifies the offset. \n

* @par Attributes:
* @li num_groups: An required int32/int64, specifying the number of group.
* @li eps: An optional float32, specifying the small value added to the 
* denominator for numerical stability. Defaults to "0.00001".
* @li data_format: An optional String, Defaults to NCHW. 
* @li activate_swish: An optional bool.  Defaults to "true".
* @li swish_scale: An optional float.  Defaults to "1". \n

* @par Outputs:
* Three outputs
* @li y: A ND Tensor of type bfloat16/float16/float32 for the normalized "x".
* @li mean: A Tensor of type bfloat16/float16/float32.
* Must be 1D. Specifies the mean of "x".
* @li rstd: A Tensor of type bfloat16/float16/float32.
* Must be 1D. Specifies the rstd of "x". \n

* @par Third-party framework compatibility
* @li Compatible with the PyTorch operator GroupNorm and Swish.

*/
REG_OP(GroupNormSwish)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(gamma, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(beta, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(mean, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(rstd, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(data_format, String, "NCHW")
    .ATTR(eps, Float, 0.00001f)
    .ATTR(activate_swish, Bool, true)
    .ATTR(swish_scale, Float, 1.0)
    .OP_END_FACTORY_REG(GroupNormSwish)

/**
* @brief Performs the backward operation of group normalization and swish.

 * @par Inputs:
 * Six input, including:
 * @li dy: A Tensor. Group grad. Datatype support float32, float16, bfloat16. Format support ND. 
 * @li mean: A Tensor. Mean of each group. Datatype support float32, float16, bfloat16. Format support ND.
 * @li rstd: A Tensor. Reciprocal standard deviation of each group. Datatype support float32, float16, bfloat16. Format support ND.
 * @li x: A Tensor. Specifies the offset. Datatype support float32, float16, bfloat16. Format support ND. Same shape as mean.
 * @li gamma: A Tensor. Specifies the scaling factor. Datatype support float32, float16, bfloat16. Format support ND. Same shape as dy.
 * @li beta: A Tensor. Specifies the intercept. Datatype support float32, float16, bfloat16. Format support ND. Same shape as gamma.

* @par Attributes:
* @li num_groups: Int. Number specifying the number of group.
* @li data_format: An optional String, Defaults to NCHW. 
* @li swish_scale: An optional float. Defaults to "1.0".
* @li dgamma_is_require: An optional bool, controls whether to return weight.grad. Defaults to true.
* @li dbeta_is_require: An optional bool, controls whether to return beta.grad. Defaults to true.

 * @par Outputs:
 * Three output, including:
 * @li dx: A Tensor. x factor grad. Datatype is the same as the input Datatype. Format support ND.
 * @li dgamma: A Tensor. scale factor grad. Datatype is the same as the input Datatype. Format support ND.
 * @li dbeta: A Tensor. offset factor grad. Datatype is the same as the input Datatype. Format support ND.

* @par Third-party framework compatibility
* @li Compatible with the backward of PyTorch operator GroupNorm and Swish.

*/
REG_OP(GroupNormSwishGrad)
    .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(rstd, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(beta, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(dx, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(dgamma, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(dbeta, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(num_groups, Int)
    .ATTR(data_format, String, "NCHW")
    .ATTR(swish_scale, Float, 1.0)
    .ATTR(dgamma_is_require, Bool, true)
    .ATTR(dbeta_is_require, Bool, true)
    .OP_END_FACTORY_REG(GroupNormSwishGrad)

/**
* @brief AddRmsNorm operator interface implementation. \n
*  calculating: x1, x2, gamma \n
*  x = x1 + x2 \n
*  rstd = np.rsqrt(np.mean(np.power(x,2), reduce_axis, keepdims=True) + epsilon)) \n
*  y = gamma * (x * rstd)

* @par Inputs
* Three inputs, including:
* @li x1: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li x2: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li gamma: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].

* @par Attributes
* epsilon: Input eps in the formula, which is used to prevent division-by-zero errors.
* A optional attribute, the type is float. Defaults to 1e-6.

* @par Outputs
* Three outputs, including:
* @li y: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li rstd: A Tensor. Support dtype: [float32], support format: [ND].
* @li x: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
*/
REG_OP(AddRmsNorm)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(rstd, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(epsilon, Float, 1e-6f)
    .OP_END_FACTORY_REG(AddRmsNorm)

/**
* @brief QuantizeAddLayerNorm operator interface implementation.

* @par Inputs
* @li x1: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li x2: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li gamma: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li beta: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li bias: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li scales: A Tensor. Support dtype: [float32, bfloat16], support format: [ND].
* @li zero_points: A optional Tensor. Support dtype: [float32, bfloat16], support format: [ND].

* @par Attributes
* @li dtype: A required attribute, the type is int. No defaults value.
* @li axis: A optional attribute, the type is int. Defaults to -1.
* @li epsilon: A optional attribute, the type is float. Defaults to 1e-5.
* @li additional_output: A optional attribute, the type is bool. Defaults to false.

* @par Outputs
* @li y: A Tensor. Support dtype: [int8], support format: [ND].
* @li x: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
*/
REG_OP(QuantizeAddLayerNorm)
    .INPUT(x1, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x2, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(gamma, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(beta, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(bias, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(scales, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(zero_points, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, ge::TensorType({DT_INT8, DT_INT8, DT_INT8}))
    .OUTPUT(x, ge::TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(axis, Int, -1)
    .ATTR(epsilon, Float, 1e-5f)
    .ATTR(additional_output, Bool, false)
    .OP_END_FACTORY_REG(QuantizeAddLayerNorm)

/**
* @brief DuaQuantizeAddLayerNorm operator interface implementation.

* @par Inputs
* @li x1: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li x2: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li gamma: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li beta: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li bias: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li scales1: A Tensor. Support dtype: [float32, bfloat16], support format: [ND].
* @li scales2: A Tensor. Support dtype: [float32, bfloat16], support format: [ND].
* @li zero_points1: A optional Tensor. Support dtype: [int8, uint8, bfloat16, int32], support format: [ND].
* @li zero_points2: A optional Tensor. Support dtype: [int8, uint8, bfloat16, int32], support format: [ND].

* @par Attributes
* @li dtype: A required attribute, the type is int. No defaults value.
* @li axis: A optional attribute, the type is float. Defaults to -1.
* @li epsilon: A optional attribute, the type is float. Defaults to 1e-5.
* @li additional_output: A optional attribute, the type is bool. Defaults to false.

* @par Outputs
* @li y1: A Tensor. Support dtype: [int8, uint8, int32], support format: [ND].
* @li y2: A Tensor. Support dtype: [int8, uint8, int32], support format: [ND].
* @li x: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
*/
REG_OP(DuaQuantizeAddLayerNorm)
    .INPUT(x1, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x2, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(gamma, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(beta, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(bias, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(scales1, ge::TensorType({DT_BF16, DT_FLOAT}))
    .INPUT(scales2, ge::TensorType({DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(zero_points1, ge::TensorType({DT_INT8, DT_UINT8, DT_BF16, DT_INT32}))
    .OPTIONAL_INPUT(zero_points2, ge::TensorType({DT_INT8, DT_UINT8, DT_BF16, DT_INT32}))
    .OUTPUT(y1, ge::TensorType({DT_INT8, DT_UINT8, DT_INT32}))
    .OUTPUT(y2, ge::TensorType({DT_INT8, DT_UINT8, DT_INT32}))
    .OUTPUT(x, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(axis, Int, -1)
    .ATTR(epsilon, Float, 1e-5f)
    .ATTR(additional_output, Bool, false)
    .OP_END_FACTORY_REG(DuaQuantizeAddLayerNorm)

/**
* @brief InplaceAddRmsNorm operator interface implementation. \n
*  calculating: x1, x2, gamma \n
*  x2 = x1 + x2 \n
*  rstd = np.rsqrt(np.mean(np.power(x,2), reduce_axis, keepdims=True) + epsilon)) \n
*  x1 = gamma * (x2 * rstd)

* @par Inputs
* Three inputs, including:
* @li x1: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li x2: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li gamma: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].

* @par Attributes:
* epsilon: A optional attribute, the type is float. Defaults to 1e-6.

* @par Outputs
* Three outputs, including:
* @li x1: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li rstd: A Tensor. Support dtype: [float32], support format: [ND].
* @li x2: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
*/
REG_OP(InplaceAddRmsNorm)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(rstd, TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(epsilon, Float, 1e-6f)
    .OP_END_FACTORY_REG(InplaceAddRmsNorm)

/**
* @brief AddRmsNormQuant operator interface implementation.
* Calculating input: x1, x2, gamma, scales1, scales2, zero_points1, zero_points2
* Calculating process:
*  x = x1 + x2
*  rstd = np.rsqrt(np.mean(np.power(x, 2), reduce_axis, keepdims=True) + epsilon))
*  rmsnorm_out = x * rstd * gamma
*  if div_mode is true:
*    y1 = rmsnorm_out / scales1 + zero_points1
*    y2 = rmsnorm_out / scales2 + zero_points2
*  if div_mode is false:
*    y1 = rmsnorm_out * scales1 + zero_points1
*    y2 = rmsnorm_out * scales2 + zero_points2

* @par Inputs
* @li x1: A tensor. Input x1 for the add operation.
*         Support dtype: float32/float16/bfloat16, support format: ND.
* @li x2: A tensor. Input x2 for the add operation.
*         Support dtype: float32/float16/bfloat16, support format: ND.
* @li gamma: A tensor. Describing the weight of the rmsnorm operation.
*            Support dtype: float32/float16/bfloat16, support format: ND.
* @li scales1: A tensor. Describing the weight of the first quant operation.
*              Support dtype: float32/float16/bfloat16, support format: ND.
* @li scales2: A optional input tensor. Describing the weight of the secend quant operation.
*              Support dtype: float32/float16/bfloat16, support format: ND.
* @li zero_points1: An optional input tensor. Describing the bias of the first quant operation.
*                   Support dtype: int32/float32/float16/bfloat16, support format: ND.
* @li zero_points2: An optional input tensor. Describing the bias of the secend quant operation.
*                   Support dtype: int32/float32/float16/bfloat16, support format: ND.

* @par Attributes
* @li axis: An optional attribute. Describing the axis of the quant operation, does not take effect now.
*           The type is int. Defaults to -1.
* @li epsilon: An optional attribute. Describing the epsilon of the rmsnorm operation.
*              The type is float. Defaults to 1e-6.
* @li div_mode: An optional attribute. When div_mode is true, the quant opertaion uses division, otherwise, uses multiplication.
*               The type is bool. Defaults to true.

* @par Outputs
* @li y1: A tensor. Describing the output of the first quant operation.
*                   Support dtype: int8, support format: ND.
* @li y2: A tensor. Describing the output of the second quant operation.
*                   Support dtype: int8, support format: ND.
* @li x: A tensor. Describing the output of the x1+x2 add operation.
*                  Support dtype: float32/float16/bfloat16, support format: ND.
*/
REG_OP(AddRmsNormQuant)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(scales1, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(scales2, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(zero_points1, TensorType({DT_INT32, DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(zero_points2, TensorType({DT_INT32, DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y1, TensorType({DT_INT8}))
    .OUTPUT(y2, TensorType({DT_INT8}))
    .OUTPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(axis, Int, -1)
    .ATTR(epsilon, Float, 1e-6f)
    .ATTR(div_mode, Bool, true)
    .OP_END_FACTORY_REG(AddRmsNormQuant)

/**
* @brief Fused Operator of Add and LayerNorm.

* @par Inputs
* @li x1: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li x2: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li gamma: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li beta: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li bias: A optional input Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].

* @par Attributes
* @li epsilon: A optional attribute, the type is float. Defaults to 1e-5.
* @li additional_output: A optional attribute, the type is bool. Defaults to false.

* @par Outputs
* @li x1: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
* @li mean: A Tensor. Support dtype: [float32], support format: [ND].
* @li rstd: A Tensor. Support dtype: [float32], support format: [ND].
* @li x2: A Tensor. Support dtype: [float32, float16, bfloat16], support format: [ND].
*/
REG_OP(InplaceAddLayerNorm)
    .INPUT(x1, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x2, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(gamma, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(beta, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(x1, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(mean, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(rstd, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(x2, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .ATTR(epsilon, Float, 1e-5f)
    .ATTR(additional_output, Bool, false)
    .OP_END_FACTORY_REG(InplaceAddLayerNorm)

/**
* @brief Fused Operator of AddLayerNorm and Quantize. \n
*  calculating: x1, x2, gamma, beta, bias, scales1, scales2, zero_points1, zero_points1 \n
*  x = x1 + x2 + bias \n
*  rstd = 1 / (sqrt(Var(x) + eps)) \n
*  y = (x - E(x)) * rstd * gamma + beta \n
*  when quant_mode = "static", output out_scales1 and out_scales2 are invalid: \n
*  y1 = round(y / scale1) + zero_points1 \n
*  y2 = round(y / scale2) + zero_points2 \n
*  when quant_mode = "dynamic": \n
*  tmp1 = y * scale1 \n
*  tmp2 = y * scale2 \n
*  out_scales1 = reduce_max(abs(tmp1))/127
*  out_scales2 = reduce_max(abs(tmp2))/127
*  y1 = round(tmp1 / out_scales1) \n
*  y2 = round(tmp2 / out_scales2) \n

* @par Inputs
* @li x1: A tensor for add compute. Support dtype: float32, float16, bfloat16, support format: ND.
* @li x2: A tensor for add compute. Support dtype: float32, float16, bfloat16, support format: ND.
* @li gamma: A tensor for layer norm weight params. Support dtype: float32, float16, bfloat16, support format: ND.
* @li beta: A tensor for layer norm weight params. Support dtype: float32, float16, bfloat16, support format: ND.
* @li bias: An optional input tensor for add compute. Support dtype: float32, float16, bfloat16, support format: ND.
* @li scales1: An optional input tensor for one of quant scale. Support dtype: float32, float16, bfloat16, support
format: ND.
* @li scales2: An optional input tensor for another quant scale. Support dtype: float32, float16, bfloat16, support
format: ND.
* @li zero_points1: An optional input tensor for one of quant offset. Support dtype: float32, float16, bfloat16,
support format: ND.
* @li zero_points2: An optional input tensor for another quant offset. Support dtype: float32, float16, bfloat16,
support format: ND.

* @par Attributes
* @li quant_mode: An optional attribute utilized to select quant mode, can be "dynamic" or "static", the type is
string. Defaults to "dynamic".
* @li epsilon: An optional attribute for layer norm compute, the type is float. Defaults to 1e-5.
* @li additional_output: An optional attribute control whether output x valid or invalid, the type is bool. Defaults
to false, which means x output is invalid.

* @par Outputs
* @li y1: Quantize result 1.
*     A tensor. Support dtype: int8, support format: ND.
* @li y2: Quantize result 2.
*     A tensor. Support dtype: int8, support format: ND.
* @li x: Describing the result of x1 + x2 + bias.
*     A tensor. Support dtype: float32, float16, bfloat16, support format: ND.
* @li out_scales1: Describing the result of dynamic quantize scales.
*     A tensor. Support dtype: float32, support format: ND.
* @li out_scales1: Describing the result of dynamic quantize scales.
*     A tensor. Support dtype: float32, support format: ND.
*/
REG_OP(AddLayerNormQuant)
    .INPUT(x1, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x2, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(gamma, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(beta, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(scales1, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(scales2, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(zero_points1, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(zero_points2, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(y1, ge::TensorType({DT_INT8, DT_INT8, DT_INT8}))
    .OUTPUT(y2, ge::TensorType({DT_INT8, DT_INT8, DT_INT8}))
    .OUTPUT(x, ge::TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(out_scales1, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(out_scales2, ge::TensorType({DT_FLOAT, DT_FLOAT, DT_FLOAT}))
    .ATTR(quant_mode, String, "dynamic")
    .ATTR(epsilon, Float, 1e-5f)
    .ATTR(additional_output, Bool, false)
    .OP_END_FACTORY_REG(AddLayerNormQuant)
}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_NORM_H_