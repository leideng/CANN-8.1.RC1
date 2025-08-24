/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2025. All rights reserved.
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
 * \file experiment_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_EXPERIMENT_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_EXPERIMENT_OPS_H_

#include "graph/operator_reg.h"
namespace ge {
/**
* @brief silentcheckV2. Detect npu silent-fault, return 0/1/2 as result represent normal/L1-error/L2-warn

* @par Inputs:
* @li val: A Tensor, dtype is float16 bfloat16 or float32.
* @li input_grad: A Tensor, dtype is float16 bfloat16 or float32.
* @li sfda: A Tensor, dtype is float32. Must be shape of [3], represent: [pre_val, min_val, max_val]
* @li step: A Tensor, dtype is int64. Must be shape of [1].
*
* @par Attributes:
* @li c_min_steps: An optional int
* @li c_thresh_l1: An optional float
* @li c_coeff_l1: An optional float
* @li c_thresh_l2: An optional float
* @li c_coeff_l2: An optional float
* @li npu_asd_detect: An optional int
*
* @par Outputs:
* @li input_grad: A ref tensor, dtype is float16 bfloat16 or float32.
* @li sfda: A ref tensor, dtype is float32.
* @li step: A ref tensor, dtype is int64.
* @li result: A tensor, dtype is int32.
*/
REG_OP(SilentCheckV2)
    .INPUT(val, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BFLOAT16}))
    .INPUT(input_grad, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BFLOAT16}))
    .INPUT(sfda, TensorType({DT_FLOAT32}))
    .INPUT(step, TensorType({DT_INT64}))
    .OUTPUT(input_grad, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BFLOAT16}))
    .OUTPUT(sfda, TensorType({DT_FLOAT32}))
    .OUTPUT(step, TensorType({DT_INT64}))
    .OUTPUT(result, TensorType({DT_INT32}))
    .ATTR(c_min_steps, Int, 7)
    .ATTR(c_thresh_l1, Float, 1000000)
    .ATTR(c_coeff_l1, Float, 100000)
    .ATTR(c_thresh_l2, Float, 10000)
    .ATTR(c_coeff_l2, Float, 5000)
    .ATTR(npu_asd_detect, Int, 1)
    .OP_END_FACTORY_REG(SilentCheckV2)
/**
* @brief silentcheckV3. Detect npu silent-fault, return 0/1/2 as result represent normal/L1-error/L2-warn
* @par Inputs:
* @li val: A Tensor, dtype is float16 bfloat16 or float32. Must be shape of [0].
* @li max: A Tensor, dtype is float16 bfloat16 or float32. Must be shape of [0].
* @li avg: A Tensor, dtype is float16 bfloat16 or float32. Must be shape of [0].
* @li input_grad: A Tensor, dtype is float16 bfloat16 or float32.
* @li step: A Tensor, dtype is int64. Must be shape of [1].
* @li dst_size: A Tensor, dtype is int64.
* @li dst_stride: A Tensor, dtype is int64.
* @li dst_offset: A Tensor, dtype is int64.
*
* @par Attributes:
* @li c_thresh_l1: An optional float
* @li c_thresh_l2: An optional float
* @li beta1: An optional float
* @li npu_asd_detect: An optional int
*
* @par Outputs:
* @li avg: A ref tensor, dtype is float16 bfloat16 or float32.
* @li input_grad: A ref tensor, dtype is float16 bfloat16 or float32.
* @li step: A ref tensor, dtype is int64.
* @li result: A tensor, dtype is int32.
*/
REG_OP(SilentCheckV3)
    .INPUT(val, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BFLOAT16}))
    .INPUT(max, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BFLOAT16}))
    .INPUT(avg, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BFLOAT16}))
    .INPUT(input_grad, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BFLOAT16}))
    .INPUT(step, TensorType({DT_INT64}))
    .INPUT(dst_size, TensorType({DT_INT64}))
    .INPUT(dst_stride, TensorType({DT_INT64}))
    .INPUT(dst_offset, TensorType({DT_INT64}))
    .OUTPUT(avg, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BFLOAT16}))
    .OUTPUT(input_grad, TensorType({DT_FLOAT32, DT_FLOAT16, DT_BFLOAT16}))
    .OUTPUT(step, TensorType({DT_INT64}))
    .OUTPUT(result, TensorType({DT_INT32}))
    .ATTR(c_thresh_l1, Float, 1000000)
    .ATTR(c_thresh_l2, Float, 10000)
    .ATTR(beta1, Float, 0.99)
    .ATTR(npu_asd_detect, Int, 1)
    .OP_END_FACTORY_REG(SilentCheckV3)
/**
* @brief Updates "var" according to the AdamW algorithm.
*
* @attention Constraints:
*  The input tensors must have the same shape.*
*
* @par Inputs:
* @li var: A Tensor of ND,
*     dtype is float16 bfloat16 or float32.
*     Should be from a Variable().
* @li m: A Tensor of ND,
*     dtype is float16 bfloat16 or float32.
*     Should be from a Variable().
* @li v: A Tensor of ND,
*     dtype is float16 bfloat16 or float32.
*     Should be from a Variable().
* @li beta1_power: A scalar of the same type as "var", value is beta1**(step-1).
* @li beta2_power: A scalar of the same type as "var", value is beta2**(step-1).
* @li lr: learning_rate. A scalar of the same type as "var".
* @li weight_decay: learning_rate. A scalar of the same type as "var".
* @li beta1: A scalar of the same type as "var".
* @li beta2: A scalar of the same type as "var".
* @li epsilon: A scalar of the same type as "var".
* @li grad: A Tensor of the same type as "var", for the gradient,
*     dtype is float16 bfloat16 or float32.
* @li max_grad_norm: A mutable Tensor of the same type as "var", an optional input,
*     dtype is float16 bfloat16 or float32.
*     Should be from a Variable().
*
* @par Attributes:
* @li amsgrad: An optional bool. Defaults to "False", only support "False".
* @li maximize: An optional bool. Defaults to "False".
*
* @par Outputs:
* @li var: A mutable tensor. Has the same type as input "var".
* @li m: A mutable tensor. Has the same type as input "m".
* @li v: A mutable tensor. Has the same type as input "v". \n
*/
REG_OP(ApplyAdamW)
    .INPUT(var, TensorType::NumberType())
    .INPUT(m, TensorType::NumberType())
    .INPUT(v, TensorType::NumberType())
    .INPUT(beta1_power, TensorType::NumberType())
    .INPUT(beta2_power, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(weight_decay, TensorType::NumberType())
    .INPUT(beta1, TensorType::NumberType())
    .INPUT(beta2, TensorType::NumberType())
    .INPUT(epsilon, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OPTIONAL_INPUT(max_grad_norm, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(m, TensorType::NumberType())
    .OUTPUT(v, TensorType::NumberType())
    .ATTR(amsgrad, Bool, false)
    .ATTR(maximize, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdamW)

/**
* @brief Updates "var" "m" "v" and "max_grad_norm" according to the AdamWV2 algorithm.
*
* @attention Constraints:
*  The input tensors must have the same shape, except for the step. The shape of step must be (1,).
*  When the data types of the input tensors var,m,v,grad,max_grad_norm are the same, the data type can be
*  float16,bfloat16 or float32.
*  The data types of the input tensors var,m and v must be the same.For example,if var tensor is float16,
*  the data types of m and v must also be float16.
*  The data tytpes of the input tensors grad and max_grad_norm must be the same.For example,if grad tensor is float16,
*  the data types of max_grad_norm must also be float16.
*  When data type of the input tensor var,m and v are different with input tensor grad and max_grad_norm,
*  the data types of var,m and v can only be float32,and the data type of grad and max_grad_norm tensor
*  can only be float16 or bfloat16.
*
* @par Inputs:
* @li var: A Tensor, dtype is float16 bfloat16 or float32, default is ND.
* @li m: A Tensor of the same type as "var", default is ND.
* @li v: A Tensor of the same type as "var", default is ND.
* @li grad: A Tensor, dtype is float16 bfloat16 or float32, for the gradient, default is ND.
* @li step: A Tensor, dtype is float32 or int64.
* @li max_grad_norm: An optional Tensor of the same type as "grad", default is ND.
*
* @par Attributes:
* @li lr: A required float, default is 0.1
* @li beta1: A required float, default is 0.1
* @li beta2: A required float,default is 0.1
* @li weight_decay: A required float, default is 0.1
* @li eps: A required float, default is 0.1
* @li amsgrad: A required bool, default is false.
* @li maximize: A required bool, default is false.\n
*/
REG_OP(ApplyAdamWV2)
    .INPUT(var, TensorType::FLOAT())
    .INPUT(m, TensorType::FLOAT())
    .INPUT(v, TensorType::FLOAT())
    .INPUT(grad, TensorType::FLOAT())
    .INPUT(step, TensorType({DT_FLOAT, DT_INT64}))
    .OPTIONAL_INPUT(max_grad_norm, TensorType::FLOAT())
    .ATTR(lr, Float, 0.1f)
    .ATTR(beta1, Float, 0.1f)
    .ATTR(beta2, Float, 0.1f)
    .ATTR(weight_decay, Float, 0.1f)
    .ATTR(eps, Float, 0.1f)
    .ATTR(amsgrad, Bool, false)
    .ATTR(maximize, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdamWV2)

/**
* @brief Calculate SQ distance.
*
* @par Inputs:
* @li ivf: A Tensor, dtype is uint8.
* @li query: A Tensor, dtype is float16 or float32.
* @li bucket_list: A Tensor, dtype is int32 or int64.
* @li bucket_limits: A Tensor, dtype is int32 or int64.
* @li bucket_offsets: A Tensor, dtype is int32 or int64.
* @li vmin: A Tensor, dtype is float16 or float32.
* @li vdiff: A Tensor, dtype is float16 or float32. \n
*
* @par Outputs:
* @li actual_count: A Tensor, dtype is int32 or int64, the actual number of
* sq_distance.
* @li sq_distance: A Tensor, dtype is float16 or float32.
* @li grouped_extreme_distance: A Tensor, dtype is float16 or float32, the
* extremum in each group of sq_distance.
* @li sq_ivf: A Tensor, dtype is int32 or int64.
* @li sq_index: A Tensor, dtype is int32 or int64. \n
*
* @par Attributes:
* @li total_limit: A required int, indicates the max length of the output
* sq_distance.
* @li group_size: An optional int, indicates the group size of the extremum.
* Defaults to 64.
* @li extreme_mode: A optional int, indicates the type of extremum, 0 means
* minimum, and 1 means maximum. Defaults to 0.\n
*
*/
REG_OP(ScanSQCodes)
    .INPUT(ivf, TensorType({DT_UINT8}))
    .INPUT(query, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(bucket_list, TensorType({DT_INT32, DT_INT64}))
    .INPUT(bucket_limits, TensorType({DT_INT32, DT_INT64}))
    .INPUT(bucket_offsets, TensorType({DT_INT32, DT_INT64}))
    .INPUT(vmin, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(vdiff, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(actual_count, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(sq_distance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(grouped_extreme_distance, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(sq_ivf, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(sq_index, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(total_limit, Int)
    .ATTR(group_size, Int, 64)
    .ATTR(extreme_mode, Int, 0)
    .OP_END_FACTORY_REG(ScanSQCodes)

/**
* @brief Performs non-maximum suppression (NMS) on the rotated boxes according
* to their intersection-over-union (IoU). Rotated NMS interatively removes lower
* scoring rotated boxes which have an IoU greater than iou_threshold with
* another (higher scoring) rotated box.

* @par Inputs:
* Three inputs, including:
* @li boxes: A 2D Tensor of float16 or float32 with shape (N, 5). Rotated boxes to
* perform NMS on. They are expected to be in (x1, y1, x2, y2, angle_degress) format.
* @li scores: A 1D Tensor of float16 or float32 with shape (N). Scores for each one of
* the rotated boxes.
* @li labels: A 1D Tensor of int32 or int64 with shape (N). Labels for each one of
* the rotated boxes.

* @par Attributes:
* iou_threshold: A required float attribute. Discards all overlapping rotated
* boxes with IoU < iou_threshold.

* @par Outputs:
* Two outputs, including:
* @li selected_detections: A 2D Tensor of float16 or float32 with shape (N, 5).
* The selected boxes that kept by Rotated NMS, sorted in decreasing order of scores.
* @li keep_indices: A 1D Tensor of int32 or int64 with shape (N). The indices of
* selected_detections.

* @attention Constraints:
* Currently, the tensor type of input (boxes, scores) only support float.
* The tensor type of keep_indices only support int32.
*/
REG_OP(RotatedNMS)
    .INPUT(boxes, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(scores, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(labels, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(selected_detections, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(keep_indices, TensorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(iou_threshold, Float)
    .ATTR(is_angle, Bool, true)
    .OP_END_FACTORY_REG(RotatedNMS)

/**
* @brief According to the indices, return the value.

* @par Inputs:
* Four inputs, including:
* @li x: A ND Tensor.
* @li indexed_sizes: A 1D Tensor of int64 with shape (N). Sizes for each one of the indexed data.
* @li indexed_strides: A 1D Tensor of int64 with shape (N). Strides for each one of the indexed data.
* @li indices: Dynamic input. A ND Tensor of int64. return the value according to the indices.

* @par Outputs:
* y: The indexed output tensor. Has the same type and format as input "x".
*/
REG_OP(Index)
    .INPUT(x, TensorType::BasicType())
    .INPUT(indexed_sizes, TensorType({DT_INT64}))
    .INPUT(indexed_strides, TensorType({DT_INT64}))
    .DYNAMIC_INPUT(indices, TensorType({DT_INT64, DT_INT32}))
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(Index)

/**
* @brief According to the index number of indexes, replace the value
* corresponding to X with the value.

* @par Inputs:
* Five inputs, including:
* @li x: A ND Tensor.
* @li value: A Tensor of the same type as "x".
* @li indexed_sizes: A 1D Tensor of int64 with shape (N). Sizes for each one of the indexed data.
* @li indexed_strides: A 1D Tensor of int64 with shape (N). Strides for each one of the indexed data.
* @li indices: Dynamic input. A Tensor of the indices.

* @par Attributes:
* @li accumulate: Does it support self accumulation. Defaults to false.

* @par Outputs:
* @li x: A Tensor.

* @par Third-party framework compatibility
* Compatible with the Pytorch operator index_put.

* @par Restrictions:
* Warning:THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(IndexPutV2)
    .INPUT(x, TensorType::BasicType())
    .INPUT(value, TensorType::BasicType())
    .INPUT(indexed_sizes, TensorType({DT_INT64}))
    .INPUT(indexed_strides, TensorType({DT_INT64}))
    .DYNAMIC_INPUT(indices, TensorType({DT_INT64, DT_INT32}))
    .OUTPUT(x, TensorType::BasicType())
    .ATTR(accumulate, Bool, false)
    .OP_END_FACTORY_REG(IndexPutV2)

/**
* @brief Performs average pooling on the input. Used in the combination of conv + avgpoolupdate to replace avgpool
* @par Inputs:
* x1: Output of upstream Conv2d. A tensor of type float16, float32.
* x2: Input feature map of upstream Conv2d. A tensor of type int8, float16, float32.

* @par Attributes:
* @li ksize: A required list of 4 ints, specifying the size (N, C, H, and W) of the sliding window,
* where N = C = 1, and H and W are positive integers within the range [1, 255].
* @li strides: A required list of 4 ints, specifying the stride of the sliding window.
* The strides of the N and C dimensions are 1.
* The strides of the H and W dimensions are positive integers within the range [1, 63].
* @li padding_mode: A required string, specifying the padding algorithm,
* either "VALID", "SAME" and "CALCULATED".
* With "SAME" means that the outputs will have the same spatial dimensions as its inputs.
* With "VALID" means no padding.
* @li pads: Pad value when padding_mode is "CALCULATED".
* @li data_format: An optional string, specifying the data format of "ksize" and "strides",
* either "NCHW", or "NHWC" (default).
* @li ceil_mode: Use ceil or floor to calculate the output size when padding_mode is "CALCULATED".
* @li exclusive: Ignore padding area or not when calculating average.

* @par Outputs:
* y: The average pooled output tensor. Has the same type and format as input "x1".

* @attention Constraints:
* @li Only single input and single output are supported.
* @li "ksize_H" and "ksize_W" are positive integers within the range [1, 255]. ksize_H * ksize_W < 256
* @li Due to instruction restrictions,
* the values of "strides_h" and "strides_w" are positive integers within the range [1, 63].
* @par Third-party framework compatibility
* Compatible with the TensorFlow/Pytorch/Onnx operator AvgPoolV2.
*/
REG_OP(AvgPoolUpdate)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(x2, TensorType({DA_INT4, DT_INT8, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(ksize, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .ATTR(padding_mode, String, "CALCULATED")
    .ATTR(pads, ListInt, {0, 0, 0, 0})
    .ATTR(data_format, String, "NHWC")
    .ATTR(ceil_mode, Bool, false)
    .ATTR(exclusive, Bool, true)
    .OP_END_FACTORY_REG(AvgPoolUpdate)

/**
* @brief YUVToRGB

* @par Inputs:
* @li x: A 4-D uint8 Tensor.
*        Must set the format, supported format list ["NYUV"].
* @li matrix: A 1-D float tensor of 2x3x3 elements

* @par Outputs:
* @li y: A 4-D uint8 Tensor.
*        Must set the format, supported format list ["NCHW, NHWC"].

* @par Attributes:
* @li matrix_type: An Int attr, Defaults to 0.
*                  support list [ 0: CSC_MATRIX_BT601_WIDE,
*                                 1: CSC_MATRIX_BT601_NARROW,
*                                 2: CSC_MATRIX_BT709_WIDE,
*                                 3: CSC_MATRIX_BT709_NARROW,
*                                 4: CSC_MATRIX_BT2020_WIDE,
*                                 5: CSC_MATRIX_BT2020_NARROW,
*                                 6: CSC_MATRIX_USR_DEFINE ]
* @li rb_swap: An Int attr, Defaults to 0.
*              support list [ 0: RGB, 1: BGR ]

* @attention Constraints:
* @li Only support in dvpp
*/

REG_OP(YUVToRGB)
    .INPUT(x, TensorType({DT_UINT8}))
    .OPTIONAL_INPUT(matrix, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_UINT8}))
    .ATTR(matrix_type, Int, 0)
    .ATTR(rb_swap, Int, 0)
    .OP_END_FACTORY_REG(YUVToRGB)

/**
* @brief DecodeJpegPre

* @par Inputs:
* @li contents: A Tensor of type string. 0-D. The JPEG-encoded image.

* @par Outputs:
* @li dvpp_support: indicates if the dvpp support this jpeg image decode.

* @par Attributes:
* @li w_range: An required listInt contains width [min, max].
* @li h_range: An required listInt contains height [min, max].

* @attention Constraints:
* @li Only support in dvpp
*/

REG_OP(DecodeJpegPre)
    .INPUT(contents, TensorType({DT_STRING}))
    .OUTPUT(dvpp_support, BOOL)
    .REQUIRED_ATTR(w_range, ListInt)
    .REQUIRED_ATTR(h_range, ListInt)
    .OP_END_FACTORY_REG(DecodeJpegPre)

/**
* @brief Computes the output as scale * (x + bias) if x+bias > 0 and scale * negative_slope * (x+bias)
* if x+bias <= 0 . \n

* @par Inputs:
* Two input:
* x: A Tensor. Must be one of the following types: float32, float16, double.
* bias: A Tensor. Must be one of the following types: float32, float16, double.
*
* @par Attributes:
* negative_slope: A float32. Defaults to "0.2".
* sacle: A float32. Defaults to "2**0.5".
*
* @par Outputs:
* y: A Tensor. Has the same type as "x".
* @par Third-party framework compatibility
* Compatible with the mmcv operator FusedBiasLeakyrelu.
*/
REG_OP(FusedBiasLeakyRelu)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE}))
    .INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE}))
    .ATTR(negative_slope, Float, 0.2f)
    .ATTR(scale, Float, 1.414213562373f)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE}))
    .OP_END_FACTORY_REG(FusedBiasLeakyRelu)

/**
* @brief Computes the output as scale * gradients if features > 0 and
* negative_slope * gradients * scale if features <= 0 . \n

* @par Inputs:
* Two inputs, including:
* @li y_grad: A Tensor. Must be one of the following types: float16, float32, double.
* @li features: A Tensor. Has the same type as "gradients" . \n

* @par Attributes:
* negative_slope: A float32. Defaults to "0.2" . \n
* scale : A float32. Defaults to "2**0.5"

* @par Outputs:
* x_grad: A Tensor. Has the same type as "y_grad" . \n

* @par Third-party framework compatibility
* Compatible with the MMCV operator FusedBiasLeakyReluGrad.
*/
REG_OP(FusedBiasLeakyReluGrad)
    .INPUT(y_grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .INPUT(features, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .ATTR(negative_slope, Float, 0.2f)
    .ATTR(scale, Float, 1.414213562373f)
    .OUTPUT(x_grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(FusedBiasLeakyReluGrad)

/**
* @brief multi-scale deformable attention grad.
*
* @par Inputs:
* @li value: A Tensor. Must be one of the following types: float32.
* @li value_spatial_shapes: A Tensor. Must be one of the following types: int32.
* @li value_level_start_index: A Tensor. Must be one of the following types: int32.
* @li sampling_locations: A Tensor. Must be one of the following types: float32.
* @li attention_weights: A Tensor. Must be one of the following types: float32.
* @li grad_output: A Tensor. Must be one of the following types: float32.
*
* @par Outputs:
* grad_value: A Tensor. Must be one of the following types: float32.
* grad_sampling_locations: A Tensor. Must be one of the following types: float32.
* grad_attention_weights: A Tensor. Must be one of the following types: float32.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(MultiScaleDeformableAttentionGrad)
    .INPUT(value, TensorType({DT_FLOAT}))
    .INPUT(value_spatial_shapes, TensorType({DT_INT32}))
    .INPUT(value_level_start_index, TensorType({DT_INT32}))
    .INPUT(sampling_locations, TensorType({DT_FLOAT}))
    .INPUT(attention_weights, TensorType({DT_FLOAT}))
    .INPUT(grad_output, TensorType({DT_FLOAT}))
    .OUTPUT(grad_value, TensorType({DT_FLOAT}))
    .OUTPUT(grad_sampling_locations, TensorType({DT_FLOAT}))
    .OUTPUT(grad_attention_weights, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(MultiScaleDeformableAttentionGrad)

/**
* @brief Set initial values for memory of sizes list . \n

* @par Attributes:
* @li sizes: sizes of workspaces. \n
* @li dtypes: data types of initial values. \n
* @li values_int: integer values to be set. \n
* @li values_float: float values to be set. \n

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL.  Please do not use.
*/
REG_OP(MemSet)
    .REQUIRED_ATTR(sizes, ListInt)
    .ATTR(dtypes, ListType, {})
    .ATTR(values_int, ListInt, {})
    .ATTR(values_float, ListFloat, {})
    .OP_END_FACTORY_REG(MemSet)

/**
* @brief Performs the backpropagation of DeformableRoiPool for training scenarios . \n

* @par Inputs:
* Four inputs, including:
* @li grad_output: A 5HD gradient input of type float32
* @li feature_map: A 5HD Tensor of type float32.
* @li rois: ROI position. A 2D Tensor of float32 with shape (N, 5). "N" indicates the number of ROIs,
* the value "5" indicates the indexes of images where the ROIs are located, "x0", "x1", "y0" and "y1".
* @li offset: An optional 5HD Tensor input, specifying the offset of sampled points . \n

* @par Attributes:
* Four attributes, including:
* @li output_size: A required list of 2 ints, obtained based on the shape of "output" of DeformableRoiPool.
* @li spatial_scale: A optional attribute of type float, specifying the scaling ratio of "feature_map"
* to the original image.
* @li sample_ratio: An optional attribute of type int, specifying the horizontal and vertical sampling
* frequency of each output.
* If this attribute is set to "0", the sampling frequency is equal to the rounded up value of "rois",
* which is a floating point number. Defaults to "0".
* @li gamma: An optional attribute of type float, specfying the scaling factor of offset . \n

* @par Outputs:
* @li grad_fm: Gradient added to input "features". Has the same 5HD shape as input "features".
* @li grad_offset: Gradient added to input "offset". Has the same 4D shape as input "offset".
*/
REG_OP(DeformableRoiPoolGrad)
    .INPUT(grad, TensorType({DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(rois, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OUTPUT(grad_x, TensorType({DT_FLOAT}))
    .OUTPUT(grad_offset, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(output_size, ListInt)
    .ATTR(spatial_scale, Float, 1.0)
    .ATTR(sampling_ratio, Int, 0)
    .ATTR(gamma, Float, 0.1f)
    .OP_END_FACTORY_REG(DeformableRoiPoolGrad)

/**
* @brief find an optimal n for shift-n. \n

* @par Inputs:
* @li x: A Tensor. indicates the output of quantizable layers.
* @li scale_d: A Tensor, one number. indicates the scale of data.
* @li scale_w: A Tensor, must be one number or the same size as dim-C when x is NHWC/NCHW.
*              indicates the scale of weight. \n

* @par Outputs:
* @li n: A Tensor, has the same shape as scale_w. indicates the optimal n. \n
*/
REG_OP(SearchN)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(scale_d, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(scale_w, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(n, TensorType({DT_INT8}))
    .OP_END_FACTORY_REG(SearchN)

/**
* @brief The operator generates three assist matrixs which will be used in AdaptiveAvgPool. \n

* @par Input:
* input_size: A Tensor of type int64.  \n
* output_size: A Tensor of type int64.  \n

* @par Outputs:
* three inputs, including:
* @li left_matrix: A Tensor of type float32.  \n
* @li right_matrix: A Tensor of type float32.  \n
* @li weight_matrix: A Tensor of type float32.  \n
*/
REG_OP(AdaptiveAvgPoolAssistMatrix)
    .INPUT(input_size, TensorType({DT_INT64, DT_INT32}))
    .INPUT(output_size, TensorType({DT_INT64, DT_INT32}))
    .OUTPUT(left_matrix, TensorType({DT_FLOAT}))
    .OUTPUT(right_matrix, TensorType({DT_FLOAT}))
    .OUTPUT(weight_matrix, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(AdaptiveAvgPoolAssistMatrix)

/**
* @brief The operator generates three assist matrixs which will be used in AdaptiveAvgPool2d. \n

* @par Input:
* input_size: A Tensor of type int64.  \n

* @par Outputs:
* three inputs, including:
* @li left_matrix: A Tensor of type float32.  \n
* @li right_matrix: A Tensor of type float32.  \n
* @li weight_matrix: A Tensor of type float32.  \n

* @par Attributes:
* output_size: A required attribute.  \n
*/
REG_OP(AdaptiveAvgPool2dAssistMatrix)
    .INPUT(input_size, TensorType({DT_INT64}))
    .OUTPUT(left_matrix, TensorType({DT_FLOAT}))
    .OUTPUT(right_matrix, TensorType({DT_FLOAT}))
    .OUTPUT(weight_matrix, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(output_size, ListInt)
    .OP_END_FACTORY_REG(AdaptiveAvgPool2dAssistMatrix)

/**
* @brief Compute correct bounding box.

* @par Inputs:
* Three inputs, including:
* @li x: A 5D Tensor of type float16 with shape (N, na, no, H, W), na indicates the number of anchors,
* no indicates the number of outputs per anchor, including [xywh, class_num, conf_score].
* @li grid: A 5D Tensor of type float16 with shape (1, na, 2, H, W) for V3/V5 and (1, 1, 2, H, W) for V7,
* the value "2" indicates offsets of coordinates.
* @li anchor_grid: A 5D Tensor of type float16 with shape (1, na, 2, H, W) for V3/V5 and (1, 1, 2, 1, 1) for V7,
* the value "2" indicates anchors relative to the original image.

* @par Attributes:
* @li stride: A required int32, scale for each box.
* @li yolo_version: A required string, specifying the YOLO version, optional [V3, V5, V7].

* @par Outputs:
* @li y: A 5D Tensor of type float16 with shape (N, na, no, H, W), same as the input x.

* @par attention Constraints:
* @li This operator applies to YOLO V3, V5 and V7 networks.
* @par Third-party framework compatibility
* It is a custom operator.
*/
REG_OP(CorrectBBox)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(grid, TensorType({DT_FLOAT16}))
    .INPUT(anchor_grid, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .REQUIRED_ATTR(stride, Int)
    .REQUIRED_ATTR(yolo_version, String)
    .OP_END_FACTORY_REG(CorrectBBox)

/**
* @brief Fusion ops for splitvd rope quantize scatter.

* @par Inputs:
* eight inputs, including:
* @li qkv: A 3D Tensor of type float16 with shape (B, S, H), H is (Nq+Nkv+Nkv)*D, format support ND
* @li cos: A 4D Tensor of type float16 with shape (B, S, 1, D), shape must same with k, format support ND
* @li sin: A 4D Tensor of type float16 with shape (B, S, 1, D), shape must same with k, format support ND
* @li quant_scale: A 1D Tensor of type float with shape (D), shape D must same with k, format support ND
* @li k_cache: A 4D Tensor of type int8 with shape (B, S, Nkv, D), shape B/N/D must same with k,
* S must large than k, format support ND
* @li v_cache: A 4D Tensor of type int8 with shape (B, S, Nkv, D), shape B/N/D must same with v,
* S must large than k, format support ND
* @li indice: A 1D Tensor of type int32 with shape (B), shape must same with qkv, format support ND

* @par Attributes:
* @li size_splits: list to split qkv input tensor to q/k/v, default is null.
* @li layout: qkv input tensor layout, like BSND/BNSD, default is BSND.
* @li kv_output: control origin k/v output or not, default is false.

* @par Outputs:
* @li q: A 4D Tensor of type float16 with shape (B, S, Nq, D), split from qkv, format support ND
* @li k: A 4D Tensor of type float16 with shape (B, S, Nkv, D), split from qkv, N must same with v, format support ND
* @li v: A 4D Tensor of type float16 with shape (B, S, Nkv, D), split from qkv, N must same with k, format support ND
* @li k_cache: A 4D Tensor of type int8 with shape (B, S, Nkv, D), shape B/N/D must same with k, S must large than k,
* format support ND
* @li v_cache: A 4D Tensor of type int8 with shape (B, S, Nkv, D), shape B/N/D must same with v, S must large than v,
* format support ND
* It is a custom operator.
*/
REG_OP(RopeQuantKvcache)
    .INPUT(qkv, TensorType({DT_FLOAT16}))
    .INPUT(cos, TensorType({DT_FLOAT16}))
    .INPUT(sin, TensorType({DT_FLOAT16}))
    .INPUT(quant_scale, TensorType({DT_FLOAT32}))
    .INPUT(quant_offset, TensorType({DT_INT32}))
    .INPUT(k_cache, TensorType({DT_INT8}))
    .INPUT(v_cache, TensorType({DT_INT8}))
    .INPUT(indice, TensorType({DT_INT32}))
    .OUTPUT(q, TensorType({DT_FLOAT16}))
    .OUTPUT(k, TensorType({DT_FLOAT16}))
    .OUTPUT(v, TensorType({DT_FLOAT16}))
    .OUTPUT(k_cache, TensorType({DT_INT8}))
    .OUTPUT(v_cache, TensorType({DT_INT8}))
    .ATTR(size_splits, ListInt, {})
    .ATTR(layout, String, "BSND")
    .ATTR(kv_output, Bool, false)
    .OP_END_FACTORY_REG(RopeQuantKvcache)

/**
* @brief Obtains the ROI feature matrix from the feature map. It is a customized FasterRcnn operator . \n

* @par Inputs:
* Three inputs, including:
* @li features: A 5HD Tensor of type float32 or float16.
* @li rois: ROI position. A 2D Tensor of float32 or float16 with shape (N, 5). "N" indicates the number of ROIs,
*     the value "5" indicates the indexes of images where the ROIs are located, "x0", "y0", "x1", and "y1".
* @li offset: An optional input of type float32 or float16, offset of height and width defaults to a Tensor of zero . \n

* @par Attributes:
* @li spatial_scale: A required attribute of type float32, specifying the scaling ratio of "features"
*     to the original image.
* @li pooled_height: A required attribute of type int32, specifying the H dimension.
* @li pooled_width: A required attribute of type int32, specifying the W dimension.
* @li sampling_ratio: An optional attribute of type int32, specifying the horizontal and vertical sampling frequency
*     of each output. If this attribute is set to "0",
* the sampling frequency is equal to the rounded up value of "rois", which is a floating point number. Defaults to "0".
* @li gamma: An optional attribute of type float32. Defaults to "0.1" . \n
* @par Outputs:
* output: Outputs the feature sample of each ROI position. The format is 5HD Tensor of type float32 or float16.
  The axis N is the number of input ROIs. Axes H, W, and C are consistent
* with the values of "pooled_height",
* "pooled_width", and "features", respectively.
*/
REG_OP(DeformableRoiPool)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(rois, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(spatial_scale, Float, 1.0)
    .REQUIRED_ATTR(output_size, ListInt)
    .ATTR(sampling_ratio, Int, 0)
    .ATTR(gamma, Float, 0.1f)
    .OP_END_FACTORY_REG(DeformableRoiPool)

/**
 * @brief Generate the attention map of Point-wise Spatial Attention(PSA) \n

 * @par Inputs:
 * x: A Tensor of BasicType that indicates the global attention map from upstream computing. \n

 * @par Outputs:
 * y: A Tensor of BasicType that indicates the generated pixel-wise global attention map. \n

 * @par Attributes:
 * @li psa_type: An Int value of 1 or 2 that indicates the method used to generate pixel-wise global attention map.
 * @li num: An Int value that indicates the batch_size of input x.
 * @li h_feature: An Int value that indicates the hight of input feature map.
 * @li w_feature: An Int value that indicates the width of input feature map.
 * @li h_mask: An Int value that indicates the hight of the over-completed map.
 * @li w_mask: An Int value that indicates the width of the over-completed map.
 * @li half_h_mask: An Int value that indicates half of the hight of input feature map.
 * @li half_w_mask: An Int value that indicates half of the width of the over-completed map. \n

 * @par Third-party framework compatibility
 * Compatible with the mmcv operator PSAMask.\n
 */
REG_OP(PSAMask)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(psa_type, Int)
    .REQUIRED_ATTR(num, Int)
    .REQUIRED_ATTR(h_feature, Int)
    .REQUIRED_ATTR(w_feature, Int)
    .REQUIRED_ATTR(h_mask, Int)
    .REQUIRED_ATTR(w_mask, Int)
    .REQUIRED_ATTR(half_h_mask, Int)
    .REQUIRED_ATTR(half_w_mask, Int)
    .OP_END_FACTORY_REG(PSAMask)

/**
 * @brief Calculate the gradient of operator PSAMask \n

 * @par Inputs:
 * y_grad: A Tensor of BasicType that indicates the passed gradient. \n

 * @par Outputs:
 * x_grad: A Tensor of BasicType that indicates the calculated gradient. \n

 * @par Attributes:
 * @li psa_type: An Int value of 1 or 2 that indicates the method used to generate pixel-wise global attention map.
 * @li num: An Int value that indicates the batch_size of input x.
 * @li h_feature: An Int value that indicates the hight of input feature map.
 * @li w_feature: An Int value that indicates the width of input feature map.
 * @li h_mask: An Int value that indicates the hight of the over-completed map.
 * @li w_mask: An Int value that indicates the width of the over-completed map.
 * @li half_h_mask: An Int value that indicates half of the hight of input feature map.
 * @li half_w_mask: An Int value that indicates half of the width of the over-completed map. \n

 * @par Third-party framework compatibility
 * Compatible with the mmcv operator PSAMask.\n
 */
REG_OP(PSAMaskGrad)
    .INPUT(y_grad, TensorType::BasicType())
    .OUTPUT(x_grad, TensorType::BasicType())
    .REQUIRED_ATTR(psa_type, Int)
    .REQUIRED_ATTR(num, Int)
    .REQUIRED_ATTR(h_feature, Int)
    .REQUIRED_ATTR(w_feature, Int)
    .REQUIRED_ATTR(h_mask, Int)
    .REQUIRED_ATTR(w_mask, Int)
    .REQUIRED_ATTR(half_h_mask, Int)
    .REQUIRED_ATTR(half_w_mask, Int)
    .OP_END_FACTORY_REG(PSAMaskGrad)

/**
* @brief Find nearby points in spherical space or spherical layer. \n

* @par Inputs:
* Two inputs, including:
* @li xyz: A 3D Tensor of type float16 or float32, xyz coordinates of the features.
* @li center_xyz: A 3D Tensor of type float16 or float32. centers coordinates of the ball query. \n

* @par Attributes:
* @li min_radius: A required float, minimum radius of the balls.
* @li max_radius: A required float, maximum radius of the balls.
* @li sample_num: A required int, maximum number of features in the balls. \n

* @par Outputs:
* One outputs:
* @li idx: A 3D(B, M, sample_num) Tensor of type int32 with the indices of the features that form the query balls. \n

* @par Third-party framework compatibility
* Compatible with the MMCV operator BallQuery(BallQuery branch).
*/
REG_OP(BallQuery)
    .INPUT(xyz, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(center_xyz, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(idx, TensorType({DT_INT32}))
    .REQUIRED_ATTR(min_radius, Float)
    .REQUIRED_ATTR(max_radius, Float)
    .REQUIRED_ATTR(sample_num, Int)
    .OP_END_FACTORY_REG(BallQuery)

/**
* @brief Find nearby points in spherical space. \n

* @par Inputs:
* Four inputs, including:
* @li xyz: A 2D Tensor of type float16 or float32, xyz coordinates of the features.
* @li center_xyz: A 2D Tensor of type float16 or float32. Centers coordinates of the ball query.
* @li xyz_batch_cnt: A 1D Tensor of type int32 or int64, Stacked input xyz coordinates nums in
     each batch, just like (N1, N2, ...).
* @li center_xyz_batch_cnt: A 1D Tensor of type int32 or int64. Stacked input centers coordinates nums in
     each batch, just like (M1, M2, ...). \n

* @par Attributes:
* @li max_radius: A required float, maximum radius of the balls.
* @li sample_num: A required int, maximum number of features in the balls. \n

* @par Outputs:
* One outputs:
* @li idx: A 2D(M, sample_num) Tensor of type int32 with the indices of the features that form the query balls. \n

* @par Third-party framework compatibility
* Compatible with the MMCV operator BallQuery(StackBallQuery branch).
*/
REG_OP(StackBallQuery)
    .INPUT(xyz, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(center_xyz, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(xyz_batch_cnt, TensorType({DT_INT32, DT_INT64}))
    .INPUT(center_xyz_batch_cnt, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(idx, TensorType({DT_INT32}))
    .REQUIRED_ATTR(max_radius, Float)
    .REQUIRED_ATTR(sample_num, Int)
    .OP_END_FACTORY_REG(StackBallQuery)

/**
* @brief Group the points in the point cloud according to the group they belong to. \n

* @par Inputs:
* Four inputs, including:
* @li features:  Tensor of features to group, input shape is (N1 + N2 ..., C).
* @li features_batch_cnt:  Input features nums in each batch, just like (N1, N2, ...). Defaults to None.
* @li indices: The indices of features to group with, input shape is (M1 + M2 ..., nsample).
* @li indices_batch_cnt: Input indices nums in each batch, just like (M1, M2, ...). Defaults to None. \n

* @par Outputs:
* One outputs: Grouped features, the shape is (M1 + M2 ..., C, nsample).

* @par Third-party framework compatibility
* Compatible with the MMCV operator GroupPoints(StackGroupPoints branch).
*/
REG_OP(StackGroupPoints)
    .INPUT(features, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(features_batch_cnt, TensorType({DT_INT32, DT_INT64}))
    .INPUT(indices, TensorType({DT_INT32, DT_INT64}))
    .INPUT(indices_batch_cnt, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(StackGroupPoints)

/**
 * @brief Find and get the corresponding value from the corresponding ps according to the keys
 * @par Inputs:
 * @li keys: A tensor. Must be int64 type.
 * @li table_id: A tensor. Must be int32 type.
 * @par Outputs:
 * @li values: A Tensor. Must be float32 type.
 * @par Attributes:
 * @li tag: A required integer identifying the hccl operator tag.
 * @li insert_option: Indicates whether lookup supports new value. Defaults to "0".
 * @li max_num: A required integer identifying the keys max num.
 * @li embedding_dim: Apply memory usage for output or infer shape.
 */
REG_OP(HcomRemoteLookup)
    .INPUT(keys, TensorType({DT_INT64}))
    .INPUT(table_id, Int)
    .OUTPUT(values, TensorType({DT_FP32}))
    .REQUIRED_ATTR(tag, Int)
    .ATTR(insert_option, Int, 0)
    .REQUIRED_ATTR(max_num, Int)
    .REQUIRED_ATTR(embedding_dim, Int)
    .OP_END_FACTORY_REG(HcomRemoteLookup)

/**
 * @brief Workers all find and get the corresponding value from the corresponding ps according to the keys
 * @par Inputs:
 * @li table_id: A tensor. Must be int32 type.
 * @li keys: A tensor. Must be int64 type.
 * @par Outputs:
 * @li values: A Tensor. Must be float32 type.
 * @par Attributes:
 * @li tag: A required integer identifying the hccl operator tag.
 * @li insert_option: Indicates whether lookup supports new value. Defaults to "0".
 * @li group: A string identifying the group name of ranks participating in
  the op. Defaults to "hccl_world_group".
 * @li max_num: A required integer identifying the keys max num.
 * @li embedding_dim: A required integer identifying Apply memory usage for output or infer shape.
 * @li flags: An integer identifying counter filter feature.
 */
REG_OP(HcomCollRemoteLookup)
    .INPUT(table_id, TensorType({DT_INT32}))
    .INPUT(keys, TensorType({DT_INT64}))
    .OUTPUT(values, TensorType({DT_FP32}))
    .REQUIRED_ATTR(tag, Int)
    .ATTR(insert_option, Int, 0)
    .ATTR(group, String, "hccl_world_group")
    .REQUIRED_ATTR(max_num, Int)
    .REQUIRED_ATTR(embedding_dim, Int)
    .ATTR(flags, Int, 0)
    .OP_END_FACTORY_REG(HcomCollRemoteLookup)

/**
 * @brief Workers send the keys and values to ps according to keys
 * @par Inputs:
 * @li table_id: A tensor. Must be int32 type.
 * @li keys: A tensor. Must be int64 type.
 * @li values: A Tensor. Must be float32 type.
 * @par Attributes:
 * @li tag: A required integer identifying the hccl operator tag.
 * @li group: A string identifying the group name of ranks participating in
  the op. Defaults to "hccl_world_group".
 * @li max_num: A required integer identifying the keys max num.
 * @li embedding_dim: Apply memory usage for output or infer shape.
 */
REG_OP(HcomCollRemoteUpdate)
    .INPUT(table_id, TensorType({DT_INT32}))
    .INPUT(keys, TensorType({DT_INT64}))
    .INPUT(values, TensorType({DT_FP32}))
    .REQUIRED_ATTR(tag, Int)
    .ATTR(group, String, "hccl_world_group")
    .REQUIRED_ATTR(max_num, Int)
    .REQUIRED_ATTR(embedding_dim, Int)
    .OP_END_FACTORY_REG(HcomCollRemoteUpdate)

/**
 * @brief Workers all find and get the corresponding value from the corresponding ps according to the keys. Used with
 * HcomCollRemoteUpdatePaired.
 * @par Inputs:
 * @li table_id: A tensor. Must be int32 type.
 * @li keys: A tensor. Must be int64 type.
 * @par Outputs:
 * @li values: A Tensor. Must be float32 type.
 * @li indices: A Tensor. Recovery matrix. Must be int64 type.
 * @li num_uniqued: A Tensor. Number of Recovery matrix. Must be int64 type.
 * @li ps_segments: A Tensor. Offset and size of buffer for pss. Must be int64 type.
 * @li ps_segments_num: A Tensor. Number of ps_segments. Must be int64 type.
 * @par Attributes:
 * @li tag: A required integer identifying the hccl operator tag.
 * @li insert_option: Indicates whether lookup supports new value. Defaults to "0".
 * @li group: A string identifying the group name of ranks participating in
  the op. Defaults to "hccl_world_group".
 * @li max_num: A required integer identifying the keys max num.
 * @li embedding_dim: A required integer identifying Apply memory usage for output or infer shape.
 * @li flags: An integer identifying counter filter feature.
 */
REG_OP(HcomCollRemoteLookupPaired)
    .INPUT(table_id, TensorType({DT_INT32}))
    .INPUT(keys, TensorType({DT_INT64}))
    .OUTPUT(values, TensorType({DT_FP32}))
    .OUTPUT(indices, TensorType({DT_INT64}))
    .OUTPUT(num_uniqued, TensorType({DT_INT64}))
    .OUTPUT(ps_segments, TesnorType({DT_INT64}))
    .OUTPUT(ps_segments_num, TesnorType({DT_INT64}))
    .REQUIRED_ATTR(tag, Int)
    .ATTR(insert_option, Int, 0)
    .ATTR(group, String, "hccl_world_group")
    .REQUIRED_ATTR(max_num, Int)
    .REQUIRED_ATTR(embedding_dim, Int)
    .ATTR(flags, Int, 0)
    .OP_END_FACTORY_REG(HcomCollRemoteLookupPaired)

/**
 * @brief Workers all find and get the corresponding value from the corresponding ps according to the keys. Used with
 * HcomCollRemoteLookupUniquedAndPaired.
 * @par Inputs:
 * @li table_id: A tensor. Must be int32 type.
 * @li keys: A tensor. Must be int64 type.
 * @li key_num_input: A tensor. Must be int64 type.
 * @li unique_indices: A tensor. Must be int32 type.
 * @li key_count: A tensor. Must be int32 type.
 * @par Outputs:
 * @li values: A Tensor. Must be float32 type.
 * @li indices: A Tensor. Recovery matrix. Must be int64 type.
 * @li num_uniqued: A Tensor. Number of Recovery matrix. Must be int64 type.
 * @li ps_segments: A Tensor. Offset and size of buffer for pss. Must be int64 type.
 * @li ps_segments_num: A Tensor. Number of ps_segments. Must be int64 type.
 * @par Attributes:
 * @li tag: A required integer identifying the hccl operator tag.
 * @li insert_option: Indicates whether lookup supports new value. Defaults to "0".
 * @li group: A string identifying the group name of ranks participating in
  the op. Defaults to "hccl_world_group".
 * @li max_num: A required integer identifying the keys max num.
 * @li embedding_dim: A required integer identifying Apply memory usage for output or infer shape.
 * @li flags: An integer identifying counter filter feature.
 */
REG_OP(HcomCollRemoteLookupUniquedAndPaired)
    .INPUT(table_id, TensorType({DT_INT32}))
    .INPUT(keys, TensorType({DT_INT64}))
    .INPUT(key_num_input, TensorType({DT_INT64}))
    .INPUT(unique_indices, TesnorType({DT_INT32}))
    .OPTIONAL_INPUT(key_count, TensorType({DT_INT32}))
    .OUTPUT(values, TensorType({DT_FP32}))
    .OUTPUT(indices, TensorType({DT_INT64}))
    .OUTPUT(num_uniqued, TensorType({DT_INT64}))
    .OUTPUT(ps_segments, TesnorType({DT_INT64}))
    .OUTPUT(ps_segments_num, TesnorType({DT_INT64}))
    .REQUIRED_ATTR(tag, Int)
    .ATTR(insert_option, Int, 0)
    .ATTR(group, String, "hccl_world_group")
    .REQUIRED_ATTR(max_num, Int)
    .REQUIRED_ATTR(embedding_dim, Int)
    .REQUIRED_ATTR(flags, Int)
    .OP_END_FACTORY_REG(HcomCollRemoteLookupUniquedAndPaired)

/**
 * @brief Workers send the keys and values to ps according to keys. Used with HcomCollRemoteLookupPaired.
 * @par Inputs:
 * @li table_id: A tensor. Must be int32 type.
 * @li keys: A tensor. Must be int64 type.
 * @li values: A Tensor. Must be float32 type.
 * @li indices: A Tensor. Recovery matrix. Must be int64 type.
 * @li num_uniqued: A Tensor. Number of Recovery matrix. Must be int64 type.
 * @li ps_segments: A Tensor. Offset and size of buffer for pss. Must be int64 type.
 * @li ps_segments_num: A Tensor. Number of ps_segments. Must be int64 type.
 * @par Attributes:
 * @li tag: A required integer identifying the hccl operator tag.
 * @li group: A string identifying the group name of ranks participating in
  the op. Defaults to "hccl_world_group".
 * @li max_num: A required integer identifying the keys max num.
 * @li embedding_dim: Apply memory usage for output or infer shape.
 */
REG_OP(HcomCollRemoteUpdatePaired)
    .INPUT(table_id, TensorType({DT_INT32}))
    .INPUT(keys, TensorType({DT_INT64}))
    .INPUT(values, TensorType({DT_FP32}))
    .INPUT(indices, TesnorType({DT_INT64}))
    .INPUT(num_uniqued, TesnorType({DT_INT64}))
    .INPUT(ps_segments, TesnorType({DT_INT64}))
    .INPUT(ps_segments_num, TesnorType({DT_INT64}))
    .OPTIONAL_INPUT(global_step, TesnorType({DT_INT32, DT_INT64}))
    .REQUIRED_ATTR(tag, Int)
    .ATTR(group, String, "hccl_world_group")
    .ATTR(padding_key, Int, 0)
    .ATTR(flags, Int, 0)
    .REQUIRED_ATTR(max_num, Int)
    .REQUIRED_ATTR(embedding_dim, Int)
    .OP_END_FACTORY_REG(HcomCollRemoteUpdatePaired)

/**
 * @brief Calculate that aggregates input data
 * @par Inputs:
 * @li x: A tensor of type float32, int32, int8, int16, float16, int64, uint64
 * @par Outputs:
 * @li y: A tensor of type float32, int32, int8, int16, float16, int64, uint64.
 * @par Attributes:
 * @li tag: A required integer identifying the hccl operator root_rank.
 * @li group: A string identifying the group name of ranks participating in
  the op.
 * @li rank_size: A required integer identifying the rank size.
 */
REG_OP(HcomGather)
    .INPUT(x, TensorType({DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_INT64, DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32, DT_INT8, DT_INT16, DT_FLOAT16, DT_INT64, DT_UINT64}))
    .REQUIRED_ATTR(root_rank, Int)
    .REQUIRED_ATTR(group, String)
    .REQUIRED_ATTR(rank_size, Int)
    .OP_END_FACTORY_REG(HcomGather)

/**
* @brief Find a min polygon from the point set in the operator MinAreaPolygons. \n

* @par Inputs:
* @li pointsets: A 2D Tensor with shape (N, 18), format ND, dtype must be one
 of the following types: float16, float32, double. \n

* @par Outputs:
* @li polygons: A 2D Tensor with shape (N, 8), format ND, dtype must be one of
 the following types: float16, float32, double.  \n
*/
REG_OP(MinAreaPolygons)
    .INPUT(pointsets, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(polygons, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .OP_END_FACTORY_REG(MinAreaPolygons)

/**
* @brief Determine if the target points set is inside polygons. \n

* @par Inputs:
* @li points: A 2D Tensor with shape (N, 2), format ND, dtype must be float32. \n
* @li polygons: A 2D Tensor with shape (M, 8), format ND, dtype must be float32.
*     This parameter will be transposed to be (8, M) before passed to the operator. \n

* @par Outputs:
* @li output: A 2D Tensor with shape (N, M), format ND, dtype must be float32.  \n
*/
REG_OP(PointsInPolygons)
.INPUT(points, TensorType({DT_FLOAT}))
.INPUT(polygons, TensorType({DT_FLOAT}))
.OUTPUT(output, TensorType({DT_FLOAT}))
.OP_END_FACTORY_REG(PointsInPolygons)

/**
* @brief Calculate the index and distance of the nearest three point to the target point.
* @par Inputs:
* Two input:
* xyz1: The set of target points.
* xyz2: The set of compare points. \n

* @par Outputs:
* dist: A Tensor, the distance of the nearest point to the target point.
* idx: A Tensor, the index of the nearest point to the target point. \n
*/
REG_OP(ThreeNN)
    .INPUT(xyz1, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(xyz2, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(dist, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(idx, TensorType({DT_INT32}))
    .OP_END_FACTORY_REG(ThreeNN)

/**
* @brief three interpolate.
* @par Inputs:
* Two input:
* features: The set of features points
* idx: The set of index
* weight : The set of weight points

* @par y:
* y: A Tensor, the interpolate point
*/
REG_OP(ThreeInterpolate)
    .INPUT(features, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(idx, TensorType({DT_INT32, DT_INT64}))
    .INPUT(weight, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ThreeInterpolate)

/**
* @brief three interpolate backward.
* @par Inputs:
* three input:
* grad_x: The set of features points with dtype of float32 and float16 with shape [b,c,n]
* idx: The set of index with dtype of int32 and int64 with shape [b,n,3]
* weight : The set of weight points with dtype of float32 and float16 with shape[b,n,3]
* m: The dims m of output with dtype int
* @par y:
* grad_y: A Tensor, the interpolate backward output with dtype of float32 and float16 with shape[b,c,m]
*/
REG_OP(ThreeInterpolateBackward)
    .INPUT(grad_x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(idx, TensorType({DT_INT32, DT_INT64}))
    .INPUT(weight, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(grad_y, TensorType({DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(m, Int)
    .OP_END_FACTORY_REG(ThreeInterpolateBackward)

/**
 * @brief Calculate the voxels of cloud points \n

 * @par Inputs:
 * Three inputs, including:
 * @li points: the shape is [M, C], points[:3] contain xyz points and points[3:] contain other information.
 * @li voxel_size: the size of voxel with the shape of [3].
 * @li coors_range:the coordinate range of voxel with the shape of [6]. \n

 * @par Outputs:
 * Four outputs, including:
 * @li voxels: the output voxels with the shape of [M, max_points, C].
 * @li coors: the voxel coordinates with shape of [M, 3].
 * @li num_points_per_voxel: the number of points per voxel with the shape of [M].
 * @li voxel_num: the number of voxels. \n

 * @par Attributes:
 * Three attrs, including:
 * @li max_points: maximum points contained in a voxel.
 * @li max_voxels: maximum voxels this op create.
 * @li deterministic: An optional attr, only support true now, false is faster. \n

 * @par Third-party framework compatibility
 * Compatible with the mmcv operator Voxelization.\n
 */
REG_OP(Voxelization)
    .INPUT(points, TensorType({DT_DOUBLE,DT_FLOAT,DT_FLOAT16}))
    .INPUT(voxel_size, TensorType({DT_DOUBLE,DT_FLOAT,DT_FLOAT16}))
    .INPUT(coors_range, TensorType({DT_DOUBLE, DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(voxels, TensorType({DT_DOUBLE,DT_FLOAT,DT_FLOAT16}))
    .OUTPUT(coors, TensorType({DT_INT32}))
    .OUTPUT(num_points_per_voxel, TensorType({DT_INT32}))
    .OUTPUT(voxel_num, TensorType({DT_INT32}))
    .ATTR(max_points, Int, 35)
    .ATTR(max_voxels, Int, 20000)
    .ATTR(deterministic, Bool, true)
    .OP_END_FACTORY_REG(Voxelization)

/**
 * @brief Encoding the orientation information and generating orientation-sensitive features. \n

 * @par Inputs:
 * Two inputs, including:
 * @li x: Input features with shape [num_output_planes, num_input_planes, num_orientations, H, W].
 * @li indices: Indices with shape [num_orientations, H, W, num_rotations]. \n

 * @par Outputs:
 * One output, including:
 * @li y: Refined features with shape [num_output_planes * num_rotations, num_input_planes * num_orientations, H, W]. \n

 * @par Third-party framework compatibility
 * Compatible with the mmcv operator ActiveRotatedFilter.\n
 */
REG_OP(ActiveRotatedFilter)
    .INPUT(x, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .INPUT(indices, TensorType({DT_INT32,DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .OP_END_FACTORY_REG(ActiveRotatedFilter)

/**
 * @brief The backward of ActiveRotatedFilter. \n

 * @par Inputs:
 * Two inputs, including:
 * @li y_grad: Input features with shape [num_output_planes * num_rotations, num_input_planes * num_orientations, H, W].
 * @li indices: Indices with shape [num_orientations, H, W, num_rotations]. \n

 * @par Outputs:
 * One output, including:
 * @li x_grad: Refined features with shape [num_output_planes, num_input_planes, num_orientations, H, W]. \n

 * @par Third-party framework compatibility
 * Compatible with the mmcv operator ActiveRotatedFilterGrad.\n
 */
REG_OP(ActiveRotatedFilterGrad)
    .INPUT(y_grad, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .INPUT(indices, TensorType({DT_INT32,DT_INT64}))
    .OUTPUT(x_grad, TensorType({DT_FLOAT16,DT_FLOAT,DT_INT32}))
    .OP_END_FACTORY_REG(ActiveRotatedFilterGrad)

/**
* @brief Blend face iamge to the backgroud.
*
* @par Inputs:
* @li face_img: A 3D Tensor, format is ND, dtype is uint8 or float32, shape is (H, W, 3). The input face image.
* @li face_rect: A 1D Tensor, format is ND, dtype is int32, shape is (4,). The coordinates of the face image in the backgroud.
* @li face_mask: A 3D Tensor, format is ND, dtype is float32, shape is (H, W, 1).
* @li acc_face: A 3D Tensor, format is ND, dtype is float32, shape is (H, W, 3).
* @li acc_mask: A 3D Tensor, format is ND, dtype is float32, shape is (H, W, 3).
* @li max_mask: A 3D Tensor, format is ND, dtype is float32, shape is (H, W, 3).
*
* @par Outputs:
* @li acc_face: A 3D Tensor, format is ND. It has the same type and shape as input "acc_face".
* @li acc_mask: A 3D Tensor, format is ND. It has the same type and shape as input "acc_mask".
* @li max_mask: A 3D Tensor, format is ND. It has the same type and shape as input "max_mask". \n
*/
REG_OP(BlendFaceBgPartOne)
    .INPUT(face_img, TensorType({DT_UINT8, DT_FLOAT}))
    .INPUT(face_rect, TensorType({DT_INT32}))
    .INPUT(face_mask, TensorType({DT_FLOAT}))
    .INPUT(acc_face, TensorType({DT_FLOAT}))
    .INPUT(acc_mask, TensorType({DT_FLOAT}))
    .INPUT(max_mask, TensorType({DT_FLOAT}))
    .OUTPUT(acc_face, TensorType({DT_FLOAT}))
    .OUTPUT(acc_mask, TensorType({DT_FLOAT}))
    .OUTPUT(max_mask, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BlendFaceBgPartOne)

/**
* @brief Blend face iamge to the backgroud Part Two.
*
* @par Inputs:
* @li acc_face: A 3D Tensor, format is ND, dtype is float32, shape is (H, W, 3).
* @li acc_mask: A 3D Tensor, format is ND, dtype is float32, shape is (H, W, 3).
* @li max_mask: A 3D Tensor, format is ND, dtype is float32, shape is (H, W, 3).
* @li bg_img: A 3D Tensor, format is ND, dtype is float32 or uint8, shape is (H, W, 3), the input background image.
*
* @par Attributes:
* @li epsilon: A scalar of the same type as "var".
*
* @par Outputs:
* @li fused_img: A 3D Tensor, format is ND. It has the same type and shape as input "acc_face". \n
*/
REG_OP(BlendFaceBgPartTwo)
    .INPUT(acc_face, TensorType({DT_FLOAT}))
    .INPUT(acc_mask, TensorType({DT_FLOAT}))
    .INPUT(max_mask, TensorType({DT_FLOAT}))
    .INPUT(bg_img, TensorType({DT_UINT8, DT_FLOAT}))
    .OUTPUT(fused_img, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 1e-12f)
    .OP_END_FACTORY_REG(BlendFaceBgPartTwo)

/**
* @brief Convert the image from YUV to Raw.
*
* @par Inputs:
* @li img_channel_0: A 2D Tensor, format is ND, dtype is uint16, shape is (h, w).
* The input image of channel 0.
* @li img_channel_1: A 2D Tensor, format is ND, dtype is uint16, shape is (h, w).
* The input image of channel 1.
* @li img_channel_2: A 2D Tensor, format is ND, dtype is uint16, shape is (h, w).
* The input image of channel 2.
* @li img_channel_3: A 2D Tensor, format is ND, dtype is uint16, shape is (h, w).
* The input image of channel 3.
* @li img_size: A 1D Tensor, format is ND, dtype is int32, shape is (2,).
* The data is h_out and w_out, which indicates the output height and width.
* @li gamma: A 1D Tensor, format is ND, dtype is float32, shape is (4,).
*
* @par Attributes:
* bayer_pattern: A optional string. Choice calculate mode, the value must
* be one of ["binning", "quad"]. Default: "binning".
*
* @par Outputs:
* raw_img: A 2D Tensor, format is ND, dtype is uint16, shape is (h_out, w_out).
* The output raw image. \n
*/
REG_OP(ImgRawDecodePostHandle)
    .INPUT(img_channel_0, TensorType({DT_UINT16}))
    .INPUT(img_channel_1, TensorType({DT_UINT16}))
    .INPUT(img_channel_2, TensorType({DT_UINT16}))
    .INPUT(img_channel_3, TensorType({DT_UINT16}))
    .INPUT(img_size, TensorType({DT_INT32}))
    .INPUT(gamma, TensorType({DT_FLOAT}))
    .OUTPUT(raw_img, TensorType({DT_UINT16}))
    .ATTR(bayer_pattern, String, "binning")
    .OP_END_FACTORY_REG(ImgRawDecodePostHandle)

/**
* @brief Convert the image from YUV to Raw.
*
* @par Inputs:
* @li img_channel_0: A 2D Tensor, format is ND, dtype is uint16, shape is (h, w).
* The input image of channel 0.
* @li img_channel_1: A 2D Tensor, format is ND, dtype is uint16, shape is (h, w).
* The input image of channel 1.
* @li img_channel_2: A 2D Tensor, format is ND, dtype is uint16, shape is (h, w).
* The input image of channel 2.
* @li img_channel_3: A 2D Tensor, format is ND, dtype is uint16, shape is (h, w).
* The input image of channel 3.
* @li gamma: A 1D Tensor, dtype is float32, format is ND, shape is (4,).
* @li bayer_coordinate: A 1D Tensor, format is ND, dtype is int32, shape is (4,).
* The data is supplied as [lt_x, lt_y, rb_x, rb_y], where the (lt_x, lt_y), (rb_x, rb_y)
* are the left top and right bottom coordinates, respectively.
* @li bayer_params: A 1D Tensor, format is ND, dtype is float32, shape is (8,).
* The data is supplied as
* [r_gain, g_gain, b_gain, iso, ev_gain, iso_long, evSL, exposure_gain].
* @li bayer_ptn: A 1D Tensor, format is ND, dtype is int32, shape is (4,).
* The bayer_ptn is used as index to obtain the value of rgb_gain.
* @par Outputs:
* raw_img: A 2D Tensor, format is ND, dtype is float32,
* shape is (h_out, w_out), where h_out = rb_y - rb_x, w_out = lt_y - lt_x.
* The output raw image. \n
*/
REG_OP(ImgRawDecodePostHandleV2)
    .INPUT(img_channel_0, TensorType({DT_UINT16}))
    .INPUT(img_channel_1, TensorType({DT_UINT16}))
    .INPUT(img_channel_2, TensorType({DT_UINT16}))
    .INPUT(img_channel_3, TensorType({DT_UINT16}))
    .INPUT(gamma, TensorType({DT_FLOAT}))
    .INPUT(bayer_coordinate, TensorType({DT_INT32}))
    .INPUT(bayer_params, TensorType({DT_FLOAT}))
    .INPUT(bayer_ptn, TensorType({DT_INT32}))
    .OUTPUT(raw_img, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(ImgRawDecodePostHandleV2)

/**
* @brief YUV4442YUV422. Convert the image from yuv444 to yuv422.

* @par Inputs:
* @li x: A 3D Tensor, dtype is float16, shape is (h, w, 4). The input yuv444
* data. The format must be ND. \n

* @par Outputs:
* @li y: A 3D Tensor, dtype is uint8, shape is (h, w, 2). The output yuv422
* data. The format must be ND. \n
*/
REG_OP(YUV4442YUV422)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_UINT8}))
    .OP_END_FACTORY_REG(YUV4442YUV422)

/**
* @brief RGB2YUV422. Convert the image from rgb to yuv422.

* @par Inputs:
* rgb: A 3D Tensor of dtype uint8 with shape (H, W, 3).
* The value of W is a multiple of 16. \n
* @par Outputs:
* yuv: A 3D Tensor of dtype uint8 with shape (H, W, 2).
* The value of H and W are same as rgb. \n

* @attention Constraints:
* Input images is a tensor of 3 dimensions. The last dimension is
* interpretted as channels, and must be three . \n
*/
REG_OP(RGB2YUV422)
    .INPUT(rgb, TensorType({DT_UINT8}))
    .OUTPUT(yuv, TensorType({DT_UINT8}))
    .OP_END_FACTORY_REG(RGB2YUV422)

/**
* @brief Function MultiHeadAttentionScore. \n

* @par Inputs:
* six inputs, including:
* @li query: A matrix Tensor. The type support float16, float32 .
* @li key: A matrix Tensor. The type support float16, float32.
* @li value: A matrix Tensor. The type support float16, float32.
* @li pse_shift: A matrix Tensor. The type support float16, float32.
* @li drop_mask: A matrix Tensor. The type support uint8.
* @li padding_mask: A matrix Tensor. The type support float16, float32.
* @li atten_mask: A matrix Tensor. The type support float16, float32.

* @par Attributes:
* @li scale_value: A float. The scale value. Default: 1.0.
* @li keep_prob: A float. The keep probability of dropout. Default: 1.0.
* @li pre_tockens: A int. Previous tokens.
* @li next_tockens: A int. Next tokens.
* @li head_num: A int. The number of the heads.
* @li input_layout: A string. Specifies the layout of `query`, the value must be one of ["BSH", "SBH"]. Default: "BSH".
*
* @par Outputs:
* softmax_out: A matrix Tensor. The type support float16, float32.
* attention_out: A matrix Tensor. The type support float16, float32.


* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(MultiHeadAttentionScore)
    .INPUT(query, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .OPTIONAL_INPUT(drop_mask, TensorType({DT_UINT1, DT_UINT8}))
    .OPTIONAL_INPUT(padding_mask, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .OUTPUT(softmax_out, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .ATTR(scale_value, Float, 1.0)
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(pre_tockens, Int, 2147483647)
    .ATTR(next_tockens, Int, 2147483647)
    .REQUIRED_ATTR(head_num, Int)
    .REQUIRED_ATTR(input_layout, String)
    .OP_END_FACTORY_REG(MultiHeadAttentionScore)

/**
* @brief Function MultiHeadAttentionScoreGrad. \n

* @par Inputs:
* twelve inputs, including:
* @li query: A matrix Tensor. The type support float32.
* @li key: A matrix Tensor. The type support float32.
* @li value: A matrix Tensor. The type support float32.
* @li dy: A matrix Tensor. The type support float32.
* @li pse_shift: A scalar. The type support float32.
* @li drop_mask: A matrix Tensor. The type support uint8.
* @li padding_mask: A matrix Tensor. The type support float32.
* @li atten_mask: A matrix Tensor. The type support float32.
* @li softmax_max: A matrix Tensor. The type support float32.
* @li softmax_sum: A matrix Tensor. The type support float32.
* @li softmax_in: A matrix Tensor. The type support float32.
* @li attention_in: A matrix Tensor. The type support float32.


* @par Attributes:
* @li scale_value: A float. The scale value. Default: 1.0.
* @li keep_prob: A float. The keep probability of dropout. Default: 1.0.
* @li pre_tockens: A int. Previous tokens. Default: 65536.
* @li next_tockens: A int. Next tokens. Default: 65536.
* @li head_num: A int. The number of the heads.
* @li input_layout: A string. Specifies the layout of `query`, the value must be one of ["BSH", "SBH"]. Default: "BSH".


* @par Outputs:
* dq: A matrix Tensor. The type support float32.
* dk: A matrix Tensor. The type support float32.
* dv: A matrix Tensor. The type support float32.
* dpse: A matrix Tensor. The type support float32.


* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(MultiHeadAttentionScoreGrad)
    .INPUT(query, TensorType({DT_FLOAT32}))
    .INPUT(key, TensorType({DT_FLOAT32}))
    .INPUT(value, TensorType({DT_FLOAT32}))
    .INPUT(dy, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(drop_mask, TensorType({DT_UINT8}))
    .OPTIONAL_INPUT(padding_mask, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(softmax_in, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(attention_in, TensorType({DT_FLOAT32}))
    .OUTPUT(dq, TensorType({DT_FLOAT32}))
    .OUTPUT(dk, TensorType({DT_FLOAT32}))
    .OUTPUT(dv, TensorType({DT_FLOAT32}))
    .OUTPUT(dpse, TensorType({DT_FLOAT32}))
    .ATTR(scale_value, Float, 1.0)
    .ATTR(keep_prob, Float, 1.0)
    .ATTR(pre_tockens, Int, 65536)
    .ATTR(next_tockens, Int, 65536)
    .REQUIRED_ATTR(head_num, Int)
    .REQUIRED_ATTR(input_layout, String)
    .OP_END_FACTORY_REG(MultiHeadAttentionScoreGrad)

/**
* @brief paste sub img.
*
* @par Inputs:
* @li patch_img: A 3D Tensor, format is ND, dtype is uint8 or float16 or float32,
* shape is (H, W, C). The input image.
* @li patch_coord: A 1D Tensor, format is ND, dtype is int32, shape is (4,). The coordinates
* in the combined img.
* @li core_area_coord: A 1D Tensor, format is ND, dtype is int32, shape is (4,). The
* coordinates in the patch img
* @li combine_img: A 3D Tensor, format is ND, dtype is uint8 or float16 or float32, shape is
* (H, W, C). \n
*
* @par Outputs:
* @li combine_img: A 3D Tensor, format is ND. It has the same type and shape as input
 "combine_img". \n
*
* @par Attr
* @li scale: A required float, scale of coordinates. \n
*/
REG_OP(PasteSubImg)
    .INPUT(patch_img, TensorType({DT_UINT8, DT_FLOAT16, DT_FLOAT32}))
    .INPUT(patch_coord, TensorType({DT_INT32}))
    .INPUT(core_area_coord, TensorType({DT_INT32}))
    .INPUT(combine_img, TensorType({DT_UINT8, DT_FLOAT16, DT_FLOAT32}))
    .OUTPUT(combine_img, TensorType({DT_UINT8, DT_FLOAT16, DT_FLOAT32}))
    .REQUIRED_ATTR(scale, Float)
    .OP_END_FACTORY_REG(PasteSubImg)


/**
* @brief RotatedFeatureAlign:Calculate the output features according to
* the input features. \n

* @par Inputs:
* @li x: A tensor of type float32. The input features.
* @li bboxes: A tensor of type float32. The position information of bboxes. \n

* @par Outputs:
* @li y: A tensor of type float32. The output features. \n

* @par Attributes:
* @li spatial_scale: A required float32. The scale of feature map to initial image.
* @li points: An optional int. Defaults to "1". The number of sample points. \n

* @par Third-party framework compatibility
* Compatible with MMCV RotatedFeatureAlign operator.
*/

REG_OP(RotatedFeatureAlign)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(bboxes, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(spatial_scale, Float)
    .ATTR(points, Int, 1)
    .OP_END_FACTORY_REG(RotatedFeatureAlign)

/**
* @brief RotatedFeatureAlignGrad:Calculate the gradient of input features according to
* the gradient of output features. \n

* @par Inputs:
* @li dy: A tensor of type float32. The gradient of output features.
* @li bboxes: A tensor of type float32. The position information of bboxes. \n

* @par Outputs:
* @li dx: A tensor of type float32. The gradient of input features. \n

* @par Attributes:
* @li spatial_scale: A required float32. The scale of feature map to initial image.
* @li points: An optional int. Defaults to "1". The number of sample points. \n

* @par Third-party framework compatibility
* Compatible with MMCV RotatedFeatureAlign operator.
*/

REG_OP(RotatedFeatureAlignGrad)
    .INPUT(dy, TensorType({DT_FLOAT}))
    .INPUT(bboxes, TensorType({DT_FLOAT}))
    .OUTPUT(dx, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(spatial_scale, Float)
    .ATTR(points, Int, 1)
    .OP_END_FACTORY_REG(RotatedFeatureAlignGrad)

/**
* @brief Computes the transpose of convolution 2d with respect to the input.
* @par Inputs:
* Five inputs:
* @li x: A Tensor of type int8.
* The format is NHWC or NCHW.
* @li filter_compress: A Tensor of type int8. Must have the same type as "x".
* The format is NHWC or NCHW or HWCN.
* @li compress_index: A Tensor of type int8. Index for decompression.
* Must have the same type and format as "filter_compress".
* @li bias: An optional 1D tensor of the same type as "y".
* @li offset_w: An optional 1D tensor for quantized inference. Type is int8.
* @par Required Attributes:
* @li input_size: An integer vector representing the shape of input.
* @li strides: A tuple/list of 4 integers.
* Specifies the stride of the sliding window for each dimension of "x".
* The N and C dimensions must be 1. Has the same format as "x".
* @li pads: A required list or tuple of int32. Padding added to each dimension
* of the input.
* @par Attributes:
* Six attributes:
* @li dilations: A tuple/list of 4 integers. The dilation factor for each dimension
* of input. The N and C dimensions must be 1. Has the same format as "x".
* @li groups: Number of blocked connections from input channels to output channels.
* Defaults to "1".
* @li data_format: An optional string from: "NHWC", "NCHW". Defaults to "NHWC".
* Specify the data format of the input and output data.
* @li output_padding: The size will be added in the output shape. Defaults
* to [0, 0, 0, 0].
* @li offset_x: An optional int. Input offset, used for quantized inference.
* Defaults to "0".
* @li alg: An optional string from "weiight_unzip", "weight_sparse_4_2"
* @par Outputs:
* y: A Tensor of type int32.
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(Conv2DTransposeDCompress)
    .INPUT(x, TensorType({DT_INT8}))
    .INPUT(filter_compress, TensorType({DT_INT8}))
    .INPUT(compress_index, TensorType({DT_INT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_INT32}))
    .REQUIRED_ATTR(input_size, ListInt)
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NHWC")
    .ATTR(output_padding, ListInt, {0, 0, 0, 0})
    .ATTR(offset_x, Int, 0)
    .ATTR(alg, String, "weight_sparse_4_2")
    .OP_END_FACTORY_REG(Conv2DTransposeDCompress)

/**
* @brief Detect whether there is Inf or Nan in scaled_grads, set found_inf to 1 if it exists,
* and do not operate on found_inf if it does not. Finally, multiply all values of scaled_grads by inv_scale
* @par Inputs:
 * Three inputs:
 * @li scaled_grads: A tensor list containing multiple tensors, can be float16, float, bfloat16,
 * meanwhile, this value is also an output, store the value multiplied by inv_scale.
 * Shape support 1D ~ 8D. The format is ND.
 * @li found_inf: A tensor with only one element, the shape must be (1,), must be float,
 * meanwhile, this value is also an output, indicating whether there is Inf or Nan present. The format is ND.
 * @li inv_scale: A tensor with only one element, the shape must be (1,), must be float. The format is ND.
*/
REG_OP(ForeachNonFiniteCheckAndUnscale)
    .DYNAMIC_INPUT(scaled_grads, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(found_inf, TensorType({DT_FLOAT}))
    .INPUT(inv_scale, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(ForeachNonFiniteCheckAndUnscale)

/**
* @brief Check if there are non-finite numbers (+inf/-inf/nan) in the tensor_list.
* If there are, set found_flag to 1, otherwise, set found_flag to 0.
* @par Inputs:
 * One input:
 * tensor_list: Dynamic input, A tensor list containing multiple ND format tensors,
 * Support 1D ~ 8D, dtype can be float16, bfloat16, float32.
 * The dtype of each tensor in the tensor_list must be consistent,
 * and the tensor_list can contain a maximum of 256 tensors.
* @par Outputs:
 * found_flag: A tensor with only one element, the shape must be (1,), must be float.
*/
REG_OP(NonFiniteCheck)
    .DYNAMIC_INPUT(tensor_list, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(found_flag, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(NonFiniteCheck)

/**
* @brief Provide the universal template for foreach operators, which have one tensorlist input,
* and one tensorlist output.
* @par Inputs:
* x: A tensor list containing multiple tensors, can be bfloat16, float16, float, int32.
* @par Required Attributes:
* op_code: Determine operator type.Each number represents an operator, Please refer to API related materials.
* @par Outputs:
* y:A tensor list containing multiple tensors. dtype and format of output are same as x1.
*/
REG_OP(ForeachUnaryOp)
    .DYNAMIC_INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .REQUIRED_ATTR(op_code, Int)
    .OP_END_FACTORY_REG(ForeachUnaryOp)

/**
* @brief Provide the universal template for foreach operators, which have one tensorlist input,
* and a scalar input.
* @par Inputs:
* x1: A tensor list containing multiple tensors, can be bfloat16, float16, float, int32.
* x2: A scalar, dtype and format of alpha are same as x1.
* @par Required Attributes:
* op_code: Determine operator type.
* @par Outputs:
* y:A tensor list containing multiple tensors. dtype and format of output are same as x1.
*/
REG_OP(ForeachUnaryWithScalarOp)
    .DYNAMIC_INPUT(x1, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .INPUT(x2, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .REQUIRED_ATTR(op_code, Int)
    .OP_END_FACTORY_REG(ForeachUnaryWithScalarOp)

/**
* @brief Provide the universal template for foreach operators, which have two tensorlist inputs,
* and one tensorlist output.
* @par Inputs:
* @li x1: A tensor list containing multiple tensors, can be bfloat16, float16, float, int32, int64,
* int16, int8, uint8, double, complex128, complex64, complex32.
* @li x2: A tensor list containing multiple tensors. dtype and format of input1 are same as x1.
* @par Required Attributes:
* op_code: Determine operator type.Each number represents an operator, Please refer to API related materials.
* @par Outputs:
* y:A tensor list containing multiple tensors. dtype and format of output are same as x1.
*/
REG_OP(ForeachBinaryOp)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_BF16, DT_INT16,
                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128, DT_COMPLEX64, DT_COMPLEX32}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_BF16, DT_INT16,
                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128, DT_COMPLEX64, DT_COMPLEX32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_BF16, DT_INT16,
                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128, DT_COMPLEX64, DT_COMPLEX32}))
    .REQUIRED_ATTR(op_code, Int)
    .OP_END_FACTORY_REG(ForeachBinaryOp)

/**
* @brief Provide the universal template for foreach operators, which have two tensorlist inputs,
* a scalar input, and one tensorlist output.
* @par Inputs:
* @li x1: A tensor list containing multiple tensors, can be bfloat16, float16, float, int32, int64,
* int16, int8, uint8, double, complex128, complex64, complex32.
* @li x2: A tensor list containing multiple tensors. dtype and format of input1 are same as x1.
* @li x3: A scalar, dtype and format of alpha are same as x1.
* @par Required Attributes:
* op_code: Determine operator type.Each number represents an operator, Please refer to API related materials.
* @par Outputs:
* y:A tensor list containing multiple tensors. dtype and format of output are same as x1.
*/
REG_OP(ForeachBinaryWithScalarOp)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_BF16, DT_INT16,
                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128, DT_COMPLEX64, DT_COMPLEX32}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_BF16, DT_INT16,
                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128, DT_COMPLEX64, DT_COMPLEX32}))
    .INPUT(x3, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_BF16, DT_INT16,
                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128, DT_COMPLEX64, DT_COMPLEX32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32, DT_INT64, DT_BF16, DT_INT16,
                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128, DT_COMPLEX64, DT_COMPLEX32}))
    .REQUIRED_ATTR(op_code, Int)
    .OP_END_FACTORY_REG(ForeachBinaryWithScalarOp)

/**
* @brief Provide the universal template for foreach operators, which have three tensorlist inputs,
* and one tensorlist output.
* @par Inputs:
* @li x1: A tensor list containing multiple tensors, can be bfloat16, float16, float, int32,
* int16, complex64, complex32.
* @li x2: A tensor list containing multiple tensors. dtype and format of input1 are same as x1.
* @li x3: A tensor list containing multiple tensors. dtype and format of input2 are same as x1.
* @par Required Attributes:
* op_code: Determine operator type.Each number represents an operator, Please refer to API related materials.
* @par Outputs:
* y:A tensor list containing multiple tensors. dtype and format of output are same as x1.
*/
REG_OP(ForeachTernaryOp)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_INT16, DT_INT32, DT_FLOAT16, DT_BF16, DT_COMPLEX32, DT_COMPLEX64}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_INT16, DT_INT32, DT_FLOAT16, DT_BF16, DT_COMPLEX32, DT_COMPLEX64}))
    .DYNAMIC_INPUT(x3, TensorType({DT_FLOAT, DT_INT16, DT_INT32, DT_FLOAT16, DT_BF16, DT_COMPLEX32, DT_COMPLEX64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .REQUIRED_ATTR(op_code, Int)
    .OP_END_FACTORY_REG(ForeachTernaryOp)

/**
* @brief Provide the universal template for foreach operators, which have three tensorlist inputs,
* one scalar input, and one output.
* @par Inputs:
* @li x1: A tensor list containing multiple tensors, can be bfloat16, float16, float, int32,
* int16, complex64, complex32.
* @li x2: A tensor list containing multiple tensors. dtype and format of input1 are same as x1.
* @li x3: A tensor list containing multiple tensors. dtype and format of input2 are same as x1.
* @li x4: A scalar, dtype and format of alpha are same as x1.
* @par Required Attributes:
* op_code: Determine operator type.Each number represents an operator, Please refer to API related materials.
* @par Outputs:
* y:A tensor list containing multiple tensors. dtype and format of output are same as x1.
*/
REG_OP(ForeachTernaryWithScalarOp)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_INT16, DT_INT32, DT_FLOAT16, DT_BF16, DT_COMPLEX32, DT_COMPLEX64}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_INT16, DT_INT32, DT_FLOAT16, DT_BF16, DT_COMPLEX32, DT_COMPLEX64}))
    .DYNAMIC_INPUT(x3, TensorType({DT_FLOAT, DT_INT16, DT_INT32, DT_FLOAT16, DT_BF16, DT_COMPLEX32, DT_COMPLEX64}))
    .INPUT(x4, TensorType({DT_FLOAT, DT_INT16, DT_INT32, DT_FLOAT16, DT_BF16, DT_COMPLEX32, DT_COMPLEX64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_INT16, DT_INT32, DT_FLOAT16, DT_BF16, DT_COMPLEX32, DT_COMPLEX64}))
    .REQUIRED_ATTR(op_code, Int)
    .OP_END_FACTORY_REG(ForeachTernaryWithScalarOp)


/**
* @brief round off number foreach element in each tensor in tesnorlist, this is an in-place operation.
* @par Inputs:
 * Two inputs
 * @li x: A tensor list containing multiple tensors, can be float16, float.
 * @li roundMode: mode of round off which currently supports 2(floor) and 3(ceil).
*/
REG_OP(ForeachRoundOffNumberInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(roundMode, TensorType({DT_INT8}))
    .OP_END_FACTORY_REG(ForeachRoundOffNumberInplace)


/**
* @brief round off number foreach element in each tensor in tesnorlist, this is an in-place operation.
* @par Inputs:
 * Two inputs
 * @li x: A tensor list containing multiple tensors, can be float16, float.
 * @li roundMode: mode of round off which currently supports 2(floor) and 3(ceil).
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are produced by round off
*/
REG_OP(ForeachRoundOffNumber)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(roundMode, TensorType({DT_INT8}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachRoundOffNumber)


/**
* @brief multiply scalar foreach element in each tensor in tesnorlist, this is an in-place operation.
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors, can be float16, float, and int32.
 * @li scalar: A scalar to be multiplied, the data type must be the same as tensors.
*/
REG_OP(ForeachMulScalarInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .INPUT(scalar, TensorType({DT_FLOAT16, DT_FLOAT, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachMulScalarInplace)


/**
* @brief Writes the input data of the corresponding subscript to the specified register.
* @par Inputs:
* Two inputs:
* @li x1: A 1D tensor, dtype is int32, format is ND, shape is (1,).
* @li x2: A 1D tensor, dtype is uint64, the format is ND.
*/
REG_OP(SwitchByIndex)
    .INPUT(x1, TensorType({DT_INT32}))
	.INPUT(x2, TensorType({DT_UINT64}))
    .OP_END_FACTORY_REG(SwitchByIndex)


/**
* @brief Quant Batch Matmul Calculation.

* @par Inputs:
* Four inputs:
* @li x1: A matrix Tensor. The format support ND. The shape ranges from 2D to 3D,
          with the same dimensions as that of x2.
          Must be one of the following types: int8. Boardcasting is not supported between x1 and x2.
          The data types of x1 and x2 must meet the deduction relationship.
          The shape is (batch,m,k), where batch is optional.
* @li x2: A matrix Tensor. The format support ND. The shape ranges from 2D to 3D,
          with the same dimensions as that of x1.
          Must be one of the following types: int8. Boardcasting is not supported between x1 and x2.
          The data types of x1 and x2 must meet the deduction relationship.
          The shape is (batch,k,n), where batch is optional.
* @li deq_scale: A quantization parameter Tensor. The format support ND. Must be one of the following types: uint64.
* @li bias: A 1D optional matrix Tensor. The format support ND.
            The shape is (n,), where n is the same as that of x2. Must be one of the following types: int32. \n

* @par Attributes:
* @li adj_x1: A bool. If true, changes the shape of "x1" from [m, k] to
              [k, m] before multiplication. Default: false.
* @li adj_x2: A bool. If true, changes the shape of "x2" from [k, m] to
              [m, k] before multiplication. Default: false. \n

* @par Outputs:
* y: A matrix Tensor. Must be one of the following types: float16.
     The format support ND. The shape must be deduced from x1 and x2.
     The shape is (batch,m,n), where batch is optional. \n

* @attention Constraints:
    1. The data type, format, or shape of x1, x2, bias and out should be supported.
    2. Data type deduction can be performed for x1 and x2.
    3. The input shapes of x1 adn x2 must meet the matrix multiplication relationship.
*/
REG_OP(QuantBatchMatmul)
    .INPUT(x1, TensorType({DT_INT8}))
    .INPUT(x2, TensorType({DT_INT8}))
    .INPUT(deq_scale, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(bias, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .ATTR(adj_x1, Bool, false)
    .ATTR(adj_x2, Bool, false)
    .OP_END_FACTORY_REG(QuantBatchMatmul)

/**
* @brief The fusion operator of antiquant function and matmul.

* @par Inputs:
* @li input_x: A Tensor of type float16. The format is ND.
* @li input_y: A Tensor of type int8. The format is ND.
* @li diagonal_matrix: A Tensor of type int8. The format is ND.
* @li q_bias: A Tensor of type int32. The format is ND.
* @li deq_scale: A tensor for quantized inference. The format is NHWC.
* Type is uint64.
* @li bias: An Optional Tensor of type float. The format is ND.

* @par Attributes:
* @li adj_x1: A bool, if true means input_x is transposed.
* @li adj_x2: A bool, if true means input_y is transposed.

* @par Outputs:
* y: A matrix Tensor of type float16. The format is ND.

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(WeightQuantBatchmatmul)
    .INPUT(input_x, TensorType({DT_FLOAT16}))
    .INPUT(input_y, TensorType({DT_INT8}))
    .INPUT(diagonal_matrix, TensorType({DT_INT8}))
    .INPUT(q_bias, TensorType({DT_INT32}))
    .INPUT(deq_scale, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .ATTR(adj_x1, Bool, false)
    .ATTR(adj_x2, Bool, false)
    .OP_END_FACTORY_REG(WeightQuantBatchmatmul)

/**
* @brief The fusion operator of antiquant function and matmul.

* @par Inputs:
* @li input_x: A Tensor of type float16. The format is ND.
* @li input_y: A Tensor of type int8. The format is ND.
* @li diagonal_matrix: A Tensor of type int8. The format is ND.
* @li q_bias: A Tensor of type int32. The format is ND.
* @li deq_scale: A tensor for quantized inference. The format is NHWC.
* Type is uint64.
* @li mul_scale: A Tensor of type fp16. The format is ND.
* @li add_offset: A Tensor of type fp16. The format is ND.
* @li bias: An Optional Tensor of type float. The format is ND.

* @par Attributes:
* @li adj_x1: A bool, if true means input_x is transposed.
* @li adj_x2: A bool, if true means input_y is transposed.
* @li scale: A float, means antiquant param.
* @li offset: A float, means antiquant param.

* @par Outputs:
* y: A matrix Tensor of type float16. The format is ND.

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(WeightQuantBatchmatmulV3)
    .INPUT(input_x, TensorType({DT_FLOAT16}))
    .INPUT(input_y, TensorType({DT_INT8}))
    .INPUT(diagonal_matrix, TensorType({DT_INT8}))
    .INPUT(q_bias, TensorType({DT_INT32}))
    .INPUT(deq_scale, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(mul_scale, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(add_offset, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .ATTR(adj_x1, Bool, false)
    .ATTR(adj_x2, Bool, false)
    .ATTR(scale, Float, 1)
    .ATTR(offset, Float, 0)
    .OP_END_FACTORY_REG(WeightQuantBatchmatmulV3)

/**
* @brief Fusion op for batchmatmul-fixpipe.
* @par Inputs:
* @li x1: A Tensor of type float16. The format is ND.
* @li X2: A Tensor of type float16. Must have the same type as "x". The format is ND.
* @li quant_pre: A tensor for quantized inference. The format is NHWC. Type is uint64.
* @li bias: A Tensor of type float16. The format is ND.
* @par Outputs:
* @li y: A Tensor of type int8. The format is ND.
* @par Attributes:
* @li adj_x1: A bool, true means x1 is transposed.
* @li adj_x2: A bool, true means x2 is transposed.
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(BatchMatmulFixpipe)
    .INPUT(x1, TensorType({DT_FLOAT16}))
    .INPUT(x2, TensorType({DT_FLOAT16}))
    .INPUT(quant_pre, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_INT8}))
    .ATTR(adj_x1, Bool, false)
    .ATTR(adj_x2, Bool, false)
    .OP_END_FACTORY_REG(BatchMatmulFixpipe)

/**
* @brief Apply add operation for each tensor in tensor list with a scalar in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value add by the scalar.
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
*/
REG_OP(ForeachAddScalarInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachAddScalarInplace)


/**
* @brief Apply add operation for each tensor in tensor list with a scalar in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are add by the scalar
*/
REG_OP(ForeachAddScalar)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachAddScalar)


/**
* @brief Apply add operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors,
 * meanwhile, this value is also an output, store the value add by the scalar.
 * @li scalars: A scalar list in form of tensor with only multiple elements
*/
REG_OP(ForeachAddScalarListInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachAddScalarListInplace)


/**
* @brief Apply add operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalars: A scalar list in form of tensor with only multiple elements
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are add by the scalars in scalar list
*/
REG_OP(ForeachAddScalarList)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachAddScalarList)


/**
* @brief Apply add operation for each tensor in a tensor list with each tensor in another
* tensor list in manner of element-wise
* @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value add by the scalar.
 * @li x2: Another tensor list containing multiple tensors
 * @li alpha: The elements in x2 should perform multipy with alpha which is a scalar
*/
REG_OP(ForeachAddListInplace)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OPTIONAL_INPUT(alpha, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachAddListInplace)


/**
* @brief Apply add operation for each tensor in a tensor list with each tensor in another
* tensor list in manner of element-wise
* @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Another tensor list containing multiple tensors
 * @li alpha: The elements in x2 should perform multipy with alpha which is a scalar
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are add by the scalars in scalar list
*/
REG_OP(ForeachAddList)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(alpha, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachAddList)


/**
* @brief Apply sub operation for each tensor in tensor list with a scalar in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value sub by the scalar.
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
*/
REG_OP(ForeachSubScalarInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachSubScalarInplace)


/**
* @brief Apply sub operation for each tensor in tensor list with a scalar in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are sub by the scalar
*/
REG_OP(ForeachSubScalar)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachSubScalar)


/**
* @brief Apply sub operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value sub by the scalar.
 * @li scalars: A scalar list in form of tensor with only multiple elements
*/
REG_OP(ForeachSubScalarListInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachSubScalarListInplace)


/**
* @brief Apply sub operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalars: A scalar list in form of tensor with only multiple elements
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are sub by the scalars in scalar list
*/
REG_OP(ForeachSubScalarList)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachSubScalarList)


/**
* @brief Apply sub operation for each tensor in a tensor list with each tensor in another
* tensor list in manner of element-wise
* @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value sub by the scalar.
 * @li x2: Another tensor list containing multiple tensors
 * @li alpha: The elements in x2 should perform multipy with alpha which is a scalar
*/
REG_OP(ForeachSubListInplace)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OPTIONAL_INPUT(alpha, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachSubListInplace)


/**
* @brief Apply sub operation for each tensor in a tensor list with each tensor in another
* tensor list in manner of element-wise
* @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Another tensor list containing multiple tensors
 * @li alpha: The elements in x2 should perform multipy with alpha which is a scalar
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are sub by the scalars in scalar list
*/
REG_OP(ForeachSubList)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(alpha, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachSubList)


/**
* @brief Apply mul operation for each tensor in tensor list with a scalar in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are mul by the scalar
*/
REG_OP(ForeachMulScalar)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachMulScalar)


/**
* @brief Apply mul operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value mul by the scalar.
 * @li scalars: A scalar list in form of tensor with only multiple elements
*/
REG_OP(ForeachMulScalarListInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachMulScalarListInplace)


/**
* @brief Apply mul operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalars: A scalar list in form of tensor with only multiple elements
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are mul by the scalars in scalar list
*/
REG_OP(ForeachMulScalarList)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachMulScalarList)


/**
* @brief Apply mul operation for each tensor in a tensor list with each tensor in another
* tensor list in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x1: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value mul by the scalar.
 * @li x2: Another tensor list containing multiple tensors
*/
REG_OP(ForeachMulListInplace)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachMulListInplace)


/**
* @brief Apply mul operation for each tensor in a tensor list with each tensor in another
* tensor list in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Another tensor list containing multiple tensorsr
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are mul by the scalars in scalar list
*/
REG_OP(ForeachMulList)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachMulList)


/**
* @brief Apply div operation for each tensor in tensor list with a scalar in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value div by the scalar.
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
*/
REG_OP(ForeachDivScalarInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachDivScalarInplace)


/**
* @brief Apply div operation for each tensor in tensor list with a scalar in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are div by the scalar
*/
REG_OP(ForeachDivScalar)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachDivScalar)

/**
* @brief Apply div operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value div by the scalar.
 * @li scalars: A scalar list in form of tensor with only multiple elements
*/
REG_OP(ForeachDivScalarListInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachDivScalarListInplace)


/**
* @brief Apply div operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalars: A scalar list in form of tensor with only multiple elements
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are div by the scalars in scalar list
*/
REG_OP(ForeachDivScalarList)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(scalars, TensorType({DT_FLOAT}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachDivScalarList)


/**
* @brief Apply erf operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the erf value of the x
*/
REG_OP(ForeachErf)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachErf)


/**
* @brief Apply erfc operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the erfc value of the x
*/
REG_OP(ForeachErfc)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachErfc)


/**
* @brief Apply Div operation for each tensor in a tensor list with each tensor in another
* tensor list in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x1: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value Div by the scalar.
 * @li x2: Another tensor list containing multiple tensors
*/
REG_OP(ForeachDivListInplace)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachDivListInplace)


/**
* @brief Apply div operation for each tensor in a tensor list with each tensor in another
* tensor list in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Another tensor list containing multiple tensorsr
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are div by the scalars in scalar list
*/
REG_OP(ForeachDivList)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachDivList)


/**
* @brief Apply maximum operation for each tensor in tensor list with a scalar in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value maximum with the scalar.
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
*/
REG_OP(ForeachMaximumScalarInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachMaximumScalarInplace)


/**
* @brief Apply maximum operation for each tensor in tensor list with a scalar in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are maximum with the scalar
*/
REG_OP(ForeachMaximumScalar)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachMaximumScalar)


/**
* @brief Apply maximum operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value maximum with the scalar.
 * @li scalars: A scalar list in form of tensor with only multiple elements
*/
REG_OP(ForeachMaximumScalarListInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachMaximumScalarListInplace)


/**
* @brief Apply maximum operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalars: A scalar list in form of tensor with only multiple elements
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are maximum with the scalars in scalar list
*/
REG_OP(ForeachMaximumScalarList)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachMaximumScalarList)


/**
* @brief Apply maximum operation for each tensor in a tensor list with each tensor in another
* tensor list in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x1: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value maximum with the scalar.
 * @li x2: Another tensor list containing multiple tensors
*/
REG_OP(ForeachMaximumListInplace)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachMaximumListInplace)


/**
* @brief Apply maximum operation for each tensor in a tensor list with each tensor in another
* tensor list in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Another tensor list containing multiple tensorsr
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are maximum with the scalars in scalar list
*/
REG_OP(ForeachMaximumList)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachMaximumList)


/**
* @brief Apply minimum operation for each tensor in tensor list with a scalar in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value minimum with the scalar.
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
*/
REG_OP(ForeachMinimumScalarInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachMinimumScalarInplace)


/**
* @brief Apply minimum operation for each tensor in tensor list with a scalar in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are minimum with the scalar
*/
REG_OP(ForeachMinimumScalar)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachMinimumScalar)


/**
* @brief Apply minimum operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value minimum with the scalar.
 * @li scalars: A scalar list in form of tensor with only multiple elements
*/
REG_OP(ForeachMinimumScalarListInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachMinimumScalarListInplace)


/**
* @brief Apply minimum operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalars: A scalar list in form of tensor with only multiple elements
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are minimum with the scalars in scalar list
*/
REG_OP(ForeachMinimumScalarList)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachMinimumScalarList)


/**
* @brief Apply minimum operation for each tensor in a tensor list with each tensor in another
* tensor list in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x1: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value minimum with the scalar.
 * @li x2: Another tensor list containing multiple tensors
*/
REG_OP(ForeachMinimumListInplace)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachMinimumListInplace)


/**
* @brief Apply minimum operation for each tensor in a tensor list with each tensor in another
* tensor list in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Another tensor list containing multiple tensorsr
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are minimum with the scalars in scalar list
*/
REG_OP(ForeachMinimumList)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachMinimumList)


/**
* @brief Apply power operation for each tensor in tensor list with a scalar in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value power with the scalar.
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
*/
REG_OP(ForeachPowScalarInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachPowScalarInplace)


/**
* @brief Apply power operation for each tensor in tensor list with a scalar in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalar: A scalar in form of tensor with only one element, the shape must be (1,)
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are power with the scalar
*/
REG_OP(ForeachPowScalar)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachPowScalar)


/**
* @brief Apply power operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value power with the scalar.
 * @li scalars: A scalar list in form of tensor with only multiple elements
*/
REG_OP(ForeachPowScalarListInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachPowScalarListInplace)


/**
* @brief Apply power operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalars: A scalar list in form of tensor with only multiple elements
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are power with the scalars in scalar list
*/
REG_OP(ForeachPowScalarList)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachPowScalarList)


/**
* @brief Apply power operation for each tensor in tensor list with a scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Two inputs:
 * @li x: A tensor list containing multiple tensors
 * @li scalar: A scalar
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are power with the scalars in scalar list
*/
REG_OP(ForeachPowScalarAndTensor)
    .INPUT(scalar, TensorType({DT_FLOAT, DT_INT64}))
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachPowScalarAndTensor)


/**
* @brief Apply power operation for each tensor in a tensor list with each tensor in another
* tensor list in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x1: A tensor list containing multiple tensors
 * meanwhile, this value is also an output, store the value power with the scalar.
 * @li x2: Another tensor list containing multiple tensors
*/
REG_OP(ForeachPowListInplace)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachPowListInplace)


/**
* @brief Apply power operation for each tensor in a tensor list with each tensor in another
* tensor list in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x1: A scalar
 * @li x2: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are powering the scalar in scalar list
*/
REG_OP(ForeachScalarPowTensor)
    .INPUT(scalar, TensorType({DT_FLOAT, DT_INT64}))
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachScalarPowTensor)


/**
* @brief Apply power operation for a scalar with each tensor in a tensor list
* in manner of element-wise
* @par Inputs:
 * Two inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Another tensor list containing multiple tensorsr
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are power with the scalars in scalar list
*/
REG_OP(ForeachPowList)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachPowList)


/**
* @brief Apply abs operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachAbsInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachAbsInplace)


/**
* @brief Apply abs operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the abs value of the x
*/
REG_OP(ForeachAbs)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachAbs)


/**
* @brief Apply copy operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * x: A tensor list containing multiple tensors. The data type can only be
 * float16, float, bfloat16, int8, int16, int32, uint8, uint16, uint32, int64, float64, bool.
 * The format support ND. Shape support 1D ~ 8D.
* @par Outputs:
 * y: A tensor list which store the tensors whose value are the copy value of the x.
 * The data type can only be float16, float, bfloat16, int8, int16, int32, uint8, uint16, uint32, int64, float64, bool.
 * The format support ND. Shape support 1D ~ 8D. Has the same dtype adn shape as input "x".
*/
REG_OP(ForeachCopy)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_UINT32, DT_INT64, DT_DOUBLE, DT_BOOL}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16, DT_INT8, DT_UINT8, DT_INT16, DT_UINT16, DT_UINT32, DT_INT64, DT_DOUBLE, DT_BOOL}))
    .OP_END_FACTORY_REG(ForeachCopy)


/**
* @brief Apply sign operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output.
 * The data type can only be float16, float, int32. The format support ND. Shape support 1D ~ 8D.
*/
REG_OP(ForeachSignInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachSignInplace)


/**
* @brief Apply sign operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * x: A tensor list containing multiple tensors. The data type can only be float16, float, int32, int8, int64, bfloat16.
 * The format support ND. Shape support 1D ~ 8D.
* @par Outputs:
 * y: A tensor list which store the tensors whose value are the sign value of the x.
 * The format support ND. The data type can only be float16, float, int32, int8, int64, bfloat16.
 * Shape support 1D ~ 8D. The data type and shape are same as input "x".
*/
REG_OP(ForeachSign)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_INT64, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT8, DT_INT64, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachSign)


/**
* @brief Apply arc cos operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachACosInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachACosInplace)


/**
* @brief Apply arc cos operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the arc cos value of the x
*/
REG_OP(ForeachAcos)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachAcos)


/**
* @brief Apply arc sin operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachASinInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachASinInplace)


/**
* @brief Apply arc sin operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the arc sin value of the x
*/
REG_OP(ForeachAsin)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachAsin)


/**
* @brief Apply arc tan operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachATanInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachATanInplace)


/**
* @brief Apply arc tan operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the arc tan value of the x
*/
REG_OP(ForeachAtan)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachAtan)


/**
* @brief Apply cos operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachCosInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachCosInplace)


/**
* @brief Apply cos operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the cos value of the x
*/
REG_OP(ForeachCos)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachCos)


/**
* @brief Apply cos operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachSinInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachSinInplace)


/**
* @brief Apply cos operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the cos value of the x
*/
REG_OP(ForeachSin)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachSin)


/**
* @brief Apply tan operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachTanInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachTanInplace)


/**
* @brief Apply tan operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the tan value of the x
*/
REG_OP(ForeachTan)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachTan)


/**
* @brief Apply cosh operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachCoshInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachCoshInplace)


/**
* @brief Apply cosh operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the cosh value of the x
*/
REG_OP(ForeachCosh)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachCosh)


/**
* @brief Apply sinh operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachSinhInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachSinhInplace)


/**
* @brief Apply sinh operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the sinh value of the x
*/
REG_OP(ForeachSinh)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachSinh)


/**
* @brief Apply tanh operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachTanhInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachTanhInplace)


/**
* @brief Apply tanh operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the tanh value of the x
*/
REG_OP(ForeachTanh)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachTanh)


/**
* @brief Apply sqrt operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachSqrtInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachSqrtInplace)


/**
* @brief Apply sqrt operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the sqrt value of the x
*/
REG_OP(ForeachSqrt)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachSqrt)


/**
* @brief Apply neg operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachNegInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachNegInplace)


/**
* @brief Apply neg operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the neg value of the x
*/
REG_OP(ForeachNeg)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachNeg)


/**
* @brief Apply norm operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the norm value of the x
*/
REG_OP(ForeachNorm)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_INT64}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachNorm)


/**
* @brief Apply exp operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachExpInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachExpInplace)


/**
* @brief Apply exp operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the exp value of the x
*/
REG_OP(ForeachExp)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachExp)


/**
* @brief Apply expm1 operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachExpm1Inplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachExpm1Inplace)


/**
* @brief Apply expm1 operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the expm1 value of the x
*/
REG_OP(ForeachExpm1)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachExpm1)


/**
* @brief Apply log operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachLogInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachLogInplace)


/**
* @brief Apply log operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the log value of the x
*/
REG_OP(ForeachLog)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachLog)


/**
* @brief Apply log2 operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachLog2Inplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachLog2Inplace)


/**
* @brief Apply log2 operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the log2 value of the x
*/
REG_OP(ForeachLog2)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachLog2)


/**
* @brief Apply log10 operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachLog10Inplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachLog10Inplace)


/**
* @brief Apply log10 operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the log10 value of the x
*/
REG_OP(ForeachLog10)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachLog10)


/**
* @brief Apply log1p operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachLog1pInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachLog1pInplace)


/**
* @brief Apply log1p operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the log1p value of the x
*/
REG_OP(ForeachLog1p)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachLog1p)


/**
* @brief Apply reciprocal operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachReciprocalInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachReciprocalInplace)


/**
* @brief Apply reciprocal operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the reciprocal value of the x
*/
REG_OP(ForeachReciprocal)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachReciprocal)


/**
* @brief Apply zero operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachZeroInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachZeroInplace)


/**
* @brief Apply sigmoid operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors meanwhile, this value is also an output
*/
REG_OP(ForeachSigmoidInplace)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachSigmoidInplace)


/**
* @brief Apply sigmoid operation for each tensor in a tensor list in manner of element-wise
* @par Inputs:
 * One inputs:
 * @li x: A tensor list containing multiple tensors
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are the sigmoid value of the x
*/
REG_OP(ForeachSigmoid)
    .DYNAMIC_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachSigmoid)

/**
* @brief Performs the backpropagation of ROI Align Rotated . \n

* @par Inputs:
* @li x: A tensor of type float32, describing the feature_map.
* @li rois: A tensor of type float32, with shape(n, 6) with each roi decoded as
*     (batch_index, center_x, center_y, w, h, angle). The angle is in radian.

* @par Attributes:
* @li pooled_h: A required int32, specifying the pooled H. Must be greater than 0.
* @li pooled_w: A required int32, specifying the pooled W. Must be greater than 0.
* @li spatial_scale: An required scaling factor for mapping the input coordinates
*     to the ROI coordinates.
* @li sampling_ratio: An required number of inputs samples to take for each output sample.
*     0 to take samples densely for current models.
* @li aligned: A required bool, if False, use the legacy implementation.
*     If True, align the results more perfectly. Default: True.
* @li clockwise: A required bool, if True, the angle in each proposal follows a clockwise
*     fashion in image space, Otherwise, the angle is counterclockwise. Default: False. \n

* @par Outputs:
* @li y: A tensor of type float32, describing the result. \n

* @par Third-party framework compatibility
* It has a corresponding operator in MMCV.
*/
REG_OP(RoiAlignRotatedGrad)
    .INPUT(x_grad, TensorType({DT_FLOAT}))
    .INPUT(rois, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(y_grad_shape, ListInt)
    .REQUIRED_ATTR(pooled_h, Int)
    .REQUIRED_ATTR(pooled_w, Int)
    .REQUIRED_ATTR(spatial_scale, Float)
    .ATTR(sampling_ratio, Int, 0)
    .ATTR(aligned, Bool, true)
    .ATTR(clockwise, Bool, false)
    .OUTPUT(y_grad, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(RoiAlignRotatedGrad)


/**
* @brief Apply AddcDiv operation for each tensor in tensor list with a scalar in manner
* of element-wise
* @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors, which is also used for the output
 * @li x2: Second tensor list containing multiple tensors
 * @li x3: Third tensor list containing multiple tensors
 * @li scalar: A scalar value
*/
REG_OP(ForeachAddcdivScalarInplace)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_INPUT(x3, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachAddcdivScalarInplace)


/**
* @brief Apply AddcDiv operation for each tensor in tensor list with a scalar in manner
* of element-wise
* @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Second tensor list containing multiple tensors
 * @li x3: Third tensor list containing multiple tensors
 * @li scalar: A scalar value
* @par Outputs:
    * @li y: A tensor list which store the tensors whose value are AddcDiv with the scalar
*/
REG_OP(ForeachAddcdivScalar)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(x3, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachAddcdivScalar)

/**
* @brief Apply AddcDiv operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Four inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Second tensor list containing multiple tensors
 * @li x3: Third tensor list containing multiple tensors
 * @li scalars: A list of scalar value
* @par Outputs:
    * @li y: A tensor list which store the tensors whose value are AddcDiv with the scalar
*/
REG_OP(ForeachAddcdivScalarList)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(x3, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachAddcdivScalarList)

/**
* @brief Apply AddcDiv operation for each tensor in tensor list with a scalar in scalar list
*  or a tensor in manner of element-wise
* @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors, which is also used for the output
 * @li x2: Second tensor list containing multiple tensors
 * @li x3: Third tensor list containing multiple tensors
 * @li scalars: A scalar list or a tensor
*/
REG_OP(ForeachAddcdivListInplace)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_INPUT(x3, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachAddcdivListInplace)


/**
* @brief Apply AddcDiv operation for each tensor in tensor list with a scalar in scalar list
*  or a tensor in manner of element-wise
* @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Second tensor list containing multiple tensors
 * @li x3: Third tensor list containing multiple tensors
 * @li scalar: A scalar list or a tensor
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are AddcDiv with the scalars
*/
REG_OP(ForeachAddcdivList)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(x3, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachAddcdivList)


/**
* @brief Apply AddcMul operation for each tensor in tensor list with a scalar in manner
* of element-wise
* @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors, which is also used for the output
 * @li x2: Second tensor list containing multiple tensors
 * @li x3: Third tensor list containing multiple tensors
 * @li scalar: A scalar value
*/
REG_OP(ForeachAddcmulScalarInplace)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_INPUT(x3, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachAddcmulScalarInplace)


/**
* @brief Apply AddcMul operation for each tensor in tensor list with a scalar in manner
* of element-wise
* @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Second tensor list containing multiple tensors
 * @li x3: Third tensor list containing multiple tensors
 * @li scalar: A scalar value
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are AddcMul with the scalar
*/
REG_OP(ForeachAddcmulScalar)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x3, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalar, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachAddcmulScalar)

/**
* @brief Apply AddcMul operation for each tensor in tensor list with a list of scalar in manner
* of element-wise the number of tensors in tensor list shall be equal to the number of scalars
* in scalar list
* @par Inputs:
 * Four inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Second tensor list containing multiple tensors
 * @li x3: Third tensor list containing multiple tensors
 * @li scalars: A list of scalars value
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are AddcMul with the scalar
*/
REG_OP(ForeachAddcmulScalarList)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x3, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachAddcmulScalarList)

/**
* @brief Apply AddcMul operation for each tensor in tensor list with a scalar in scalar list
*  or a tensor in manner of element-wise
* @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors, which is also used for the output
 * @li x2: Second tensor list containing multiple tensors
 * @li x3: Third tensor list containing multiple tensors
 * @li scalars: A scalar list or a tensor
*/
REG_OP(ForeachAddcmulListInplace)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .DYNAMIC_INPUT(x3, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
    .OP_END_FACTORY_REG(ForeachAddcmulListInplace)


/**
* @brief Apply AddcMul operation for each tensor in tensor list with a scalar in scalar list
*  or a tensor in manner of element-wise
* @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Second tensor list containing multiple tensors
 * @li x3: Third tensor list containing multiple tensors
 * @li scalar: A scalar list or a tensor
* @par Outputs:
    * @li y: A tensor list which store the tensors whose value are AddcMul with the scalars
*/
REG_OP(ForeachAddcmulList)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_INPUT(x3, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .INPUT(scalars, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachAddcmulList)


/**
* @brief Apply lerp operation for each tensor in tensor list with tensors in another tensor list and
* a scalar in manner of element-wise
* @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors. meanwhile, this value is also an output,
 * store the value produced by lerp.
 * @li x2: Another tensor list containing multiple tensors
 * @li weight: A scalar in form of tensor with only one element, the shape must be (1,)
*/
REG_OP(ForeachLerpScalarInplace)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(weight, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachLerpScalarInplace)


/**
* @brief Apply lerp operation for each tensor in tensor list with tensors in another tensor list and
* a scalar in manner of element-wise
* @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Another tensor list containing multiple tensors
 * @li weight: A scalar in form of tensor with only one element, the shape must be (1,)
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are produced by lerp
*/
REG_OP(ForeachLerpScalar)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(weight, TensorType({DT_FLOAT}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachLerpScalar)


/**
* @brief Apply lerp operation for each tensor in tensor list with tensors in another tensor list and
* an additonal tensor in manner of element-wise
* @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors. meanwhile, this value is also an output,
 * store the value produced by lerp.
 * @li x2: Another tensor list containing multiple tensors
 * @li weights: A tensor contain multiple elements
*/
REG_OP(ForeachLerpListInplace)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(weights, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(ForeachLerpListInplace)


/**
* @brief Apply lerp operation for each tensor in tensor list with tensors in another tensor list and
* an additonal tensor in manner of element-wise
* @par Inputs:
 * Three inputs:
 * @li x1: A tensor list containing multiple tensors
 * @li x2: Another tensor list containing multiple tensors
 * @li weights: A tensor contain multiple elements
* @par Outputs:
 * @li y: A tensor list which store the tensors whose value are produced by lerp
*/
REG_OP(ForeachLerpList)
    .DYNAMIC_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(weight, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DYNAMIC_OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OP_END_FACTORY_REG(ForeachLerpList)


/**
* @brief The GELUV2 activation function is x*Φ(x),
* where Φ(x) the standard Gaussian cumulative distribution function.

* @par Inputs:
* One input, including:
* x: A Tensor. Must be one of the following types: bfloat16, float16, float32.

* @par Outputs:
* y: A Tensor. Has the same type as "x".

* @par Attributes:
* approximate: A optional string. The gelu approximation algorithm to use: 'none' or 'tanh', default is 'none'.

* @par Third-party framework compatibility:
* Compatible with the Pytorch operator Gelu.
*/
REG_OP(GeluV2)
    .INPUT(x, "T")
    .OUTPUT(y, "T")
    .DATATYPE(T, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(approximate, String, "none")
    .OP_END_FACTORY_REG(GeluV2)

/**
* @brief Apply power operation for a scalar
* in manner of element-wise
* @par Inputs:
* Two inputs:
* @li x1: A tensor to be power
* @li x2: Another tensor contains a power scalar
* @par Outputs:
* @li y: A tensor which value are power with the scalar

* @par Third-party framework compatibility:
* New operator Pows.

* @par Restrictions:
* Warning:THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(Pows)
    .INPUT(x1, "T")
    .INPUT(x2, "T")
    .OUTPUT(y, "T")
    .DATATYPE(T, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OP_END_FACTORY_REG(Pows)

/**
* @brief Computes the gradient for the gelu of "x" .

* @par Inputs:
* Two inputs, including:
* @li dy: A Tensor. Support 1D ~ 8D. Must be one of the following types:bfloat16, float16, float32.
* @li x: A Tensor of the same type and format as "dy".

* @par Outputs:
* z: A Tensor. Has the same type, shape and format as "dy".

* @par Attributes:
* approximate: A optional string.
* The gelu grad approximation algorithm to use: 'none' or 'tanh', default is 'none'. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator GeluGrad.

* @attention Constraints:
* if the GeluGradV2 operator has approximate='none':
* when x is -inf, the computation result is 0.
* when x is inf, the computation result is dy.

*/
REG_OP(GeluGradV2)
    .INPUT(dy, "T")
    .INPUT(x, "T")
    .OUTPUT(z, "T")
    .DATATYPE(T, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(approximate, String, "none")
    .OP_END_FACTORY_REG(GeluGradV2)

/**
* @brief Multiplies matrix "a" by matrix "b", producing "a * b". \n
* @par Inputs:
* Five inputs, including:
* @li x1: A matrix Tensor. 2D. Must be one of the following types: int8.
* @li x2: A matrix Tensor. 2D. Must be one of the following types: int8.
* @li compress_index: A compress index matrix of type int8.
* @li bias: An optional Tensor. 1D. Must be one of the following types: int32.
* @li offset_w: An optional matrix Tensor. 2D. Must be one of the following
* types: int8. \n

* @par Attributes:
* @li transpose_x1: A bool. If True, changes the shape of "x1" from [K, M] to
* [M, K] before multiplication.
* @li transpose_x2: A bool. If True, changes the shape of "x2" from [N, K] to
* [K, N] before multiplication.
* @li offset_x: An optional integer for quantized MatMulV2CompressDequant.
* @li tiling_k: An optional integer for binary quantized MatMulV2CompressDequant.
* @li tiling_n: An optional integer for binary quantized MatMulV2CompressDequant.
* The negative offset added to the input x1 for int8 type. Ensure offset_x
* within the effective range of int8 [-128, 127]. Defaults to "0". \n

* @par Outputs:
* y: The result matrix Tensor. 2D. Must be one of the following types: int32,
* float16. \n

* @attention Constraints:
* if performances better in format NZ, please close
* "MatmulTransdataFusionPass" in fusion configuration.

*/
REG_OP(MatMulV2CompressDequant)
    .INPUT(x1, TensorType({DT_INT8}))
    .INPUT(x2, TensorType({DT_INT8}))
    .INPUT(compress_index, TensorType({DT_INT8}))
    .INPUT(deq_scale, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(bias, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(compress_info, ListInt, {1, 1, 1, 1, 1})
    .ATTR(offset_x, Int, 0)
    .ATTR(alg, String, "weight_unzip")
    .OP_END_FACTORY_REG(MatMulV2CompressDequant)

/**
* @brief Multiplies matrix "x1" by matrix "x2", producing "x1 * x2". \n
* @par Inputs:
* Four inputs, including:
* @li x1: A matrix Tensor. 2D. Must be one of the following types: bfloat16,
* float16, float32. Has format [ND, NHWC, NCHW].
* @li x2: A matrix Tensor. 2D. Must be one of the following types: bfloat16,
* float16, float32. Has format [ND, NHWC, NCHW].
* @li bias: A 1D Tensor. Must be one of the following types: float32,
* float16, float32. Has format [ND, NHWC, NCHW]. \n
* @li offset_w: An optional matrix Tensor. 2D. Must be one of the following
* types: int8, int4.

* @par Attributes:
* @li transpose_x1: A bool. If True, changes the shape of "x1" from [M, K] to
* [K, M] before multiplication.
* @li transpose_x2: A bool. If True, changes the shape of "x2" from [K, N] to
* [N, K] before multiplication.
* @li offset_x: An optional integer for quantized MatMulV2Compress.
* The negative offset added to the input x1 for int8 type. Ensure offset_x
* within the effective range of int8 [-128, 127]. Defaults to "0".
* @li enable_hf32: An optional bool for MatMulV3. If True, enable enable_hi_float_32_execution
* before multiplication. \n

* @par Outputs:
* y: The result matrix Tensor. 2D. Must be one of the following types: bfloat16,
* float16, float32. Has format [ND, NHWC, NCHW]. \n
*/
REG_OP(MatMulV3)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8, DT_INT4}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(offset_x, Int, 0)
    .ATTR(enable_hf32, Bool, false)
    .OP_END_FACTORY_REG(MatMulV3)

/**
* @brief Multiplies matrix "x1" by matrix "x2", producing "x1 * x2".
* @par Inputs:
* Four inputs, including:
* @li x1: A matrix Tensor. Must be one of the following types: float16,
* float32, bfloat16. 2D-6D. Has format [ND, NHWC, NCHW].
* @li x2: A matrix Tensor. Must be one of the following types: float16,
* float32, bfloat16. 2D-6D. Has format [ND, NHWC, NCHW].
* @li bias: A optional Tensor. Must be one of the following types: float16,
* float32, bfloat16. Has format [ND, NHWC, NCHW].
* @li offset_w: A optional Tensor. Must be one of the following types:
* int8, int4. Has format [ND, NHWC, NCHW].

* @par Attributes:
* @li adj_x1: A bool. If True, changes the shape of "x1" from [B, M, K] to
* [B, K, M] before multiplication.
* @li adj_x2: A bool. If True, changes the shape of "x2" from [B, K, N] to
* [B, N, K] before multiplication.
* @li offset_x: An optional integer for quantized BatchMatMulV3.
* @li enable_hf32: An optional bool for BatchMatMulV3. If True, enable enable_hi_float_32_execution
* before multiplication. \n

* @par Outputs:
* y: The result matrix Tensor. Must be one of the following types: float16,
* float32, bfloat16. 2D-6D. Has format [ND, NHWC, NCHW]. BatchMatMulV3 supports broadcasting in the batch dimensions.
*/

REG_OP(BatchMatMulV3)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8, DT_INT4}))
    .ATTR(adj_x1, Bool, false)
    .ATTR(adj_x2, Bool, false)
    .ATTR(offset_x, Int, 0)
    .ATTR(enable_hf32, Bool, false)
    .OP_END_FACTORY_REG(BatchMatMulV3)

/**
* @brief Finds values and indices of the "k" largest elements for the last
* dimension . \n

* @par Inputs:
* Two inputs, including:
* @li x: A 1D-8D tensor, with the last dimension at least "k".
* Supported type: float16, float32. Supported format: ND.
* @li k: A 0D Tensor of type int32. Supported format: ND.
* Number of top elements to look for along the last dimension (along each row
* for matrices) . \n

* @par Attributes:
* @li sorted: An optional bool. Defaults to "True".
* If "True", the returned "k" elements are themselves sorted.
* If "False", the returned "k" elements are not sorted.
* @li dim: An optional int. Defaults to "-1". For reserved use.
* @li largest: An optional bool, controls whether to return largest or smallest elements. Defaults to "True".
* If "True", the "k" largest elements are returned in descending order.
* If "False", the "k" smallest elements are returned in ascending order. \n

* @par Outputs:
* @li values: A Tensor, specifying the sorted data. Has the same type and format as
* "input".
* @li indices: A Tensor of type int32, specifying the indices of sorted data. Supported format: ND . \n

* @see TopK()
* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator TopKV2.
*/
REG_OP(TopKV3)
    .INPUT(x, TensorType::RealNumberType())
    .INPUT(k, TensorType({DT_INT32}))
    .OUTPUT(values, TensorType::RealNumberType())
    .OUTPUT(indices, TensorType({DT_INT32}))
    .ATTR(sorted, Bool, true)
    .ATTR(dim, Int, -1)
    .ATTR(largest, Bool, true)
    .OP_END_FACTORY_REG(TopKV3)

/**
* @brief multi-scale deformable attention.
*
* @par Inputs:
* @li value: A Tensor. Must be one of the following types: float16, float32.
* @li value_spatial_shapes: A Tensor. Must be one of the following types: int32, int64.
* @li value_level_start_index: A Tensor. Must be one of the following types: int32, int64.
* @li sampling_locations: A Tensor. Must be one of the following types: float16, float32.
* @li attention_weights: A Tensor. Must be one of the following types: float16, float32.
*
* @par Outputs:
* output: A Tensor. Must be one of the following types: float16, float32.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(MultiScaleDeformableAttnFunction)
    .INPUT(value, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(value_spatial_shapes, TensorType({DT_UINT64, DT_INT32}))
    .INPUT(value_level_start_index, TensorType({DT_UINT64, DT_INT32}))
    .INPUT(sampling_locations, TensorType({DT_FLOAT, DT_FLOAT16}))
    .INPUT(attention_weights, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OUTPUT(output, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OP_END_FACTORY_REG(MultiScaleDeformableAttnFunction)

/**
*@brief Computes the gradients of convolution 3d with respect to the input.
*@par Inputs:
 * @li input_size: A 1-D Tensor of type int32 or int64.
 * The tensor representing the shape of feature map (the input of the convolution),
 * where feature map is a 5-D tensor.
 * The order of integers in the tensor is determined by feature map format,
 * and integers represent the length of each dimension of feature map.
 * The axes sequence that can be entered are as follows:
 * [batch, in_depth, in_height, in_width, in_channels] or
 * [batch, in_channels, in_depth, in_height, in_width].
 * @li filter: A 5-D Tensor. Must be one of the following types: float16, float32, bfloat16.
 * The format of the filter tensor must be one of the followings:
 * [out_channels, in_channels/groups, filter_depth, filter_height, filter_width] or
 * [filter_depth, filter_height, filter_width, in_channels/groups, out_channels].
 * kernel_height (H) and kernel_width (W) must have dimension in [1, 511].
 * @li out_backprop: A 5-D Tensor. Must have the same type as filter.
 * The format of the out_backprop tensor must be one of the followings:
 * [batch, out_depth, out_height, out_width, out_channels] or
 * [batch, out_channels, out_depth, out_height, out_width].
 * Gradients with respect to the "output" of the convolution.
*@par Attributes:
 * @li strides: Required. A list of 5 integers. Specifies the stride of the sliding window
 * for each dimension of feature map. The strides have the same axes sequence as feature map:
 * [batch, stride_depth, stride_height, stride_width, channels] or
 * [batch, channels, stride_depth, stride_height, stride_width].
 * The batch(N) and channels(C) dimensions must be 1.
 * The width (W) and height (H) dimension must be in [1, 63].
 * The depth (D) dimension must be in [1, 255].
 * @li pads: Required. A list of 6 integers. Specifies the pads factor of
 * feature map in each directions. Supports only pads along the depth(D),
 * height(H) and width(W) dimensions.
 * The pads sequence is as follows: [front, tail, top, bottom, left, right].
 * Modes "SAME" and "VAILD" padding can be achieved with appropriate values of each direction in pads.
 * All dimensions must be in [0, 255].
 * @li dilations: Optional. Defaults to [1, 1, 1, 1, 1].
 * A tuple/list of 5 integers, The dilation factor for each dimension of filter.
 * The dilations has the same axes sequence as filter:
 * [out_channels, in_channels/groups, depth, dilation_height, dilation_width] or
 * [depth, dilation_height, dilation_width, in_channels/groups, out_channels].
 * Currently all dimensions must be 1.
 * @li groups: Optional. Default to 1.
 * Number of blocked connections from in_channels to out_channels.
 * Currently when groups > 1, only supports
 * groups=in_channels=out_channels and input_data_type equals to bfloat16.
 * @li data_format: Optional. Defaults to "NDHWC". A string from: "NDHWC", "NCDHW".
 * The correspondence is as follows: batch(N), depth(D), height(H), width(W), channels(C).
 * Specify the data format of the feature map, out_backprop and output.
*@par Outputs:
 * y: A Tensor. It has the same format as feature map and out_backprop.
 * The type is float16, bfloat16, float32.
 * The gradients of feature map.
*@par Third-party framework compatibility
 * Compatible with Tensorflow's conv3d_backprop_input
*/
REG_OP(Conv3DBackpropInputV2)
    .INPUT(input_size, TensorType({DT_INT32, DT_INT64}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(out_backprop, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NDHWC")
    .OP_END_FACTORY_REG(Conv3DBackpropInputV2)

/**
*@brief Computes the gradients of convolution 2d with respect to the input.
*@par Inputs:
 * @li input_size: A 1-D Tensor of type int32 or int64.
 * The tensor representing the shape of feature map (the input of the convolution),
 * where feature map is a 4-D tensor.
 * The order of integers in the tensor is determined by feature map format,
 * and integers represent the length of each dimension of feature map.
 * The axes sequence that can be entered are as follows:
 * [batch, in_height, in_width, in_channels] or
 * [batch, in_channels, in_height, in_width].
 * @li filter: A 4-D Tensor. Must be one of the following types: float16, float32, bfloat16.
 * The format of the filter tensor must be one of the followings:
 * [out_channels, in_channels/groups, filter_height, filter_width] or
 * [filter_height, filter_width, in_channels/groups, out_channels].
 * kernel_height (H) and kernel_width (W) must have dimension in [1, 511].
 * @li out_backprop: A 4-D Tensor. Must have the same type as filter.
 * The format of the out_backprop tensor must be one of the followings:
 * [batch, out_height, out_width, out_channels] or
 * [batch, out_channels, out_height, out_width].
 * Gradients with respect to the "output" of the convolution.
*@par Attributes:
 * @li strides: Required. A list of 4 integers. Specifies the stride of the sliding window
 * for each dimension of feature map. The strides have the same axes sequence as feature map:
 * [batch, stride_depth, stride_height, stride_width, channels] or
 * [batch, channels, stride_depth, stride_height, stride_width].
 * The batch(N) and channels(C) dimensions must be 1.
 * The width (W) and height (H) dimension must be in [1, 63].
 * @li pads: Required. A list of 4 integers. Specifies the pads factor of
 * feature map in each directions. Supports only pads along the height(H) and width(W) dimensions.
 * The pads sequence is as follows: [top, bottom, left, right].
 * Modes "SAME" and "VAILD" padding can be achieved with appropriate values of each direction in pads.
 * All dimensions must be in [0, 255].
 * @li dilations: Optional. Defaults to [1, 1, 1, 1].
 * A tuple/list of 4 integers, The dilation factor for each dimension of filter.
 * The dilations has the same axes sequence as filter:
 * [out_channels, in_channels/groups, dilation_height, dilation_width] or
 * [dilation_height, dilation_width, in_channels/groups, out_channels].
 * Currently all dimensions must be 1.
 * @li groups: Optional. Default to 1.
 * Number of blocked connections from in_channels to out_channels.
 * Currently when groups > 1, only supports
 * groups=in_channels=out_channels and input_data_type equals to bfloat16.
 * @li data_format: Optional. Defaults to "NHWC". A string from: "NHWC", "NCHW".
 * The correspondence is as follows: batch(N), height(H), width(W), channels(C).
 * Specify the data format of the feature map, out_backprop and output.
*@par Outputs:
 * y: A Tensor. It has the same format as feature map and out_backprop.
 * The type is float16, bfloat16, float32.
 * The gradients of feature map.
*@par Third-party framework compatibility
 * Compatible with Tensorflow's conv3d_backprop_input
*/
REG_OP(Conv2DBackpropInputV2)
    .INPUT(input_size, TensorType({DT_INT32, DT_INT64}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(out_backprop, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(Conv2DBackpropInputV2)

/**
*@brief Computes the transpose of convolution 3d with respect to the input.

*@par Inputs:
 * @li input_size: A Tensor of type int32 or int64. An integer vector
 * representing the shape of input.
 * @li x: A Tensor of the following types, float16, float32, bfloat16. The format
 * is NDHWC or NCDHW.
 * @li filter: A Tensor of the following types, float16, float32, bfloat16.
 * The format is NDHWC, NCDHW or DHWCN.
 * height (H), width (W) dimension must be in [1, 511].
 * @li bias: Optional. An optional 1D tensor of type float16 or float32. Reserved.
 * @li offset_w: Optional. An optional 1D tensor of type int8 for quantized deconvolution.
 *  Reserved. \n

*@par Attributes:
 * @li strides: Required. A tuple/list of 5 integers. Specifies the stride of
 * the sliding window for each dimension of "x".
 * The N and C dimensions must be 1. Has the same format as "x".
 * The height (H) and width (W) dimensions must be in [1, 63].
 * The width (W) dimension must be in [1, 255].
 * @li pads: Required. A tuple/list of 6 integers.
 * All dimensions must be in [0, 255].
 * @li dilations: Optional. A tuple/list of 5 integers,
 * The dilation factor for each dimension of input.
 * All dimensions must be 1. Has the same format as "x".
 * Defaults to [1, 1, 1, 1, 1].
 * @li groups: Optional. Number of blocked connections from input channels to
 * output channels. Defaults to 1. Currently when groups > 1, only supports
 * groups=in_channels=out_channels and input_data_type equals to bfloat16.
 * @li data_format: Optional. An string from: "NDHWC", "NCDHW".
 * Defaults to "NDHWC". Specify the data format of the input and output data.
 * @li output_padding: Optional. The size will be added in the output shape.
 * Defaults to [0, 0, 0, 0, 0]
 * @li offset_x: Optional. Input offset_x value. Defaults to 0. Reserved. \n

*@par Outputs:
 * y: A Tensor. Has the same format as "x", has the type float16, float32, bfloat16.
*/
REG_OP(Conv3DTransposeV2)
    .INPUT(input_size, TensorType({DT_INT32, DT_INT64}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NDHWC")
    .ATTR(output_padding, ListInt, {0, 0, 0, 0, 0})
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(Conv3DTransposeV2)

/**
*@brief Computes the transpose of convolution 2d with respect to the input.

*@par Inputs:
 * @li input_size: A Tensor of type int32 or int64. An integer vector
 * representing the shape of input.
 * @li x: A Tensor of the following types, float16, float32, bfloat16. The format
 * is NHWC or NCHW.
 * @li filter: A Tensor of the following types, float16, float32, bfloat16.
 * The format is NHWC, NCHW or HWCN.
 * height (H), width (W) dimension must be in [1, 511].
 * @li bias: Optional. An optional 1D tensor of type float16 or float32. Reserved.
 * @li offset_w: Optional. An optional 1D tensor of type int8 for quantized deconvolution.
 *  Reserved. \n

*@par Attributes:
 * @li strides: Required. A tuple/list of 4 integers. Specifies the stride of
 * the sliding window for each dimension of "x".
 * The N and C dimensions must be 1. Has the same format as "x".
 * The height (H) and width (W) dimensions must be in [1, 63].
 * The width (W) dimension must be in [1, 255].
 * @li pads: Required. A tuple/list of 4 integers.
 * All dimensions must be in [0, 255].
 * @li dilations: Optional. A tuple/list of 4 integers,
 * The dilation factor for each dimension of input.
 * All dimensions must be 1. Has the same format as "x".
 * Defaults to [1, 1, 1, 1].
 * @li groups: Optional. Number of blocked connections from input channels to
 * output channels. Defaults to 1. Currently when groups > 1, only supports
 * groups=in_channels=out_channels and input_data_type equals to bfloat16.
 * @li data_format: Optional. An string from: "NHWC", "NCHW".
 * Defaults to "NHWC". Specify the data format of the input and output data.
 * @li output_padding: Optional. The size will be added in the output shape.
 * Defaults to [0, 0, 0, 0]
 * @li offset_x: Optional. Input offset_x value. Defaults to 0. Reserved. \n

*@par Outputs:
 * y: A Tensor. Has the same format as "x", has the type float16, float32, bfloat16.
*/
REG_OP(Conv2DTransposeV2)
    .INPUT(input_size, TensorType({DT_INT32, DT_INT64}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(filter, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT32}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NHWC")
    .ATTR(output_padding, ListInt, {0, 0, 0, 0})
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(Conv2DTransposeV2)

/**
*@brief Computes the gradients of convolution with respect to the filter
*@par Inputs:
 * Three inputs:
 * @li x: A 4D Tensor of input image. With the format "NHWC" which shape is
 * [batch, in_height, in_width, in_channels] or the format "NCHW" which shape
 * is [batch, in_channels, in_height, in_width].
 * support type float16, bfloat16, float32 and double.
 * @li filter_size: A Tensor of type int32. Currently does not support
 * data tensor. An integer vector representing the tensor shape of filter,
 * where filter is a 4-D tensor [filter_height, filter_width, in_channels,
 * out_channels] or [out_channels, filter_height, filter_width, in_channels]
 * or [out_channels, in_channel, filter_height, filter_width].
 * @li out_backprop: A Tensor. Must have the same type as x. 4-D with shape
 * [batch, out_height, out_width, out_channels] or [batch, out_channels,
 * out_height, out_width]. Gradients with respect to the output of the
 * convolution.
 *\n
 *\n
 * The following are the supported data types and data formats:\n
 *\n
 *\n
    | Tensor    | x        | out_backprop | y       |\n
    |-----------|--------- |--------------|---------|\n
    | Data Type | float16  |    float16   | float32 |\n
    |           | bfloat16 |    bfloat16  | float32 |\n
    |           | float32  |    float32   | float32 |\n
    |           | double   |    double    | double  |\n
    | Format    | NCHW     |     NCHW     | NCHW    |\n
    |           | NHWC     |     NHWC     | HWCN    |\n
 *\n
 * For float32 and float64 type of x and outbackprop, the actual calculation on the chip is
 * based on float16 before V220.
 *\n
 *
*@par Attributes:
 * Five attributes:
 * @li strides: A tuple/list of 4 integers. The stride of the sliding window
 * for H/W dimension. The index of H/W is same as data_format.
 * @li pads: A tuple/list of 4 integers, [top, bottom, left, right] pads on
 * feature map.
 * @li dilations: A tuple/list of 4 integers, The dilation factor for each
 * dimension of input, defaults to [1,1,1,1].
 * @li groups: Number of blocked connections from input channels to output
 * channels.
 * @li data_format: An optional string from: "NHWC", "NCHW". Defaults to
 * "NHWC". Specify the data format of the input and output data.
 *\n
 *\n
 * The following value range restrictions must be met:\n
 *\n
 *\n
    | Name             | Field    | Scope        |\n
    |------------------|----------|--------------|\n
    | x(fmap)          | H        | [1, 4096]  |\n
    |                  | W        | [1, 4096]    |\n
    | Filter Size      | H        | [1, 255]     |\n
    |                  | W        | [1, 255]     |\n
    | out_backprop     | H        | [1, 4096]  |\n
    |                  | W        | [1, 4096]    |\n
    | y                | H        | [1, 4096]  |\n
    |                  | W        | [1, 4096]    |\n
    | Stride           | H        | [1, 63]      |\n
    |                  | W        | [1, 63]      |\n
    | Padding          | Top      | [0, 255]     |\n
    |                  | Bottom   | [0, 255]     |\n
    |                  | Left     | [0, 255]     |\n
    |                  | Right    | [0, 255]     |\n
    | Dilation         | H        | [1, 255]     |\n
    |                  | W        | [1, 255]     |\n
 *\n
*@par Outputs:
 * y: A Tensor of type float16, float32 or double, has the same format as filter_size.
 *\n
 *     out_backprop_height = (in_height + pad_top + pad_bottom -
 *                           (dilation_h * (filter_height - 1) + 1))
 *                           / stride_h + 1
 *\n
 *     out_backprop_width = (in_width + pad_left + pad_right -
 *                          (dilation_w * (filter_width - 1) + 1))
 *                          / stride_w + 1
 *\n
 *
*@par Third-party framework compatibility
 * Compatible with Tensorflow's conv2d_backprop_filter
*/
REG_OP(Conv2DBackpropFilterV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16}))
    .INPUT(filter_size, TensorType({DT_INT32}))
    .INPUT(out_backprop, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(Conv2DBackpropFilterV2)

/**
*@brief Computes the gradients of convolution3D with respect to the filter
*@par Inputs:
 * @li x: A Tensor. Must be one of the following types: float16, float32, bfloat16.
 * 5-D with shape [batch, in_depth, in_height, in_width, in_channels]
 * or [batch, in_channels, in_depth, in_height, in_width].
 * @li filter_size: A Tensor of type int32. An integer vector representing the
 * tensor shape of filter, where filter is a 5-D tensor
 * [filter_depth, filter_height, filter_width, in_channels, out_channels]
 * [out_channels, in_channels, filter_depth, filter_height, filter_width]
 * or [out_channels, filter_depth, filter_height, filter_width, in_channels].
 * height (H) and width (W) dimensions must be in [1, 511].
 * depth (D) dimension must be in [1, 255].
 * @li out_backprop: A Tensor. Must have the same type as x.
 * 5-D with shape [batch, out_depth, out_height, out_width, out_channels]
 * or [batch, out_channels, out_depth, out_height, out_width].
 * Gradients with respect to the output of the convolution. \n

*@par Attributes:
 * @li strides: Required. A tuple/list of 5 integers. Specifies the stride
 * of the sliding window for each dimension of "x".
 * The N and C dimensions must be 1.
 * height (H) and width (W) dimensions must be in [1, 63].
 * depth (D) dimension must be in [1, 255].
 * Has the same format as "x".
 * @li pads: Required. A tuple/list of 6 integers, [front, back, top, bottom,
 * left, right] pads on feature map.
 * height (H) and width (W) and depth (D) dimensions must be in [0, 255].
 * @li dilations: Optional. A tuple/list of 5 integers, The dilation factor
 * for each dimension of input. Defaults to [1, 1, 1, 1, 1]
 * All dimensions must be 1. Has the same format as "x".
 * @li groups: Optional. Number of blocked connections from input channels
 * to output channels. Defaults to 1. Currently when groups > 1, only supports
 * groups=in_channels=out_channels and input_data_type equals to bfloat16.
 * @li data_format: Optional. An string from: "NDHWC", "NCDHW".
 * Defaults to "NDHWC". Specify the data format of the input and output data. \n

*@par Outputs:
 * y: A Tensor that has the type float16, float32, and the format
 * is NDHWC, NCDHW or DHWCN. \n

*@par Third-party framework compatibility
 * Compatible with Tensorflow's conv3d_backprop_filter
*/
REG_OP(Conv3DBackpropFilterV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(filter_size, TensorType({DT_INT32}))
    .INPUT(out_backprop, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NDHWC")
    .OP_END_FACTORY_REG(Conv3DBackpropFilterV2)

/**
*@brief Computes the gradients of convolution2D with respect to the filter
*@par Inputs:
 * @li x: A Tensor. Must be one of the following types: float16, float32, bfloat16.
 * 4-D with shape [batch, in_height, in_width, in_channels]
 * or [batch, in_channels, in_height, in_width].
 * @li filter_size: A Tensor of type int32. An integer vector representing the
 * tensor shape of filter, where filter is a 4-D tensor
 * [filter_height, filter_width, in_channels, out_channels]
 * [out_channels, in_channels, filter_height, filter_width]
 * or [out_channels, filter_height, filter_width, in_channels].
 * height (H) and width (W) dimensions must be in [1, 511].
 * @li out_backprop: A Tensor. Must have the same type as x.
 * 4-D with shape [batch, out_height, out_width, out_channels]
 * or [batch, out_channels, out_height, out_width].
 * Gradients with respect to the output of the convolution. \n

*@par Attributes:
 * @li strides: Required. A tuple/list of 4 integers. Specifies the stride
 * of the sliding window for each dimension of "x".
 * The N and C dimensions must be 1.
 * height (H) and width (W) dimensions must be in [1, 63].
 * Has the same format as "x".
 * @li pads: Required. A tuple/list of 6 integers, [top, bottom, left, right] pads on feature map.
 * height (H) and width (W) dimensions must be in [0, 255].
 * @li dilations: Optional. A tuple/list of 4 integers, The dilation factor
 * for each dimension of input. Defaults to [1, 1, 1, 1]
 * All dimensions must be 1. Has the same format as "x".
 * @li groups: Optional. Number of blocked connections from input channels
 * to output channels. Defaults to 1. Currently when groups > 1, only supports
 * groups=in_channels=out_channels and input_data_type equals to bfloat16.
 * @li data_format: Optional. An string from: "NHWC", "NCHW".
 * Defaults to "NHWC". Specify the data format of the input and output data. \n

*@par Outputs:
 * y: A Tensor that has the type float16, float32, and the format
 * is NHWC, NCHW or HWCN. \n

*@par Third-party framework compatibility
 * Compatible with Tensorflow's conv3d_backprop_filter
*/
REG_OP(Conv2DBackpropFilterV3)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(filter_size, TensorType({DT_INT32}))
    .INPUT(out_backprop, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NHWC")
    .OP_END_FACTORY_REG(Conv2DBackpropFilterV3)

/**
* @brief GemmV2 matrix "a" by matrix "b" and add matrix "c", producing "alpha * op(a) * op(b) + beta * op(c)". \n
* @par Inputs:
* Four inputs, including:
* @li a: A matrix Tensor. 2D. Must be one of the following types: bfloat16,
* float16. Has format [ND].
* @li b: A matrix Tensor. 2D. Must be one of the following types: bfloat16,
* float16. Has format [ND].
* @li alpha: A 1D Tensor. Must be one of the following types: bfloat16,
* float16. Has format [ND]. \n
* @li beta: A 1D Tensor. Must be one of the following types: bfloat16,
* float16. Has format [ND]. \n
* @li c: A matrix Tensor. 2D. Must be one of the following types: bfloat16,
* float16. Has format [ND].

* @par Attributes:
* @li transpose_a: A bool. If True, changes the shape of "a" from [M, K] to
* [K, M] before multiplication.
* @li transpose_b: A bool. If True, changes the shape of "b" from [K, N] to
* [N, K] before multiplication. \n

* @par Outputs:
* c: The result matrix Tensor. 2D. Must be one of the following types: bfloat16,
* float16. Has format [ND]. \n
*/
REG_OP(GemmV2)
    .INPUT(a, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(b, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(alpha, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(beta, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(c, TensorType({DT_FLOAT}))
    .OUTPUT(c, TensorType({DT_FLOAT}))
    .ATTR(transpose_a, Bool, false)
    .ATTR(transpose_b, Bool, false)
    .OP_END_FACTORY_REG(GemmV2)
/**
* @brief GroupedMatmulSwiglu". \n
* @par Inputs:
* Four inputs, including:
* @li x: A matrix Tensor. 2D. Must be one of the following types: int8.
* Has format [ND].
* @li weight: A matrix Tensor. 5D. Must be one of the following types: int8.
* Has format [NZ].
* @li perChannelScale: A 2D Tensor. Must be one of the following types: bfloat16,
* float32, float16. Has format [ND]. \n
* @li perTokenScale: A 1D Tensor. Must be one of the following types: float32.
* Has format [ND]. \n
* @li groupList: A matrix Tensor. 1D. Must be one of the following types: int64.
* Has format [ND].

* @par Outputs:
* @li quantOutput: The result matrix Tensor. 2D. Must be one of the following types: int8.
* Has format [ND]. \n
* @li quantScaleOutput: The result matrix Tensor. 2D. Must be one of the following types: float32.
* Has format [ND]. \n
*/
REG_OP(GroupedMatmulSwiglu)
    .INPUT(x, TensorType({DT_INT8}))
    .INPUT(weight, TensorType({DT_INT8}))
    .INPUT(perChannelScale, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(perTokenScale, TensorType({DT_FLOAT}))
    .INPUT(groupList, TensorType({DT_INT64}))
    .OUTPUT(quantOutput, TensorType({DT_INT8}))
    .OUTPUT(quantScaleOutput, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(GroupedMatmulSwiglu)

/**
* @brief This operation samples input x by using interpolation based on flow field grid,
  which is usually generated by affine_grid.

* @par Inputs:
* @li x: 4-D Tensor. If the tensor is 4-D, the shape is `[batch, channels, height, width]` or
  `[batch, height, width, channels]`. The position of the channels axis depends on the 'channel_last' attribute.
* @li grid: flow field grid, 4-D Tensor with shape `[batch, output_height, output_width, 2]`.

* @par Attributes:
* @li interpolation_mode: An optional string specifying the interpolation method.
  The default value is "bilinear". Currently, only support "bilinear" and "bicubic".
* @li padding_mode: An optional string specifying the pad method,
  either "zeros", "border", or "reflection". The default value is "zeros".
* @li align_corners: An optional bool. If "true", the centers of the corner
  pixels of the input and output tensors are aligned. Defaults to "false".
* @li channel_last: An optional bool specifying the intput x shape. If "true", the x shape is
`[batch, height, width, channels]`, else if "false", the x shape is `[batch, channels, height, width]`. Defaults to "false" .
* @li scheduler_mode: An optional int. 0: general; 1: sliding window.
  The value 1 is available only in the channel last scenario. The default value is 0.

* @par Outputs:
* y: Returns 4-D Tensor with the same dtype as `x`, the shape is `[batch, channels, output_height, output_width]`.

* @attention Constraints:
* @li The input value of the grid multiplied by the image (length or width) is
  greater than a 24 bit binary number (16777216), there may be errors in the sampling points,
  and the accuracy may be biased.
* @li If the grid contains data beyond the range of [-1, 1], errors may occur
  in the calculation of data in the small value range when bicubic interplation
  is used, and the precision may be inaccurate.
* @li If the grid contains a large amount of data that exceeds the range of [-1, 1],
  a large number of duplicate values in the calculation result will be obtained
  when the zeros or border padding policy is used.
* @li When bilinear or bicubic interpolation is used, the workspace memory
  is required for the float16 data type.

* @par Restrictions:
* Warning:THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(GridSample)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(grid, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(interpolation_mode, String, "bilinear")
    .ATTR(padding_mode, String, "zeros")
    .ATTR(align_corners, Bool, false)
    .ATTR(channel_last, Bool, false)
    .ATTR(scheduler_mode, Int, 1)
    .OP_END_FACTORY_REG(GridSample)

/**
* @brief AdaLayerNorm operator interface implementation
*  calculating: x, scale, shift
*  mean  = np.mean(x, reduce_axis, keepdims=True)
*  rstd = np.rsqrt(np.mean(np.power((x - mean),2), reduce_axis, keepdims=True) + epsilon))
*  y = ((x - mean) * rstd) * (1 + scale) + shift \n

*@par Inputs:
*Three inputs, including:
* @li x: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li scale: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li shift: A Tensor. Must be one of the following types: float16, float32, bfloat16. \n

*@par Attributes:
* @li epsilon: A optional attribute, the type is float32. Defaults to 1e-5 . \n

*@par Outputs:
*Four outputs, including:
* @li y: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li ln_res: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li mean: A Tensor. Must be one of the following types: float32.
* @li rstd: A Tensor. Must be one of the following types: float32. \n
*/
REG_OP(AdaLayerNorm)
    .INPUT(x, "T1")
    .INPUT(scale, "T1")
    .INPUT(shift, "T1")
    .OUTPUT(y, "T1")
    .OUTPUT(ln_res, "T1")
    .OUTPUT(mean, "T2")
    .OUTPUT(rstd, "T2")
    .ATTR(epsilon, Float, 0.00001f)
    .DATATYPE(T1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T2, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(AdaLayerNorm)

/**
* @brief AdaLayerNormGrad operator interface implementation
*  calculating: dy, x, mean, rstd, scale, ln_res
*  pd_xl = dy * (1 + scale)
*  pd_var = np.sum(((-0.5)*pd_xl*(x - data_mean)
*           np.power(rstd, 3)),
*           reduce_axis, keepdims=True)
*  pd_mean = np.sum(((-1.0)*pd_xl*rstd), reduce_axis, keepdims=True)
*            + pd_var*(1.0/m)
*            np.sum(((-2.0)*(x - data_mean)), reduce_axis, keepdims=True)
*  dx = pd_xl*rstd +
*         pd_var*(2.0/m)*(x - data_mean) + pd_mean*(1.0/m)
*  dscale = np.sum(dy * ln_res, param_axis, keepdims=True)
*  dshift = np.sum(dy, param_axis, keepdims=True) \n

*@par Inputs:
*Six inputs, including:
* @li dy: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li x: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li mean: A Tensor. Must be one of the following types: float32.
* @li rstd: A Tensor. Must be one of the following types: float32.
* @li scale: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li ln_res: A Tensor. Must be one of the following types: float16, float32, bfloat16. \n

*@par Outputs:
*Three outputs, including:
* @li dx: A Tensor. Must be one of the following types: float16, float32, bfloat16.
* @li dshift: A Tensor. Must be one of the following types: float32.
* @li dscale: A Tensor. Must be one of the following types: float32. \n
*/
REG_OP(AdaLayerNormGrad)
    .INPUT(dy, "T1")
    .INPUT(x, "T1")
    .INPUT(mean, "T2")
    .INPUT(rstd, "T2")
    .INPUT(scale, "T1")
    .INPUT(ln_res, "T1")
    .OUTPUT(dx, "T1")
    .OUTPUT(dshift, "T2")
    .OUTPUT(dscale, "T2")
    .DATATYPE(T1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .DATATYPE(T2, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(AdaLayerNormGrad)

/**
* @brief amp_update_scale.
*
* @par Inputs:
* @li current_scale: The current loss scaling factor. Must be float32.
* @li growth_tracker: Tracks the number of steps without overflow. Must be int32.
* @li found_inf: Include a boolean value indicating whether an overflow
* occurred during forward and backward propagation. Must be float32.
*
* @par Outputs:
* updates_current_scale: Updated loss scaling factor. Must be float32.
* updated_growth_tracker: Updated growth tracker. Must be float32.

* @par Attributes:
* @li growth_factor: used to multiply and increase the loss scaling factor.
* Must be float.
* @li backoff_factor: used to multiply and decrease the loss scaling factor.
* Must be float.
* @li growth_interval: the number of steps without overflow before increasing
* the loss scaling factor. Must be int.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(AmpUpdateScale)
    .INPUT(current_scale, TensorType({DT_FLOAT}))
    .INPUT(growth_tracker, TensorType({DT_INT32}))
    .INPUT(found_inf, TensorType({DT_FLOAT}))
    .OUTPUT(updated_scale, TensorType({DT_FLOAT}))
    .OUTPUT(updated_growth_tracker, TensorType({DT_INT32}))
    .REQUIRED_ATTR(growth_factor, Float)
    .REQUIRED_ATTR(backoff_factor, Float)
    .REQUIRED_ATTR(growth_interval, Int)
    .OP_END_FACTORY_REG(AmpUpdateScale)

/**
* @brief Calculates the reversed outputs of the function "embedding". \n

* @par Inputs:
* Two inputs, including:
* @li grad: A mutable Tensor of word grad. Must be one of the following types:
*     float32
* @li sort_indices: A mutable word index Tensor of the int32 type.\n
* @li pos_idx: A mutable position of sort_indices in the grad Tensor of the int32 type.\n

* @par Attributes:
* @li num_weights: An int attr which use to judge how many words in dict. \n

* @li padding_idx: An int attr judge which word to fill zeros. Defaults to "-1". \n

* @li scale_grad_by_freq: An optional bool. Defaults to "False".
*     If "True", "grad_weight" will be scale by word_frequency.
*     If "False", "grad_weight" will not be scale by word_frequency. \n

* @par Outputs:
* y: A mutable output Tensor of new word grad has the same type as "grads". \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator EmbeddingDenseGradV2.
*/
REG_OP(EmbeddingDenseGradV2)
    .INPUT(grad, TensorType({ DT_FLOAT32 }))  /* "First operand." */
    .INPUT(sort_indices, TensorType({ DT_INT32 }))  /* "Second operand." */
    .INPUT(pos_idx, TensorType({ DT_INT32 }))  /* "Thrid operand." */
    .OUTPUT(y, TensorType({ DT_FLOAT32 }))  /* "Result, has same element type as two inputs" */
    .REQUIRED_ATTR(num_weights, Int)
    .ATTR(padding_idx, Int, -1)
    .ATTR(scale_grad_by_freq, Bool, false)
    .OP_END_FACTORY_REG(EmbeddingDenseGradV2)

/**
* @brief Conv3DV2 performs a 3-dimensional convolution, using high-level AscendC API\n
*
* @par Inputs:
* @li x: Input feature map. A tensor of type bfloat16 of shape [NCDHW].
* @li filter: Input filter/weights sliding window. A tensor of type bfloat16 of shape [NCDHW].
* Only cases with filter.dim(3) <= 511 and filter.dim(4) <= 511 are supported.
* @li bias: Optional input bias. A tensor of type bfloat16 of shape [ND].
* @li offset_w: Optional input offset w. A tensor of type int8. Currently not supported.
*
* @par Attributes:
* @li strides: A required list of 5 ints, specifying the stride of the sliding window.
* The strides of the N and C dimensions are 1.
* The strides of the D, H, W dimensions are positive integers.
* The strides of H and W dimensions must be within the range [1, 63].
* @li pads: A required list of 5 ints, specifying the padding of the input feature map.
* The pads of the N and C dimensions are 1.
* The pads of the D, H, W dimensions are positive integers within the range [1, 255].
* @li dilations: An optional list of 5 ints, specifying the padding of the input feature map.
* The dilations of the N and C dimensions are 1.
* The dilations of the D, H, W dimensions are positive integers within the range [1, 255].
* The default value is [1, 1, 1, 1, 1].
* @li groups: An optional int parameter. Number of blocked connections from input channels
* to output channels. The default value is 1, and the only supported value.
* @li data_format: An optional string parameter, indicating the input and output data format.
* The default value is "NCDHW" and the only supported one.
* @li groups: An optional int parameter. The default value is 0, and the only supported value.
* @par Outputs:
*/
REG_OP(Conv3DV2)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT, DT_FLOAT16}))
    .INPUT(filter, TensorType({DT_BF16, DT_FLOAT, DT_FLOAT16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT, DT_FLOAT16}))
    .REQUIRED_ATTR(strides, ListInt)
    .REQUIRED_ATTR(pads, ListInt)
    .ATTR(dilations, ListInt, {1, 1, 1, 1, 1})
    .ATTR(groups, Int, 1)
    .ATTR(data_format, String, "NCDHW")
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(Conv3DV2)

/**
 * @brief The fusion operator of RMSNorm, RotaryPositionEmbedding and Update KVCache.
 *
 * @par Inputs:
 * @li kv: A Tensor. The type support float16. Format: ND.
 * @li gamma: A Tensor, used in RMS Norm. The type support float16. Format: ND.
 * @li cos: A Tensor, from position embedding. The type support float16. Format: ND.
 * @li sin: A Tensor, from position embedding. The type support float16. Format: ND.
 * @li index: A Tensor. The type support int64. Format: ND.
 * @li k_cache: A Tensor. The type support float16. Format: ND.
 * @li v_cache: A Tensor. The type support float16. Format: ND.
 *
 * @par Outputs:
 * @li k_cache: A Tensor. The type support float16. Format: ND.
 * @li v_cache: A Tensor. The type support float16. Format: ND.
 *
 * @par Attributes:
 * epsilon: A float32. The epsilon value for RMSNorm. Default: 1e-5.
 *
 * @par Restrictions:
 * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
 */
REG_OP(KvRmsNormRopeCache)
    .INPUT(kv, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(cos, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(sin, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(index, TensorType({DT_INT64}))
    .INPUT(k_cache, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8}))
    .INPUT(ckv_cache, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8}))
    .OPTIONAL_INPUT(k_rope_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(c_kv_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(k_rope_offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(c_kv_offset, TensorType({DT_FLOAT}))
    .OUTPUT(k_cache, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(ckv_cache, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(k_rope, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(c_kv, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(epsilon, Float, 1e-5)
    .ATTR(cache_mode, String, "Norm")
    .ATTR(is_output_kv, Bool, false)
    .OP_END_FACTORY_REG(KvRmsNormRopeCache)

/**
 * @brief The fusion operator of Interleave RotaryPositionEmbedding.
 *
 * @par Inputs:
 * @li x: A tensor. The type support float16, bfloat16. Format: ND.
 * @li cos: A tensor. The type support float16, bfloat16. Format: ND.
 * @li sin: A tensor. The type support float16, bfloat16. Format: ND.
 *
 * @par Outputs:
 * y: A tensor. The type support float16, bfloat16. Format: ND.
 *
 * @par Restrictions:
 * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
 */

 REG_OP(InterleaveRope)
     .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
     .INPUT(cos, TensorType({DT_FLOAT16, DT_BF16}))
     .INPUT(sin, TensorType({DT_FLOAT16, DT_BF16}))
     .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
     .OP_END_FACTORY_REG(InterleaveRope)


  /**
   * @brief compute softmax and topk for moe input.
   * @par Inputs:
   * @li x: A Tensor. Shape is: (B*S, E). Type is:BFloat16, Float16 or Float32. Format support ND.
   * @li bias: A Tensor. Shape is: (E). Type is:BFloat16, Float16 or Float32. The data type must be the same as that of x.
   *     Format support ND. Currently, passing a value is required.
   * @par Outputs:
   * @li y: A Tensor. Type is:BFloat16, Float16 or Float32. The data type must be the same as that of x.
         The size of the non-1 axis must be the same as that of the corresponding axis of x.
         The size of the -1 axis must be the same as that of k. Format support ND.
   * @li expert_idx: A Tensor. Type is:Int32. The shape must be the same as that of y. Format support ND.
   * @li out: A Tensor. Type is:Float32. The shape must be the same as that of x. Format support ND.
   * @par Attributes:
   * @li k: Required parameter. Type is:Int32. The value must greater than 0 and less than or equal to the size
         of the -1 axis of x, and k must not greater than 32.
   * @li k_group: Optional parameter. Type is:Int32. Currently only support 4.
   * @li group_count: Optional parameter. Type is:Int32. Currently only support 8.
   * @li group_select_mode: Optional parameter. Type is:Int32. 0-Select the maximum value in the group;
   *     1-Select the sum of the top two values in the group.Currently only support 1.
   * @li renorm: Optional parameter. Type is:Int32. 0-No renormalization; 1-Apply L1 renormalization. Currently only support 0.
   * @li norm_type: Optional parameter. Type is:Int32. 0-Use softmax; 1-Use sigmoid. Currently only support 1.
   * @li out_flag: Optional parameter. Type is:Bool. True-Output the normalization result;
   *     False-Do not output the normalization result.Currently only support False.
   * @li routed_scaling_factor: Optional parameter. Type is:Float32.
   * @li eps: Optional parameter. Type is:Float32.
   * @par Restrictions:
   * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
   */
REG_OP(MoeGatingTopK)
  .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
  .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
  .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
  .OUTPUT(expert_idx, TensorType({DT_INT32}))
  .OUTPUT(out, TensorType({DT_FLOAT}))
  .REQUIRED_ATTR(k, Int)
  .ATTR(k_group, Int, 1)
  .ATTR(group_count, Int, 1)
  .ATTR(group_select_mode, Int, 0)
  .ATTR(renorm, Int, 0)
  .ATTR(norm_type, Int, 0)
  .ATTR(out_flag, Bool, false)
  .ATTR(routed_scaling_factor, Float, 1.0)
  .ATTR(eps, Float, 1e-20)
  .OP_END_FACTORY_REG(MoeGatingTopK)


  /**
   * @brief compute init routing for moe.
   * @par Inputs:
   * @li x: A 2D tensor. Shape is: (B*S, H). Type is:Int8, BFloat16, Float16 or Float32. Format support ND.
   * @li expert_idx: A 2D tensor. Shape is: (B*S, K). Type is:Int32. Expert index. Format support ND.
   * @li scale: A 1D or 2D tensor. Shape is: (B*S) or (B*S, H). Type is:Float32. Format support ND.
   * @li offset: A 2D tensor. Shape is: (expert_end - expert_start, 1) or (expert_end - expert_start, H).
                 Type is:Float32. Format support ND.
   * @par Outputs:
   * @li expanded_x: A 2D tensor. Shape is: (B*S*K, H). Type is:Int8, BFloat16, Float16 or Float32. 
                     The data type must be the same as that of x. Format support ND.
   * @li expanded_row_idx: A 1D tensor. Shape is: (B*S*K). Type is:Int32. Format support ND.
   * @li expert_tokens_count_or_cumsum: A 1D tensor. represents the number of tokens processed by each expert and the
                                        cumulative value. The value is controlled by expert_tokens_num_flag to output.
                                        Type is:Int64. shape is (expert_end - expert_start, ). Format support ND.
   * @li expanded_scale: A 1D tensor. Shape is: (B*S*K). Type is:Float32. 
                         The data type must be the same as that of scale. Format support ND.
   * @par Attributes:
   * @li active_num: Optional parameter. Type is:Int32. identify activate scenario. The value 0 indicates a non-active
   *                 scenario, and a value greater than 0 indicates an active scenario. In the active scenario, the size
   *                 of axis 0 of grad_expanded_x must be equal to the value of active_num. Default: -1.
   * @li expert_capacity: Optional parameter. Type is:Int32. The max tokens count of every expert. Default: -1.
   * @li expert_num: Optional parameter. Type is:Int32. Number of experts. Default: -1.
   * @li drop_pad_mode: Optional parameter. Type is:Int32. The value is 0(dropless) or 1(dropPad). Default: 0.
   * @li expert_tokens_num_type: Optional parameter. Type is:Int32. The value is 0(compute tokens cumsum) or
                                 1(compute tokens count), which in dropPad scenario. Default: 0.
   * @li expert_tokens_num_flag: Optional parameter. Type is:Bool. The value is true (compute tokens) or
                                 false(do not compute tokens), which in dropPad scenario. Default: false.
   * @li quant_mode: Optional parameter. Type is:Int. The value is -1(no quant) or 0(static quant) or 1(dynamic quant). Default: -1.
   * @li active_expert_range: Optional parameter. Type is:ListInt. Like [expert_start, expert_end].
                              expert_start must be greater than or equal to 0, expert_end must be less than or equal to 10240,
                              expert_start must be less than expert_end. Default: [].
   * @li row_idx_type: Optional parameter. Type is:Int. The value is 0(gather) or 1(scatter). Default: 0.
   */
  REG_OP(MoeInitRoutingV3)
  .INPUT(x, TensorType({DT_INT8, DT_FLOAT16, DT_FLOAT, DT_BF16}))
  .INPUT(expert_idx, TensorType({DT_INT32}))
  .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT}))
  .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
  .OUTPUT(expanded_x, TensorType({DT_INT8, DT_FLOAT16, DT_FLOAT, DT_BF16}))
  .OUTPUT(expanded_row_idx, TensorType({DT_INT32}))
  .OUTPUT(expert_tokens_count_or_cumsum, TensorType({DT_INT64}))
  .OUTPUT(expanded_scale, TensorType({DT_FLOAT}))
  .ATTR(active_num, Int, -1)
  .ATTR(expert_capacity, Int, -1)
  .ATTR(expert_num, Int, -1)
  .ATTR(drop_pad_mode, Int, 0)
  .ATTR(expert_tokens_num_type, Int, 0)
  .ATTR(expert_tokens_num_flag, Bool, false)
  .ATTR(quant_mode, Int, -1)
  .ATTR(active_expert_range, ListInt, {})
  .ATTR(row_idx_type, Int, 0)
  .OP_END_FACTORY_REG(MoeInitRoutingV3)


/**
* @brief Combine Dequant + Swiglu + Quant.

* @par Inputs:
* Seven inputs, including:
* @li x: A tensor. Shape is (X..., H), dim must > 2, and H must be even.
* @li weight_scale: Dequantization scale of weight. An optional tensor. Type is float32. Shape is (1..., H).
* @li activation_scale: Dequantization scale of activation. An optional tensor. Type is float32. Shape is  (X..., 1).
* @li bias: Bias for matmul. An optional tensor. Type is float16/bfloat16/int32/float32. Shape is (X..., H).
* @li quant_scale: Quantized scale. An optional tensor. Type is float16/bfloat16/float32. Shape is (1..., H).
* @li quant_offset: Quantized offset. An optional tensor. Type is float16/bfloat16/float32. Shape is (1..., H).
* @li group_index: Mean group index. An optional tensor. Type is int32/int64. Shape is (1,). \n

* @par Outputs:
* @li y: A tensor. Type is int8.
* @li scale: A tensor. Type is float32.

* @par Attributes:
* @li activate_left: Type is bool.
* The swi activate_left algorithm to use:
*     'false'(activate right) or 'true'(activate left), defalut is 'false'(activate right).
* @li quant_mode: Type is string. The quant mode to use: 'static' or 'dynamic', defalut is 'static'.

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(DequantSwigluQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16, DT_INT32}))
    .OPTIONAL_INPUT(weight_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(activation_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16, DT_INT32, DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(quant_offset, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(group_index, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_INT8}))
    .OUTPUT(scale, TensorType({DT_FLOAT}))
    .ATTR(activate_left, Bool, false)
    .ATTR(quant_mode, String, "static")
    .OP_END_FACTORY_REG(DequantSwigluQuant)


/**
* @brief Fused Operator of AddRmsNorm and DynamicQuant . \n

* @par Inputs:
* @li x1: A tensor of type float16/bfloat16. Supported format "ND". \n
* @li x2: A tensor of type float16/bfloat16. Supported format "ND". \n
* @li gamma: A tensor of type float16/bfloat16. Supported format "ND". \n
* @li smooth_scale1: Optional Input. A tensor of type float16/bfloat16. Supported format "ND". \n
* @li smooth_scale2: Optional Input. A tensor of type float16/bfloat16. Supported format "ND". \n

* @par Attributes:
* @li epsilon: A optional float, default value is 1e-6.

* @par Outputs:
* @li y1: A tensor of type int8, quantize result for rmsnorm(x1+x2)*smooth1. Supported format "ND". \n
* @li y2: A tensor of type int8, quantize result for rmsnorm(x1+x2)*smooth2. Supported format "ND". \n
* @li x: A tensor of type float16/bfloat16, describing the result of x1 + x2. Supported format "ND". \n
* @li scale1: A tensor of type float32, describing the result of dynamic quantize scales. Supported format "ND". \n
* @li scale2: A tensor of type float32, describing the result of dynamic quantize scales. Supported format "ND". \n
*/
REG_OP(AddRmsNormDynamicQuant)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(smooth_scale1, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(smooth_scale2, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y1, TensorType({DT_INT8, DT_INT8}))
    .OUTPUT(y2, TensorType({DT_INT8, DT_INT8}))
    .OUTPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(scale1, TensorType({DT_FLOAT, DT_FLOAT}))
    .OUTPUT(scale2, TensorType({DT_FLOAT, DT_FLOAT}))
    .ATTR(epsilon, Float, 1e-6)
    .OP_END_FACTORY_REG(AddRmsNormDynamicQuant)

/**
* @brief AddRmsNormCast operator interface implementation. \n
*  calculating: x1, x2, gamma \n
*  x = x1 + x2 \n
*  rstd = np.rsqrt(np.mean(np.power(x,2), reduce_axis, keepdims=True) + epsilon)) \n
*  y2 = gamma * (x * rstd)
*  y1 = cast16232(y2)

* @par Inputs
* Three inputs, including:
* @li x1: A Tensor. Support dtype: [float16, bfloat16], support format: [ND].
* @li x2: A Tensor. Support dtype: [float16, bfloat16], support format: [ND].
* @li gamma: A Tensor. Support dtype: [float16, bfloat16], support format: [ND].

* @par Attributes
* epsilon: Input eps in the formula, which is used to prevent division-by-zero errors.
* A optional attribute, the type is float. Defaults to 1e-6.

* @par Outputs
* Three outputs, including:
* @li y1: A Tensor. Support dtype: [float32], support format: [ND].
* @li y2: A Tensor. Support dtype: [float16, bfloat16], support format: [ND].
* @li rstd: A Tensor. Support dtype: [float32], support format: [ND].
* @li x: A Tensor. Support dtype: [float16, bfloat16], support format: [ND].
*/
REG_OP(AddRmsNormCast)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y1, TensorType({DT_FLOAT}))
    .OUTPUT(y2, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(rstd, TensorType({DT_FLOAT}))
    .OUTPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .ATTR(epsilon, Float, 1e-6f)
    .OP_END_FACTORY_REG(AddRmsNormCast)

/**
* @brief MoeDistributeDispatch operator interface implementation.

* @par Inputs
* Five inputs, including:
* @li x: A tensor. Support dtype: float16,bfloat16, dimension must be 2. Shape supports (BS, H), support format: ND.
* @li expertIds: A tensor. Support dtype: int32, indicates top k experts of each token, dimension must be 2. Shape supports (BS, K), support format: ND.
* @li scales: An optional tensor. Support dtype: float32, dimension must be 2, support format: ND.
* @li x_active_mask: An optional tensor. Support dtype: bool, support format: ND.
* @li expert_scales: An optional tensor. Support dtype: float32. Shape supports (BS, K), support format: ND.

* @par Attributes
* @li group_ep: Required. Input ep comm group name, ep means experts parallelism, dtype: String.
* @li ep_world_size: Required. Input ep comm world size, dtype: int64.
* @li ep_rank_id: Required. Input ep comm rank Id, dtype: int64.
* @li moe_expert_num: Required. Input moe expert num, dtype: int64.
* @li group_tp: Input tp comm group name, tp means tensor parallelism, dtype: String.
* @li tp_world_size: Input tp comm world size, dtype: int64.
* @li tp_rank_id: Input tp comm rank Id, dtype: int64.
* @li expert_shard_type: Input moe shard type, dtype: int64.
* @li shared_expert_num: Input shared expert num, dtype: int64.
* @li shared_expert_rank_num: Input shared expert rank num, dtype: int64.
* @li quant_mode: Input quant mode. The options are 0 (non-quantization), 1 (static quantization), and 2 (dynamic quantization). dtype: int64.
* @li global_bs: Input global batch size, dtype: int64.
* @li expert_token_nums_type: Input expert token nums type, dtype: int64.

* @par Outputs
* Seven outputs, including:
* @li expand_x: A tensor. Result of each expert after dispatching. Support dtype: float16,bfloat16,int8. Shape supports (A, H), support format: ND.
* @li dynamic_scales: If quant is enabled, scale value of each token. A tensor. Support dtype: float32. Shape supports (A, ), support format: ND.
* @li expand_idx: A tensor. Support dtype: int32. Shape supports (BS*K, ), support format: ND.
* @li expert_token_nums: A tensor. Tokens nums of expand_x. Support dtype: int64, support format: ND.
* @li ep_recv_count: A tensor. Received token nums after dispatching. Support dtype: int32, support format: ND.
* @li tp_recv_count: A tensor. Received token nums after allgather. Support dtype: int32, support format: ND.
* @li expand_scales: A tensor. Scales of each token to sum for combine. Support dtype: float32. Shape supports (A, ), support format: ND.
*/
REG_OP(MoeDistributeDispatch)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(expert_ids, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(scales, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(x_active_mask, TensorType({DT_BOOL}))
    .OPTIONAL_INPUT(expert_scales, TensorType({DT_FLOAT}))
    .OUTPUT(expand_x, TensorType({DT_BF16, DT_INT8, DT_FLOAT16}))
    .OUTPUT(dynamic_scales, TensorType({DT_FLOAT}))
    .OUTPUT(expand_idx, TensorType({DT_INT32}))
    .OUTPUT(expert_token_nums, TensorType({DT_INT64}))
    .OUTPUT(ep_recv_count, TensorType({DT_INT32}))
    .OUTPUT(tp_recv_count, TensorType({DT_INT32}))
    .OUTPUT(expand_scales, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(group_ep, String)
    .REQUIRED_ATTR(ep_world_size, Int)
    .REQUIRED_ATTR(ep_rank_id, Int)
    .REQUIRED_ATTR(moe_expert_num, Int)
    .ATTR(group_tp, String, "")
    .ATTR(tp_world_size, Int, 0)
    .ATTR(tp_rank_id, Int, 0)
    .ATTR(expert_shard_type, Int, 0)
    .ATTR(shared_expert_num, Int, 1)
    .ATTR(shared_expert_rank_num, Int, 0)
    .ATTR(quant_mode, Int, 0)
    .ATTR(global_bs, Int, 0)
    .ATTR(expert_token_nums_type, Int, 1)
    .OP_END_FACTORY_REG(MoeDistributeDispatch)

    /**
    * @brief Return values and indices where values contains the median of each row of input in the dim,
    * and indices contains the index of the median values found in the dim.
    * @par Inputs:
    * One input:
    * @li self: A Tensor with any format. Support float, float16.

    * @par Attributes:
    * dim: An optional int32, specifying the dim to reduce. Defaults to -1.
    * keepdim: An optional bool, specifying whether the output has dim retained or not.
    * 
    * @par Outputs:
    * valuesOut: A Tensor, which is the same dtype as self. Support float, float16.
    * indicesOut: A Tensor, which contains the index of the median values found in the dim. Support int32.
    * @attention Constraints:
    * "dim" must be within the rank of the input tensor.

    * @par Third-party framework compatibility
    * Compatible with the PyTorch operator the median.
    */
    REG_OP(MedianDim)
        .INPUT(self, TensorType({ DT_FLOAT, DT_FLOAT16 }))
        .OUTPUT(valuesOut, TensorType({ DT_FLOAT, DT_FLOAT16 }))
        .OUTPUT(indicesOut, TensorType({ DT_INT32 }))
        .ATTR(dim, Int, -1)
        .ATTR(keepdim, Bool, false)
        .OP_END_FACTORY_REG(MedianDim)

/**
* @brief MoeDistributeCombine operator interface implementation.

* @par Inputs
* Ten inputs, including:
* @li expand_x: A tensor. Support dtype: float16, bfloat16, dimension must be 2, Support Shape: (A * world_size, H), support format: ND.
* @li expert_ids: A tensor. Support dtype: int32, dimension must be 2, Support Shape: (BS, K), support format: ND.
* @li expand_idx: A tensor. Support dtype: int32, dimension must be 1, Support Shape: (BS*K, ), support format: ND.
* @li ep_send_counts: A tensor. Support dtype: int32, Support Shape: (expert_nums + 2 * globalBs * K * server_num, ), support format: ND.
* @li expert_scales: A tensor. Support dtype: float32, Support Shape: (BS, K), support format: ND.
* @li tp_send_counts: A tensor. Support dtype: int32, support format: ND.
* @li x_active_mask: An optional tensor. Support dtype: bool, support format: ND.
* @li activation_scale: An optional tensor. Support dtype: float32, support format: ND.
* @li weight_scale: An optional tensor. Support dtype: float32, support format: ND.
* @li group_list: An optional tensor. Support dtype: int64, support format: ND.
* @li expand_scales: A tensor. Support dtype: float32, Support Shape: (A, ), support format: ND.

* @par Attributes
* @li group_ep: Input ep comm group name, ep means experts parallelism, dtype: String.
* @li ep_world_size: Input ep comm world size, dtype: Int64.
* @li ep_rank_id: Input ep comm rank Id, dtype: Int64.
* @li moe_expert_num: Input moe expert num, dtype: Int64.
* @li group_tp: Input tp comm group name, tp means tensor parallelism, dtype: String.
* @li tp_world_size: Input tp comm world size, dtype: Int64.
* @li tp_rank_id: Input tp comm rank Id, dtype: Int64.
* @li expert_shard_type: Input moe shard type, dtype: Int64.
* @li shared_expert_num: Input shared expert num, dtype: Int64.
* @li shared_expert_rank_num: Input shared expert rank num, dtype: Int64.
* @li global_bs: Input global batch size, dtype: Int64.
* @li out_dtype: Dtype of output, 0 for bfloat16, 1 for float16, dtype: Int64.
* @li comm_quant_mode: communication quantization mode, 1 for enable, 0 for disable, dtype: Int64.
* @li group_list_type: type of input group_list, dtype: Int64.

* @par Outputs
* One outputs, including:
* @li x: A tensor. Result of combine. Support dtype: float16,bfloat16,  Support Shape: (BS, H), support format: ND.
*/
REG_OP(MoeDistributeCombine)
    .INPUT(expand_x, TensorType({DT_BF16, DT_FLOAT16, DT_INT32}))
    .INPUT(expert_ids, TensorType({DT_INT32}))
    .INPUT(expand_idx, TensorType({DT_INT32}))
    .INPUT(ep_send_counts, TensorType({DT_INT32}))
    .INPUT(expert_scales, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(tp_send_counts, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(x_active_mask, TensorType({DT_BOOL}))
    .OPTIONAL_INPUT(activation_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(weight_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(group_list, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(expand_scales, TensorType({DT_FLOAT}))
    .OUTPUT(x, TensorType({DT_BF16, DT_FLOAT16}))
    .REQUIRED_ATTR(group_ep, String)
    .REQUIRED_ATTR(ep_world_size, Int)
    .REQUIRED_ATTR(ep_rank_id, Int)
    .REQUIRED_ATTR(moe_expert_num, Int)
    .ATTR(group_tp, String, "")
    .ATTR(tp_world_size, Int, 0)
    .ATTR(tp_rank_id, Int, 0)
    .ATTR(expert_shard_type, Int, 0)
    .ATTR(shared_expert_num, Int, 1)
    .ATTR(shared_expert_rank_num, Int, 0)
    .ATTR(global_bs, Int, 0)
    .ATTR(out_dtype, Int, 0)
    .ATTR(comm_quant_mode, Int, 0)
    .ATTR(group_list_type, Int, 0)
    .OP_END_FACTORY_REG(MoeDistributeCombine)

/**
* @brief Rearrange tokens from rank order to expert order
* @par Inputs:
* @li tokens: A 2D tensor, represents tokens in rank-order. Type is BFloat16, Float16 or
      Int8. Shape supports (A, H). Format supports ND.
* @li expert_token_num_per_rank: A 2D tensor, represents numbers of tokens belong to an expert on specific rank.
      Type is Int32 or Int64. Shape supports (N, E). Format supports ND.
* @li per_token_scales: A 1D tensor, optional, represents tokens scale in rank-order. Type is Float32.
      Shape supports (A). Format supports ND.
* @par Outputs:
* @li permute_tokens: A 2D tensor, represents tokens in expert-order. Type is BFloat16, Float16 or
      Int8. Shape supports (A, H). Format supports ND.
* @li permute_per_token_scales: A 1D tensor, represents tokens scale in expert-order. Type is Float32.
      Shape supports (A). Format supports ND.
* @li permute_token_idx: A 1D tensor, represents token idx in rank-order. Type is Int32.
      Shape supports (A). Format supports ND.
* @li expert_token_num: A 1D tensor, represents tokens nums of experts. Type is Int32 or Int64.
      Shape supports (A). Format supports ND.
* @par Attributes:
* @li expert_token_num_type: Optional integer, represents the cumsum or count mode. Type is Int. Default: 1. Value
      supports 0-cumsum or 1-count.
* @li idx_type: Optional integer, represents the gather or scatter index. Type is Int. Default: 0. Value
      supports 0-gather idx or 1-scatter idx.
*/
REG_OP(MoeReRouting)
    .INPUT(tokens, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .INPUT(expert_token_num_per_rank, TensorType({DT_INT32, DT_INT64}))
    .OPTIONAL_INPUT(per_token_scales, TensorType({DT_FLOAT}))
    .OUTPUT(permute_tokens, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .OUTPUT(permute_per_token_scales, TensorType({DT_FLOAT}))
    .OUTPUT(permute_token_idx, TensorType({DT_INT32}))
    .OUTPUT(expert_token_num, TensorType({DT_INT32, DT_INT64}))
    .ATTR(expert_token_num_type, Int, 1)
    .ATTR(idx_type, Int, 0)
    .OP_END_FACTORY_REG(MoeReRouting)

/**
* @brief QuantMatmulDequant operator interface implementation.

* @par Inputs
* @li x: A tensor. Quantized input data in combination operations. Support dtype: float16, dimension must be 2, support format: ND.
* @li quantized_weight: A tensor. Weights used for quantitative calculations. Support dtype: int8, dimension must be 2, support format: NZ.
* @li weight_scale: A tensor. Quantization coefficient for weight. Support dtype: float32, support format: ND.
* @li bias: An optional input bias tensor. Support dtype: int32, support format: ND.
* @li x_scale: A optional tensor. Indicates the quantization coefficient of x. Support dtype: float32, support format: ND.
* @li x_offset: A optional tensor. Indicates the offset of the input x. Support dtype: float32, support format: ND.
* @li smooth_scale: A optional tensor. The smoothing coefficient for x. Support dtype: float16, support format: ND.

* @par Attributes
* @li x_quant_mode: dtype: String. Quantization mode for input x. The value range is [pertoken,pertensor]. The default value is'pertoken'.
* @li transpose_weight: dtype: Bool. Indicates whether the input weight is transposed.

* @par Outputs
* y: A tensor. The result of the combination operation. Has the same dtype and format as x.
* The shape supports 2D, with each dimension representing: (m, n). Where m is consistent with the m of x,
* and n is consistent with the n of quantized_weight.
*/
REG_OP(QuantMatmulDequant)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(quantized_weight, TensorType({DT_INT8}))
    .INPUT(weight_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(x_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(x_offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(smooth_scale, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .ATTR(x_quant_mode, String, "pertoken")
    .ATTR(transpose_weight, Bool, true)
    .OP_END_FACTORY_REG(QuantMatmulDequant)

/**
* @brief QuantGroupedMatmulDequant operator interface implementation.

* @par Inputs
* @li x: A tensor. Quantized input data in combination operations. Support dtype: float16, dimension must be 2, support format: ND.
* @li quantized_weight: A tensor. Weights used for quantitative calculations. Support dtype: int8, dimension must be 3, support format: NZ.
* @li weight_scale: A tensor. Quantization coefficient for weight. Support dtype: float32, support format: ND.
* @li group_list: A tensor. The cumsum result (cumulative sum) of the matmul size distribution representing the x and out group axis direction.
* Support dtype: int64, support format: ND.
* @li bias: An optional input bias tensor. Support dtype: int32, support format: ND.
* @li x_scale: A optional tensor. Indicates the quantization coefficient of x. Support dtype: float32, support format: ND.
* @li x_offset: A optional tensor. Indicates the offset of the input x. Support dtype: float32, support format: ND.
* @li smooth_scale: A optional tensor. The smoothing coefficient for x. Support dtype: float16, support format: ND.

* @par Attributes
* @li x_quant_mode: dtype: String. Quantization mode for input x.
* @li transpose_weight: dtype: Bool. Indicates whether the input weight is transposed.

* @par Outputs
* y: A tensor. The result of the combination operation. Has the same dtype and format as x.
* The shape supports 2D, with each dimension representing: (m, n). Where m is consistent with the m of x,
* and n is consistent with the n of quantized_weight.
*/
REG_OP(QuantGroupedMatmulDequant)
    .INPUT(x, TensorType({DT_FLOAT16}))
    .INPUT(quantized_weight, TensorType({DT_INT8}))
    .INPUT(weight_scale, TensorType({DT_FLOAT}))
    .INPUT(group_list, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(bias, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(x_scale, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(x_offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(smooth_scale, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .ATTR(x_quant_mode, String, "pertoken")
    .ATTR(transpose_weight, Bool, true)
    .OP_END_FACTORY_REG(QuantGroupedMatmulDequant)
}  // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_EXPERIMENT_OPS_H_
