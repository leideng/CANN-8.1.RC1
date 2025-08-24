#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
sigmoid_cross_entropy_with_logits_grad_v2
"""

import functools

import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util


# pylint: disable=raise-missing-from,too-many-arguments,too-many-locals,unused-argument
def _broadcast_shape_check(input_shape, target_shape):
    """
    _broadcast_shape_check
    """
    try:
        shape_util.produce_shapes(input_shape, target_shape)
    except RuntimeError:
        raise RuntimeError("input_shape can't be broadcast to target_shape")


@register_operator_compute("SigmoidCrossEntropyWithLogitsGradV2", op_mode="static", support_fusion=True)
def sigmoid_cross_entropy_with_logits_grad_v2_compute(predict, target, dout, weight, pos_weight, reduction="mean"):
    """
    :param predict: TVM tensor, the placeholder of predict
    :param target: TVM tensor, the placeholder of target
    :param dout: TVM tensor, the placeholder of dout
    :param weight: TVM tensor, the placeholder of weight
    :param pos_weight: TVM tensor, the placeholder of pos_weight
    :param reduction: str, specifies the reduction mode :'none' | 'mean' | 'sum'
    :return: TVM tensor
    """
    predict_shape = shape_util.shape_to_list(predict.shape)
    predict_dtype = predict.dtype

    precision_dtype = "float32"

    if predict.dtype.lower() == "float16":
        predict = tbe.cast_to(predict, precision_dtype)
        target = tbe.cast_to(target, precision_dtype)

    # calculate sigmoid(predict)
    const_num_neg_one = tvm.const(-1, dtype="float32")
    const_num_one = tvm.const(1, dtype="float32")
    tmp_negative = tbe.vmuls(predict, const_num_neg_one)
    tmp_exp = tbe.vexp(tmp_negative)
    tmp_sum = tbe.vadds(tmp_exp, const_num_one)
    tensor_one = tbe.broadcast(tvm.const(1, "float32"), predict_shape)
    sigmoid_res = tbe.vdiv(tensor_one, tmp_sum)

    # calculate the result of gradient = ((log_weight + 1 - target) * sigmoid(predict) - log_weight) * dout
    if pos_weight is not None:
        pos_weight_shape = shape_util.shape_to_list(pos_weight.shape)
        if pos_weight_shape != predict_shape:
            _, _, broadcast_pos_shape = shape_util.produce_shapes(pos_weight_shape, predict_shape)
            pos_weight = tbe.broadcast(pos_weight, broadcast_pos_shape, precision_dtype)

        log_weight = tbe.vmul(pos_weight, target)
        weight_tmp = tbe.vadds(log_weight, tvm.const(1, precision_dtype))
        weight_sub = tbe.vsub(weight_tmp, target)
        grad_tmp = tbe.vmul(weight_sub, sigmoid_res)
        grad_cur = tbe.vsub(grad_tmp, log_weight)
        grad_output = tbe.vmul(grad_cur, dout)
    else:
        grad_cur = tbe.vsub(sigmoid_res, target)
        grad_output = tbe.vmul(grad_cur, dout)

    # calculate the result of gradient = gradient * weight
    if weight is not None:
        weight_shape = shape_util.shape_to_list(weight.shape)
        if weight_shape != predict_shape:
            _, _, broadcast_weight_shape = shape_util.produce_shapes(weight_shape, predict_shape)
            weight = tbe.broadcast(weight, broadcast_weight_shape, precision_dtype)

        grad_output = tbe.vmul(grad_output, weight)

    # calculate the result of gradient = gradient / num
    if reduction == "mean":
        num = functools.reduce(lambda x, y: x * y, predict_shape)
        norm = 1.0 / num
        grad_output = tbe.vmuls(grad_output, norm)

    grad_output = tbe.cast_to(grad_output, predict_dtype)
    return grad_output


def optional_weight(tensor_list, predict_shape, dtype_list, weight, pos_weight):
    """
    optional_weight
    """
    weight_data = None
    pos_weight_data = None
    if weight is not None:
        weight_shape = weight.get("shape")
        weight_dtype = weight.get("dtype").lower()
        para_check.check_dtype(weight_dtype, dtype_list)
        _broadcast_shape_check(weight_shape, predict_shape)

        weight_shape = tuple([1] * (len(predict_shape) - len(weight_shape))) + tuple(weight_shape)
        weight_data = tvm.placeholder(weight_shape, weight_dtype, name="weight_data")
        tensor_list.append(weight_data)

    if pos_weight is not None:
        pos_weight_shape = pos_weight.get("shape")
        pos_weight_dtype = pos_weight.get("dtype").lower()

        para_check.check_dtype(pos_weight_dtype, dtype_list)
        _broadcast_shape_check(pos_weight_shape, predict_shape)

        pos_weight_shape = tuple([1] * (len(predict_shape) - len(pos_weight_shape))) + tuple(pos_weight_shape)
        pos_weight_data = tvm.placeholder(pos_weight_shape, pos_weight_dtype,
                                          name="pos_weight_data")
        tensor_list.append(pos_weight_data)

    return weight_data, pos_weight_data


@para_check.check_op_params(dict, dict, dict, dict, dict, dict, str, str)
def sigmoid_cross_entropy_with_logits_grad_v2(predict, target, dout, weight, pos_weight, gradient,
                                              reduction="mean",
                                              kernel_name="sigmoid_cross_entropy_with_logits_grad_v2"):
    """
    Function: it measures the gradient of Binary Cross Entropy With Logits.
    -----------
    :param predict: dict, shape and dtype of input, required
    :param target: dict,shape and dtype of target, should be same shape and type as predict, required
    :param dout: dict,shape and dtype of dout, should be same shape and type as predict, required
    :param weight: dict,shape and dtype of weight, should be same shape and type as predict, optional
    :param pos_weight: dict,shape and dtype of pos_weight, should be same shape and type as predict, optional
    :param gradient: dict,shape and dtype of target, should be same shape and type as predict, required
    :param reduction: str, specifies the reduction mode: 'none' | 'mean' | 'sum', default to 'mean'
    :param kernel_name: str, kernel name, default to 'sigmoid_cross_entropy_with_logits_grad_v2'
    :return: None
    """
    predict_shape = predict.get("shape")
    predict_dtype = predict.get("dtype").lower()
    target_shape = target.get("shape")
    target_dtype = target.get("dtype").lower()
    dout_shape = dout.get("shape")
    dout_dtype = dout.get("dtype").lower()

    shape_util.compare_tensor_dict_key(predict, target, "shape")
    shape_util.compare_tensor_dict_key(predict, dout, "shape")
    shape_util.compare_tensor_dict_key(predict, target, "dtype")
    shape_util.compare_tensor_dict_key(predict, dout, "dtype")

    dtype_list = ["float16", "float32"]
    para_check.check_dtype(predict_dtype, dtype_list)
    para_check.check_shape(predict_shape)

    reduction_list = ["none", "mean", "sum"]
    if reduction not in reduction_list:
        raise RuntimeError("reduction should be one of ['none','mean','sum']")

    para_check.check_kernel_name(kernel_name)

    tensor_list = []

    predict_data = tvm.placeholder(predict_shape, predict_dtype, name="predict_data")
    target_data = tvm.placeholder(target_shape, target_dtype, name="target_data")
    dout_data = tvm.placeholder(dout_shape, dout_dtype, name="dout_data")

    tensor_list.append(predict_data)
    tensor_list.append(target_data)
    tensor_list.append(dout_data)

    weight_data, pos_weight_data = optional_weight(tensor_list, predict_shape, dtype_list, weight, pos_weight)

    res = sigmoid_cross_entropy_with_logits_grad_v2_compute(predict_data, target_data, dout_data, weight_data,
                                                            pos_weight_data, reduction)

    tensor_list.append(res)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    tbe.build(schedule, config)
