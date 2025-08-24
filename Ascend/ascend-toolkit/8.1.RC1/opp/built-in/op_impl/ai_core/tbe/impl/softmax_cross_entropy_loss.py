#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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
softmax_cross_entropy_loss
"""

from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import shape_util
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm

# compute needed,scalar -1
SCALAR_MINUS_ONE = -1


# 'pylint: disable=unused-argument
@register_operator_compute("softmax_cross_entropy_loss", op_mode="static", support_fusion=True)
def softmax_cross_entropy_loss_compute(
        scores,
        labels,
        weights,
        loss,
        log_prop,
        reduction,
        weights_flag,
        kernel_name="softmax_cross_entropy_loss"):
    """Computes softmax cross entropy cost.
    softmax = e^(x-max) / ∑(e^(x-max))
    log(softmax) = (x-max) - log(∑e^(x-max))
    cross_entropy = -∑(y * log⁡(softmax))

    Parameters
    # ----------
    scores: TVM tensor
        input tensor contains shape and dtype attributes.
        source data type support "float16", "float32".
    labels: TVM tensor
        input tensor contains shape and dtype attributes.
        labels data type support "int32", "int64".
    weights: TVM tensor
        If given, it has to be a 1D Tensor
    loss: dict
        when reduction=none:TVM tensor, output tensor
        when reduction=sum/mean, A Scalar
        Must have the same type as 'scores'.
    log_prop: dict
        data of output.
        Must have the same type as 'scores'.
    reduction: str
        reduce configuration mean/sum/none. Default: mean
    kernel_name: str
        kernel name, default value is "softmax_cross_entropy_loss"

    Returns:
    res: TVM tensor
        output tensor. Has the same type as "scores".
    """
    shape_scores = shape_util.shape_to_list(scores.shape)
    shape_labels = shape_util.shape_to_list(labels.shape)

    dtype = scores.dtype.lower()

    has_improve_precision = False
    if dtype == "float16" and \
            tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        scores = tbe.cast_to(scores, "float32")
        has_improve_precision = True

    data_max = tbe.reduce_max(scores, axis=1, keepdims=True)
    data_max_broadcast = tbe.broadcast(data_max, shape_scores)
    data_sub = tbe.vsub(scores, data_max_broadcast)
    data_exp = tbe.vexp(data_sub)
    data_sum = tbe.sum(data_exp, axis=1, keepdims=True)
    data_sum_broadcast = tbe.broadcast(data_sum, shape_scores)
    data_log_tmp = tbe.vlog(data_sum_broadcast)
    log_prop = tbe.vsub(data_sub, data_log_tmp)

    data_muls = tbe.vmuls(log_prop, SCALAR_MINUS_ONE)
    if weights_flag:
        input_weights = tbe.broadcast(weights, shape_scores)
        loss = tbe.vmul(data_muls, input_weights)
    else:
        loss = data_muls

    if has_improve_precision:
        loss = tbe.cast_to(loss, "float16")
        log_prop = tbe.cast_to(log_prop, "float16")

    res = [loss, log_prop]

    return res


# 'pylint: disable=unused-argument,too-many-locals,invalid-name
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def softmax_cross_entropy_loss(
        scores,
        labels,
        weights,
        loss,
        log_prop,
        ignore_index=0,
        reduction='mean',
        kernel_name="softmax_cross_entropy_loss"):
    """
    Computes softmax cross entropy cost.

    Parameters
    ----------
    scores: dict
        input tensor contains shape and dtype attributes.
        source data type support "float16", "float32".
    labels: dict
        input tensor contains shape and dtype attributes.
        labels data type support "int32", "int64".
    weights: dict
        A manual rescaling weight given to each class.
        If given, it has to be a 1D Tensor assigning weight to each of the classes.
        Otherwise, it is treated as if having all ones.
    loss: dict
        data of output.
        Must have the same type as 'scores'.
    log_prop: dict
        data of output.
        Must have the same type as 'scores'.
    ignore_index : int
        Specifies a target value that is ignored and does not contribute to the input gradient.
        It's an optional value.
    reduction: str (default is mean)
        Type of reduction to apply to loss: none, sum, mean(default). 
        'none': no reduction will be applied,
        'sum': the output will be summed.
        'mean': the sum of the output will be divided by the number of elements in the output.
    kernel_name: str
        kernel name, default value is "softmax_cross_entropy_loss"

    Returns:
    None
    """
    shape_scores = scores.get("shape")
    shape_labels = shape_util.shape_to_list(labels.get("shape"))
    shape_labels.insert(1, 1)
    shape_labels = tuple(shape_labels)
    input_dtype = scores.get("dtype").lower()
    labels_dtype = labels.get("dtype").lower()
    if weights:
        shape_weights = shape_util.shape_to_list(weights.get("shape"))
        list_new = [1 for i in range(len(shape_scores) - 1)]
        list_new.insert(1, shape_weights[0])
        shape_weights = tuple(list_new)
        para_check.check_shape(shape_weights, param_name="weights")
        data_weights = tvm.placeholder(shape_weights, dtype=input_dtype,
                                       name="data_weights")
        weights_flag = True
    else:
        data_weights = None
        weights_flag = False

    para_check.check_shape(shape_scores, param_name="scores")
    para_check.check_shape(shape_labels, param_name="labels")

    check_list = ("float16", "float32", "float64", "bfloat16")
    para_check.check_dtype(input_dtype, check_list, param_name="scores")

    data_scores = tvm.placeholder(shape_scores, dtype=input_dtype,
                                  name="data_scores")
    data_labels = tvm.placeholder(shape_labels, dtype=labels_dtype,
                                  name="data_labels")
    res = softmax_cross_entropy_loss_compute(data_scores,
                                             data_labels,
                                             data_weights,
                                             loss,
                                             log_prop,
                                             reduction,
                                             weights_flag)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    if weights_flag:
        tensor_list = [data_scores, data_labels, data_weights] + list(res)
    else:
        tensor_list = [data_scores, data_labels] + list(res)

    config = {"name": kernel_name,
              "tensor_list": tensor_list}
    tbe.cce_build_code(sch, config)
