#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019 Huawei Technologies Co., Ltd
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
sigmoid_cross_entropy_with_logits_grad
"""
import functools
from tbe.dsl.api import auto_schedule
from tbe.dsl.api import build
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals
@register_operator_compute("sigmoid_cross_entropy_with_logits_grad", op_mode="static", support_fusion=True)
def sigmoid_cross_entropy_with_logits_grad_compute(
        predict,
        target,
        dout,
        gradient,
        kernel_name):
    """calculating sigmoid_cross_entropy_with_logits_grad_compute

    Parameters
    ----------
    predict : TVM tensor
        the output of previous layer
    target : TVM tensor
        label
    dout : TVM tensor
        last gradient
    gradient : TVM tensor
        result after compute
    Returns
    -------
    output tensor
    """
    dtype = predict.dtype
    if dtype == "float16" and tbe_platform.api_check_support(
            "tbe.dsl.vmul", "float32"):
        predict = tbe.cast_to(predict, "float32")
        target = tbe.cast_to(target, "float32")
        dout = tbe.cast_to(dout, "float32")

    # e^x
    val1 = tbe.vexp(predict)
    # 1 + e^x
    val2 = tbe.vadds(val1, tvm.const(1, dtype="float32"))

    val3 = tbe.vdiv(val1, val2)
    # -target
    val4 = tbe.vmuls(target, tvm.const(-1, dtype="float32"))

    val5 = tbe.vadd(val3, val4)

    result = tbe.vmul(val5, dout)

    if dtype == "float16":
        result = tbe.cast_to(result, dtype)
    return result


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def sigmoid_cross_entropy_with_logits_grad(
        predict,
        target,
        dout,
        gradient,
        kernel_name="sigmoid_cross_entropy_with_logits_grad"):
    """calculating data

    Parameters
    ----------
    predict : dict
        the output of previous layer
    target : dict
        label
    dout : dict
        last gradient
    gradient : dict
        result after compute
    kernel_name : str
        kernel name, default value is "sigmoid_cross_entropy_with_logits_grad"

    Returns
    -------
    None
    """
    check_list = ("float16", "float32")
    predict_shape = predict.get("shape")
    predict_dtype = predict.get("dtype")
    gradient_dtype = gradient.get("dtype").lower()
    predict_dtype_lower = predict_dtype.lower()
    para_check.check_dtype(gradient_dtype, check_list, param_name="gradient")
    para_check.check_dtype(predict_dtype_lower, check_list, param_name="predict")

    para_check.check_shape(predict_shape, param_name="predict")

    target_shape = target.get("shape")
    target_dtype = target.get("dtype")
    target_dtype_lower = target_dtype.lower()
    para_check.check_dtype(target_dtype_lower, check_list, param_name="target")

    para_check.check_shape(target_shape, param_name="target")

    dout_shape = dout.get("shape")
    dout_dtype = dout.get("dtype")
    dout_dtype_lower = dout_dtype.lower()
    para_check.check_dtype(dout_dtype_lower, check_list, param_name="dout")

    para_check.check_shape(dout_shape, param_name="dout")
    shape_util.compare_tensor_dict_key(predict, target, "shape")
    shape_util.compare_tensor_dict_key(predict, dout, "shape")
    shape = (functools.reduce(lambda x, y: x * y, predict_shape[:]),)
    predict_data_input = tvm.placeholder(
        shape, name="predict_data_input", dtype=predict_dtype_lower)
    target_data_input = tvm.placeholder(
        shape, name="target_data_input", dtype=target_dtype_lower)
    dout_data_input = tvm.placeholder(
        shape, name="dout_data_input", dtype=dout_dtype_lower)

    res = sigmoid_cross_entropy_with_logits_grad_compute(
        predict_data_input, target_data_input, dout_data_input, gradient,
        kernel_name)

    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {
        "name":
            kernel_name,
        "tensor_list": [
            predict_data_input, target_data_input, dout_data_input, res
        ]
    }

    build(sch, config)
