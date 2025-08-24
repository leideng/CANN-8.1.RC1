#!/usr/bin/python
# -*- coding: utf-8 -*-
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
xdivy_grad
"""
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import shape_util
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm


# 'pylint: disable=unused-argument
# 'pylint: disable=invalid-name
def _broadcast_gradient_args(x, y):
    """
    Return the reduction indices for computing gradients of
    x op y with broadcast.

    Parameters
    ----------
    x : the shape of data input
    y : the shape of data input

    Returns
    -------
    rx : the reduction indices for computing gradients of x
    ry : the reduction indices for computing gradients of y
    """
    rx = []
    ry = []
    for i, item in enumerate(x):
        if item < y[i]:
            rx.append(i)
        elif item > y[i]:
            ry.append(i)

    return rx, ry


# 'pylint: disable=too-many-locals
@register_operator_compute("xdivy_grad", op_mode="static", support_fusion=True)
def xdivy_grad_compute(placeholders, shape_max, dtype, rx, ry):
    """
    do element-wise xdivy_grad compute

    Parameters
    ----------
    placeholders : the placeholder of data input
    shape_max : the shape of broadcast
    dtype : the type of data input
    rx : the reduction indices of data input with broadcast
    ry : the reduction indices for data input with broadcast

    Returns
    -------
    output_y1 : result of xdivy_grad
    output_y2 : result of xdivy_grad
    None
    """
    x1_ori = placeholders[0]
    x2_ori = placeholders[1]
    grad_ori = placeholders[2]

    fp32_support = tbe_platform.api_check_support("tbe.vdiv", "float32")
    if not fp32_support:
        para_check.check_dtype(dtype, ("float16", ), param_name="x1")

    if dtype == "float16" and fp32_support:
        x1 = tbe.cast_to(x1_ori, "float32")
        x2 = tbe.cast_to(x2_ori, "float32")
        grad = tbe.cast_to(grad_ori, "float32")
        x1 = tbe.broadcast(x1, shape_max)
        x2 = tbe.broadcast(x2, shape_max)
        grad = tbe.broadcast(grad, shape_max)
    else:
        x1 = tbe.broadcast(x1_ori, shape_max)
        x2 = tbe.broadcast(x2_ori, shape_max)
        grad = tbe.broadcast(grad_ori, shape_max)

    if dtype == "float16" and not fp32_support:
        esp_min = tvm.const(1.18e-7, dtype="float16")
        neg_one = tvm.const(-1, dtype="float16")
    else:
        esp_min = tvm.const(1.18e-38, dtype="float32")
        neg_one = tvm.const(-1, dtype="float32")
    x1_addespmin = tbe.vadds(x1, esp_min)
    not_zero_x1 = tbe.vdiv(x1, x1_addespmin)
    partial_x1 = tbe.vdiv(not_zero_x1, x2)
    partial_x1g = tbe.vmul(partial_x1, grad)

    neg_x1 = tbe.vmuls(x1, neg_one)
    partial_x1pow = tbe.vmul(partial_x1, partial_x1)
    partial_x2 = tbe.vmul(neg_x1, partial_x1pow)
    partial_x2g = tbe.vmul(partial_x2, grad)

    output_y1 = tbe.sum(partial_x1g, rx, keepdims=True)
    output_y2 = tbe.sum(partial_x2g, ry, keepdims=True)

    if dtype == "float16" and fp32_support:
        output_y1 = tbe.cast_to(output_y1, "float16")
        output_y2 = tbe.cast_to(output_y2, "float16")

    return output_y1, output_y2


# 'pylint: disable=too-many-locals,too-many-arguments
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def xdivy_grad(x1, x2, grad, y1, y2, kernel_name="xdivy_grad"):
    """
    Returns gradient of xdivy(x, y) with respect to x and y.

    Parameters
    ----------
    x1 : dict
        shape and dtype of input, only support float16, float32
    x2 : dict
        shape and dtype of input, only support float16, float32
    grad : dict
        shape and dtype of input, only support float16, float32
    y1 : dict
        shape and dtype of output, should be same shape and type as input
    y2 : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "xdivygrad"

    Returns
    -------
    None
    """
    shape_x1 = x1.get("shape")
    dtype_x1 = x1.get("dtype").lower()
    shape_x2 = x2.get("shape")
    dtype_x2 = x2.get("dtype").lower()
    shape_grad = grad.get("shape")
    dtype_grad = grad.get("dtype").lower()

    if dtype_x1 != dtype_x2:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "x1", "x2", dtype_x1, dtype_x2)
    if dtype_x1 != dtype_grad:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "x1", "grad", dtype_x1, dtype_grad)
    if dtype_x2 != dtype_grad:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "x2", "grad", dtype_x2, dtype_grad)

    para_check.check_shape(shape_x1, param_name="x1")
    para_check.check_shape(shape_x2, param_name="x2")
    para_check.check_shape(shape_grad, param_name="grad")
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_x1, check_list, param_name="x1")
    shape_x1, shape_x2, shape_max_x1x2 = shape_util.produce_shapes(shape_x1, shape_x2)
    if len(shape_max_x1x2) < len(shape_grad):
        error_manager_vector.raise_err_input_shape_invalid(
            kernel_name, "grad", "the length of shape_grad can not be longer than the maximum length of x1 and x2.")

    shape_grad, _, shape_max = shape_util.produce_shapes(shape_grad, shape_max_x1x2)

    for (x, y) in zip(shape_max_x1x2, shape_grad):
        if x < y:
            error_manager_vector.raise_err_input_shape_invalid(
                kernel_name, "grad", "the dim of grad's shape can not be bigger than the maximum dim of x1 and x2.")

    para_check.check_shape(shape_max, param_name="x")
    rx, ry = _broadcast_gradient_args(shape_x1, shape_x2)

    x1 = tvm.placeholder(shape_x1, name="x", dtype=dtype_x1)
    x2 = tvm.placeholder(shape_x2, name="y", dtype=dtype_x1)
    grad = tvm.placeholder(shape_grad, name="grad", dtype=dtype_x1)

    output_y1, output_y2 = xdivy_grad_compute([x1, x2, grad], shape_max, dtype_x1, rx, ry)

    with tvm.target.cce():
        sch = tbe.auto_schedule([output_y1, output_y2])

    config = {"name": kernel_name, "tensor_list": [x1, x2, grad, output_y1, output_y2]}
    tbe.cce_build_code(sch, config)
