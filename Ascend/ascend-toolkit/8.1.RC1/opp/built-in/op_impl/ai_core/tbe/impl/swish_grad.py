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
swish_grad
y = sigmoid(scale*x) + x*sigmoid'(scale*x)
sigmoid' = sigmoid*(1 - sigmoid)
let:
A = fwd_input = x                   # input of swish forward
B = fwd_output = x*sigmoid(scale*x) # output of swish forward
y = scale*B + B/A*(1 - scale*B)
"""

from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name
@register_operator_compute("swish_grad", op_mode="static", support_fusion=True, support_bfp16=True)
def swish_grad_compute(input_gradients, fwd_input, fwd_output, bkwd_output, beta, kernel_name="swish_grad"):
    """
    algorithm : swish grad compute
    let:
    A = fwd_input = x                   # input of swish forward
    B = fwd_output = x*sigmoid(scale*x) # output of swish forward
    then,
    swish_grad = scale*B + B/A*(1 - scale*B)
    Parameters:
    ----------
    input_gradients : dictionary of gradient
    fwd_input : dictionary of swish input
    fwd_output : dictionary of swish output
    bkwd_output: dictionary of output
    beta: scale for exponent in sigmoid
    kernel_name : default value is "swish_grad"
    Returns
    -------
    a tenosr
    """
    dtype = fwd_input.dtype.lower()
    # calculate B/A
    sigmoid_value = tbe.vdiv(fwd_output, fwd_input)
    # calculate 1-beta*B
    one_tensor = tbe.broadcast(tvm.const(1, dtype=dtype), fwd_output.shape)
    beta_tensor = tbe.broadcast(tvm.const(beta, dtype=dtype), fwd_output.shape)
    beta_output = tbe.vmul(fwd_output, beta_tensor)
    one_minus_B = tbe.vsub(one_tensor, beta_output)
    # calculate B/A*(1 - scale*B)
    swish_part = tbe.vmul(sigmoid_value, one_minus_B)
    # calculate scale*B + B/A*(1 - scale*B)
    grad_x = tbe.vadd(beta_output, swish_part)
    return tbe.vmul(input_gradients, grad_x)


def check_op_dtype(dtype_input, dtype_x0, dtype_x1):
    """
    check dtypes
    :param dtype_input: str
    :param dtype_x0: str
    :param dtype_x1: str
    :return: none
    """
    if dtype_input != dtype_x0:
        error_manager_vector.raise_err_two_input_dtype_invalid('swish_grad', "input_data", "x0",
                                                               "the dtype of input_data, x0, must be the same")
    if dtype_input != dtype_x1:
        error_manager_vector.raise_err_two_input_dtype_invalid('swish_grad', "input_data", "x1",
                                                               "the dtype of input_data, x1, must be the same")
    check_list = ["bfloat16", "float16", "float32"]
    para_check.check_dtype(dtype_input, check_list)


# 'pylint: disable=too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def swish_grad(input_data, x0, x1, output_y, scale=1.0, kernel_name="swish_grad"):
    """
    do swish grad

    let:
    A = fwd_input = x                   # input of swish forward
    B = fwd_output = x*sigmoid(scale*x) # output of swish forward
    then,
    swish_grad = scale*B + B/A*(1 - scale*B)
    Parameters:
    ----------
    input_data : dictionary of gradient
    x0 : dictionary of swish input
    x1 : dictionary of swish output
    y: dictionary of output
    scale: scale for exponent in sigmoid
    kernel_name : default value is "swish_grad"
    Returns
    -------
    None
    """
    shape_input = input_data.get("shape")
    shape_x0 = x0.get("shape")
    shape_x1 = x1.get("shape")
    dtype_input = input_data.get("dtype").lower()
    dtype_x0 = x0.get("dtype").lower()
    dtype_x1 = x1.get("dtype").lower()

    shape_input = shape_util.scalar2tensor_one(shape_input)
    shape_x0 = shape_util.scalar2tensor_one(shape_x0)
    shape_x1 = shape_util.scalar2tensor_one(shape_x1)

    para_check.check_shape(shape_input)
    para_check.check_shape(shape_x0)
    para_check.check_shape(shape_x1)

    if shape_x0 != shape_x1:
        error_detail = "The shape of input x0 and x1 are not match for dynamic swish_grad."
        error_manager_vector.raise_err_two_input_shape_invalid(
            kernel_name, "x0", "x1", error_detail)
    if shape_x0 != shape_input:
        error_detail = "The shape of input x0 and input_data are not match for dynamic swish_grad."
        error_manager_vector.raise_err_two_input_shape_invalid(
            kernel_name, "x0", "input_data", error_detail)

    para_check.check_kernel_name(kernel_name)
    check_op_dtype(dtype_input, dtype_x0, dtype_x1)

    data_input = tvm.placeholder(
        shape_input, dtype=dtype_input, name="data_input")
    data_x0 = tvm.placeholder(shape_input, dtype=dtype_input, name="data_x0")
    data_x1 = tvm.placeholder(shape_input, dtype=dtype_input, name="data_x1")

    res = swish_grad_compute(data_input, data_x0, data_x1,
                             output_y, scale, kernel_name="swish_grad")

    with tvm.target.cce():
        auto_sch = auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data_input, data_x0, data_x1, res]}

    build(auto_sch, config)
