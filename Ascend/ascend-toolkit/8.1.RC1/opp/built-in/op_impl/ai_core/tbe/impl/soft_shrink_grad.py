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
soft_shrink_grad.py
"""
from impl.util.platform_adapter import build
from impl.util.platform_adapter import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute


@register_operator_compute("soft_shrink_grad", op_mode="static", support_fusion=True)
# 'pylint: disable=unused-argument
def soft_shrink_grad_compute(input_grad, input_x, output_y, lambd=0.5, kernel_name="soft_shrink_grad"):
    """calculating data

    Parameters
    ----------
    input_grad : TVM tensor
        the input gradient
    input_x : TVM tensor
        the placeholder of input_x
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    lambda : float
        calculation threshold
    kernel_name : str
        kernel name, default value is "soft_shrink_grad"

    Returns
    -------
    output tensor
    """
    dtype = input_x.dtype
    shape = input_x.shape
    one_tensor = tbe.broadcast(tvm.const(1, dtype), shape)
    zero_tensor = tbe.broadcast(tvm.const(0, dtype), shape)
    lambd_tensor = tbe.broadcast(tvm.const(lambd, dtype), shape)
    ratio = tbe.vcmpsel(tbe.vabs(input_x), lambd_tensor, 'le', zero_tensor, one_tensor)
    result = tbe.vmul(input_grad, ratio)
    return result


@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def soft_shrink_grad(input_grad, input_x, output_y, lambd=0.5, kernel_name="soft_shrink_grad"):
    """calculating data

    Parameters
    ----------
    input_grad : TVM tensor
        the input gradient
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    lambda : float
        calculation threshold
    kernel_name : str
        kernel name, default value is "soft_shrink_grad"

    Returns
    -------
    None
    """
    shape_x = input_x.get("shape")
    shape_grad = input_grad.get("shape")

    if list(shape_x) != list(shape_grad):
        raise RuntimeError("Input_grad and input_x must have the same shape.")
    para_check.check_shape(shape_x, param_name="input_x")

    check_list = ("float16", "float32")
    input_dtype = input_x.get("dtype").lower()
    grad_dtype = input_grad.get("dtype").lower()
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")
    para_check.check_dtype(grad_dtype, check_list, param_name="input_grad")
    para_check.check_kernel_name(kernel_name)
    if lambd < 0:
        raise RuntimeError("Only support lambd >= 0 while lambd is {}".format(lambd))

    data_input_grad = tvm.placeholder(shape_grad, name="data_input_grad", dtype=input_dtype)
    data_input_x = tvm.placeholder(shape_x, name="data_input_x", dtype=input_dtype)
    res = soft_shrink_grad_compute(data_input_grad, data_input_x, output_y, lambd, kernel_name)

    with tvm.target.cce():
        schedule = auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input_grad, data_input_x, res]}

    build(schedule, config)
