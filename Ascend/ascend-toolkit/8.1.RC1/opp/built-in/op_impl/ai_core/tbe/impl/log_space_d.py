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
log_space_d
"""

import math
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check


# 'pylint: disable=unused-argument,too-many-arguments,too-many-locals
@register_operator_compute("log_space_d", op_mode="static", support_fusion=True)
def log_space_d_compute(assist, y, start, end, steps=100, base=10.0, dtype=1, kernel_name="log_space_d"):
    """
    calculating data

    Parameters
    ----------
    assist : TVM tensor
        the placeholder of assist
    y : dict
        dict of y, include keys(shape and dtype)
    start : float
        Set the starting value
    end : float
        Set the termination value
    steps : int
        The number of sample points
    base : float
        The base of the exponential function
    dtype : int
        The dtype of output
    kernel_name : str
        kernel name, default value is "log_space_d"

    Returns
    -------
    output tensor
    """
    output_dtype_dict = {0:"float16", 1:"float32"}
    step_minus_one = tvm.const(steps - 1, "int32")
    scalar_one = tvm.const(1.0, "float32")
    if steps <= 1:
        diff = end - start
    else:
        diff = (end - start) / (step_minus_one)
    if assist.dtype != "float32":
        assist = tbe.cast_to(assist, "float32")
    diff = tbe.vmuls(assist, diff)
    x = tbe.vadds(diff, start)
    if base > 0:
        log_base = math.log(base)
        log_base = tvm.const(log_base, "float32")
        index = tbe.vmuls(x, log_base)
        rs = tbe.vexp(index)
    elif base < 0:
        scalar_two = tvm.const(2.0, "float32")
        one_half = 1.0 / scalar_two
        negative_two = tvm.const(-2.0, "float32")
        x_abs = tbe.vabs(x)
        x_div_two = tbe.vmuls(x_abs, one_half)
        x_div_two_floor = tbe.floor(x_div_two)
        x_remainder1 = tbe.vmuls(x_div_two_floor, scalar_two)
        x_remainder2 = tbe.vsub(x_abs, x_remainder1)
        negative_two_x = tbe.vmuls(x_remainder2, negative_two)
        rs1 = tbe.vadds(negative_two_x, scalar_one)
        abs_base = math.fabs(base)
        log_base = math.log(abs_base)
        log_base = tvm.const(log_base, "float32")
        index = tbe.vmuls(x, log_base)
        rs2 = tbe.vexp(index)
        rs = tbe.vmul(rs1, rs2)
    elif base == 0:
        scalar_zero = tvm.const(0.0, "float32")
        rs = tbe.vcmpsel(x, scalar_zero, 'eq', scalar_one, scalar_zero)
    if steps == 0:
        rs = assist
    if output_dtype_dict.get(dtype) == "float16":
        rs = tbe.cast_to(rs, output_dtype_dict[dtype])
    return rs


# 'pylint: disable=too-many-arguments,too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.REQUIRED_ATTR_FLOAT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def log_space_d(assist, y, start, end, steps=100, base=10.0, dtype=1, kernel_name="log_space_d"):
    """
    calculating data

    Parameters
    ----------
    assist : dict
        shape and dtype of input
    y : dict
        shape and dtype of output
    start : float
        Set the starting value
    end : float
        Set the termination value
    steps : int
        The number of sample points
    base : float
        The base of the exponential function
    dtype : int
        The dtype of output
    kernel_name : str
        kernel name, default value is "log_space_d"

    Returns
    -------
    None
    """

    shape = assist.get("shape")

    assist_dtype = assist.get("dtype").lower()

    para_check.check_shape_rule(shape)
    para_check.check_shape(shape)
    para_check.check_kernel_name(kernel_name)

    check_tuple = ("float16", "float32")
    para_check.check_dtype_rule(assist_dtype, check_tuple)

    is_onedim = len(shape)

    if steps < 0:
        raise RuntimeError("please input steps > 0")
    if is_onedim != 1:
        raise RuntimeError("assist.shape only support one dim")
    if shape[0] != steps:
        raise RuntimeError("assist shape should equal to steps")
    if dtype not in [0, 1]:
        raise RuntimeError("only support dtype 0,1")

    data_assist = tvm.placeholder(shape, name="data_assist", dtype=assist_dtype)
    res = log_space_d_compute(data_assist, y, start, end, steps, base, dtype, kernel_name)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_assist, res]}

    tbe.build(schedule, config)
