#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpImplMode
from impl.util.util_common import check_op_impl_mode


# 'pylint: disable=unused-argument,invalid-name,too-many-locals
@register_operator_compute("LogSpaceD", op_mode="dynamic", support_fusion=True)
def log_space_d_compute(assist, y, start, end, steps=100, base=10.0, dtype=1, kernel_name="log_space_d", 
                        impl_mode=OpImplMode.HIGH_PERFORMANCE):
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
    input_dtype = assist.dtype.lower()
    if input_dtype == "bfloat16":
        assist = tbe.cast_to(assist, "float32")
    output_dtype_dict = {0:"float16", 1:"float32"}
    if output_dtype_dict.get(dtype) == "float16" and impl_mode == OpImplMode.HIGH_PRECISION:
        assist = tbe.cast_to(assist, "float32")
    check_support_flag = False    
    if steps <= 1:
        ratio = end - start
    else:
        ratio = (end - start) / (steps - 1)
    diff = tbe.vmuls(assist, tvm.const(ratio, "float32"))
    xi = tbe.vadds(diff, tvm.const(start, "float32"))
    if base > 0:
        log_base = math.log(base) # ln(base)
        log_base = tvm.const(log_base, "float32")
        index = tbe.vmuls(xi, log_base) # xi*ln(base)
        if not tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
            index = tbe.cast_to(index, "float16")
            check_support_flag = True
        rs = tbe.vexp(index) # e^(xi*ln(base)) = base^xi
        if check_support_flag:
            rs = tbe.cast_to(rs, "float32")
    elif base < 0:
        scalar_two = tvm.const(2.0, "float32")
        one_half = 1.0 / scalar_two
        negative_two = tvm.const(-2.0, "float32")
        x_abs = tbe.vabs(xi)
        x_div_two = tbe.vmuls(x_abs, one_half)
        x_div_two_floor = tbe.floor(x_div_two)
        if x_div_two_floor.dtype != "float32":
            x_div_two_floor = tbe.cast_to(x_div_two_floor, "float32")
        x_remainder1 = tbe.vmuls(x_div_two_floor, scalar_two)
        x_abs = tbe.cast_to(x_abs, "float32")
        x_remainder2 = tbe.vsub(x_abs, x_remainder1)
        negative_two_x = tbe.vmuls(x_remainder2, negative_two)
        rs1 = tbe.vadds(negative_two_x, tvm.const(1.0, "float32"))
        abs_base = math.fabs(base)
        log_base = math.log(abs_base)
        log_base = tvm.const(log_base, "float32")
        if xi.dtype != "float32":
            xi = tbe.cast_to(xi, "float32")
        index = tbe.vmuls(xi, log_base)
        if not tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
            index = tbe.cast_to(index, "float16")
            check_support_flag = True
        rs2 = tbe.vexp(index)
        if check_support_flag:
            rs2 = tbe.cast_to(rs2, "float32")
        rs = tbe.vmul(rs1, rs2)
    elif base == 0:
        rs = tbe.vcmpsel(xi, tvm.const(0.0, "float32"), 'eq',
                        tvm.const(1.0, "float32"), tvm.const(0.0, "float32"))
    if steps == 0:
        rs = assist
    if input_dtype == "bfloat16":
        rs = tbe.round(rs, "bfloat16")
    if output_dtype_dict.get(dtype) == "float16":
        rs = tbe.cast_to(rs, output_dtype_dict.get(dtype))
    return  rs 


# 'pylint: disable=too-many-arguments,too-many-locals
@register_operator("LogSpaceD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.REQUIRED_ATTR_FLOAT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def log_space_d(assist, y, start, end, steps=100, base=10.0, dtype=1, kernel_name="log_space_d", 
                impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    calculating data

    Parameters
    ----------
    assist : TVM tensor
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
    
    check_op_impl_mode(
        impl_mode, [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION], kernel_name)
    shape = assist.get("shape")
    assist_dtype = assist.get("dtype").lower()

    para_check.check_shape(shape)
    para_check.check_kernel_name(kernel_name)

    check_tuple = ("float16", "float32", "bfloat16")
    para_check.check_dtype(assist_dtype, check_tuple)

    is_onedim = len(shape)

    if steps < 0:
        raise RuntimeError("please input steps > 0")
    if is_onedim != 1:
        raise RuntimeError("assist.shape only support one dim")
    if dtype not in [0, 1]:
        raise RuntimeError("only support dtype 0,1")

    ins = classify([assist], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_x,) in ins:
        with tbe.compute():
            shape_x = shape_util.variable_shape([_x])
            data_assist = tvm.placeholder(shape_x[0], name="data_assist", dtype=assist_dtype)
            res = log_space_d_compute(data_assist, y, start, end, steps, base, dtype, kernel_name, impl_mode)
            tensors.append([data_assist, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)