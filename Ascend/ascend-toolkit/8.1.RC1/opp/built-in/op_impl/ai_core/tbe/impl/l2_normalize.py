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
l2_normalize
"""
from impl.util.platform_adapter import build
from impl.util.platform_adapter import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=unused-argument
@register_operator_compute("l2_normalize", op_mode="static", support_fusion=True)
def l2_normalize_compute(input_x,
                         output_y,
                         axis,
                         epsilon,
                         kernel_name="l2_normalize"):
    """
    calculating data

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    axis : list
        the axis which to be computed
    epsilon : float
        the minimum value, in case the denominator is zero
    kernel_name : str
        kernel name, default value is "l2_normalize"

    Returns
    -------
    output tensor
    """
    dtype = input_x.dtype
    if dtype == "float16" and tbe_platform.api_check_support(
            "tbe.dsl.vmul", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
    x_square = tbe.vmul(input_x, input_x)
    x_square_sum = tbe.sum(x_square, axis, keepdims=True)
    const_epsilon = tvm.const(epsilon, "float32")
    x_l2norm = tbe.vmaxs(x_square_sum, const_epsilon)
    x_l2norm_sqrt = tbe.vsqrt(x_l2norm)
    x_l2norm_sqrt = tbe.broadcast(x_l2norm_sqrt,
                                  shape_util.shape_to_list(input_x.shape))

    result = tbe.vdiv(input_x, x_l2norm_sqrt)

    if dtype == "float16":
        result = tbe.cast_to(result, "float16")
    return result


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.KERNEL_NAME)
def l2_normalize(input_x, output_y, axis, epsilon, kernel_name="l2_normalize"):
    """
    calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    axis : list
        the axis which to be computed
    epsilon : float
        the minimum value, in case the denominator is zero
    kernel_name : str
        kernel name, default value is "l2_normalize"

    Returns
    -------
    None
    """
    shape = input_x.get("shape")
    dtype = input_x.get("dtype")
    input_dtype = dtype.lower()

    para_check.check_shape(shape, param_name="input_x")
    para_check.check_dtype(input_dtype, ("float16", "float32"), param_name="input_x")

    for i in axis:
        if not isinstance(i, int):
            error_manager_vector.raise_err_input_value_invalid(kernel_name, "axis", "int", "not int")
        if i >= len(shape) or i < -len(shape):
            error_manager_vector.raise_err_input_param_not_in_range(kernel_name, "axis", -len(shape), len(shape), i)

    data_input = tvm.placeholder(shape, name="data_input", dtype=input_dtype)
    res = l2_normalize_compute(data_input, output_y,
                               axis, epsilon, kernel_name)

    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_input, res]}

    build(sch, config)
