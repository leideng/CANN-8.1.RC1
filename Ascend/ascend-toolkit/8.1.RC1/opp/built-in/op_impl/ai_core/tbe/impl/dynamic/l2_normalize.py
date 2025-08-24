#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2022 Huawei Technologies Co., Ltd
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
from ..util.platform_adapter import tbe
from ..util.platform_adapter import para_check
from ..util.platform_adapter import shape_util
from ..util.platform_adapter import tbe_platform
from ..util.platform_adapter import tvm
from ..util.platform_adapter import register_operator
from ..util.platform_adapter import register_operator_compute
from ..util.platform_adapter import classify
from ..util.platform_adapter import OpPatternMode
from ..util.platform_adapter import error_manager_vector


# 'pylint: disable=unused-argument
@register_operator_compute("L2Normalize", op_mode="dynamic", support_fusion=True)
def l2_normalize_compute(x,
                         y,
                         axis,
                         eps,
                         kernel_name="l2_normalize"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of x
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
    dtype = x.dtype
    if dtype == "float16" and tbe_platform.api_check_support(
            "te.lang.cce.vmul", "float32"):
        x = tbe.cast_to(x, "float32")
    x_square = tbe.vmul(x, x)
    x_square_sum = tbe.reduce_sum(x_square, axis, keepdims=True)
    const_epsilon = tvm.const(eps, "float32")
    x_l2norm = tbe.vmaxs(x_square_sum, const_epsilon)
    x_l2norm_sqrt = tbe.vsqrt(x_l2norm)
    x_l2norm_sqrt = tbe.broadcast(x_l2norm_sqrt, x.shape)

    result = tbe.vdiv(x, x_l2norm_sqrt)

    if dtype == "float16":
        result = tbe.cast_to(result, "float16")
    return result


@register_operator("L2Normalize")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.KERNEL_NAME)
def l2_normalize(x, y, axis, eps, kernel_name="l2_normalize"):
    """
    calculating data

    Parameters
    ----------
    x : dict
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
    shape = x.get("shape")
    dtype = x.get("dtype")
    input_dtype = dtype.lower()

    para_check.check_shape(shape, param_name="x")
    para_check.check_dtype(input_dtype, ("float16", "float32"), param_name="x")

    for dim in axis:
        if not isinstance(dim, int):
            error_manager_vector.raise_err_input_value_invalid(kernel_name, "axis", "int", "not int")
        if dim >= len(shape) or dim < -len(shape):
            error_manager_vector.raise_err_input_param_not_in_range(kernel_name, "axis", -len(shape), len(shape), dim)

    schedules, tensors = [], []
    extra_params = {"keepdims": True}

    ins = classify([x, axis], OpPatternMode.NORM, extra_params)

    for (_x, reduce_axis) in ins:
        with tbe.compute():
            x_shape,  = shape_util.variable_shape([_x], op_mode="norm")
            data_input = tvm.placeholder(x_shape, dtype=input_dtype, name="data_input")
            res = l2_normalize_compute(data_input, y=y, axis=reduce_axis, eps=eps, kernel_name=kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {
        "name": kernel_name,
        "tensor_list": tensors,
        "bool_storage_as_1bit": False
    }
    tbe.build(schedules, config)
