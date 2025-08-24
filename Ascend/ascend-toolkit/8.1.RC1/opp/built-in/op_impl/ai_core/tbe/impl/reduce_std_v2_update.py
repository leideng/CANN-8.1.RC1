#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
reduce_std_v2_update
"""
import operator as op
import tbe.dsl as tbe
from tbe import tvm
from tbe.common.utils import para_check
from tbe.common.utils import shape_to_list
from tbe.common.utils import axis_check
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=invalid-name,too-many-locals,unused-argument,too-many-arguments
@register_operator_compute("reduce_std_v2_update", op_mode="static", support_fusion=True)
def reduce_std_v2_update_compute(x, mean, dim, if_std, unbiased, keepdim, correction,
                                 kernel_name="reduce_std_v2_update"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of X
    mean : TVM tensor
        the mean of X
    dim : intlist
        dimension to calculate
    if_std : bool
        control whether the output is standard deviation or variance, default value is False
    unbiased : bool
        control Bessel deviation, default value is True
    keepdim : bool
        hold dimension or not, default value is False
    correction: int
        if unbiased is true, Bessel's correction will be used, default value is 1
    kernel_name: str
        kernel name

    Returns
    -------
    output TVM tensors
    """
    x_type = x.dtype.lower()

    if x_type == "float16":
        x = tbe.cast_to(x, "float32")
        mean = tbe.cast_to(mean, "float32")

    shape_x = shape_to_list(x.shape)

    reduce_ele = 1.0
    for i in dim:
        reduce_ele *= shape_x[i]
    cof = reduce_ele ** (-1)

    x_sub = tbe.vsub(x, mean)
    var_mul = tbe.vmul(x_sub, x_sub)

    if unbiased:
        cof_unbiased = (reduce_ele - correction) ** (-1)
        var_muls = tbe.vmuls(var_mul, cof_unbiased)
    else:
        var_muls = tbe.vmuls(var_mul, cof)

    var = tbe.reduce_sum(var_muls, axis=dim, keepdims=keepdim)

    if if_std:
        std = tbe.vsqrt(var, impl_mode="high_precision")
        if std.dtype != x_type:
            std = tbe.cast_to(std, x_type)
        return std

    if var.dtype != x_type:
        var = tbe.cast_to(var, x_type)
    return var


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def reduce_std_v2_update(x, mean, output_var, dim, if_std=False, unbiased=True, keepdim=False, correction=1,
                         kernel_name="reduce_std_v2_update"):
    """
    calculating data

    Parameters
    ----------
    x: dict
        input tensor
    mean: dict
        mean value of input tensor
    output_var: dict
        output, variance or standard deviation
    dim: list[int]
        dimension to calculate
    if_std : bool
        control whether the output is standard deviation or variance, default value is False
    unbiased: bool
        control Bessel deviation, default value is True
    keepdims: bool
        hold dimension or not, default value is False
    correction: int
        if unbiased is true, Bessel's correction will be used, default value is 1
    kernel_name: str
        cce kernel name, default value is reduce_std_with_mean

    Returns
    -------
    None
    """
    check_list = ("float16", "float32")

    shape_x = x.get("shape")
    dtype_x = x.get("dtype").lower()
    para_check.check_dtype(dtype_x, check_list, param_name="x")
    para_check.check_shape(shape_x, param_name="x")

    shape_mean = mean.get("shape")
    para_check.check_shape(shape_mean, param_name="mean")

    if not op.eq(shape_x, shape_mean):
        raise RuntimeError("the x and mean should have the same shape.")

    dim = list(dim)

    dim = axis_check(len(shape_x), dim)

    data_x = tvm.placeholder(shape_x, dtype=x.get("dtype"), name="data_x")
    data_mean = tvm.placeholder(shape_mean, dtype=mean.get("dtype"), name="data_mean")

    res = reduce_std_v2_update_compute(data_x, data_mean, dim, if_std, unbiased, keepdim, correction, kernel_name)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_x, data_mean, res]}
    tbe.build(schedule, config)