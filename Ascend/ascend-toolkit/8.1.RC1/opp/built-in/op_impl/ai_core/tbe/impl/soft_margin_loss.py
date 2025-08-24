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
soft_margin_loss
"""

from impl.util.platform_adapter import register_operator_compute
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check


SHAPE_SIZE_LIMIT = 2147483648


# 'pylint: disable=unused-argument,too-many-locals
@register_operator_compute("soft_margin_loss", op_mode="static", support_fusion=True)
def soft_margin_loss_compute(input_x, input_y, output_z, reduction='mean', kernel_name="soft_margin_loss"):
    """calculating data

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    input y : TVM tensor
        the placeholder of input_y
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "soft_margin_loss"

    Returns
    -------
    output tensor
    """

    result_type = input_x.dtype
    x_shape = input_x.shape
    if input_x.dtype.lower() == "float16":
        input_x = tbe.cast_to(input_x, "float32")
    if input_y.dtype.lower() == "float16":
        input_y = tbe.cast_to(input_y, "float32")

    x_mul = tbe.vmuls(input_x, tvm.const(-1, dtype=input_x.dtype))
    x_y_mul = tbe.vmul(x_mul, input_y)
    res_exp = tbe.vexp(x_y_mul)
    res_add = tbe.vadds(res_exp, tvm.const(1, dtype=input_x.dtype))
    result = tbe.vlog(res_add)

    if reduction != 'none':
        ax = list(range(len(x_shape)))
        result = tbe.sum(result, ax, False)
        if reduction == 'mean':
            size = 1.0
            for val in x_shape:
                size = size * val
            result = tbe.vmuls(result, 1.0 / size)

    if result_type == "float16":
        result = tbe.cast_to(result, "float16")
    return result


# 'pylint: disable=unused-argument,too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_STR,
                            para_check.KERNEL_NAME)
def soft_margin_loss(input_x, input_y, output_z, reduction='mean', kernel_name="soft_margin_loss"):
    """calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input_predict
    input_y: dict
        shape and dtype of input label
    output_z : dict
     if reduction is none, shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "soft_margin_loss"

    Returns
    -------
    None
    """
    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype")
    dtype_x = dtype_x.lower()
    shape_y = input_y.get("shape")
    dtype_y = input_y.get("dtype")
    dtype_y = dtype_y.lower()

    #reduction value must be in reduction_list
    reduction_list = ("mean", "none", "sum")
    if reduction.lower() not in reduction_list:
        raise RuntimeError("The reduction value is invalid."
                           "Please input a valid value.")

    # dtype of input must be float16,float32
    check_tuple = ("float16", "float32")
    para_check.check_shape_rule(shape_y)
    para_check.check_shape_rule(shape_y)
    para_check.check_dtype_rule(dtype_x, check_tuple)
    para_check.check_dtype_rule(dtype_y, check_tuple)
    para_check.check_kernel_name(kernel_name)
    para_check.check_shape_size(shape_x, SHAPE_SIZE_LIMIT)

    data_input1 = tvm.placeholder(shape_x, name="data_input1", dtype=dtype_x)
    data_input2 = tvm.placeholder(shape_y, name="data_input2", dtype=dtype_y)
    res = soft_margin_loss_compute(data_input1, data_input2, output_z, reduction, kernel_name)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)
    config = {"name": kernel_name,
              "tensor_list": (data_input1, data_input2, res)}
    tbe.cce_build_code(schedule, config)
