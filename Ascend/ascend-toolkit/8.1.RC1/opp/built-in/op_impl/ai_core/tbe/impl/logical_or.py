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
logical_or
"""
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=unused-argument,invalid-name,unused-variable
# 'pylint: disable=locally-disabled,too-many-arguments,too-many-locals
@register_operator_compute("logical_or", op_mode="static", support_fusion=True)
def logical_or_compute(x1, x2, y, kernel_name="logical_or"):
    """
    algorithm : logical_or_compute
    calculating the value of x1 OR x2 element-wise

    Parameters
    ----------
    x1 : the placeholders of x1

    x2 : the placeholders of x2

    y : the dict of y

    kernel_name : string, cce kernel name, default value is "logical_or"

    Returns
    -------
    result res
    """
    _, _, shape_max = shape_util.broadcast_shapes(shape_util.shape_to_list(
        x1.shape), shape_util.shape_to_list(x2.shape), param_name_input1="x1",
        param_name_input2="x2")
    x1 = tbe.cast_to(x1, "float16")
    x2 = tbe.cast_to(x2, "float16")
    x1 = tbe.broadcast(x1, shape_max)
    x2 = tbe.broadcast(x2, shape_max)
    res = tbe.vmax(x1, x2)
    res = tbe.cast_to(res, "int8")

    return res


# @register_operator("LogicalOr")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def logical_or(x1, x2, y, kernel_name="logical_or"):
    """
    algorithm : logical_or
    calculating the value of x1 OR x2 element-wise

    Parameters
    ----------
    x1 : the dict of x1,
         include shape and dtype,
         dtype support int8, the value only support 0, 1

    x2 : the dict of x2,
         include shape and dtype,
         dtype support int8, the value only support 0, 1

    y : the dict of y, include shape and dtype

    kernel_name : string, cce kernel name, default value is "logical_or"

    Returns
    -------
    None
    """

    shape_x1 = x1.get("shape")
    shape_x2 = x2.get("shape")
    dtype_x1 = x1.get("dtype")
    dtype_x2 = x2.get("dtype")
    if dtype_x1 == "bool" or dtype_x2 == "bool":
        dtype_x1 = "int8"
        dtype_x2 = "int8"

    para_check.check_shape(shape_x1, param_name="x1")
    para_check.check_shape(shape_x2, param_name="x2")

    check_tuple = ("int8",)
    para_check.check_dtype(dtype_x1, check_tuple, param_name="x1")
    para_check.check_dtype(dtype_x2, check_tuple, param_name="x2")

    shape_x1, shape_x2, shape_max = shape_util.broadcast_shapes(shape_x1, shape_x2,
                                                     param_name_input1="x1",
                                                     param_name_input2="x2")
    dtype = dtype_x1.lower()
    data_x1 = tvm.placeholder(shape_x1, name="data_x1", dtype=dtype)
    data_x2 = tvm.placeholder(shape_x2, name="data_x2", dtype=dtype)

    res = logical_or_compute(data_x1, data_x2, y, kernel_name)

    with tvm.target.cce():
        schedule = auto_schedule(res)

    config = {"print_ir": False,
              "need_build": False,
              "name": kernel_name,
              "tensor_list": (data_x1, data_x2, res)}
    build(schedule, config)
