#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
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
dot
"""

import te.lang.cce as tbe
from impl.util.platform_adapter import build
from impl.util.platform_adapter import auto_schedule
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check


# 'pylint: disable=unused-argument
@register_operator_compute("dot", op_mode="static", support_fusion=True)
def dot_compute(input_x1, input_x2, output, kernel_name="dot"):
    """
    :param input_x1: one tesnor must be 1d
    :param input_x2: another tensor must be 1d
    :param output: must be 1d
    :param kernel_name: dot
    :return: the dot result
    """
    dtype = input_x1.dtype
    if dtype == 'float16':
        input_x1 = tbe.cast_to(input_x1, 'float32')
        input_x2 = tbe.cast_to(input_x2, 'float32')

    mul_tmp = tbe.vmul(input_x1, input_x2)
    red_res = tbe.sum(mul_tmp, 0)
    if dtype == 'float16':
        red_res = tbe.cast_to(red_res, 'float16')
    return red_res


# 'pylint: disable=unused-argument,too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def dot(input_x1, input_x2, output, kernel_name="dot"):
    """
    :param input_x1: one tesnor must be 1d
    :param input_x2: another tensor must be 1d
    :param output: must be 1d
    :param kernel_name: dot
    :return: the dot result
    """
    # shape check
    shape_x1 = input_x1.get("shape")
    shape_x2 = input_x2.get("shape")

    if len(shape_x1) != 1:
        raise RuntimeError("Input tensor1 must be 1D.")
    if len(shape_x2) != 1:
        raise RuntimeError("Input tensor2 must be 1D.")

    shape = shape_x1

    # type check
    dtype_set = ("float16", "float32", "int8", "int32", "uint8")
    dtype_x1 = input_x1.get("dtype")
    dtype_x2 = input_x2.get("dtype")
    para_check.check_dtype_rule(dtype_x1, dtype_set)
    para_check.check_dtype_rule(dtype_x2, dtype_set)
    if dtype_x1 != dtype_x2:
        raise RuntimeError("Input dtype must be the same.")

    data_type = dtype_x1
    data_input1 = tvm.placeholder(shape, name="data_input1", dtype=data_type)
    data_input2 = tvm.placeholder(shape, name="data_input2", dtype=data_type)

    res = dot_compute(data_input1, data_input2, output, kernel_name)

    with tvm.target.cce():
        schedule = auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input1, data_input2, res]}

    build(schedule, config)
