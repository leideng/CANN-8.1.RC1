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
is_close
"""
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
@register_operator_compute("is_close", op_mode="static", support_fusion=True)
def is_close_compute(input_x1, input_x2, output_y, rtol=1e-05, atol=1e-08, equal_nan=False, kernel_name="is_close"):
    """
    calculating a new tensor with bool elements representing if each element of input_x1 is "close" to the corresponding
    element of input_x2.
    Closeness is defined as:∣input_x1−input_x2∣≤atol+rtol×∣input_x2∣

    Parameters
    ----------
    input_x1: TVM tensor
        the placeholder of first input data
    input_x2: TVM tensor
        the placeholder of second input data
    output_y: dict
        shape and dtype of output, should be broadcast shape and bool type
    rtol: float
        absolute tolerance, default value 1e-08
    atol: float
        relative tolerance, default value is 1e-05
    equal_nan: bool
        if True, then two NaN s will be considered equal, default value is False
    kernel_name: str
        cce kernel name, default value is is_close

    Returns
    -------
    res : output of the data's isclose
    """
    shape_x1 = shape_util.shape_to_list(input_x1.shape)
    shape_x2 = shape_util.shape_to_list(input_x2.shape)

    shape_x1, shape_x2, shape_max = shape_util.broadcast_shapes(shape_x1, shape_x2)
    input_x1 = tbe.broadcast(input_x1, shape_max)
    input_x2 = tbe.broadcast(input_x2, shape_max)
    lhs = tbe.vabs(tbe.vsub(input_x1, input_x2))
    temp = tbe.vabs(tbe.vmuls(input_x2, rtol))
    rhs = tbe.vadds(temp, atol)
    return tbe.vcmp(lhs, rhs, operation='le')


# 'pylint: disable=too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def is_close(input_x1, input_x2, output_y, rtol=1e-05, atol=1e-08, equal_nan=False, kernel_name="is_close"):
    """
    algorithm: is_close
    calculating a new tensor with bool elements representing if each element of input_x1 is "close" to the corresponding
    element of input_x2.
    Closeness is defined as:∣input_x1−input_x2∣≤atol+rtol×∣input_x2∣

    Parameters
    ----------
    input_x1 : dict
        shape and dtype of first input, only support float16, float32, int32
    input_x2 : dict
        shape and dtype of second input, only support float16, float32, int32
    output_y: dict
        shape and dtype of output, should be broadcast shape and bool type
    rtol: float
        absolute tolerance, default value 1e-08
    atol: float
        relative tolerance, default value is 1e-05
    equal_nan: bool
        if True, then two NaN s will be considered equal, default value is False
    kernel_name : str
        cce kernel name, default value is is_close

    Returns
    -------
    None
    """
    shape_x1 = input_x1.get("shape")
    shape_x2 = input_x2.get("shape")
    para_check.check_kernel_name(kernel_name)
    para_check.check_shape_rule(shape_x1)
    para_check.check_shape_rule(shape_x2)
    para_check.check_shape_size(shape_x1)
    para_check.check_shape_size(shape_x2)

    input_data_type = input_x1.get("dtype").lower()

    check_tuple = ("float16", "float32", "int32")
    para_check.check_dtype_rule(input_data_type, check_tuple)

    shape_x1, shape_x2, shape_max = shape_util.broadcast_shapes(shape_x1, shape_x2)
    if shape_x1[-1] == 1 and shape_x2[-1] == 1 and shape_max[-1] == 1:
        shape_x1 = shape_x1 if len(shape_x1) == 1 else shape_x1[:-1]
        shape_x2 = shape_x2 if len(shape_x2) == 1 else shape_x2[:-1]
        shape_max = shape_max if len(shape_max) == 1 else shape_max[:-1]

    para_check.check_shape_size(shape_max)
    data_x1 = tvm.placeholder(shape_x1, name="data_1", dtype=input_data_type)
    data_x2 = tvm.placeholder(shape_x2, name="data_2", dtype=input_data_type)
    if input_data_type == "float16":
        data_x1_trans = tbe.cast_to(data_x1, "float32")
        data_x2_trans = tbe.cast_to(data_x2, "float32")
        res = is_close_compute(data_x1_trans, data_x2_trans, output_y, rtol, atol, equal_nan, kernel_name)
    else:
        res = is_close_compute(data_x1, data_x2, output_y, rtol, atol, equal_nan, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    # operator build
    config = {"name": kernel_name,
              "tensor_list": [data_x1, data_x2, res],
              "bool_storage_as_1bit": False}

    tbe.build(schedule, config)
