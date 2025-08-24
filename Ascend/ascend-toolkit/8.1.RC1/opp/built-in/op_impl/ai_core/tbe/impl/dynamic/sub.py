#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright 2018 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

sub
"""
from __future__ import absolute_import

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util import util_common


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals,invalid-name
@register_operator_compute("Sub", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def sub_compute(input_x, input_y, output_z, kernel_name="sub"):
    """
    calculating data's sub, c = a - b

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of first input data
    input_y: TVM tensor
        the placeholder of second input data
    output_z: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name: str
        cce kernel name, default value is sub

    Returns
    -------
    res : output of the data's sub
    """
    shape_x, shape_y, shape_max = shape_util.broadcast_shapes(input_x.shape, input_y.shape,
                                                              param_name_input1="input_x",
                                                              param_name_input2="input_y")
    input_dtype = input_x.dtype.lower()
    if input_dtype in ("uint8", "int8", "bool"):
        input_x = tbe.cast_to(input_x, "float16")
        input_y = tbe.cast_to(input_y, "float16")

    input_x = tbe.broadcast(input_x, shape_max)
    input_y = tbe.broadcast(input_y, shape_max)
    res = tbe.vsub(input_x, input_y)

    if input_dtype in ("uint8", "int8"):
        res = util_common.uint8_int8_overflow_proc(res, input_dtype)

    if input_dtype == "bool":
        res = tbe.vabs(res)
        res = tbe.cast_to(res, "bool")

    return res


@register_operator("Sub")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def sub(input_x, input_y, output_z, kernel_name="sub"):
    """
    do element-wise sub operation between two input tensors

    Parameters:
    ----------
    input_x : dict
        shape and dtype of input, only support bfloat16, float16, float32, int32, int64, complex32, complex64,
        uint8, int8, bool
    input_y : dict
        shape and dtype of input, only support bfloat16, float16, float32, int32, int64, complex32, complex64,
        uint8, int8, bool
    output_z: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : kernel name, default value is "sub"

    Returns
    -------
    None
    """

    check_list = ["bfloat16", "float16", "float32", "int32", "int64", "complex32", "complex64",
                  "uint8", "int8", "bool"]
    x_dtype = input_x.get("dtype").lower()
    y_dtype = input_y.get("dtype").lower()
    para_check.check_dtype(x_dtype, check_list, param_name="input_x")
    para_check.check_dtype(y_dtype, check_list, param_name="input_y")

    ins = classify([input_x, input_y], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (x1, x2) in ins:
        with tbe.compute():
            x_shape, y_shape = shape_util.variable_shape([x1, x2])
            data1 = tvm.placeholder(x_shape, x_dtype, "data1")
            data2 = tvm.placeholder(y_shape, y_dtype, "data2")
            res = sub_compute(data1, data2, output_z, kernel_name)
            tensors.append([data1, data2, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    # build
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
