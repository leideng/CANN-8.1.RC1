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

gcd
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


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals,invalid-name
@register_operator_compute("Gcd", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def gcd_compute(input_x1, input_x2, output_y, kernel_name="gcd"):
    """
    calculating data's gcd, c = a - b

    Parameters
    ----------
    input_x1: TVM tensor
        the placeholder of first input data
    input_x2: TVM tensor
        the placeholder of second input data
    output_y: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name: str
        cce kernel name, default value is gcd

    Returns
    -------
    result : output of the data's gcd
    """
    shape_x1, shape_x2, shape_max = shape_util.broadcast_shapes(input_x1.shape, input_x2.shape,
                                                                param_name_input1="input_x1",
                                                                param_name_input2="input_x2")
    input_x1 = tbe.broadcast(input_x1, shape_max)
    input_x2 = tbe.broadcast(input_x2, shape_max)
    result = tbe.vgcd(input_x1, input_x2)

    return result


@register_operator("Gcd")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def gcd(input_x1, input_x2, output_y, kernel_name="gcd"):
    """
    do element-wise gcd operation between two input tensors

    Parameters:
    ----------
    input_x1 : dict
        shape and dtype of input, only support int16, int32, int64
    input_x2 : dict
        shape and dtype of input, only support int16, int32, int64
    output_z: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : kernel name, default value is "gcd"

    Returns
    -------
    None
    """

    check_list = ["int16", "int32", "int64"]
    x1_dtype = input_x1.get("dtype").lower()
    x2_dtype = input_x2.get("dtype").lower()
    error_detal = "gcd only support int16, int32, int64"
    if x1_dtype not in check_list or x2_dtype not in check_list:
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name,
                                                               "input_x1",
                                                               "input_x2",
                                                               error_detal)

    ins = classify([input_x1, input_x2], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedule, tensors = [], []
    for (x1, x2) in ins:
        with tbe.compute():
            x1_shape, x2_shape = shape_util.variable_shape([x1, x2])
            data1 = tvm.placeholder(x1_shape, x1_dtype, "data1")
            data2 = tvm.placeholder(x2_shape, x2_dtype, "data2")
            res = gcd_compute(data1, data2, output_y, kernel_name)
            tensors.append([data1, data2, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedule.append(sch)
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedule, config)
