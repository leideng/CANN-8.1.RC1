#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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
hard_shrink
"""
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=superfluous-parens,unused-argument
@register_operator_compute("hard_shrink", op_mode="static", support_fusion=True)
def hard_shrink_compute(input_x, output_y, lambd, kernel_name="hard_shrink"):
    """
    calculating data

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    lambda : float
        calculation threshold
    kernel_name : str
        kernel name, default value is "hard_shrink"

    Returns
    -------
    output tensor
    """
    dtype = input_x.dtype
    shape = input_x.shape
    input_x_abs = tbe.vabs(input_x)
    lambd_tensor = tbe.broadcast(tvm.const(lambd, dtype), shape)
    zero_tensor = tbe.broadcast(tvm.const(0, dtype), shape)
    result = tbe.vcmpsel(input_x_abs, lambd_tensor, 'le', zero_tensor, input_x)
    result = tbe.cast_to(result, dtype)
    return result


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def hard_shrink(input_x, output_y, lambd=0.5, kernel_name="hard_shrink"):
    """
    calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    lambda : float
        calculation threshold
    kernel_name : str
        kernel name, default value is "hard_shrink"

    Returns
    -------
    None
    """
    input_shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()

    para_check.check_shape(input_shape, param_name="input_x")
    check_tuple = ("float16", "float32")
    para_check.check_dtype(input_dtype, check_tuple, param_name="input_x")

    if(lambd < 0):
        raise RuntimeError("Only support lambd >= 0 while lambd is {}.".format(lambd))

    data_input = tvm.placeholder(input_shape, name="data_input", dtype=input_dtype)
    res = hard_shrink_compute(data_input, output_y, lambd, kernel_name)

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res]}

    tbe.cce_build_code(schedule, config)
