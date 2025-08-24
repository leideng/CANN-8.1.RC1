#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2018 Huawei Technologies Co., Ltd
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
sub
"""
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import shape_util
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=unused-variable
@register_operator_compute("sub", op_mode="static", support_fusion=True)
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
    shape_x = shape_util.shape_to_list(input_x.shape)
    shape_y = shape_util.shape_to_list(input_y.shape)

    shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x, shape_y, param_name_input1="input_x",
                                                              param_name_input2="input_y")
    input_x = tbe.broadcast(input_x, shape_max)
    input_y = tbe.broadcast(input_y, shape_max)
    res = tbe.vsub(input_x, input_y)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def sub(input_x, input_y, output_z, kernel_name="sub"):
    """
    do element-wise sub operation between two input tensors

    Parameters:
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32,int32
    input_y : dict
        shape and dtype of input, only support float16, float32,int32
    output_z: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : kernel name, default value is "sub"

    Returns
    -------
    None
    """
    shape_x = shape_util.scalar2tensor_one(input_x.get("shape"))
    shape_y = shape_util.scalar2tensor_one(input_y.get("shape"))
    para_check.check_shape(shape_x, param_name="input_x")
    para_check.check_shape(shape_y, param_name="input_y")

    check_list = ["float16", "float32", "int32"]
    dtype = input_x.get("dtype").lower()
    para_check.check_dtype(dtype, check_list, param_name="input_x")

    shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x, shape_y, param_name_input1="input_x",
                                                              param_name_input2="input_y")

    data1 = tvm.placeholder(shape_x, dtype=dtype, name="data1")
    data2 = tvm.placeholder(shape_y, dtype=dtype, name="data2")

    res = sub_compute(data1, data2, output_z, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data1, data2, res]}
    tbe.cce_build_code(sch, config)
