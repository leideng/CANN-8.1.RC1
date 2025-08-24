#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

fused_mul_add_add
"""

from collections import namedtuple

import tbe.dsl as tbe
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.utils import para_check
from impl.util.platform_adapter import error_manager_vector


def _check_format_pattern(input0, input1, input2, input3):
    """
    check format pattern of inputs. must be NZ+ND+ND+NZ
    """
    format_input0 = input0.get("format").upper()
    format_input1 = input1.get("format").upper()
    format_input2 = input2.get("format").upper()
    format_input3 = input3.get("format").upper()

    list_format = [format_input0, format_input1, format_input2, format_input3]
    list_pattern = ["FRACTAL_NZ", "ND", "ND", "FRACTAL_NZ"]

    if list_format != list_pattern:
        error_detail = 'The format pattern of inputs are not supported!'
        error_manager_vector.raise_err_specific_reson("fused_mul_add_add", error_detail)


def _infer_shape(input0, input1, input2, input3):
    """
    shape_input0 : FRACTAL_NZ, [N,...,A,B,16,16]
    last_two_dims : [B*16, A*16]
    """
    shape_input0 = list(shape_util.scalar2tensor_one(input0.get("shape")))
    shape_input1 = list(shape_util.scalar2tensor_one(input1.get("shape")))
    shape_input2 = list(shape_util.scalar2tensor_one(input2.get("shape")))
    shape_input3 = list(shape_util.scalar2tensor_one(input3.get("shape")))

    last_two_dims = [shape_input0[-2] * shape_input0[-3], shape_input0[-4] * shape_input0[-1]]

    # infer shape for input1
    if (len(shape_input1) >= 2) and (shape_input1[-1] == 1) and (shape_input1[-2] == last_two_dims[-2]):
        shape_input1.insert(-2, 1)
        shape_input1.insert(-2, 1)
        shape_input1[-2] = shape_input0[-2]
        shape_input1[-3] = shape_input0[-3]
    else:
        error_detail = 'either shape of input1 or input0 is illegal'
        error_manager_vector.raise_err_specific_reson("fused_mul_add_add", error_detail)

    # infer shape for input2
    if shape_input2[-1] == last_two_dims[-1]:
        shape_input2.insert(-1, 1)
        shape_input2.insert(-1, 1)
        shape_input2.insert(-1, 1)
        shape_input2[-4] = shape_input0[-4]
        shape_input2[-1] = shape_input0[-1]
    elif len(shape_input2) == 1 and shape_input2[-1] == 1:
        shape_input2.insert(-1, 1)
        shape_input2.insert(-1, 1)
        shape_input2.insert(-1, 1)
    else:
        error_detail = 'either shape of input2 or input0 is illegal'
        error_manager_vector.raise_err_specific_reson("fused_mul_add_add", error_detail)

    # check shape of input3
    if shape_input3 != shape_input0:
        error_detail = 'shape of input3 is not same as input0'
        error_manager_vector.raise_err_specific_reson("fused_mul_add_add", error_detail)

    InputShape = namedtuple('InputShape', ['input0_shape', 'input1_shape', 'input2_shape', 'input3_shape'])

    res = InputShape(shape_input0, shape_input1, shape_input2, shape_input3)

    return res


def _data_broadcast(data_1, data_2):
    """
    broadcast data to same shape by their max shape of inputs

    Parameters
    ----------
    data_1: TVM tensor. the placeholder of first input data
    data_2: TVM tensor. the placeholder of second input data

    Returns
    -------
    data_1: TVM tensor. the placeholder of first input broadcast
    data_2: TVM tensor. the placeholder of second input broadcast
    """
    shape_x = shape_util.shape_to_list(data_1.shape)
    shape_y = shape_util.shape_to_list(data_2.shape)
    if shape_x != shape_y:
        shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x, shape_y,
                                                                  param_name_input1="data_1",
                                                                  param_name_input2="data_2")
        data_1 = tbe.broadcast(data_1, shape_max)
        data_2 = tbe.broadcast(data_2, shape_max)

    return data_1, data_2


# 'pylint: disable=unused-argument,too-many-arguments
def fused_mul_add_add_compute(data_input0, data_input1, data_input2, data_input3, output,
                              kernel_name="fused_mul_add_add"):
    """
    fused_mul_add_add compute function

    Parameters
    ----------
    data_input0: TVM tensor. the input tensor of mul
    data_input1: TVM tensor. the input tensor of mul
    data_input2: TVM tensor. the input tensor of first add
    data_input3: TVM tensor. the input tensor of second add
    output: TVM tensor. the output tensor of add
    kernel_name : str. kernel name, default value is "fused_mul_add_add"

    Returns
    -------
    res tensor
    """
    data_input0, data_input1 = _data_broadcast(data_input0, data_input1)
    data_input0, data_input2 = _data_broadcast(data_input0, data_input2)
    # mul
    mul_res = tbe.vmul(data_input0, data_input1)

    # first add
    first_add_res = tbe.vadd(mul_res, data_input2)

    # second add
    res = tbe.vadd(first_add_res, data_input3)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def fused_mul_add_add(input0, input1, input2, input3, output, kernel_name="fused_mul_add_add"):
    """
    The interface of FusedMulAddAdd operator

    Parameters
    ----------
    input0: dict
         the dict of input of mul, support float16, float32, int32
    input1: dict
         the dict of input of mul, support float16, float32, int32
    input2: dict
         the dict of input of first add, support float16, float32, int32
    input3: dict
         the dict of input of second add, support float16, float32, int32
    output: dict
         the dict of output
    kernel_name: str
        cce kernel name, default value is fused_mul_add_add

    Returns
    -------
    None
    """
    # check data type for all input
    dtype_tuple = ("float16", "float32", "int32")
    dtype_input0 = input0.get("dtype").lower()
    dtype_input1 = input1.get("dtype").lower()
    dtype_input2 = input2.get("dtype").lower()
    dtype_input3 = input3.get("dtype").lower()
    para_check.check_dtype(dtype_input0, dtype_tuple, param_name="x1")
    para_check.check_dtype(dtype_input1, dtype_tuple, param_name="x2")
    para_check.check_dtype(dtype_input2, dtype_tuple, param_name="x3")
    para_check.check_dtype(dtype_input3, dtype_tuple, param_name="x4")

    # check format pattern of inputs
    _check_format_pattern(input0, input1, input2, input3)

    # infer shape
    res = _infer_shape(input0, input1, input2, input3)

    data_input0 = tvm.placeholder(res.input0_shape, name="x1", dtype=dtype_input0)
    data_input1 = tvm.placeholder(res.input1_shape, name="x2", dtype=dtype_input1)
    data_input2 = tvm.placeholder(res.input2_shape, name="x3", dtype=dtype_input2)
    data_input3 = tvm.placeholder(res.input3_shape, name="x4", dtype=dtype_input3)

    # compute
    res = fused_mul_add_add_compute(data_input0, data_input1, data_input2, data_input3,
                                    output, kernel_name="fused_mul_add_add")

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": (data_input0, data_input1, data_input2, data_input3, res)}

    tbe.build(sch, config)
