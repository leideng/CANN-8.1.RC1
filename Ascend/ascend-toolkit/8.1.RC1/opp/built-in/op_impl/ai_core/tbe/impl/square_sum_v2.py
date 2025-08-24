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
square_sum_v2
"""
import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util
from tbe import tvm
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info

MIN_FP32 = 2**(-126)
# min float16 value
MIN_FP16 = 2**(-24)
VALUE_ONE = 1

SHAPE_SIZE_LIMIT = 200000000


# 'pylint: disable=unused-argument
# 'pylint: disable=too-many-arguments,too-many-locals
def get_op_support_info(input_x,
                        output1,
                        output2,
                        attr1,
                        attr2=True,
                        kernel_name="square_sum_v2"):
    """
    get_op_support_info
    """
    shape_x = shape_util.shape_to_list(input_x.get("shape"))
    axis_d = []
    for i, _ in enumerate(shape_x):
        axis_d.append(i)
    format_x = input_x.get("format").upper()
    axis_split = [i for i in axis_d if i not in attr1]
    if format_x == "ND":
        if attr2:
            axis_split_matrix = []
            for i in axis_split:
                split_0 = [SplitInput([0, [i], [-1], [-1]]), SplitOutput([0, [i]], [1, [i]])]
                axis_split_matrix.append(split_0)
            axis_reduce_list = None
        else:
            axis_split_matrix = None
            axis_reduce_list = None
    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
# 'pylint: disable=locally-disabled,redefined-builtin,too-many-locals,unused-variable
def reduce_sum_d_compute(x,
                         y,
                         axis=None,
                         keepdims=None,
                         kernel_name="reduce_sum_d"):
    """redusce_sum_d compute

    Parameters:
    ----------
    x: TVM tensor
        input tensor.
    y: dict
        the dict of output tensor.
    axis: int, list, tuple or NONETYPE
        the axis for reduce.
    keepdims: bool or NONETYPE
        if true, retains reduced dimensions with length 1.
    kernel_name: str
        cce kernel name, default value is "reduce_sum_d".

    Returns
    -------
    res: TVM tensor
        output tensor, has the same shape and type as input tensor.
    """
    dtype = x.dtype
    if dtype == "float16" and tbe_platform.cce_conf.api_check_support(
            "te.lang.cce.sum", "float32"):
        x = tbe.cast_to(x, "float32")
    res_sum = tbe.sum(x, axis=axis, keepdims=keepdims)
    res = tbe.cast_to(res_sum, dtype)

    return res


def square_compute(input_x, output_y, kernel_name="square"):
    """
    algorithm: square
    calculating data's square,y= x*x

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is square

    Returns
    -------
    res : tvm.tensor
        the result of square
    """
    res = tbe.vmul(input_x, input_x)
    return res


@tbe_platform.fusion_manager.fusion_manager.register("square_sum_v2")
def suqare_sum_v2_compute(input_x,
                          output1,
                          output2,
                          attr1,
                          attr2,
                          kernel_name="square_sum_v2"):
    """
    calculating data

    Parameters
    ----------
    Input and output of fusion graph

    Returns
    -------
    output tensor
    """
    shape = shape_util.shape_to_list(input_x.shape)
    axis_d = []
    if not attr1:
        for i, _ in enumerate(shape):
            axis_d.append(i)
    else:
        axis_d = attr1
    square = square_compute(input_x, {}, kernel_name)

    sum0 = reduce_sum_d_compute(square, {},
                                axis_d,
                                keepdims=attr2,
                                kernel_name=kernel_name)

    return [sum0, square]


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def square_sum_v2(input_x,
                  output1,
                  output2,
                  attr1,
                  attr2=True,
                  kernel_name="square_sum_v2"):
    """
    calculating data

    Parameters
    ----------
    Input and output of fusion graph

    Returns
    -------
    None
    """
    shape = input_x.get("shape")
    dtype = input_x.get("dtype")

    input_dtype = dtype.lower()

    para_check.check_shape(shape, param_name="input_x")

    data_input = tvm.placeholder(shape, name="data_input", dtype=input_dtype)

    res = suqare_sum_v2_compute(data_input, output1, output2, attr1, attr2,
                                kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_input] + list(res)}

    tbe.cce_build_code(sch, config)
