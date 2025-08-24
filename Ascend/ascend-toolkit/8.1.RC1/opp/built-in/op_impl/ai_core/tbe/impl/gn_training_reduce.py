#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
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
gn_training_reduce
"""
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
# 'pylint: disable=locally-disabled,redefined-builtin
def op_select_format(x, sum, square_sum, num_groups,
                     kernel_name="gn_training_reduce"):
    """
    select format dynamically
    """
    input0 = gen_param(classify="input0", name="x",
                       datatype="float16,float,float16,float",
                       format="NCHW,NHWC,NCHW,NHWC")
    output0 = gen_param(classify="output0", name="sum",
                        datatype="float,float,float,float",
                        format="ND,ND,ND,ND")
    output1 = gen_param(classify="output1", name="square_sum",
                        datatype="float,float,float,float",
                        format="ND,ND,ND,ND")

    param_list = [input0, output0, output1]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def _shape_check(shape_x, data_format, num_groups):
    """
    Function to check if the shape is in line with norms.

    Parameters
    ----------
    shape_x: list or tuple
        x's data shape
    data_format: str
        data format of input x
    num_groups: int
        groups of channel
    Returns
    -------
    None
    """
    if data_format == "NCHW":
        c_index_ = 1
    elif data_format == "NHWC":
        c_index_ = 3
    else:
        para_check.check_format(data_format, ("NCHW", "NHWC"), param_name="x")

    para_check.check_shape(shape_x, min_rank=4, max_rank=4, param_name="x")
    if shape_x[c_index_] % num_groups != 0:
        error_manager_vector.raise_err_check_params_rules("gn_training_reduce", "num_groups must divide C channel",
                                                          "channel and num_groups",
                                                          "{} and {}".format(shape_x[c_index_], num_groups))


@register_operator_compute("gn_training_reduce", op_mode="static", support_fusion=True)
def gn_training_reduce_compute(x, data_format, kernel_name="gn_training_reduce"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of input_x
    data_format: str
        format string of input x
    kernel_name : str
        kernel name, default value is "gn_training_reduce"

    Returns
    -------
    output tensor
    """
    if data_format == "NCHW":
        reduce_axis = [2, 3, 4]
    else:
        reduce_axis = [1, 2, 4]
    dtype = x.dtype
    if dtype == "float16":
        x = tbe.cast_to(x, "float32")
    square_x = tbe.vmul(x, x)
    sum_x, square_sum_x = tbe.tuple_sum([x, square_x], reduce_axis, True)
    res = [sum_x, square_sum_x]
    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def gn_training_reduce(x, sum, square_sum, num_groups=2, kernel_name="gn_training_reduce"):
    """
    calculating data

    Parameters
    ----------
    x: dict
        dict of input, A 5HD Tensor for input data.
    sum: dict
        dict of sum, A `Tensor`. Sum of x.
    square_sum: dict
        dict of square_sum, A `Tensor`. Square sum of x.
    num_groups: int
        A integer value indicates the group in channel.
    kernel_name : str
        kernel name, default value is "gn_training_reduce"

    Returns
    -------
    None
    """
    shape_x = x.get("shape")
    dtype_x = x.get("dtype")
    data_format = x.get("format")
    input_dtype = dtype_x.lower()

    _shape_check(shape_x, data_format, num_groups)
    para_check.check_dtype(input_dtype, ("float16", "float32"), param_name="x")

    # Reshape NCHW -> N[GD]HW
    if data_format == "NCHW":
        shape_x = [shape_x[0], num_groups, shape_x[1] // num_groups, shape_x[2], shape_x[3]]

    # Reshape NHWC -> NHW[GD]
    elif data_format == "NHWC":
        shape_x = [shape_x[0], shape_x[1], shape_x[2], num_groups, shape_x[3] // num_groups]

    x_input = tvm.placeholder(shape_x, name="x_input", dtype=input_dtype)

    res = gn_training_reduce_compute(x_input, data_format, kernel_name)

    with tvm.target.cce():
        schedule = auto_schedule(res)

    tensor_list = [x_input] + list(res)

    config = {"name": kernel_name,
              "tensor_list": tensor_list}

    build(schedule, config)
