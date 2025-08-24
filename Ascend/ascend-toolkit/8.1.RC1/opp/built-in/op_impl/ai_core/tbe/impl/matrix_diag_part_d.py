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
matrix_diag_part_d
"""
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import build
from impl.util.platform_adapter import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info
from impl.util.platform_adapter import register_operator_compute


# `define a scaler, value = -2`
SCALER_NEGATIVE_TWO = -2


# 'pylint: disable = unused-argument
def get_op_support_info(input_diagonal, input_help,
                        output_diagonal, kernel_name="matrix_diag_part_d"):
    """
    get_op_support_info
    """
    format_diagonal = input_diagonal.get("format").upper()
    shape_input_diagonal = input_diagonal.get("shape")
    if format_diagonal in ("ND", "NCHW", "NHWC"):
        if len(shape_input_diagonal) > 2:
            axis_split_matrix = [[SplitInput([0, [0], [-1], [-1]], [1, [0], [-1], [-1]]), \
                                SplitOutput([0, [0]])]]
        else:
            axis_split_matrix = None

    else:
        axis_split_matrix = None
    axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


# 'pylint: disable=locally-disabled,unused-argument
@register_operator_compute("matrix_diag_part_d", op_mode="static", support_fusion=True)
def matrix_diag_part_d_compute(input_diagonal, input_help, output_diagonal,
                               kernel_name="matrix_diag_part_d"):
    """
    compute for matrix_diag_part_d

    Parameters
    ----------
    input_diagonal: TVM tensor
        the placeholder of input diagonal
    input_help: TVM tensor
        the placeholder of input help
    output_diagonal: dict
        dict of output_diagonal
    kernel_name: str
        cce kernel name, default value is "matrix_diag_part_d"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    shape_input_diagonal = shape_util.shape_to_list(input_diagonal.shape)
    dtype_input_diagonal = input_diagonal.dtype

    res_vmul = tbe.vmul(input_diagonal, input_help)
    if shape_input_diagonal[-2] < shape_input_diagonal[-1]:
        if dtype_input_diagonal == "int32":
            res_vmul = tbe.cast_to(res_vmul, "float32")
        res = tbe.sum(res_vmul, -1)
        if dtype_input_diagonal == "int32":
            res = tbe.cast_to(res, "int32")
    else:
        if dtype_input_diagonal == "int32":
            res_vmul = tbe.cast_to(res_vmul, "float32")
        res = tbe.sum(res_vmul, SCALER_NEGATIVE_TWO)
        if dtype_input_diagonal == "int32":
            res = tbe.cast_to(res, "int32")

    if dtype_input_diagonal in ("int8", "uint8"):
        res = tbe.cast_to(res, dtype_input_diagonal,
                                  f1628IntegerFlag=True)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def matrix_diag_part_d(input_diagonal, input_help,
                       output_diagonal, kernel_name="matrix_diag_part_d"):
    """
    Returns the batched diagonal part of a batched tensor

    Parameters
    ----------
    input_diagonal: dict
        dict of input_diagonal, include keys(shape and dtype)
    input_help: dict
        dict of help Matrix, Its Diagonal Line value is 1 else value is 0
    output_diagonal: dict
        dict of output
    kernel_name: str
        cce kernel name, default value is "matrix_diag_part_d"

    Returns
    -------
    None
    """
    shape_input_diagonal = input_diagonal.get("shape")
    dtype_input_diagonal = input_diagonal.get("dtype")
    shape_input_help = input_help.get("shape")
    dtype_input_help = input_help.get("dtype")

    para_check.check_shape(shape_input_diagonal, param_name="input_diagonal")
    para_check.check_shape(shape_input_help, param_name="input_help")

    if len(shape_input_diagonal) < 2:
        error_detail = "Only the rank of input tensors >= 2 are supported!"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "input_diagonal", error_detail)
    if list(shape_input_diagonal) != list(shape_input_help):
        error_detail = "the shape of input_diagonal and input_help must be equal!"
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "input_diagonal", \
                                                               "input_help", error_detail)

    check_list = ("float16", "float32", "int32", "int8", "uint8")
    dtype_input_diagonal = dtype_input_diagonal.lower()
    para_check.check_dtype(dtype_input_diagonal, check_list, param_name="input_diagonal")
    dtype_input_help = dtype_input_help.lower()
    para_check.check_dtype(dtype_input_help, check_list, param_name="input_help")

    data_input_diagonal = tvm.placeholder(shape_input_diagonal,
                                          name="data_input_diagonal",
                                          dtype=dtype_input_diagonal)
    data_input_help = tvm.placeholder(shape_input_help, name="data_input_help",
                                      dtype=dtype_input_help)

    res = matrix_diag_part_d_compute(data_input_diagonal, data_input_help,
                                     output_diagonal, kernel_name)
    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input_diagonal, data_input_help, res]}
    build(sch, config)
