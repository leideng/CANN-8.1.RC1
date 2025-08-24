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
matrix_diag_d
"""
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info

# `define a scalar, value = 2`
SCALAR_TWO = 2


# 'pylint: disable = unused-argument
def get_op_support_info(input_diagonal, input_help, output_diagonal,
                        kernel_name="matrix_diag_d"):
    """
    get_op_support_info
    """
    format_diagonal = input_diagonal.get("format").upper()
    shape_input_diagonal = input_diagonal.get("shape")
    if format_diagonal == "ND":
        if len(shape_input_diagonal) > 1:
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
@register_operator_compute("matrix_diag_d", op_mode="static", support_fusion=True)
def matrix_diag_d_compute(input_diagonal, input_help, output_diagonal,
                          kernel_name="matrix_diag_d"):
    """
    compute for matrix_diag_d

    Parameters
    ----------
    input_diagonal: TVM tensor
        the placeholder of input diagonal
    input_help: TVM tensor
        the placeholder of input help
    output_diagonal: dict
        dict info of output_diagonal
    kernel_name: str
        cce kernel name, default value is "matrix_diag_d"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    input_help_shape = shape_util.shape_to_list(input_help.shape)
    input_diagonal_dtype = input_diagonal.dtype

    if input_diagonal_dtype in ("int8", "uint8"):
        input_diagonal = tbe.cast_to(input_diagonal, "float16")
    res_broadcast = tbe.broadcast(input_diagonal, input_help_shape)
    res = tbe.vmul(res_broadcast, input_help)
    if input_diagonal_dtype in ("int8", "uint8"):
        res = tbe.cast_to(res, input_diagonal_dtype,
                                  f1628IntegerFlag=True)

    return res


# @register_operator("MatrixDiagD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def matrix_diag_d(input_diagonal, input_help, output_diagonal,
                  kernel_name="matrix_diag_d"):
    """
    Returns a batched diagonal tensor with a given batched diagonal values

    Parameters
    ----------
    input_diagonal: dict
        dict of input_diagonal, include keys(shape and dtype)
    input_help: dict
        dict of input_help, include keys(shape and dtype),
        Its diagonal Line value is 1 else value is 0
    output_diagonal: dict
        dict of  output
    kernel_name: str
        cce kernel name, default value is "matrix_diag_d"

    Returns
    -------
    None
    """
    input_diagonal_shape = input_diagonal.get("shape")
    input_diagonal_dtype = input_diagonal.get("dtype")
    input_help_shape = input_help.get("shape")
    input_help_dtype = input_help.get("dtype")

    para_check.check_shape(input_diagonal_shape, param_name="input_diagonal")
    para_check.check_shape(input_help_shape, param_name="input_help")

    input_diagonal_shape_chgshape = list(input_diagonal_shape)
    # The penultimate dimension of the input_diagonal_shape is
    # extended for broadcast
    input_diagonal_shape_chgshape.insert(-1, 1)
    para_check.check_shape(input_diagonal_shape_chgshape, param_name="input_diagonal")

    if len(input_help_shape) < SCALAR_TWO:
        error_detail = "Only the rank of input tensors >= 2 are supported!"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "input_help", error_detail)
    if input_help_shape[-1] != input_help_shape[-2]:
        error_detail = "the last two dimensions of shape_b must be the same!"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "input_help", error_detail)

    check_list = ("float16", "float32", "int32", "int8", "uint8")
    input_diagonal_dtype = input_diagonal_dtype.lower()
    para_check.check_dtype(input_diagonal_dtype, check_list, param_name="input_diagonal")
    input_help_dtype = input_help_dtype.lower()
    para_check.check_dtype(input_help_dtype, check_list, param_name="input_help")

    input_diagonal = tvm.placeholder(input_diagonal_shape_chgshape,
                                     name="input_diagonal",
                                     dtype=input_diagonal_dtype)
    input_help = tvm.placeholder(input_help_shape, name="input_help",
                                 dtype=input_help_dtype)

    res = matrix_diag_d_compute(input_diagonal, input_help, output_diagonal,
                                kernel_name)
    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_diagonal, input_help, res]}
    build(sch, config)
