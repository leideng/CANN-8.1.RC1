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
MatrixSetDiagD: Returns a batched matrix tensor with new batched diagonal values
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


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant.
    """
    SCALAR_NEGATIVE_ONE = -1
    SCALAR_TWO = 2


# 'pylint: disable = unused-argument
def get_op_support_info(input_matrix, input_diagonal, input_help, output_matrix,
                        kernel_name="matrix_set_diag_d"):
    """
    get_op_support_info
    """
    format_matrix = input_matrix.get("format").upper()
    format_diagonal = input_diagonal.get("format").upper()
    shape_input_matrix = input_matrix.get("shape")
    if format_matrix in ("ND", "NHWC", "NCHW") and format_diagonal in ("ND", "NHWC", "NCHW"):
        if len(shape_input_matrix) > 2:
            axis_split_matrix = [[SplitInput([0, [0], [-1], [-1]], [1, [0], [-1], [-1]], [2, [0], [-1], [-1]]), \
                                SplitOutput([0, [0]])]]
        else:
            axis_split_matrix = None

    else:
        axis_split_matrix = None
    axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


def _check_tensor_size(shape_x, shape_y):
    """
    Check whether matrix_set_diag_d is supported or not.

    Parameters
    ----------
    shape_x: list
        shape of the first tensor x
    shape_y: list
        shape of the second tensor y with the same type and shape with x

    Returns
    -------
    None
    """
    len_x = len(shape_x)
    len_y = len(shape_y)

    if (len_x < Constant.SCALAR_TWO) or (len_y < Constant.SCALAR_TWO):
        error_detail = "Only the rank of input tensors >= 2 are supported!"
        error_manager_vector.raise_err_two_input_shape_invalid("matrix_set_diag_d", "input_matrix", \
                                                               "input_help", error_detail)
    if len_x == len_y:
        for i in range(len_x):
            if shape_x[i] != shape_y[i]:
                error_detail = "The input_matrix and input_help are not with the same dimension!"
                error_manager_vector.raise_err_two_input_shape_invalid("matrix_set_diag_d", "input_matrix", \
                                                                       "input_help", error_detail)
    else:
        error_detail = "The input_x and input_y are not with the same rank!"
        error_manager_vector.raise_err_two_input_shape_invalid("matrix_set_diag_d", "input_matrix", \
                                                               "input_help", error_detail)


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals
@register_operator_compute("matrix_set_diag_d", op_mode="static", support_fusion=True)
def matrix_set_diag_d_compute(input_matrix, input_diagonal, input_help,
                              output_matrix, kernel_name="matrix_set_diag_d"):
    """
    how to make matrix_set_diag_d compute these tensors.
    -----------
    According to the auxiliary matrix and diagonal, res1 matrix is generated,
    then the matrix points with all diagonal zeros are multiplied by the input
    matrix, and finally the sum is added.

    Parameters
    ----------
    input_matrix: TVM tensor
        the placeholder of input_matrix
    input_diagonal: TVM tensor
        the placeholder of input_diagonal
    input_help: TVM tensor
        the placeholder of input_help
    output_matrix: dict
        dict of output
    kernel_name: str
        kernel name, default value is "matrix_set_diag_d"

    Returns
    -------
    res: TVM tensor
        the result of matrix_set_diag_d_compute
    """
    shape_input = shape_util.shape_to_list(input_matrix.shape)
    input_dtype = input_matrix.dtype

    if input_dtype in ("int8", "uint8"):
        input_matrix = tbe.cast_to(input_matrix, "float16")
        input_diagonal = tbe.cast_to(input_diagonal, "float16")
        input_help = tbe.cast_to(input_help, "float16")

    diag_tmp = tbe.broadcast(input_diagonal, shape_input)
    help_tmp = tbe.vadds(input_help, Constant.SCALAR_NEGATIVE_ONE)
    help_y = tbe.vabs(help_tmp)

    res_vmul_x = tbe.vmul(input_matrix, help_y)
    res_vmul_y = tbe.vmul(diag_tmp, input_help)
    res = tbe.vadd(res_vmul_x, res_vmul_y)

    if input_dtype in ("int8", "uint8"):
        res = tbe.cast_to(res, input_dtype, f1628IntegerFlag=True)

    return res


# 'pylint: disable=locally-disabled,too-many-locals
# @register_operator("MatrixSetDiagD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def matrix_set_diag_d(input_matrix, input_diagonal, input_help, output_matrix,
                      kernel_name="matrix_set_diag_d"):
    """
    algorithm: matrix_set_diag_d

    Parameters
    ----------
    input_matrix: dict with keys(shape and dtype)
        dtype only support float16, float32, int32, int8, uint8.
    input_diagonal: dict with keys(shape and dtype)
        dtype only support float16, float32, int32, int8, uint8.
    input_help: dict with keys(shape and dtype)
        dtype only support float16, float32, int32, int8, uint8.
    output_matrix: dict
        dict of output
    kernel_name: str
        kernel name, default value is "matrix_set_diag_d"

    Returns
    -------
    None
    """
    shape_input = input_matrix.get("shape")
    shape_diag = input_diagonal.get("shape")
    dtype_diagonal = input_diagonal.get("dtype")
    dtype_help = input_help.get("dtype")
    help_matrix = input_help.get("shape")
    dtype = input_matrix.get("dtype")

    para_check.check_shape(shape_input, param_name="input_matrix")
    para_check.check_shape(shape_diag, param_name="input_diagonal")
    para_check.check_shape(help_matrix, param_name="input_help")


    # Check help_matrix can really help.
    _check_tensor_size(shape_input, help_matrix)

    # Adjust diag's shape according to input shape.
    # Extend the shape_diag dimension for broadcast.
    if shape_input[-2] <= shape_input[-1]:
        shape_b_newshape = list(shape_diag) + [1]
    # The penultimate dimension of the shape_diag is extended for broadcast.
    else:
        shape_b_newshape = list(shape_diag)
        shape_b_newshape.insert(-1, 1)

    check_list = ("float16", "float32", "int32", "int8", "uint8")
    input_dtype = dtype.lower()
    input_dtype_diagonal = dtype_diagonal.lower()
    input_dtype_help = dtype_help.lower()
    para_check.check_dtype(input_dtype, check_list, param_name="input_matrix")
    para_check.check_dtype(input_dtype_diagonal, check_list, param_name="input_diagonal")
    para_check.check_dtype(input_dtype_help, check_list, param_name="input_help")

    data_a = tvm.placeholder(shape_input, name="data_a", dtype=input_dtype)
    data_b = tvm.placeholder(shape_b_newshape, name="data_b", dtype=input_dtype)
    help_x = tvm.placeholder(help_matrix, name="help_x", dtype=input_dtype)

    res = matrix_set_diag_d_compute(data_a, data_b, help_x,
                                    output_matrix, kernel_name)
    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_a, data_b, help_x, res]}
    build(sch, config)
