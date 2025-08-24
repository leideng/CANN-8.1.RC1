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
read_select
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tvm
from impl.util import util_common
from impl.util.platform_adapter import error_manager_vector
from impl.util.util_select_op_base import get_op_cal_info
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput

READ_SELECT_TAG = "read_select"
PARA_LIST_LEN = 5
EMPTY_LIST_LEN = 0


# 'pylint: disable=unused-argument
def get_op_support_info(input_x, output_x, stride_list, output_tensor_dim, kernel_name="read_select"):
    """
    get_op_support_info, lxfusion slice inference for read_select
    """
    ori_input_shape = input_x.get("ori_shape")
    ori_output_shape = output_x.get("ori_shape")
    input_format = input_x.get("format").upper()
    ori_format = input_x.get("ori_format").upper()
    axis_split_matrix = []
    axis_reduce_list = []
    if ori_input_shape and ori_output_shape and len(ori_input_shape) == len(ori_output_shape):
        for i, _ in enumerate(ori_input_shape):
            axis = i
            if input_format != ori_format:
                axis = util_common.update_axis_for_other_format(ori_input_shape, axis, input_format, ori_format, False)
            if ori_input_shape[i] == ori_output_shape[i]:
                split_0 = [SplitInput([0, [axis], [-1], [-1]]), SplitOutput([0, [axis]])]
                axis_split_matrix.append(split_0)
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


def _check_para_list_len(total_shape, valid_shape, slice_offset, stride_list):
    if len(total_shape) != PARA_LIST_LEN:
        error_manager_vector.raise_err_specific_reson("read_select", "the len of input shape should be 5")

    if (len(valid_shape) != PARA_LIST_LEN) and (len(valid_shape) != EMPTY_LIST_LEN):
        error_manager_vector.raise_err_specific_reson("read_select", "the len of valid shape should be 5 or 0")

    if (len(slice_offset) != PARA_LIST_LEN) and (len(slice_offset) != EMPTY_LIST_LEN):
        error_manager_vector.raise_err_specific_reson("read_select", "the len of slice offset should be 5 or 0")

    if len(stride_list) != PARA_LIST_LEN:
        error_manager_vector.raise_err_specific_reson("read_select", "the len of stride list should be 5")


# 'pylint: disable=locally-disabled,too-many-locals,unused-argument,dangerous-default-value
@register_operator_compute("read_select", op_mode="static", support_fusion=True)
def read_select_compute(input_tensor,
                        output_x,
                        stride_list=[1, 1, 1, 1, 1],
                        output_tensor_dim=4,
                        kernel_name="read_select"):
    """
    calculating data

    Parameters
    ----------
    input_tensor : TVM tensor
        the placeholder of input_x
    output_x : dict
        dict of output_x, include keys(shape and dtype)
    stride_list : list
        list of stride for 5HD shape
    output_tensor_dim: int
        output tensor 4d or 5d
    kernel_name : str
        kernel name, default value is "read_select"

    Returns
    -------
    output tensor
    """
    total_shape = input_tensor.shape
    n_total, c1_total, h_total, w_total, c0_total = total_shape

    # valid_shape and slice_offset are all 5HD shape
    valid_shape = input_tensor.op.attrs['valid_shape']
    slice_offset = input_tensor.op.attrs['slice_offset']
    _check_para_list_len(total_shape, valid_shape, slice_offset, stride_list)

    if len(valid_shape) == EMPTY_LIST_LEN:
        valid_shape = [
            n_total, c1_total, (h_total + stride_list[2] - 1) // stride_list[2],
            (w_total + stride_list[3] - 1) // stride_list[3], c0_total
        ]

    if len(slice_offset) == EMPTY_LIST_LEN:
        slice_offset = [0, 0, 0, 0, 0]

    n_valid, c1_valid, h_valid, w_valid, c0_valid = valid_shape

    input_tensor.op.attrs["dma_copy"] = True
    output_shape = valid_shape
    output_ub_5d = \
        tvm.compute(output_shape,
                    lambda n, c1, h, w, c0:
                    input_tensor(n, c1, slice_offset[2] + h*stride_list[2],
                                 w*stride_list[3], c0),
                    name="output_ub_5d", attrs=input_tensor.op.attrs)

    if output_tensor_dim == 5:
        return output_ub_5d

    output_shape_4d = (n_valid, c1_valid, h_valid * w_valid, c0_valid)
    output_ub_4d = \
        tvm.compute(output_shape_4d,
                    lambda n, c1, hw, c0: output_ub_5d(n, c1,
                    hw // w_valid, hw % w_valid, c0),
                    name="output_ub_4d")

    return output_ub_4d


# 'pylint: disable=locally-disabled,unexpected-keyword-arg,unnecessary-lambda
@para_check.check_input_type(dict, dict, (tuple, list), int, str)
def read_select(input_x, output_x, stride_list=[1, 1, 1, 1, 1], output_tensor_dim=4, kernel_name="read_select"):
    """
    Read data with offset and stride

    Parameters
    ----------
    input_x : dict
        dict of input_x, include keys(shape and dtype)
    output_x : dict
        dict of output_x, include keys(shape and dtype)
    stride_list : list
        list of stride for 5HD shape
    output_tensor_dim: int
        output tensor 4d or 5d
    kernel_name : str
        kernel name, default value is "read_select"

    Returns
    -------
    output tensor
    """
    total_shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()
    valid_shape = input_x.get("valid_shape")
    slice_offset = input_x.get("slice_offset")

    para_check.check_shape_rule(total_shape)
    if len(valid_shape) != EMPTY_LIST_LEN:
        para_check.check_shape_rule(valid_shape)
        para_check.check_tensor_shape_size(valid_shape)
    para_check.check_tensor_shape_size(total_shape)
    para_check.check_kernel_name(kernel_name)

    _check_para_list_len(total_shape, valid_shape, slice_offset, stride_list)

    check_list = ["float16", "int8", "int16"]
    if input_dtype not in check_list:
        error_manager_vector.raise_err_input_dtype_not_supported("read_select", "input_x", "float16, int8, int16",
                                                                 str(input_dtype))

    src_in_flag = "DDR"
    if "src_in_flag" in input_x:
        src_in_flag = input_x.get("src_in_flag")

    input_tensor = tvm.placeholder(total_shape,
                                   name="input_tensor",
                                   dtype=input_dtype,
                                   attrs={
                                       "valid_shape": valid_shape,
                                       "slice_offset": slice_offset,
                                       "src_in_flag": src_in_flag
                                   })

    output_tensor = read_select_compute(input_tensor, output_x, stride_list, output_tensor_dim, kernel_name=kernel_name)
    if output_tensor_dim == 5:
        output_5d_tensor = output_tensor
    else:
        output_5d = output_tensor.op.input_tensors
        output_5d_tensor = output_5d[0]

    res = tvm.compute(output_5d_tensor.shape,
                      lambda *indice: output_5d_tensor(*indice),
                      name="res",
                      tag=READ_SELECT_TAG)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [input_tensor, res]}
    tbe.build(sch, config)
