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
batch_to_space_d
"""
from impl.util.platform_adapter import tbe_build
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.batch_to_space_nd_d import BatchToSpaceNdFive
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info


# 'pylint: disable = unused-argument,too-many-locals,invalid-name
def get_op_support_info(x, y, block_size, crops, kernel_name="batch_to_space_d"):
    """get op support info
    """
    format_x = x.get("format").upper()
    if format_x == "NC1HWC0":
        axis_split_matrix = [[SplitInput([0, [1], [-1], [-1]]), SplitOutput([0, [1]])]]
        axis_reduce_list = None

    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT,
                            (para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_LIST_INT),
                            para_check.KERNEL_NAME)
def batch_to_space_d(x, y, block_size, crops, kernel_name="batch_to_space_d"):
    """BatchToSpace for tensor.

    Parameters
    ----------
    x: dict
        the dict of input tensor.
    y: dict
        the dict of output tensor.
    block_size: int
        the size of block.
    crops: list or tuple
        2-D with shape [2, 2], crops[i] = [crop_start, crop_end].
    kernel_name: str
        cce kernel name, default value is "batch_to_space".

    Returns
    -------
    None.
    """
    input_shape = x.get("shape")
    input_dtype = x.get("dtype").lower()
    if len(crops) == 4:
        crops = [[crops[0], crops[1]], [crops[2], crops[3]]]
    para_check.check_shape(input_shape, param_name="x")
    check_list = {"float16", "float32"}
    para_check.check_dtype(input_dtype, check_list, param_name="x")

    if len([x for x in input_shape if isinstance(x, int) and x > 0]) != len(input_shape):
        error_detail = "input_shape of x should be positive integer"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "x", error_detail)

    dim_cnt = 5
    if len(input_shape) != dim_cnt:
        error_detail = "the length of input_shape must be 5,while it is: %d" % len(input_shape)
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "x", error_detail)

    crops_len = 2
    if not (len(crops) == crops_len and len(crops[0]) == crops_len and len(crops[1]) == crops_len):
        error_detail = "shape of crops should be 2*2"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "crops", error_detail)

    if not (isinstance(crops[0][0], int) and crops[0][0] >= 0 and isinstance(crops[0][1], int) and crops[0][1] >= 0 and
            isinstance(crops[1][0], int) and crops[1][0] >= 0 and isinstance(crops[1][1], int) and crops[1][1] >= 0):
        error_detail = "crops  must be >= 0"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "crops", error_detail)

    batch_size = input_shape[0]
    if batch_size % (block_size * block_size) != 0:
        error_detail = "batch_size of x should be divisible by the square of block_size"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "x", error_detail)
    output_shape = (input_shape[0] // block_size // block_size, input_shape[1],
                    input_shape[2] * block_size - crops[0][0] - crops[0][1],
                    input_shape[3] * block_size - crops[1][0] - crops[1][1], input_shape[4])
    para_check.check_shape(output_shape, param_name="y")

    block_shape = [block_size, block_size]
    data = tvm.placeholder(input_shape, name="data", dtype=input_dtype)
    batch = BatchToSpaceNdFive(input_shape, input_dtype, block_shape, crops)
    res = tvm.extern([batch.output_shape], [data],
                     lambda ins, outs: batch.kernel_ir(outs, ins),
                     dtype=input_dtype,
                     name="res")
    sch = tvm.create_schedule(res.op)
    with tbe_build.build_config():
        tvm.build(sch, [data, res], "cce", name=kernel_name)
