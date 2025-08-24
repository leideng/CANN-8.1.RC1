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
space_to_batch_d
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_build
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import error_manager_vector
from impl.space_to_batch_nd_d import SpaceToBatchNdFive
from impl.util.util_select_op_base import get_op_cal_info


# 'pylint: disable=invalid-name,unused-argument
def _check_param(x, y, paddings, block_size, kernel_name):
    """check the parameters including shape, dtype, block_shape, paddings and kernel_name.
    """
    shape = x.get("shape")
    dtype = x.get("dtype").lower()
    dtype_list = ("float16", "float32")
    para_check.check_shape(shape, param_name="x")
    para_check.check_dtype(dtype, dtype_list, param_name="x")

    if len(shape) != 5:
        error_detail = "the shape of image_input should be 5, but got: %d" % len(shape)
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "x", error_detail)
    if block_size < 2:
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "block_size", "greater than one", block_size)

    _check_padding(paddings)

    padding_shape = (shape[0], shape[1], shape[2] + paddings[0][0] + paddings[0][1],
                     shape[3] + paddings[1][0] + paddings[1][1], shape[4])
    para_check.check_shape(padding_shape, param_name="paddings")

    padding_height, padding_width = padding_shape[2], padding_shape[3]
    if padding_height % block_size != 0 or padding_width % block_size != 0:
        error_detail = "both height_pad and width_pad must be divisible by block_size"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "paddings", error_detail)

    output_shape = (padding_shape[0] * block_size * block_size, padding_shape[1], padding_shape[2] // block_size,
                    padding_shape[3] // block_size, padding_shape[4])
    para_check.check_shape(output_shape, param_name="y")


def _check_padding(paddings):
    """check the paddings
    """
    if len(paddings) != 2 or len(paddings[0]) != 2 or len(paddings[1]) != 2:
        error_detail = "the shape of paddings should be 2x2"
        error_manager_vector.raise_err_input_shape_invalid("space_to_batch_d", "paddings", error_detail)

    def _check_padding_val(val):
        """check the padding value
        """
        if not (isinstance(val, int) and val >= 0):
            error_detail = "paddings should be integer and must be >= 0"
            error_manager_vector.raise_err_input_shape_invalid("space_to_batch_d", "paddings", error_detail)

    _check_padding_val(paddings[0][0])
    _check_padding_val(paddings[0][1])
    _check_padding_val(paddings[1][0])
    _check_padding_val(paddings[1][1])


def get_op_support_info(x, y, block_size, paddings, kernel_name="space_to_batch_d"):
    """get op support info
    """
    axis_split_list = None
    axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_list, axis_reduce_list)
    return op_cal_info_in_json


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT,
                            (para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_LIST_INT),
                            para_check.KERNEL_NAME)
def space_to_batch_d(x, y, block_size, paddings, kernel_name="space_to_batch_d"):
    """SpaceToBatch for tensors

    Parameters
    ----------
    x: dict,shape and datatype,datatype supports float16,float32
    y: dict,shape and datatype,datatype supports float16,float32
    block_size: must be greater than one. It indicates the block size
    paddings: (tuple, list),the padding of the input with zeros across the
              spatial dimensions as follows:
              paddings = [[pad_top, pad_bottom], [pad_left, pad_right]]
    kernel_name: cce kernel name, default value is "space_to_batch_d"
    Returns
    -------
    None
    """
    if len(paddings) == 4:
        paddings = [[paddings[0], paddings[1]], [paddings[2], paddings[3]]]

    _check_param(x, y, paddings, block_size, kernel_name)

    input_shape = x.get("shape")
    input_dtype = x.get("dtype").lower()
    block_shape = [block_size, block_size]

    data = tvm.placeholder(input_shape, name="data", dtype=input_dtype)
    space = SpaceToBatchNdFive(input_shape, input_dtype, block_shape, paddings)
    res = tvm.extern([space.output_shape], [data],
                     lambda ins, outs: space.kernel_ir(outs, ins),
                     dtype=input_dtype,
                     name="res")
    sch = tvm.create_schedule(res.op)
    with tbe_build.build_config():
        tvm.build(sch, [data, res], "cce", name=kernel_name)
