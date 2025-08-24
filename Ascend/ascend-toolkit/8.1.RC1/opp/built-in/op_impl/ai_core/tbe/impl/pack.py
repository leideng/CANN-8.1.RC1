#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
pack
"""

from impl.util import util_common
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import error_manager_vector
from impl.util import util_select_op_base
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info
from impl.concat_v2_d import concat_v2_d_util


def op_select_format(x, y, axis, kernel_name="pack"):
    """
    Select format dynamically\n
    1. When the ori_shapes of y and all elements of list x are not -2 and each ori_shape's length of them
    (shape_len for short) is not less then 2, the parameter axis is not euqal to any one of -1, -2, shape_len, 
    (shape_len -1) at the same time, the FRACTAL_NZ is supported;
    2. In other sences, only ND format is supported.
    """
    shape_len = 1
    for i, input_dict in enumerate(x):
        shape_input = input_dict.get("ori_shape")
        shape_input = shape_util.scalar2tensor_one(shape_input)
        if -2 not in shape_input:
            shape_len = len(shape_input)
    if axis is not None:
        pack_axis = axis % (shape_len + 1)

    # check whether support FRACTAL_NZ
    is_support_nz = False
    if shape_len >= 2 and axis is not None:
        is_pack_shape_len_dim = pack_axis == shape_len
        is_pack_last_one_dim = pack_axis == shape_len - 1
        # condition
        # do not pack the tensor with the -1 or -2 dim
        if not (is_pack_shape_len_dim or is_pack_last_one_dim):
            is_support_nz = True

    base_data_type = \
        ["float", "float16", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool"]
    other_data_type = ["float", "float16", "int16", "int32", "uint16", "uint32"]
    bfp16_support = tbe_platform.intrinsic_check_support("Intrinsic_vconv", "bf162f32")
    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        base_data_type.remove("float")
        other_data_type.remove("float")
    elif bfp16_support:
        base_data_type.append("bfloat16")
        other_data_type.append("bfloat16")

    dtype_base_out = base_data_type[:]
    format_base_out = ["ND"] * len(dtype_base_out)
    if is_support_nz and not util_common.is_dynamic_input(x):
        other_format = "FRACTAL_NZ"
        dtype_base_out = dtype_base_out + other_data_type
        format_base_out = format_base_out + [other_format] * len(other_data_type)

    dtype_str = ",".join(dtype_base_out)
    format_str = ",".join(format_base_out)
    input0 = util_select_op_base.gen_param(
        classify="input0", name="x", datatype=dtype_str, format=format_str, unknownshape_format=format_str)
    output0 = util_select_op_base.gen_param(
        classify="output0", name="y", datatype=dtype_str, format=format_str, unknownshape_format=format_str)
    param_list = [input0, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


# 'pylint: disable = unused-argument
def get_op_support_info(x, y, axis, kernel_name="pack"):
    """
    get_op_support_info
    """
    x_len = len(x)
    shape_x_len = len(x[0].get("shape"))
    format_x = x[0].get("format").upper()
    if axis < -1:
        axis = axis + 1
    if axis < 0:
        axis += shape_x_len
    if format_x in ("ND", "NC1HWC0"):
        axis_split_matrix = []
        for i in range(0, shape_x_len-1):
            if i != axis:
                input_list = []
                for j in range(0, x_len):
                    input_0 = [j, [i], [-1], [-1]]
                    input_list.append(input_0)
                split_0 = [SplitInput(*input_list), SplitOutput([0, [i]])]
                axis_split_matrix.append(split_0)

    else:
        axis_split_matrix = None
    axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


def convert_pack_2_concat_input(x, y, axis):
    if axis < -1:
        return x, y, axis + 1

    ori_shape = x[0].get("ori_shape")
    shape = x[0].get("shape")
    if len(shape) == len(ori_shape) and axis in (-1, len(shape)):
        for item in x:
            item["shape"] = list(item["shape"])
            item["ori_shape"] = list(item["ori_shape"])
            item["shape"].append(1)
            item["ori_shape"].append(1)

    return x, y, axis


@para_check.check_op_params(para_check.DYNAMIC_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def pack(x, y, axis, kernel_name="pack"):
    """
    algorithm: pack
    Concatenates tensors along one dimension.
    Parameters
    ----------
    x : A list of `dict`.dict include keys shape and dtype
    y: dict of output_data, dict include keys shape and dtype
    axis : int, in the range [-rank(values), rank(values)
    kernel_name : cce kernel name, default value is "pack"
    Returns
    -------
    None
    """
    check_list = ("int8", "int16", "int32", "int64", "uint8", "bool",
                  "uint16", "uint32", "uint64", "float16", "float32")
    data = []
    for i, input_dict in enumerate(x):
        shape_input = input_dict.get("shape")
        input_format = input_dict.get("format")
        if input_format == "FRACTAL_NZ":
            align_len = 16
            para_check.check_shape(shape_input, min_rank=2, param_name="x")
            last_one_dim = shape_input[-1]
            last_second_dim = shape_input[-2]
            if last_one_dim != align_len or last_second_dim != align_len:
                error_detail = "when input format is FRACTAL_NZ, all of the last two dims must be 16"
                error_manager_vector.raise_err_input_shape_invalid(kernel_name, "x", error_detail)
        else:
            para_check.check_shape(shape_input, param_name="x")
        para_check.check_dtype(input_dict.get("dtype").lower(), check_list, param_name="x")
        input_dtype = (input_dict.get("dtype")).lower()
        input_dtype = input_dtype if input_dtype != "bool" else "int8"
        input_dict["dtype"] = input_dtype
        data.append(tvm.placeholder(shape_input, name="data_%d" % i,
                                    dtype=input_dtype))

    left_value = -len((x[0].get("shape")))-1
    right_value = len((x[0].get("shape")))
    if axis < left_value or axis > right_value:
        expect_value = "[%s, %s]".format(str(left_value), str(right_value))
        error_manager_vector.raise_err_input_value_invalid("pack",
                                                           "axis",
                                                           expect_value, str(axis))

    x, y, axis = convert_pack_2_concat_input(x, y, axis)
    concat_v2_d_util(x, y, axis, kernel_name, op_type="Pack")
