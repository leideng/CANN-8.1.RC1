#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

slice with axes
"""
from __future__ import absolute_import
from impl.util import util_select_op_base
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from impl.util.util_common import is_unknown_rank_input
from impl.util.util_common import update_shape_base_other_format
from .slice import get_dtype_and_format
from .slice import slice_compute
from .slice import update_params_for_other_format


# 'pylint: disable=unused-argument,invalid-name
def op_select_format(x, offsets, size, y, axes, kernel_name="slice_with_axes"):
    """
    define the op_select_format for SliceWithAxes Op

    dtype_support:
          "float", "float16", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool"

    format_support:
        1. when both offsets and size are not const, the Op Select can support ND.

        2. when both offsets and size are const,
           slice can support 5HD by meeting any of the following conditions.
            conditions as follows:
                2.1: C dim in start is c0 align and C dim in size is c0 align
                    or size_c = shape_c - start_c(means will slice all remain data from start_c)
                2.2: C Dim in start is 0, C Dim in size == C Dim in shape(means will slice without c)

            for example:
                inputs: 
                    x : Tensor of (shape=(-1, 128, -1, -1), "NCHW")
                    begin: value is [2, 16, 4, 7]  C begin is 16
                    size: value is [-1, -1, -1, -1]  C size is -1
                the Op Select can process with NC1HWC0:
                    x : Tensor of (shape=(-1, 8, -1, -1, 16), "NC1HWC0")
                    y : Tensor of (shape=(-1, 7, -1, -1, 16), "NC1HWC0")

        3. when both offsets and size are const,
           slice can support FRACTAL_Z and FRACTAL_Z_3D by meeting any of the following conditions.
            conditions as follows:
                3.1: C/N dim in start is c0 align and
                    C/N dim in size is c0 align or size_c/size_n = shape_c/shape_n - start_c/start_n
                    means will slice all remain data from start_c/start_n

            for example:
                    inputs: 
                        x : Tensor of (shape=(128, 128, -1, -1), "NCHW")
                        begin: value is [0, 16, 4, 7]  C begin is 16
                        size: value is [-1, -1, -1, -1]  C size is -1
                    the Op Select can process with NC1HWC0:
                        x : Tensor of (shape=(-1, -1, 8, 8, 16, 16), "FRACTAL_Z")
                        y : Tensor of (shape=(-1, -1, 8, 7, 16, 16), "FRACTAL_Z")
        
        4. when both offsets and size are const,
           slice can support FRACTAL_NZ by meeting any of the following conditions.
            conditions as follows:
                3.1: last two dims in start is c0 align and
                     (last two dimss in size is c0 align or
                        size of last two dims = shape of last two dims - start of last two dims)
                      means will slice all remain data from last two dims

            for example:
                    inputs: 
                        x : Tensor of (shape=(128, 128, 128), "NCHW")
                        begin: value is [120, 16, 0]  C begin is 16
                        size: value is [-1, -1, -1]  C size is -1
                    the Op Select can process with NC1HWC0:
                        x : Tensor of (shape=(128, 8, 8, 16, 16), "FRACTAL_NZ")
                        y : Tensor of (shape=(8, 7, 8, 16, 16), "FRACTAL_NZ")
    """
    bfp16_support = tbe_platform.intrinsic_check_support("Intrinsic_vconv", "bf162f32")
    if not bfp16_support:
        base_x_type = ("float", "float16", "int8", "int16", "int32", "int64", "uint8", "uint16",
                        "uint32", "uint64", "bool")
    else:
        base_x_type = ("bfloat16", "float", "float16", "int8", "int16", "int32", "int64", "uint8", "uint16",
                        "uint32", "uint64", "bool")
    dtype_x_out = list(base_x_type)
    format_x_out = ["ND"] * len(base_x_type)

    ori_offsets_value = offsets.get("const_value")
    ori_size_value = size.get("const_value")

    if ori_offsets_value and ori_size_value and not is_unknown_rank_input(x):
        dims_x = len(x.get("ori_shape"))
        dims_axes = len(axes)
        offsets_value = [0 for _ in range(dims_x)]
        size_value = [-1 for _ in range(dims_x)]

        for i in range(dims_axes):
            offsets_value[axes[i]] = ori_offsets_value[i]
            size_value[axes[i]] = ori_size_value[i]

        dtype_x_out, format_x_out = get_dtype_and_format(x, offsets_value, size_value, dtype_x_out, format_x_out)

    base_format_len = len(format_x_out)
    dtype_x_out = dtype_x_out * 2
    format_x_out = format_x_out * 2
    other_input_type = ["int32"] * base_format_len + ["int64"] * base_format_len
    other_input_format_type = ["ND"] * base_format_len * 2

    x_dtype_str = ','.join(dtype_x_out)
    x_format_str = ','.join(format_x_out)
    other_input_dtype_str = ','.join(other_input_type)
    other_input_format_str = ','.join(other_input_format_type)

    input0 = util_select_op_base.gen_param(classify="input0",
                                           name="x",
                                           datatype=x_dtype_str,
                                           format=x_format_str,
                                           unknownshape_format=x_format_str)
    input1 = util_select_op_base.gen_param(classify="input1",
                                           name="offsets",
                                           datatype=other_input_dtype_str,
                                           format=other_input_format_str,
                                           unknownshape_format=other_input_format_str)
    input2 = util_select_op_base.gen_param(classify="input2",
                                           name="size",
                                           datatype=other_input_dtype_str,
                                           format=other_input_format_str,
                                           unknownshape_format=other_input_format_str)
    output0 = util_select_op_base.gen_param(classify="output0",
                                            name="y",
                                            datatype=x_dtype_str,
                                            format=x_format_str,
                                            unknownshape_format=x_format_str)
    param_list = [input0, input1, input2, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def update_input_params(x, offsets, size, axes):
    """
    update input params in known shape
    """
    input_format = x.get("format")

    ori_offsets_value = offsets.get("const_value")
    ori_size_value = size.get("const_value")

    if ori_offsets_value and ori_size_value:
        dims_x = len(x.get("ori_shape"))
        dims_axes = len(axes)
        offsets_value = [0 for _ in range(dims_x)]
        size_value = [-1 for _ in range(dims_x)]

        for i in range(dims_axes):
            offsets_value[axes[i]] = ori_offsets_value[i]
            size_value[axes[i]] = ori_size_value[i]

        if not is_unknown_rank_input([x]) and offsets_value and size_value and \
            input_format in ("NDC1HWC0", "NC1HWC0", "FRACTAL_NZ", "FRACTAL_Z", "FRACTAL_Z_3D"):
            # reshape (C1HW)NiNoC0/(DC1HW)NiNoC0 to C1HWNiNoC0/DC1HWNiNoC0
            x = update_shape_base_other_format(x)

            # update offsets, size base on ori_format
            offsets_value, size_value = update_params_for_other_format(x.get("ori_shape"), offsets_value, size_value,
                                                                    input_format, x.get("ori_format"))

        # update offsets/size const value
        offsets["const_value"] = offsets_value
        size["const_value"] = size_value

        # update offsets/size shape and ori_shape
        offsets["shape"] = [len(offsets_value)]
        size["shape"] = [len(offsets_value)]
        offsets["ori_shape"] = [len(offsets_value)]
        size["ori_shape"] = [len(offsets_value)]


# 'pylint: disable=unused-argument,invalid-name
def slice_with_axes_dsl(x, offsets, size, y, axes, kernel_name="slice_with_axes"):
    """
    slice interface for dsl
    """
    update_input_params(x, offsets, size, axes)

    x_dtype = x.get("dtype").lower()
    offsets_dtype = offsets.get("dtype").lower()
    size_dtype = size.get("dtype").lower()

    tbe_context.get_context().add_compile_info("is_with_axes", True)
    ins = classify([x, offsets, size], "slice", {"end_mode": "size"})
    schedules, tensors = [], []
    for shape_x, shape_offsets, shape_size in ins:
        with tbe.compute():
            x_var, offsets_list, size_list = \
                shape_util.variable_shape([shape_x, shape_offsets, shape_size], "slice")
            x_tensor = tvm.placeholder(x_var, name="x", dtype=x_dtype)
            offsets_tensor = tvm.placeholder([len(offsets_list)], name="offsets", dtype=offsets_dtype)
            size_tensor = tvm.placeholder([len(size_list)], name="size", dtype=size_dtype)
            res = slice_compute(x_tensor, offsets_list, size_list, y, kernel_name)
            tensors.append([x_tensor, offsets_tensor, size_tensor, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)


# 'pylint: disable=locally-disabled,too-many-arguments,invalid-name,unused-argument
# 'pylint: disable=unused-argument,too-many-locals,redefined-builtin
@register_operator("SliceWithAxes")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_LIST_INT, para_check.KERNEL_NAME)
def slice_with_axes(x, offsets, size, y, axes, kernel_name="slice_with_axes"):
    """
    algorithm: slice_with_axes
    calculating: this operation extracts a slice of size size
                 from a tensor input
                 starting at the location specified by begin.

    Parameters
    ----------
    x: dict
        contains shape and dtype information of input tensor
    y: dict
        contains shape and dtype information of output tensor
    offsets: dict
        represents the index of the first value to select
    size: dict
        represents the shape of output tensor
    axes: ListInt
        the axes of input tensor to select
    kernel_name: str
        cce kernel name, default value is "slice".

    Returns
    -------
    the result of slice_with_axes
    """
    x_dtype = x.get("dtype").lower()
    offsets_dtype = offsets.get("dtype").lower()
    size_dtype = size.get("dtype").lower()
    check_list_x = ("bfloat16", "float32", "float16", "bool", "int8", "int16", "int32", "int64",
                    "uint8", "uint16", "uint32", "uint64")
    check_list_offsets = ("int32", "int64")
    check_list_size = ("int32", "int64")
    para_check.check_dtype(x_dtype, check_list_x, param_name="x")
    para_check.check_dtype(offsets_dtype, check_list_offsets, param_name="offsets")
    para_check.check_dtype(size_dtype, check_list_size, param_name="size")

    slice_with_axes_dsl(x, offsets, size, y, axes, kernel_name)

