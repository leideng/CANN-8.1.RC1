#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

dynamic split_v_d
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import check_support_block_size_16
from impl.split_last_dim import SplitWith5HD
from impl.util import util_common
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.util.util_select_op_base import gen_param
from te.utils.error_manager import error_manager_vector


def op_select_format(input_value, output_data, size_splits, split_dim, num_split, kernel_name="split_v_d"):
    """
    1.when input input_value's ori_shape in ["NHWC"] and split_d by dim N,H,W, and
    the dim C of input_value's ori_shape can be divisible by 16(32 when dtype is int8).
    the Op SplitVD can support ND and NC1HWC0

        for example:
        input_value : Tensor of (shape=(16, 16, 16, 16), "NHWC")
        the Op Select can process with NC1HWC0:
        input_value : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")

    2.when input input_value's ori_shape in ["NDHWC"] and split_d by dim N,D,H,W, and
    the dim C of input_value's ori_shape can be divisible by 16(32 when dtype is int8).
    the Op SplitVD can support ND and NDC1HWC0.

        for example:
        input_value : Tensor of (shape=(16, 1, 1, 16, 16, 16), "NDC1HWC0")

    3.when input input_value's original shape dimension is greater than two and
    split_dim is 0 and the first dim of input_value's ori_shape is N, the Op
    SplitVD can support ND and FRACTAL_NZ.

        for example:
        input_value : Tensor of (shape=(16, 16, 16, 16), "NCHW")
    """
    dtype = input_value.get("dtype").lower()
    if dtype == "int8":
        c0_len = 32 if not check_support_block_size_16() else 16
    else:
        c0_len = 16 if not check_support_block_size_16() else 8
    output_org_shape_list = []
    output_org_format_list = []
    is_support_hd = True
    support_ori_format = \
        util_common.get_fused_format_str(["N", "D", "H", "W", "C"]) \
        + util_common.get_fused_format_str(["N", "H", "W", "C"])
    input_ori_shape = input_value.get("ori_shape")
    input_ori_shape = shape_util.scalar2tensor_one(input_ori_shape)
    input_ori_format = input_value.get("ori_format")
    split_dim = split_dim % len(input_ori_shape)

    for _, output_dict in enumerate(output_data):
        ori_format = output_dict.get("ori_format").upper()
        ori_shape = output_dict.get("ori_shape")
        ori_shape = shape_util.scalar2tensor_one(ori_shape)
        output_org_shape_list.append(ori_shape)
        output_org_format_list.append(ori_format)

        if ori_format not in support_ori_format or len(input_ori_shape) != len(input_ori_format) \
                or len(ori_format) != len(ori_shape):
            is_support_hd = False
            break

        # when split_d by N, H, W, support NDC1HWC0
        if ori_format[split_dim] != "C":
            break

        # when split_d by C, but output size not C0 align donot support NC1HWC0
        if ori_shape[split_dim] % c0_len != 0:
            is_support_hd = False
            break

    is_support_nz = False
    if input_ori_format[0] == "N" and split_dim == 0 and len(input_ori_shape) > 2:
        is_support_nz = True

    split_with_5hd_not_align = SplitWith5HD(input_value, output_data, split_dim, num_split, kernel_name)
    is_support_other_5hd = split_with_5hd_not_align.check_op_select() and not check_support_block_size_16()
    size_equal = len(set(size_splits))
    if size_equal != 1:
        is_support_other_5hd = False

    if output_org_format_list[0] in ("NCHW",) and input_ori_shape[0] == 1:
        is_support_other_5hd = False

    dtype_base = ["float16", "float", "int32", "int8", "int16", "int64", "uint8", "uint16", "uint32", "uint64", "bool"]
    dtype_5hd = ["float16", "float", "int32", "int8", "int16", "uint16", "uint32"]
    if tbe_platform.intrinsic_check_support("Intrinsic_vconv", "bf162f32"):
        dtype_base.append("bfloat16")
        dtype_5hd.append("bfloat16")

    dtype_base_out = dtype_base.copy()
    format_base_out = ["ND"] * len(dtype_base)

    if is_support_hd:
        other_format = "NC1HWC0" if len(input_ori_shape) == 4 else "NDC1HWC0"
        dtype_base_out = dtype_base_out + dtype_5hd
        format_base_out = format_base_out + [other_format] * len(dtype_5hd)

    if is_support_nz:
        dtype_base_out = dtype_base_out + dtype_base
        format_base_out = format_base_out + ["FRACTAL_NZ"] * len(dtype_base)

    if is_support_other_5hd:
        dtype_base_out = dtype_base_out + ["float16", "int16", "uint16"]
        format_base_out = format_base_out + ["NC1HWC0"] * 3

    dtype_str = ','.join(dtype_base_out)
    format_str = ','.join(format_base_out)

    input0 = gen_param(classify="input0", name="x", datatype=dtype_str, format=format_str)
    output0 = gen_param(classify="output0", name="y", datatype=dtype_str, format=format_str)
    param_list = [input0, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


# 'pylint: disable=too-many-arguments
def check_supported(input_value, output_data, size_splits, split_dim, num_split, kernel_name="split_v_d"):
    """
    Check whether input is supported
    """
    ori_shape = input_value.get("ori_shape")
    input_format = input_value.get("format")
    ori_format = input_value.get("ori_format")

    if 0 not in ori_shape:
        input_value = util_common.update_shape_base_other_format(input_value)

    split_dim = util_common.update_axis_for_other_format(ori_shape, split_dim, input_format, ori_format)
    if input_format == "NC1HWC0":
        split_with_5hd_not_align = SplitWith5HD(input_value, output_data,
                                                split_dim, num_split, kernel_name)
        if split_with_5hd_not_align.check_5hd_vnchw() and not check_support_block_size_16():
            return False, "the format is not supported by DSL now"

    return True, ""


# 'pylint: disable=unused-argument
@register_operator_compute("SplitVD", op_mode="dynamic", support_fusion=False)
def split_v_d_compute(input_tensors, y, input_size_splits, axis_, num_split, kernel_name="split_v_d"):
    """
    Split_v_d compute

    Parameters
    ----------
    input_tensors: dict
        the dict of input tensor.
    input_size_splits: dict
        the dict of input size_splits tensor.
        Specifies a list containing the sizes of each output tensor along the split dimension.
    axis_: dict
        the dict of input split_dim tensor.
        An int, specifies the dimension along which to split.
    y: list or tuple
        the list of output tensor.
    num_split: int
        an integer indicating the number of outputs.
    kernel_name: str
        cce kernel name, default value is "split_v".

    Returns
    -------
    res: TVM tensor
        the result of Split_v_d
    """
    res = tbe.split(input_tensors, axis_, input_size_splits)
    return res


@register_operator("SplitVD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.DYNAMIC_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
def split_v_d(input_value, output_data, size_splits, split_dim, num_split, kernel_name="split_v_d"):
    """Split a tensor into len(size_splits) tensors along one dimension.

    Parameters
    ----------
    input_value: dict
        the dict of input tensor.
    output_data: list or tuple
        the list of output tensor.
    size_splits: list or tuple
        a Python list containing the sizes of each output tensor
        along `split_dim`.
    split_dim: int
        the dimension along which to split_d.
    num_split: int
        used to specify the number of outputs.
    kernel_name: str
        cce kernel name, default value is "split_v_d".

    Returns
    -------
    None.
    """
    input_value = util_common.update_shape_base_other_format(input_value)

    input_dtype = input_value.get("dtype")
    input_shape = input_value.get("shape")
    input_format = input_value.get("format")
    input_ori_format = input_value.get("ori_format")
    input_ori_shape = input_value.get("ori_shape")
    size_splits = list(size_splits)

    size_splits_sum = 0
    for size_split in size_splits:
        if size_split != -1:
            size_splits_sum = size_splits_sum + size_split

    if size_splits_sum != input_ori_shape[split_dim]:
        for idx, size_split in enumerate(size_splits):
            if size_split == -1:
                size_splits[idx] = input_ori_shape[split_dim] - size_splits_sum

    if size_splits[-1] == 0:
        del(size_splits[-1])
        num_split = num_split - 1

    split_dim = util_common.update_axis_for_other_format(input_ori_shape, split_dim, input_format, input_ori_format)

    if (split_dim == 1 and input_format == "NC1HWC0") or (split_dim == 2 and input_format == "NDC1HWC0"):
        c0_size = input_shape[-1]
        size_splits = [size // c0_size for size in size_splits]

    if input_format == "FRACTAL_NZ" and input_ori_format != "FRACTAL_NZ" and split_dim >= len(input_shape) - 4:
        size_splits = [size // 16 for size in size_splits]

    dim = input_shape[split_dim]
    if len(size_splits) + 1 == num_split:
        split_list = []
        split_sum = 0
        for i, _ in enumerate(size_splits):
            split_list.append(size_splits[i])
            split_sum = split_sum + size_splits[i]
        if dim - split_sum > 0:
            split_list.append(dim - split_sum)
        size_splits = split_list
    
    size_sum = 0
    for size in size_splits:
        if size < 1:
            expected_value = "The size of size_splits must be greater or equal to 1"
            real_value = "less to 1"
            error_manager_vector.raise_err_input_value_invalid(kernel_name, "size_splits", expected_value, real_value)
        size_sum = size_sum + size
    if size_sum != input_shape[split_dim]:
        expected_value = "The sum size of size_splits must be equal to the length of split_dim"
        real_value = "The sum size is (%d) and the length of split_dim is (%d)"\
                     % (size_sum, input_shape[split_dim])
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "size_splits", expected_value, real_value)
    if len(size_splits) != num_split:
        expected_value = "The length of size_splits must be equal to num_split"
        real_value = "The length of size_splits is (%d) and the num_split is (%d)" \
                     % (len(size_splits), num_split)
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "size_splits", expected_value, real_value)

    input_size_splits = {"shape": (num_split, ), "const_value": size_splits, 
                         "range": [num_split, num_split], "dtype": "int32",
                         "ori_range": (num_split, ), "format": "ND", "ori_format": "ND"}
    input_split_dim = {"shape": (1, ), "const_value": (split_dim, ), "range": [1, 1],
                       "dtype": "int32", "ori_range": (1, ), "format": "ND", "ori_format": "ND"}
    extra_params = {"avg_split": False, "num_split":num_split}
    ins = classify([input_value, input_split_dim, input_size_splits], "split", extra_params)
    schedules, tensors = [], []
    for input_x_, axis_, size_splits_ in ins:
        with tbe.compute():
            shape_x, input_size_splits = shape_util.variable_shape([input_x_, size_splits_], "split")
            input_tensors = tvm.placeholder(shape_x, dtype=input_dtype, name="data_x")

            res = split_v_d_compute(input_tensors, output_data, input_size_splits, axis_, num_split, kernel_name)

            tensors.append([input_tensors, *res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name":kernel_name, "tensor_list":tensors}
    tbe.build(schedules, config)
