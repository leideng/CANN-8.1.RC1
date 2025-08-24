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

split_d
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
from impl.util import util_common
from impl.util import util_select_op_base
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl import split_last_dim


# 'pylint: disable=unused-argument,invalid-name,no-self-use
def op_select_format(input_value, output_data, split_dim, num_split, kernel_name="split_d"):
    """
    1.when input input_value's ori_shape in ["NCHW", "NHWC"] and split_d by
    dim N, H, W and dim C of input_value's ori_shape can be divisible by 16(32
    when dtype is int8). the Op SplitD can support ND and NC1HWC0.

        for example:
        input_value : Tensor of (shape=(2, 16, 32), "ND")
        the Op Select can process with NC1HWC0:
        input_value : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")

    2.when input input_value's ori_shape dimension is greater then 2 and
    do not split with last 2 dim. the Op SplitD can support ND and FRACTAL_NZ.

        for example:
        input_value : Tensor of (shape=(16, 16, 16, 16), "NCHW")
        the Op Select can process with NC1HWC0:
        input_value : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")
    """
    dtype = input_value.get("dtype").lower()
    if dtype == "int8":
        c0_len = 32 if not check_support_block_size_16() else 16
    else:
        c0_len = 16 if not check_support_block_size_16() else 8
    output_org_shape_list = []
    output_org_format_list = []
    is_support_hd = False
    is_support_nz = False
    is_support_other_5hd = False
    support_ori_format =  util_common.get_fused_format_str(["N", "D", "H", "W", "C"]) \
                          + util_common.get_fused_format_str(["N", "H", "W", "C"])
    input_ori_shape = input_value.get("ori_shape")
    input_ori_format = input_value.get("ori_format")
    if split_dim is not None:
        is_support_hd = True
        split_dim = split_dim % len(input_ori_shape)

        for _, output_dict in enumerate(output_data):
            ori_format = output_dict.get("ori_format").upper()
            ori_shape = output_dict.get("ori_shape")
            output_org_shape_list.append(ori_shape)
            output_org_format_list.append(ori_format)

            if ori_format not in support_ori_format or len(input_ori_shape) != len(input_ori_format) \
                    or len(ori_format) != len(ori_shape):
                is_support_hd = False
                break

            # when split_d by N,H,W, support NC1HWC0
            if ori_format[split_dim] != "C":
                break

            # when split_d by C, but output size not C0 align donot support NC1HWC0
            if ori_shape[split_dim] % c0_len != 0:
                is_support_hd = False
                break

        is_support_nz = False
        if len(input_ori_shape) > 2:
            # if do not split with last two dim, will support nz
            if split_dim < len(input_ori_shape) - 2:
                is_support_nz = True

        split_with_5hd_not_align = \
            split_last_dim.SplitWith5HD(input_value, output_data,
                                        split_dim, num_split, kernel_name)
        is_support_other_5hd = split_with_5hd_not_align.check_op_select() and not check_support_block_size_16()

    dtype_base = ["float16", "float", "int32", "int8", "int16", "int64", "uint8",
                  "uint16", "uint32", "uint64", "bool"]
    dtype_5hd = ["float16", "float", "int32", "int8", "int16", "uint16", "uint32"]
    if tbe_platform.intrinsic_check_support("Intrinsic_vconv", "bf162f32"):
        dtype_base.append("bfloat16")
        dtype_5hd.append("bfloat16")

    dtype_base_out = dtype_base.copy()
    format_base_out = ["ND"] * len(dtype_base)

    if is_support_hd and not util_common.is_dynamic_input([input_value]):
        other_format = "NC1HWC0" if len(input_ori_shape) == 4 else "NDC1HWC0"
        dtype_base_out = dtype_base_out + dtype_5hd
        format_base_out = format_base_out + [other_format] * len(dtype_5hd)

    if is_support_nz and not util_common.is_dynamic_input([input_value]):
        dtype_base_out = dtype_base_out + dtype_base
        format_base_out = format_base_out + ["FRACTAL_NZ"] * len(dtype_base)

    if is_support_other_5hd and not util_common.is_dynamic_input([input_value]):
        dtype_base_out = dtype_base_out + ["float16", "int16", "uint16"]
        format_base_out = format_base_out + ["NC1HWC0"] * 3

    dtype_str = ','.join(dtype_base_out)
    format_str = ','.join(format_base_out)

    is_dynamic_input = _check_dynamic_input(input_value)

    unknownshape_format_str = ','.join(len(format_base_out) * ['ND']) if is_dynamic_input else format_str
    
    input0 = util_select_op_base.gen_param(classify="input0", name="x", datatype=dtype_str,
                                           format=format_str, unknownshape_format=unknownshape_format_str)
    output0 = util_select_op_base.gen_param(classify="output0", name="y", datatype=dtype_str,
                                            format=format_str, unknownshape_format=unknownshape_format_str)
    param_list = [input0, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def _check_dynamic_input(input_values):
    shape_input = input_values.get("ori_shape")
    if -1 in shape_input or -2 in shape_input:
        return True
    return False


def check_supported(x, y, split_dim, num_split, kernel_name="split_d"):
    """
    Check whether input is supported
    """
    is_dynamic_input = _check_dynamic_input(x)
    if is_dynamic_input:
        return True, ""
    ori_shape = x.get("ori_shape")
    input_format = x.get("format")
    ori_format = x.get("ori_format")

    if 0 not in ori_shape:
        x = util_common.update_shape_base_other_format(x)

    split_dim = util_common.update_axis_for_other_format(ori_shape, split_dim, input_format, ori_format)

    split_with_5hd_not_align = split_last_dim.SplitWith5HD(x, y, split_dim, num_split, kernel_name)
    if split_with_5hd_not_align.check_5hd_vnchw() and not check_support_block_size_16():
        return False, "the format is not supported by DSL now"

    return True, ""


@register_operator_compute("SplitD", op_mode="dynamic", support_fusion=False)
def split_d_compute(input_tensors, size_splits, axis_, num_split, kernel_name):
    """
    Split_d compute

    Parameters
    ----------
    input_tensors: dict
        the dict of input tensor.
    size_splits: dict
        the dict of input size_splits tensor.
        Specifies a list containing the sizes of each output tensor along the split dimension.
    axis_: dict
        the dict of input split_dim tensor.
        An int, specifies the dimension along which to split.
    num_split: int
        an integer indicating the number of outputs.
    kernel_name: str
        cce kernel name, default value is "split_v".

    Returns
    -------
    res: TVM tensor
        the result of Split_d
    """
    res = tbe.split(input_tensors, axis_, size_splits)
    return res


@register_operator("SplitD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.DYNAMIC_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def split_d(x, y, split_dim, num_split, kernel_name="split_d"):
    """
    Split a tensor into `num_split` tensors along one dimension.

    Parameters
    ----------
    x: dict
        the dict of input tensor.
    y: list or tuple
        the list of output tensor.
    split_dim: int
        the dimension along which to split_d.
    num_split: int
        an integer indicating the number of split_d along `split_dim`.
    kernel_name: str
        cce kernel name, default value is "split_d".

    Returns
    -------
    compile info
    """
    x = util_common.update_shape_base_other_format(x)
    dtype_x = x.get("dtype").lower()
    input_format = x.get("format")
    ori_format = x.get("ori_format")
    ori_shape = x.get("ori_shape")
    if split_dim is None:
        tbe_context.get_context().add_compile_info("split_dim_idx", 0)
    dtype_list = ("float16", "float32", "int32", "int8", "int16", "int64", "uint8", "uint16",
                  "uint32", "uint64", "bfloat16")
    para_check.check_dtype(dtype_x, dtype_list, param_name="x")

    split_dim = util_common.update_axis_for_other_format(ori_shape, split_dim,
                                                         input_format, ori_format)

    if num_split is None:
        num_split = len(y)

    extra_params = {"avg_split": True, "num_split":num_split}

    schedules, tensors = [], []
    ins = classify([x, split_dim], "split", extra_params)

    for input_x_, axis_, size_splits_ in ins:
        with tbe.compute():
            shape_x, size_splits = shape_util.variable_shape([input_x_, size_splits_], "split")
            input_tensors = tvm.placeholder(shape_x, dtype=dtype_x, name="data_x")

            res = split_d_compute(input_tensors, size_splits, axis_, num_split, kernel_name)

            tensors.append([input_tensors, *res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name":kernel_name, "tensor_list":tensors}
    tbe.build(schedules, config)
