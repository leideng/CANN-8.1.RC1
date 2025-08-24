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

strided slice
"""
from __future__ import absolute_import
from impl.util.platform_adapter import error_manager_vector
from impl.constant_util import C0_SIZE
from impl.util import util_select_op_base
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tik
from impl.util.util_common import ceil
from impl.util.util_common import get_fused_str
from impl.util.util_common import is_support_fractal_z_input
from impl.util.util_common import is_unknown_rank_input
from impl.util.util_common import update_shape_base_other_format
from impl.util import util_common
from .strided_slice import StridedSlice


def check_support_hd_and_fz(x, offsets, size):
    """
    check whether support 5HD 6HD and FRACTAL_Z/FRACTAL_Z_3D
    """
    hd_support_format = get_fused_str(["N", "C", "H", "W"]) + get_fused_str(["N", "D", "C", "H", "W"])
    hd_format_c0 = 16
    fz_format_n0 = 16

    is_support_hd = False
    is_support_fz = False

    input_ori_shape = x.get("ori_shape")
    input_ori_format = x.get("ori_format")

    if len(input_ori_format) == len(input_ori_shape) and input_ori_format in hd_support_format:
        dict_zip_begin = dict(zip(list(input_ori_format), offsets))
        dict_zip_size = dict(zip(list(input_ori_format), size))
        dict_zip_shape = dict(zip(list(input_ori_format), input_ori_shape))
        begin_c_align_flag = dict_zip_begin["C"] % hd_format_c0 == 0
        begin_n_align_flag = dict_zip_begin["N"] % fz_format_n0 == 0

        is_size_c_support = \
            dict_zip_size["C"] % hd_format_c0 == 0 or dict_zip_shape["C"] == dict_zip_size["C"] + dict_zip_begin["C"]
        # charge whether support 5HD 6HD
        # `info: condition:
        # one: C dim in start is c0 align
        # two: C dim in size is c0 align or size_c = shape_c - start_c(means will slice all remain data from start_c)
        # three: C Dim in start is 0, C Dim in size == C Dim in shape(means will slice without c)
        if begin_c_align_flag and is_size_c_support:
            is_support_hd = True

        is_slice_without_c = dict_zip_begin["C"] == 0 and (dict_zip_size["C"] == -1 or
                                                           dict_zip_size["C"] == dict_zip_shape["C"])
        is_support_hd = is_support_hd or is_slice_without_c

        is_size_n_support = \
            dict_zip_size["N"] % fz_format_n0 == 0 or dict_zip_shape["N"] == dict_zip_size["N"] + dict_zip_begin["N"]
        # charge whether support FRACTAL_Z，FRACTAL_Z_3D
        # `info: condition:
        # one: both N and C dim in start is c0 align
        # two: C dim in size is c0 align or size_c = shape_c - start_c
        # three: N dim in size is n0 align or size_n = shape_n - start_n
        if begin_c_align_flag and begin_n_align_flag and is_support_fractal_z_input(x) and \
                is_size_c_support and is_size_n_support:
            is_support_fz = True

    return is_support_hd, is_support_fz


def check_support_nz(ori_shape, offsets, size):
    """
    check whether support FRACTAL_NZ
    """
    is_support_nz = False
    nz_format_align = 16
    if len(ori_shape) >= 2:
        is_first_last_size_support = size[-1] % nz_format_align == 0 or ori_shape[-1] == size[-1] + offsets[-1]
        is_second_last_size_support = size[-2] % nz_format_align == 0 or ori_shape[-2] == size[-2] + offsets[-2]
        # `info: condition:`
        # `one: len >= 2;`
        # `two: the value begin[-1] and begin[-2] is align`
        # `three: -1 dim in size is align or size = shape - start`
        # `four: -2 dim in size is align or size = shape - start`
        if offsets[-1] % nz_format_align == 0 and offsets[-2] % nz_format_align == 0 \
                and is_first_last_size_support and is_second_last_size_support:
            is_support_nz = True
    return is_support_nz


def get_dtype_and_format(x, offsets, size, dtype_x_out, format_x_out):
    """
    get_dtype_and_format
    """
    input_ori_shape = x.get("ori_shape")
    # update size the size = -1
    size = list(size)
    if not (len(input_ori_shape) == len(offsets) and len(input_ori_shape) == len(size)):
        expected_value = "must be equal to shape!"
        real_value = "not equal to shape!"
        error_manager_vector.raise_err_input_value_invalid("slice", "length of offsets and size", expected_value,
                                                           real_value)
    for i, item in enumerate(size):
        if item == -1 and input_ori_shape[i] >= 0:
            size[i] = input_ori_shape[i] - offsets[i]

    is_support_hd, is_support_fz = check_support_hd_and_fz(x, offsets, size)
    is_support_nz = check_support_nz(input_ori_shape, offsets, size)

    other_x_type = ["float", "float16", "int16", "int32", "uint16", "uint32"]
    bfp16_support = tbe_platform.intrinsic_check_support("Intrinsic_vconv", "bf162f32")
    if bfp16_support:
        other_x_type.append("bfloat16")

    if is_support_hd:
        other_format = "NC1HWC0" if len(input_ori_shape) == 4 else "NDC1HWC0"
        dtype_x_out = dtype_x_out + other_x_type
        format_x_out = format_x_out + [other_format] * len(other_x_type)
    if is_support_fz:
        other_format = "FRACTAL_Z" if len(input_ori_shape) == 4 else "FRACTAL_Z_3D"
        dtype_x_out = dtype_x_out + other_x_type
        format_x_out = format_x_out + [other_format] * len(other_x_type)
    if is_support_nz:
        other_format = "FRACTAL_NZ"
        dtype_x_out = dtype_x_out + other_x_type
        format_x_out = format_x_out + [other_format] * len(other_x_type)

    return dtype_x_out, format_x_out


# 'pylint: disable=unused-argument,invalid-name
def op_select_format(x, offsets, size, y, kernel_name="slice"):
    """
    define the op_select_format for Slice Op

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

    offsets_value = offsets.get("const_value")
    size_value = size.get("const_value")
    if offsets_value and size_value and not is_unknown_rank_input(x):
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


# 'pylint: disable=unused-argument,invalid-name
def slice_compute(x, offsets, size, y, kernel_name="slice"):
    """
    slice compute

    Parameters
    ----------
    x: input params shape, dtype and range
    offsets: input offsets shape, dtype and range
    size: input size shape, dtype and range
    y: output shape, dtype and range
    kernel_name: kernel name of slice op

    Returns
    -------
    res: TVM tensor
        the result of gather
    """
    res = tbe.slice(x, offsets, size)
    return res


# 'pylint: disable=too-many-locals
def update_params_for_other_format(shape, begin, size, input_format, ori_format):
    """
    update begin, size base on  ori_format
    """
    # modify size base size value if value = -1 size = shape - begin
    size_new = []
    for i, item in enumerate(size):
        if item != -1 or shape[i] <= 0:
            size_new.append(item)
        else:
            size_new.append(shape[i] - begin[i])
    size = size_new

    if input_format in ["NDC1HWC0", "NC1HWC0", "FRACTAL_Z", "FRACTAL_Z_3D"]:
        # when NDC1HWC0 or NC1HWC0 will update the C1 and C0 for begin and size
        # ex: begin [N, D, C, H, W] -> [N, D, C // 16, H, W, 0]
        #     size  [N, D, C, H, W] -> [N, D, (C + 15) // 16, H, W, -1]
        # when FRACTAL_Z or FRACTAL_Z_3D will update the C1 and C0 and N1 and N0
        # ex: begin [N, D, C, H, W] -> [D, C // 16, H, W, N // 16, 0, 0]
        #     size  [N, D, C, H, W] -> [D, (C + 15) // 16, H, W, (N + 15) // 16, 0, 0]
        begin_nchw = [
            begin[ori_format.index("N")], begin[ori_format.index("C")], begin[ori_format.index("H")],
            begin[ori_format.index("W")]
        ]
        size_nchw = [
            size[ori_format.index("N")], size[ori_format.index("C")], size[ori_format.index("H")],
            size[ori_format.index("W")]
        ]
        begin_c1 = begin_nchw[1] // C0_SIZE
        begin_c0 = 0
        begin_n1 = begin_nchw[0] // C0_SIZE
        begin_n0 = 0
        size_c1 = -1 if size_nchw[1] == -1 else ceil(size_nchw[1], C0_SIZE)
        size_c0 = -1
        size_n1 = -1 if size_nchw[0] == -1 else ceil(size_nchw[0], C0_SIZE)
        size_n0 = -1

        if input_format == "NDC1HWC0":
            begin_new = [begin_nchw[0], begin[ori_format.index("D")], begin_c1, begin_nchw[2], begin_nchw[3], begin_c0]
            size_new = [size_nchw[0], size[ori_format.index("D")], size_c1, size_nchw[2], size_nchw[3], size_c0]
        elif input_format == "NC1HWC0":
            begin_new = [begin_nchw[0], begin_c1, begin_nchw[2], begin_nchw[3], begin_c0]
            size_new = [size_nchw[0], size_c1, size_nchw[2], size_nchw[3], size_c0]
        elif input_format == "FRACTAL_Z_3D":
            begin_new = [
                begin[ori_format.index("D")], begin_c1, begin_nchw[2], begin_nchw[3], begin_n1, begin_n0, begin_c0
            ]
            size_new = [size[ori_format.index("D")], size_c1, size_nchw[2], size_nchw[3], size_n1, size_n0, size_c0]
        else:
            begin_new = [begin_c1, begin_nchw[2], begin_nchw[3], begin_n1, begin_n0, begin_c0]
            size_new = [size_c1, size_nchw[2], size_nchw[3], size_n1, size_n0, size_c0]

        return begin_new, size_new

    if input_format in ["FRACTAL_NZ"]:
        # when FRACTAL_NZ will update last two dim
        # ex: begin [A, B, C, D] -> [A, B, D // 16,  C // 16, 0 , 0]
        #     size  [A, B, C, D] -> [A, B, (D + 15) // 16,  (C + 15) // 16, -1 , -1]
        begin_fisrt_last_dim_one = begin[-1] // C0_SIZE
        begin_fisrt_last_dim_two = 0

        begin_second_last_dim_one = begin[-2] // C0_SIZE
        begin_second_last_dim_two = 0

        size_fisrt_last_dim_one = -1 if size[-1] == -1 else ceil(size[-1], C0_SIZE)
        size_fisrt_last_dim_two = -1

        size_second_last_dim_one = -1 if size[-2] == -1 else ceil(size[-2], C0_SIZE)
        size_second_last_dim_two = -1

        begin_new = list(begin[0:-2]) + [
            begin_fisrt_last_dim_one, begin_second_last_dim_one, begin_second_last_dim_two, begin_fisrt_last_dim_two
        ]
        size_new = size[0:-2] + [
            size_fisrt_last_dim_one, size_second_last_dim_one, size_second_last_dim_two, size_fisrt_last_dim_two
        ]

        return begin_new, size_new

    return None, None


def update_input_params(x, offsets, size):
    """
    update input params in known shape
    """
    input_format = x.get("format")

    offsets_value = offsets.get("const_value")
    size_value = size.get("const_value")

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
def slice_dsl(x, offsets, size, y, kernel_name="slice"):
    """
    slice interface for dsl
    """
    update_input_params(x, offsets, size)

    x_dtype = x.get("dtype").lower()
    x_dtype = "int8" if x_dtype == "bool" else x_dtype
    offsets_dtype = offsets.get("dtype").lower()
    size_dtype = size.get("dtype").lower()
    is_unknown = util_common.is_unknown([x, offsets, size, y])

    ins = classify([x, offsets, size], "slice", {"end_mode": "size"})
    schedules, tensors = [], []
    
    
    for shape_x, shape_offsets, shape_size in ins:
        with tbe.compute():
            if not is_unknown and "const_value" not in offsets:
                offsets_tensor = tvm.placeholder([len(shape_x["shape"])], name="offsets", dtype=offsets_dtype)
                shape_offsets["tensor"] = offsets_tensor
                x_var, offsets_list, size_list = \
                shape_util.variable_shape([shape_x, shape_offsets, shape_size], "slice")
                x_tensor = tvm.placeholder(x_var, name="x", dtype=x_dtype)
                size_tensor = tvm.placeholder([len(size_list)], name="size", dtype=size_dtype)
                res = slice_compute(x_tensor, offsets_list, size_list, y, kernel_name)
                tensors.append([x_tensor, offsets_tensor, size_tensor, res])
            else:
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


# 'pylint: disable=unused-argument,invalid-name
def slice_tik(x, offsets, size, y, kernel_name="slice"):
    """
    slice interface for tik
    """
    strided_slice_instance = StridedSlice(x, None, 0, 0, 0, 0, 0, kernel_name)
    strided_slice_instance.strided_slice()
    inst = strided_slice_instance.tik_instance
    opt_config = strided_slice_instance.get_opt_config()
    tbe_context.get_context().add_compile_info("vars", strided_slice_instance.get_vars_info())
    # It is used to distinguish between Tik implementation and DSL implementation in the tilling phase
    tbe_context.get_context().add_compile_info("is_tik", True)
    inst.BuildCCE(kernel_name=strided_slice_instance.kernel_name,
                  inputs=(strided_slice_instance.input_gm, strided_slice_instance.begin_gm,
                          strided_slice_instance.end_gm),
                  outputs=(strided_slice_instance.output_gm,),
                  flowtable=[strided_slice_instance.tiling_param.tiling_gm],
                  config=opt_config,
                  enable_l2=False)

    return inst


# 'pylint: disable=locally-disabled,too-many-arguments,invalid-name,unused-argument
# 'pylint: disable=unused-argument,too-many-locals,redefined-builtin
@register_operator("Slice")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def slice(x, offsets, size, y, kernel_name="slice"):
    """
    algorithm: slice
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
    kernel_name: str
        cce kernel name, default value is "slice".

    Returns
    -------
    tik instance
    """
    # dynamic slice does not use offsets, end params.
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

    if tbe_platform.api_check_support("tbe.dsl.slice", "float32"):
        slice_dsl(x, offsets, size, y, kernel_name)
    else:
        slice_tik(x, offsets, size, y, kernel_name)
