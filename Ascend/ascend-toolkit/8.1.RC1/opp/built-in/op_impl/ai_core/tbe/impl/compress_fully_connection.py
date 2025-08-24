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
compress_fully_connection
"""
from __future__ import absolute_import
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import gemm
from impl.util.platform_adapter import get_matmul_performance_format
from impl.util.platform_adapter import auto_schedule
from impl.util.platform_adapter import error_manager_cube as err_man_cube
from impl.util.util_gemm import get_op_support_info_fc
from impl.util.util_gemm import get_prebuild_pattern
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json


def get_format(x, w, b, offset_w, y, num_output, transpose, axis, offset_x, format_x_ori):
    """
    get format from get_matmul_performance_format function
    """
    dtype_x = x.get('dtype')
    shape_x_ori = x.get("ori_shape")
    length_x_ori = len(shape_x_ori)
    if dtype_x == 'float32':
        dtype_x = 'float16'

    k_shape = 1
    for i in range(1, length_x_ori):
        k_shape *= shape_x_ori[i]

    shape_x = (shape_x_ori[0], k_shape)
    tensor_x = tvm.placeholder(shape_x, dtype=dtype_x, name='tensor_a')

    shape_w_ori = w.get('shape')
    dtype_w = w.get('dtype')
    if dtype_w == 'float32':
        dtype_w = 'float16'
    if not transpose:
        kw_shape = shape_w_ori[1] * shape_w_ori[2] * shape_w_ori[3]
        n_shape = shape_w_ori[0]
        shape_w = (kw_shape, shape_w_ori[0])
    else:
        kw_shape = shape_w_ori[0] * shape_w_ori[1] * shape_w_ori[2]
        n_shape = shape_w_ori[3]
        shape_w = (shape_w_ori[3], kw_shape)

    tensor_w = tvm.placeholder(shape_w, dtype=dtype_w, name='tensor_b')

    if b is not None:
        dtype_b = b.get('dtype')
        if dtype_b  == 'float32':
            dtype_b = 'float16'
        shape_bias = (n_shape,)
        tensor_b = tvm.placeholder(shape_bias, dtype=dtype_b, name='tensor_bias')
    else:
        tensor_b = None

    format_x = get_matmul_performance_format(tensor_x, tensor_w,
                            False, transpose, "ND", "FRACTAL_Z",
                            1.0, 1.0, 'float16', tensor_b, None)

    if format_x_ori in ("NCHW", "NHWC"):
        if format_x == "FRACTAL_NZ":
            format_x = "FRACTAL_Z"

    format_list = format_x+","+format_x
    return format_list


def get_op_specific_info(x, w, compress_index, b, offset_w, y, num_output, transpose, axis, offset_x,
                         kernel_name="fully_connection"):
    """
    get the fullyconnection prebuild pattern

    """
    return get_prebuild_pattern(x, op_pattern="Matmul")


def op_select_format(x, w, compress_index, b, offset_w, y, num_output, transpose, axis, offset_x,
                     kernel_name="fully_connection"):
    """
    select format dynamically
    """
    shape_x_ori = x.get("ori_shape")
    length_x_ori = len(shape_x_ori)
    format_x_ori = x.get("format")

    shape_y_ori = y.get("ori_shape")
    length_y_ori = len(shape_y_ori)

    if axis == 1:
        if shape_x_ori[0] == 1:
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16, int8",
                               format="NC1HWC0, NC1HWC0")
            input1 = gen_param(classify="input1", name="w",
                               datatype="float16, int8",
                               format="FRACTAL_Z, FRACTAL_Z")
            input2 = gen_param(classify="input2", name="compress_index",
                               datatype="int8, int8",
                               format="ND, ND")
            input3 = gen_param(classify="input3", name="b",
                               datatype="float16, int32",
                               format="NC1HWC0, NC1HWC0")
            input4 = gen_param(classify="input4", name="offset_w",
                               datatype="int8, int8",
                               format="ND, ND")
            output0 = gen_param(classify="output0", name="y",
                                datatype="float16,int32",
                                format="NC1HWC0, NC1HWC0")
        elif length_y_ori == 4:
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16, int8",
                               format="NC1HWC0, NC1HWC0")
            input1 = gen_param(classify="input1", name="w",
                               datatype="float16, int8",
                               format="FRACTAL_Z, FRACTAL_Z")
            input2 = gen_param(classify="input2", name="compress_index",
                               datatype="int8, int8",
                               format="ND, ND")
            input3 = gen_param(classify="input3", name="b",
                               datatype="float16,int32",
                               format="NC1HWC0, NC1HWC0")
            input4 = gen_param(classify="input4", name="offset_w",
                               datatype="int8, int8",
                               format="ND, ND")
            output0 = gen_param(classify="output0", name="y",
                                datatype="float16, int32",
                                format="NC1HWC0, NC1HWC0")
        elif length_x_ori == 2:
            input0 = gen_param(classify="input0", name="x",
                               datatype="float16, int8",
                               format="FRACTAL_NZ, FRACTAL_NZ")
            input1 = gen_param(classify="input1", name="w",
                               datatype="float16, int8",
                               format="FRACTAL_Z, FRACTAL_Z")
            input2 = gen_param(classify="input2", name="compress_index",
                               datatype="int8, int8",
                               format="ND, ND")
            input3 = gen_param(classify="input3", name="b",
                               datatype="float16, int32",
                               format="NC1HWC0, NC1HWC0")
            input4 = gen_param(classify="input4", name="offset_w",
                               datatype="int8, int8",
                               format="ND, ND")
            output0 = gen_param(classify="output0", name="y",
                                datatype="float16, int32",
                                format="FRACTAL_NZ, FRACTAL_NZ")
        else:
            format_x = get_format(x, w, b, offset_w, y, num_output, transpose, axis, offset_x, format_x_ori)

            input0 = gen_param(classify="input0", name="x",
                               datatype="float16, int8",
                               format=format_x)
            input1 = gen_param(classify="input1", name="w",
                               datatype="float16, int8",
                               format="FRACTAL_Z, FRACTAL_Z")
            input2 = gen_param(classify="input2", name="compress_index",
                               datatype="int8, int8",
                               format="ND,ND")
            input3 = gen_param(classify="input3", name="b",
                               datatype="float16,int32",
                               format="NC1HWC0,NC1HWC0")
            input4 = gen_param(classify="input4", name="offset_w",
                               datatype="int8,int8",
                               format="ND,ND")
            output0 = gen_param(classify="output0", name="y",
                                datatype="float16,int32",
                                format="FRACTAL_NZ, FRACTAL_NZ")
    else:
        input0 = gen_param(classify="input0", name="x",
                           datatype="float16, int8",
                           format="FRACTAL_NZ, FRACTAL_NZ")
        input1 = gen_param(classify="input1", name="w",
                           datatype="float16, int8",
                           format="FRACTAL_Z, FRACTAL_Z")
        input2 = gen_param(classify="input2", name="compress_index",
                           datatype="int8, int8",
                           format="ND,ND")
        input3 = gen_param(classify="input3", name="b",
                           datatype="float16,int32",
                           format="NC1HWC0,NC1HWC0")
        input4 = gen_param(classify="input4", name="offset_w",
                           datatype="int8,int8",
                           format="ND,ND")
        output0 = gen_param(classify="output0", name="y",
                            datatype="float16,int32",
                            format="FRACTAL_NZ, FRACTAL_NZ")

    param_list = [input0, input1, input2, input3, input4, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def fully_connection_check_rule(x, w, compress_index, b, offset_w, y,
                                num_output, transpose, axis, offset_x,
                                kernel_name="compress_fully_connection"):
    """check input params"""
    # x info
    shape_x = x.get('shape')
    dtype_x = x.get('dtype')
    format_x = x.get('format')

    if axis == 2:
        km_shape = shape_x[1] * shape_x[-1]
    else:
        if format_x == "NC1HWC0":
            km_shape = shape_x[1] * shape_x[2] * shape_x[3] * shape_x[4]
        else:
            km_shape = shape_x[0] * shape_x[-1]


    if shape_x[-1] not in (16, 32):
        err_man_cube.raise_err_specific("compress_fully_connection",
                                        "for axis = 1: C0 must be 16 when non-quant condition!, \
                                        C0 must be 32 when quant condition! \
                                        for axis = 2: the last dim must be 16!")

    para_check.check_dtype(dtype_x, ['float16', 'int8'], param_name="x")
    para_check.check_format(format_x, ('NC1HWC0', 'FRACTAL_NZ', 'FRACTAL_Z'), param_name="x")

    # w info
    shape_w = w.get('shape')
    dtype_w = w.get('dtype')
    format_w = w.get('format')

    para_check.check_dtype(dtype_w, ['float16', 'int8'], param_name="w")

    para_check.check_format(format_w, ["FRACTAL_Z"], param_name="w")
    # format shape info
    if dtype_x == 'float16' and (shape_w[2] != 16 or shape_w[3] != 16):
        err_man_cube.raise_err_three_paras("E62305",
                                           "compress_fully_connection",
                                           "w last two dims for no quant",
                                           "16", "shape_w={}".format(shape_w))
    if dtype_x == 'int8' and (shape_w[2] != 16 or shape_w[3] != 32):
        err_man_cube.raise_err_three_paras("E62305",
                                           "compress_fully_connection",
                                           "w last two dims for quant",
                                           "16 or 32", "shape_w={}".format(shape_w))

    kn_shape = shape_w[0] * shape_w[3]
    n_shape = shape_w[1] * shape_w[2]

    # Check shape
    if km_shape != kn_shape:
        err_man_cube.raise_err_three_paras("E62305",
                                           "compress_fully_connection",
                                           "km_shape and kn_shape", "equal",
                                           "km_shape={}, kn_shape={}".format(km_shape, kn_shape))

    # b info
    if b is not None:
        shape_b = b.get('shape')
        dtype_b = b.get('dtype')
        format_b = b.get('format')
        b_size = shape_b[1] * shape_b[4]

        # Check info
        para_check.check_dtype(dtype_b, ['float16', 'int32'], param_name="b")
        para_check.check_format(format_b, ('NC1HWC0'), param_name="b")

    # axis info
    if axis not in (1, 2):
        err_man_cube.raise_err_one_para("E62006", "compress_fully_connection",
                                        "axis only support 1, 2 when reduce from channel!")


def compress_fully_connection_compute(
        x, w, compress_index, b, offset_w, y, num_output, transpose, axis, offset_x,
        kernel_name="compress_fully_connection"):
    """
    x : the tensor of input x

    w: the tensor of intput w

    b: the tensor of bias

    offset_w : unused

    num_output : output neuron number

    transpose: is weight transpose

    axis: the beginning reduce axis, reduce axis is [axis, last_axis]

    offset_x: unused

    Returns:
    -------
    None
    """
    format_x = x.op.attrs["format"]
    format_out = None
    if axis == 2:
        format_a = 'FRACTAL_NZ'
        trans_a = True
    else:
        if format_x == 'NC1HWC0':
            format_a = 'ND'
            trans_a = False
            if x.shape[0].value == 1:
                format_out = 'FRACTAL_NZ'
        else:
            format_a = 'FRACTAL_NZ'
            trans_a = True
    format_b = "FRACTAL_Z"
    trans_b = False
    out_type = y.get('dtype')
    out_format = y.get('format')
    quantize_params = None

    if format_out is not None:
        out_format = format_out
    if offset_w is not None:
        err_man_cube.raise_err_one_para("E62006", "compress_fully_connection",
                                        "tensor offset_w must be None!")

    para_dict = {
            "trans_a": trans_a,
            "trans_b": trans_b,
            "format_a": format_a,
            "format_b": format_b,
            "dst_dtype": out_type,
            "tensor_c": b,
            "quantize_params": quantize_params,
            "format_out": out_format,
            "compress_index": compress_index,
            "offset_a": offset_x,
            "offset_b": offset_w,
            "kernel_name": kernel_name
        }
    result = gemm(tensor_a=x, tensor_b=w, para_dict=para_dict)

    return result


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.OPTION_ATTR_BOOL, para_check.REQUIRED_ATTR_INT,
                            para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
def get_op_support_info(x, w, compress_index, b, offset_w, y, num_output, transpose, axis, offset_x=0,
                        kernel_name="fully_connection"):
    """
    get the compress_fully_connection split

    """
    inputs = [x, w, b]
    return get_op_support_info_fc(inputs, y, axis, op_type="compress_fully_connection")


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.OPTION_ATTR_BOOL, para_check.REQUIRED_ATTR_INT,
                            para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
def compress_fully_connection(x, w, compress_index, b, offset_w, y,
                              num_output, transpose, axis, offset_x,
                              kernel_name="compress_fully_connection"):
    """
    x : the dict of input x

    w: the dict of intput w

    compress_index: the dict of intput compress index

    b: the dict of bias

    offset_w : unused

    num_output : output neuron number

    transpose: is weight transpose

    axis: the beginning reduce axis, reduce axis is [axis, last_axis]

    offset_x: unused

    Returns:
    -------
    None
    """
    # Check params
    fully_connection_check_rule(x, w, compress_index, b, offset_w, y,
                                num_output, transpose, axis, offset_x,
                                kernel_name="compress_fully_connection")

    # x info
    shape_x = x.get('shape')
    dtype_x = x.get('dtype')
    format_x = x.get('format')

    if axis == 2:
        shape_x_final = (shape_x[0], shape_x[1], shape_x[2], shape_x[3], shape_x[4])
    else:
        if format_x == "NC1HWC0":
            shape_x_final = (shape_x[0], shape_x[1] * shape_x[2] * shape_x[3] * shape_x[4])
        else:
            shape_x_final = (shape_x[0], shape_x[1], shape_x[2], shape_x[3])

    # set tensor attrs
    addr_type = x.get("addr_type", 0)
    valid_shape = x.get("valid_shape", [])
    slice_offset = x.get("slice_offset", [])
    l1_addr_flag = x.get("L1_addr_flag", -1)
    L1_addr_offset = x.get("L1_addr_offset", -1)
    L1_valid_size = x.get("L1_valid_size", -1)
    l1_fusion_type = x.get("L1_fusion_type", -1)
    attr = {"addr_type": addr_type,
            "valid_shape": valid_shape,
            "slice_offset": slice_offset,
            "L1_addr_flag": l1_addr_flag,
            "L1_addr_offset": L1_addr_offset,
            "L1_valid_size": L1_valid_size,
            "L1_fusion_type": l1_fusion_type,
            "format": format_x}

    tensor_x = tvm.placeholder(shape_x_final, dtype=dtype_x,
                               name='tensor_a', attrs=attr)

    # w info
    shape_w = w.get('shape')
    dtype_w = w.get('dtype')
    tensor_w = tvm.placeholder(shape_w, dtype=dtype_w, name='tensor_b')

    # b info
    if b is not None:
        shape_b = b.get('shape')
        dtype_b = b.get('dtype')
        shape_bias = (shape_b[1] * shape_b[4],)
        tensor_b = tvm.placeholder(
            shape_bias, dtype=dtype_b, name='tensor_bias', attrs={'ori_shape': shape_bias})
    else:
        tensor_b = None

    # offset_w info
    if offset_w is None:
        tensor_offset_w = None
    else:
        err_man_cube.raise_err_one_para("E62006", "compress_fully_connection",
                                        "tensor offset_w must be None!")

    index_size = tvm.var("index_size", dtype="int32")
    compress_index = tvm.placeholder([index_size, ],
                                     name='compress_index', dtype="int8")

    # Compute
    result = compress_fully_connection_compute(
        tensor_x, tensor_w, compress_index, tensor_b, tensor_offset_w, y,
        num_output, False, axis, offset_x, kernel_name)

    out_addr_type = y.get("addr_type", 0)
    result.op.attrs['addr_type'] = out_addr_type

    # Schedule
    with tvm.target.cce():
        schedule = auto_schedule(result)

    # CCE build
    if b is not None:
        tensor_list = [tensor_x, tensor_w, compress_index, tensor_b, result]
    else:
        tensor_list = [tensor_x, tensor_w, compress_index, result]

    config = {"print_ir": False, "need_build": True, "need_print": True,
              "name": kernel_name, "tensor_list": tensor_list}

    tbe.build(schedule, config)
