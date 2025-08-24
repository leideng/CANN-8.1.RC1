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
reduction
"""
import functools

from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tvm
from impl.util import util_select_op_base


# 'pylint: disable=redefined-outer-name, too-many-arguments, E1101
# 'pylint: disable=unused-argument,too-many-locals
def op_select_format(input_x, output_y, operation=1, axis=0, coeff=1.0, kernel_name="reduction"):
    """
    1. when input x's ori_shape in ["NHWC", "NCHW"] and attr axis
    can't be dim C. the Op Reduction can support ND.
    > for example:
    > for example:
    > x : Tensor of (shape=(16, 16), "ND")
    > the Op Select can process with NC1HWC0:
    > x : Tensor of (shape=(16, 1, 16, 16, 16), "NC1HWC0")
    """
    input_ori_shape = input_x.get("ori_shape")
    input_ori_format = input_x.get("ori_format")

    if axis < 0:
        axis = len(input_ori_shape) + axis

    is_support_5hd = True

    if input_ori_format not in ("NCHW", "NHWC"):
        is_support_5hd = False

    if (input_ori_format == "NCHW" and axis == 1) \
            or (input_ori_format == "NHWC" and axis == 3):
        is_support_5hd = False

    if len(input_ori_shape) < 4:
        is_support_5hd = False

    if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        dtype_base = ["float16"]
    else:
        dtype_base = ["float16", "float32"]

    format_base = ["ND"] * len(dtype_base)
    if is_support_5hd:
        dtype_base = dtype_base + ["float16"]
        format_base = format_base + ["NC1HWC0"]

    dtype_base = ','.join(dtype_base)
    format_base = ','.join(format_base)

    input0 = util_select_op_base.gen_param(classify="input0", name="x", datatype=dtype_base, format=format_base)
    output0 = util_select_op_base.gen_param(classify="output0", name="y", datatype=dtype_base, format=format_base)
    param_list = [input0, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


@register_operator_compute("reduction", op_mode="static", support_fusion=True)
def reduction_compute(data_info, product_verion, operation, axis, coeff):
    """
    Reduce a tensor on a certain axis, and scale output with coeff
    Parameters
    ----------
    data_info: include TVM tensor,shape and dtype
    product_verion: include mini("1.1"、"1.3"),cloud("1.6"),es("5.10"),DC("2.3")
    operation : can only be one of "1:SUM, 2:ASUM (sum of abs), 3:SUMSQ (sum of sqr), 4:MEAN"
    axis : the axis to reduce
    coeff : scale for output
    Returns
    -------
    output of the data's reduction
    """
    input_data = data_info.get("tensor")
    input_data_dtype = data_info.get("dtype")
    mean_size = input_data.op.attrs["mean_size"].value

    if product_verion not in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        if input_data_dtype == "float16":
            input_data = tbe.cast_to(input_data, "float32")

    # computational process
    if operation == 2:
        data_tmp_input = tbe.vabs(input_data)
        tmp = tbe.vmuls(data_tmp_input, coeff)

    elif operation == 3:
        data_tmp_input = tbe.vmul(input_data, input_data)
        tmp = tbe.vmuls(data_tmp_input, coeff)

    elif operation == 4:
        cof = float(coeff * (mean_size**(-0.5)))
        tmp = tbe.vmuls(input_data, cof)

    elif operation == 1:
        tmp = tbe.vmuls(input_data, coeff)

    res = tbe.sum(tmp, axis=axis)

    if operation == 4:
        size_reci = float(mean_size**(-0.5))
        res = tbe.vmuls(res, size_reci)

    if product_verion not in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        if input_data_dtype == "float16":
            res = tbe.cast_to(res, "float16")

    return res


# 'pylint: disable=redefined-outer-name, too-many-arguments, E1101
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def reduction(input_x, output_y, operation=1, axis=0, coeff=1.0, kernel_name="reduction"):
    """
    Reduce a tensor on a certain axis, and scale output with coeff
    Parameters
    ----------
    input_x : input tensor
    output_y: output tensor
    operation : can only be one of "1:SUM, 2:ASUM (sum of abs), 3:SUMSQ (sum of sqr), 4:MEAN"
    axis : the first axis to reduce, may be negative to index from the end
            (e.g., -1 for the last axis).If axis == 0, the output Blob always has
            the empty shape (count 1), performing reduction across the entire input.
    coeff : scale for output
    kernel_name : cce kernel name, default value is "cce_reductionLayer"
    Returns
    -------
    None
    """
    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")

    # input_x's shape check
    ori_shape = list(input_x.get("ori_shape"))
    para_check.check_shape(ori_shape, param_name="input_x")

    # input_x' dtype check
    inp_dtype = input_x.get("dtype").lower()
    para_check.check_dtype(inp_dtype, ("float16", "float32"), param_name="input_x")
    if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403") and inp_dtype == "float32":
        error_manager_vector.raise_err_specific_reson(
            "reduction", "ES is not supported while the x's dtype\
                                                      of input is [{}].".format(inp_dtype))

    # axis parameter check
    if axis >= len(ori_shape) or axis < -len(ori_shape):
        error_manager_vector.raise_err_input_param_range_invalid("reduction", "axis", str(-len(ori_shape)),
                                                                 str(len(ori_shape) - 1), str(axis))

    # operation parameter check
    if operation not in (1, 2, 3, 4):
        error_manager_vector.raise_err_input_value_invalid("reduction", "operation", "1, 2, 3 or 4", str(operation))

    # Preprocess
    if axis < 0:
        axis = len(ori_shape) + axis

    shape = list(input_x.get("shape"))
    mean_size = 0
    if input_x.get("format") == "NC1HWC0":
        axis = shape_util.axis_transform_5d(axis, input_x.get("ori_format"))
        if axis > 1:
            shape = shape[:axis] + [functools.reduce(lambda x, y: x * y, shape[axis:-1])] + [shape[-1]]
            mean_size = shape[-2]
        if axis == 1:
            rule_desc = "The C axis does not support reduction when the data format is NC1HWC0"
            error_manager_vector.raise_err_check_params_rules("reduction", rule_desc, "axis", axis)
        if axis == 0:
            shape = [functools.reduce(lambda x, y: x * y, shape)]
            ori_shape = [functools.reduce(lambda x, y: x * y, ori_shape)]
            mean_size = ori_shape[-1]
    else:
        shape = shape[:axis] + [functools.reduce(lambda x, y: x * y, shape[axis:])]
        mean_size = shape[-1]
    attr = {"mean_size": mean_size}

    # define input
    data = tvm.placeholder(shape, name="data_input", dtype=inp_dtype, attrs=attr)
    data_info = {"tensor": data, "shape": shape, "dtype": inp_dtype}

    res = reduction_compute(data_info, cce_product, operation, axis, coeff)

    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": [data, res]}
    build(sch, config)
