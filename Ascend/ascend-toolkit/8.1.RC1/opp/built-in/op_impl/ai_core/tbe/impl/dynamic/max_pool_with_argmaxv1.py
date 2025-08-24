#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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
max_pool_with_argmaxv1
"""
from impl.util.util_common import is_unknown_rank_input
from impl.dynamic.max_pool_with_argmaxv2 import MaxPoolWithargmaxPytorch
from impl.dynamic.max_pool_with_argmaxv2 import _check_param
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """

    DT_INT32 = 3
    # dimension n
    DIM_N = 0
    # dimension h
    DIM_H = 1
    # dimension w
    DIM_W = 2
    # dimension c
    DIM_C = 3
    DEFAULT_DTYPE = 3
    WINDOW_AXES = "WINDOW_AXES"
    ATTR_AXES = "ATTR_AXES"
    WINDOW_DIMENSIONS = "WINDOW_DIMENSIONS"
    WINDOW_STRIDES = "WINDOW_STRIDES"
    WINDOW_PADDINGS = "WINDOW_PADDINGS"
    WINDOW_DILATIONS = "WINDOW_DILATIONS"
    CEIL_MODE = "CEIL_MODE"

    def __init__(self):
        pass


def max_pool_with_argmaxv1_compute(x, y, argmax, ksize, strides, pads, dtype, dilation, ceil_mode=False,
                                   kernel_name="max_pool_with_argmax_v1"):
    """
    max_pool_with_argmax_v1 compute for dsl
    """
    round_mode = "CEIL" if ceil_mode else "FLOOR"
    window_axes = (x.op.attrs["window_axes"][0].value, x.op.attrs["window_axes"][1].value)
    return tbe.reduce_window(x, "MAX", window_axes, ksize, strides, dilation, "CALCULATED", pads, round_mode, True)


def max_pool_with_argmax_v1_dsl(x, y, argmax, ksize, strides, pads, dtype=Constant.DEFAULT_DTYPE,
                                dilation=(1, 1, 1, 1), ceil_mode=False, kernel_name="max_pool_with_argmaxv1"):
    """
    max_pool_with_argmax_v1 interface for dsl
    :param x: dict of shape and dtype of the input x
    :param y: dict of shape and dtype of the output y
    :param argmax: dict of shape and dtype of the output argmax
    :param ksize: the size of the window to take a max over
    :param strides: the stride of the window
    :param pads: implicit zero padding to be added on both sides
    :param dilation: a parameter that controls the stride of elements \
                     in the window
    :param ceil_mode: when True, will use ceil instead of floor to compute \
                      the output shape
    :param dtype: input data type, only support int32 or int64
    :param kernel_name: the kernel's name
    """
    dtype = x.get("dtype")
    dtype_lower = dtype.lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_lower, check_list, param_name="x")
    windows_axes = [2, 3]
    attr_axes = [1, 2]
    if is_unknown_rank_input(x):
        if x.get("format") == "NC1HWC0":
            x["shape"] = (-1, -1, -1, -1, 16)
            x["range"] = ((1, None), (1, None), (1, None), (1, None), (16, 16))
        else:
            x["shape"] = (-1, -1, -1, -1)
            x["range"] = ((1, None), (1, None), (1, None), (1, None))
        extra_params = {
            Constant.WINDOW_AXES: windows_axes,
            Constant.ATTR_AXES: attr_axes,
            Constant.WINDOW_DIMENSIONS: [None, None, None, None],
            Constant.WINDOW_STRIDES: [None, None, None, None],
            Constant.WINDOW_PADDINGS: [[None, None], [None, None], [None, None], [None, None]],
            Constant.WINDOW_DILATIONS: [1, 1, 1, 1],
            Constant.CEIL_MODE: ceil_mode
        }
    else:
        extra_params = {
            Constant.WINDOW_AXES: windows_axes,
            Constant.ATTR_AXES: attr_axes,
            Constant.WINDOW_DIMENSIONS: ksize,
            Constant.WINDOW_STRIDES: strides,
            Constant.WINDOW_PADDINGS: [[pads[0], pads[0]], [pads[1], pads[1]],
                                       [pads[2], pads[2]], [pads[3], pads[3]]],
            Constant.WINDOW_DILATIONS: [1, 1, 1, 1],
            Constant.CEIL_MODE: ceil_mode
        }

    ins = classify([x, ], "pooling_with_arg", extra_params)
    schedules = []
    tensors = []
    for _x, _axes, _ksize, _strides, _paddings, _dilations, _mode in ins:
        with tbe.compute():
            shape_var, window_axes, window_dimensions, window_strides, padding_dimensions, window_dilations \
                = shape_util.variable_shape([_x, _axes, _ksize, _strides, _paddings, _dilations, _mode],
                                            op_mode="pooling_with_arg")

            data_input = tvm.placeholder(shape_var, name="data_input", dtype=dtype_lower,
                                         attrs={"window_axes": window_axes})
            res = max_pool_with_argmaxv1_compute(data_input, y, argmax, window_dimensions, window_strides,
                                                 padding_dimensions, 3, window_dilations, _mode)
            tensors.append([data_input, ] + res)

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    tbe_context.get_context().add_compile_info("dimensions_attr_idx", 0)
    tbe_context.get_context().add_compile_info("strides_attr_idx", 1)
    tbe_context.get_context().add_compile_info("pads_attr_idx", 2)
    tbe_context.get_context().add_compile_info("dilations_attr_idx", 4)
    tbe_context.get_context().add_compile_info("ceil_mode_idx", 5)
    tbe_context.get_context().add_compile_info("dimensions_attr_name", "ksize")
    tbe_context.get_context().add_compile_info("strides_attr_name", "strides")
    tbe_context.get_context().add_compile_info("pads_attr_name", "pads")
    tbe_context.get_context().add_compile_info("dilations_attr_name", "dilation")
    tbe_context.get_context().add_compile_info("ceil_mode_name", "ceil_mode")
    # It is used to distinguish between Tik implementation and DSL implementation in the tilling phase
    tbe_context.get_context().add_compile_info("is_dsl", True)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)


def max_pool_with_argmax_v1_tik(x, y, argmax, ksize, strides, pads, dtype, dilation, ceil_mode, kernel_name):
    """
    max_pool_with_argmax_v1 interface for tik

    :param x: dict of shape and dtype of the input x
    :param y: dict of shape and dtype of the output y
    :param argmax: dict of shape and dtype of the output argmax
    :param ksize: the size of the window to take a max over
    :param strides: the stride of the window
    :param pads: implicit zero padding to be added on both sides
    :param dilation: a parameter that controls the stride of elements \
                     in the window
    :param ceil_mode: when True, will use ceil instead of floor to compute \
                      the output shape
    :param dtype: input data type, only support int32 or int64
    :param kernel_name: the kernel's name
    :return: tik_instance
    """
    x_type = x.get("dtype")
    if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") == "Ascend310P" and x_type == "float32":
        obj = MaxPoolWithargmaxPytorch(x.get("shape"), ksize, strides, pads, x_type.lower(),
                                       dilation, ceil_mode, kernel_name)
    else:
        dim_n = Constant.DIM_N
        dim_h = Constant.DIM_H
        dim_w = Constant.DIM_W
        dim_c = Constant.DIM_C

        ksize = [ksize[dim_n], ksize[dim_c], ksize[dim_h], ksize[dim_w]]
        strides = [strides[dim_n], strides[dim_c], strides[dim_h], strides[dim_w]]
        pads = [pads[dim_n], pads[dim_c], pads[dim_h], pads[dim_w]]
        dilation = [dilation[dim_n], dilation[dim_c], dilation[dim_h], dilation[dim_w]]

        [_, _, dim_h, dim_w] = _check_param(x, ksize, strides, pads, dtype, dilation, ceil_mode, kernel_name)
        obj = MaxPoolWithargmaxPytorch(x.get("shape"), [1, ksize[dim_h], ksize[dim_w], 1],
                                       [1, strides[dim_h], strides[dim_w], 1],
                                       [1, pads[dim_h], pads[dim_w], 1], x_type.lower(),
                                       dilation, ceil_mode, kernel_name)

    return obj.max_pool_operator()


# 'pylint: disable=unused-argument
@register_operator("MaxPoolWithArgmaxV1")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def max_pool_with_argmax_v1(x, y, argmax, ksize, strides, pads, dtype=Constant.DT_INT32, dilation=(1, 1, 1, 1),
                            ceil_mode=False, kernel_name="max_pool_with_argmax_v1"):
    """
    implementation of max_pool_with_argmax for pytorch and return the \
    tik instance
    :param x: dict of shape and dtype of the input x
    :param y: dict of shape and dtype of the output y
    :param argmax: dict of shape and dtype of the output argmax
    :param ksize: the size of the window to take a max over
    :param strides: the stride of the window
    :param pads: implicit zero padding to be added on both sides
    :param dilation: a parameter that controls the stride of elements \
                     in the window
    :param ceil_mode: when True, will use ceil instead of floor to compute \
                      the output shape
    :param dtype: input data type, only support int32 or int64
    :param kernel_name: the kernel's name
    """
    x_type = x.get("dtype")
    if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") == "Ascend310P" and x_type == "float32":
        max_pool_with_argmax_v1_tik(x, y, argmax, ksize, strides, pads, dtype, dilation, ceil_mode, kernel_name)
    elif is_unknown_rank_input(x) or x_type == "float32" or x.get("format") == "NCHW":
        max_pool_with_argmax_v1_dsl(x, y, argmax, ksize, strides, pads, dtype, dilation, ceil_mode, kernel_name)
    else:
        max_pool_with_argmax_v1_tik(x, y, argmax, ksize, strides, pads, dtype, dilation, ceil_mode, kernel_name)
