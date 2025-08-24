#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
dynamic max pool v3
"""
from impl.util import util_common
from impl.util.util_common import is_unknown_rank_input
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.pooling_pattern_adapter import PoolingPattern
from impl.util.pooling_pattern_adapter import ReduceWindowAttr


# 'pylint: disable=too-many-arguments,huawei-too-many-arguments
def check_supported(input_data,
                    output_data,
                    ksize,
                    strides,
                    padding_mode="CALCULATED",
                    pads=(0, 0, 0, 0),
                    data_format="NHWC",
                    global_pooling=False,
                    ceil_mode=False,
                    kernel_name="max_pool_v3"):
    """
    check whether ai_core is supported
    """
    if util_common.is_unknown([input_data, output_data]):
        return True, ""

    input_dtype = input_data.get("dtype").lower()
    if input_dtype in ("float32",):
        return True, ""

    return False, ""


@register_operator_compute("MaxPoolV3", op_mode="dynamic", support_fusion=False)
def max_pool_v3_compute(input_data, output_data, window_axes, ksize, strides, padding_mode="CALCULATED",
                        pads=(0, 0, 0, 0), data_format="NHWC", global_pooling=False, ceil_mode=False,
                        kernel_name="max_pool_v3"):
    """
    Performs max pooling v3 on the input.

    Parameters
    ----------
    input_data: TVM tensor
        A `Tensor`. Must be one of the following types: `float16`.
        5-D input to pool over.
    output_data: dict
        dict of output_data, include keys(shape and dtype).
    window_axes: list
        A list of `ints`
    ksize: list or tuple
        A list of `ints`.
        The size of the window for each dimension of the input tensor.
    strides: list or tuple
        A list of `ints`.
        The stride of the sliding window for each dimension of the input tensor.
    padding_mode: str
         A `string` from: `"SAME", "VALID"`.The type of padding algorithm to use.
    pads: list
        A list of 'ints'.
    data_format: str
        A `string` from: `"NC1HWC0", "NHWC", "NCHW"`.
    global_pooling: bool
        Flag of global_pooling
    ceil_mode: bool
        Flag of ceil mode
    kernel_name: str
        kernel name, default value is 'max_pool_v3'

    Returns:
    -------
    res: TVM tensor
        output tensor. Has the same type as `input_data`.
    """
    computation = ReduceWindowAttr.GMP if global_pooling else ReduceWindowAttr.MAX
    rounding_mode = ReduceWindowAttr.CEIL if ceil_mode else ReduceWindowAttr.FLOOR
    return tbe.reduce_window(
        input_data, computation, window_axes, ksize, strides, [1] * len(window_axes),
        padding_mode, pads, rounding_mode
    )


def _get_window_info_and_add_compile_info(ksize, strides, pads, data_format):
    """
    get window info and add compile info in unknown_rank cases
    """
    if data_format == "NHWC":
        h_index = 1
        w_index = 2
    else:
        h_index = 2
        w_index = 3
    has_unknown_info = False
    if ksize is None:
        window_dimensions = None
        has_unknown_info = True
        operation.add_compile_info(PoolingPattern.WINDOW_DIMENSIONS_ATTR_IDNEX, 0)
    else:
        window_dimensions = [ksize[h_index], ksize[w_index]]

    if strides is None:
        window_strides = None
        has_unknown_info = True
        operation.add_compile_info(PoolingPattern.WINDOW_STRIDES_ATTR_IDNEX, 1)
    else:
        window_strides = [strides[h_index], strides[w_index]]

    if pads is None:
        padding_dimensions = None
        has_unknown_info = True
        operation.add_compile_info(PoolingPattern.PADDING_DIMENSIONS_ATTR_IDNEX, 3)
    else:
        padding_dimensions = [[pads[0], pads[1]], [pads[2], pads[3]]]

    if has_unknown_info:
        actual_indices_list = [h_index, w_index]
        operation.add_compile_info(PoolingPattern.ACTUAL_WINDOW_ORI_INDICES, actual_indices_list)

    return window_dimensions, window_strides, padding_dimensions


@register_operator("MaxPoolV3")
def max_pool_v3(input_data,
                output_data,
                ksize,
                strides,
                padding_mode="CALCULATED",
                pads=(0, 0, 0, 0),
                data_format="NHWC",
                global_pooling=False,
                ceil_mode=False,
                kernel_name="max_pool_v3"):
    """
    Performs max pooling v3 on the input.

    Parameters
    ----------
    input_data: dict
        dict of input_data, include keys(shape and dtype).
    output_data: dict
        dict of output_data, include keys(shape and dtype).
    ksize: list or tuple
        A list of `ints` that has length 4.
        The size of the window for each dimension of the input tensor.
    strides: list or tuple
        A list of `ints` that has length 4.
        The stride of the sliding window for each dimension of the input tensor.
    padding_mode: str
        A `string` from: `"SAME", "VALID", "CALCULATED"`.The type of padding algorithm to use.
    pads: list
        A list of 'ints' that has length 4.
    data_format: str
        A `string` from: `"NC1HWC0", "NHWC", "NCHW"`.
    global_pooling: bool
        Flag of global_pooling
    ceil_mode: bool
        Flag of ceil mode
    kernel_name: str
        kernel name, default value is 'max_pool_v3'

    Returns:
    -------
    None
    """
    dtype = input_data.get("dtype")
    dtype_lower = dtype.lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_lower, check_list, param_name="x")

    if is_unknown_rank_input(input_data):
        input_data["shape"] = (-1, -1, -1, -1, 16)
        input_data["range"] = ((1, None), (1, None), (1, None), (1, None), (16, 16))

    window_dimensions, window_strides, padding_dimensions =\
        _get_window_info_and_add_compile_info(ksize, strides, pads, data_format)
    extra_params = {
        PoolingPattern.WINDOW_DIMENSIONS: window_dimensions,
        PoolingPattern.WINDOW_STRIDES: window_strides,
        PoolingPattern.PADDING_DIMENSIONS: padding_dimensions,
        PoolingPattern.GLOBAL_POOLING: global_pooling,
    }
    ins = classify([input_data, [2, 3]], OpPatternMode.POOLING, extra_params)

    schedules = []
    tensors = []
    for (x, axis) in ins:
        with tbe.compute():
            shape_var_new = shape_util.variable_shape([x], op_mode=OpPatternMode.POOLING)[0]
            window_dimensions = x.get(PoolingPattern.WINDOW_DIMENSIONS)
            window_strides = x.get(PoolingPattern.WINDOW_STRIDES)
            padding_dimensions = x.get(PoolingPattern.PADDING_DIMENSIONS)

            data_input = tvm.placeholder(shape_var_new, name="data_input", dtype=dtype_lower)
            res = max_pool_v3_compute(data_input, output_data, axis, window_dimensions, window_strides,
                                      padding_mode, padding_dimensions, data_format, global_pooling, ceil_mode,
                                      kernel_name)
            tensors.append([data_input, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
