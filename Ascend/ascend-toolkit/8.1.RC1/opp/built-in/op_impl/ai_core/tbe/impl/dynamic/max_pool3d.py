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
dynamic max pool3d
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
from impl.util.platform_adapter import tbe_register
from impl.util.pooling_pattern_adapter import PoolingPattern
from impl.util.pooling_pattern_adapter import ReduceWindowAttr
from impl.max_pool3d import op_select_format as max_pool3d_op_select_format


# 'pylint: disable=too-many-arguments,huawei-too-many-arguments
def check_supported(x, y, ksize, strides, padding="SAME", pads=(0, 0, 0, 0, 0, 0),
                    dilation=(1, 1, 1, 1, 1), ceil_mode=0, data_format="NDHWC",
                    kernel_name="max_pool3d"):
    """
    check whether ai_core is supported
    1. dynamic shape
    2. input dtype float32
    3. dilation is not 1
    """
    if util_common.is_unknown([x, y]):
        return True, ""

    input_dtype = x.get("dtype").lower()
    if input_dtype in ("float32",):
        return True, ""

    default_dilations = [[1, 1, 1, 1, 1], [1, 1, 1], [1]]
    dilation_list = list(dilation)
    if dilation_list not in default_dilations:
        return True, ""
    if ceil_mode == 1 and len(x.get("shape")) == 6:
        return True

    return False, "not supported in dynamic, change to static"


@tbe_register.register_param_generalization("MaxPool3D")
def max_pool3d_generalization(x, y, ksize, strides, padding="SAME", pads=(0, 0, 0, 0, 0, 0),
                              dilation=(1, 1, 1, 1, 1), ceil_mode=0, data_format="NDHWC",
                              kernel_name="max_pool3d", generalize_config=None):
    """
    Performs max pooling 3d on the input.
    Parameters
    ----------
    x : dict, shape and dtype of input_data,
        only support float16, shape is 5 dims, format is NDHWC

    y : dict, shape and dtype of output_data, only support float16

    ksize : list or tuple, the window of max_pool3d,
            only support max_pool3d in D or H or W

    strides : list or tuple, the stride of max_pool3d window,
            only support max_pool3d in D or H or W

    padding : str, the mode of padding, support SAME or VALID

    pads :  list or tuple, like (2),(2,2,2)

    dilation : list or tuple

    ceil_mode: int, default = 0

    data_format : str, default = "NDHWC"

    kernel_name : cce kernel name, default value is "max_pool3d"

    Returns
    -------
    parmas list
    """

    if generalize_config["mode"] == "keep_rank":
        return None

    default_dilation = [1, 1, 1, 1, 1]
    dilation_list = list(dilation)
    if dilation_list != default_dilation:
        dilation = None

    x["shape"] = [-2]
    y["shape"] = [-2]
    ksize = None
    strides = None
    pads = None

    generalization_res = [x, y, ksize, strides, padding, pads, dilation, ceil_mode, data_format]
    return [generalization_res]


def op_select_format(x, y, ksize, strides, padding="SAME",
                     pads=(0, 0, 0, 0, 0, 0),
                     dilation=(1, 1, 1), ceil_mode=0, data_format="NDHWC",
                     kernel_name="max_pool3d"):
    """
    max_pool3d ops not performance optimazation yet ,use this function to
    support covid_19 scenario.
    when performance optimazation is done, delete this function
    """
    return max_pool3d_op_select_format(x, y, ksize, strides, padding, pads, dilation,
                                       ceil_mode, data_format, kernel_name)


@register_operator_compute("MaxPool3D", op_mode="dynamic", support_fusion=False)
def max_pool3d_compute(x, y, window_axes, ksize, strides, padding="SAME", pads=(0, 0, 0, 0, 0, 0),
                       dilation=(1, 1, 1, 1, 1), ceil_mode=0, data_format="NDHWC", kernel_name="max_pool3d"):
    """
    Performs max pooling v3 on the input.

    Parameters
    ----------
    x: TVM tensor
        A `Tensor`. Must be one of the following types: `float16`, `uint8`, `int8`.
        6-D input to pool over.
    y: dict
        dict of output_data, include keys(shape and dtype).
    window_axes: list
        A list of `ints`
    ksize: list or tuple
        A list of `ints`.
        The size of the window for each dimension of the input tensor.
    strides: list or tuple
        A list of `ints`.
        The stride of the sliding window for each dimension of the input tensor.
    padding: str
         A `string` from: `"SAME", "VALID", "CALCULATED"`.The type of padding algorithm to use.
    pads: list
        A list of 'ints'.
    dilation: list
        A list of 'ints'.
    ceil_mode: bool
        Flag of ceil mode
    data_format: str
        A `string` from: `"NDC1HWC0", "NDHWC", "NDCHW"`.
    kernel_name: str
        kernel name, default value is 'max_pool3d'

    Returns:
    -------
    res: TVM tensor
        output tensor. Has the same type as `x`.
    """
    rounding_mode = ReduceWindowAttr.FLOOR if not ceil_mode else ReduceWindowAttr.CEIL
    return tbe.reduce_window(
        x, ReduceWindowAttr.MAX, window_axes, ksize, strides, dilation, padding, pads, rounding_mode
    )


def _get_d_h_w_item(ori_list, default_d_index, default_h_index, default_w_index):
    if len(ori_list) == 1:
        return [ori_list[0], ori_list[0], ori_list[0]]
    elif len(ori_list) == 3:
        return [ori_list[0], ori_list[1], ori_list[2]]

    return [ori_list[default_d_index], ori_list[default_h_index], ori_list[default_w_index]]


def _get_window_info_and_add_compile_info(ksize, strides, dilation, pads, data_format):
    """
    get window info and add compile info in unknown_rank cases
    """
    if data_format == "NDHWC":
        d_index = 1
        h_index = 2
        w_index = 3
    else:
        d_index = 2
        h_index = 3
        w_index = 4
    has_unknown_info = False
    if ksize is None:
        window_dimensions = None
        has_unknown_info = True
        operation.add_compile_info(PoolingPattern.WINDOW_DIMENSIONS_ATTR_IDNEX, 0)
    else:
        window_dimensions = _get_d_h_w_item(ksize, d_index, h_index, w_index)

    if strides is None:
        window_strides = None
        has_unknown_info = True
        operation.add_compile_info(PoolingPattern.WINDOW_STRIDES_ATTR_IDNEX, 1)
    else:
        window_strides = _get_d_h_w_item(strides, d_index, h_index, w_index)

    if dilation is None:
        window_dilations = None
        has_unknown_info = True
        operation.add_compile_info(PoolingPattern.WINDOW_DILATIONS_ATTR_IDNEX, 4)
    else:
        window_dilations = _get_d_h_w_item(dilation, d_index, h_index, w_index)

    if pads is None:
        padding_dimensions = None
        has_unknown_info = True
        operation.add_compile_info(PoolingPattern.PADDING_DIMENSIONS_ATTR_IDNEX, 3)
    else:
        padding_dimensions = [[pads[0], pads[1]], [pads[2], pads[3]], [pads[4], pads[5]]]

    if has_unknown_info:
        actual_indices_list = [d_index, h_index, w_index]
        operation.add_compile_info(PoolingPattern.ACTUAL_WINDOW_ORI_INDICES, actual_indices_list)

    res = (window_dimensions, window_strides, window_dilations, padding_dimensions)

    return res


@register_operator("MaxPool3D")
def max_pool3d(x, y, ksize, strides, padding="SAME", pads=(0, 0, 0, 0, 0, 0),
               dilation=(1, 1, 1, 1, 1), ceil_mode=0, data_format="NDHWC",
               kernel_name="max_pool3d"):
    """
    Performs max pooling 3d on the input.
    Parameters
    ----------
    x : dict, shape and dtype of input_data, only support float16, float32

    y : dict, shape and dtype of output_data, only support float16, float32

    ksize : list or tuple with length 1 or 3 or 5, the window of max_pool3d

    strides : list or tuple with length 1 or 3 or 5, the stride of max_pool3d window

    padding : str, the mode of padding, support SAME or VALID or CALCULATED

    pads :  list or tuple with length 6

    dilation : list or tuple with length 1 or 3 or 5, the dilation of max_pool3d window

    data_format : str, support NDHWC or NCDHW, default = "NDHWC"

    kernel_name : cce kernel name, default value is "max_pool3d"

    Returns
    -------
    None
    """
    dtype = x.get("dtype")
    dtype_lower = dtype.lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_lower, check_list, param_name="x")

    if is_unknown_rank_input(x):
        x["shape"] = (-1, -1, -1, -1, -1, 16)
        x["range"] = ((1, None), (1, None), (1, None), (1, None), (1, None), (16, 16))

    window_dimensions, window_strides, window_dilations, padding_dimensions =\
        _get_window_info_and_add_compile_info(ksize, strides, dilation, pads, data_format)
    extra_params = {
        PoolingPattern.WINDOW_DIMENSIONS: window_dimensions,
        PoolingPattern.WINDOW_STRIDES: window_strides,
        PoolingPattern.WINDOW_DILATIONS: window_dilations,
        PoolingPattern.PADDING_DIMENSIONS: padding_dimensions
    }
    ins = classify([x, [1, 3, 4]], OpPatternMode.POOLING, extra_params)

    schedules = []
    tensors = []
    for (x, axis) in ins:
        with tbe.compute():
            shape_var_new = shape_util.variable_shape([x], op_mode=OpPatternMode.POOLING)[0]
            window_dimensions = x.get(PoolingPattern.WINDOW_DIMENSIONS)
            window_strides = x.get(PoolingPattern.WINDOW_STRIDES)
            window_dilations = x.get(PoolingPattern.WINDOW_DILATIONS)
            padding_dimensions = x.get(PoolingPattern.PADDING_DIMENSIONS)

            data_input = tvm.placeholder(shape_var_new, name="data_input", dtype=dtype_lower)
            res = max_pool3d_compute(data_input, y, axis, window_dimensions, window_strides, padding,
                                     padding_dimensions, window_dilations, ceil_mode, data_format,
                                     kernel_name)
            tensors.append([data_input, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
