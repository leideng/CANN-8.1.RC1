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
normalize_scale
"""
import te.lang.cce as tbe
from impl.util.platform_adapter import build
from impl.util.platform_adapter import auto_schedule
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util import util_select_op_base
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable = unused-argument
def get_op_support_info(x1, x2, x3, y, across_spatial=True, channel_shared=True,
                        eps=1e-10, kernel_name="normalize_scale"):
    format_x = x1.get("format")
    axis_split_list = []
    # temp modify, next step will provide a reverse infershape interface
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(None, None, 0, 0)
    return op_cal_info_in_json

    # 'pylint: disable=unreachable
    if format_x == "NCHW" or format_x == "NHWC":
        split_0 = [util_select_op_base.SplitInput([0, [0], [-1], [-1]], [1, [0], [-1], [-1]], [2, [0], [-1], [-1]]),
                   util_select_op_base.SplitOutput([0, [0]])]
        split_1 = [util_select_op_base.SplitInput([0, [1], [-1], [-1]], [1, [1], [-1], [-1]], [2, [1], [-1], [-1]]),
                   util_select_op_base.SplitOutput([0, [1]])]
        split_2 = [util_select_op_base.SplitInput([0, [2], [-1], [-1]], [1, [2], [-1], [-1]], [2, [2], [-1], [-1]]),
                   util_select_op_base.SplitOutput([0, [2]])]
        split_3 = [util_select_op_base.SplitInput([0, [3], [-1], [-1]], [1, [3], [-1], [-1]], [2, [3], [-1], [-1]]),
                   util_select_op_base.SplitOutput([0, [3]])]
        axis_split_list = [split_0, split_1, split_2, split_3]
    else:
        axis_split_list = None

    axis_reduce_list = None
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(axis_split_list, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


# 'pylint: disable=locally-disabled,unused-argument,invalid-name,too-many-locals
# 'pylint: disable=locally-disabled,too-many-arguments,protected-access
# 'pylint: disable=locally-disabled,too-many-branches
@register_operator_compute("normalize_scale", op_mode="static", support_fusion=True)
def normalize_scale_compute(x1, x2, x3, y,
                            across_spatial=True, eps=1e-10,
                            kernel_name="normalize_scale"):
    """
    calculating data

    Parameters
    ----------
    x1 : TVM tensor
        the placeholder of x1
    x2 : TVM tensor
        the placeholder of x2
    x3 : TVM tensor
        the placeholder of x3
    y : dict
        dict of y, include keys(shape and dtype, format)
    across_spatial: bool
        indicates whether standardization should cross spatial locations.
        Default(True)
    eps: float
        prevent dividing by 0.
        Default(1e-10)
    kernel_name : str
        kernel name, default value is "normalize_scale"

    Returns
    -------
    output tensor
    """

    # set intermediate dtype
    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        # hisi es, cs
        intermediate_dtype = "float16"
        dtype_cast_mapping = {"int8": "float16"}
        dtype_reverse_cast_mapping = {"float16": "int8"}
    else:
        # mini, cloud
        intermediate_dtype = "float32"
        dtype_cast_mapping = {"int8": "float16", "float16": "float32"}
        dtype_reverse_cast_mapping = {"float16": "int8",
                                      "float32": "float16"}

    x1_shape = shape_util.shape_to_list(x1.shape)

    x1_cast = x1
    while x1_cast.dtype in dtype_cast_mapping:
        x1_cast = tbe.cast_to(x1_cast,
                              dtype_cast_mapping[x1_cast.dtype])
    x2_cast = x2
    while x2_cast.dtype in dtype_cast_mapping:
        x2_cast = tbe.cast_to(x2_cast,
                              dtype_cast_mapping[x2_cast.dtype])

    x3_cast = x3
    while x3_cast.dtype in dtype_cast_mapping:
        x3_cast = tbe.cast_to(x3_cast,
                              dtype_cast_mapping[x3_cast.dtype])

    x1_sqr_sum = tbe.vadds(x3_cast,
                           tvm.const(eps, dtype=intermediate_dtype))

    x2_cast_broadcast = tbe.broadcast(x2_cast, x1_shape)

    x1_scaled = tbe.vmul(x1_cast, x2_cast_broadcast)

    if cce_product in ("Ascend910", "Ascend610", "Ascend310P"):
        x1_sqr_sum_sqrt = tbe.vsqrt(x1_sqr_sum)
        x1_sqr_sum_sqrt_broadcast = tbe.broadcast(x1_sqr_sum_sqrt,
                                                  x1_shape)
        x1_normalized = tbe.vdiv(x1_scaled, x1_sqr_sum_sqrt_broadcast)
    elif cce_product in ("Ascend310",):
        # customized for mini, using newton
        x1_sqr_sum_sqrt = tbe.vsqrt(x1_sqr_sum)

        for _ in range(1):
            res = tbe.vdiv(x1_sqr_sum, x1_sqr_sum_sqrt)
            res = tbe.vadd(res, x1_sqr_sum_sqrt)
            res = tbe.vmuls(res, tvm.const(0.5, intermediate_dtype))
            x1_sqr_sum_sqrt = res
        x1_sqr_sum_rsqrt = tbe.vrec(x1_sqr_sum_sqrt)
        x1_sqr_sum_rsqrt_broadcast = tbe.broadcast(x1_sqr_sum_rsqrt,
                                                   x1_shape)
        x1_normalized = tbe.vmul(x1_scaled, x1_sqr_sum_rsqrt_broadcast)
    else:
        # for mini and hisi-es
        x1_sqr_sum_rsqrt = tbe.vrsqrt(x1_sqr_sum)
        x1_sqr_sum_rsqrt_broadcast = tbe.broadcast(x1_sqr_sum_rsqrt,
                                                   x1_shape)
        x1_normalized = tbe.vmul(x1_scaled, x1_sqr_sum_rsqrt_broadcast)

    x1_normalized_cast = x1_normalized
    while x1_normalized_cast.dtype != x1.dtype and \
            x1_normalized_cast.dtype in dtype_reverse_cast_mapping:
        x1_normalized_cast = tbe.cast_to(x1_normalized_cast,
                                         dtype_reverse_cast_mapping[
                                         x1_normalized_cast.dtype])

    return x1_normalized_cast


def _check_format(data_format, data_format_3):
    """
    check the format for x1 and x3

    Parameters
    ----------
    data_format : str
        the format for x1
    data_format_3 : str
        the format for x3

    Returns
    -------
    None
    """

    if data_format != data_format_3:
        error_manager_vector.raise_err_two_input_format_invalid("normalize_scale", "data_format", "data_format3",
                                                                "the parameter[data_format] must should be equal to \
                                                                the parameter[data_format3]in format")

    para_check.check_format(data_format, ("NCHW", "NHWC"), param_name="x1")


def _check_dtype(dtype_1, dtype_3):
    """
    check the dtype for x1, x3

    Parameters
    ----------
    dtype_1 : str
        dtype for x1
    dtype_3 : str
        dtype for x3

    Returns
    -------
    None
    """

    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        # hisi es, cs
        para_check.check_dtype(dtype_1, ("int8", "float16",), param_name="x1")
        para_check.check_dtype(dtype_3, ("int8", "float16",), param_name="x3")
    else:
        para_check.check_dtype(dtype_1, ("int8", "float16", "float32",), param_name="x1")
        para_check.check_dtype(dtype_3, ("int8", "float16", "float32",), param_name="x3")


def _check_shape_1(shape_1):
    """
    check the shape for x1

    Parameters
    ----------
    shape_1 : list or tuple
        shape for x1

    Returns
    -------
    None
    """

    para_check.check_shape(shape_1, param_name="x1")
    para_check.check_shape(shape_1, min_rank=4, max_rank=4, param_name="x1")


def _check_shape_2(shape_1, data_format, channel_shared):
    """
    check the shape for x2

    Parameters
    ----------
    shape_1 : list or tuple
        shape for x1
    data_format : str
        format for x1
    channel_shared: bool
        used to control whether x2 are shared by multiple channels.
        Default(True)

    Returns
    -------
    the expand shape for x2, used for placeholder
    """

    if channel_shared:
        shape_2 = [1, 1, 1, 1]
    elif data_format == "NCHW":
        shape_2 = [1, shape_1[1], 1, 1]
    elif data_format == "NHWC":
        shape_2 = [1, 1, 1, shape_1[3]]

    return shape_2


def _check_shape_3(shape_1, shape_3, data_format, across_spatial):
    """
    check the shape for x3

    Parameters
    ----------
    shape_1 : list or tuple
        shape for x1
    shape_3 : list or tuple
        shape for x3
    data_format : str
        format for x1 and x3
    across_spatial: bool
        indicates whether standardization should cross spatial locations.
        Default(True)

    Returns
    -------
    None
    """
    para_check.check_shape(shape_3, param_name="x3")
    para_check.check_shape(shape_3, min_rank=4, max_rank=4, param_name="x3")

    if across_spatial:
        if not (shape_3[0] == shape_1[0] and shape_3[1] == 1 and
                shape_3[2] == 1 and shape_3[3] == 1):
            error_manager_vector.raise_err_inputs_shape_not_equal("normalize_scale", "x3", "x1",
                                                                  str(shape_3), str(shape_1),
                                                                  "(" + str(shape_1[0]) + ", 1, 1, 1)")

    elif data_format == "NCHW":
        if not (shape_3[0] == shape_1[0] and shape_3[1] == 1 and
                shape_3[2] == shape_1[2] and shape_3[3] == shape_1[3]):
            error_manager_vector.raise_err_inputs_shape_not_equal("normalize_scale", "x3", "x1",
                                                                  str(shape_3), str(shape_1),
                                                                  "(" + str(shape_1[0]) + ", 1, " +
                                                                  str(shape_1[2]) + ", " +
                                                                  str(shape_1[3]) + ")")

    elif data_format == "NHWC":
        if not (shape_3[0] == shape_1[0] and shape_3[1] == shape_1[1] and
                shape_3[2] == shape_1[2] and shape_3[3] == 1):
            error_manager_vector.raise_err_inputs_shape_not_equal("normalize_scale", "x3", "x1",
                                                                  str(shape_3), str(shape_1),
                                                                  "(" + str(shape_1[0]) + ", " +
                                                                  str(shape_1[1]) + ", " +
                                                                  str(shape_1[2]) + ", 1)")


# 'pylint: disable=locally-disabled,invalid-name,too-many-arguments
# 'pylint: disable=locally-disabled,too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def normalize_scale(x1, x2, x3, y, across_spatial=True,
                    channel_shared=True, eps=1e-10,
                    kernel_name="normalize_scale"):
    """
    calculating data

    Parameters
    ----------
    x1 : dict
        shape and dtype, format of input 1
    x2 : dict
        shape and dtype, format of input 2
    x3 : dict
        shape and dtype, format of input 3
    y : dict
        shape and dtype, format of output,
        should be same shape and type, format as input 1
    across_spatial: bool
        indicates whether standardization should cross spatial locations.
        Default(True)
    channel_shared: bool
        used to control whether x2 are shared by multiple channels.
        Default(True)
    eps: float
        prevent dividing by 0.
        Default(1e-10)
    kernel_name : str
        kernel name, default value is "normalize_scale"

    Returns
    -------
    None
    """

    shape_1 = x1.get("shape")
    dtype_1 = x1.get("dtype").lower()
    data_format = x1.get("format")

    shape_3 = x3.get("shape")
    dtype_3 = x3.get("dtype").lower()
    data_format_3 = x3.get("format")

    _check_format(data_format, data_format_3)
    _check_dtype(dtype_1, dtype_3)
    if len(list(shape_1)) == 2:
        if data_format == "NCHW":
            shape_1 = [shape_1[0], shape_1[1], 1, 1]
        elif data_format == "NHWC":
            shape_1 = [shape_1[0], 1, 1, shape_1[1]]
    _check_shape_1(shape_1)
    _check_shape_3(shape_1, shape_3, data_format, across_spatial)

    # the expand shape for x2, used for placeholder
    shape_2 = _check_shape_2(shape_1, data_format, channel_shared)
    dtype_2 = dtype_1

    data_x1 = tvm.placeholder(shape_1, name="data_1", dtype=dtype_1)
    data_x2 = tvm.placeholder(shape_2, name="data_2", dtype=dtype_2)
    data_x3 = tvm.placeholder(shape_3, name="data_3", dtype=dtype_3)
    res = normalize_scale_compute(data_x1, data_x2, data_x3, y,
                                  across_spatial, eps, kernel_name)

    # 'pylint: disable=no-member
    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {"print_ir": False,
              "need_build": True,
              "name": kernel_name,
              "tensor_list": [data_x1, data_x2, data_x3, res]}

    build(sch, config)
