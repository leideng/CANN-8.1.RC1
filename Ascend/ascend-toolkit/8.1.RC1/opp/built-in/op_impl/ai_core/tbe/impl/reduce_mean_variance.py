#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
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
reduce_mean_d
"""

import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util
from tbe import tvm


# 'pylint: disable=locally-disabled,unused-argument,invalid-name,too-many-arguments
@tbe_platform.fusion_manager.fusion_manager.register("reduce_mean_variance")
def reduce_mean_variance_compute(x, mean, square_sum, axes, keep_dims, kernel_name="reduce_mean_variance"):
    """
    Reduce a tensor on a certain axes based on mean and square_sum.

    Parameters:
    ----------
    x : dict
        shape and dtype of input
    mean: dict
        shape and dtype of output
    square_sum: dict
        shape and dtype of output
    axes : int, list, tuple
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(x), rank(x)).
    keep_dims : bool
        if true, retains reduced dimensions with length 1,
        default value is None.
    kernel_name : str
        cce kernel name, default value is reduce_mean_square_sum

    Returns
    -------
    mean: TVM tensor
    square_sum: TVM tensor
    """
    shape = shape_util.shape_to_list(x.shape)
    reduce_elts = 1.0
    for i in axes:
        reduce_elts *= shape[i]
    mean_cof = reduce_elts**(-1)
    x = tbe.cast_to(x, "float32")
    mean_muls = tbe.vmuls(x, mean_cof)

    x_square = tbe.vmul(x, x)
    x_square = tbe.vmuls(x_square, mean_cof)
    mean, square_sum = tbe.tuple_sum([mean_muls, x_square], axis=axes, keepdims=keep_dims)

    return mean, square_sum


# 'pylint: disable=locally-disabled,unused-argument,invalid-name,too-many-arguments
def check_supported(x, mean, square_sum, axes, keep_dims=True, kernel_name="reduce_mean_variance"):
    """
    verify the types of reduce_mean_variance supported by tbe
    """
    support_shapes = [[224, 224, 160, 32], [112, 112, 80, 64], [56, 56, 40, 128], [28, 28, 20, 256], [14, 14, 10, 320],
                      [7, 7, 5, 320]]
    shape_x = x.get("ori_shape")
    temp_shape = list(shape_x)

    if len(axes) != 3:
        reason = "when axes don't equal to 3, ai_core can not support"
        return False, reason

    format_x = x.get("format")
    if format_x not in ("NDC1HWC0", "NDHWC"):
        reason = "when format isn't in ('NDC1HWC0', 'NDHWC'), ai_core can not support"
        return False, reason

    if len(temp_shape) != 5:
        reason = "when length of ori_shape don't equal to 5, ai_core can not support"
        return False, reason

    if not tbe_platform.api_check_support("tik.vbi"):
        reason = "reduce_mean_variance can't support current platform"
        return False, reason

    del temp_shape[0]
    if temp_shape in support_shapes:
        return True, ""

    reason = "when ori_shape isn't in white list, ai_core can not support"
    return False, reason


# 'pylint: disable=locally-disabled,unused-argument,invalid-name,too-many-arguments
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            (para_check.REQUIRED_ATTR_INT, para_check.REQUIRED_ATTR_LIST_INT),
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def reduce_mean_variance(x, mean, square_sum, axes, keep_dims=True, kernel_name="reduce_mean_variance"):
    """
    Reduce a tensor on a certain axes based on mean and square_sum.

    Parameters:
    ----------
    x : dict
        shape and dtype of input
    mean: dict
        shape and dtype of output
    square_sum: dict
        shape and dtype of output
    axes : int, list, tuple
        The dimensions to reduce. If None (the default), reduces all dimensions.
        Must be in the range [-rank(x), rank(x)).
    keep_dims : bool
        if true, retains reduced dimensions with length 1,
        default value is None.
    kernel_name : str
        cce kernel name, default value is reduce_mean_square_sum

    Returns
    -------
    None
    """
    shape_x = x.get("shape")
    format_x = x.get("format")
    dtype_x = x.get("dtype").lower()
    axis = list(axes)
    if format_x in ("NDC1HWC0", ):
        axis = [1, 3, 4]

    data_input = tvm.placeholder(shape_x, name="data_input", dtype=dtype_x)
    res_mean, res_square_sum = reduce_mean_variance_compute(data_input, mean, square_sum, axis, keep_dims)

    with tvm.target.cce():
        sch = tbe.auto_schedule([res_mean, res_square_sum])
    config = {"print_ir": False, "name": kernel_name, "tensor_list": [data_input, res_mean, res_square_sum]}
    tbe.cce_build_code(sch, config)
