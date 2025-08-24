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
dynamic expand
"""
from functools import reduce

from impl.util.util_common import is_unknown_rank_input
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.util_common import gen_range


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
@register_operator_compute("Expand", op_mode="dynamic", support_fusion=False)
def expand_compute(x, shape):
    """
    TVM calculation process, used for fusion operation.

    Parameters
    ----------
    x: list of placeholders.
        Input data.
    shape : list or tuple.
        Number of the axis replicates.
    shape: dict.
        dict of output.
    Returns
    -------
    res
    """
    dtype = x.dtype
    compute_dtype = dtype
    if dtype in ('uint8', 'int8'):
        x = tbe.cast_to(x, 'float16')
        compute_dtype = 'float16'

    shape_in = x.shape
    _, _, shape_max = shape_util.broadcast_shapes(shape_in, shape)
    output_tensor = tbe.broadcast(x, shape_max, compute_dtype)

    if dtype in ('uint8', 'int8'):
        return tbe.cast_to(output_tensor, dtype, f1628IntegerFlag=True)

    return output_tensor


def fuse_shape(x, shape):
    """
    get broadcast shape just like tiling
    x [2, 1, 5]
    shape [6, 5, 2, 5, 5]
    return [30, 2, 5, 5]
    """
    len_diff = len(shape) - len(x)
    shape_fuse = reduce(lambda x, y: x * y, shape[:len_diff], 1)
    shape_left = [shape[index + len_diff] if shape[index + len_diff] > 1 else x[index] for index in range(len(x))]
    return [shape_fuse] + shape_left


def get_shape_adapt(input_x_shape, input_shape_shape, input_x_range, shape, kernel_name):
    """
    get shape adapt is shape and range
    """
    dims_value = input_shape_shape[0]

    if dims_value < -1:
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "shape", "shape[0] should be greater than -1")

    if dims_value == -1:
        dims_value = len(input_x_shape)

    if len(input_x_shape) > dims_value:
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "x", "shape", \
            "the dimensions of x should not be greater than shape[0]")

    shape_shape_adapt = []
    shape_range_adapt = []

    const_value = shape.get('const_value')
    if const_value:
        const_value = list(const_value)
        shape_shape_adapt = fuse_shape(input_x_shape, const_value)
        shape_range_adapt = gen_range(shape_shape_adapt)

    else:
        for shape_i, range_i in zip(input_x_shape, input_x_range):
            if shape_i == 1 or (shape_i == -1 and range_i[0] <= 1):
                shape_shape_adapt.append(-1)
                shape_range_adapt.append((1, None))
            else:
                shape_shape_adapt.append(shape_i)
                shape_range_adapt.append(range_i)

        shape_shape_adapt = [-1] + shape_shape_adapt
        shape_range_adapt = [(1, None)] + shape_range_adapt

    return [shape_shape_adapt, shape_range_adapt]


# 'pylint: disable=too-many-locals,too-many-statements,invalid-name
@register_operator("Expand")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def expand(x, shape, y, kernel_name="expand"):
    """algorithm: expand.
    The expand in tensorflow can multiple the shape of the given tensor.
    For example, tiling [a b c d] by [2] produces [a b c d a b c d].
    The expand op in TBE is compatible with the tensorflow operator BroadcastTo
    Abnormal condition:
    1. The length of shape must be equal to or less than the shape of multiples.

    Parameters
    ----------
    x : dict
        shape and dtype of input
    shape : dict
        shape and dtype of multiples
    y: dict
        dict of output.
    kernel_name : str.
        kernel name, default value is "expand".

    Returns
    -------
    None
    """
    input_x_dtype = x.get("dtype").lower()
    input_shape_dtype = shape.get("dtype").lower()

    if input_x_dtype == "bool":
        input_x_dtype = "int8"

    check_list_x_dtype = ('float16', 'float32', 'int8', 'uint8', 'int32', 'bfloat16')
    para_check.check_dtype(input_x_dtype, check_list_x_dtype, param_name="x")
    check_list_shape_dtype = ('int32', 'int64')
    para_check.check_dtype(input_shape_dtype, check_list_shape_dtype, param_name="shape")

    input_x_shape = list(x.get("shape"))
    input_shape_shape = list(shape.get("shape"))

    if is_unknown_rank_input([x, shape]):
        x, shape = [x, x] if is_unknown_rank_input(x) else [shape, shape]
    else:
        input_x_range = list(x.get("range"))
        shape_shape_adapt, shape_range_adapt = \
            get_shape_adapt(input_x_shape, input_shape_shape, input_x_range, shape, kernel_name)

        shape["shape"] = shape_shape_adapt
        shape["range"] = shape_range_adapt

    ins = classify([x, shape], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_x, _shape) in ins:
        with tbe.compute():
            shape_x, shape_shape = shape_util.variable_shape([_x, _shape])
            shape_input = tvm.placeholder(shape_shape, name="shape_input", dtype=input_shape_dtype)
            x_input = tvm.placeholder(shape_x, name="x_input", dtype=input_x_dtype)
            res = expand_compute(x_input, shape_shape)
            tensors.append([x_input, shape_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
