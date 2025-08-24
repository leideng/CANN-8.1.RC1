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
cdist
"""

import math
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpTbeImplMode


def tensor_pow(tensor, power):
    """
    tensor_pow
    """
    log_value = tbe.vlog(tensor)
    ret = tbe.vexp(tbe.vmuls(log_value, power))
    return ret


# 'pylint: disable=unused-argument,invalid-name,redefined-argument-from-local
@register_operator_compute("Cdist", op_mode="dynamic", support_fusion=True)
def cdist_compute(input_x1, input_x2, output_y, axis, p, kernel_name="cdist"):
    """
    calculating data

    Parameters
    ----------
    input_x1 : TVM tensor
        the placeholder of input_x1
    input_x2 : TVM tensor
        the placeholder of input_x2
    p : float
        norm number
    kernel_name : str
        kernel name, default value is "cdist"

    Returns
    -------
    :return: TVM tensor
        result tensor
    """
    dtype = input_x1.dtype

    if  dtype == 'float16' and \
            tbe_platform.api_check_support("te.lang.cce.vlog", "float32") and \
            tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        x1_data = tbe.cast_to(input_x1, 'float32')
        x2_data = tbe.cast_to(input_x2, 'float32')
    else:
        x1_data = input_x1
        x2_data = input_x2

    if math.isclose(p, 0.0):
        zero = tvm.const(0, dtype='float32')
        one = tvm.const(1, dtype='float32')
        elements = tbe.vcmpsel(x1_data, x2_data,
                               operation='eq', slhs=zero, srhs=one)
        res = tbe.reduce_sum(elements, axis=axis["value"])
    else:
        diff = tbe.vsub(x1_data, x2_data)
        diff = tbe.vabs(diff)

        if math.isclose(p, -1): # Note: here -1 means inf!
            scalar_zero = tvm.const(0, diff.dtype)
            mask = tbe.vcmp(diff, diff, "eq", "bit")
            diff = tbe.vsel(mask, diff, scalar_zero)
            res = tbe.reduce_max(diff, axis=axis["value"])
        elif math.isclose(p, 1):
            res = tbe.reduce_sum(diff, axis=axis["value"])
        elif math.isclose(p, 2):
            pow_x = tbe.vmul(diff, diff)
            sum_pow = tbe.reduce_sum(pow_x, axis=axis["value"])
            res = tbe.vsqrt(sum_pow, impl_mode="high_precision")
        else:
            p_scalar = tvm.const(p, dtype='float32')
            p_r_scalar = tvm.const(1 / p, dtype='float32')

            elements = tensor_pow(diff, p_scalar)
            summation = tbe.reduce_sum(elements, axis=axis["value"])
            res = tensor_pow(summation, p_r_scalar)

    if dtype == 'float16':
        ret = tbe.cast_to(res, 'float16')
    else:
        ret = res
    return ret


# 'pylint: disable=too-many-locals
@register_operator("Cdist")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def cdist(input_x1, input_x2, output_y, p=2.0, kernel_name="cdist"):
    """
    calculating data

    Parameters
    ----------
    input_x1 : dict
        shape and dtype of input1
    input_x2 : dict
        shape and dtype of input2
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    p : float
        norm number
    kernel_name : str
        kernel name, default value is "cdist"

    Returns
    -------
    :return: None
    """

    # To improve performance, the "broadcast" moves to the adaptation layer.
    # Therefore, the input_x1.shape and input_x2.shape are equal.

    para_check.check_kernel_name(kernel_name)
    x1_shape = shape_util.shape_to_list(input_x1.get("shape"))
    x1_dtype = input_x1.get("dtype")
    x2_shape = shape_util.shape_to_list(input_x2.get("shape"))
    x2_dtype = input_x2.get("dtype")

    check_data_list = ['float16', 'float32']

    para_check.check_tensor_shape_size(x1_shape)
    para_check.check_tensor_shape_size(x2_shape)

    para_check.check_dtype_rule(x1_dtype, check_data_list)
    para_check.check_dtype_rule(x2_dtype, check_data_list)

    # check data shape
    d = min(len(x1_shape), len(x2_shape))
    pos = -1
    while pos >= -d:
        if (pos != -2) and (x1_shape[pos] != x2_shape[pos]) and (x1_shape[pos] != 1) and (x2_shape[pos] != 1):
            raise RuntimeError("The shape of x1 and x2 must be equal: got %d and %d in %d dim."
                               % (x1_shape[pos], x2_shape[pos], pos))
        pos -= 1

    if p < 0 and p != -1:
        raise RuntimeError("Cdist only supports non-negative p values.")

    input_x1["rel_pos_to_reduce"] = "before"
    input_x2["rel_pos_to_reduce"] = "before"

    axis = [-1]
    input_axis = {"shape": [len(axis), ], "value": axis, "rel_pos_to_reduce": "axis"}

    # initialize data
    schedules = []
    ins = classify([input_x1, input_x2, input_axis], OpPatternMode.REDUCE, {"keepdims": False})
    tensors = []

    for (_input_x1, _input_x2, _axis) in ins:
        with tbe.compute():
            shape_x1, shape_x2 = shape_util.variable_shape([_input_x1, _input_x2, _axis], op_mode="reduce")[:2]
            data_x1 = tvm.placeholder(shape_x1, name="data_x1", dtype=x1_dtype)
            data_x2 = tvm.placeholder(shape_x2, name="data_x2", dtype=x2_dtype)

            res = cdist_compute(data_x1, data_x2, output_y, _axis, p)
            tensors.append([data_x1, data_x2, res])
        with tvm.target.cce():
            schedule = tbe.auto_schedule(res)
        schedules.append(schedule)

    # build
    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)