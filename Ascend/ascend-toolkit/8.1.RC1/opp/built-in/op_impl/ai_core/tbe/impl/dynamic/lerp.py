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
lerp
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import OpImplMode
from impl.util.util_common import check_op_impl_mode


@register_operator_compute("Lerp", op_mode="dynamic", support_fusion=True, support_bfp16=True)
# 'pylint: disable=unused-argument
def lerp_compute(start, end, weight, y, kernel_name="lerp", impl_mode=OpImplMode.HIGH_PRECISION):
    """
    Compute

    Parameters
    ----------
    start: dict
        data of input
        datatype suports float32,float16
    end: dict
        data of input
        datatype suports float32,float16
    weight: dict
        data of input
        datatype suports float32,float16
    y: dict
        data of output
    kernel_name: str
        the name of the operator
    Returns
    -------
    None
    """
    x_dtype = start.dtype
    if x_dtype == "float16" and impl_mode == OpImplMode.HIGH_PRECISION:
        start = tbe.cast_to(start, "float32")
        end = tbe.cast_to(end, "float32")
        weight = tbe.cast_to(weight, "float32")

    # Broadcast the shape of start, end and weight
    shape_x, shape_y, shape_tmp = shape_util.broadcast_shapes(start.shape, end.shape)
    shape_tmp, shape_z, shape_max = shape_util.broadcast_shapes(shape_tmp, weight.shape)
    start = tbe.broadcast(start, shape_max)
    end = tbe.broadcast(end, shape_max)
    weight = tbe.broadcast(weight, shape_max)

    half_scalar = tvm.const(0.5, dtype=weight.dtype)
    one_tensor = tbe.broadcast(tvm.const(1.0, dtype=weight.dtype), weight.shape)
    re_weight = tbe.vsub(one_tensor, weight)
    mask = tbe.vcmp(weight, half_scalar, 'lt')

    sub_val = tbe.vsub(end, start)
    mul_val = tbe.vmul(weight, sub_val)
    re_mul_val = tbe.vmul(re_weight, sub_val)

    left_res = tbe.vadd(start, mul_val)
    right_res = tbe.vsub(end, re_mul_val)

    res = tbe.vsel(mask, left_res, right_res)

    if res.dtype != x_dtype:
        res = tbe.cast_to(res, x_dtype)

    return res


# 'pylint: disable=too-many-locals
@register_operator("Lerp")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def lerp(start, end, weight, y, kernel_name="lerp", impl_mode=OpImplMode.HIGH_PRECISION):
    """
    Lerp

    Parameters
    ----------
    start: dict
        data of input
        datatype suports float32,float16
    end: dict
        data of input
        datatype suports float32,float16
    weight: dict
        data of input
        datatype suports float32,float16
    y: dict
        data of output
    kernel_name: str
        the name of the operator
    Returns
    -------
    None
    """
    check_op_impl_mode(
        impl_mode, [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION], kernel_name)
    dtype = start.get("dtype")
    input_dtype = dtype.lower()
    para_check.check_kernel_name(kernel_name)
    check_tuple = ("bfloat16", "float16", "float32")
    para_check.check_dtype_rule(input_dtype, check_tuple)
    ins = classify([start, end, weight], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for(_x, _y, _z) in ins:
        with tbe.compute():
            shape_x, shape_y, shape_z = shape_util.variable_shape([_x, _y, _z])
            data_x = tvm.placeholder(shape_x, name="data_1", dtype=input_dtype)
            data_y = tvm.placeholder(shape_y, name="data_2", dtype=input_dtype)
            data_z = tvm.placeholder(shape_z, name="data_3", dtype=input_dtype)
            res = lerp_compute(data_x, data_y, data_z, y, kernel_name, impl_mode)
            tensors.append([data_x, data_y, data_z, res])
        with tvm.target.cce():
            schedule = tbe.auto_schedule(res)
        schedules.append(schedule)
    config = {"name": kernel_name,
              "tensor_list": tensors}

    tbe.build(schedules, config)
