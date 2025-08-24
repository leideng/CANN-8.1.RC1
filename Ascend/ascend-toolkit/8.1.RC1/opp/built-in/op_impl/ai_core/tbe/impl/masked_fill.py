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
masked_fill
"""
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm


# 'pylint: disable=invalid-name,unused-argument,unused-variable,too-many-locals
@register_operator_compute("masked_fill", op_mode="static", support_fusion=True)
def masked_fill_compute(x, mask, value, y, kernel_name="masked_fill"):
    """
    calculating masked_fill
    :param x: TVM tensor
                   the output of previous layer
    :param mask: TVM tensor
                    mask dtype is bool
    :param value: scalar or TVM tensor
                    the value to fill in with
    :param kernel_name: str
                    kernel name, default value is "masked_fill"
    :return:y
            TVM tensor
    """

    ori_dtype = x.dtype
    if x.dtype == 'int8':
        x = tbe.cast_to(x, 'float16')

    x_shape = shape_util.shape_to_list(x.shape)
    mask_shape = shape_util.shape_to_list(mask.shape)
    # computer output shape
    x_shape, mask_shape, target_shape = shape_util.broadcast_shapes(x_shape, mask_shape)
    target_dtype = x.dtype

    if x_shape != target_shape:
        x = tbe.broadcast(x, target_shape)

    if x.dtype == 'int32':
        mask = tbe.cast_to(mask, 'float16')
    mask = tbe.cast_to(mask, x.dtype)
    if mask_shape != target_shape:
        mask = tbe.broadcast(mask, target_shape)

    if value.dtype != x.dtype:
        value = tbe.cast_to(value, x.dtype)
    value = tbe.broadcast(value, target_shape)

    tensor_ones = tbe.broadcast(tvm.const(1, target_dtype), target_shape)

    if x.dtype == 'int32':
        y = masked_fill_compute_int32(x, mask, value, tensor_ones)
        return y

    y = tbe.vcmpsel(mask, tensor_ones, 'ne', x, value)

    if y.dtype != ori_dtype:
        y = tbe.cast_to(y, ori_dtype)

    return y


def masked_fill_compute_int32(x, mask, value, tensor_ones):
    """
    calculating masked_fill dtype is int32
    :param x: TVM tensor
                   the output of previous layer
    :param mask: TVM tensor
                    mask dtype is bool
    :param value: scalar or TVM tensor
                    the value to fill in with
    :param tensor_ones: TVM tensor
                    the tensor of 1
    :return:y
            TVM tensor
    """
    tensor_mask_value = tbe.vmul(mask, value)  # mask * value
    tensor_mask_mul = tbe.vsub(tensor_ones, mask)  # tensor_ones - mask
    tensor_x_mul = tbe.vmul(x, tensor_mask_mul)  # x*[tensor_ones-mask]
    y = tbe.vadd(tensor_x_mul, tensor_mask_value)  # x*[tensor_ones-mask] + mask*value

    return y


# @register_operator("MaskedFill")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def masked_fill(x, mask, value, y, kernel_name="masked_fill"):
    """
    :param x: dict
                    shape and dtype of tensor x input
    :param mask: dict
                    shape and dtype of tensor mask,
                    can be boardcast as shape as x
    :param value: dict
                    shape and dtype of value
    :param y: dict
                    the output of masked _fill
    :param kernel_name: str
                      kernel name, default value is "masked _fill"
    :return: none
    """

    x_shape = x.get("shape")
    x_dtype = x.get("dtype")
    x_dtype_lower = x_dtype.lower()

    mask_shape = mask.get("shape")
    mask_dtype = mask.get("dtype")

    value_shape = value.get("shape")
    value_dtype = value.get("dtype")
    value_dtype_lower = value_dtype.lower()

    # check dtype
    if x_dtype_lower == "bool":
        x_dtype_lower = "int8"

    x_dtype_list = ("float16", "float32", "int8", "int32")
    para_check.check_dtype(x_dtype_lower, x_dtype_list)

    mask_dtype_list = ("bool", "int8")
    para_check.check_dtype(mask_dtype, mask_dtype_list)

    if mask_dtype == "bool":
        mask_dtype = "int8"

    if value_dtype_lower == "bool":
        value_dtype_lower = "int8"

    value_dtype_list = ("float16", "float32", "int8", "int32")
    para_check.check_dtype(value_dtype_lower, value_dtype_list)

    # check shape
    para_check.check_shape(x_shape)
    para_check.check_shape(mask_shape)
    para_check.check_shape(value_shape)

    # check boardcast shape
    x_shape, mask_shape, out_shape = shape_util.broadcast_shapes(x_shape, mask_shape)
    para_check.check_shape(out_shape)

    # check kernel_name
    para_check.check_kernel_name(kernel_name)

    pos_mask_shape = tuple([1] * (len(x_shape) - len(mask_shape))) + tuple(mask_shape)
    pos_value = tuple([1] * (len(x_shape) - len(value_shape))) + tuple(value_shape)
    data_x = tvm.placeholder(x_shape, dtype=x_dtype_lower, name="data_x")

    data_mask = tvm.placeholder(pos_mask_shape, dtype=mask_dtype, name="data_mask")

    data_value = tvm.placeholder(pos_value, dtype=value_dtype_lower, name="data_value")

    y = masked_fill_compute(data_x, data_mask, data_value, kernel_name)

    with tvm.target.cce():
        schedule = auto_schedule(y)

    config = {"name": kernel_name,
              "tensor_list": [data_x, data_mask, data_value, y]}
    build(schedule, config)
