#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
from impl.util import util_common
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util_soc_common import is_v200


def masked_fill_int64_compute(x, mask, value):
    """
    masked_fill_int64_compute
    """
    x_shape, mask_shape, value_shape, target_shape = shape_util.unify_broadcast_shapes(
        [x.shape, mask.shape, value.shape])
    mask = tbe.cast_to(mask, "float16")
    mask = tbe.broadcast(mask, target_shape)
    mask = tbe.cast_to(mask, "bool")
    x = tbe.broadcast(x, target_shape)
    value = tbe.broadcast(value, target_shape)
    return tbe.vsel(mask, value, x)


# 'pylint: disable=invalid-name,unused-argument,unused-variable,too-many-locals
@register_operator_compute("MaskedFill", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def masked_fill_compute(x, mask, value, y, kernel_name="masked_fill"):
    """
    calculating masked_fill
    for MaskedFill, the mask is byte type, the vsel can not support byte mode.
    and it donot have high performance with vcmpsel, will use vector formula to calculate
    formula as follows:   mask is [0, 1, 0, 1]
        value_output = value * mask
        x_output = x * (1 - mask)
        output = value_output + x_output

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
    if ori_dtype == "int64":
        return masked_fill_int64_compute(x, mask, value)

    if x.dtype in ('int8',):
        x = tbe.cast_to(x, 'float16')
    target_dtype = x.dtype

    if x.dtype == 'int32':
        mask = tbe.cast_to(mask, 'float16')
    mask = tbe.cast_to(mask, x.dtype)

    if value.dtype != x.dtype:
        value = tbe.cast_to(value, x.dtype)

    # computer output shape
    x_shape, mask_shape, value_shape, target_shape = shape_util.unify_broadcast_shapes(
        [x.shape, mask.shape, value.shape])
    mask = tbe.broadcast(mask, target_shape)
    x = tbe.broadcast(x, target_shape)
    value = tbe.broadcast(value, target_shape)

    # gen mask: 1-mask
    # vcmp condition
    # 1. static case and dtype is float type, will use vcmp
    # 2. soc in v200 v220, support infnan, will use vcmp
    is_float_type = x.dtype in ('float32', 'float16', 'bfloat16')
    is_need_infnan = is_v200() or tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ("Ascend310P",)
    is_use_vcmp = (not util_common.is_unknown([y]) and x.dtype != 'int32') or (is_float_type and is_need_infnan)
    tensor_ones = tbe.broadcast(tvm.const(1, target_dtype), target_shape)
    if is_use_vcmp:
        mask_tmp = tbe.vcmp(mask, tensor_ones, 'ne', 'bit')
        y = tbe.vsel(mask_tmp, x, value)
    else:
        tensor_mask_sub = tbe.vsub(tensor_ones, mask)
        # do calculate
        tensor_mask_value = tbe.vmul(value, mask)
        tensor_x_mul = tbe.vmul(x, tensor_mask_sub)
        y = tbe.vadd(tensor_x_mul, tensor_mask_value)

    if y.dtype != ori_dtype:
        y = tbe.cast_to(y, ori_dtype)

    return y


@register_operator("MaskedFill")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
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
    # value must be scalar
    value["shape"] = [1]
    value["range"] = [[1, 1]]
    x_shape = x.get("shape")
    x_dtype = x.get("dtype")
    x_dtype_lower = x_dtype.lower()
    x_range = list(x.get("range"))

    mask_shape = mask.get("shape")
    mask_dtype = mask.get("dtype")
    mask_dtype_lower = mask_dtype.lower()
    mask_range = list(mask.get("range"))

    value_shape = value.get("shape")
    value_dtype = value.get("dtype")
    value_dtype_lower = value_dtype.lower()
    value_range = list(value.get("range"))

    # check dtype
    if x_dtype_lower == "bool":
        x_dtype_lower = "int8"

    x_dtype_list = ("bfloat16", "float16", "float32", "int8", "int32", "int64")
    para_check.check_dtype(x_dtype_lower, x_dtype_list)

    mask_dtype_list = ("bool", "int8")
    para_check.check_dtype(mask_dtype, mask_dtype_list)

    if mask_dtype_lower == "bool":
        mask_dtype_lower = "int8"

    if value_dtype_lower == "bool":
        value_dtype_lower = "int8"

    value_dtype_list = ("bfloat16", "float16", "float32", "int8", "int32", "int64")
    para_check.check_dtype(value_dtype_lower, value_dtype_list)

    # check kernel_name
    para_check.check_kernel_name(kernel_name)

    ins = classify([x, mask, value], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_x, _mask, _value) in ins:
        with tbe.compute():
            mask_dtype_lower = "bool" if x_dtype_lower == "int64" else mask_dtype_lower
            shape_x, shape_mask, shape_value = shape_util.variable_shape([_x, _mask, _value])
            data_x = tvm.placeholder(shape_x, dtype=x_dtype_lower, name="data_x")
            data_mask = tvm.placeholder(shape_mask, dtype=mask_dtype_lower, name="data_mask")
            data_value = tvm.placeholder(shape_value, dtype=value_dtype_lower, name="data_value")
            res = masked_fill_compute(data_x, data_mask, data_value, y, kernel_name)
            tensors.append([data_x, data_mask, data_value, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {
        "name": kernel_name,
        "tensor_list": tensors,
    }
    tbe.build(schedules, config)
