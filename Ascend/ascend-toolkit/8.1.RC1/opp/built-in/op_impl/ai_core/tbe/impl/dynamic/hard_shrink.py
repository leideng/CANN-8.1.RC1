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
hard_shrink
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_attr_common import HardShrinkAttrInfo
from impl.util.util_attr_common import get_attr_by_cls
from impl.util.util_soc_common import after_v200


# 'pylint: disable=invalid-name,unused-argument
@register_operator_compute("HardShrink", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def hard_shrink_compute(input_x, output_y, lambd, kernel_name="hard_shrink"):
    """
    Calculating function
    HardShrink is defined as:
        H(x) = {
                x, if |x| > lambd,
                0, otherwise
                }
    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    lambda : float
        calculation threshold
    kernel_name : str
        kernel name, default value is "hard_shrink"

    Returns
    -------
    output tensor
    """
    shape = input_x.shape
    dtype = input_x.dtype
    if dtype == "float32" and not tbe_platform.api_check_support("te.lang.cce.vcmpsel", "float32"):
        input_x = tbe.cast_to(input_x, "float16")
    input_x_abs = tbe.vabs(input_x)
    
    lambd_scalar = get_attr_by_cls(lambd, HardShrinkAttrInfo.ATTR_LAMBD, dtype)
    lambd_tensor = tbe.broadcast(lambd_scalar, shape)
    zero_tensor = tbe.broadcast(tvm.const(0, dtype), shape)
    if after_v200() and (dtype in ["float16", "float32", "bfloat16"]):
        mask = tbe.vcmp(input_x_abs, lambd_tensor, operation='le')
        result = tbe.vsel(mask, zero_tensor, input_x)
        result = tbe.cast_to(result, dtype)
        return result
    result = tbe.vcmpsel(input_x_abs, lambd_tensor, 'le', zero_tensor, input_x)
    result = tbe.cast_to(result, dtype)
    return result


@register_operator("HardShrink")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def hard_shrink(input_x, output_y, lambd=0.5, kernel_name="hard_shrink"):
    """
    A activation funciton which applies the Hard Shrinkage(HardShrink) element-wise

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    lambda : float
        calculation threshold for the HardShrink formulation. Default: 0.5
    kernel_name : str
        kernel name, default value is "hard_shrink"

    Returns
    -------
    None
    """
    check_tuple = ("float16", "float32", "bfloat16")
    input_dtype = input_x.get("dtype").lower()
    para_check.check_dtype(input_dtype, check_tuple, param_name="input_x")
    ins = classify([input_x], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_x,) in ins:
        with tbe.compute():
            input_shape = shape_util.variable_shape([_x])
            data_input = tvm.placeholder(input_shape[0], name="data_input", dtype=input_dtype)
            res = hard_shrink_compute(data_input, output_y, lambd, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)