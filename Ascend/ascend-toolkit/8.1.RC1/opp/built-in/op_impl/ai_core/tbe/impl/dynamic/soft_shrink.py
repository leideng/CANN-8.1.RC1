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
soft_shrink
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpImplMode
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_common import check_op_impl_mode
from impl.util.util_attr_common import get_attr_by_cls
from impl.util.util_attr_common import SoftShrinkInfo
from impl.util.util_soc_common import after_v200


# 'pylint: disable=unused-argument,invalid-name,too-many-locals
def _softshrink_compute(input_x, lambd, kernel_name="soft_shrink"):
    """
    function of soft_shrink compute

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    lambd : float
        calculation threshold
    kernel_name : str
        kernel name, default value is "soft_shrink"

    Returns
    -------
    output tensor
    """

    shape = input_x.shape
    dtype = input_x.dtype
    input_x_abs = tbe.vabs(input_x)
    lambd_var = get_attr_by_cls(lambd, SoftShrinkInfo.ATTR_LAMBD, dtype)
    lambd_tensor = tbe.broadcast(lambd_var, shape)
    if after_v200() and (dtype in ["float16", "float32", "bfloat16"]):
        nan_mask = tbe.vcmp(input_x, input_x, operation='eq')
        input_x_nan = tbe.vsel(nan_mask, input_x, tvm.const(0, dtype))
        input_x_nan_abs = tbe.vabs(input_x_nan)
        mask1 = tbe.vcmp(input_x_nan_abs, lambd_var, operation='le')
        res1 = tbe.vsel(mask1, tvm.const(0, dtype), input_x_nan)
        sub_res = tbe.vsub(res1, lambd_tensor)
        mask2 = tbe.vcmp(res1, lambd_var, 'gt')
        res2 = tbe.vsel(mask2, sub_res, res1)
        add_res = tbe.vadd(res2, lambd_tensor)
        neg_lambd_var = tbe.vmuls(lambd_tensor, tvm.const(-1, dtype))
        mask3 = tbe.vcmp(res2, neg_lambd_var, 'lt')
        result = tbe.vsel(mask3, add_res, res2)
        return result
    res1 = tbe.vcmpsel(input_x_abs, lambd_var, 'le', tvm.const(0, dtype), input_x)
    sub_res = tbe.vsub(res1, lambd_tensor)
    res2 = tbe.vcmpsel(res1, lambd_var, 'gt', sub_res, res1)
    add_res = tbe.vadd(res2, lambd_tensor)
    neg_lambd_var = tbe.vmuls(lambd_tensor, tvm.const(-1, dtype))
    result = tbe.vcmpsel(res2, neg_lambd_var, 'lt', add_res, res2)
    return result


# 'pylint: disable=unused-argument,invalid-name,too-many-locals
@register_operator_compute("SoftShrink", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def soft_shrink_compute(input_x, output_y, lambd, kernel_name="soft_shrink", impl_mode="high_performance"):
    """calculating data

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    lambd : float
        calculation threshold
    kernel_name : str
        kernel name, default value is "soft_shrink"
    impl_mode : assign high_performance or high_precision

    Returns
    -------
    output tensor
    """
    check_support_flag = False
    ori_dtype = input_x.dtype
    if ori_dtype == "float16" and impl_mode == "high_precision":
        check_support_flag = True
        input_x = tbe.cast_to(input_x, "float32")
        result = _softshrink_compute(input_x, lambd, kernel_name="soft_shrink")
    else:
        if ori_dtype == "float32" and not tbe_platform.api_check_support("tbe.dsl.vcmpsel", "float32"):
            check_support_flag = True
            input_x = tbe.cast_to(input_x, "float16")
        result = _softshrink_compute(input_x, lambd, kernel_name="soft_shrink")
    if check_support_flag:
        result = tbe.cast_to(result, ori_dtype)
    return result


# 'pylint: disable=redefined-builtin
@register_operator("SoftShrink")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def soft_shrink(input_x, output_y, lambd=0.5, kernel_name="soft_shrink", impl_mode="high_performance"):
    """calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    lambd : float
        calculation threshold
    kernel_name : str
        kernel name, default value is "soft_shrink"
    impl_mode : assign high_performance or high_precision

    Returns
    -------
    None
    """
    check_op_impl_mode(impl_mode, [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION], kernel_name)
    input_dtype = input_x.get("dtype").lower()
    shape = input_x.get("shape")
    check_tuple = ("float16", "float32", "bfloat16")
    para_check.check_shape(shape, param_name="input_x")
    para_check.check_kernel_name(kernel_name)
    para_check.check_dtype(input_dtype, check_tuple, param_name="input_x")

    ins = classify([input_x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([x])
            tensor_x = tvm.placeholder(x_shape[0], input_dtype, "tensor_x")
            res = soft_shrink_compute(tensor_x, output_y, lambd, kernel_name, impl_mode)
            tensors.append((tensor_x, res))
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
