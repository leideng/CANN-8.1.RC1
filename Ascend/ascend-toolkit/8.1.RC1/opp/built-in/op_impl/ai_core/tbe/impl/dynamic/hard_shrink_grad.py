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
hard_shrink_grad
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
from impl.util.util_attr_common import HardShrinkGradAttrInfo
from impl.util.util_attr_common import get_attr_by_cls
from impl.util.util_soc_common import after_v200


# 'pylint: disable=invalid-name,unused-argument
@register_operator_compute("HardShrinkGrad", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def hard_shrink_grad_compute(gradients, features, backprops, lambd=0.5, kernel_name="hard_shrink_grad"):
    """
    Calculating function
    HardShrinkGrad is defined as:
        dH(x) = {
                1, if |x| > lambd,
                0, otherwise
                }

    Parameters
    ----------
    gradients : TVM tensor
        the placeholder gradients.
    features : TVM tensor
        the placeholder of features
    backprops : dict
        dict of backprops, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "hard_shrink_grad"
    lambd : float
         the lambd value for the Hardshrink formulation. Default: 0.5.

    Returns
    -------
    output tensor
    """
    ori_dtype = features.dtype
    dtype = features.dtype
    lambd_scalar = get_attr_by_cls(lambd, HardShrinkGradAttrInfo.ATTR_LAMBD, dtype)
    if ori_dtype == "float32" and not tbe_platform.api_check_support("te.lang.cce.vsel", "float32"):
        gradients = tbe.cast_to(gradients, "float16")
        features = tbe.cast_to(features, "float16")
    features_abs = tbe.vabs(features)
    if after_v200() and (dtype in ["float16", "float32"]):
        mask = tbe.vcmp(features_abs, lambd_scalar, operation='le')
        ratio = tbe.vsel(mask, tvm.const(0, "float16"), tvm.const(1, "float16"))
        if dtype != "float16":
            ratio = tbe.cast_to(ratio, dtype)
        result = tbe.vmul(gradients, ratio)
        if ori_dtype != dtype:
            result = tbe.cast_to(result, ori_dtype)
        return result
    ratio = tbe.vcmpsel(features_abs, lambd_scalar, 'le', tvm.const(0, dtype), tvm.const(1, dtype))
    result = tbe.vmul(gradients, ratio)
    if ori_dtype != dtype:
        result = tbe.cast_to(result, ori_dtype)
    return result


@register_operator("HardShrinkGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, 
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,  
                            para_check.KERNEL_NAME)
def hard_shrink_grad(gradients, features, backprops, lambd=0.5, kernel_name="hard_shrink_grad"):
    """
    Gradients of a activation funciton which applies the Hard Shrinkage(HardShrink) element-wise

    Parameters
    ----------
    gradients : dict
        shape and dtype of input
    features : dict
        shape and dtype of input
    backprops : dict
        shape and dtype of output, should be same shape and type as input
    lambda : float
        calculation threshold for the HardShrink formulation. Default: 0.5
    kernel_name : str
        kernel name, default value is "hard_shrink_grad"

    Returns
    -------
    None
    """
    gradients_dtype = gradients.get("dtype").lower()
    features_dtype = features.get("dtype").lower()
    check_tuple = ("float16", "float32", "bfloat16")
    para_check.check_dtype_rule(gradients_dtype, check_tuple)
    para_check.check_dtype_rule(features_dtype, check_tuple)
    para_check.check_kernel_name(kernel_name)

    ins = classify([gradients, features], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (grad, x) in ins:
        with tbe.compute():
            grad_shape, x_shape = shape_util.variable_shape([grad, x])
            grad_input = tvm.placeholder(grad_shape, name="grad_input", dtype=gradients_dtype)
            x_input = tvm.placeholder(x_shape, name="x_input", dtype=features_dtype)
            result = hard_shrink_grad_compute(grad_input, x_input, backprops, lambd, kernel_name)
            tensors.append([grad_input, x_input, result])
        with tvm.target.cce():
            sch = tbe.auto_schedule(result)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)