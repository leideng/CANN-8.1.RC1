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
soft_shrink_grad
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_attr_common import get_attr_by_cls
from impl.util.util_attr_common import SoftShrinkGradInfo
from impl.util.util_soc_common import after_v200


# 'pylint: disable=unused-argument,invalid-name,too-many-locals
@register_operator_compute("SoftShrinkGrad", op_mode="dynamic", support_fusion=True)
def soft_shrink_grad_compute(input_grad, input_x, output_y, lambd=0.5, kernel_name="soft_shrink_grad"):
    """calculating data

    Parameters
    ----------
    input_grad : TVM tensor
        the input gradient
    input_x : TVM tensor
        the placeholder of input_x
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    lambda : float
        calculation threshold
    kernel_name : str
        kernel name, default value is "soft_shrink_grad"

    Returns
    -------
    output tensor
    """
    
    input_dtype = input_x.dtype.lower()
    if input_dtype in ("float16", "bfloat16",) :
        input_x = tbe.cast_to(input_x, "float32")
        input_grad = tbe.cast_to(input_grad, "float32")
    lambd_tensor = get_attr_by_cls(lambd, SoftShrinkGradInfo.ATTR_LAMBD, "float32")
    result = tbe.vcmpsel(tbe.vabs(input_x), lambd_tensor, 'le', \
                        tvm.const(0, "float32"), input_grad)    
    mask = tbe.vcmp(input_x, input_x, 'eq')
    result = tbe.vsel(mask, result, input_grad)
    if input_dtype in ("float16",) :
        result = tbe.cast_to(result, input_dtype)
    elif input_dtype in ("bfloat16",) :
        result = tbe.round(result, input_dtype)
    return result


@register_operator("SoftShrinkGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def soft_shrink_grad(input_grad, input_x, output_y, lambd=0.5, kernel_name="soft_shrink_grad"):
    """calculating data

    Parameters
    ----------
    input_grad : TVM tensor
        the input gradient
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    lambda : float
        calculation threshold
    kernel_name : str
        kernel name, default value is "soft_shrink_grad"

    Returns
    -------
    None
    """
    shape_x = input_x.get("shape")
    shape_grad = input_grad.get("shape")
    para_check.check_shape(shape_x, param_name="input_x")
    para_check.check_shape(shape_grad, param_name="input_grad")

    check_list = ("float16", "float32", "bfloat16")
    input_dtype = input_x.get("dtype").lower()
    grad_dtype = input_grad.get("dtype").lower()
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")
    para_check.check_dtype(grad_dtype, check_list, param_name="input_grad")
    para_check.check_kernel_name(kernel_name)

    ins = classify([input_grad, input_x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (g, x) in ins:
        with tbe.compute():
            g_shape, x_shape = shape_util.variable_shape([g, x])
            tensor_g = tvm.placeholder(g_shape, grad_dtype, "tensor_g")
            tensor_x = tvm.placeholder(x_shape, input_dtype, "tensor_x")
            res = soft_shrink_grad_compute(tensor_g, tensor_x, output_y, lambd, kernel_name)
            tensors.append([tensor_g, tensor_x, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)