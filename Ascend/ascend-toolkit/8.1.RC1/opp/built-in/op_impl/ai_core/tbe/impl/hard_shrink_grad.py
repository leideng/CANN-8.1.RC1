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
hard_shrink_grad
"""
from tbe.dsl.api import auto_schedule
from tbe.dsl.api import build
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=unused-argument,too-many-locals,invalid-name
@register_operator_compute("hard_shrink_grad", op_mode="static", support_fusion=True)
def hard_shrink_grad_compute(gradients, features, backprops, lambd=0.5, kernel_name="hard_shrink_grad"):
    """
    calculating data

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
    lambd : int
         the lambd value for the Hardshrink formulation. Default: 0.5.

    Returns
    -------
    output tensor
    """
    dtype = features.dtype
    shape = features.shape
    one_tensor = tbe.broadcast(tvm.const(1, dtype), shape)
    zero_tensor = tbe.broadcast(tvm.const(0, dtype), shape)
    lambd_tensor = tbe.broadcast(tvm.const(lambd, dtype), shape)
    ratio = tbe.vcmpsel(tbe.vabs(features), lambd_tensor, 'le', zero_tensor, one_tensor)
    result = tbe.vmul(gradients, ratio)
    return result


# 'pylint: disable=invalid-name
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def hard_shrink_grad(gradients, features, backprops, lambd=0.5, kernel_name="hard_shrink_grad"):
    """
    calculating data

    Parameters
    ----------
    features : dict
        shape and dtype of input
    backprops : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "hard_shrink_grad"

    Returns
    -------
    None
    """
    shape_x = features.get("shape")
    shape_grad = gradients.get("shape")
    dtype = features.get("dtype")
    input_dtype = dtype.lower()

    para_check.check_shape_rule(shape_x)
    para_check.check_shape_rule(shape_grad)
    para_check.check_tensor_shape_size(shape_x)
    para_check.check_tensor_shape_size(shape_grad)
    if list(shape_x) != list(shape_grad):
        raise RuntimeError("Gradients and features must have the same shape.")
    check_list = ("float16", "float32")
    input_dtype = features.get("dtype").lower()
    grad_dtype = gradients.get("dtype").lower()
    para_check.check_dtype_rule(input_dtype, check_list)
    para_check.check_dtype_rule(grad_dtype, check_list)
    para_check.check_kernel_name(kernel_name)
    if lambd < 0:
        raise RuntimeError("Only support lambd >= 0 while lambd is {}.".format(lambd))

    data_gradients = tvm.placeholder(shape_grad, name="data_gradients", dtype=input_dtype)
    data_features = tvm.placeholder(shape_x, name="data_features", dtype=input_dtype)
    res = hard_shrink_grad_compute(data_gradients, data_features, backprops, lambd, kernel_name)
    with tvm.target.cce():
        schedule = auto_schedule(res)
    config = {"name": kernel_name,
              "tensor_list": [data_gradients, data_features, res]}

    build(schedule, config)
