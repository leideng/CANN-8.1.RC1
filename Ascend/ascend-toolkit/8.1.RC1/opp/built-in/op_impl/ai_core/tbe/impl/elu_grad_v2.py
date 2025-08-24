#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2019 Huawei Technologies Co., Ltd
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
elu_grad_v2
"""
import operator
from tbe.dsl.api import auto_schedule
from tbe.dsl.api import build
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import shape_util


# 'pylint: disable=unused-argument,too-many-locals,invalid-name
@register_operator_compute("elu_grad_v2", op_mode="static", support_fusion=True)
def elu_grad_v2_compute(grads, activations, y, alpha=1.0):
    """
    calculating data

    Parameters
    ----------
    grads : TVM tensor, input parameter
    activations : TVM tensor, input parameter, the value of elu()
    alpha: scalar parameter, default value = 1.0
    kernel_name : str
    kernel name, default value is "elu_grad_v2"
    Returns
    f'(x)={ 1            when x>0
            f(x)+alpha   when x<=0
    -------
    output tensor
    """
    dtype = grads.dtype
    shape = grads.shape
    if dtype.lower() == "float16":
        grads = tbe.cast_to(grads, "float32")
        activations = tbe.cast_to(activations, "float32")

    scalar_param_zero = tvm.const(0.0, grads.dtype)
    scalar_param_alpha = tvm.const(alpha, grads.dtype)
    scalar_param_one = tvm.const(1.0, grads.dtype)

    tensor_scalar_one = tbe.broadcast(scalar_param_one, shape)
    tensor_scalar_zero = tbe.broadcast(scalar_param_zero, shape)
    tensor_ret_x_ge_zero = tbe.vadds(activations, scalar_param_alpha)

    mid_ret = tbe.vcmpsel(activations, tensor_scalar_zero, 'le', tensor_ret_x_ge_zero, tensor_scalar_one)
    res = tbe.vmul(mid_ret, grads)

    if dtype.lower() == "float16":
        res = tbe.cast_to(res, "float16")

    return res


# 'pylint: disable=unused-argument
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def elu_grad_v2(grads, activations, y, alpha=1.0, kernel_name="elu_grad_v2"):
    """
    grads : shape and dtype
    activations : shape  dtype
    alpha: scalar parameter, default value = 1.0
    kernel name, default value is "elu_grad_v2"
    """
    shape_gradient = grads.get("shape")
    shape_activation = activations.get("shape")
    dtype_gradient = grads.get("dtype")
    dtype_activation = activations.get("dtype")
    para_check.check_kernel_name(kernel_name)

    para_check.check_shape(shape_gradient)
    para_check.check_shape(shape_activation)
    if alpha < 0:
        raise RuntimeError("Alpha couldn't less than zero.")
    if not operator.eq(shape_gradient, shape_activation):
        raise RuntimeError("All input shape must be equal.")
    shape_gradient, _ = shape_util.refine_shape_axes(shape_gradient, [])
    shape_activation, _ = shape_util.refine_shape_axes(shape_activation, [])

    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_gradient, check_list)
    para_check.check_dtype(dtype_activation, check_list)
    if dtype_gradient.lower() != dtype_activation.lower():
        raise RuntimeError("All input dtype must be same.")

    dtype = dtype_gradient.lower()
    data_gradient = tvm.placeholder(shape_gradient, dtype=dtype, name="data_gradient")
    data_activation = tvm.placeholder(shape_activation, dtype=dtype, name="data_activation")
    res = elu_grad_v2_compute(data_gradient, data_activation, y, alpha)

    with tvm.target.cce():
        auto_sch = auto_schedule(res)

    config = {"name": kernel_name,
              "print_ir": False,
              "tensor_list": [data_gradient, data_activation, res]}
    build(auto_sch, config)
