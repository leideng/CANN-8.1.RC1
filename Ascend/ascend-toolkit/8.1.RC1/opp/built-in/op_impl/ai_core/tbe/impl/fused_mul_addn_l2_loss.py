#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
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
fused_mul_addn_l2_loss
"""
import functools
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector


@register_operator_compute("fused_mul_addn_l2loss", op_mode="static", support_fusion=True)
def fused_mul_addn_l2loss_compute(weight, weight_grad, const_input):
    """
    calculating data

    Parameters
    ----------
    weight : TVM tensor
        the placeholder of weight
    weight_grad : TVM tensor
        the placeholder of weight_grad
    const_input : TVM tensor
        the placeholder of const_input

    Returns
    -------
    output tensor
    """

    # cal vmul and addn
    const_input = tbe.broadcast(const_input, weight.shape)
    data_mul = tbe.vmul(weight, const_input)
    data_addn = tbe.vadd(data_mul, weight_grad)

    axis = list(range(len(weight.shape)))
    # cal l2 loss
    coeff_sqrt = tvm.const(1.0 / (2**(0.5)), dtype=weight.dtype)
    l2_loss_vmuls = tbe.vmuls(weight, coeff_sqrt)
    l2_loss_sqr = tbe.vmul(l2_loss_vmuls, l2_loss_vmuls)
    l2_loss = tbe.sum(l2_loss_sqr, axis)

    return data_addn, l2_loss


# 'pylint: disable=too-many-locals,too-many-arguments,unused-argument
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def fused_mul_addn_l2loss(input_x, input_y, input_z, output_x, output_y, kernel_name="fused_mul_addn_l2loss"):
    """
    calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input_x
    input_y : dict
        shape and dtype of input_y
    input_z : dict
        shape and dtype of input_z
    output_x : dict
        shape and dtype of first output, which should have shape (1,) and dtype
        as input
    output_y : dict
        shape and dtype of second output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "fused_mul_addn_l2loss"

    Returns
    -------
    None
    """

    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype").lower()

    shape_y = input_y.get("shape")
    dtype_y = input_y.get("dtype").lower()

    shape_z = [1] * len(shape_x)
    dtype_z = input_z.get("dtype").lower()

    check_list = ("float16", "float32")
    # check input x attr
    para_check.check_shape(shape_x, param_name="input_x")
    para_check.check_dtype(dtype_x, check_list, param_name="input_x")
    # check input y attr
    para_check.check_shape(shape_y, param_name="input_y")
    para_check.check_dtype(dtype_y, check_list, param_name="input_y")
    # check input z attr
    para_check.check_shape(shape_z, param_name="input_z")
    para_check.check_dtype(dtype_z, check_list, param_name="input_z")

    def _check_dtype_same_with_input_x(param, dtype):
        if dtype_x != dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, 'input_x', param, dtype_x, dtype)

    _check_dtype_same_with_input_x('input_y', dtype_y)
    _check_dtype_same_with_input_x('input_z', dtype_z)

    if dtype_x == "float32":
        if not tbe_platform.api_check_support("tbe.dsl.vmul", "float32"):
            error_manager_vector.raise_err_input_dtype_not_supported(kernel_name, 'input_x', ('float16',), dtype_x)

        if not tbe_platform.api_check_support("tbe.dsl.vmuls", "float32"):
            error_manager_vector.raise_err_input_dtype_not_supported(kernel_name, 'input_x', ('float16',), dtype_x)

        if not tbe_platform.api_check_support("tbe.dsl.reduce_sum", "float32"):
            error_manager_vector.raise_err_input_dtype_not_supported(kernel_name, 'input_x', ('float16',), dtype_x)

        if not tbe_platform.api_check_support("tbe.dsl.vadd", "float32"):
            error_manager_vector.raise_err_input_dtype_not_supported(kernel_name, 'input_x', ('float16',), dtype_x)

    fused_x_shape = [functools.reduce(lambda a, b: a * b, shape_x[:])]
    fused_y_shape = [functools.reduce(lambda a, b: a * b, shape_y[:])]
    fused_z_shape = [functools.reduce(lambda a, b: a * b, shape_z[:])]
    weight = tvm.placeholder(fused_x_shape, name="weight", dtype=dtype_x)
    weight_grad = tvm.placeholder(fused_y_shape, name="weight_grad", dtype=dtype_y)
    const_input = tvm.placeholder(fused_z_shape, name="const_input", dtype=dtype_z)

    res1, res2 = fused_mul_addn_l2loss_compute(weight, weight_grad, const_input)
    res_list = [res1, res2]
    with tvm.target.cce():
        sch = auto_schedule(res_list)
    config = {"name": kernel_name, "tensor_list": [weight, weight_grad, const_input] + res_list}

    build(sch, config)
