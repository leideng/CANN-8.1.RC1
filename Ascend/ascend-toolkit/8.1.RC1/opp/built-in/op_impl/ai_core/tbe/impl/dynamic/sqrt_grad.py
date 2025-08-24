#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

sqrt_grad
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name,too-many-locals
@register_operator_compute("SqrtGrad", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def sqrt_grad_compute(x, dx, out, kernel_name="sqrt_grad"):
    """
    algorithm: sqrt_grad_compute
    output_grad = input_grad/(2*input)

    Parameters
    ----------
    x: a tensor of input data

    dx : a tensor of grad

    Returns
    -------
    output data

    """

    dtype = x.dtype.lower()
    mul_support = tbe_platform.api_check_support("tbe.dsl.vmuls", "float32")
    if dtype == "float32" and not mul_support:
        error_detail = "not support dtype(float32) on this platform!"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "x", 'dx', error_detail)
    const_val_half = tvm.const(0.5, dtype)
    div_val = tbe.vdiv(dx, x)
    res = tbe.vmuls(div_val, const_val_half)
    return res


# 'pylint: disable=redefined-argument-from-local
@register_operator("SqrtGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def sqrt_grad(x, dx, out, kernel_name="sqrt_grad"):
    """
    algorithm: sqrt_grad_cce

    Parameters
    ----------
    x : dict of data: dict

    dx : dict of data_grad: dict

    out : dict of output: dict

    kernel_name : cce kernel name, default value is "sqrt_grad": str

    Returns
    -------
    None

    """
    x_dtype = x.get("dtype").lower()
    dx_dtype = dx.get("dtype").lower()
    shape_x = x.get("shape")
    shape_dx = dx.get("shape")
    check_list = ("bfloat16", "float16", "float32")
    para_check.check_dtype(x_dtype, check_list, param_name="x")
    para_check.check_dtype(dx_dtype, check_list, param_name="dx")
    if len(shape_x) != len(shape_dx):
        error_detail = "The shapes of two input parameters are not match for dynamic sqrt_grad"
        error_manager_vector.raise_err_two_input_dtype_invalid("sqrt_grad", "x", "dx", error_detail)
    if x_dtype != dx_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "x", "dx",
                                                              x_dtype, dx_dtype)
    ins = classify([x, dx], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (x, dx) in ins:
        with tbe.compute():
            x_shape, dx_shape = shape_util.variable_shape([x, dx])
            tensor_x = tvm.placeholder(x_shape, x_dtype, "tensor_x")
            tensor_dx = tvm.placeholder(dx_shape, dx_dtype, "tensor_dx")
            res = sqrt_grad_compute(tensor_x, tensor_dx, out, kernel_name)
            tensors.append([tensor_x, tensor_dx, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
