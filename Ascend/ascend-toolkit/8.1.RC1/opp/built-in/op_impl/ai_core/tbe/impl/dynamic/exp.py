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

exp
"""
import math
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpImplMode
from impl.util.util_common import check_op_impl_mode


# 'pylint: disable=too-many-locals,redefined-argument-from-local
def isclose(valuex, valuey, rel_tol=1e-08, abs_tol=0.0):
    """
    determines whether the values of two floating-point numbers are close or equal
    """
    return math.isclose(valuex, valuey, rel_tol=rel_tol, abs_tol=abs_tol)


# 'pylint: disable=locally-disabled,unused-argument,too-many-arguments
@register_operator_compute("Exp", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def exp_compute(input_x, output_y, base=-1.0, scale=1.0, shift=0.0,
                kernel_name="exp", impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    algorithm: exp
    calculating data's exp
    if base == -1:
       y = exp(shift + scale * x)
    if base > 0:
       y = exp((shift+scale*x)*ln(base))

    Parameters
    ----------
    input_x : TVM tensor, the placeholders of input data
    output_y : dict, shape and dtype of output, should be same shape and type as input
    base: (optional, default -1 for a value of e the base gamma
    scale: (optional, default 1) the scale alpha
    shift: (optional, default 0) the shift beta
    kernel_name : str, kernel name, default value is "exp"

    Returns
    -------
    res : the result of compute
    """
    x_dtype = input_x.dtype
    api_check = tbe_platform.api_check_support("tbe.dsl.vexp", "float32")
    if (not api_check) and (input_x.dtype == "float32"):
        input_x = tbe.cast_to(input_x, "float16")

    if api_check and input_x.dtype == "float16":
        if impl_mode == OpImplMode.HIGH_PRECISION:
            input_x = tbe.cast_to(input_x, "float32")
    input_x_dtype = input_x.dtype
    if isclose(scale, 1.0) and isclose(shift, 0.0):
        input_x_vadds = input_x
    else:
        scale_const = tvm.const(scale, dtype=input_x_dtype)
        shift_const = tvm.const(shift, dtype=input_x_dtype)
        input_x_vmuls = tbe.vmuls(input_x, scale_const)
        input_x_vadds = tbe.vadds(input_x_vmuls, shift_const)
    if base > 0:
        base_const = tvm.const(math.log(base), dtype=input_x.dtype)
        input_x_bases = tbe.vmuls(input_x_vadds, base_const)
        res = tbe.vexp(input_x_bases)
    # base is -1 value
    else:
        res = tbe.vexp(input_x_vadds)

    if input_x.dtype != x_dtype:
        res = tbe.cast_to(res, x_dtype)
    return res


@register_operator("Exp")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def exp(input_x, output_y, base=-1.0, scale=1.0, shift=0.0, kernel_name="exp", impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    algorithm: exp
        calculating data's exp
    if base == -1:
       y = exp(shift + scale * x)
    if base > 0:
       y = exp((shift+scale*x)*ln(base))

    Parameters
    ----------
    input_x : dict,shape and dtype of input, only support float16,float32,bfloat16
    output_y: dict,shape and dtype of output, should be same shape and type as input
    base: (optional, default -1 for a value of e the base gamma
    scale: (optional, default 1) the scale alpha
    shift: (optional, default 0) the shift beta
    kernel_name : str, kernel name, default value is "exp"

    Returns
    -------
    None
    """
    check_op_impl_mode(impl_mode, [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION], kernel_name)

    dtype = input_x.get("dtype")
    # input_x' dtype check, only supports fp16 and fp32
    check_list = ("float16", "float32", "bfloat16")
    input_dtype = dtype.lower()
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")
    if base <= 0 and (not isclose(base, -1.0)):
        expect_value = "strictly positive or -1"
        real_value = "base < 0 or base notequal with -1"
        error_manager_vector.raise_err_input_value_invalid(
            kernel_name, "base", expect_value, real_value)
    ins = classify([input_x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (input_x,) in ins:
        with tbe.compute():
            shape_x = shape_util.variable_shape([input_x])
            data_input = tvm.placeholder(shape_x[0], name="data_input",
                                         dtype=input_dtype)
            res = exp_compute(data_input, output_y, base, scale, shift,
                              kernel_name, impl_mode)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
