#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.
You may not use this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

is_neg_inf
"""

from __future__ import absolute_import

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


@register_operator_compute("IsNegInf", op_mode="dynamic", support_fusion=True)
def is_neg_inf_compute(x, y, kernel_name="is_neg_inf"):
    """
    Parameters
    ----------
    x: the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name

    Returns
    -------
    res : result of is_neg_inf
    """
    if x.dtype.lower() == 'bfloat16':
        x = tbe.cast_to(x, 'float32')

    inp_dtype = x.dtype
    inp_shape = x.shape

    neg_inf = tbe.broadcast(tvm.const(float('-inf'), inp_dtype), inp_shape)
    neg_res = tbe.vcmp(
        lhs=x, 
        rhs=neg_inf,
        operation='eq',
        mode='bool'
    )
    return neg_res


@register_operator("IsNegInf")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def is_neg_inf(x, y, kernel_name="is_neg_inf"):
    """
    Parameters
    ----------
    Algorithm: is_neg_inf

    Parameters:

    x: dynamic input, include shape, dtype and range

    y: the dict of output

    kernel_name: kernel name, must be string, default value is "is_neg_inf".

    Returns
    -------
    None
    """

    # check input tensor data_type
    dtype_x = x.get("dtype").lower()
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(dtype_x, check_list, param_name="x")

    ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (x,) in ins:
        with tbe.compute():
            shape_x = shape_util.variable_shape([x])
            input_data = tvm.placeholder(shape_x[0], name="input_data", dtype=dtype_x)
            res = is_neg_inf_compute(input_data, y, kernel_name)

            tensors.append([input_data, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {
        "name": kernel_name, 
        "tensor_list": tensors,
        "bool_storage_as_1bit": False
        }

    tbe.build(schedules, config)
