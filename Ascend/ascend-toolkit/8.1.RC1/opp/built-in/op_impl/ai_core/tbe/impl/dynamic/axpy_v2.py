#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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
axpy_v2
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=unused-variable,unused-argument,invalid-name
@register_operator_compute("AxpyV2", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def axpy_v2_compute(x1, x2, alpha, y, kernel_name="axpy_v2"):
    """
    calculating data

    Parameters
    ----------
    x1 : TVM tensor
        the placeholder of input_x
    x2 : TVM tensor
        the placeholder of x2
    y : dict
        dict of y, include keys(shape and dtype)
    alpha : TVM tensor
        scalar of mul-factor
    kernel_name : str
        kernel name, default value is "axpy_v2"

    Returns
    -------
    output tensor
    """
    # broadcast
    shape_x1 = shape_util.shape_to_list(x1.shape)
    shape_x2 = shape_util.shape_to_list(x2.shape)
    dtype_alpha = alpha.dtype.lower()
    dtype = x1.dtype.lower()
    precision_dtype = "float32"
    need_cast_dtype = "float16"
    # cast dtype
    if dtype in ("float32", "int32"):
        if dtype_alpha != dtype:
            alpha = tbe.cast_to(alpha, dtype)
    elif dtype == need_cast_dtype:
        x1 = tbe.cast_to(x1, precision_dtype)
        x2 = tbe.cast_to(x2, precision_dtype)
        if dtype_alpha != precision_dtype:
            alpha = tbe.cast_to(alpha, precision_dtype)

    if shape_x1 != shape_x2:
        # if shape not equal, then apply broadcast.
        shape_x, shape_y, shape_max = shape_util.broadcast_shapes(x1.shape,
                                                                  x2.shape,
                                                                  param_name_input1='x1',
                                                                  param_name_input2='x2')
        x1 = tbe.broadcast(x1, shape_max)
        x2 = tbe.broadcast(x2, shape_max)
        alpha = tbe.broadcast(alpha, shape_max)
    else:
        alpha = tbe.broadcast(alpha, x1.shape)

    if dtype == "int32":
        res = tbe.vmul(x2, alpha)
        res = tbe.vadd(x1, res)
    else:
        res = tbe.vmla(x2, alpha, x1)
        if dtype == need_cast_dtype:
            res = tbe.cast_to(res, dtype)
    return res


# 'pylint: disable=too-many-locals,invalid-name
@register_operator("AxpyV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def axpy_v2(x1, x2, alpha, y, kernel_name="axpy_v2"):
    """
    calculating data of axpy

    Parameters
    ----------
    x1 : dict
        shape and dtype of input_x
    x2 : dict
        shape and dtype of input_y
    alpha : dict
        shape and dtype of alpha
        scalar apply to input_y:input_y*alpha
    y : dict
        shape and dtype of output, should be same shape and type as input

    kernel_name : str
        kernel name, default value is "axpy"

    Returns
    -------
    None
    """
    # check kernel name
    para_check.check_kernel_name(kernel_name)
    dtype_x1 = x1.get("dtype").lower()
    dtype_x2 = x2.get("dtype").lower()
    alpha_dtype = alpha.get("dtype").lower()
    # check dtype
    dtype_list0 = ("bfloat16", "float16", "float32", "int32")

    para_check.check_dtype(dtype_x1, dtype_list0)
    para_check.check_dtype(dtype_x2, dtype_list0)
    para_check.check_dtype(alpha_dtype, dtype_list0)
    para_check.check_elewise_shape_range([x1, x2])
    ins = classify([x1, x2, alpha], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for(ins_x1, ins_x2, ins_alpha) in ins:
        with tbe.compute():
            x_shape, y_shape, alpha_shape = shape_util.variable_shape([ins_x1, ins_x2, ins_alpha])
            data1 = tvm.placeholder(x_shape, dtype=dtype_x1, name="data1")
            data2 = tvm.placeholder(y_shape, dtype=dtype_x2, name="data2")
            alpha_input = tvm.placeholder(alpha_shape, dtype=alpha_dtype, name="alpha_input")
            res = axpy_v2_compute(data1, data2, alpha_input, y, kernel_name)
            tensors.append([data1, data2, alpha_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
