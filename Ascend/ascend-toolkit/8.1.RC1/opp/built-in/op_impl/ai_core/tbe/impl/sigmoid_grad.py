#!usr/bin/env python
# -*- coding:utf-8 -*-
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
sigmoid_grad
"""
import functools
import operator
from tbe.dsl.api import auto_schedule
from tbe.dsl.api import build
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=too-many-arguments,unused-argument,invalid-name
@register_operator_compute("sigmoid_grad", op_mode="static", support_fusion=True)
def sigmoid_grad_compute(x, y, z, kernel_name="sigmoid_grad"):
    """
    algorithm : sigmoid grad compute

    sigmoid_grad = (sigmoid - sigmoid*sigmoid)*grad

    Parameters:
    ----------
    x : a tensor of input data

    y : a tensor of grad

    z : output dict

    kernel_name : cce kernel name, default value is "sigmoid_grad"

    Returns
    -------
    a tenosr
    """
    dtype = x.dtype.lower()
    cast_support = tbe_platform.api_check_support("tbe.dsl.cast_to", "f322f16")
    if not cast_support:
        para_check.check_dtype(dtype, ("float16", ), param_name="x")
    vmul_support = tbe_platform.api_check_support("tbe.dsl.vmul", "float32")
    vsub_support = tbe_platform.api_check_support("tbe.dsl.vsub", "float32")
    if dtype == "float16":
        x = tbe.cast_to(x, "float32")
        y = tbe.cast_to(y, "float32")
    sigmoid_square = tbe.vmul(x, x)
    if dtype == "float32" and not vmul_support:
        sigmoid_square = tbe.cast_to(sigmoid_square, "float16")
    tensor_sub = tbe.vsub(x, sigmoid_square)
    if dtype == "float32" and not vsub_support:
        tensor_sub = tbe.cast_to(tensor_sub, "float16")
    res = tbe.vmul(tensor_sub, y)
    if dtype == "float16":
        res = tbe.cast_to(res, "float16")
    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def sigmoid_grad(x, y, z, kernel_name="sigmoid_grad"):
    """
    do sigmoid grad

    sigmoid_grad = (sigmoid - sigmoid*sigmoid)*grad

    Parameters:
    ----------
    x : dictionary shape of sigmoid input

    y : dictionary shape of grad

    z: dictionary output

    kernel_name : cce kernel name, default value is "sigmoid_grad_cce"

    Returns
    -------
    None
    """
    shape_sig = x.get("shape")
    shape_d = y.get("shape")
    dtype = x.get("dtype")
    dtype_y = y.get("dtype")
    if dtype != dtype_y:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, 'x', 'y', dtype, dtype_y)
    if not operator.eq(list(shape_sig), list(shape_d)):
        error_manager_vector.raise_err_inputs_shape_not_equal(kernel_name, 'x', 'y', shape_sig, shape_d, shape_sig)
    para_check.check_shape(shape_sig, param_name="x")
    input_dtype = dtype.lower()
    para_check.check_dtype(input_dtype, ("float16", "float32"), param_name="x")

    shape_sig = [functools.reduce(lambda x, y: x * y, shape_sig[:])]
    input_sigmoid = tvm.placeholder(shape_sig, name="input_sigmoid", dtype=input_dtype)
    input_grad = tvm.placeholder(shape_sig, name="input_grad", dtype=input_dtype)

    with tvm.target.cce():
        res = sigmoid_grad_compute(input_sigmoid, input_grad, z, kernel_name)
        auto_sch = auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [input_sigmoid, input_grad, res]}

    build(auto_sch, config)
