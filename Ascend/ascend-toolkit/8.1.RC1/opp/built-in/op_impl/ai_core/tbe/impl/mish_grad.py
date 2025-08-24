#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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
mish grad
"""
import functools

import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=unused-argument,too-many-locals
def mish_compute(data_grad, data_x):
    """
    algorithm: mish_grad
    calculating data's mish_grad:
            y1 = 1 / ((1 + exp(x)) ^ 2 + 1)
            x_grad = (4 * x * exp(x) * (exp(x) + 1)) * (y1 ^ 2) - 2 * y1 + 1
            output_grad = input_grad * x_grad
    Parameters
    ----------
    data_grad: TVM tensor
        the placeholder of input data, should be same shape and type as data_x
    data_x: TVM tensor
        shape and dtype of output, only support float16, float32
    Returns
    -------
    res : tvm.tensor
        the result of mish_grad
    """
    dtype = data_x.dtype
    exp_val = tbe.vexp(data_x)
    add_exp_val = tbe.vadds(exp_val, tvm.const(1, dtype))
    pow_add_exp_val = tbe.vmul(add_exp_val, add_exp_val)
    add_pow_add_exp_val = tbe.vadds(pow_add_exp_val, tvm.const(1, dtype))
    rec_add_pow_add_exp_val = tbe.vrec(add_pow_add_exp_val)
    val_1_1 = tbe.vmuls(data_x, tvm.const(4, dtype))
    val_2_1 = tbe.vmul(val_1_1, exp_val)
    val_3_1 = tbe.vmul(val_2_1, add_exp_val)
    val_4_1 = tbe.vmul(val_3_1, rec_add_pow_add_exp_val)
    val_5_1 = tbe.vmul(val_4_1, rec_add_pow_add_exp_val)
    val_1_2 = tbe.vmuls(rec_add_pow_add_exp_val, tvm.const(-2, dtype))
    val_1_3 = tbe.vadd(val_5_1, val_1_2)
    val_1_3 = tbe.vadds(val_1_3, tvm.const(1, dtype))

    res = tbe.vmul(val_1_3, data_grad)
    return res


# 'pylint: disable=unused-argument
def mish_compute_tanh(data_grad, data_x, data_tanh):
    """
    algorithm: mish_grad
    calculating data's mish_grad:
            tanhx = tanh(ln(1+exp(x)))
            x_grad = tanhx - x*(tanhx^2 - 1) / (1+exp(x)) * exp(x)
            output_grad = input_grad * x_grad
    Parameters
    ----------
    data_grad: TVM tensor
        the placeholder of input data, should be same shape and type as data_x
    data_x: dict
        shape and dtype of output, only support float16, float32
    data_tanh: dict
        shape and dtype of output, should be same shape and type as data_x
    Returns
    -------
    res : tvm.tensor
        the result of mish_grad
    """
    dtype = data_x.dtype
    exp_val = tbe.vexp(data_x)
    add_exp_val = tbe.vadds(exp_val, tvm.const(1, dtype))
    rec_add_exp_val = tbe.vrec(add_exp_val)
    pow_tanh = tbe.vmul(data_tanh, data_tanh)
    sub_pow_tanh = tbe.vadds(pow_tanh, tvm.const(-1, dtype))
    val_1 = tbe.vmul(data_x, exp_val)
    val_2 = tbe.vmul(val_1, sub_pow_tanh)
    val_3 = tbe.vmul(val_2, rec_add_exp_val)
    val_4 = tbe.vsub(data_tanh, val_3)
    res = tbe.vmul(val_4, data_grad)
    return res


# 'pylint: disable=too-many-locals
def check_params(input_grad, input_x, input_tanhx, output_grad, kernel_name):
    """
    check params
    """
    excepted_value = "equal"
    real_value = "not equal"
    input_shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()
    input_format = input_x.get("format")
    if input_tanhx is not None:
        params = (("input_grad", input_grad), ("input_tanhx", input_tanhx), ("output_grad", output_grad))
    else:
        params = (("input_grad", input_grad), ("output_grad", output_grad))
    for param_name, param in params:
        param_shape = param.get("shape")
        param_dtype = param.get("dtype").lower()
        param_format = param.get("format")
        if param_shape != input_shape:
            error_manager_vector.raise_err_input_value_invalid(
                kernel_name, "shape of input_x and {}".format(param_name), excepted_value, real_value)
        if param_dtype != input_dtype:
            error_manager_vector.raise_err_input_value_invalid(
                kernel_name, "dtype of input_x and {}".format(param_name), excepted_value, real_value)
        if param_format != input_format:
            error_manager_vector.raise_err_input_value_invalid(
                kernel_name, "format of input_x and {}".format(param_name), excepted_value, real_value)


# 'pylint: disable=too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def mish_grad(input_grad, input_x, input_tanhx, output_grad, kernel_name="mish_grad"):
    """
    algorithm: mish_grad
    calculating data's mish_grad:
            tanhx = tanh(ln(1+exp(x)))
            x_grad = tanhx - x*(tanhx^2 - 1) / (1+exp(x)) * exp(x)
            output_grad = input_grad * x_grad
    Parameters
    ----------
    input_grad:
        shape and dtype of input_grad,should be same shape and type as input_x
    input_x : dict
        shape and dtype of input_x, only support float16, float32
    input_tanhx:
        shape and dtype of input_tanhx, should be same shape and type as input_x
    output_grad: dict
        shape and dtype of output_grad, should be same shape and type as input_x
    kernel_name : str
        cce kernel name, default value is mish_grad
    Returns
    -------
    None
    """
    input_shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()
    input_format = input_x.get("format")
    check_list = ("float16", "float32")
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")
    para_check.check_shape(input_shape, param_name="input_x")
    para_check.check_format(input_format)
    check_params(input_grad, input_x, input_tanhx, output_grad, kernel_name)
    # fuse single axis
    fuseshape = [1]
    fuseshape[0] = functools.reduce(lambda x, y: x * y, input_shape)

    data_grad = tvm.placeholder(fuseshape, dtype=input_dtype, name="data_grad_input")
    data_x = tvm.placeholder(fuseshape, dtype=input_dtype, name="data_x")
    if input_tanhx is None:
        res = mish_compute(data_grad, data_x)
        tensor_list = [data_grad, data_x, res]
    else:
        data_tanh = tvm.placeholder(fuseshape, dtype=input_dtype, name="data_tanh")
        res = mish_compute_tanh(data_grad, data_x, data_tanh)
        tensor_list = [data_grad, data_x, data_tanh, res]

    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensor_list}
    tbe.cce_build_code(schedule, config)
