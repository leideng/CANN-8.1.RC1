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
log1p
"""
import functools

import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    This class for Constant.
    """
    # `define a scalar, value = -1`
    SCALAR_NEG_ONE = -1.0
    # `define a scalar, value = 1`
    SCALAR_ONE = 1.0


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals
@register_operator_compute("log1p", op_mode="static", support_fusion=True)
def log1p_compute(input_x, output_y, kernel_name="log1p"):
    """
    algorithm: log1p
    calculating data's log1p, y = log(x + 1)
    in cloud scene, for all inputs :
    y = log(x + 1)
    in mini scene :
    y(n+1) = y(n) - (e^y(n) - 1 - x(n))/e^y(n)
    f(y) = e^y(n),        y(n) <= TAYLOR_NEGATIVE_THRESHOLD or y(n) >= TAYLOR_POSITIVE_THRESHOLD
    f(y) = seventh taylor computer, TAYLOR_NEGATIVE_THRESHOLD < y(n) < TAYLOR_POSITIVE_THRESHOLD

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    output_y: dict
        dict info of output_y
    kernel_name: str
        kernel name, default value is "log1p"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    dtype = input_x.dtype
    shape = input_x.shape

    cloud_check = tbe_platform.api_check_support("tbe.dsl.vlog",
                                                          "float32")
    mini_check = tbe_platform.api_check_support("tbe.dsl.vadd",
                                                         "float32")

    if dtype == "float16" and cloud_check:
        input_x = tbe.cast_to(input_x, "float32")

    data_add = tbe.vadds(input_x, tvm.const(Constant.SCALAR_ONE, "float32"))
    res = tbe.vlog(data_add)

    if (not cloud_check) and mini_check:
        input_x = tbe.cast_to(input_x, "float32")
        res = _log1p_mini_compute(res, input_x, shape)

    if dtype == "float16" and (cloud_check or mini_check):
        res = tbe.cast_to(res, "float16")

    return res


def _log1p_mini_compute(mini_res, input_x, shape):
    """
    do element-wise log(x + 1) compute in mini scene
    f(y) = e^y(n),        y(n) <= TAYLOR_NEGATIVE_THRESHOLD or y(n) >= TAYLOR_POSITIVE_THRESHOLD
    f(y) = seventh taylor computer, TAYLOR_NEGATIVE_THRESHOLD < y(n) < TAYLOR_POSITIVE_THRESHOLD

    Parameters:
    ----------
    mini_res: TVM tensor, the tensor of log(x + 1)
    input_x : TVM tensor, the placeholder of input_x
    shape : tuple, the shape of input_x

    Returns : A Tensor. Has the same type as mini_res.
    -------
    """
    # define taylor negative threshold , value = -1.7
    taylor_negative_threshold = -1.7
    # define taylor positive threshold , value = 0.7
    taylor_positive_threshold = 0.7
    input_y = tbe.cast_to(mini_res, "float32")
    input_x = tbe.cast_to(input_x, "float32")
    newton_taylor_res = _newton_taylor_log1p(input_x, input_y)
    newton_exp_res = _newton_exp_log1p(input_x, input_y)

    input_left_border = tvm.const(taylor_negative_threshold, "float32")
    tensor_input_left_border = tbe.broadcast(input_left_border, shape)

    input_right_border = tvm.const(taylor_positive_threshold, "float32")
    tensor_input_right_border = tbe.broadcast(input_right_border,
                                              shape)

    b_gt_left_border = tbe.vcmp(input_y,
                                tensor_input_left_border, 'gt')
    exp_taylor_neg = tbe.vsel(b_gt_left_border, newton_taylor_res,
                              newton_exp_res)

    b_lt_right_border = tbe.vcmp(input_y,
                                 tensor_input_right_border, 'lt')
    mini_res = tbe.vsel(b_lt_right_border, exp_taylor_neg,
                        newton_exp_res)

    return mini_res


def _exp_taylor_compute(input_x):
    """
    calculate e^x, use seventh order taylor expansion
    e^x = 1 + x + (x^2 / 2!) + (x^3 / 3!) +  (x^4 / 4!) + (x^5 / 5!) + (x^6 / 6!) + (x^7 / 7!)

    Parameters:
    ----------
    input_x : TVM tensor, the placeholder of input_x

    Returns : A Tensor. Has the same type as input_x.
    -------
    """
    # define second order parameter , value = 1 / 2.0
    taylor_second_order_param = 1 / 2.0
    # define third order parameter , value = 1 / 6.0
    taylor_third_order_param = 1 / 6.0
    # define fourth order parameter , value = 1 / 24.0
    taylor_fourth_order_param = 1 / 24.0
    # define fifth order parameter , value = 1 / 120.0
    taylor_fifth_order_param = 1 / 120.0
    # define sixth order parameter , value = 1 / 720.0
    taylor_sixth_order_param = 1 / 720.0
    # define seventh order parameter , value = 1 / 5040.0
    taylor_seventh_order_param = 1 / 5040.0
    # calculate second order tayloy section : x^2 / 2!
    taylor_second_order_param = tvm.const(taylor_second_order_param, "float32")
    data_power_2 = tbe.vmul(input_x, input_x)
    data_power_2_div_2 = tbe.vmuls(data_power_2,
                                   taylor_second_order_param)

    # calculate third order tayloy section : x^3 / 3!
    taylor_third_order_param = tvm.const(taylor_third_order_param, "float32")
    data_power_3 = tbe.vmul(data_power_2, input_x)
    data_power_3_div_6 = tbe.vmuls(data_power_3,
                                   taylor_third_order_param)

    # calculate fourth order tayloy section : x^4 / 4!
    taylor_fourth_order_param = tvm.const(taylor_fourth_order_param, "float32")
    data_power_4 = tbe.vmul(data_power_3, input_x)
    data_power_4_div_24 = tbe.vmuls(data_power_4,
                                    taylor_fourth_order_param)

    # calculate fifth order tayloy section : x^5 / 5!
    taylor_fifth_order_param = tvm.const(taylor_fifth_order_param, "float32")
    data_power_5 = tbe.vmul(data_power_4, input_x)
    data_power_5_div_120 = tbe.vmuls(data_power_5,
                                     taylor_fifth_order_param)

    # xcalculate sixth order tayloy section : ^6 / 6!
    taylor_sixth_order_param = tvm.const(taylor_sixth_order_param, "float32")
    data_power_6 = tbe.vmul(data_power_5, input_x)
    data_power_6_div_720 = tbe.vmuls(data_power_6,
                                     taylor_sixth_order_param)

    # calculate seventh order tayloy section : x^7 / 7!
    taylor_seventh_order_param = tvm.const(taylor_seventh_order_param,
                                           "float32")
    data_power_7 = tbe.vmul(data_power_6, input_x)
    data_power_7_div_5040 = tbe.vmuls(data_power_7,
                                      taylor_seventh_order_param)

    # calculate first order tayloy plus one section : 1 + x
    res_first_taylor = tbe.vadds(input_x,
                                 tvm.const(Constant.SCALAR_ONE, "float32"))
    res_second_taylor = tbe.vadd(res_first_taylor, data_power_2_div_2)
    res_third_taylor = tbe.vadd(res_second_taylor, data_power_3_div_6)
    res_fourth_taylor = tbe.vadd(res_third_taylor, data_power_4_div_24)
    res_fifth_taylor = tbe.vadd(res_fourth_taylor, data_power_5_div_120)
    res_sixth_taylor = tbe.vadd(res_fifth_taylor, data_power_6_div_720)
    res = tbe.vadd(res_sixth_taylor, data_power_7_div_5040)

    return res


def _newton_exp_iter(input_x, input_y):
    """
    do element-wise Newton compute
    y(n+1) = y(n) - (e^y(n) - 1 - x(n))/e^y(n)

    Parameters:
    ----------
    input_x: TVM tensor, the placeholder of input_x
    input_y: start value of Newton iteration

    Returns : A Tensor. Has the same type as input_y.
    -------
    """
    # `Newton begin:y(n+1) = y(n) - 1 + e^-y(n) + x(n)*e^-y(n)`
    newton_exp = tbe.vadds(input_y, tvm.const(Constant.SCALAR_NEG_ONE,
                                              "float32"))
    input_y_mul = tbe.vmuls(input_y, tvm.const(Constant.SCALAR_NEG_ONE,
                                               "float32"))
    input_y_exp = tbe.vexp(input_y_mul)
    newton_exp = tbe.vadd(newton_exp, input_y_exp)
    input_y_res = tbe.vmul(input_x, input_y_exp)
    newton_exp = tbe.vadd(newton_exp, input_y_res)
    #Newton end
    return newton_exp


def _newton_taylor_iter(input_x, input_y):
    """
    do element-wise Newton compute
    y(n+1) = y(n) - (e^y(n) - 1 - x(n))/e^y(n)

    Parameters:
    ----------
    input_x: TVM tensor, the placeholder of input_x
    input_y: start value of Newton iteration

    Returns: A Tensor. Has the same type as input_y.
    -------
    """
    # `Newton begin:y(n+1) = y(n) - 1 + e^-y(n) + x(n)*e^-y(n)`
    newton_taylor = tbe.vadds(input_y, tvm.const(Constant.SCALAR_NEG_ONE,
                                                 "float32"))
    input_y_mul = tbe.vmuls(input_y, tvm.const(Constant.SCALAR_NEG_ONE,
                                               "float32"))
    input_y_taylor = _exp_taylor_compute(input_y_mul)
    newton_taylor = tbe.vadd(newton_taylor, input_y_taylor)
    input_y_res = tbe.vmul(input_x, input_y_taylor)
    newton_taylor = tbe.vadd(newton_taylor, input_y_res)
    #Newton end
    return newton_taylor


def _newton_exp_log1p(input_x, output_y):
    """
    do element-wise Newton compute
    y(n+1) = y(n) - (e^y(n) - 1 - x(n))/e^y(n)

    Parameters:
    ----------
    input_x: TVM tensor, the placeholder of input_x
    output_y: TVM tensor, start value of log1p's Newton iteration

    Returns: A Tensor. Has the same type as output_y.
    -------
    """
    for _ in range(2):
        output_y = _newton_exp_iter(input_x, output_y)
    return output_y


def _newton_taylor_log1p(input_x, output_y):
    """
    do element-wise Newton compute
    y(n+1) = y(n) - (e^y(n) - 1 - x(n))/e^y(n)

    Parameters:
    ----------
    input_x: TVM tensor, the placeholder of input_x
    output_y: TVM tensor, start value of log1p's Newton iteration

    Returns: A Tensor. Has the same type as output_y.
    -------
    """
    for _ in range(2):
        output_y = _newton_taylor_iter(input_x, output_y)
    return output_y


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def log1p(input_x, output_y, kernel_name="log1p"):
    """
    algorithm: log1p
    calculating data's log1p, y = log(x + 1)

    Parameters
    ----------
    input_x: dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "log1p"

    Returns
    -------
    None
    """
    shape = input_x.get("shape")
    dtype = input_x.get("dtype")

    para_check.check_shape(shape, param_name="input_x")

    check_list = ("float16", "float32")
    input_dtype = dtype.lower()
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")

    shape = (functools.reduce(lambda x, y: x*y, shape[:]),)
    data_input = tvm.placeholder(shape, name="data_input", dtype=input_dtype)
    res = log1p_compute(data_input, output_y, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_input, res],
              "bool_storage_as_1bit": False}
    tbe.cce_build_code(sch, config)
