#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
from collections import namedtuple
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import tbe_platform
from impl.util.util_common import check_op_impl_mode
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


def compute_x_grad(input_x, x_dtype, x_shape):
    ones = tbe.broadcast(tvm.const(1, x_dtype), x_shape)
    # x > 0 : (2e^(-x) + 1) / (2e^(-2x) + 2e^(-x) + 1)
    logic_type = 'bool'
    mask1 = tbe.vcmp(input_x, 0, 'gt', mode=logic_type)
    neg_x_one = tbe.vmuls(input_x, -1)
    neg_x_two = tbe.vmuls(input_x, -2)
    exp_val_neg_one = tbe.vexp(neg_x_one)
    exp_val_neg_two = tbe.vexp(neg_x_two)
    two_exp_neg_one = tbe.vmuls(exp_val_neg_one, 2)
    two_exp_neg_two = tbe.vmuls(exp_val_neg_two, 2)
    tmp1 = tbe.vadds(two_exp_neg_one, 1)
    tmp2 = tbe.vadd(two_exp_neg_one, two_exp_neg_two)
    tmp2 = tbe.vadds(tmp2, 1)
    tanh_val_1 = tbe.vdiv(tmp1, tmp2)

    s_acc = tbe.vadds(exp_val_neg_one, 1)
    s_acc = tbe.vdiv(input_x, s_acc)
    val1 = tbe.vmul(tanh_val_1, tanh_val_1)
    val1 = tbe.vsub(ones, val1)
    val1 = tbe.vmul(s_acc, val1)
    val1 = tbe.vadd(tanh_val_1, val1)
    val1 = tbe.vsel(mask1, val1, 0)

    # x <= 0: (2e^x + e^2x) / (2 + 2e^x + e^2x)
    mask2 = tbe.vcmp(input_x, 0, 'le', mode=logic_type)
    two_x = tbe.vmuls(input_x, 2)
    exp_val = tbe.vexp(input_x)
    two_exp_val = tbe.vmuls(exp_val, 2)
    exp_val_two = tbe.vexp(two_x)
    tmp1 = tbe.vadd(two_exp_val, exp_val_two)
    tmp2 = tbe.vadds(tmp1, 2)
    tanh_val_2 = tbe.vdiv(tmp1, tmp2)

    s_acc = tbe.vmul(input_x, exp_val)
    s_acc = tbe.vdiv(s_acc, tbe.vadds(exp_val, 1))
    val2 = tbe.vmul(tanh_val_2, tanh_val_2)
    val2 = tbe.vsub(ones, val2)
    val2 = tbe.vmul(s_acc, val2)
    val2 = tbe.vadd(tanh_val_2, val2)
    val2 = tbe.vsel(mask2, val2, 0)

    x_grad = tbe.vadd(val1, val2)
    
    # handling nan values
    mask = tbe.vadd(tbe.cast_to(mask1, 'float16'), tbe.cast_to(mask2, 'float16'))
    mask = tbe.cast_to(mask, logic_type)
    x_grad = tbe.vsel(mask, x_grad, input_x)
    
    return x_grad


# 'pylint: disable=unused-argument
@register_operator_compute("MishGrad", op_mode="dynamic", support_fusion=True)
def mish_grad_compute(input_grad, input_x, input_tanhx, output_grad, kernel_name, impl_mode):
    """
    algorithm: mish_grad
    when input_tanhx is none
        calculating data's mish_grad:
            y1 = 1 / ((1 + exp(x)) ^ 2 + 1)
            x_grad = (4 * x * exp(x) * (exp(x) + 1)) * (y1 ^ 2) - 2 * y1 + 1
            output_grad = input_grad * x_grad
    else:
        calculating data's mish_grad:
            tanhx = tanh(ln(1+exp(x)))
            x_grad = tanhx - x*(tanhx^2 - 1) / (1+exp(x)) * exp(x)
            output_grad = input_grad * x_grad

    Parameters
    ----------
    input_grad: TVM tensor
        the placeholder of input data's gradient, should be same shape and type as data_x
    input_x: TVM tensor
        the placeholder of input data, only support float16, float32, bfloat16
    input_tanhx: TVM tensor
        the placeholder of input data, only support float16, float32, bfloat16
    output_grad: TVM tensor
        the output tensor of mish grad
    kernel_name : str
        cce kernel name, default value is mish_grad
    impl_mode : str
        the specified mode in which the operator runs

    Returns
    -------
    res : tvm.tensor
        the result of mish_grad compute
    """
    ori_dtype = input_x.dtype
    x_dtype = ori_dtype
    x_shape = input_x.shape
    vexp_support_fp32 = tbe_platform.api_check_support("tbe.dsl.vexp", "float32")

    if ori_dtype == "float16" or ori_dtype == "bfloat16":
        x_dtype = "float32"
        input_grad = tbe.cast_to(input_grad, x_dtype)
        input_tanhx = tbe.cast_to(input_tanhx, x_dtype) if input_tanhx is not None else None
        if vexp_support_fp32:
            input_x = tbe.cast_to(input_x, x_dtype)
            exp_val = tbe.vexp(input_x)
        else:
            exp_val = tbe.cast_to(tbe.vexp(input_x), x_dtype)
            input_x = tbe.cast_to(input_x, x_dtype)
    else:
        if vexp_support_fp32:
            exp_val = tbe.vexp(input_x)
        else:
            exp_val = tbe.cast_to(tbe.vexp(tbe.cast_to(input_x, "float16")), x_dtype)

    if input_tanhx is None:
        x_grad = compute_x_grad(input_x, x_dtype, x_shape)
    else:
        ones = tbe.broadcast(tvm.const(1, x_dtype), x_shape)
        add_exp_val = tbe.vadds(exp_val, tvm.const(1, x_dtype))
        rec_add_exp_val = tbe.vdiv(ones, add_exp_val)
        pow_tanh = tbe.vmul(input_tanhx, input_tanhx)
        sub_pow_tanh = tbe.vadds(pow_tanh, tvm.const(-1, x_dtype))
        x_mul_exp_x = tbe.vmul(input_x, exp_val)
        sub_pow_tanh_mul_x_mul_exp_x = tbe.vmul(x_mul_exp_x, sub_pow_tanh)
        val_except_sub = tbe.vmul(sub_pow_tanh_mul_x_mul_exp_x, rec_add_exp_val)
        x_grad = tbe.vsub(input_tanhx, val_except_sub)

    res = tbe.vmul(x_grad, input_grad)
    if "bfloat16" == ori_dtype:
        res = tbe.round(res, ori_dtype)
    elif x_dtype != ori_dtype:
        res = tbe.cast_to(res, ori_dtype)

    return res


# 'pylint: disable=too-many-locals
def check_params(inputs, output_grad, kernel_name, impl_mode):
    """
    check params
    """
    check_op_impl_mode(impl_mode, ["high_performance", "high_precision"], kernel_name)
    input_grad = inputs[0]
    input_x = inputs[1]
    input_tanhx = inputs[2]
    excepted_value = "equal"
    real_value = "not equal"
    input_dtype = input_x.get("dtype").lower()
    input_format = input_x.get("format")
    if input_tanhx is not None:
        params = (("input_grad", input_grad), ("input_tanhx", input_tanhx), ("output_grad", output_grad))
    else:
        params = (("input_grad", input_grad), ("output_grad", output_grad))
    for param_name, param in params:
        param_dtype = param.get("dtype").lower()
        param_format = param.get("format")
        if param_dtype != input_dtype:
            error_manager_vector.raise_err_input_value_invalid(
                kernel_name, "dtype of input_x and {}".format(param_name), excepted_value, real_value)
        if param_format != input_format:
            error_manager_vector.raise_err_input_value_invalid(
                kernel_name, "format of input_x and {}".format(param_name), excepted_value, real_value)


@register_operator("MishGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME, para_check.OPTION_ATTR_STR)
def mish_grad(input_grad, input_x, input_tanhx, output_grad, kernel_name="mish_grad", impl_mode="high_performance"):
    """
    algorithm: mish_grad

    Parameters
    ----------
    input_grad : dict
        shape and dtype of input_grad,should be same shape and type as input_x
    input_x : dict
        shape and dtype of input_data, only support float16, float32, bfloat16
    input_tanhx : dict
        shape and dtype of input_tanhx, if not None, should be same shape and type as input_x
    output_grad: dict
        shape and dtype of output_grad, should be same shape and type as input_x
    kernel_name : str
        cce kernel name, default value is mish_grad
    impl_mode : str
        the specified mode in which the operator runs, default value is high_performance

    Returns
    -------
    None
    """
    input_dtype = input_x.get("dtype").lower()
    check_tuple = ("float16", "float32", "bfloat16")
    para_check.check_dtype(input_dtype, check_tuple, param_name="input_data")
    name_tuple = namedtuple('inputs', 'input_grad, input_x, input_tanhx')
    inputs = name_tuple(input_grad=input_grad, input_x=input_x, input_tanhx=input_tanhx)
    check_params(inputs, output_grad, kernel_name, impl_mode)
    if input_tanhx is None:
        schedules, tensors = [], []
        ins = classify([input_grad, input_x], OpPatternMode.ELEWISE)
        for (grad, data) in ins:
            with tbe.compute():
                shape_grad, shape_data = shape_util.variable_shape([grad, data])
                data_grad = tvm.placeholder(shape_data, dtype=input_dtype, name="data_grad_input")
                data_data = tvm.placeholder(shape_data, dtype=input_dtype, name="data_data")
                res = mish_grad_compute(data_grad, data_data, input_tanhx, output_grad, kernel_name, impl_mode)
                tensors.append([data_grad, data_data, res])
            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)
    else:
        schedules, tensors = [], []
        ins = classify([input_grad, input_x, input_tanhx], OpPatternMode.ELEWISE)
        for (grad, data, tanhx) in ins:
            with tbe.compute():
                shape_grad, shape_data, shape_tanh = shape_util.variable_shape([grad, data, tanhx])
                data_grad = tvm.placeholder(shape_data, dtype=input_dtype, name="data_grad_input")
                data_data = tvm.placeholder(shape_data, dtype=input_dtype, name="data_data")
                data_tanh = tvm.placeholder(shape_data, dtype=input_dtype, name="data_tanh")
                res = mish_grad_compute(data_grad, data_data, data_tanh, output_grad, kernel_name, impl_mode)
                tensors.append([data_grad, data_data, data_tanh, res])
            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)
    config = {"name": kernel_name,
              "tensor_list": tensors,
              "bool_storage_as_1bit": False}
    tbe.build(schedules, config)
