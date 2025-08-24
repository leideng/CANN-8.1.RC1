#!/usr/bin/env python
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
gelu_grad
"""

import operator
import functools

from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant.
    """
    CSVALUE = tvm.const(0.044715, "float32")
    SQURT = tvm.const(0.7978846, "float32")
    CSVALUE_4 = tvm.const(0.0535161122, "float32")
    CSVALUE_5 = tvm.const(0.3989422804, "float32")
    MIN_FP32 = 2**(-126)


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals
def tanh_compute(input_x, output_y, kernel_name="tanh"):
    """
    algorithm: tanh
    calculating data's tanh,y= (e^(2x)-1)/(e^(2x)+1)

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is tanh

    Returns
    -------
    res : tvm.tensor
        the result of tanh
    """
    input_dtype = input_x.dtype

    support_fp32 = tbe_platform.api_check_support("tbe.dsl.vexp", "float32")
    has_improve_precision = False
    if input_dtype == "float16" and support_fp32:
        input_x = tbe.cast_to(input_x, "float32")
        has_improve_precision = True

    input_abs = tbe.vabs(input_x)
    power_val = tbe.vmuls(input_abs, tvm.const(-2, "float32"))
    exp_val = tbe.vexp(power_val)

    up_val_tmp = tbe.vmul(exp_val, input_x)
    up_val = tbe.vsub(input_x, up_val_tmp)

    input_x_tmp = tbe.vadds(input_abs, Constant.MIN_FP32)
    down_val_tmp = tbe.vadds(exp_val, tvm.const(1, "float32"))
    down_val = tbe.vmul(down_val_tmp, input_x_tmp)

    res = tbe.vdiv(up_val, down_val)

    if has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res


def _math_four_compute(placeholders):
    """
    placeholders: data_x
    return: math_four
    math_four equals (np.sqrt(2 / np.pi)*(x + 0.044715*tf.pow(x, 3)))
    """
    data_x = placeholders
    datax_pow = tbe.vmul(data_x, data_x)
    datax_pow1 = tbe.vmul(datax_pow, data_x)
    datax_muls_c = tbe.vmuls(datax_pow1, Constant.CSVALUE)
    datax_addx = tbe.vadd(datax_muls_c, data_x)
    datax_muls_s = tbe.vmuls(datax_addx, Constant.SQURT)

    return datax_muls_s



def _result2_compute(placeholders):
    """
    placeholders: data_x
    return: result
    result equals np.sqrt(2 / np.pi) (1 + 3*0.044715*x2)
    """
    data_x = placeholders
    val1 = Constant.CSVALUE_5
    data_x_sqr = tbe.vmul(data_x, data_x)
    data_x_sqr_vmul = tbe.vmuls(data_x_sqr, Constant.CSVALUE_4)
    data_x_sqr_vmul_add1 = tbe.vadds(data_x_sqr_vmul, val1)

    return data_x_sqr_vmul_add1


def _result3_compute(placeholders):
    """
    placeholders: data_x
    return: result3
    result3 equals x*0.5*(1 - tanh(math_four)*tanh(math_four))
    """
    data_x = placeholders
    val1 = tvm.const(1.0, "float32")
    math_four = _math_four_compute(data_x)
    tanh_math_four = tanh_compute(math_four, placeholders[1])
    tanh_math_four_squ = tbe.vmul(tanh_math_four, tanh_math_four)
    val3 = tvm.const(-1.0, "float32")
    math_four_squ_n = tbe.vmuls(tanh_math_four_squ, val3)
    add_compute = tbe.vadds(math_four_squ_n, val1)
    result3 = tbe.vmul(add_compute, data_x)

    return result3, tanh_math_four


def _result_grad_compute(placeholders):
    """
    placeholders: data_x, data_gelu
    return: res_grad
    res_grad = `res/x +`
       `x*0.5*(1 - tanh(math_four)*tanh(math_four))*`
       `np.sqrt(2 / np.pi)*(1 + 3*0.044715*x2)`
    """
    data_x = placeholders[0]

    result2 = _result2_compute(data_x)
    result3, tanh_math_four_result = _result3_compute(data_x)
    mul_result2_3 = tbe.vmul(result2, result3)

    # `compute res1 = res/x = f1 = x*(0.5*(1+tanh_math_four_result))`
    mul_compute_1 = tbe.vadds(tanh_math_four_result, 1)
    mul_compute_2 = tbe.vmuls(mul_compute_1, 0.5)

    res_grad = tbe.vadd(mul_compute_2, mul_result2_3)

    return res_grad


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,no-member
@register_operator_compute("gelu_grad", op_mode="static", support_fusion=True)
def gelu_grad_compute(input_dy, input_x, input_y,
                      output_z, kernel_name="gelu_grad"):
    """
    algorithm: gelu_grad
    calculating: dy*res'
    res' = `res/x +`
           `x*0.5*(1 - tanh(math_four)*tanh(math_four))*`
           `np.sqrt(2 / np.pi)*(1 + 3*0.044715*x2)`
    math_four = `(np.sqrt(2 / np.pi)*(x + 0.044715*tf.pow(x, 3)))`

    Parameters
    ----------
    input_dy: TVM tensor.
        the placeholder of input input_dy
    input_x: TVM tensor.
        the placeholder of input input_x
    input_y: TVM tensor.
        the placeholder of input input_y
    output_z: dict
        shape and dtype of output
    kernel_name: str
        cce kernel name, default value is "gelu_grad"
    Returns:
    -------
    A TVM tensor same as input placeholders.
    """
    input_dtype = input_dy.dtype.lower()

    has_improve_precision = False
    if input_dtype == "float16" and \
            tbe_platform.api_check_support("tbe.dsl.vexp",
                                                    "float32"):
        input_dy = tbe.cast_to(input_dy, "float32")
        input_x = tbe.cast_to(input_x, "float32")
        input_y = tbe.cast_to(input_y, "float32")
        has_improve_precision = True
    # compute res'
    result5 = _result_grad_compute([input_x, output_z])
    # compute dy*res'
    result_temp1 = tbe.vmul(input_dy, result5)

    # input_y must be involved in order to keep it
    input_y_temp_1 = tbe.vmuls(input_y, 0)

    result = tbe.vadd(result_temp1, input_y_temp_1)
    if has_improve_precision:
        result = tbe.cast_to(result, "float16")

    return result


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def gelu_grad(input_dy, input_x, input_y, output_z, kernel_name="gelu_grad"):
    """
    algorithm: gelu_grad
    calculating: dy*res'
    res' = `res/x +`
           `x*0.5*(1 - tanh(math_four)*tanh(math_four))*`
           `np.sqrt(2 / np.pi)*(1 + 3*0.044715*x2)`
    math_four = `(np.sqrt(2 / np.pi)*(x + 0.044715*tf.pow(x, 3)))`

    Parameters
    ----------
    input_dy : dict
        shape and dtype of dy input, only support float16, float32
    input_x : dict
        shape and dtype of x input, only support float16, float32
    input_y : dict
        shape and dtype of y input, only support float16, float32
    output_z: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is gelu_grad

    Returns:
    -------
    none.
    """
    shape_dy = input_dy.get("shape")
    shape_x = input_x.get("shape")
    shape_y = input_y.get("shape")

    para_check.check_shape(shape_dy, param_name="input_dy")
    para_check.check_shape(shape_x, param_name="input_x")
    para_check.check_shape(shape_y, param_name="input_y")
    input_dtype = input_dy.get("dtype").lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(input_dtype, check_list, param_name="input_dy")
    shape_dy = list(shape_dy)
    shape_x = list(shape_x)
    shape_y = list(shape_y)

    if not (operator.eq(shape_dy, shape_x) and operator.eq(shape_dy, shape_y)):
        error_detail = "all input shape must be equal, shape_dy:%s, shape_x:%s, shape_y:%s" % shape_dy, shape_x, shape_y
        error_manager_vector.raise_err_two_input_shape_invalid("gelu_grad", "shape_dy or shape_x",
                                                               "shape_dy or shape_y", error_detail)

    fuseshape = [1]
    fuseshape[0] = functools.reduce(lambda x, y: x*y, shape_dy)
    data_dy = tvm.placeholder(fuseshape, name="data_dy", dtype=input_dtype)
    data_x = tvm.placeholder(fuseshape, name="data_x", dtype=input_dtype)
    data_gelu = tvm.placeholder(fuseshape, name="data_gelu", dtype=input_dtype)
    res = gelu_grad_compute(data_dy, data_x, data_gelu, output_z, kernel_name)

    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data_dy, data_x, data_gelu, res]}

    build(sch, config)
