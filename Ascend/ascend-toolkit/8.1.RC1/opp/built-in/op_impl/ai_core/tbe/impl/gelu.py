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
gelu
"""
import functools

from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpImplMode
from impl.util.util_common import check_op_impl_mode
from .fast_gelu_v2 import fast_gelu_v2_compute


# 'pylint: disable=too-many-locals,invalid-name
def _tanh_parameter_compute(placeholders):
    """
    compute the parameter of tanh:
    :param placeholders: input data
    return: result equals (x+0.044715*tf.pow(x,3))
    """
    cs_value = tvm.const(0.044715, "float32")
    mul_0 = tbe.vmul(placeholders, placeholders)
    pow_0 = tbe.vmul(mul_0, placeholders)
    mul_1 = tbe.vmuls(pow_0, cs_value)
    result = tbe.vadd(placeholders, mul_1)

    return result


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,no-member
@register_operator_compute("gelu", op_mode="static", support_fusion=True)
def gelu_compute(input_x, output_y, kernel_name="gelu",
                 impl_mode=OpImplMode.HIGH_PRECISION):
    """
    mathematical formula of gelu(x):
    gelu(x) = 0.5*x*(1.0+tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3))))
    tanh(y) = 2/(1+exp(-2y)) - 1
    convert gelu to result(x) =
      x/(1+e(-2*(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3)))))

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input input_x
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is gelu
    impl_mode: str
        impl_mode, default value is None

    Returns
    -------
     A TVM tensor same as input placeholders.
    """
    if impl_mode == OpImplMode.HIGH_PERFORMANCE:
        return fast_gelu_v2_compute(input_x, output_y, kernel_name, impl_mode)

    dtype = input_x.dtype
    has_improve_precision = False
    if dtype == "float16" and \
            tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        if impl_mode == OpImplMode.HIGH_PRECISION or (impl_mode is None):
            has_improve_precision = True
            input_x = tbe.cast_to(input_x, "float32")

    # `gelu(x) = 0.5*x*(1.0+tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3))))`
    # `tanh(y) = 2/(1+exp(-2y)) - 1`

    # simplize
    # `gelu(x) = x/(1+e^(-y))`
    # the formula is y = 2*np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3))

    # to avoid overflow, keep exp negative
    # `gelu(x) = x/(1+e^(-|y|)) * v_const`
    # the formula is y = 2*np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3))
    # `v_const = 1 if x > 0  , e^y  if x < 0`
    const_0 = tvm.const(1.5957691, "float32") # 2*np.sqrt(2/np.pi)
    const_1 = tvm.const(1.0, "float32")
    const_2 = tvm.const(-1.0, "float32")
    const_3 = tvm.const(0.0, "float32")
    tanh_parameter = _tanh_parameter_compute(input_x)
    mul_0 = tbe.vmuls(tanh_parameter, const_0)  # y

    mul_0_min = tbe.vmins(mul_0, const_3)
    right_mul = tbe.vexp(mul_0_min)

    mul_0_abs = tbe.vabs(mul_0)   # abs(y)
    mul_0_abs_neg = tbe.vmuls(mul_0_abs, const_2)  # -abs(y)

    # the formula is e^(-abs(y))
    mul_0_abs_neg_exp = tbe.vexp(mul_0_abs_neg)

    # the formula is e^(-abs(y)) + 1
    mul_0_abs_neg_exp_add = tbe.vadds(mul_0_abs_neg_exp, const_1)
    left_mul = tbe.vdiv(input_x, mul_0_abs_neg_exp_add)

    result = tbe.vmul(left_mul, right_mul)

    if has_improve_precision:
        result = tbe.cast_to(result, "float16")

    return result


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def gelu(input_x, output_y, kernel_name="gelu", impl_mode=OpImplMode.HIGH_PRECISION):
    """
    mathematical formula of gelu(x):
    gelu(x) = 0.5*x*(1.0+tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3))))
    tanh(y) = 2/(1+exp(-2y)) - 1
    convert gelu to result(x) =
     x/(1+e(-2*(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3)))))

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is gelu
    impl_mode:str
        impl_mode, default value is None

    Returns
    -------
    None.
    """
    check_op_impl_mode(impl_mode, [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION], kernel_name)

    input_shape = input_x.get("shape")
    para_check.check_shape(input_shape, param_name="input_x")

    check_list = ("float16", "float32")
    input_dtype = input_x.get("dtype").lower()
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")

    fuseshape = [1]
    fuseshape[0] = functools.reduce(lambda x, y: x*y, input_shape)
    data = tvm.placeholder(fuseshape, name="data", dtype=input_dtype)
    result = gelu_compute(data, output_y, kernel_name, impl_mode=impl_mode)

    with tvm.target.cce():
        sch = tbe.auto_schedule(result)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data, result]}

    tbe.build(sch, config)
