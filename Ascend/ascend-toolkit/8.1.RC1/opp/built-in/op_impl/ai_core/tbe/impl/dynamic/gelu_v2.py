#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2023 Huawei Technologies Co., Ltd
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
gelu_v2
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import OpImplMode
from impl.util.platform_adapter import tbe_platform
from impl.util.util_common import check_op_impl_mode
from impl.util.util_soc_common import after_v200
from impl.dynamic.erf import erf_high_precision_compute
from impl.dynamic.gelu import gelu_compute as gelu_compute_tanh


def phi(input_x):
    '''
    phi(x) = 0.5 + 0.5 * erf(x/sqrt2)
    '''
    const_rev_sqrt2 = tvm.const(0.70710678118, "float32")
    const_one = tvm.const(1.0, "float32")
    const_half = tvm.const(0.5, "float32")

    erf_input = tbe.vmuls(input_x, const_rev_sqrt2)
    res_erf = erf_high_precision_compute(erf_input)
    adds = tbe.vadds(res_erf, const_one)
    res = tbe.vmuls(adds, const_half)

    return res


def gelu_compute_none(input_x):
    phi_res = phi(input_x)
    result = tbe.vmul(input_x, phi_res)
    return result


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,no-member
def gelu_compute_erf(input_x):
    """
    supported dtype is float32
    res = x/(1+exp(((((((a1*x^2+a2)*x^2+a3)*x^2+a4)*x^2+a5)*x^2+a6)*x^2+a7)*x))
    """
    if not after_v200():
        # v200/v100 donot suported infnan, must do vmaxs to clip 
        input_x = tbe.vmaxs(input_x, tvm.const(-13.25, "float32"))
    x1 = tbe.vmins(input_x, tvm.const(5.75, "float32"))
    x_pow = tbe.vmul(x1, x1)
    y = tbe.vmuls(x_pow, tvm.const(-0.3512339572e-8, "float32"))
    y = tbe.vadds(y, tvm.const(0.2645266170e-6, "float32"))
    y = tbe.vmul(y, x_pow)
    y = tbe.vadds(y, tvm.const(-0.7929488134e-5, "float32"))
    y = tbe.vmul(y, x_pow)
    y = tbe.vadds(y, tvm.const(0.1106123840e-3, "float32"))
    y = tbe.vmul(y, x_pow)
    y = tbe.vadds(y, tvm.const(0.6518995814e-4, "float32"))
    y = tbe.vmul(y, x_pow)
    y = tbe.vadds(y, tvm.const(-0.7266616915e-1, "float32"))
    y = tbe.vmul(y, x_pow)
    y = tbe.vadds(y, tvm.const(-0.1595769883e1, "float32"))
    y = tbe.vmul(y, x1)
    y = tbe.vexp(y)
    y = tbe.vadds(y, tvm.const(1.0, "float32"))
    res = tbe.vdiv(input_x, y)

    return res


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,no-member
@register_operator_compute("GeluV2", op_mode="dynamic", support_fusion=True, support_bfp16=False)
def gelu_v2_compute(input_x, output_y, approximate="none", kernel_name="gelu_v2", impl_mode=OpImplMode.HIGH_PRECISION):
    """
    mathematical formula of gelu_v2(x):
    approximate="tanh":
    tanh(y) = 2/(1+exp(-2y)) - 1
        gelu(x) = 0.5*x*(1.0+tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3))))
    convert gelu to result(x) =
        tanh(y) = 2/(1+exp(-2y)) - 1
     x/(1+e(-2*(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3)))))
        convert gelu to result(x) = x/(1+e(-2*(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3)))))

    approximate="none":
        gelu_v2(x) = x * phi(x)
        phi(x) = 0.5 + 0.5 * erf(x/sqrt2)

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input input_x
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    approximate: str
        The gelu approximation algorithm to use: 'none' or 'tanh', default is 'none'.
    kernel_name: str
        cce kernel name, default value is gelu_v2
    impl_mode: str
        impl mode, must in [OpImplMode.HIGH_PRECISION, OpImplMode.HIGH_PERFORMANCE]

    Returns
    -------
     A TVM tensor same as input placeholders.
    """
    x_dtype = input_x.dtype
    if x_dtype in ("float16", "bfloat16"):
        input_x = tbe.cast_to(input_x, "float32")

    if approximate == "none":
        result = gelu_compute_erf(input_x)
    else:
        result = gelu_compute_tanh(input_x, output_y, kernel_name, impl_mode)

    if x_dtype != result.dtype:
        result = tbe.cast_to(result, x_dtype)

    return result


# 'pylint: disable=invalid-name
@register_operator("GeluV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME, para_check.OPTION_ATTR_STR)
def gelu_v2(x, y, approximate="none", kernel_name="gelu_v2", impl_mode=OpImplMode.HIGH_PRECISION):
    """
    mathematical formula of gelu_v2(x):
    approximate="tanh":
    tanh(y) = 2/(1+exp(-2y)) - 1
        gelu(x) = 0.5*x*(1.0+tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3))))
    convert gelu to result(x) =
        tanh(y) = 2/(1+exp(-2y)) - 1
     x/(1+e(-2*(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3)))))
        convert gelu to result(x) = x/(1+e(-2*(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3)))))

    approximate="none":
        gelu_v2(x) = x * phi(x)
        phi(x) = 0.5 + 0.5 * erf(x/sqrt2)

    Parameters
    ----------
    x : dict
        shape and dtype of input, only support bfloat16 float16, float32
    y: dict
        shape and dtype of output, should be same shape and type as input
    approximate: str
        The gelu approximation algorithm to use: 'none' or 'tanh', default is 'none'.
    kernel_name : str
        cce kernel name, default value is gelu_v2
    impl_mode: str
        impl mode, must in [OpImplMode.HIGH_PRECISION, OpImplMode.HIGH_PERFORMANCE]

    Returns
    -------
    None.
    """

    check_op_impl_mode(impl_mode, [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION], kernel_name)
    dtype_x = x.get("dtype").lower()
    check_list = ("bfloat16", "float16", "float32",)
    para_check.check_dtype(dtype_x, check_list, param_name="x")

    ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (x1,) in ins:
        with tbe.compute():
            shape_x = shape_util.variable_shape([x1])
            input_data = tvm.placeholder(shape_x[0], name="input_data",
                                         dtype=dtype_x)
            res = gelu_v2_compute(input_data, y, approximate, kernel_name, impl_mode)
            tensors.append([input_data, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors, "build_args": {"status_check": False}}
    tbe.build(schedules, config)
