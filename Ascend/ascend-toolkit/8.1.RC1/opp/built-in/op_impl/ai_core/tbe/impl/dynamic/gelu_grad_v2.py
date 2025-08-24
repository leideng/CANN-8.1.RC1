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
gelu_grad_v2
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import OpImplMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_platform
from impl.util.util_common import check_op_impl_mode
from impl.dynamic.erf import erf_high_precision_compute
from impl.dynamic.erf import erf_compute_with_simplified_formula
from impl.dynamic.gelu_grad import gelu_grad_compute as gelu_grad_compute_tanh
from tbe.common.register import set_fusion_buildcfg


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    # CSVALUE equals 1 / np.sqrt(2)
    CSVALUE = 0.70710678118
    # CSVALUE_1 equals 0.5
    CSVALUE_1 = 0.5
    # CSVALUE_2 equals 1 / np.sqrt(2 * np.pi)
    CSVALUE_2 = 0.3989422804
    
    # min float32 value
    MIN_FP32 = 2**(-126)

    COEFFICIENT_ARRAY = [-0.7978849236182753, -6.223065142195392, -20.451359645895863,
                        -29.633095101286557, -4.474629239730099, 30.9817411426,
                        7.79950327625, 26.6304065815, 44.965039885, 30.9817411527]


def cdf(x):
    """
    alpha = 1 / sqrt(2)
    return erf(alpha * x) *0.5 + 0.5
    """
    erf_input = tbe.vmuls(x, tvm.const(Constant.CSVALUE, x.dtype))
    erf_res = erf_compute_with_simplified_formula(erf_input)
    mul_res = tbe.vmuls(erf_res, tvm.const(Constant.CSVALUE_1, x.dtype))
    res = tbe.vadds(mul_res, tvm.const(Constant.CSVALUE_1, x.dtype))

    return res


def pdf(x):
    """
    beta = 1 / sqrt(2 * pi)
    return beta * exp(-0.5*t*t)
    """
    mul_res = tbe.vmul(x, x)
    muls_res = tbe.vmuls(mul_res, tvm.const(-0.5, x.dtype))
    exp_res = tbe.vexp(muls_res)
    res = tbe.vmuls(exp_res, tvm.const(Constant.CSVALUE_2, x.dtype))

    return res


def gelu_grad_compute_none(input_dy, input_x):
    """
    result = cdf(x) + x * pdf(x)
    """
    cdf_res = cdf(input_x)
    pdf_res = pdf(input_x)
    mul_res = tbe.vmul(input_x, pdf_res)
    add_res = tbe.vadd(cdf_res, mul_res)
    result = tbe.vmul(add_res, input_dy)
    return result


def gelu_grad_compute_none_v2(input_dy, input_x):
    """
        gelu_grad_erf = erfc(-\hat{x}) / 2 + (1 /sqrt(Pi)) * (\hat{x}) * exp(-\hat{x}^2)
        erfc(x) = 1 + sgn(x) - sgn(x) * num(x) / den(x) * exp(-x^2)
        num(x) = a0 + (x * (a1 + (x * a2 + x * (a3 + x * (a4 + a5 * x)))))
        den(x) = b0 + (x * (b1 + x * (b2 + x * (b3 + x))))
    """
    dtype = input_x.dtype
    input_shape = input_x.shape

    x1 = tbe.vmuls(input_x, tvm.const(-0.70710678118654752, dtype))
    input_x_abs = tbe.vabs(x1)
    
    input_x_abs = tbe.vadds(input_x_abs, tvm.const(Constant.MIN_FP32, dtype))
    input_x_sign = tbe.vdiv(x1, input_x_abs)
    
    input_x_abs_reduce = tbe.vmins(input_x_abs, tvm.const(7.5, dtype))
    
    num = tbe.vmuls(input_x_abs_reduce, tvm.const(-5.49061387205456025e-6, dtype))
    num = tbe.vadds(num, tvm.const(0.000183133812184063945, dtype))
    num = tbe.vmul(num, input_x_abs_reduce)
    num = tbe.vadds(num, tvm.const(0.561518149524517221, dtype))
    num = tbe.vmul(num, input_x_abs_reduce)
    num = tbe.vadds(num, tvm.const(3.30140769823242503, dtype))
    num = tbe.vmul(num, input_x_abs_reduce)
    num = tbe.vadds(num, tvm.const(7.81521876717522386, dtype))
    num = tbe.vmul(num, input_x_abs_reduce)
    num = tbe.vadds(num, tvm.const(8.99592084642248623, dtype))
    
    den = tbe.vadds(input_x_abs_reduce, tvm.const(5.81125804356746743, dtype))
    den = tbe.vmul(den, input_x_abs_reduce)
    den = tbe.vadds(den, tvm.const(14.5781556895088744, dtype))
    den = tbe.vmul(den, input_x_abs_reduce)
    den = tbe.vadds(den, tvm.const(17.9660160055262283, dtype))
    den = tbe.vmul(den, input_x_abs_reduce)
    den = tbe.vadds(den, tvm.const(8.99592098051391175, dtype))

    res = tbe.vdiv(num, den)
    x_square = tbe.vmul(input_x, input_x)
    x_square = tbe.vmuls(x_square, tvm.const(-0.5, dtype))
    
    x_exp = tbe.vexp(x_square)
    res = tbe.vmul(x_exp, res)
    res = tbe.vmul(res, input_x_sign)
    
    const_zero = tbe.broadcast(tvm.const(0, dtype), input_shape)
    input_x_sign = tbe.vsub(const_zero, input_x_sign)
    tmp1 = tbe.vadds(input_x_sign, tvm.const(1.0, dtype))
    erfc_res = tbe.vadd(res, tmp1)
    
    erfc_res = tbe.vmuls(erfc_res, tvm.const(0.5, dtype))
    
    res1 = tbe.vmul(x_exp, x1)
    res1 = tbe.vmuls(res1, tvm.const(-0.5641895835477562869, dtype))
    res1 = tbe.vadd(erfc_res, res1)
    
    result = tbe.vmul(input_dy, res1)
    return result


def gelu_grad_compute_none_v3(input_dy, input_x):
    """
        gelu_grad_erf = erfc(-\hat{x}) / 2 + (1 /sqrt(Pi)) * (\hat{x}) * exp(-\hat{x}^2)
        erfc(x) = 1 + sgn(x) - sgn(x) * num(x) / den(x) * exp(-x^2)
        num(x) = a0 + (x * (a1 + (x * a2 + x * (a3 + x * (a4 + a5 * x)))))
        den(x) = b0 + (x * (b1 + x * (b2 + x * (b3 + x))))
    """
    dtype = input_x.dtype
    input_shape = input_x.shape
    
    x1 = tbe.vabs(input_x)
    x1 = tbe.vmins(x1, 30.0)
    num = tbe.vmuls(x1, Constant.COEFFICIENT_ARRAY[0])
    num = tbe.vadds(num, Constant.COEFFICIENT_ARRAY[1])
    num = tbe.vmul(num, x1)
    num = tbe.vadds(num, Constant.COEFFICIENT_ARRAY[2])
    num = tbe.vmul(num, x1)
    num = tbe.vadds(num, Constant.COEFFICIENT_ARRAY[3])
    num = tbe.vmul(num, x1)
    num = tbe.vadds(num, Constant.COEFFICIENT_ARRAY[4])
    num = tbe.vmul(num, x1)
    num = tbe.vadds(num, Constant.COEFFICIENT_ARRAY[5])

    den = tbe.vadds(x1, Constant.COEFFICIENT_ARRAY[6])
    den = tbe.vmul(den, x1)
    den = tbe.vadds(den, Constant.COEFFICIENT_ARRAY[7])
    den = tbe.vmul(den, x1)
    den = tbe.vadds(den, Constant.COEFFICIENT_ARRAY[8])
    den = tbe.vmul(den, x1)
    den = tbe.vadds(den, Constant.COEFFICIENT_ARRAY[9])

    res = tbe.vdiv(num, den)
    x2 = tbe.vmul(x1, x1)
    x2 = tbe.vmuls(x2, tvm.const(-0.5, input_x.dtype))
    x_exp = tbe.vexp(x2)
    res = tbe.vmul(res, x_exp)
    sign_x = tbe.vsignbit(input_x)
    left_tmp = tbe.vadds(sign_x, tvm.const(-0.5, input_x.dtype))
    left_tmp = tbe.vmul(left_tmp, res)
    right_tmp = tbe.vadds(sign_x, -1.0)
    input_x_out = tbe.vsub(left_tmp, right_tmp)
    result = tbe.vmul(input_dy, input_x_out)
    
    return result


@register_operator_compute("GeluGradV2", op_mode="dynamic", support_fusion=True, support_bfp16=False)
def gelu_grad_v2_compute(input_dy,
                         input_x,
                         output_z,
                         approximate="none",
                         kernel_name="gelu_grad_v2",
                         impl_mode=OpImplMode.HIGH_PRECISION):
    """
    algorithm: gelu_grad_v2
    calculating: dy*res'

    approximate="tanh":
           `np.sqrt(2 / np.pi)*(1 + 3*0.044715*x2)`
        res' = `res/x +`
    math_four = `(np.sqrt(2 / np.pi)*(x + 0.044715*tf.pow(x, 3)))`
            `x*0.5*(1 - tanh(math_four)*tanh(math_four))*`
            `np.sqrt(2 / np.pi)*(1 + 3*0.044715*x2)`
        math_four = `(np.sqrt(2 / np.pi)*(x + 0.044715*tf.pow(x, 3)))`

    approximate="none"
        res' = `cdf(x) + x * pdf(x)`
        cdf(x) = `erf(alpha * x) *0.5 + 0.5`
        alpha = 1 / sqrt(2)
        pdf(x) = `beta * exp(-0.5*t*t)`
        beta = 1 / sqrt(2 * pi)

    Parameters
    ----------
    input_dy: TVM tensor.
        the placeholder of input input_dy
    input_x: TVM tensor.
        the placeholder of input input_x
    output_z: dict
        shape and dtype of output
    approximate: str
        The gelu grad approximation algorithm to use: 'none' or 'tanh', default is 'none'.
    kernel_name: str
        cce kernel name, default value is "gelu_grad_v2"
    impl_mode: str
        impl mode, must in [OpImplMode.HIGH_PRECISION, OpImplMode.HIGH_PERFORMANCE]

    Returns:
    -------
    A TVM tensor same as input placeholders.
    """
    # set fusion build config to avoid the problem: fusion pass set dummy_placeholder default = False
    # when the input is unused, the cce will miss the input gm addr and trigger 0x800000
    build_cfg = {'dummy_placeholder': True}
    set_fusion_buildcfg("GeluGradV2", build_cfg)

    x_dtype = input_x.dtype
    if x_dtype in ("float16", "bfloat16"):
        input_x = tbe.cast_to(input_x, "float32")

    dy_dtype = input_dy.dtype
    if dy_dtype in ("float16", "bfloat16"):
        input_dy = tbe.cast_to(input_dy, "float32")

    if approximate == "none":
        cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
        if cce_product == "Ascend910B":
            result = gelu_grad_compute_none_v3(input_dy, input_x)
        elif cce_product == "Ascend310P":
            result = gelu_grad_compute_none_v2(input_dy, input_x)
        else:
            result = gelu_grad_compute_none(input_dy, input_x)
    else:
        result = gelu_grad_compute_tanh(input_dy, input_x, input_x, output_z, kernel_name, impl_mode)

    if x_dtype != result.dtype or dy_dtype != result.dtype:
        result = tbe.cast_to(result, x_dtype)

    return result


# 'pylint: disable=invalid-name
@register_operator("GeluGradV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_STR,
                            para_check.KERNEL_NAME, para_check.OPTION_ATTR_STR)
def gelu_grad_v2(input_dy, input_x, output_z, approximate="none",
                 kernel_name="gelu_grad_v2", impl_mode=OpImplMode.HIGH_PRECISION):
    """
    algorithm: gelu_grad_v2
    calculating: dy*res'

    approximate="tanh":
           `np.sqrt(2 / np.pi)*(1 + 3*0.044715*x2)`
        res' = `res/x +`
    math_four = `(np.sqrt(2 / np.pi)*(x + 0.044715*tf.pow(x, 3)))`
            `x*0.5*(1 - tanh(math_four)*tanh(math_four))*`
            `np.sqrt(2 / np.pi)*(1 + 3*0.044715*x2)`
        math_four = `(np.sqrt(2 / np.pi)*(x + 0.044715*tf.pow(x, 3)))`

    approximate="none"
        res' = `cdf(x) + x * pdf(x)`
        cdf(x) = `erf(alpha * x) *0.5 + 0.5`
        alpha = 1 / sqrt(2)
        pdf(x) = `beta * exp(-0.5*t*t)`
        beta = 1 / sqrt(2 * pi)

    Parameters
    ----------
    input_dy : dict
        shape and dtype of dy input, only support bfloat16, float16, float32
    input_x : dict
        shape and dtype of x input, only support bfloat16, float16, float32
    output_z: dict
        shape and dtype of output, should be same shape and type as input
    approximate: str
        The gelu grad approximation algorithm to use: 'none' or 'tanh', default is 'none'.
    kernel_name : str
        cce kernel name, default value is gelu_grad_v2
    impl_mode: str
        impl mode, must in [OpImplMode.HIGH_PRECISION, OpImplMode.HIGH_PERFORMANCE]

    Returns:
    -------
    none.
    """

    check_op_impl_mode(impl_mode, [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION], kernel_name)
    dy_dtype = input_dy.get("dtype").lower()
    x_dtype = input_x.get("dtype").lower()
    check_list = ("bfloat16", "float16", "float32")
    para_check.check_dtype(dy_dtype, check_list, param_name="input_dy")
    para_check.check_dtype(x_dtype, check_list, param_name="input_x")

    if dy_dtype != x_dtype:
        raise RuntimeError("Type of input_dy[%s] is not same as type of input_x[%s]."%(dy_dtype, x_dtype))

    ins = classify([input_dy, input_x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (g, x) in ins:
        with tbe.compute():
            dy_shape, x_shape = shape_util.variable_shape([g, x])
            data_dy = tvm.placeholder(dy_shape, dtype=dy_dtype, name="dy_dtype")
            data_x = tvm.placeholder(x_shape, dtype=x_dtype, name="x_dtype")
            res = gelu_grad_v2_compute(data_dy, data_x, output_z, approximate, kernel_name, impl_mode)
            tensors.append((data_dy, data_x, res))
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
