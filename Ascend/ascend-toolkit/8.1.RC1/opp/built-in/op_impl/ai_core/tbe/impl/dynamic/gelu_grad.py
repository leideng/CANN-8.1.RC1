#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
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
from impl.util.util_common import check_op_impl_mode
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import OpImplMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import register_operator
from impl.util.util_soc_common import after_v200
from tbe.common.register import set_fusion_buildcfg


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    # CSVALUE equals 0.044715
    CSVALUE = 0.044715
    # SQURT equals np.sqrt(2 / np.pi)
    SQURT = 0.7978846

    # CSVALUE_4 equals 0.5*np.sqrt(2 / np.pi)*3*CSVALUE
    CSVALUE_4 = 0.0535161122
    # CSVALUE_5 equals 0.5*np.sqrt(2 / np.pi)
    CSVALUE_5 = 0.3989422804

    # min float32 value
    MIN_FP32 = 2**(-126)

    # min float16 value
    MIN_FP16 = 2**(-14)


# 'pylint: disable=locally-disabled,unused-argument
# 'pylint: disable=too-many-locals
def tanh_compute(input_x, output_y, impl_mode):
    """
    algorithm: tanh
    calculating data's tanh,y= (e^(2x)-1)/(e^(2x)+1)

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    impl_mode: str
        impl mode, must in [OpImplMode.HIGH_PRECISION, OpImplMode.HIGH_PERFORMANCE]

    Returns
    -------
    res : tvm.tensor
        the result of tanh
    """
    input_dtype = input_x.dtype

    has_improve_precision = False
    is_support_fp32 = tbe_platform.api_check_support("tbe.dsl.vexp", "float32")
    if impl_mode == OpImplMode.HIGH_PRECISION and input_dtype == "float16" and is_support_fp32:
        input_x = tbe.cast_to(input_x, "float32")
        has_improve_precision = True

    input_abs = tbe.vabs(input_x)
    power_val = tbe.vmuls(input_abs, tvm.const(-2, input_abs.dtype))

    if not is_support_fp32 and input_dtype == "float32":
        power_val_fp16 = tbe.cast_to(power_val, "float16")
        exp_val = tbe.vexp(power_val_fp16)
    else:
        exp_val = tbe.vexp(power_val)
    exp_val_fp32 = tbe.cast_to(exp_val, input_dtype)

    up_val_tmp = tbe.vmul(exp_val_fp32, input_x)
    up_val = tbe.vsub(input_x, up_val_tmp)

    min_value = Constant.MIN_FP32 if input_abs.dtype == "float32" else Constant.MIN_FP16
    input_x_tmp = tbe.vadds(input_abs, tvm.const(min_value, input_abs.dtype))
    down_val_tmp = tbe.vadds(exp_val_fp32, tvm.const(1, input_abs.dtype))
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
    datax_muls_c = tbe.vmuls(datax_pow1, tvm.const(Constant.CSVALUE, datax_pow1.dtype))
    datax_addx = tbe.vadd(datax_muls_c, data_x)
    datax_muls_s = tbe.vmuls(datax_addx, tvm.const(Constant.SQURT, datax_addx.dtype))

    return datax_muls_s


def _result2_compute(placeholders):
    """
    placeholders: data_x
    return: result
    result equals np.sqrt(2 / np.pi) (1 + 3*0.044715*x2)
    """
    data_x = placeholders
    data_x_sqr = tbe.vmul(data_x, data_x)
    data_x_sqr_vmul = tbe.vmuls(data_x_sqr, tvm.const(Constant.CSVALUE_4, data_x_sqr.dtype))
    data_x_sqr_vmul_add1 = tbe.vadds(data_x_sqr_vmul, tvm.const(Constant.CSVALUE_5, data_x_sqr_vmul.dtype))

    return data_x_sqr_vmul_add1


def _result3_compute(placeholders, impl_mode):
    """
    placeholders: data_x
    impl_mode: impl mode, must in [OpImplMode.HIGH_PRECISION, OpImplMode.HIGH_PERFORMANCE]
    return: result3
    result3 equals x*0.5*(1 - tanh(math_four)*tanh(math_four))
    """
    data_x = placeholders
    math_four = _math_four_compute(data_x)
    tanh_math_four = tanh_compute(math_four, placeholders[1], impl_mode)
    tanh_math_four_squ = tbe.vmul(tanh_math_four, tanh_math_four)
    math_four_squ_n = tbe.vmuls(tanh_math_four_squ, tvm.const(-1.0, tanh_math_four_squ.dtype))
    add_compute = tbe.vadds(math_four_squ_n, tvm.const(1.0, math_four_squ_n.dtype))
    result3 = tbe.vmul(add_compute, data_x)

    return result3, tanh_math_four


def _result_grad_compute(placeholders, impl_mode):
    """
    `placeholders: data_x, data_gelu`
    `impl_mode: impl mode, must in [OpImplMode.HIGH_PRECISION, OpImplMode.HIGH_PERFORMANCE]`
    `return: res_grad`
    `res_grad = `res/x +``
       `x*0.5*(1 - tanh(math_four)*tanh(math_four))*`
       `np.sqrt(2 / np.pi)*(1 + 3*0.044715*x2)`
    """
    data_x = placeholders[0]

    result2 = _result2_compute(data_x)
    result3, tanh_math_four_result = _result3_compute(data_x, impl_mode)
    mul_result2_3 = tbe.vmul(result2, result3)

    # `compute res1 = res/x = f1 = x*(0.5*(1+tanh_math_four_result))`
    mul_compute_1 = tbe.vadds(tanh_math_four_result, 1)
    mul_compute_2 = tbe.vmuls(mul_compute_1, 0.5)

    res_grad = tbe.vadd(mul_compute_2, mul_result2_3)

    return res_grad


def gelu_grad_compute_tanh(input_dy, input_x):
    """
    g1(x): 1.0 / (exp(x*(x^2*a1+a2))+1)
    g2(x): x^2*a3 + a4
    f(x): (x*(g1-1)*g2+1)*g1*dy
    """
    dtype = input_x.dtype
    ori_dtype = input_x.dtype
    input_shape = input_x.shape

    if dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        input_dy = tbe.cast_to(input_dy, "float32")
        input_x = tbe.cast_to(input_x, "float32")
        dtype = input_x.dtype

    x_pow = tbe.vmul(input_x, input_x)
    g1 = tbe.vmuls(x_pow, tvm.const(-0.0713548162726002527220, dtype))
    g1 = tbe.vadds(g1, tvm.const(-1.5957691216057308, dtype))
    g1 = tbe.vmul(g1, input_x)
    g1 = tbe.vexp(g1)
    g1 = tbe.vadds(g1, tvm.const(1.0, dtype))
    const_one = tbe.broadcast(tvm.const(1.0, dtype), input_shape)
    g1 = tbe.vdiv(const_one, g1)

    res = tbe.vadds(g1, tvm.const(-1.0, dtype))
    res = tbe.vmul(res, input_x)

    g2 = tbe.vmuls(x_pow, tvm.const(-0.21406444881780074632901625683959062, dtype))
    g2 = tbe.vadds(g2, tvm.const(-1.5957691216057307117597842397375274738, dtype))

    res = tbe.vmul(res, g2)
    res = tbe.vadds(res, tvm.const(1.0, dtype))
    res = tbe.vmul(res, g1)
    res = tbe.vmul(input_dy, res)

    if dtype != ori_dtype:
        res = tbe.cast_to(res, ori_dtype)

    return res


def gelu_grad_compute_tanh_v2(input_dy, input_x):
    """
    gelu = (x / 2) * (1 + tanh(f(x)))
    f(x) = sqrt(2.0f / PI) * x * (1.0f + 0.044715f * x * x)
    f'(x) = sqrt(2.0f / PI) * (1.0f + 0.134145f * x * x)
    
    gelu_grad = (1.0f + res_p * 2 * x * f'(x)) * t
    exp_f(x) = exp(-2 * f(x))
    t = 1 / (1 + exp_f(x))
    if x > 0:
        res_p = exp_f(x) * t
    else:
        res_p = 1 - t
    """
    dtype = input_x.dtype
    ori_dtype = input_x.dtype
    input_shape = input_x.shape

    if dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        input_dy = tbe.cast_to(input_dy, "float32")
        input_x = tbe.cast_to(input_x, "float32")
        dtype = input_x.dtype

    x_square = tbe.vmul(input_x, input_x)
    px = tbe.vmuls(x_square, tvm.const(-0.0713548162726002527220, dtype))
    px = tbe.vadds(px, tvm.const(-1.595769121605730711759, dtype))
    px = tbe.vmul(px, input_x)
    px = tbe.vexp(px)

    res0 = tbe.vmuls(x_square, tvm.const(0.2140644488178007, dtype))
    res0 = tbe.vadds(res0, tvm.const(1.595769121605730711759, dtype))
    res0 = tbe.vmul(res0, input_x)

    t = tbe.vadds(px, tvm.const(1.0, dtype))
    const_one = tbe.broadcast(tvm.const(1.0, dtype), input_shape)
    t = tbe.vdiv(const_one, t)

    resp = tbe.vmul(px, t)
    resp = tbe.vmul(resp, res0)
    resp = tbe.vmul(resp, t)
    mask_select = tbe.vcmp(resp, resp, "eq", "bool")
    resp = tbe.vsel(mask_select, resp, tvm.const(0.0, dtype))
    resp = tbe.vadd(resp, t)
    resp = tbe.vmul(input_dy, resp)
    
    if dtype != ori_dtype:
        resp = tbe.cast_to(resp, ori_dtype)
    
    return resp


@register_operator_compute("GeluGrad", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def gelu_grad_compute(input_dy,
                      input_x,
                      input_y,
                      output_z,
                      kernel_name="gelu_grad",
                      impl_mode=OpImplMode.HIGH_PRECISION):
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
    impl_mode: str
        impl mode, must in [OpImplMode.HIGH_PRECISION, OpImplMode.HIGH_PERFORMANCE]

    Returns:
    -------
    A TVM tensor same as input placeholders.
    """
    # set fusion build config to avoid the problem: fusion pass set dummy_placeholder default = False
    # when the input is unused, the cce will miss the input gm addr and trigger 0x800000
    build_cfg = {'dummy_placeholder': True}
    set_fusion_buildcfg("GeluGrad", build_cfg)

    if after_v200():
        cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
        if cce_product == "Ascend910B" or cce_product == "Ascend310P":
            res = gelu_grad_compute_tanh_v2(input_dy, input_x)
        else:
            res = gelu_grad_compute_tanh(input_dy, input_x)
        return res

    input_dtype = input_dy.dtype.lower()

    has_improve_precision = False
    has_improve_performance = False
    is_support_fp32 = tbe_platform.api_check_support("tbe.dsl.vexp", "float32")
    if impl_mode == OpImplMode.HIGH_PRECISION and input_dtype == "float16" and is_support_fp32:
        input_dy = tbe.cast_to(input_dy, "float32")
        input_x = tbe.cast_to(input_x, "float32")
        has_improve_precision = True
    elif impl_mode == OpImplMode.HIGH_PERFORMANCE and input_dtype == "float32":
        input_dy = tbe.cast_to(input_dy, "float16")
        input_x = tbe.cast_to(input_x, "float16")
        has_improve_performance = True
    # compute res'
    result5 = _result_grad_compute([input_x, output_z], impl_mode)
    # compute dy*res'
    result = tbe.vmul(input_dy, result5)

    if has_improve_precision:
        result = tbe.cast_to(result, "float16")
    elif has_improve_performance:
        result = tbe.cast_to(result, "float32")

    return result


# 'pylint: disable=invalid-name
@register_operator("GeluGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME, para_check.OPTION_ATTR_STR)
def gelu_grad(input_dy, input_x, input_y, output_z, kernel_name="gelu_grad", impl_mode=OpImplMode.HIGH_PRECISION):
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
        shape and dtype of dy input, only support bfloat16, float16, float32
    input_x : dict
        shape and dtype of x input, only support bfloat16, float16, float32
    input_y : dict
        shape and dtype of y input, only support bfloat16, float16, float32
    output_z: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is gelu_grad

    Returns:
    -------
    none.
    """
    check_op_impl_mode(impl_mode, [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION], kernel_name)
    dy_dtype = input_dy.get("dtype").lower()
    x_dtype = input_x.get("dtype").lower()
    y_dtype = input_y.get("dtype").lower()
    check_list = ("bfloat16", "float16", "float32")
    para_check.check_dtype(dy_dtype, check_list, param_name="input_dy")
    para_check.check_dtype(x_dtype, check_list, param_name="input_x")
    para_check.check_dtype(y_dtype, check_list, param_name="input_y")
    ins = classify([input_dy, input_x, input_y], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (g, x, y) in ins:
        with tbe.compute():
            dy_shape, x_shape, y_shape = shape_util.variable_shape([g, x, y])
            data_dy = tvm.placeholder(dy_shape, dtype=dy_dtype, name="dy_dtype")
            data_x = tvm.placeholder(x_shape, dtype=x_dtype, name="x_dtype")
            data_y = tvm.placeholder(y_shape, dtype=y_dtype, name="y_dtype")
            res = gelu_grad_compute(data_dy, data_x, data_y, output_z, kernel_name, impl_mode)
            tensors.append((data_dy, data_x, data_y, res))
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
