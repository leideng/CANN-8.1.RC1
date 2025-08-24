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
gelu
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import OpImplMode
from impl.util.util_common import check_op_impl_mode
from impl.util.util_soc_common import after_v200

FP16_MIN_V2 = -4.6
FP16_MAX_V2 = 4.6
DOUBLE_X = 2


# 'pylint: disable=too-many-locals
def _tanh_parameter_compute(placeholders):
    csvalue = tvm.const(0.044715, "float32")
    mul_0 = tbe.vmul(placeholders, placeholders)
    pow_0 = tbe.vmul(mul_0, placeholders)
    mul_1 = tbe.vmuls(pow_0, csvalue)
    result = tbe.vadd(placeholders, mul_1)

    return result


def high_performance_compute(input_x, output_y, kernel_name="gelu", impl_mode=None):
    """
    mathematical formula of gelu(x):
    sgn(x) = (x+0.000000000001)/|(x+0.000000000001)|
    gelu(x) = x*(sgn(x)*[(a/2)*(clip(|x|,max=-b) + b)^2 + 0.5] + 0.5)

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input input_x
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is fast_gelu_v2
    impl_mode: str
        impl_mode, default value is None

    Returns
    -------
     A TVM tensor same as input placeholders.
    """
    dtype = input_x.dtype
    has_improve_precision = False

    if dtype == "float16" and \
            tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        if impl_mode == OpImplMode.HIGH_PRECISION or (impl_mode is None):
            has_improve_precision = True
            input_x = tbe.cast_to(input_x, "float32")
    const_b = tvm.const(-1.769, "float32")
    const_b_ = tvm.const(1.769, "float32")
    const_a_half = tvm.const(-0.1444, "float32")
    const_c = tvm.const(0.7071, "float32")
    const_offset = tvm.const(0.000000000001, "float32")
    if has_improve_precision == False and dtype == "float16":
        const_offset = tvm.const(0.0000001, "float32")
    const_d = tvm.const(0.5, "float32")

    muls_0 = tbe.vmuls(input_x, const_c)
    abs_muls_0 = tbe.vabs(muls_0)
    max_abs_muls_0 = tbe.vmins(abs_muls_0, const_b_)
    vadds = tbe.vadds(max_abs_muls_0, const_b)
    temp = tbe.vmul(vadds, vadds)
    temp_0 = tbe.vmuls(temp, const_a_half)
    temp_0 = tbe.vadds(temp_0, const_d)
    x_adds = tbe.vadds(input_x, const_offset)
    abs_x = tbe.vabs(x_adds)
    if impl_mode == OpImplMode.HIGH_PERFORMANCE:
        vrec_abs = tbe.vrec(abs_x)
        cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
        if cce_product == "Ascend310P":
            vrec_abs = tbe.vmins(vrec_abs, 65504)
        sgn = tbe.vmul(x_adds, vrec_abs)
    else:
        sgn = tbe.vdiv(x_adds, abs_x)
    temp_1 = tbe.vmul(temp_0, sgn)
    temp_1 = tbe.vadds(temp_1, const_d)
    result = tbe.vmul(input_x, temp_1)

    if has_improve_precision:
        result = tbe.cast_to(result, "float16")

    return result


def tanh_compute_v2(input_x):
    """
    algorithm: tanh
    calculating data's tanh,y= (e^(2x)-1)/(e^(2x)+1)

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data

    Returns
    -------
    res : tvm.tensor
        the result of tanh
    """
    input_dtype = input_x.dtype
    min_fp_data = tvm.const(FP16_MIN_V2, dtype=input_dtype)
    max_fp_data = tvm.const(FP16_MAX_V2, dtype=input_dtype)

    input_x = tbe.vmins(input_x, max_fp_data)
    input_x = tbe.vmaxs(input_x, min_fp_data)

    power_val = tbe.vmuls(input_x, tvm.const(DOUBLE_X, input_dtype))
    exp_val = tbe.vexp(power_val)
    upper = tbe.vadds(exp_val, tvm.const(-1., input_dtype))
    down = tbe.vadds(exp_val, tvm.const(1., input_dtype))
    res = tbe.vdiv(upper, down)

    return res


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,no-member
def gelu_compute_v2(input_x, impl_mode):
    """
    gelu(x): x / (1 + exp(-sqrt(8/pi)(x + 0.044715*x^3)))
    """
    dtype = input_x.dtype
    ori_dtype = input_x.dtype

    if dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        if impl_mode == OpImplMode.HIGH_PRECISION or (impl_mode is None):
            input_x = tbe.cast_to(input_x, "float32")
            dtype = input_x.dtype
    
    const_0 = tvm.const(-1.5957691, dtype)
    const_1 = tvm.const(1.0, dtype)

    tanh_parameter = _tanh_parameter_compute(input_x)
    mul_0 = tbe.vmuls(tanh_parameter, const_0)

    temp = tbe.vexp(mul_0)
    temp = tbe.vadds(temp, const_1)
    res = tbe.vdiv(input_x, temp)

    if dtype != ori_dtype:
        res = tbe.cast_to(res, ori_dtype)

    return res


def gelu_compute_with_tanh(input_x, impl_mode):
    """
    mathematical formula of gelu(x):
    0.5*x*[1.0 + tanh(np.sqrt(2/np.pi)*(x + 0.044715*x*x*x))], np.sqrt(2/np.pi) : 0.797884
    """
    dtype = input_x.dtype.lower()
    check_list = ("float16")
    para_check.check_dtype(dtype, check_list)

    const_1 = tvm.const(1.0, dtype)
    const_2 = tvm.const(0.5, dtype)
    const_3 = tvm.const(0.044715, dtype)
    const_4 = tvm.const(0.797884, dtype)

    mul_0 = tbe.vmul(input_x, input_x)
    pow_0 = tbe.vmul(mul_0, input_x)
    mul_1 = tbe.vmuls(pow_0, const_3)
    add_0 = tbe.vadd(input_x, mul_1)
    mul_2 = tvm.compute(add_0.shape, lambda *indices: add_0[indices] * const_4,
                        name="mul_2", tag="elewise_binary_mul")
    tanh_0 = tanh_compute_v2(mul_2)
    add_res = tbe.vadds(tanh_0, const_1)
    mul_3 = tbe.vmuls(add_res, const_2)
    res = tbe.vmul(mul_3, input_x)
        
    return res


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,no-member
@register_operator_compute("Gelu", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def gelu_compute(input_x, output_y, kernel_name="gelu", impl_mode=None):
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
        return high_performance_compute(input_x, output_y, kernel_name, impl_mode)

    dtype = input_x.dtype

    if tbe_platform.intrinsic_check_support("Intrinsic_vtanh"):
        res = gelu_compute_with_tanh(input_x, impl_mode)
        return res
    elif (after_v200() 
        or (tbe_platform.get_soc_spec("SHORT_SOC_VERSION") == "Ascend310P" and dtype == "float16")):
        res = gelu_compute_v2(input_x, impl_mode)
        return res

    has_improve_precision = False

    if dtype == "float16" and \
            tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        if impl_mode == OpImplMode.HIGH_PRECISION or (impl_mode is None):
            has_improve_precision = True
            input_x = tbe.cast_to(input_x, "float32")
            dtype = input_x.dtype

    # `formula; gelu(x) = 0.5*x*(1.0+tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3))))`
    # `formula; tanh(y) = 2/(1+exp(-2y)) - 1`

    # simplize
    # `formula; gelu(x) = x/(1+e^(-y))`
    # the formula is y = 2*np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3))

    # to avoid overflow, keep exp negative
    # `formula; gelu(x) = x/(1+e^(-|y|)) * v_const`
    # the formula is y = 2*np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3))
    # `formula; v_const = 1 if x > 0  , e^y  if x < 0`
    # formula; `2*np.sqrt(2/np.pi)`
    const_0 = tvm.const(1.5957691, "float32")
    const_1 = tvm.const(1.0, "float32")
    const_2 = tvm.const(-1.0, "float32")
    const_3 = tvm.const(0.0, "float32")
    tanh_parameter = _tanh_parameter_compute(input_x)
    mul_0 = tbe.vmuls(tanh_parameter, const_0)
    # y
    mul_0_min = tbe.vmins(mul_0, const_3)
    if not tbe_platform.api_check_support("tbe.dsl.vexp", "float32") and dtype == "float32":
        mul_0_min_fp16 = tbe.cast_to(mul_0_min, "float16")
        right_mul = tbe.vexp(mul_0_min_fp16)
    else:
        right_mul = tbe.vexp(mul_0_min)
    right_mul_fp32 = tbe.cast_to(right_mul, dtype)

    # `abs(y)`
    mul_0_abs = tbe.vabs(mul_0)
    # `-abs(y)`
    mul_0_abs_neg = tbe.vmuls(mul_0_abs, const_2)

    # the formula is e^(-abs(y))
    if not tbe_platform.api_check_support("tbe.dsl.vexp", "float32") and dtype == "float32":
        mul_0_abs_neg_fp16 = tbe.cast_to(mul_0_abs_neg, "float16")
        mul_0_abs_neg_exp = tbe.vexp(mul_0_abs_neg_fp16)
    else:
        mul_0_abs_neg_exp = tbe.vexp(mul_0_abs_neg)
    mul_0_abs_neg_exp_fp32 = tbe.cast_to(mul_0_abs_neg_exp, dtype)

    # the formula is e^(-abs(y)) + 1
    mul_0_abs_neg_exp_add = tbe.vadds(mul_0_abs_neg_exp_fp32, const_1)
    left_mul = tbe.vdiv(input_x, mul_0_abs_neg_exp_add)

    result = tbe.vmul(left_mul, right_mul_fp32)

    if has_improve_precision:
        result = tbe.cast_to(result, "float16")

    return result


# 'pylint: disable=invalid-name
@register_operator("Gelu")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME, para_check.OPTION_ATTR_STR)
def gelu(x, y, kernel_name="gelu", impl_mode=OpImplMode.HIGH_PRECISION):
    """
    mathematical formula of gelu(x):
    gelu(x) = 0.5*x*(1.0+tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3))))
    tanh(y) = 2/(1+exp(-2y)) - 1
    convert gelu to result(x) =
     x/(1+e(-2*(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3)))))

    Parameters
    ----------
    x : dict
        shape and dtype of input, only support bfloat16 float16, float32
    y: dict
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
            res = gelu_compute(input_data, y, kernel_name, impl_mode=impl_mode)

            tensors.append([input_data, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}

    tbe.build(schedules, config)
