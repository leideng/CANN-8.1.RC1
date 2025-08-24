"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

fast_gelu
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import OpImplMode
from impl.util.util_common import check_op_impl_mode
from impl.util.util_soc_common import support_inf_nan


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,no-member
# 'pylint: disable=too-many-locals,unused-variable
@register_operator_compute("FastGelu", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def fast_gelu_compute(input_x, output_y, kernel_name="fast_gelu", impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    mathematical formula of fast_gelu(x):
    For Atlas 200/300/500 Inference Product, Atlas Training Series Product, 
    Atlas Inference Series Product(Ascend 310P Processor), Ascend 610 AI Processor, 
    the calculation formula: fast_gelu(x) = xe^(0.851x)(x-|x|)/(1+e^(-1.702|x|)).
    For other chips, the calculation formula: fast_gelu(x) = x/(1+e^(-1.702x)).

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input input_x
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is fast_gelu

    Returns
    -------
     A TVM tensor same as input placeholders.
    """
    # define const value
    muls_const_value = -1.702
    adds_const_value = 1
    ori_dtype = input_x.dtype.lower()
    calc_dtype = input_x.dtype.lower()

    vexp_float32_support = tbe_platform.api_check_support("tbe.dsl.vexp", "float32")
    # some socs not support float32 vexp calculation, cast to float16
    if not vexp_float32_support and ori_dtype == "float32":
        calc_dtype = "float16"
        input_x = tbe.cast_to(input_x, calc_dtype)

    is_promote_dtype = ori_dtype == "bfloat16"
    if is_promote_dtype:
        calc_dtype = "float32"
        input_x = tbe.cast_to(input_x, calc_dtype)

    # define tvm const with calc dtype
    muls_const = tvm.const(muls_const_value, calc_dtype)
    adds_const = tvm.const(adds_const_value, calc_dtype)
    if support_inf_nan():
        # use formula_2 for performance and precision balance
        muls_x = tbe.vmuls(input_x, muls_const)
        exp_x = tbe.vexp(muls_x)
        adds_x = tbe.vadds(exp_x, adds_const)
        # vdiv perf is better than vrec + vmul
        res = tbe.vdiv(input_x, adds_x)
    else:
        # use formula_1 for no overflow calculation
        exp_half_value = 0.851
        exp_half_const = tvm.const(exp_half_value, calc_dtype)
        abs_x = tbe.vabs(input_x)
        mul_abs_x = tbe.vmuls(abs_x, muls_const)
        exp_abs_x = tbe.vexp(mul_abs_x)
        div_down = tbe.vadds(exp_abs_x, adds_const)
        pn_x = tbe.vsub(input_x, abs_x)
        mul_pn_x = tbe.vmuls(pn_x, exp_half_const)
        exp_pn_x = tbe.vexp(mul_pn_x)
        div_up = tbe.vmul(input_x, exp_pn_x)
        # vrec + vmul perf is better than vdiv
        div_down_rec = tbe.vrec(div_down, impl_mode=impl_mode)
        res = tbe.vmul(div_up, div_down_rec)
    # cast to ori_dtype for output
    if ori_dtype != calc_dtype:
        res = tbe.cast_to(res, ori_dtype)
    return res


@register_operator("FastGelu")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME,
                            para_check.OPTION_ATTR_STR)
def fast_gelu(input_x, output_y, kernel_name="fast_gelu", impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    mathematical formula of fast_gelu(x):
    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is fast_fast_gelu

    Returns
    -------
    none.
    """
    check_op_impl_mode(impl_mode, [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION], kernel_name)

    shape = input_x.get("shape")
    para_check.check_shape(shape, param_name="input_x")

    check_list = ("float16", "float32", "bfloat16")
    input_dtype = input_x.get("dtype").lower()
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")

    ins = classify([input_x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_input_x,) in ins:
        with tbe.compute():
            shape = shape_util.variable_shape([_input_x])
            data = tvm.placeholder(shape[0], name="data", dtype=input_dtype)
            result = fast_gelu_compute(data, output_y, kernel_name, impl_mode)

            tensors.append([data, result])
        with tvm.target.cce():
            sch = tbe.auto_schedule(result)
        schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}

    tbe.build(schedules, config)
