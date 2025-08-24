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
lp_norm
"""

import math
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import OpTbeImplMode


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for Constant
    """
    CONST_INF = 2147483647
    CONST_EPSILON_FP16 = 1e-7
    CCE_PLAT = tbe_platform.get_soc_spec('SHORT_SOC_VERSION')


# 'pylint: disable=invalid-name,unused-argument,too-many-locals,too-many-arguments
def lp_norm_inf_compute(abs_x, x_type, y, p, axes, keepdim, kernel_name):
    """
    Compute norm for p = "inf" or p = "-inf"
    When p equals inf, lp_norm equals the max absolute value of elements;
    when -inf, lp_norm equals the min absolute value of elements.
    """
    if p in ("inf", Constant.CONST_INF):
        reduce_support_fp32 = tbe_platform.api_check_support("tbe.dsl.reduce_max", "float32")
        if x_type == "float16" and reduce_support_fp32:
            abs_x = tbe.cast_to(abs_x, "float32")
        elif x_type == "float32" and not reduce_support_fp32:
            cast_support_f322f16 = tbe_platform.api_check_support("tbe.cast_to", "f322f16")
            if cast_support_f322f16 and x_type == "float32":
                abs_x = tbe.cast_to(abs_x, "float16")
            else:
                raise RuntimeError("Type of input x must be float16 since cast op donot support f322f16")
        res = tbe.reduce_max(abs_x, axis=axes, keepdims=keepdim)
    else:
        # p is "-inf"
        reduce_support_fp32 = tbe_platform.api_check_support("tbe.dsl.reduce_min", "float32")
        if x_type == "float16" and reduce_support_fp32:
            abs_x = tbe.cast_to(abs_x, "float32")
        elif x_type == "float32" and not reduce_support_fp32:
            cast_support_f322f16 = tbe_platform.api_check_support("tbe.cast_to", "f322f16")
            if cast_support_f322f16 and x_type == "float32":
                abs_x = tbe.cast_to(abs_x, "float16")
            else:
                raise RuntimeError("Type of input x must be float16 since cast op cannot support f322f16")
        res = tbe.reduce_min(abs_x, axis=axes, keepdims=keepdim)

    if res.dtype != x_type:
        res = tbe.cast_to(res, dtype=x_type)
    return res


# 'pylint: disable=invalid-name,unused-argument,too-many-locals,too-many-arguments
def lp_norm0_compute(abs_x, x_type, y, axes, keepdim, kernel_name):
    """
    Compute norm for p = 0.
    When p = 0, lp_norm equals the number of nonzero-elements
    """
    mul_support_fp32 = tbe_platform.api_check_support("tbe.dsl.vmuls", "float32")
    if mul_support_fp32 and x_type == "float16":
        abs_x = tbe.cast_to(abs_x, "float32")
    elif not mul_support_fp32 and x_type == "float32":
        cast_support_f322f16 = tbe_platform.api_check_support("tbe.cast_to", "f322f16")
        if cast_support_f322f16:
            abs_x = tbe.cast_to(abs_x, "float16")
        else:
            raise RuntimeError("Type of input x must be float16 since cast op cannot support f322f16")
    zero_tensor = tbe.vmuls(abs_x, tvm.const(0, dtype=abs_x.dtype))
    one_tensor = tbe.vadds(zero_tensor, tvm.const(1, dtype=abs_x.dtype))
    ele_tensor = tbe.vcmpsel(abs_x, zero_tensor, 'ne', one_tensor, zero_tensor)
    res = tbe.reduce_sum(ele_tensor, axis=axes, keepdims=keepdim)
    if abs_x.dtype != x_type:
        res = tbe.cast_to(res, dtype=x_type)
    return res


# 'pylint: disable=invalid-name,unused-argument,too-many-locals,too-many-arguments
def lp_norm1_compute(abs_x, x_type, y, axes, keepdim, kernel_name):
    """
    Compute norm for p = 1.
    When p = 1, lp_norm equals the sum of elements' absolute value
    """
    sum_support_fp32 = tbe_platform.api_check_support("tbe.dsl.reduce_sum", "float32")
    if sum_support_fp32 and x_type == "float16":
        abs_x = tbe.cast_to(abs_x, "float32")
    elif not sum_support_fp32 and x_type == "float32":
        cast_support_f322f16 = tbe_platform.api_check_support("tbe.cast_to", "f322f16")
        if cast_support_f322f16:
            abs_x = tbe.cast_to(abs_x, "float16")
        else:
            raise RuntimeError("Type of input x must be float16 since cast op cannot support f322f16")

    res = tbe.reduce_sum(abs_x, axis=axes, keepdims=keepdim)
    if res.dtype != x_type:
        res = tbe.cast_to(res, dtype=x_type)
    return res


# 'pylint: disable=invalid-name,unused-argument,too-many-locals
def lp_norm2_compute(abs_x, x_type, y, axes, keepdim, kernel_name):
    """
    Compute norm for p = 2.
    For precision considering, separate it from lp_norm_compute without using vlog.
    """
    mul_support_fp32 = tbe_platform.api_check_support("tbe.dsl.vmul", "float32")
    if mul_support_fp32 and x_type == "float16":
        abs_x = tbe.cast_to(abs_x, "float32")
    elif not mul_support_fp32 and x_type == "float32":
        cast_support_f322f16 = tbe_platform.api_check_support("tbe.cast_to", "f322f16")
        if cast_support_f322f16:
            abs_x = tbe.cast_to(abs_x, "float16")
        else:
            raise RuntimeError("Type of input x must be float16 since cast op cannot support f322f16")
    pow_x = tbe.vmul(abs_x, abs_x)
    sum_pow = tbe.reduce_sum(pow_x, axis=axes, keepdims=keepdim)
    res = tbe.vsqrt(sum_pow, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
    if res.dtype != x_type:
        res = tbe.cast_to(res, dtype=x_type)
    return res


# 'pylint: disable=invalid-name,unused-argument,too-many-locals,too-many-arguments
@register_operator_compute("LpNorm", op_mode="dynamic", support_fusion=False)
def lp_norm_compute(abs_x, x_type, y, p, axes, keepdim, kernel_name):
    """
    Compute norm for p >= 3.

    When p equals other int value, lp_norm = pow(sum(pow(abs(input),p)),1/p).
    """
    mul_support_fp32 = tbe_platform.api_check_support("tbe.dsl.vmul", "float32")
    if mul_support_fp32 and x_type == "float16":
        abs_x = tbe.cast_to(abs_x, "float32")
    elif not mul_support_fp32 and x_type == "float32":
        cast_support_f322f16 = tbe_platform.api_check_support("tbe.cast_to", "f322f16")
        if cast_support_f322f16:
            abs_x = tbe.cast_to(abs_x, "float16")
        else:
            raise RuntimeError("Type of input x must be float16 since cast op cannot support f322f16")
    prod_x = abs_x
    for _ in range(1, p):
        prod_x = tbe.vmul(prod_x, abs_x)
    sum_prod_x = tbe.reduce_sum(prod_x, axis=axes, keepdims=keepdim)
    # extraction can be transformed like x^p =  y --> x = exp(log(y)/p)
    if "910" in Constant.CCE_PLAT:
        log_sum_x = tbe.vlog(sum_prod_x, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
    else:
        log_sum_x = tbe.vlog(sum_prod_x)
    zero_tensor = tbe.vmuls(log_sum_x, tvm.const(0, dtype=log_sum_x.dtype))
    p_tensor = tbe.vadds(zero_tensor, tvm.const(p, dtype=log_sum_x.dtype))
    div_log_x = tbe.vdiv(log_sum_x, p_tensor)
    exp_div_x = tbe.vexp(div_log_x)
    if exp_div_x.dtype != x_type:
        exp_div_x = tbe.cast_to(exp_div_x, dtype=x_type)
    return exp_div_x


# 'pylint: disable=too-many-arguments
# 'pylint: disable=too-many-branches
@register_operator("LpNorm")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_LIST_INT, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def lp_norm(x, y, p=2, axes=None, keepdim=False, epsilon=1e-12, kernel_name="lp_norm"):
    """
    Computes norm for p equals 0, 1, 2, -inf, inf, or other integers.
    Parameters
    ----------
    x: tensor
       The input tensor.0
       Required.
    y: tensor
       The output tensor.
       Required.
    p: int, inf, -inf
       The order of norm.
       Optional. Default: 2.
    axes: int list, None.
          The dimension on which the norm will be applied. None means all dimensions will be applied.
          Optional. Default: None.
    keepdim: bool
             Whether the output tensors should have dim keeped or not.
             Optional. Default: False
    epsilon: float
             The number used for safe considering as norm usually served as denominator.
             Optional. Default: 1e-7 for fp16, 1e-12 for fp32
    kernel_name: str
                 Kernel name.
                 Optional. Default: "lp_norm".
    Returns
    -------
    None
    """
    para_check.check_kernel_name(kernel_name)
    type_list = ["float16", "float32"]
    x_type = x.get("dtype").lower()
    x_shape = x.get("shape")
    para_check.check_dtype(x_type, type_list)
    para_check.check_shape(x_shape)
    p_inf_list = ("inf", "-inf")

    if isinstance(axes, int):
        axes = [axes]
    if axes is None:
        axes = list(range(len(x_shape)))
    if len(axes) == 0:
        axes = list(range(len(x_shape)))

    x["rel_pos_to_reduce"] = "before"

    input_axis = {"shape": [len(axes), ], "value": axes, "rel_pos_to_reduce": "axis"}
    schedules = []
    tensors = []
    ins = classify([x, input_axis], OpPatternMode.REDUCE, {"keepdims": keepdim is True})

    for (_x, axes_dict) in ins:
        with tbe.compute():
            shape_var_new = shape_util.variable_shape([_x, axes_dict], op_mode="reduce")[0]
            input_data = tvm.placeholder(shape_var_new, name="input_data", dtype=x_type)
            abs_data = tbe.vabs(input_data)

            if (p in p_inf_list) or (p == Constant.CONST_INF) or (p == -Constant.CONST_INF - 1):
                res = lp_norm_inf_compute(abs_data, x_type, y, p, axes_dict.get("value"), keepdim, kernel_name)
            elif p == 0:
                res = lp_norm0_compute(abs_data, x_type, y, axes_dict.get("value"), keepdim, kernel_name)
            elif p == 1:
                res = lp_norm1_compute(abs_data, x_type, y, axes_dict.get("value"), keepdim, kernel_name)
            elif p == 2:
                res = lp_norm2_compute(abs_data, x_type, y, axes_dict.get("value"), keepdim, kernel_name)
            else:
                res = lp_norm_compute(abs_data, x_type, y, p, axes_dict.get("value"), keepdim, kernel_name)

            if x_type == "float16" and float(epsilon) <= Constant.CONST_EPSILON_FP16:
                if math.isclose(epsilon, 0.0):
                    std_no = tvm.const(0.0, dtype=x_type)
                else:
                    std_no = tvm.const(Constant.CONST_EPSILON_FP16, dtype=x_type)
            else:
                std_no = tvm.const(float(epsilon), dtype=x_type)
            res = tbe.vmaxs(res, std_no)

            tensors.append([input_data, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
