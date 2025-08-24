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
lp_norm_update_v2
The difference from lp_norm_update is that the attribute `p` is float type
"""

import math
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import OpTbeImplMode


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    CONST_EPSILON_FP16 = 1e-7
    CONST_INF = float('inf')
    CONST_DEC_INF = float(2147483647)
    CONST_NEG_INF = float('-inf')
    CONST_DEC_NEG_INF = float(-2147483648)


# 'pylint: disable=invalid-name,unused-argument,too-many-locals
def _lp_norm_update_v2_compute(x, y, p, kernel_name):
    """
    Compute norm for p = 2.
    For precision considering, separate it from lp_norm_update_compute without using vlog.
    Compute norm for other float value.
    When p equals other float value, lp_norm_update = pow(x, 1/p).
    """
    # extraction can be transformed like x^p =  y --> x = exp(log(y)/p)
    x_type = x.dtype
    if math.isclose(p, 2.0):
        res = tbe.vsqrt(x, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
    else:
        if not tbe_platform.api_check_support("tbe.dsl.vlog", "float32"):
            if x.dtype == "float32":
                x = tbe.cast_to(x, "float16")
            log_x = tbe.vlog(x)
        else:
            log_x = tbe.vlog(x, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
        log_x = tbe.vmuls(log_x, tvm.const(1 / p, dtype=log_x.dtype))
        res = tbe.vexp(log_x)
    if res.dtype != x_type:
        res = tbe.cast_to(res, x_type)
    return res


# 'pylint: disable=invalid-name,unused-argument,too-many-locals,too-many-arguments,huawei-too-many-arguments
@register_operator_compute("LpNormUpdateV2", op_mode="dynamic", support_fusion=True)
def lp_norm_update_v2_compute(x, x_type, y, p, epsilon, kernel_name):
    """
    Select the calculation branch based on the value of p.
    """
    p_inf_list = ("inf", "-inf", Constant.CONST_INF, -Constant.CONST_INF, Constant.CONST_NEG_INF)

    if x_type == "bfloat16":
        x = tbe.cast_to(x, dtype="float32")
    
    if math.isclose(p, 0.0) or math.isclose(p, 1.0) \
        or (p in p_inf_list or math.isclose(p, Constant.CONST_DEC_INF) or math.isclose(p, Constant.CONST_DEC_NEG_INF)):
        res = x
    else:
        res = _lp_norm_update_v2_compute(x, y, p, kernel_name)

    if x_type == "float16" and float(epsilon) <= Constant.CONST_EPSILON_FP16:
        if math.isclose(epsilon, 0.0):
            std_no = tvm.const(0.0, dtype=x_type)
        else:
            std_no = tvm.const(Constant.CONST_EPSILON_FP16, dtype=x_type)
    else:
        std_no = tvm.const(float(epsilon), dtype="float16" if x_type == "float16" else "float32")
    res = tbe.vmaxs(res, std_no)

    if res.dtype != x_type:
        if x_type == "float16":
            res = tbe.cast_to(res, x_type)
        elif x_type == "bfloat16":
            res = tbe.round(res, x_type)
    return res


@register_operator("LpNormUpdateV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def lp_norm_update_v2(x, y, p=2, epsilon=1e-12, kernel_name="lp_norm_update_v2"):
    """
    Computes norm for p equals 0, 1, 2, -inf, inf, or other float value.
    p == 2.0 (default)          sqrt(x)
    p == inf, -inf, 0, 1.0      x
    other int or float          x ^ {1/p}

    Parameters
    ----------
    x: dict
       The input dict, only support float16, bfloat16, float32.
       Required.
    y: dict
       The output dict, only support float16, bfloat16, float32.
       Required.
    p: float, inf, -inf
       The order of norm.
       Optional. Default: 2.0.
    epsilon: float
             The number used for safe considering as norm usually served as denominator.
             Optional. Default: 1e-7 for fp16, 1e-12 for fp32
    kernel_name: str
                 Kernel name.
                 Optional. Default: "lp_norm_update_v2".
    Returns
    -------
    None
    """
    para_check.check_kernel_name(kernel_name)
    xtype_list = ["float16", "float32", "bfloat16"]
    x_type = x.get("dtype").lower()
    x_shape = x.get("shape")
    para_check.check_dtype(x_type, xtype_list)
    para_check.check_shape(x_shape)

    schedules = []
    tensors = []
    ins = classify([x], OpPatternMode.ELEWISE)

    for (_x,) in ins:
        with tbe.compute():
            shape_var_new = shape_util.variable_shape([_x, ])[0]
            input_data = tvm.placeholder(shape_var_new, name="input_data", dtype=x_type)

            res = lp_norm_update_v2_compute(input_data, x_type, y, p, epsilon, kernel_name)
            tensors.append([input_data, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
