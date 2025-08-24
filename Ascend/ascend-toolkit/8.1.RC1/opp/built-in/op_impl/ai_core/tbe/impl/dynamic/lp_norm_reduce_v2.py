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
lp_norm_reduce_v2
The difference from lp_norm_reduce is that the attribute `p` is float type
"""

import math
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import OpTbeImplMode


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    CONST_INF = float('inf')
    CONST_DEC_INF = float(2147483647)
    CONST_NEG_INF = float('-inf')
    CONST_DEC_NEG_INF = float(-2147483648)
    CONST_LOOP_MIN = 2
    CONST_LOOP_MAX = 15


def method_check_and_cast(method, ori_type, data):
    """
    check the method support the fp32 dtype compution whether or not,
    and cast data to target dtype.
    """
    support_fp32 = tbe_platform.api_check_support(method, "float32")
    if ori_type == "float16" and support_fp32:
        data = tbe.cast_to(data, "float32")
    elif ori_type == "float32" and not support_fp32:
        if tbe_platform.api_check_support("tbe.cast_to", "f322f16"):
            data = tbe.cast_to(data, "float16")
        else:
            raise RuntimeError("Type of input x must be float16 since cast op donot support f322f16")
        data = tbe.cast_to(data, "float16")
    return data


# 'pylint: disable=invalid-name,unused-argument,too-many-locals,too-many-arguments,huawei-too-many-arguments
def lp_norm_reduce_v2_inf_compute(abs_x, x_type, y, p, axes, keepdim, kernel_name):
    """
    Compute norm for p = "inf" or p = "-inf"
    p == inf, lp_norm = max(abs(x))
    p == -inf, lp_norm = min(abs(x))
    """
    if p in ('inf', Constant.CONST_INF) or math.isclose(p, Constant.CONST_DEC_INF):
        abs_x = method_check_and_cast("tbe.dsl.reduce_max", x_type, abs_x)
        res = tbe.reduce_max(abs_x, axis=axes, keepdims=keepdim, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
    else:  # p is "-inf"
        abs_x = method_check_and_cast("tbe.dsl.reduce_min", x_type, abs_x)
        res = tbe.reduce_min(abs_x, axis=axes, keepdims=keepdim, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
    return res


# 'pylint: disable=invalid-name,unused-argument,too-many-locals,too-many-arguments,huawei-too-many-arguments
def lp_norm_reduce_v2_p0_compute(abs_x, x_type, y, axes, keepdim, kernel_name):
    """
    Compute norm for p = 0.
    When p == 0, lp_norm_reduce equals the number of nonzero-elements
    """
    abs_x = method_check_and_cast("tbe.dsl.vcmpsel", x_type, abs_x)
    ele_tensor = tbe.vcmpsel(abs_x, tvm.const(0, dtype=abs_x.dtype), 'eq',
                             tvm.const(0, dtype=abs_x.dtype), tvm.const(1, dtype=abs_x.dtype))
    res = tbe.reduce_sum(ele_tensor, axis=axes, keepdims=keepdim)
    return res


# 'pylint: disable=invalid-name,unused-argument,too-many-locals,too-many-arguments,huawei-too-many-arguments
def lp_norm_reduce_v2_p1_compute(abs_x, x_type, y, axes, keepdim, kernel_name):
    """
    Compute norm for p = 1.0.
    p == 1.0, lp_norm = sum(abs(x))
    """
    abs_x = method_check_and_cast("tbe.dsl.reduce_sum", x_type, abs_x)
    res = tbe.reduce_sum(abs_x, axis=axes, keepdims=keepdim)
    return res


# 'pylint: disable=invalid-name,unused-argument,too-many-locals,too-many-arguments,huawei-too-many-arguments
def _lp_norm_reduce_v2_compute(abs_x, x_type, y, p, axes, keepdim, kernel_name):
    """
    Compute norm for other float value.
    lp_norm_reduce = sum(pow(abs(input),p)).
    """
    if math.isclose(int(p), p) and p >= Constant.CONST_LOOP_MIN and p <= Constant.CONST_LOOP_MAX:
        abs_x = method_check_and_cast("tbe.dsl.vmul", x_type, abs_x)
        pow_x = abs_x
        for _ in range(1, int(p)):
            pow_x = tbe.vmul(pow_x, abs_x)
    else:
        abs_x = method_check_and_cast("tbe.dsl.vlog", x_type, abs_x)
        if not tbe_platform.api_check_support("tbe.dsl.vlog", "float32"):  # soc only support high performance mode
            log_x = tbe.vlog(abs_x)
        else:
            log_x = tbe.vlog(abs_x, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
        log_x = tbe.vmuls(log_x, tvm.const(p, log_x.dtype))
        pow_x = tbe.vexp(log_x)
    res = tbe.reduce_sum(pow_x, axis=axes, keepdims=keepdim)
    return res


# 'pylint: disable=invalid-name,unused-argument,too-many-locals,too-many-arguments,huawei-too-many-arguments
@register_operator_compute("LpNormReduceV2", op_mode="dynamic", support_fusion=True)
def lp_norm_reduce_v2_compute(input_data, x_type, y, p, axes, keepdim, kernel_name):
    """
    Select the calculation branch based on the value of p.
    """
    p_inf_list = ("inf", "-inf", Constant.CONST_INF, -Constant.CONST_INF, Constant.CONST_NEG_INF)
    
    if x_type == "bfloat16":
        input_data = tbe.cast_to(input_data, "float32")
    abs_data = tbe.vabs(input_data)

    if math.isclose(p, 2.0):
        res = _lp_norm_reduce_v2_compute(abs_data, x_type, y, p, axes, keepdim, kernel_name)
    elif math.isclose(p, 1.0):
        res = lp_norm_reduce_v2_p1_compute(abs_data, x_type, y, axes, keepdim, kernel_name)
    elif math.isclose(p, 0.0):
        res = lp_norm_reduce_v2_p0_compute(abs_data, x_type, y, axes, keepdim, kernel_name)
    elif p in p_inf_list or math.isclose(p, Constant.CONST_DEC_INF) or math.isclose(p, Constant.CONST_DEC_NEG_INF):
        res = lp_norm_reduce_v2_inf_compute(abs_data, x_type, y, p, axes, keepdim, kernel_name)
    else:
        res = _lp_norm_reduce_v2_compute(abs_data, x_type, y, p, axes, keepdim, kernel_name)

    if res.dtype != x_type:
        if x_type == "float16":
            res = tbe.cast_to(res, dtype=x_type)
        elif x_type == "bfloat16":
            res = tbe.round(res, dtype=x_type)
    return res


# 'pylint: disable=invalid-name,unused-argument,too-many-locals,too-many-arguments,huawei-too-many-arguments
@register_operator("LpNormReduceV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_LIST_INT, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def lp_norm_reduce_v2(x, y, p=2.0, axes=None, keepdim=False, epsilon=1e-12, kernel_name="lp_norm_reduce_v2"):
    """
    Computes norm for p equals 0, 1, 2, -inf, inf, or other float value.
    p == 2.0 (default)      sum(abs(x) ^ 2)
    p == inf                max(abs(x))
    p == -inf               min(abs(x))
    p == 0                  sum(x != 0), number of non zero elements
    p == 1.0                sum(abs(x))
    other int or float      sum(abs(x) ^ {p})

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
                 Optional. Default: "lp_norm_reduce_v2".
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
    x["rel_pos_to_reduce"] = "before"

    if epsilon is None:
        input_axis = {"shape": [-1], "rel_pos_to_reduce": "axis"}
    else:
        no_shape = len(x_shape)
        if len(axes) == 0 or axes is None:
            axes = list(range(no_shape))
        if isinstance(axes, int):
            axes = [axes]

        input_axis = {"shape": [len(axes)], "value": axes, "rel_pos_to_reduce": "axis"}

    schedules = []
    tensors = []
    ins = classify([x, input_axis], OpPatternMode.REDUCE, {"keepdims": keepdim is True})

    for (_x, _axes) in ins:
        with tbe.compute():
            shape_var_new = shape_util.variable_shape([_x, _axes], op_mode="reduce")[0]
            input_data = tvm.placeholder(shape_var_new, name="input_data", dtype=x_type)

            res = lp_norm_reduce_v2_compute(input_data, x_type, y, p, _axes.get("value"), keepdim, kernel_name)
            tensors.append([input_data, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
