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
lp_norm_reduce
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    CONST_INF = 2147483647


# 'pylint: disable=invalid-name,unused-argument,too-many-locals,too-many-arguments
def lp_norm_reduce_inf_compute(abs_x, x_type, y, p, axes, keepdim, kernel_name):
    """
    Compute norm for p = "inf" or p = "-inf"
    When p equals inf, lp_norm_reduce equals the max absolute value of elements;
    when -inf, lp_norm_reduce equals the min absolute value of elements.
    """
    if p in ('inf', Constant.CONST_INF):
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

    return res


# 'pylint: disable=invalid-name,unused-argument,too-many-locals,too-many-arguments
def lp_norm_reduce0_compute(abs_x, x_type, y, axes, keepdim, kernel_name):
    """
    Compute norm for p = 0.
    When p = 0, lp_norm_reduce equals the number of nonzero-elements
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
    ele_tensor = tbe.vcmpsel(abs_x, tvm.const(0, dtype=abs_x.dtype), 'ne',
                             tvm.const(1, dtype=abs_x.dtype), tvm.const(0, dtype=abs_x.dtype))
    res = tbe.reduce_sum(ele_tensor, axis=axes, keepdims=keepdim)
    return res


# 'pylint: disable=invalid-name,unused-argument,too-many-locals,too-many-arguments
def lp_norm_reduce1_compute(abs_x, x_type, y, axes, keepdim, kernel_name):
    """
    Compute norm for p = 1.
    When p = 1, lp_norm_reduce equals the sum of elements' absolute value
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
    return res


# 'pylint: disable=invalid-name,unused-argument,too-many-locals,too-many-arguments
@register_operator_compute("LpNormReduce", op_mode="dynamic", support_fusion=True)
def lp_norm_reduce_compute(abs_x, x_type, y, p, axes, keepdim, kernel_name):
    """
    Compute norm for p >= 2.

    When p equals other int value, lp_norm_reduce = pow(sum(pow(abs(input),p)),1/p).
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
    res = tbe.reduce_sum(prod_x, axis=axes, keepdims=keepdim)
    return res


# 'pylint: disable=invalid-name,unused-argument,too-many-locals,too-many-arguments
@register_operator("LpNormReduce")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_LIST_INT, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def lp_norm_reduce(x, y, p=2, axes=None, keepdim=False, epsilon=1e-12, kernel_name="lp_norm_reduce"):
    """
    Computes norm for p equals 0, 1, 2, -inf, inf, or other integers.
    Parameters
    ----------
    x: dict
       The input dict, only support float16, float32.
       Required.
    y: dict
       The output dict, only support float16, float32.
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
                 Optional. Default: "lp_norm_reduce".
    Returns
    -------
    None
    """
    para_check.check_kernel_name(kernel_name)
    xtype_list = ["float16", "float32"]
    x_type = x.get("dtype").lower()
    x_shape = x.get("shape")
    para_check.check_dtype(x_type, xtype_list)
    para_check.check_shape(x_shape)
    p_inf_list = ("inf", "-inf")
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
            abs_data = tbe.vabs(input_data)

            if (p in p_inf_list) or (p == Constant.CONST_INF) or (p == -Constant.CONST_INF - 1):
                res = lp_norm_reduce_inf_compute(abs_data, x_type, y, p, _axes.get("value"), keepdim, kernel_name)
            elif p == 0:
                res = lp_norm_reduce0_compute(abs_data, x_type, y, _axes.get("value"), keepdim, kernel_name)
            elif p == 1:
                res = lp_norm_reduce1_compute(abs_data, x_type, y, _axes.get("value"), keepdim, kernel_name)
            else:
                res = lp_norm_reduce_compute(abs_data, x_type, y, p, _axes.get("value"), keepdim, kernel_name)

            if res.dtype != x_type:
                res = tbe.cast_to(res, dtype=x_type)

            tensors.append([input_data, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
