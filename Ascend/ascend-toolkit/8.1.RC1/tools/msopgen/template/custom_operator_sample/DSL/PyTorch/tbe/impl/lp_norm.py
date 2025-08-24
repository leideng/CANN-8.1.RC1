# Copyright 2020 Huawei Technologies Co., Ltd
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

import te.lang.cce as tbe

from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check

_CONST_INF = 2147483647
_CONST_EPSILON_FP16 = 1e-7
_CCE_PLAT = tbe_platform.get_soc_spec('SHORT_SOC_VERSION')


# pylint: disable=invalid-name,unused-argument,unused-variable,too-many-locals
@register_operator_compute("LpNorm", op_mode="static", support_fusion=True)
def lp_norm_inf_compute(abs_x, x_type, y, p, axes, keepdim, kernel_name):
    """
    Compute norm for p = "inf" or p = "-inf"
    When p equals inf, lp_norm equals the max absolute value of elements;
    when -inf, lp_norm equals the min absolute value of elements.
    """
    if (p == "inf") or (p == _CONST_INF):
        reduce_support_fp32 = tbe_platform.api_check_support("tbe.dsl.reduce_max", "float32")
        if x_type == "float16" and reduce_support_fp32:
            abs_x = tbe.cast_to(abs_x, "float32")
        elif x_type == "float32" and not reduce_support_fp32:
            cast_support_f322f16 = tbe_platform.api_check_support("tbe.dsl.cast_to", "f322f16")
            if cast_support_f322f16 and x_type == "float32":
                abs_x = tbe.cast_to(abs_x, "float16")
            else:
                raise RuntimeError("Type of input x must be float16 since cast op donot support f322f16")
        res = tbe.reduce_max(abs_x, axis=axes, keepdims=keepdim, priority_flag=True)
    else:
        # p is "-inf"
        reduce_support_fp32 = tbe_platform.api_check_support("tbe.dsl.reduce_min", "float32")
        if x_type == "float16" and reduce_support_fp32:
            abs_x = tbe.cast_to(abs_x, "float32")
        elif x_type == "float32" and not reduce_support_fp32:
            cast_support_f322f16 = tbe_platform.api_check_support("tbe.dsl.cast_to", "f322f16")
            if cast_support_f322f16 and x_type == "float32":
                abs_x = tbe.cast_to(abs_x, "float16")
            else:
                raise RuntimeError("Type of input x must be float16 since cast op cannot support f322f16")
        res = tbe.reduce_min(abs_x, axis=axes, keepdims=keepdim, priority_flag=True)

    if res.dtype != x_type:
        res = tbe.cast_to(res, dtype=x_type)
    return res


def lp_norm0_compute(abs_x, x_type, y, axes, keepdim, kernel_name):
    """
    Compute norm for p = 0.
    When p = 0, lp_norm equals the number of nonzero-elements
    """
    mul_support_fp32 = tbe_platform.api_check_support("tbe.dsl.vmuls", "float32")
    if mul_support_fp32 and x_type == "float16":
        abs_x = tbe.cast_to(abs_x, "float32", False)
    elif not mul_support_fp32 and x_type == "float32":
        cast_support_f322f16 = tbe_platform.api_check_support("tbe.dsl.cast_to", "f322f16")
        if cast_support_f322f16:
            abs_x = tbe.cast_to(abs_x, "float16")
        else:
            raise RuntimeError("Type of input x must be float16 since cast op cannot support f322f16")
    zero_tensor = tbe.vmuls(abs_x, tvm.const(0, dtype=abs_x.dtype))
    one_tensor = tbe.vadds(zero_tensor, tvm.const(1, dtype=abs_x.dtype))
    ele_tensor = tbe.vcmpsel(abs_x, zero_tensor, 'ne', one_tensor, zero_tensor)
    res = tbe.sum(ele_tensor, axis=axes, keepdims=keepdim)
    if abs_x.dtype != x_type:
        res = tbe.cast_to(res, dtype=x_type)
    return res


def lp_norm1_compute(abs_x, x_type, y, axes, keepdim, kernel_name):
    """
    Compute norm for p = 1.
    When p = 1, lp_norm equals the sum of elements' absolute value
    """
    sum_support_fp32 = tbe_platform.api_check_support("tbe.dsl.reduce_sum", "float32")
    if sum_support_fp32 and x_type == "float16":
        abs_x = tbe.cast_to(abs_x, "float32", False)
    elif not sum_support_fp32 and x_type == "float32":
        cast_support_f322f16 = tbe_platform.api_check_support("tbe.dsl.cast_to", "f322f16")
        if cast_support_f322f16:
            abs_x = tbe.cast_to(abs_x, "float16")
        else:
            raise RuntimeError("Type of input x must be float16 since cast op cannot support f322f16")

    res = tbe.sum(abs_x, axis=axes, keepdims=keepdim)
    if res.dtype != x_type:
        res = tbe.cast_to(res, dtype=x_type)
    return res


def lp_norm2_compute(abs_x, x_type, y, axes, keepdim, kernel_name):
    """
    Compute norm for p = 2.
    For precision considering, separate it from lp_norm_compute without using vlog.
    """
    mul_support_fp32 = tbe_platform.api_check_support("tbe.dsl.vmul", "float32")
    if mul_support_fp32 and x_type == "float16":
        abs_x = tbe.cast_to(abs_x, "float32", False)
    elif not mul_support_fp32 and x_type == "float32":
        cast_support_f322f16 = tbe_platform.api_check_support("tbe.dsl.cast_to", "f322f16")
        if cast_support_f322f16:
            abs_x = tbe.cast_to(abs_x, "float16")
        else:
            raise RuntimeError("Type of input x must be float16 since cast op cannot support f322f16")
    pow_x = tbe.vmul(abs_x, abs_x)
    sum_pow = tbe.sum(pow_x, axis=axes, keepdims=keepdim)
    res = tbe.vsqrt(sum_pow, priority_flag=1)
    if res.dtype != x_type:
        res = tbe.cast_to(res, dtype=x_type)
    return res


def lp_norm_compute(abs_x, x_type, y, p, axes, keepdim, kernel_name):
    """
    Compute norm for p >= 3.

    When p equals other int value, lp_norm = pow(sum(pow(abs(input),p)),1/p).
    """
    mul_support_fp32 = tbe_platform.api_check_support("tbe.dsl.vmul", "float32")
    if mul_support_fp32 and x_type == "float16":
        abs_x = tbe.cast_to(abs_x, "float32", False)
    elif not mul_support_fp32 and x_type == "float32":
        cast_support_f322f16 = tbe_platform.api_check_support("tbe.dsl.cast_to", "f322f16")
        if cast_support_f322f16:
            abs_x = tbe.cast_to(abs_x, "float16")
        else:
            raise RuntimeError("Type of input x must be float16 since cast op cannot support f322f16")
    prod_x = abs_x
    for p_ix in range(1, p):
        prod_x = tbe.vmul(prod_x, abs_x)
    sum_prod_x = tbe.sum(prod_x, axis=axes, keepdims=keepdim)
    # extraction can be transformed like x^p =  y --> x = exp(log(y)/p)
    if "910" in _CCE_PLAT:
        log_sum_x = tbe.vlog(sum_prod_x, priority_flag=1)
    else:
        log_sum_x = tbe.vlog(sum_prod_x)
    zero_tensor = tbe.vmuls(log_sum_x, tvm.const(0, dtype=log_sum_x.dtype))
    p_tensor = tbe.vadds(zero_tensor, tvm.const(p, dtype=log_sum_x.dtype))
    div_log_x = tbe.vdiv(log_sum_x, p_tensor)
    exp_div_x = tbe.vexp(div_log_x)
    if exp_div_x.dtype != x_type:
        exp_div_x = tbe.cast_to(exp_div_x, dtype=x_type)
    return exp_div_x


def lp_norm(x, y, p=2, axes=None, keepdim=False, epsilon=1e-12, kernel_name="lp_norm"):
    """
    Computes norm for p equals 0, 1, 2, -inf, inf, or other integers.
    Parameters
    ----------
    x: tensor
       The input tensor.
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
    xtype_list = ["float16", "float32"]
    x_type = x.get("dtype").lower()
    x_shape = x.get("shape")
    para_check.check_dtype(x_type, xtype_list)
    para_check.check_shape(x_shape)
    p_inf_list = ("inf", "-inf")
    no_shape = len(x_shape)
    if isinstance(axes, int):
        axes = [axes]
    if axes is None:
        axes = [i for i in range(no_shape)]
    if len(axes) == 0:
        axes = [i for i in range(no_shape)]
    input_data = tvm.placeholder(x_shape, dtype=x_type, name="input_data")
    abs_data = tbe.vabs(input_data)

    if (p in p_inf_list) or (p == _CONST_INF) or (p == -_CONST_INF - 1):
        res = lp_norm_inf_compute(abs_data, x_type, y, p, axes, keepdim, kernel_name)
    elif p == 0:
        res = lp_norm0_compute(abs_data, x_type, y, axes, keepdim, kernel_name)
    elif p == 1:
        res = lp_norm1_compute(abs_data, x_type, y, axes, keepdim, kernel_name)
    elif p == 2:
        res = lp_norm2_compute(abs_data, x_type, y, axes, keepdim, kernel_name)
    else:
        res = lp_norm_compute(abs_data, x_type, y, p, axes, keepdim, kernel_name)

    if x_type == "float16" and float(epsilon) <= _CONST_EPSILON_FP16:
        std_no = tvm.const(_CONST_EPSILON_FP16, dtype=x_type)
    else:
        std_no = tvm.const(float(epsilon), dtype=x_type)
    res = tbe.vmaxs(res, std_no)
    with tvm.target.cce():
        schedule = tbe.auto_schedule(res)
    config = {"name": kernel_name,
              "tensor_list": [input_data, res]}
    tbe.build(schedule, config)
