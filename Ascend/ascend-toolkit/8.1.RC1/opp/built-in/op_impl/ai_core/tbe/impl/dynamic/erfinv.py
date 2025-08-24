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
dynamic erfinv
"""
from functools import reduce as functools_reduce
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpTbeImplMode


# 'pylint: disable=too-few-public-methods
class Constant(object):
    """
    The class for constant
    """
    SCALER_ONE = 1
    SCALER_NEGATIVE_ONE = -1
    SCALER_FP16_MAX = 2 ** (15)
    SCALER_FP16_MIN = 2 ** (-15)
    SCALER_P = 0.3275911
    CENTRAL_RANGE = 0.7
    PI = 3.1415926535
    TWODIVPI = 1.1283791670955

    # The ratio needed for numerical calculation.
    # The detailed calculation process will be given in the code comments below.
    a = (0.886226899, -1.645349621, 0.914624893, -0.140543331)
    b = (-2.118377725, 1.442710462, -0.329097515, 0.012229801)
    c = (-1.970840454, -1.624906493, 3.429567803, 1.641345311)
    d = (3.543889200, 1.637067800)
    erf_scaler = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)


# 'pylint: disable=unused-argument
@register_operator_compute("Erfinv", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def erfinv_compute(input_x, output_y, kernel_name="erfinv"):
    """
    calculating data

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "erfinv"

    Returns
    -------
    output tensor
    """
    dtype = input_x.dtype
    if dtype == "float16":
        input_x = tbe.cast_to(input_x, "float32")

    x_abs = tbe.vabs(input_x)

    # yg is the value of y when input_x > CENTRAL_RANGE
    yg = cal_yg(input_x)

    # yl is the value of y when input_x <= CENTRAL_RANGE
    yl = cal_yl(input_x)

    if tbe_platform.api_check_support("te.lang.cce.vcmpsel", "float32"):
        y = tbe.vcmpsel(x_abs, Constant.CENTRAL_RANGE, 'le', yl, yg)
    else:
        x_abs = tbe.cast_to(x_abs, 'float16')
        yl = tbe.cast_to(yl, 'float16')
        yg = tbe.cast_to(yg, 'float16')
        y = tbe.vcmpsel(x_abs, Constant.CENTRAL_RANGE, 'le', yl, yg)
        y = tbe.cast_to(y, 'float32')

    # Two steps of Newton-Raphson correction
    for _ in range(0, 2):
        erf_result = erf(y)
        num = tbe.vsub(erf_result, input_x)
        temp = tbe.vmul(y, y)
        temp = tbe.vmuls(temp, Constant.SCALER_NEGATIVE_ONE)
        if tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
            temp = tbe.vexp(temp)
        else:
            temp = tbe.cast_to(temp, 'float16')
            temp = tbe.vexp(temp)
            temp = tbe.cast_to(temp, 'float32')
        dem = tbe.vmuls(temp, Constant.TWODIVPI)
        crt = tbe.vdiv(num, dem)
        y = tbe.vsub(y, crt)
    
    if dtype == "float16" or not tbe_platform.api_check_support("te.lang.cce.vcmpsel", "float32"):
        y = tbe.cast_to(y, 'float16')
        input_x = tbe.cast_to(input_x, 'float16')
    y = tbe.vcmpsel(input_x, Constant.SCALER_ONE, 'eq', float('inf'), y)
    y = tbe.vcmpsel(input_x, Constant.SCALER_NEGATIVE_ONE, 'eq', float('-inf'), y)
    y = tbe.vcmpsel(input_x, Constant.SCALER_NEGATIVE_ONE, 'lt', float('nan'), y)
    y = tbe.vcmpsel(input_x, Constant.SCALER_ONE, 'gt', float('nan'), y)
    
    if y.dtype != dtype:
        y = tbe.cast_to(y, dtype)
    return y


def copy_sign(input_x):
    data_vmuls = tbe.vmuls(input_x, Constant.SCALER_FP16_MAX)
    data_abs = tbe.vabs(data_vmuls)
    data_vadds = tbe.vadds(data_abs, Constant.SCALER_FP16_MIN)
    data_div = tbe.vdiv(data_vmuls, data_vadds)
    if not tbe_platform.api_check_support("te.lang.cce.round", "float32"):
        data_div = tbe.cast_to(data_div, 'float16')
    data_round = tbe.round(data_div)
    tensor_sign = tbe.cast_to(data_round, input_x.dtype)
    return tensor_sign


def mul_add(s, x):
    res = tbe.broadcast(tvm.const(s[-1], dtype="float32"), x.shape)
    for i in range(len(s) - 1):
        tmp = tbe.vmul(x, res)
        res = tbe.vadds(tmp, s[-2 - i])
    return res
    

# 'pylint: disable=too-many-locals
def cal_yl(input_x):
    """
    calculating data

    numl = ((a[3]*z + a[2])*z + a[1])*z + a[0]
    deml = (((b[3]*z + b[2])*z + b[1])*z + b[0])*z + 1
    yl = input_x * numl / deml

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x

    Returns
    -------
    output tensor
    """
    zl = tbe.vmul(input_x, input_x)

    # `deml = (((b[3]*z + b[2])*z + b[1])*z + b[0])*z + 1`
    tmp = tbe.vmul(mul_add(Constant.b, zl), zl)
    deml = tbe.vadds(tmp, Constant.SCALER_ONE)

    # `numl = ((a[3]*z + a[2])*z + a[1])*z + a[0]`
    numl = mul_add(Constant.a, zl)

    # `yl = input_x * numl / deml`
    xnuml = tbe.vmul(input_x, numl)
    yl = tbe.vdiv(xnuml, deml)
    return yl


# 'pylint: disable=too-many-locals
def cal_yg(input_x):
    """
    calculating data

    zg = sqrt(-log((1-|x|)/2))
    numg = ((c[3]*z + c[2])*z + c[1])*z + c[0]
    demg = (d[1]*z + d[0])*z + 1
    yg = copysign(numg, input_x) / demg

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x

    Returns
    -------
    output tensor
    """
    # `zg = sqrt(-log((1-|x|)/2))`
    x_abs = tbe.vabs(input_x)
    x_abs_minus_one = tbe.vadds(x_abs, Constant.SCALER_NEGATIVE_ONE)
    data_neg = tbe.vmuls(x_abs_minus_one, -1)
    mul_data = tbe.vmuls(data_neg, 0.5)
    if tbe_platform.api_check_support("te.lang.cce.vlog", "float32"):
        data_vlog = tbe.vlog(mul_data, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
    else:
        mul_data = tbe.cast_to(mul_data, 'float16')
        data_vlog = tbe.vlog(mul_data, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
        data_vlog = tbe.cast_to(data_vlog, 'float32')
    zg_square = tbe.vabs(data_vlog)
    zg = tbe.vsqrt(zg_square, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)

    # `demg = (d[1]*z + d[0])*z + 1`
    tmp = tbe.vmul(mul_add(Constant.d, zg), zg)
    demg = tbe.vadds(tmp, Constant.SCALER_ONE)

    # `numg = ((c[3]*z + c[2])*z + c[1])*z + c[0]`
    numg = mul_add(Constant.c, zg)

    # `yg = copysign(numg, input_x) / demg`
    numg_sign = tbe.vmul(numg, copy_sign(input_x))
    yg = tbe.vdiv(numg_sign, demg)
    return yg


# 'pylint: disable=too-many-locals
def erf(input_x):
    """
    calculating data

    t = 1.0/(1.0 + p*x)
    y = copysign * [1.0 - (((((e*t + d)*t) + c)*t + b)*t + a)*t*exp(-x*x)]

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x

    Returns
    -------
    output tensor
    """
    # `t = 1.0/(1.0 + p*x)`
    tensor_abs = tbe.vabs(input_x)
    tensor_one = tbe.broadcast(Constant.SCALER_ONE, input_x.shape, "float32")
    erf_t_vmuls = tbe.vmuls(tensor_abs, Constant.SCALER_P)
    erf_t_vadds = tbe.vadds(erf_t_vmuls, Constant.SCALER_ONE)
    erf_data_t = tbe.vdiv(tensor_one, erf_t_vadds)

    # `y = 1.0 - (((((e*t + d)*t) + c)*t + b)*t + a)*t*exp(-x*x)`
    erf_abs_square = tbe.vmul(tensor_abs, tensor_abs)
    erf_data_vmuls = tbe.vmuls(erf_abs_square, Constant.SCALER_NEGATIVE_ONE)
    if tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        erf_data_exp = tbe.vexp(erf_data_vmuls)
    else:
        erf_data_vmuls = tbe.cast_to(erf_data_vmuls, 'float16')
        erf_data_exp = tbe.vexp(erf_data_vmuls)
        erf_data_exp = tbe.cast_to(erf_data_exp, 'float32')
    erf_tmp = tbe.vmul(mul_add(Constant.erf_scaler, erf_data_t), erf_data_t)
    erf_negative_tmp = tbe.vmuls(erf_tmp, Constant.SCALER_NEGATIVE_ONE)
    erf_exp_vmul = tbe.vmul(erf_negative_tmp, erf_data_exp)
    erf_exp_vadds = tbe.vadds(erf_exp_vmul, Constant.SCALER_ONE)
    
    erf_result = tbe.vmul(copy_sign(input_x), erf_exp_vadds)
    return erf_result


@register_operator("Erfinv")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def erfinv(input_x, output_y, kernel_name="erfinv"):
    """
    calculating data

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output
    kernel_name : str
        kernel name, default value is "erfinv"

    Returns
    -------
    None
    """
    dtype_input = input_x.get("dtype").lower()
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype_rule(dtype_input, check_list)
    para_check.check_kernel_name(kernel_name)

    ins = classify([input_x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])
            reshape_input = (functools_reduce(lambda x, y: x * y, x_shape[0]),)
            data_input = tvm.placeholder(reshape_input, name="data_input",
                                         dtype=dtype_input)
            res = erfinv_compute(data_input, output_y, kernel_name)

            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
