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
erfinv
"""
from functools import reduce as functools_reduce
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    SCALER_ONE = 1
    SCALER_NEGATIVE_ONE = -1
    SCALER_P = 0.47047
    SCALER_A = 0.3480242
    SCALER_B = -0.0958798
    SCALER_C = 0.7478556
    SCALER_FP16_MAX = 32768
    SCALER_FP16_MIN = 2 ** (-15)

    # The ratio needed for numerical calculation.
    # The detailed calculation process will be given in the code comments below.
    a = (0.886226899, -1.645349621, 0.914624893, -0.140543331)
    b = (-2.118377725, 1.442710462, -0.329097515, 0.012229801)
    c = (-1.970840454, -1.624906493, 3.429567803, 1.641345311)
    d = (3.543889200, 1.637067800)

    CENTRAL_RANGE = 0.7
    PI = 3.1415926535
    TWODIVPI = 1.1283791670955


@register_operator_compute("erfinv", op_mode="static", support_fusion=True)
# 'pylint: disable=unused-argument
def erfinv_compute(input_x, output_y, kernel_name="erfinv"):
    """
    calculating data

    Parameters
    ----------
    grads : TVM tensor
        the placeholder of grads
    input_x : TVM tensor
        the placeholder of input_x
    output_y : dict
        dict of output_y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "hard_sigmoid_grad"

    Returns
    -------
    output tensor
    """
    dtype = input_x.dtype
    if dtype == "float16":
        input_x = tbe.cast_to(input_x, "float32")

    x_abs = tbe.vabs(input_x)
    x_lt_range = tbe.vcmp(x_abs, Constant.CENTRAL_RANGE, 'le', 'bool')

    # yl is the value of y when input_x <= CENTRAL_RANGE
    yl = cal_yl(input_x)

    # yg is the value of y when input_x > CENTRAL_RANGE
    yg = cal_yg(input_x)

    y = tbe.vsel(x_lt_range, yl, yg)

    # Two steps of Newton-Raphson correction
    for _ in range(0, 2):
        erf_result = erf(y)

        num = tbe.vsub(erf_result, input_x)
        temp = tbe.vmul(y, y)
        temp = tbe.vmuls(temp, -1)
        temp = tbe.vexp(temp)
        dem = tbe.vmuls(temp, Constant.TWODIVPI)
        crt = tbe.vdiv(num, dem)
        y = tbe.vsub(y, crt)

    if dtype == "float16":
        y = tbe.cast_to(y, dtype)
    return y


# 'pylint: disable=too-many-locals
def cal_yl(input_x):
    """
    calculating data

    Parameters
    ----------
    grads : TVM tensor
        the placeholder of grads
    input_x : TVM tensor
        the placeholder of input_x

    Returns
    -------
    output tensor
    """
    const_one = tvm.const(Constant.SCALER_ONE, dtype="float32")
    zl = tbe.vmul(input_x, input_x)

    # `numl = ((a[3]*z + a[2])*z + a[1])*z + a[0]`
    erfinv_vmuls_a = tbe.vmuls(zl, Constant.a[3])
    erfinv_vadds_a = tbe.vadds(erfinv_vmuls_a, Constant.a[2])
    erfinv_square_vmul_a = tbe.vmul(erfinv_vadds_a, zl)
    erfinv_square_vadds_a = tbe.vadds(erfinv_square_vmul_a, Constant.a[1])
    erfinv_cube_vmul_a = tbe.vmul(erfinv_square_vadds_a, zl)
    numl = tbe.vadds(erfinv_cube_vmul_a, Constant.a[0])

    # `deml = (((b[3]*z + b[2])*z + b[1])*z + b[0])*z + 1`
    erfinv_vmuls_b = tbe.vmuls(zl, Constant.b[3])
    erfinv_vadds_b = tbe.vadds(erfinv_vmuls_b, Constant.b[2])
    erfinv_square_vmul_b = tbe.vmul(erfinv_vadds_b, zl)
    erfinv_square_vadds_b = tbe.vadds(erfinv_square_vmul_b, Constant.b[1])
    erfinv_cube_vmul_b = tbe.vmul(erfinv_square_vadds_b, zl)
    erfinv_cube_vadds_b = tbe.vadds(erfinv_cube_vmul_b, Constant.b[0])
    erfinv_power4_vmul_b = tbe.vmul(erfinv_cube_vadds_b, zl)
    deml = tbe.vadds(erfinv_power4_vmul_b, const_one)

    # `yl = input_x * numl / deml`
    xnuml = tbe.vmul(input_x, numl)
    yl = tbe.vdiv(xnuml, deml)
    return yl


# 'pylint: disable=too-many-locals
def cal_yg(input_x):
    """
    calculating data

    Parameters
    ----------
    grads : TVM tensor
        the placeholder of grads
    input_x : TVM tensor
        the placeholder of input_x

    Returns
    -------
    output tensor
    """
    dtype = input_x.dtype
    const_one = tvm.const(Constant.SCALER_ONE, dtype="float32")
    const_negative_one = tvm.const(Constant.SCALER_NEGATIVE_ONE, dtype="float32")
    x_abs = tbe.vabs(input_x)
    fp16_max = tvm.const(Constant.SCALER_FP16_MAX, dtype=dtype)
    fp16_min = tvm.const(Constant.SCALER_FP16_MIN, dtype=dtype)

    # `zg = sqrt(-log((1-|x|)/2))`
    x_abs_minus_one = tbe.vadds(x_abs, const_negative_one)
    data_neg = tbe.vmuls(x_abs_minus_one, -1)
    mul_data = tbe.vmuls(data_neg, 0.5)
    data_vlog = tbe.vlog(mul_data, 1)
    zg_square = tbe.vabs(data_vlog)
    zg = tbe.vsqrt(zg_square, 1)

    # `numg = ((c[3]*z + c[2])*z + c[1])*z + c[0]`
    zg_vmuls_c3 = tbe.vmuls(zg, Constant.c[3])
    lr_vadds_c2 = tbe.vadds(zg_vmuls_c3, Constant.c[2])
    lr_vmul_zg = tbe.vmul(lr_vadds_c2, zg)
    lr_vadds_c1 = tbe.vadds(lr_vmul_zg, Constant.c[1])
    lr_vmul_zg = tbe.vmul(lr_vadds_c1, zg)
    numg = tbe.vadds(lr_vmul_zg, Constant.c[0])

    # `demg = (d[1]*z + d[0])*z + 1`
    zg_vmuls_d1 = tbe.vmuls(zg, Constant.d[1])
    lr_vadds_d0 = tbe.vadds(zg_vmuls_d1, Constant.d[0])
    lr_vmul_zg = tbe.vmul(lr_vadds_d0, zg)
    demg = tbe.vadds(lr_vmul_zg, const_one)

    data_vmuls = tbe.vmuls(input_x, fp16_max)
    data_abs = tbe.vabs(data_vmuls)
    data_vadds = tbe.vadds(data_abs, fp16_min)
    data_div = tbe.vdiv(data_vmuls, data_vadds)
    data_round = tbe.round(data_div)
    tensor_sign = tbe.cast_to(data_round, dtype)

    # `yg = copysign(numg, input_x) / demg`
    numg_sign = tbe.vmul(numg, tensor_sign)
    yg = tbe.vdiv(numg_sign, demg)
    return yg


# 'pylint: disable=too-many-locals
def erf(input_x):
    """
    calculating data

    Parameters
    ----------
    grads : TVM tensor
        the placeholder of grads
    input_x : TVM tensor
        the placeholder of input_x

    Returns
    -------
    output tensor
    """
    dtype = input_x.dtype
    shape = shape_util.shape_to_list(input_x.shape)
    const_one = tvm.const(Constant.SCALER_ONE, dtype="float32")
    const_negative_one = tvm.const(Constant.SCALER_NEGATIVE_ONE, dtype="float32")
    const_p = tvm.const(Constant.SCALER_P, dtype="float32")
    const_a = tvm.const(Constant.SCALER_A, dtype="float32")
    const_b = tvm.const(Constant.SCALER_B, dtype="float32")
    const_c = tvm.const(Constant.SCALER_C, dtype="float32")
    fp16_max = tvm.const(Constant.SCALER_FP16_MAX, dtype=dtype)
    fp16_min = tvm.const(Constant.SCALER_FP16_MIN, dtype=dtype)

    data_vmuls = tbe.vmuls(input_x, fp16_max)
    data_abs = tbe.vabs(data_vmuls)
    data_vadds = tbe.vadds(data_abs, fp16_min)
    data_div = tbe.vdiv(data_vmuls, data_vadds)
    data_round = tbe.round(data_div)
    tensor_sign = tbe.cast_to(data_round, dtype)

    tensor_one = tbe.broadcast(const_one, shape, "float32")
    tensor_abs = tbe.vabs(input_x)
    erf_t_vmuls = tbe.vmuls(tensor_abs, const_p)
    erf_t_vadds = tbe.vadds(erf_t_vmuls, const_one)
    erf_data_t = tbe.vdiv(tensor_one, erf_t_vadds)

    erf_abs_square = tbe.vmul(tensor_abs, tensor_abs)
    erf_data_vmuls = tbe.vmuls(erf_abs_square, const_negative_one)
    erf_data_exp = tbe.vexp(erf_data_vmuls)

    erf_data_t_square = tbe.vmul(erf_data_t, erf_data_t)
    erf_data_t_cube = tbe.vmul(erf_data_t, erf_data_t_square)

    erf_t_vmuls = tbe.vmuls(erf_data_t, const_a)
    erf_t_square_vmuls = tbe.vmuls(erf_data_t_square, const_b)
    erf_t_cube_vmuls = tbe.vmuls(erf_data_t_cube, const_c)

    erf_square_vadd = tbe.vadd(erf_t_vmuls, erf_t_square_vmuls)
    erf_cube_vadd_ = tbe.vadd(erf_square_vadd, erf_t_cube_vmuls)
    erf_cube_vmuls = tbe.vmuls(erf_cube_vadd_, const_negative_one)
    erf_exp_vmul = tbe.vmul(erf_cube_vmuls, erf_data_exp)
    erf_exp_vadds = tbe.vadds(erf_exp_vmul, const_one)
    erf_result = tbe.vmul(tensor_sign, erf_exp_vadds)

    return erf_result


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
        shape and dtype of output, if shape of input_x is should be
    kernel_name : str
        kernel name, default value is "pdist"

    Returns
    -------
    None
    """
    shape_input = input_x.get("shape")
    dtype_input = input_x.get("dtype")

    para_check.check_kernel_name(kernel_name)
    para_check.check_shape_rule(shape_input)
    para_check.check_tensor_shape_size(shape_input)

    dtype_input = dtype_input.lower()
    check_list = ("float16", "float32")
    para_check.check_dtype_rule(dtype_input, check_list)

    shape_input = shape_util.shape_refine(shape_input)
    reshape_input = (functools_reduce(lambda x, y: x * y, shape_input[:]),)
    data_input = tvm.placeholder(reshape_input, name="data_input",
                                 dtype=dtype_input)

    erf_result = erfinv_compute(data_input, output_y, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(erf_result)

    config = {"name": kernel_name,
              "tensor_list": [data_input, erf_result]}

    tbe.cce_build_code(sch, config)
