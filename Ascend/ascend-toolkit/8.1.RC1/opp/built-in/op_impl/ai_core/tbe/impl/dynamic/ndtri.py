"""
Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

ndtri
"""
import math
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_platform


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    Constant in this class
    """
    S2PI = 2.50662827463100050242E0
    LIST_P0 = (-5.99633501014107895267E1, 9.80010754185999661536E1, -5.66762857469070293439E1,
               1.39312609387279679503E1, -1.23916583867381258016E0)
    LIST_Q0 = (1.95448858338141759834E0, 4.67627912898881538453E0, 8.63602421390890590575E1,
               -2.25462687854119370527E2, 2.00260212380060660359E2, -8.20372256168333339912E1,
               1.59056225126211695515E1, -1.18331621121330003142E0)
    LIST_P1 = (4.05544892305962419923E0, 3.15251094599893866154E1, 5.71628192246421288162E1,
               4.40805073893200834700E1, 1.46849561928858024014E1, 2.18663306850790267539E0,
               -1.40256079171354495875E-1, -3.50424626827848203418E-2, -8.57456785154685413611E-4)
    LIST_Q1 = (1.57799883256466749731E1, 4.53907635128879210584E1, 4.13172038254672030440E1,
               1.50425385692907503408E1, 2.50464946208309415979E0, -1.42182922854787788574E-1,
               -3.80806407691578277194E-2, -9.33259480895457427372E-4)
    LIST_P2 = (3.23774891776946035970E0, 6.91522889068984211695E0, 3.93881025292474443415E0,
               1.33303460815807542389E0, 2.01485389549179081538E-1, 1.23716634817820021358E-2,
               3.01581553508235416007E-4, 2.65806974686737550832E-6, 6.23974539184983293730E-9)
    LIST_Q2 = (6.02427039364742014255E0, 3.67983563856160859403E0, 1.37702099489081330271E0,
               2.16236993594496635890E-1, 1.34204006088543189037E-2, 3.28014464682127739104E-4,
               2.89247864745380683936E-6, 6.79019408009981274425E-9)


def _polevl(inp_x, ans, iter_n):
    """
    do ndtri compute
    x = x
             1       2       6
    y = 1 + ---  +  ---  +  ---- + ...
             x        2       3
                     x       x

    Parameters:
    ----------
    inp_x : the placeholder of data input

    Returns : A Tensor. Has the same type as data.
    -------
    """
    res = tbe.broadcast(tvm.const(ans[0], inp_x.dtype), inp_x.shape)
    for i in range(1, iter_n):
        mul_res = tbe.vmul(res, inp_x)
        res = tbe.vadds(mul_res, tvm.const(ans[i], inp_x.dtype))

    return res


def _plevl(inp_x, ans, iter_n):
    """
    do ndtri compute
    x = x
             1       2       6
    y = 1 + ---  +  ---  +  ---- + ...
             x        2       3
                     x       x

    Parameters:
    ----------
    inp_x : the placeholder of data input

    Returns : A Tensor. Has the same type as data.
    -------
    """
    ans_res = tbe.broadcast(tvm.const(ans[0], inp_x.dtype), inp_x.shape)
    res = tbe.vadd(ans_res, inp_x)
    for i in range(1, iter_n):
        mul_res = tbe.vmul(res, inp_x)
        res = tbe.vadds(mul_res, tvm.const(ans[i], inp_x.dtype))

    return res


def polevl_plevl(inp_x, ans, bns, iter_n, iter_m):
    """
    do ndtri compute use the 15th order taylor expansion
    x = x
    y = polevl( x, ans, iter_n )/polevl( x, bns, iter_m )

    Parameters:
    ----------
    inp_x : the placeholder of data input

    Returns : A Tensor. Has the same type as data.
    -------
    """
    res_a = _polevl(inp_x, ans, iter_n)
    res_b = _plevl(inp_x, bns, iter_m)
    res = tbe.vdiv(res_a, res_b)

    return res


def cal_sub(res_y):
    """
    do ndtri compute
    x = res_y
    y = sqrt(-2.0 * log(x))
    z = y - (logy / y)

    Parameters:
    ----------
    res_y : the placeholder of data input

    Returns : A Tuple. Has the same dtype as res_y.
    -------
    """
    tmp_log = tbe.vlog(res_y)
    tmp_mul = tbe.vmuls(tmp_log, -2.0)
    res_x = tbe.vsqrt(tmp_mul)
    log_x = tbe.vlog(res_x)
    div_x = tbe.vdiv(log_x, res_x)
    res_sub = tbe.vsub(res_x, div_x)

    return res_x, res_sub


def cal_p0(res_y):
    """
    do ndtri compute use the 15th order taylor expansion
    x = res_y
    y = polevl( x, ans, iter_n )/polevl( x, bns, iter_m )

    Parameters:
    ----------
    res_y : the placeholder of data input

    Returns : A Tensor. Has the same type as res_y.
    -------
    """
    res_half = tbe.broadcast(tvm.const(0.5, res_y.dtype), res_y.shape)
    res_y = tbe.vsub(res_y, res_half)
    res_mul = tbe.vmul(res_y, res_y)
    pp_val = polevl_plevl(res_mul, Constant.LIST_P0, Constant.LIST_Q0, 5, 8)
    tmp_pp = tbe.vmul(res_mul, pp_val)
    tmp_mul = tbe.vmul(res_y, tmp_pp)
    res_x = tbe.vadd(tmp_mul, res_y)
    res = tbe.vmuls(res_x, Constant.S2PI)

    return res


def cal_p12(res_x, res_sub, res_one):
    """
    do ndtri compute use the 15th order taylor expansion
    x = res_x
    y = polevl( x, ans, iter_n )/polevl( x, bns, iter_m )
    res_one = a tensor of one

    Parameters:
    ----------
    res_y : the placeholder of data input

    Returns : A Tuple. Has the same dtype as res_y.
    -------
    """
    res_z = tbe.vdiv(res_one, res_x)
    pp_val1 = polevl_plevl(res_z, Constant.LIST_P1, Constant.LIST_Q1, 9, 8)
    pp_val2 = polevl_plevl(res_z, Constant.LIST_P2, Constant.LIST_Q2, 9, 8)
    val_x1 = tbe.vmul(res_z, pp_val1)
    val_x2 = tbe.vmul(res_z, pp_val2)
    res_1 = tbe.vsub(res_sub, val_x1)
    res_2 = tbe.vsub(res_sub, val_x2)

    return res_1, res_2


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,too-many-locals
@register_operator_compute("Ndtri", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def ndtri_compute(input_x, output_y, kernel_name="ndtri"):
    """
    compute ndtri, `y = sqrt(2) * erfinv(2 * x - 1)`.

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        data of output.
    kernel_name: str
        kernel name, default value is "ndtri"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    dtype = input_x.dtype.lower()

    has_improve_precision = False
    # Change dtype to float32
    if dtype == "float16" and \
            tbe_platform.api_check_support("te.lang.cce.vcmpsel", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        has_improve_precision = True
        dtype = "float32"

    res_y = input_x

    # val_sub is the val of `exp(-2)`
    val_sub = math.exp(-2)

    # res_exp is the val of `sub(1, val_sub)`
    res_exp = 1 - val_sub

    val_one = tbe.broadcast(tvm.const(1.0, dtype), input_x.shape)
    res_yy = tbe.vsub(val_one, res_y)

    # case one, res_y is greater than res_exp and res_yy is greater than val_sub
    res_one = cal_p0(res_yy)

    # case two, res_y is less than res_exp and res_yy is greater than val_sub
    res_two = cal_p0(res_y)

    # case three,four, res_y is less than res_exp and res_yy is less than val_sub
    res_x1, res_sub1 = cal_sub(res_y)
    res_three, res_four = cal_p12(res_x1, res_sub1, val_one)
    res_three = tbe.vmuls(res_three, tvm.const(-1, dtype))
    res_four = tbe.vmuls(res_four, tvm.const(-1, dtype))

    # case five,six, res_y is greater than res_exp and res_yy is less than val_sub
    res_x2, res_sub2 = cal_sub(res_yy)
    res_five, res_six = cal_p12(res_x2, res_sub2, val_one)

    """
    Approximation for interval res_x1 between 2 and 8, y between `exp(-2)` and `exp(-32)`;
    for res_x1 between 8 and 64, y between `exp(-32)` and `exp(-2048)`.
    """
    res_x1_lt_eight = tbe.vcmpsel(res_x1, tvm.const(8.0, dtype), 'lt', res_three, res_four)
    res_y_gt_sub = tbe.vcmpsel(res_y, tvm.const(val_sub, dtype), 'gt', res_two, res_x1_lt_eight)
    res_x2_lt_eight = tbe.vcmpsel(res_x2, tvm.const(8.0, dtype), 'lt', res_five, res_six)
    res_yy_gt_sub = tbe.vcmpsel(res_yy, tvm.const(val_sub, dtype), 'gt', res_one, res_x2_lt_eight)
    res = tbe.vcmpsel(res_y, tvm.const(res_exp, dtype), 'gt', res_yy_gt_sub, res_y_gt_sub)

    # Restore dtype
    if has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res


@register_operator("Ndtri")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def ndtri(input_x, output_y, kernel_name="ndtri"):
    """
    Computes ndtri element-wise

    Parameters
    ----------
    input_x: dict
        shape and dtype of input, only support bfloat16, float16, float32
    output_y: dict
        shape and dtype of output, should be same type as input
    kernel_name: str
        kernel name, default value is "ndtri"

    Returns
    -------
    None
    """
    dtype_input = input_x.get("dtype").lower()
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(dtype_input, check_list, param_name="input_x")

    ins = classify([input_x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (x_,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([x_])
            data_input = tvm.placeholder(x_shape[0], name="data_input", dtype=dtype_input)
            res = ndtri_compute(data_input, output_y, kernel_name)

            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
