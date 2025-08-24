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
fresnel_sin
"""
from impl.util import util_compute
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm

from impl.dynamic.cos import cos_compute
from impl.dynamic.sin import sin_compute
from impl.util import util_soc_common
from impl.dynamic.tanh import _sign_compute


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    the class for constant.
    """
    SCALAR_ZERO = 0.0
    SCALAR_HALF = 0.5
    SCALAR_ONE = 1
    SCALAR_A = 2.5625
    SCALAR_B = 36974.0
    SCALAR_HALF_PI = 1.5707963267948966192313216916398
    SCALAR_PI = 3.1415926535897932384626433832795

    SCALAR_CN = (-4.98843114573573548651E-8, 9.50428062829859605134E-6,
                 -6.45191435683965050962E-4, 1.88843319396703850064E-2,
                 -2.05525900955013891793E-1, 9.99999999999999998822E-1)
    SCALAR_CD = (3.99982968972495980367E-12, 9.15439215774657478799E-10,
                 1.25001862479598821474E-7, 1.22262789024179030997E-5,
                 8.68029542941784300606E-4, 4.12142090722199792936E-2,
                 1.00000000000000000118E0)
    SCALAR_SN = (-2.99181919401019853726E3, 7.08840045257738576863E5,
                 -6.29741486205862506537E7, 2.54890880573376359104E9,
                 -4.42979518059697779103E10, 3.18016297876567817986E11)
    SCALAR_SD = (1.0,
                 2.81376268889994315696E2,
                 4.55847810806532581675E4,
                 5.17343888770096400730E6,
                 4.19320245898111231129E8,
                 2.24411795645340920940E10,
                 6.07366389490084639049E11)

    SCALAR_FN = (4.21543555043677546506E-1, 1.43407919780758885261E-1,
                 1.15220955073585758835E-2, 3.45017939782574027900E-4,
                 4.63613749287867322088E-6, 3.05568983790257605827E-8,
                 1.02304514164907233465E-10, 1.72010743268161828879E-13,
                 1.34283276233062758925E-16, 3.76329711269987889006E-20)
    SCALAR_FD = (1.0,
                 7.51586398353378947175E-1,
                 1.16888925859191382142E-1,
                 6.44051526508858611005E-3,
                 1.55934409164153020873E-4,
                 1.84627567348930545870E-6,
                 1.12699224763999035261E-8,
                 3.60140029589371370404E-11,
                 5.88754533621578410010E-14,
                 4.52001434074129701496E-17,
                 1.25443237090011264384E-20)
    SCALAR_GN = (5.04442073643383265887E-1, 1.97102833525523411709E-1,
                 1.87648584092575249293E-2, 6.84079380915393090172E-4,
                 1.15138826111884280931E-5, 9.82852443688422223854E-8,
                 4.45344415861750144738E-10, 1.08268041139020870318E-12,
                 1.37555460633261799868E-15, 8.36354435630677421531E-19,
                 1.86958710162783235106E-22)
    SCALAR_GD = (1.0,
                 1.47495759925128324529E0,
                 3.37748989120019970451E-1,
                 2.53603741420338795122E-2,
                 8.14679107184306179049E-4,
                 1.27545075667729118702E-5,
                 1.04314589657571990585E-7,
                 4.60680728146520428211E-10,
                 1.10273215066240270757E-12,
                 1.38796531259578871258E-15,
                 8.39158816283118707363E-19,
                 1.86958710162783236342E-22)

    SN_COUNT = 5
    SD_COUNT = 6
    CN_COUNT = 5
    CD_COUNT = 6
    FN_COUNT = 9
    FD_COUNT = 10
    GN_COUNT = 10
    GD_COUNT = 11


def polevl(data_x, coef, num):
    """
    y = polevl( x, coef, num );
    DESCRIPTION:    
    Evaluates polynomial of degree num:
                        2          N
    y  =  C  + C x + C x  +...+ C x
             0    1     2          N
    Coefficients are stored in reverse order:
    coef[0] = C  , ..., coef[N] = C  .
                 N                   0
    Parameters:
    ----------
    data_x : the placeholder of data input
    coef : coef of the data
    iter_n : number of the coef
     Returns : A Tensor. Has the same type as data.
    -------
    """
    res = tbe.broadcast(tvm.const(coef[0], data_x.dtype), data_x.shape)
    for index in range(1, num + 1):
        res = tbe.vmul(res, data_x)
        res = tbe.vadds(res, tvm.const(coef[index], data_x.dtype))

    return res


def p1evl(data_x, coef, num):
    """
    y = p1evl( x, coef, num );
    DESCRIPTION:    
    Evaluates polynomial of degree num:
                        2          N
    y  =  C  + C x + C x  +...+ C x
             0    1     2          N
    Coefficients are stored in reverse order:
    coef[0] = C  , ..., coef[N] = C  .
                 N                   0
    The function p1evl() assumes that coef[N] = 1.0 and is
    omitted from the array.  Its calling arguments are
    otherwise the same as polevl().
    -------
    """
    res = tbe.vadds(data_x, tvm.const(coef[0], data_x.dtype))
    for index in range(1, num + 1):
        res = tbe.vmul(res, data_x)
        res = tbe.vadds(res, tvm.const(coef[index], data_x.dtype))

    return res


def generic_fresnel_asymp(x, y):
    """
    do internel fresnel sin asymp compute
    rec = 1 / (π * x^2), 
    f = 1 - rec^2 * polevel(rec^2, FN, 9) / plevel(rec^2, FD, 10),
    g = rec * polevel(rec^2, GN, 10) / plevel(rec^2, GD, 11)
    res = 0.5 - 1 / (π * x) * (f * cos(π * x^2 / 2) + g * sin(π * x^2 / 2))
    """
    const_one = tbe.broadcast(tvm.const(Constant.SCALAR_ONE, x.dtype), x.shape)
    tmp_x2 = tbe.vmul(x, x)
    data_pi = tbe.vmuls(tmp_x2, Constant.SCALAR_PI)
    data_rec = tbe.vdiv(const_one, data_pi)
    data_square = tbe.vmul(data_rec, data_rec)
    data_fn_polevl = polevl(data_square, Constant.SCALAR_FN, Constant.FN_COUNT)
    data_fd_p1evl = p1evl(data_square, Constant.SCALAR_FD, Constant.FD_COUNT)
    data_gn_polevl = polevl(data_square, Constant.SCALAR_GN, Constant.GN_COUNT)
    data_gd_p1evl = p1evl(data_square, Constant.SCALAR_GD, Constant.GD_COUNT)

    data_f = tbe.vmul(data_square, data_fn_polevl)
    data_f = tbe.vdiv(data_f, data_fd_p1evl)
    data_f = tbe.vsub(const_one, data_f)

    data_g = tbe.vmul(data_rec, data_gn_polevl)
    data_g = tbe.vdiv(data_g, data_gd_p1evl)

    data_z = tbe.vmuls(tmp_x2, Constant.SCALAR_HALF_PI)
    data_c = cos_compute(data_z, y, 'cos')
    data_s = sin_compute(data_z, y, 'sin')
    data_y = tbe.vmuls(x, Constant.SCALAR_PI)
    data_y = tbe.vdiv(const_one, data_y)

    const_half = tbe.broadcast(tvm.const(Constant.SCALAR_HALF, x.dtype), x.shape)
    res_1 = tbe.vmul(data_f, data_c)
    res_2 = tbe.vmul(data_g, data_s)
    res_add = tbe.vadd(res_1, res_2)
    res_3 = tbe.vmul(res_add, data_y)
    res = tbe.vsub(const_half, res_3)

    return res


def _generic_fresnel_sin_interval(x):
    """
    do internel fresnel_sin compute
    res = x^3 * polevel(x^4, SN, 5) / plevel(x^4, SD, 6)
    """
    tmp_x2 = tbe.vmul(x, x)
    tmp_x3 = tbe.vmul(x, tmp_x2)
    tmp_x4 = tbe.vmul(tmp_x2, tmp_x2)
    data_sn_polevl = polevl(tmp_x4, Constant.SCALAR_SN, Constant.SN_COUNT)
    data_sd_p1evl = p1evl(tmp_x4, Constant.SCALAR_SD, Constant.SD_COUNT)
    res = tbe.vmul(tmp_x3, data_sn_polevl)
    res = tbe.vdiv(res, data_sd_p1evl)

    return res


@register_operator_compute("FresnelSin", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def fresnel_sin_compute(x, y, kernel_name="fresnel_sin"):
    """
    do element-wise fresnel_sin compute
    |x| > b, y = 0.5
    x^2 >= a, rec = 1 / (π * x^2), 
              f = 1 - rec^2 * polevel(rec^2, FN, 9) / plevel(rec^2, FD, 10),
              g = rec * polevel(rec^2, GN, 10) / plevel(rec^2, GD, 11)
              y = 0.5 - 1 / (π * x) * (f * cos(π * x^2 / 2) + g * sin(π * x^2 / 2))
    x^2 < a, y = x^3 * polevel(x^4, SN, 5) / plevel(x^4, SD, 6)
    Parameters:
    x: the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name, default value is "fresnel_sin"

    Returns : A Tensor. Has the same type as data_input.
    -------
    """

    shape = x.shape
    dtype = x.dtype

    # Change dtype to float32
    if dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vadd", "float32"):
        x = tbe.cast_to(x, "float32")

    # Sign mask
    if util_soc_common.after_v200() and x.dtype == "float32":
        sign = _sign_compute(x)
    else:
        sign = util_compute.sign(x)

    # All positive
    x = tbe.vmul(x, sign)

    # set 0 to 1 to avoid div 0
    mask_zero = tbe.vcmpsel(x, tvm.const(0, x.dtype), 'eq', tvm.const(1, x.dtype), tvm.const(0, x.dtype))
    x_d_0 = tbe.vcmpsel(x, tvm.const(0, x.dtype), 'eq', tvm.const(1, x.dtype), x)

    res_gt_b = tbe.broadcast(tvm.const(Constant.SCALAR_HALF, x.dtype), shape)
    res_lt_a = _generic_fresnel_sin_interval(x_d_0)
    res_ge_a = generic_fresnel_asymp(x_d_0, y)

    square_x = tbe.vmul(x, x)
    res_1 = tbe.vcmpsel(square_x, rhs=Constant.SCALAR_A, operation='ge', slhs=res_ge_a, srhs=res_lt_a)
    res = tbe.vcmpsel(x, rhs=Constant.SCALAR_B, operation='gt', slhs=res_gt_b, srhs=res_1)

    # Restore 0
    res = tbe.vcmpsel(mask_zero, tvm.const(1, x.dtype), 'eq', tvm.const(0, x.dtype), res)

    # Restore sign
    res = tbe.vmul(res, sign)

    # Restore dtype
    if dtype == "float16":
        res = tbe.cast_to(res, "float16")

    return res


@register_operator("FresnelSin")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def fresnel_sin(x, y, kernel_name="fresnel_sin"):
    """
    x : the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name, default value is "fresnel_sin"

    Returns : None
    -------
    """
    shape_input = x.get("shape")
    dtype_input = x.get("dtype").lower()

    para_check.check_shape(shape_input, param_name="x")

    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(dtype_input, check_list, param_name="x")

    ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])
            data_input = tvm.placeholder(x_shape[0], dtype=dtype_input,
                                         name="data_input")
            res = fresnel_sin_compute(data_input, y, kernel_name)
            tensors.append([data_input, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
            schedules.append(sch)

    config = {"name": kernel_name,
              "print_ir": False,
              "tensor_list": tensors,
              "bool_storage_as_1bit": False}

    tbe.build(schedules, config)
