# Copyright 2019 Huawei Technologies Co., Ltd
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
sin
"""

from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpImplMode
from impl.util.util_common import check_op_impl_mode
from impl.util.util_soc_common import after_v200


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    # define a string name of "float16"
    FLOAT_16 = "float16"
    # define a string name of "float32"
    FLOAT_32 = "float32"

    PI = 3.14159265358979

    # the first factor to use Taylor series in circle
    FIRST_ORDER = 5
    # the last factor to use Taylor series in circle
    LAST_ORDER = 13
    # the first factor of Taylor series
    FIRST_FACTOR = -1.0 / 6.0

    # define the number of x div pi
    PI_FOR_X_TODIV = 0.3183098733425140380859375
    # define the PI for compute_v2
    PI_V2 = 3.140625
    KPI_FIRS_PI_MULS = 0.0009670257568359375
    KPI_TWI_PI_MULS = 6.2771141529083251953125e-7
    KPI_THIR_PI_MULS = 1.21644916362129151821136474609375e-10
    KPI_FOR_PI_MULS = -1.0290623200529979163359041220560e-13

    # define the number of sin_compute_v2
    SIN_RES_MULIT_SCA = 2.604926501e-6
    SIN_RES_ADDICT_UP = -0.0001980894471
    SIN_2ADDS = 0.008333049340
    SIN_3ADDS = -0.1666665792

    ONE_OVER_2048 = 1.0 / 2048.0
    # define the Pi for compute_v3
    pi_0 = 3.14160156
    pi_1 = -8.9071691e-06
    pi_2 = -1.74122761e-09
    pi_3 = 1.24467439e-13

    # define the Pi for compute_v4
    NUM_2048 = 2048
    PI_V4_0 = 1.5708008
    PI_V4_1 = -0.0000044535846
    PI_V4_2 = -8.706138e-10
    PI_V4_3 = 1.5703125
    PI_12 = 0.0004837513
    PI_22 = 0.000000075495336
    PI_32 = 2.5579538e-12
    PI_42 = 5.389786e-15
    PI_52 = 5.166901e-19
    PI_62 = 3.281839e-22
    PI_72 = 1.643355e-26
    PI_82 = 3.521475e-27

    INV_HALF_PI = 0.63661975

    SCOEF_4 = 0.0000027183114939898219064
    SCOEF_3 = -0.000198393348360966317347
    SCOEF_2 = 0.0083333293858894631756
    SCOEF_1 = -0.166666666416265235595

    CCOEF_4 = 0.0000243904487962774090654
    CCOEF_3 = -0.00138867637746099294692
    CCOEF_2 = 0.0416666233237390631894
    CCOEF_1 = -0.499999997251031003120

    NUMBER_POS_ONE = 1.0
    NUMBER_NEG_ONE = - 1.0


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals
def sin_compute_v2(input_x):
    """
    algorithm: sin

    Parameters
    ----------
    input_x : TVM tensor
              data of input

    Returns
    -------
    res : TVM tensor
          the result of sin
    """
    dtype = input_x.dtype
    # cast to type float32 when type is float16
    has_improve_precision = False
    if dtype.lower() == "float16" and tbe_platform.api_check_support("tbe.dsl.vmul", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        dtype = "float32"
        has_improve_precision = True

    x_vmul = tbe.vmuls(input_x, tvm.const(Constant.PI_FOR_X_TODIV, dtype=dtype))
    if tbe_platform.api_check_support("tbe.dsl.round", "f322f32"):
        round_pi_div = tbe.round(x_vmul, "float32")
    else:
        round_pi_div_int32 = tbe.round(x_vmul)
        round_pi_div = tbe.cast_to(round_pi_div_int32, dtype)

    kp_0 = tbe.vmuls(round_pi_div, tvm.const(Constant.PI_V2, dtype=dtype))
    input_x0 = tbe.vsub(input_x, kp_0)
    kp_1 = tbe.vmuls(round_pi_div, tvm.const(Constant.KPI_FIRS_PI_MULS, dtype=dtype))
    input_x1 = tbe.vsub(input_x0, kp_1)
    kp_2 = tbe.vmuls(round_pi_div, tvm.const(Constant.KPI_TWI_PI_MULS, dtype=dtype))
    input_x2 = tbe.vsub(input_x1, kp_2)
    kp_3 = tbe.vmuls(round_pi_div, tvm.const(Constant.KPI_THIR_PI_MULS, dtype=dtype))
    input_x3 = tbe.vsub(input_x2, kp_3)
    kp_4 = tbe.vmuls(round_pi_div, tvm.const(Constant.KPI_FOR_PI_MULS, dtype=dtype))
    input_x4 = tbe.vsub(input_x3, kp_4)


    x_pow = tbe.vmul(input_x4, input_x4)
    kover2 = tbe.vmuls(round_pi_div, tvm.const(0.5, dtype=dtype))
    if tbe_platform.api_check_support("tbe.dsl.floor", "f322f32"):
        kover2floor = tbe.floor(kover2, "float32")
    else:
        kover2floor_int32 = tbe.floor(kover2)
        kover2floor = tbe.cast_to(kover2floor_int32, dtype)
    kover2floorm4 = tbe.vmuls(kover2floor, tvm.const(4.0, dtype=dtype))
    k2 = tbe.vmuls(round_pi_div, tvm.const(-2.0, dtype=dtype))
    sign = tbe.vadd(kover2floorm4, k2)
    sign = tbe.vadds(sign, tvm.const(1.0, dtype=dtype))

    res_up = tbe.vmuls(x_pow, tvm.const(Constant.SIN_RES_MULIT_SCA, dtype=dtype))
    res_up = tbe.vadds(res_up, tvm.const(Constant.SIN_RES_ADDICT_UP, dtype=dtype))
    res_up = tbe.vmul(res_up, x_pow)
    res_up = tbe.vadds(res_up, tvm.const(Constant.SIN_2ADDS, dtype=dtype))
    res_up = tbe.vmul(res_up, x_pow)
    res_up = tbe.vadds(res_up, tvm.const(Constant.SIN_3ADDS, dtype=dtype))
    res_up = tbe.vmul(res_up, x_pow)
    res_up = tbe.vadds(res_up, tvm.const(1.0, dtype=dtype))
    res_up = tbe.vmul(res_up, input_x4)
    res = tbe.vmul(res_up, sign)

    if has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res


def compute_fixed(dtype, round_pi_div, x_fixed):
    x_pow = tbe.vmul(x_fixed, x_fixed)
    kover2 = tbe.vmuls(round_pi_div, tvm.const(0.5, dtype=dtype))
    if tbe_platform.api_check_support("tbe.dsl.floor", "f322f32"):
        kover2floor = tbe.floor(kover2, "float32")
    else:
        kover2floor_int32 = tbe.floor(kover2)
        kover2floor = tbe.cast_to(kover2floor_int32, dtype)
    kover2floorm4 = tbe.vmuls(kover2floor, tvm.const(4.0, dtype=dtype))
    k2 = tbe.vmuls(round_pi_div, tvm.const(-2.0, dtype=dtype))
    sign = tbe.vadd(kover2floorm4, k2)
    sign = tbe.vadds(sign, tvm.const(1.0, dtype=dtype))

    res_up = tbe.vmuls(x_pow, tvm.const(Constant.SIN_RES_MULIT_SCA, dtype=dtype))
    res_up = tbe.vadds(res_up, tvm.const(Constant.SIN_RES_ADDICT_UP, dtype=dtype))
    res_up = tbe.vmul(res_up, x_pow)
    res_up = tbe.vadds(res_up, tvm.const(Constant.SIN_2ADDS, dtype=dtype))
    res_up = tbe.vmul(res_up, x_pow)
    res_up = tbe.vadds(res_up, tvm.const(Constant.SIN_3ADDS, dtype=dtype))
    res_up = tbe.vmul(res_up, x_pow)
    res_up = tbe.vadds(res_up, tvm.const(1.0, dtype=dtype))
    res_up = tbe.vmul(res_up, x_fixed)
    res_sign = tbe.vmul(res_up, sign)
    # Ensure result is between -1 and 1
    res_mins = tbe.vmins(res_sign, tvm.const(Constant.NUMBER_POS_ONE, dtype=dtype))
    res_maxs = tbe.vmaxs(res_mins, tvm.const(Constant.NUMBER_NEG_ONE, dtype=dtype))

    return res_maxs


def sin_compute_v3(input_x):
    """
    algorithm: sin

    Parameters
    ----------
    input_x : TVM tensor
              data of input

    Returns
    -------
    res : TVM tensor
          the result of sin
    """
    dtype = input_x.dtype
    # cast to type float32 when type is float16
    has_improve_precision = False
    if dtype.lower() == "float16" and tbe_platform.api_check_support("tbe.dsl.vmul", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        dtype = "float32"
        has_improve_precision = True

    x_vmul = tbe.vmuls(input_x, tvm.const(Constant.PI_FOR_X_TODIV, dtype=dtype))
    x_vmul0 = tbe.vmuls(x_vmul, tvm.const(Constant.ONE_OVER_2048, dtype=dtype))
    if tbe_platform.api_check_support("tbe.dsl.round", "f322f32"):
        round_pi_div = tbe.round_half_up(x_vmul, "float32")
        round_pi_div0 = tbe.round_half_up(x_vmul0, "float32")
    else:
        round_pi_div_int32 = tbe.round(x_vmul)
        round_pi_div = tbe.cast_to(round_pi_div_int32, dtype)
        round_pi_div0_int32 = tbe.round(x_vmul0)
        round_pi_div0 = tbe.cast_to(round_pi_div0_int32, dtype)
    round_pi_div0 = tbe.vmuls(round_pi_div0, tvm.const(2048.0, dtype=dtype))
    round_pi_div1 = tbe.vsub(round_pi_div, round_pi_div0)

    fix = tbe.vmuls(round_pi_div0, tvm.const(Constant.pi_0, dtype=dtype))
    x_fixed = tbe.vsub(input_x, fix)
    fix = tbe.vmuls(round_pi_div1, tvm.const(Constant.pi_0, dtype=dtype))
    x_fixed = tbe.vsub(x_fixed, fix)
    fix = tbe.vmuls(round_pi_div0, tvm.const(Constant.pi_1, dtype=dtype))
    x_fixed = tbe.vsub(x_fixed, fix)
    fix = tbe.vmuls(round_pi_div1, tvm.const(Constant.pi_1, dtype=dtype))
    x_fixed = tbe.vsub(x_fixed, fix)
    fix = tbe.vmuls(round_pi_div0, tvm.const(Constant.pi_2, dtype=dtype))
    x_fixed = tbe.vsub(x_fixed, fix)
    fix = tbe.vmuls(round_pi_div1, tvm.const(Constant.pi_2, dtype=dtype))
    x_fixed = tbe.vsub(x_fixed, fix)
    fix = tbe.vmuls(round_pi_div0, tvm.const(Constant.pi_3, dtype=dtype))
    x_fixed = tbe.vsub(x_fixed, fix)
    fix = tbe.vmuls(round_pi_div1, tvm.const(Constant.pi_3, dtype=dtype))
    x_fixed = tbe.vsub(x_fixed, fix)

    res = compute_fixed(dtype, round_pi_div, x_fixed)

    if has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res


def sin_compute_v4(input_x):
    """
    algorithm: sin

    Parameters
    ----------
    input_x : TVM tensor
              data of input

    Returns
    -------
    res : TVM tensor
          the result of sin
    """
    dtype = input_x.dtype
    has_improve_precision = False
    if dtype.lower() == "float16" and tbe_platform.api_check_support("tbe.dsl.vmul", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        dtype = "float32"
        has_improve_precision = True
    
    number_2048 = tvm.const(Constant.NUM_2048, dtype=dtype)
    one_over_n = tvm.const(1.0 / Constant.NUM_2048, dtype=dtype)
    inv_half_pi = tvm.const(Constant.INV_HALF_PI, dtype=dtype)
    pi_0 = tvm.const(Constant.PI_V4_0, dtype=dtype)
    pi_1 = tvm.const(Constant.PI_V4_1, dtype=dtype)
    pi_2 = tvm.const(Constant.PI_V4_2, dtype=dtype)

    x_scaled = tbe.vmuls(input_x, one_over_n)
    x_overpi = tbe.vmuls(x_scaled, inv_half_pi)
    if tbe_platform.api_check_support("tbe.dsl.round", "f322f32"):
        n = tbe.round(x_overpi, "float32")
    else:
        n_int32 = tbe.round(x_overpi)
        n = tbe.cast_to(n_int32, dtype)
    n0 = tbe.vmuls(x_overpi, one_over_n)
    if tbe_platform.api_check_support("tbe.dsl.round", "f322f32"):
        n0 = tbe.round(n0, "float32")
    else:
        n0_int32 = tbe.round(n0)
        n0 = tbe.cast_to(n0_int32, dtype)
    n0 = tbe.vmuls(n0, number_2048)
    n1 = tbe.vsub(n, n0)

    fix = tbe.vmuls(n0, pi_0)
    x_fix = tbe.vsub(x_scaled, fix)
    fix = tbe.vmuls(n1, pi_0)
    x_fix = tbe.vsub(x_fix, fix)
    fix = tbe.vmuls(n0, pi_1)
    x_fix = tbe.vsub(x_fix, fix)
    fix = tbe.vmuls(n1, pi_1)
    x_fix = tbe.vsub(x_fix, fix)
    fix = tbe.vmuls(n0, pi_2)
    x_fix = tbe.vsub(x_fix, fix)

    pi_02 = tvm.const(Constant.PI_V4_3, dtype=dtype)
    pi_12 = tvm.const(Constant.PI_12, dtype=dtype)

    remain_x = tbe.vmuls(x_fix, number_2048)
    temp = tbe.vmuls(remain_x, inv_half_pi)
    if tbe_platform.api_check_support("tbe.dsl.round", "f322f32"):
        n2 = tbe.round(temp, "float32")
    else:
        n2_int32 = tbe.round(temp)
        n2 = tbe.cast_to(n2_int32, dtype)
    n0 = tbe.vmuls(n0, number_2048)
    n1 = tbe.vmuls(n1, number_2048)
    fix = tbe.vmuls(n0, pi_02)
    x_fix = tbe.vsub(input_x, fix)
    fix = tbe.vmuls(n1, pi_02)
    x_fix = tbe.vsub(x_fix, fix)
    fix = tbe.vmuls(n0, pi_12)
    x_fix = tbe.vsub(x_fix, fix)

    pi_22 = tvm.const(Constant.PI_22, dtype=dtype)
    fix = tbe.vmuls(n2, pi_02)
    x_fix = tbe.vsub(x_fix, fix)
    fix = tbe.vmuls(n1, pi_12)
    x_fix = tbe.vsub(x_fix, fix)
    fix = tbe.vmuls(n0, pi_22)
    x_fix = tbe.vsub(x_fix, fix)

    pi_32 = tvm.const(Constant.PI_32, dtype=dtype)
    fix = tbe.vmuls(n2, pi_12)
    x_fix = tbe.vsub(x_fix, fix)
    fix = tbe.vmuls(n1, pi_22)
    x_fix = tbe.vsub(x_fix, fix)
    fix = tbe.vmuls(n0, pi_32)
    x_fix = tbe.vsub(x_fix, fix)

    pi_42 = tvm.const(Constant.PI_42, dtype=dtype)
    fix = tbe.vmuls(n2, pi_22)
    x_fix = tbe.vsub(x_fix, fix)
    fix = tbe.vmuls(n1, pi_32)
    x_fix = tbe.vsub(x_fix, fix)
    fix = tbe.vmuls(n0, pi_42)
    x_fix = tbe.vsub(x_fix, fix)

    pi_52 = tvm.const(Constant.PI_52, dtype=dtype)
    fix = tbe.vmuls(n2, pi_32)
    x_fix = tbe.vsub(x_fix, fix)
    fix = tbe.vmuls(n1, pi_42)
    x_fix = tbe.vsub(x_fix, fix)
    fix = tbe.vmuls(n0, pi_52)
    x_fix = tbe.vsub(x_fix, fix)

    pi_62 = tvm.const(Constant.PI_62, dtype=dtype)
    fix = tbe.vmuls(n2, pi_42)
    x_fix = tbe.vsub(x_fix, fix)
    fix = tbe.vmuls(n1, pi_52)
    x_fix = tbe.vsub(x_fix, fix)
    fix = tbe.vmuls(n0, pi_62)
    x_fix = tbe.vsub(x_fix, fix)

    fix = tbe.vmuls(n2, pi_52)
    x_fix = tbe.vsub(x_fix, fix)
    fix = tbe.vmuls(n1, pi_62)
    x_fix = tbe.vsub(x_fix, fix)
    fix = tbe.vmuls(n2, pi_62)
    x_fix = tbe.vsub(x_fix, fix)

    half_n2 = tbe.vmuls(n2, tvm.const(0.5, dtype=dtype))
    half4_n2 = tbe.vmuls(n2, tvm.const(0.25, dtype=dtype))
    if tbe_platform.api_check_support("tbe.dsl.floor", "f322f32"):
        n_half2 = tbe.floor(half_n2, "float32")
    else:
        n_half2_int32 = tbe.floor(half_n2)
        n_half2 = tbe.cast_to(n_half2_int32, dtype)
    if tbe_platform.api_check_support("tbe.dsl.floor", "f322f32"):
        n_half4 = tbe.floor(half4_n2, "float32")
    else:
        n_half4_int32 = tbe.floor(half4_n2)
        n_half4 = tbe.cast_to(n_half4_int32, dtype)
    k1 = tbe.vmuls(n_half2, tvm.const(-2.0, dtype=dtype))
    k2 = tbe.vmuls(n_half4, tvm.const(4.0, dtype=dtype))
    sign = tbe.vadd(k1, k2)
    sign = tbe.vadds(sign, tvm.const(1.0, dtype=dtype))

    ifcos = tbe.vadd(n2, k1)
    ifsin = tbe.vmuls(ifcos, tvm.const(-1.0, dtype=dtype))
    ifsin = tbe.vadds(ifsin, tvm.const(1.0, dtype=dtype))

    scoef4 = tvm.const(Constant.SCOEF_4, dtype=dtype)
    scoef3 = tvm.const(Constant.SCOEF_3, dtype=dtype)
    scoef2 = tvm.const(Constant.SCOEF_2, dtype=dtype)
    scoef1 = tvm.const(Constant.SCOEF_1, dtype=dtype)
    x_pow = tbe.vmul(x_fix, x_fix)
    sin_poly = tbe.vmuls(x_pow, scoef4)
    sin_poly = tbe.vadds(sin_poly, scoef3)
    sin_poly = tbe.vmul(x_pow, sin_poly)
    sin_poly = tbe.vadds(sin_poly, scoef2)
    sin_poly = tbe.vmul(x_pow, sin_poly)
    sin_poly = tbe.vadds(sin_poly, scoef1)
    sin_poly = tbe.vmul(x_pow, sin_poly)
    sin_poly = tbe.vadds(sin_poly, tvm.const(1.0, dtype=dtype))
    sin_poly = tbe.vmul(x_fix, sin_poly)

    ccoef4 = tvm.const(Constant.CCOEF_4, dtype=dtype)
    ccoef3 = tvm.const(Constant.CCOEF_3, dtype=dtype)
    ccoef2 = tvm.const(Constant.CCOEF_2, dtype=dtype)
    ccoef1 = tvm.const(Constant.CCOEF_1, dtype=dtype)
    cos_poly = tbe.vmuls(x_pow, ccoef4)
    cos_poly = tbe.vadds(cos_poly, ccoef3)
    cos_poly = tbe.vmul(x_pow, cos_poly)
    cos_poly = tbe.vadds(cos_poly, ccoef2)
    cos_poly = tbe.vmul(x_pow, cos_poly)
    cos_poly = tbe.vadds(cos_poly, ccoef1)
    cos_poly = tbe.vmul(x_pow, cos_poly)
    cos_poly = tbe.vadds(cos_poly, tvm.const(1.0, dtype=dtype))

    temp1 = tbe.vmul(sin_poly, ifsin)
    cos_poly = tbe.vmul(cos_poly, ifcos)
    res = tbe.vadd(temp1, cos_poly)
    res = tbe.vmul(res, sign)

    if has_improve_precision:
        res = tbe.cast_to(res, "float16")

    return res


# 'pylint: disable=invalid-name
def _sin(x):
    """
    algorithm: sin
    calculating data's sin x = x-x^3/3!+x ^5/5!-x^7/7!+x^9/9!-x^11/11! (-pai/2 < x < pai/2)

    Parameters
    ----------
    x : TVM tensor
        the placeholders of input data

    Returns
    -------
    res : the res of sin
    """
    input_x_power = tbe.vmul(x, x)
    iter_value = tbe.vmul(tbe.vmuls(input_x_power, Constant.FIRST_FACTOR), x)
    res = tbe.vadd(x, iter_value)

    i = Constant.FIRST_ORDER
    while i < Constant.LAST_ORDER:
        iter_value = tbe.vmuls(tbe.vmul(input_x_power, iter_value), -1.0 / (i * (i - 1)))
        res = tbe.vadd(res, iter_value)

        # add 2 to get the next order
        i = i + 2

    return res


# 'pylint: disable=locally-disabled,unused-argument,invalid-name,too-many-locals
@register_operator_compute("Sin", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def sin_compute(x, y, kernel_name="sin", impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    algorithm: sin
    calculating data's sin x = x - x^3/3! + x ^5/5! + ... + (-1)^k*x^2(k+1)/(2(k+1))!

    Parameters
    ----------
    x : TVM tensor
        the placeholders of input data
    y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is "sin"
    impl_mode : str
        impl_mode, default value is "high_performance"

    Returns
    -------
    res : the res of sin
    """
    #1971 use v3/v4 compute
    if after_v200():
        if impl_mode == "high_precision":
            res = sin_compute_v4(x)
        else:
            res = sin_compute_v3(x)
        return res
    cur_cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if cur_cce_product in ("Ascend910",):
        res = sin_compute_v2(x)
        return res

    dtype = x.dtype

    has_improve_precision = False
    cast_dtype = dtype
    if tbe_platform.api_check_support("tbe.dsl.vmul", "float32"):
        has_improve_precision = True
        cast_dtype = Constant.FLOAT_32

    # cast to type float32 when type is float16
    if dtype == Constant.FLOAT_16 and has_improve_precision:
        x = tbe.cast_to(x, Constant.FLOAT_32)

    pai_multiple = tbe.vmuls(x, 1 / Constant.PI)
    # `pai_round = tbe.round(pai_multiple)`
    if not tbe_platform.api_check_support("tbe.dsl.round", "float32") and cast_dtype == Constant.FLOAT_32:
        pai_16 = tbe.cast_to(pai_multiple, Constant.FLOAT_16)
        round_float = tbe.cast_to(tbe.round(pai_16), cast_dtype)
    else:
        round_float = tbe.cast_to(tbe.round(pai_multiple), cast_dtype)
    # to adjust x to [-pai/2,pai/2]
    x = tbe.vsub(x, tbe.vmuls(round_float, Constant.PI))

    res = _sin(x)

    # if round is odd, the final result need to mutiply -1.Need to multipy 1/2 to get the ceil value
    ran_ = tbe.vmuls(round_float, 1 / 2)
    if not tbe_platform.api_check_support("tbe.dsl.ceil", "float32") and cast_dtype == Constant.FLOAT_32:
        ran_16 = tbe.cast_to(ran_, Constant.FLOAT_16)
        ceil_value = tbe.ceil(ran_16)
        ceil_value = tbe.cast_to(ceil_value, cast_dtype)
    else:
        ceil_value = tbe.ceil(ran_)
    # if odd, ceil*2-round is 1,if even, the value is 0
    tmp = tbe.cast_to(tbe.vmuls(ceil_value, tvm.const(2, dtype)), cast_dtype)
    sub_value = tbe.vsub(tmp, round_float)
    # `sub_value = tbe.vsub(tbe.vmuls(ceil_value, tvm.const(2, dtype)), round_float)`

    tensor_one = tbe.broadcast(tvm.const(1, cast_dtype), x.shape)
    odd_tensor = tbe.vsub(tensor_one, sub_value)
    even_tensor = tbe.vsub(odd_tensor, tensor_one)
    odd_even_tensor = tbe.vadd(odd_tensor, even_tensor)
    res = tbe.vmul(res, odd_even_tensor)

    # cast the dtype to float16
    if dtype == Constant.FLOAT_16 and has_improve_precision:
        res = tbe.cast_to(res, Constant.FLOAT_16)

    return res


@register_operator("Sin")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def sin(x, y, kernel_name="sin", impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    algorithm: sin
    calculating data's sin x = x - x^3/3! + x^5/5! + ... + (-1)^k*x^2(k+1)/(2(k+1))!

    Parameters
    ----------
    x: dict
        shape and dtype of input, only support float16, float32, bfloat16
    y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is "sin"
    impl_mode : str
        impl_mode, default value is "high_performance"

    Returns
    -------
    None
    """
    check_op_impl_mode(impl_mode, [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION], kernel_name)

    # check input x dtypey
    x_dtype = x.get("dtype").lower()
    y_dtype = y.get("dtype").lower()
    check_list = ("bfloat16", "float16", "float32", "int32", "int64")
    para_check.check_dtype(x_dtype, check_list, param_name="x")

    # check input x and output y dtype
    if x_dtype != y_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal("sin", "x", "y",
                                                              str(x_dtype), str(y_dtype))

    # op compute and schedule
    ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_x,) in ins:
        # op compute
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])
            input_data = tvm.placeholder(x_shape[0], name="input_data", dtype=x_dtype)

            res = sin_compute(input_data, y, kernel_name, impl_mode)
            tensors.append([input_data, res])
        # target auto schedule
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        # append schedule 2 schedules
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
