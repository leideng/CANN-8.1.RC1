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
dawsn

Op_description :
Computes dawsn of x element-wise

# dawsn(
#   x,
#   y,
#   kernel_name="cce_dawsn")

Supportive_dtype_format :
['float16', 'float32']
['ALL']

Constraint :
[1] All : shape size limit is 2147483648.
"""
from impl.util import util_compute
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # Taylor coefficient
    COEF_AN = (1.13681498971755972054E-11,
        8.49262267667473811108E-10,
        1.94434204175553054283E-8,
        9.53151741254484363489E-7,
        3.07828309874913200438E-6,
        3.52513368520288738649E-4,
        -8.50149846724410912031E-4,
        4.22618223005546594270E-2,
        -9.17480371773452345351E-2,
        9.99999999999999994612E-1)
    COEF_AN_COUNT = 9
    COEF_AD = (2.40372073066762605484E-11,
        1.48864681368493396752E-9,
        5.21265281010541664570E-8,
        1.27258478273186970203E-6,
        2.32490249820789513991E-5,
        3.25524741826057911661E-4,
        3.48805814657162590916E-3,
        2.79448531198828973716E-2,
        1.58874241960120565368E-1,
        5.74918629489320327824E-1,
        1.00000000000000000539E0)
    COEF_AD_COUNT = 10

    COEF_BN = (5.08955156417900903354E-1,
        -2.44754418142697847934E-1,
        9.41512335303534411857E-2,
        -2.18711255142039025206E-2,
        3.66207612329569181322E-3,
        -4.23209114460388756528E-4,
        3.59641304793896631888E-5,
        -2.14640351719968974225E-6,
        9.10010780076391431042E-8,
        -2.40274520828250956942E-9,
        3.59233385440928410398E-11)
    COEF_BN_COUNT = 10
    COEF_BD = (-6.31839869873368190192E-1,
        2.36706788228248691528E-1,
        -5.31806367003223277662E-2,
        8.48041718586295374409E-3,
        -9.47996768486665330168E-4,
        7.81025592944552338085E-5,
        -4.55875153252442634831E-6,
        1.89100358111421846170E-7,
        -4.91324691331920606875E-9,
        7.18466403235734541950E-11)
    COEF_BD_COUNT = 10
    
    COEF_CN = (-5.90592860534773254987E-1,
        6.29235242724368800674E-1,
        -1.72858975380388136411E-1,
        1.64837047825189632310E-2,
        -4.86827613020462700845E-4)
    COEF_CN_COUNT = 4
    COEF_CD = (-2.69820057197544900361E0,
        1.73270799045947845857E0,
        -3.93708582281939493482E-1,
        3.44278924041233391079E-2,
        -9.73655226040941223894E-4)
    COEF_CD_COUNT = 5

    THRESHOLD_3_25 = 3.25
    THRESHOLD_6_25 = 6.25
    THRESHOLD_1E_9 = 1.0e9


def _polevl(data_x, coef, iter_n):
    """
    y = polevl( x, coef, N );
    DESCRIPTION:    
    Evaluates polynomial of degree N:
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
    for index in range(1, iter_n + 1):
        res = tbe.vmul(res, data_x)
        res = tbe.vadds(res, tvm.const(coef[index], data_x.dtype))

    return res


def _p1evl(data_x, coef, iter_n):
    """
    y = p1evl( x, coef, N );
    DESCRIPTION:    
    Evaluates polynomial of degree N:
                        2          N
    y  =  C  + C x + C x  +...+ C x
             0    1     2          N
    Coefficients are stored in reverse order:
    coef[0] = C  , ..., coef[N] = C  .
                 N                   0
    The function p1evl() assumes that coef[N] = 1.0 and is
    omitted from the array.  Its calling arguments are
    otherwise the same as polevl().
    Parameters:
    ----------
    data_x : the placeholder of data input
    coef : coef of the data
    iter_n : number of the coef

    Returns : A Tensor. Has the same type as data.
    -------
    """
    res = tbe.vadds(data_x, tvm.const(coef[0], data_x.dtype))
    for index in range(1, iter_n):
        res = tbe.vmul(res, data_x)
        res = tbe.vadds(res, tvm.const(coef[index], data_x.dtype))
    
    return res


def _calc_condition_lt_three_p_two_five(input_x):
    """
    do dawsn compute when 0 <= x < 3.25
    x = xx*xx
    y = xx * polevl( x, AN, 9 )/polevl( x, AD, 10 )

    Parameters:
    ----------
    input_x : the placeholder of data input

    Returns : A Tensor. Has the same type as data.
    -------
    """
    data_square = tbe.vmul(input_x, input_x)
    data_polevl_an = _polevl(data_square, Constant.COEF_AN, Constant.COEF_AN_COUNT)
    data_polevl_ad = _polevl(data_square, Constant.COEF_AD, Constant.COEF_AD_COUNT)
    res = tbe.vmul(input_x, data_polevl_an)
    res = tbe.vdiv(res, data_polevl_ad)

    return res


def _calc_condition_lt_six_p_two_five(input_x):
    """
    do dawsn compute when 3.25 <= x < 6.25
    "x = 1.0/(xx*xx)"
    "y = (1.0/xx + x * polevl( x, BN, 10) / (p1evl( x, BD, 10) * xx)) * 0.5"

    Parameters:
    ----------
    input_x : the placeholder of data input

    Returns : A Tensor. Has the same type as data.
    -------
    """
    data_temp = tbe.vmul(input_x, input_x)
    data_one = tbe.broadcast(tvm.const(1.0, input_x.dtype), input_x.shape)
    data_temp = tbe.vdiv(data_one, data_temp)
    data_rec = tbe.vdiv(data_one, input_x)
    data_polevl_bn = _polevl(data_temp, Constant.COEF_BN, Constant.COEF_BN_COUNT)
    data_polevl_bn = tbe.vmul(data_polevl_bn, data_temp)
    data_plevl_bd = _p1evl(data_temp, Constant.COEF_BD, Constant.COEF_BD_COUNT)
    data_plevl_bd = tbe.vmul(data_plevl_bd, input_x)
    res = tbe.vdiv(data_polevl_bn, data_plevl_bd)
    res = tbe.vadd(data_rec, res)
    res = tbe.vmuls(res, tvm.const(0.5, input_x.dtype))

    return res


def _calc_condition_le_one_e_nine(input_x):
    """
    do dawsn compute when 6.25 <= x <= 1.0e9
    "x = 1.0/(xx*xx)"
    "y = (1.0/xx + x * polevl( x, CN, 4) / (p1evl( x, CD, 5) * xx)) * 0.5"

    Parameters:
    ----------
    input_x : the placeholder of data input

    Returns : A Tensor. Has the same type as data.
    -------
    """
    data_temp = tbe.vmul(input_x, input_x)
    data_one = tbe.broadcast(tvm.const(1.0, input_x.dtype), input_x.shape)
    data_temp = tbe.vdiv(data_one, data_temp)
    data_rec = tbe.vdiv(data_one, input_x)
    data_polevl_cn = _polevl(data_temp, Constant.COEF_CN, Constant.COEF_CN_COUNT)
    data_polevl_cn = tbe.vmul(data_polevl_cn, data_temp)
    data_plevl_cd = _p1evl(data_temp, Constant.COEF_CD, Constant.COEF_CD_COUNT)
    data_plevl_cd = tbe.vmul(data_plevl_cd, input_x)
    res = tbe.vdiv(data_polevl_cn, data_plevl_cd)
    res = tbe.vadd(data_rec, res)
    res = tbe.vmuls(res, tvm.const(0.5, input_x.dtype))

    return res


def _calc_condition_gt_one_e_nine(input_x):
    """
    do dawsn compute when x > 1.0e9
    "y = 1/xx * 0.5"

    Parameters:
    ----------
    input_x : the placeholder of data input

    Returns : A Tensor. Has the same type as data.
    -------
    """
    data_one = tbe.broadcast(tvm.const(1.0, input_x.dtype), input_x.shape)
    res = tbe.vdiv(data_one, input_x)
    res = tbe.vmuls(res, tvm.const(0.5, input_x.dtype))

    return res


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name,too-many-locals
@register_operator_compute("Dawsn", op_mode="dynamic", support_fusion=True)
def dawsn_compute(x, y, kernel_name="dawsn"):
    """
    do element-wise dawsin compute
  
    Parameters:
    ----------
    x: the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name, default value is "cce_dawsn"

    Returns : A Tensor. Has the same type as data_input.
    -------
    """

    dtype = x.dtype
    input_dtype = dtype

    if dtype == "float16":
        x = tbe.cast_to(x, "float32")
        dtype = "float32"

    # Sign mask
    sign = util_compute.sign(x)

    # All positive
    x = tbe.vmul(x, sign)

    # set 0 to 1 for avoid div 0
    mask_zero = tbe.vcmpsel(x, tvm.const(0, dtype), 'eq', tvm.const(1, dtype), tvm.const(0, dtype))
    res_temp = tbe.vcmpsel(x, tvm.const(0, dtype), 'eq', tvm.const(1, dtype), x)

    # lessthan 3.25    
    res_3_25 = _calc_condition_lt_three_p_two_five(res_temp)

    # lessthan 6.25    
    res_6_25 = _calc_condition_lt_six_p_two_five(res_temp)

    # lessequal 1.0e9    
    res_1_e9 = _calc_condition_le_one_e_nine(res_temp)

    # greatthan 1.0e9    
    res_gt_1_e9 = _calc_condition_gt_one_e_nine(res_temp)

    res_1 = tbe.vcmpsel(res_temp, tvm.const(Constant.THRESHOLD_3_25, dtype), 'lt', res_3_25, res_6_25)
    res_2 = tbe.vcmpsel(res_temp, tvm.const(Constant.THRESHOLD_1E_9, dtype), 'le', res_1_e9, res_gt_1_e9)
    res_result = tbe.vcmpsel(res_temp, tvm.const(Constant.THRESHOLD_6_25, dtype), 'lt', res_1, res_2)

    # Restore sign
    res_result = tbe.vmul(res_result, sign)
    # Restore 0
    res_result = tbe.vcmpsel(mask_zero, tvm.const(1, dtype), 'eq', tvm.const(0, dtype), res_result)

    if input_dtype == "float16":
        res_result = tbe.cast_to(res_result, "float16")

    return res_result


@register_operator("Dawsn")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def dawsn(x, y, kernel_name="dawsn"):
    """
    ----------
    Parameters:
    ----------
    x : the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name, default value is "dawsn"

    Returns : None
    -------
    """
    shape_input = x.get("shape")
    x_dtype = x.get("dtype").lower()
    y_dtype = y.get("dtype").lower()

    para_check.check_shape(shape_input, param_name="x")

    check_list = ("float16", "float32")
    para_check.check_dtype(x_dtype, check_list, param_name="x")
    if x_dtype != y_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal('dawsn', 'x_dtype', 'y_dtype',
                                                                str(x_dtype), str(y_dtype))

    ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for (_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])

            data_input = tvm.placeholder(x_shape[0], dtype=x_dtype,
                                         name="data_input")

            res = dawsn_compute(data_input, y, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)

        schedules.append(sch)

    config = {"name": kernel_name,
              "print_ir": False,
              "tensor_list": tensors,
              "bool_storage_as_1bit": False}

    tbe.build(schedules, config)
