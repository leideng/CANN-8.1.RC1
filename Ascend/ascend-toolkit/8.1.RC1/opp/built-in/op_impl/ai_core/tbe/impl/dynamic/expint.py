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
expint
"""
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    Constant in this class
    """
    EUL = 5.772156649015328606065e-1
    NEG_ONE = -1
    NUM_ZERO = 0.0
    NUM_ONE = 1.0
    LIST_A = (-5.350447357812542947283E0, 2.185049168816613393830E2, -4.176572384826693777058E3,
              5.541176756393557601232E4, -3.313381331178144034309E5, 1.592627163384945414220E6)
    LIST_B = (-5.250547959112862969197E1, 1.259616186786790571525E3, -1.756549581973534652631E4,
              1.493062117002725991967E5, -7.294949239640527645655E5, 1.592627163384945429726E6)
    LIST_A2 = (-2.106934601691916512584E0, 1.732733869664688041885E0, -2.423619178935841904839E-1,
               2.322724180937565842585E-2, 2.372880440493179832059E-4, -8.343219561192552752335E-5,
               1.363408795605250394881E-5, -3.655412321999253963714E-7, 1.464941733975961318456E-8,
               6.176407863710360207074E-10)
    LIST_B2 = (-2.298062239901678075778E-1, 1.105077041474037862347E-1, -1.566542966630792353556E-2,
               2.761106850817352773874E-3, -2.089148012284048449115E-4, 1.708528938807675304186E-5,
               -4.459311796356686423199E-7, 1.394634930353847498145E-8, 6.150865933977338354138E-10)
    LIST_A3 = (-7.657847078286127362028E-1, 6.886192415566705051750E-1, -2.132598113545206124553E-1,
               3.346107552384193813594E-2, -3.076541477344756050249E-3, 1.747119316454907477380E-4,
               -6.103711682274170530369E-6, 1.218032765428652199087E-7, -1.086076102793290233007E-9)
    LIST_B3 = (-1.888802868662308731041E0, 1.066691687211408896850E0, -2.751915982306380647738E-1,
               3.930852688233823569726E-2, -3.414684558602365085394E-3, 1.866844370703555398195E-4,
               -6.345146083130515357861E-6, 1.239754287483206878024E-7, -1.086076102793126632978E-9)
    LIST_A4 = (-2.458119367674020323359E-1, -1.483382253322077687183E-1, 7.248291795735551591813E-2,
               -1.348315687380940523823E-2, 1.342775069788636972294E-3, -7.942465637159712264564E-5,
               2.644179518984235952241E-6, -4.239473659313765177195E-8)
    LIST_B4 = (-1.044225908443871106315E-1, -2.676453128101402655055E-1, 9.695000254621984627876E-2,
               -1.601745692712991078208E-2, 1.496414899205908021882E-3, -8.462452563778485013756E-5,
               2.728938403476726394024E-6, -4.239462431819542051337E-8)
    LIST_A5 = (-1.373215375871208729803E0, -7.084559133740838761406E-1, 1.580806855547941010501E0,
               -2.601500427425622944234E-1, 2.994674694113713763365E-2, -1.038086040188744005513E-3,
               4.371064420753005429514E-5, 2.141783679522602903795E-6)
    LIST_B5 = (8.585231423622028380768E-1, 4.483285822873995129957E-1, 7.687932158124475434091E-2,
               2.449868241021887685904E-2, 8.832165941927796567926E-4, 4.590952299511353531215E-4,
               -4.729848351866523044863E-6, 2.665195537390710170105E-6)
    LIST_A6 = (1.981808503259689673238E-2, -1.271645625984917501326E0, -2.088160335681228318920E0,
               2.755544509187936721172E0, -4.409507048701600257171E-1, 4.665623805935891391017E-2,
               -1.545042679673485262580E-3, 7.059980605299617478514E-5)
    LIST_B6 = (1.476498670914921440652E0, 5.629177174822436244827E-1, 1.699017897879307263248E-1,
               2.291647179034212017463E-2, 4.450150439728752875043E-3, 1.727439612206521482874E-4,
               3.953167195549672482304E-5)
    LIST_A7 = (1.212561118105456670844E-1, -5.823133179043894485122E-1, 2.348887314557016779211E-1,
               -3.040034318113248237280E-2, 1.510082146865190661777E-3, -2.523137095499571377122E-5)
    LIST_B7 = (-1.002252150365854016662E0, 2.928709694872224144953E-1, -3.337004338674007801307E-2,
               1.560544881127388842819E-3, -2.523137093603234562648E-5)


def _polevl(inp_x, ans, iter_n):
    """
    do expint compute
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
    do expint compute
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
    _res = tbe.vadd(res, inp_x)
    for i in range(1, iter_n):
        mul_res = tbe.vmul(_res, inp_x)
        _res = tbe.vadds(mul_res, tvm.const(ans[i], inp_x.dtype))

    return _res


def polevl_plevl(inp_x, ans, bns, iter_n, iter_m):
    """
    do expint compute use the 15th order taylor expansion
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


def expint_cal_two(inp_x, ans, bns, iter_n, iter_m):
    """
    do expint compute use the 15th order taylor expansion when  (0 < x < 2)
    x = x
    y = EUL + log(x) + x * (polevl(x,A,5) / p1evl(x,B,6))

    Parameters:
    ----------
    inp_x : the placeholder of data input

    Returns : A Tensor. Has the same type as data.
    -------
    """
    _res = polevl_plevl(inp_x, ans, bns, iter_n, iter_m)
    mul_res = tbe.vmul(inp_x, _res)
    log_res = tbe.vlog(inp_x)
    add_res = tbe.vadd(mul_res, log_res)
    res = tbe.vadd(tbe.broadcast(tvm.const(Constant.EUL, inp_x.dtype), inp_x.shape), add_res)

    return res


def expint_cal_oth(inp_x, ans, bns, iter_n, iter_m):
    """
    do expint compute use the 15th order taylor expansion when  (2 <= x)
    x = x
    y = exp(x) * (1.0/x) * (1.0 + (1.0/x) * (polevl(w,ans,iter_n) / p1evl(w,bns,iter_m)))

    Parameters:
    ----------
    inp_x : the placeholder of data input

    Returns : A Tensor. Has the same type as data.
    -------
    """
    data_one = tbe.broadcast(tvm.const(Constant.NUM_ONE, inp_x.dtype), inp_x.shape)
    data_w = tbe.vdiv(data_one, inp_x)
    _res = polevl_plevl(data_w, ans, bns, iter_n, iter_m)
    mul_res = tbe.vmul(_res, data_w)
    add_res = tbe.vadd(mul_res, data_one)
    mul_val = tbe.vmul(add_res, data_w)
    exp_val = tbe.vexp(inp_x)
    res = tbe.vmul(mul_val, exp_val)

    return res


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name,too-many-locals
@register_operator_compute("Expint", op_mode="dynamic", support_fusion=True)
def expint_compute(x, y, kernel_name="expint"):
    """
    do element-wise expint compute

    Parameters:
    ----------
    x: the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name, default value is "cce_expint"

    Returns : A Tensor. Has the same type as data_input.
    -------
    """

    shape = x.shape
    dtype = x.dtype
    inp_dtype = dtype

    if dtype == "float16":
        x = tbe.cast_to(x, "float32")
        dtype = "float32"

    # 0 cannot be a divisor
    res_zero = tbe.vcmpsel(x, tvm.const(Constant.NUM_ZERO, dtype), 'gt', None,
                           tbe.broadcast(tvm.const(Constant.NEG_ONE, dtype), shape))
    final_zero = tbe.vcmpsel(x, tvm.const(Constant.NUM_ZERO, dtype), 'gt',
                             tbe.broadcast(tvm.const(Constant.NUM_ONE, dtype), shape),
                             tbe.broadcast(tvm.const(Constant.NUM_ZERO, dtype), shape))

    # Reference from cephes to calculate every condition result

    # x is divided into 2**1(2),2**2(4),2**3(8),2**4(16),2**5(32),2**6(64) parts
    res_two = expint_cal_two(res_zero, Constant.LIST_A, Constant.LIST_B, 6, 6)
    res_four = expint_cal_oth(res_zero, Constant.LIST_A6, Constant.LIST_B6, 8, 7)
    res_eight = expint_cal_oth(res_zero, Constant.LIST_A5, Constant.LIST_B5, 8, 8)
    res_sixteen = expint_cal_oth(res_zero, Constant.LIST_A2, Constant.LIST_B2, 10, 9)
    res_thirtytwo = expint_cal_oth(res_zero, Constant.LIST_A4, Constant.LIST_B4, 8, 8)
    res_sixtyfour = expint_cal_oth(res_zero, Constant.LIST_A7, Constant.LIST_B7, 6, 5)
    res_oth = expint_cal_oth(res_zero, Constant.LIST_A3, Constant.LIST_B3, 9, 9)

    # confirm x belongs to (0.0, 2.0) and [2.0, 4.0) and [4.0, 8.0)
    val_two = tbe.vcmpsel(x, tvm.const(2.0, dtype), 'lt', res_two, res_four)
    val_eight = tbe.vcmpsel(x, tvm.const(8.0, dtype), 'lt', res_eight, res_sixteen)
    tmp_four = tbe.vcmpsel(x, tvm.const(4.0, dtype), 'lt', val_two, val_eight)

    # confirm x belongs to [8.0, 16.0) and [16.0, 32.0)
    val_thirtytwo = tbe.vcmpsel(x, tvm.const(32.0, dtype), 'lt', res_thirtytwo, res_sixtyfour)
    tmp_sixteen = tbe.vcmpsel(x, tvm.const(16.0, dtype), 'lt', tmp_four, val_thirtytwo)

    # confirm x belongs to [32.0, 64.0) and [64.0, inf)
    tmp_sixtyfour = tbe.vcmpsel(x, tvm.const(64.0, dtype), 'lt', tmp_sixteen, res_oth)
    res = tbe.vmul(tmp_sixtyfour, final_zero)

    if inp_dtype == "float16":
        res = tbe.cast_to(res, inp_dtype)

    return res


@register_operator("Expint")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def expint(x, y, kernel_name="expint"):
    """
    ----------
    Parameters:
    ----------
    x : the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name, default value is "expint"

    Returns : None
    -------
    """
    dtype_input = x.get("dtype")
    inp_dtype = dtype_input.lower()

    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_input, check_list, param_name="x")

    ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for (_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])

            data_input = tvm.placeholder(x_shape[0], dtype=inp_dtype, name="data_x")

            res = expint_compute(data_input, y, kernel_name)
            tensors.append([data_input, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)

        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}

    tbe.build(schedules, config)
