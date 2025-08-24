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
spence
"""
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm

from impl.dynamic.log import log_compute


class Constant:
    """
    the class for constant.
    """
    SCALAR_NAN = float("nan")
    SCALAR_ZERO = 0.0
    SCALAR_HALF = 0.5
    SCALAR_REV_HALF = -0.5
    SCALAR_ONE = 1.0
    SCALAR_THREE_HALF = 1.5
    SCALAR_TWO = 2.0
    SCALAR_PI = 3.1415926535897932384626433832795
    SCALAR_PI206 = SCALAR_PI * SCALAR_PI / 6.0

    SCALAR_A = (4.65128586073990045278E-5, 7.31589045238094711071E-3,
                1.33847639578309018650E-1, 8.79691311754530315341E-1,
                2.71149851196553469920E0, 4.25697156008121755724E0,
                3.29771340985225106936E0, 1.00000000000000000126E0)
    SCALAR_B = (6.90990488912553276999E-4, 2.54043763932544379113E-2,
                2.82974860602568089943E-1, 1.41172597751831069617E0,
                3.63800533345137075418E0, 5.03278880143316990390E0,
                3.54771340985225096217E0, 9.99999999999999998740E-1)

    A_COUNT = 7
    B_COUNT = 7


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


def _generic_spence_interval(x, output_y):
    """
    x < 2, y = x
    x >= 2, y = 1 / x

    y > 1.5, w = 1 / y - 1
    0.5 <= y <= 1.5, w = y - 1
    y < 0.5, w = -y

    output_y = -w * polevel(w, A, 7) / polevel(w, B, 7)
    y < 0.5, output_y = π * π / 6.0 - log(y) * log(1.0 - y) - output_y
    x > 1.5, output_y = - 0.5 * (log(y))^2 - output_y
    Parameters:
    ----------
    x: the placeholder of data input

    output_y : the dict of output

    Returns : A Tensor. Has the same type as data_input.
    -------
    """
    const_one = tbe.broadcast(tvm.const(Constant.SCALAR_ONE, x.dtype), x.shape)
    data_rec = tbe.vdiv(const_one, x)
    y = tbe.vcmpsel(x, rhs=Constant.SCALAR_TWO, operation='lt', slhs=x, srhs=data_rec)

    y_rec = tbe.vdiv(const_one, y)
    w_gt_three_half = tbe.vsub(y_rec, const_one)

    w_ge_half = tbe.vsub(y, const_one)

    rev_y = tbe.vmuls(y, tvm.const(-1.0, x.dtype))

    w_1 = tbe.vcmpsel(y, rhs=Constant.SCALAR_THREE_HALF, operation='gt', slhs=w_gt_three_half, srhs=w_ge_half)
    w = tbe.vcmpsel(y, rhs=Constant.SCALAR_HALF, operation='lt', slhs=rev_y, srhs=w_1)

    data_a_polevl = polevl(w, Constant.SCALAR_A, Constant.A_COUNT)
    data_b_polevl = polevl(w, Constant.SCALAR_B, Constant.B_COUNT)

    spence_pol = tbe.vmul(w, data_a_polevl)
    spence_pol = tbe.vdiv(spence_pol, data_b_polevl)
    spence_pol = tbe.vmuls(spence_pol, tvm.const(-1.0, x.dtype))

    z = log_compute(y, output_y)

    const_pi_206 = tbe.broadcast(tvm.const(Constant.SCALAR_PI206, x.dtype), x.shape)
    m = tbe.vsub(const_one, y)
    n = log_compute(m, output_y)
    spence_y_lt_half = tbe.vmul(z, n)
    spence_y_lt_half = tbe.vsub(const_pi_206, spence_y_lt_half)
    spence_y_lt_half = tbe.vsub(spence_y_lt_half, spence_pol)
    spence_1 = tbe.vcmpsel(y, rhs=Constant.SCALAR_HALF, operation='lt', slhs=spence_y_lt_half, srhs=spence_pol)

    sqare_z = tbe.vmul(z, z)
    spence_x_gt_three_half = tbe.vmuls(sqare_z, tvm.const(Constant.SCALAR_REV_HALF, x.dtype))
    spence_x_gt_three_half = tbe.vsub(spence_x_gt_three_half, spence_1)

    spence_res = tbe.vcmpsel(x, rhs=Constant.SCALAR_THREE_HALF, operation='gt', slhs=spence_x_gt_three_half,
                             srhs=spence_1)

    return spence_res


@register_operator_compute("Spence", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def spence_compute(x, y, kernel_name="spence"):
    """
    do element-wise spence compute
    x < 0, y = nan
    x = 0, y = π * π / 6.0
    x = 1, y = 0
    else, y = _generic_spence_interval(x)
    Parameters:
    ----------
    x: the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name, default value is "spence"

    Returns : A Tensor. Has the same type as data_input.
    -------
    """

    shape = x.shape
    dtype = x.dtype

    # Change dtype to float32
    if dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vadd", "float32"):
        x = tbe.cast_to(x, "float32")
    if tbe_platform.api_check_support("tik.vcopy"):
        zeros = tbe.vmuls(x, tvm.const(Constant.SCALAR_NAN, x.dtype))
    else:
        zeros = tbe.vmuls(x, tvm.const(Constant.SCALAR_ZERO, x.dtype))
    res_pi206 = tbe.broadcast(tvm.const(Constant.SCALAR_PI206, x.dtype), shape)
    res_zero = tbe.broadcast(tvm.const(Constant.SCALAR_ZERO, x.dtype), shape)
    res_other = _generic_spence_interval(x, y)

    res_1 = tbe.vcmpsel(x, rhs=Constant.SCALAR_ZERO, operation='lt', slhs=zeros, srhs=res_other)
    res_2 = tbe.vcmpsel(x, rhs=Constant.SCALAR_ZERO, operation='eq', slhs=res_pi206, srhs=res_1)
    res = tbe.vcmpsel(x, rhs=Constant.SCALAR_ONE, operation='eq', slhs=res_zero, srhs=res_2)

    # Restore dtype
    if dtype == "float16":
        res = tbe.cast_to(res, "float16")

    return res


@register_operator("Spence")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def spence(x, y, kernel_name="spence"):
    """
    Parameters:
    ----------
    x : the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name, default value is "asin"

    Returns : None
    -------
    """
    shape_input = x.get("shape")
    dtype_input = x.get("dtype").lower()

    para_check.check_shape(shape_input, param_name="x")

    check_list = ("bfloat16", "float16", "float32")
    para_check.check_dtype(dtype_input, check_list, param_name="x")

    ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])
            data_input = tvm.placeholder(x_shape[0], dtype=dtype_input,
                                         name="data_input")
            res = spence_compute(data_input, y, kernel_name)
            tensors.append([data_input, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
            schedules.append(sch)

    config = {"name": kernel_name,
              "print_ir": False,
              "tensor_list": tensors,
              "bool_storage_as_1bit": False}

    tbe.build(schedules, config)
