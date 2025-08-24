# Copyright 2021 Huawei Technologies Co., Ltd
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
dynamic xlogy
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import OpTbeImplMode
from impl.util.util_soc_common import after_v200


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant.
    """
    SCALAR_NEG_ONE = -1.0
    SCALAR_ONE = 1.0
    TAYLOR_NEGATIVE_THRESHOLD = -1.7
    TAYLOR_POSITIVE_THRESHOLD = 0.7
    TAYLOR_SECOND_ORDER_PARAM = 1 / 2.0
    TAYLOR_THIRD_ORDER_PARAM = 1 / 6.0
    TAYLOR_FOURTH_ORDER_PARAM = 1 / 24.0
    TAYLOR_FIFTH_ORDER_PARAM = 1 / 120.0
    TAYLOR_SIXTH_ORDER_PARAM = 1 / 720.0
    TAYLOR_SEVENTH_ORDER_PARAM = 1 / 5040.0


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals
@register_operator_compute("Xlogy", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def xlogy_compute(input_x, input_y, output_z, kernel_name="xlogy"):
    """
    algorithm: xlogy
    calculating data's xlogy, res = 0 if x == 0 else x*log(y)
    in cloud scene, for all inputs :
    res = 0 if x == 0 else x*log(y)
    in mini scene :
    z(n+1) = z(n) - (e^(z(n)*x(n)^-1) - y(n))/x(n)^-1*e^(z(n)*x(n)^-1)
    f(z) = e^(z(n)*x(n)^-1)
    z(n)*x(n)^-1 <= Constant.TAYLOR_NEGATIVE_THRESHOLD or z(n)*x(n)^-1 >=
    Constant.TAYLOR_POSITIVE_THRESHOLD
    f(z) = seventh taylor computer
    Constant.TAYLOR_NEGATIVE_THRESHOLD < z(n)*x(n)^-1 < Constant.TAYLOR_POSITIVE_THRESHOLD

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    input_y: TVM tensor
        the placeholder of input_y
    output_z: dict
        dict info of output_z
    kernel_name: str
        kernel name, default value is "xlogy"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    shape_list = shape_util.broadcast_shapes(
        input_x.shape,
        input_y.shape,
        param_name_input1="input_x",
        param_name_input2="input_y")

    shape = shape_list[2]
    dtype = input_x.dtype
    dtype_change = input_x.dtype
    cloud_check = tbe_platform.api_check_support("tbe.dsl.vlog", "float32")
    mini_check = tbe_platform.api_check_support("tbe.dsl.vmul", "float32")
    data_x_broad = tbe.broadcast(input_x, shape_list[2])
    data_y_broad = tbe.broadcast(input_y, shape_list[2])
    if dtype == "float16" and cloud_check:
        dtype_change = "float32"
        data_x_broad = tbe.cast_to(data_x_broad, dtype_change)
        data_y_broad = tbe.cast_to(data_y_broad, dtype_change)

    if dtype_change == "float32" and not cloud_check:
        data_y_broad_16 = tbe.cast_to(data_y_broad, "float16")
        data_log_16 = tbe.vlog(data_y_broad_16)
        data_log = tbe.cast_to(data_log_16, dtype_change)
    else:
        data_log = tbe.vlog(data_y_broad)

    res = tbe.vmul(data_log, data_x_broad)

    if after_v200():
        scalar_zero = tvm.const(0, data_x_broad.dtype)
        mask = tbe.vcmp(data_x_broad, scalar_zero, "eq", "bit")
        res = tbe.vsel(mask, scalar_zero, res)
    elif (not cloud_check) and mini_check:
        dtype_change = "float32"
        data_x_broad = tbe.cast_to(data_x_broad, dtype_change)
        data_y_broad = tbe.cast_to(data_y_broad, dtype_change)
        res = _xlogy_mini_compute(res, data_x_broad, data_y_broad, shape, dtype_change)

    if dtype == "float16" and (cloud_check or mini_check):
        res = tbe.cast_to(res, "float16")

    return res


def _xlogy_mini_compute(res_mini, input_x, input_y, shape, dtype_change):
    """
    do element-wise x*log(y) compute in mini scene
    f(z) = e^(z(n)*x(n)^-1)
    z(n)*x(n)^-1 <= Constant.TAYLOR_NEGATIVE_THRESHOLD or z(n)*x(n)^-1 >=
    Constant.TAYLOR_POSITIVE_THRESHOLD
    f(z) = seventh taylor computer
    Constant.TAYLOR_NEGATIVE_THRESHOLD < z(n)*x(n)^-1 < Constant.TAYLOR_POSITIVE_THRESHOLD

    Parameters:
    ----------
    mini_res: TVM tensor, the tensor of x*log(y)
    input_x : TVM tensor, the placeholder of input_x
    input_y : TVM tensor, the placeholder of input_y
    shape : tuple, the shape of mini_res

    Returns : A Tensor. Has the same type as mini_res.
    -------
    """
    input_z = tbe.cast_to(res_mini, dtype_change)

    input_x_rec = tbe.vrec(input_x, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
    input_z_compare = tbe.vmul(input_z, input_x_rec)

    newton_taylor_res = _newton_taylor_xlogy(input_x, input_y, input_z)
    newton_exp_res = _newton_exp_xlogy(input_x, input_y, input_z, dtype_change)

    input_left_border = tvm.const(Constant.TAYLOR_NEGATIVE_THRESHOLD, "float32")
    tensor_input_left_border = tbe.broadcast(input_left_border, shape)

    input_right_border = tvm.const(Constant.TAYLOR_POSITIVE_THRESHOLD, "float32")
    tensor_input_right_border = tbe.broadcast(input_right_border, shape)

    if not tbe_platform.api_check_support("tbe.dsl.vcmp", "float32"):
        input_z_compare = tbe.cast_to(input_z_compare, "float16")
        newton_taylor_res = tbe.cast_to(newton_taylor_res, "float16")
        newton_exp_res = tbe.cast_to(newton_exp_res, "float16")
        tensor_input_left_border = tbe.cast_to(tensor_input_left_border, "float16")
        tensor_input_right_border = tbe.cast_to(tensor_input_right_border, "float16")

    b_gt_left_border = tbe.vcmp(input_z_compare, tensor_input_left_border, 'gt')
    exp_taylor_neg = tbe.vsel(b_gt_left_border, newton_taylor_res, newton_exp_res)

    b_lt_right_border = tbe.vcmp(input_z_compare, tensor_input_right_border, 'lt')
    data_xlogy = tbe.vsel(b_lt_right_border, exp_taylor_neg, newton_exp_res)

    return data_xlogy


def _exp_taylor_compute(input_x):
    """
    Calculate e^x, Use seventh order taylor expansion
    e^x = 1 + x + (x^2 / 2!) + (x^3 / 3!) +  (x^4 / 4!) +
    (x^5 / 5!) + (x^6 / 6!) + (x^7 / 7!)

    Parameters:
    ----------
    input_x : TVM tensor, the placeholder of input_x

    Returns : A Tensor. Has the same type as input_x.
    -------
    """
    # calculate second order tayloy section : x^2 / 2!
    taylor_second_order_param = tvm.const(Constant.TAYLOR_SECOND_ORDER_PARAM, "float32")
    data_power_2 = tbe.vmul(input_x, input_x)
    data_power_2_div_2 = tbe.vmuls(data_power_2, taylor_second_order_param)

    # calculate third order tayloy section : x^3 / 3!
    taylor_third_order_param = tvm.const(Constant.TAYLOR_THIRD_ORDER_PARAM, "float32")
    data_power_3 = tbe.vmul(data_power_2, input_x)
    data_power_3_div_6 = tbe.vmuls(data_power_3, taylor_third_order_param)

    # calculate fourth order tayloy section : x^4 / 4!
    taylor_fourth_order_param = tvm.const(Constant.TAYLOR_FOURTH_ORDER_PARAM, "float32")
    data_power_4 = tbe.vmul(data_power_3, input_x)
    data_power_4_div_24 = tbe.vmuls(data_power_4, taylor_fourth_order_param)

    # calculate fifth order tayloy section : x^5 / 5!
    taylor_fifth_order_param = tvm.const(Constant.TAYLOR_FIFTH_ORDER_PARAM, "float32")
    data_power_5 = tbe.vmul(data_power_4, input_x)
    data_power_5_div_120 = tbe.vmuls(data_power_5, taylor_fifth_order_param)

    # xcalculate sixth order tayloy section : ^6 / 6!
    taylor_sixth_order_param = tvm.const(Constant.TAYLOR_SIXTH_ORDER_PARAM, "float32")
    data_power_6 = tbe.vmul(data_power_5, input_x)
    data_power_6_div_720 = tbe.vmuls(data_power_6, taylor_sixth_order_param)

    # calculate seventh order tayloy section : x^7 / 7!
    taylor_seventh_order_param = tvm.const(Constant.TAYLOR_SEVENTH_ORDER_PARAM,
                                           "float32")
    data_power_7 = tbe.vmul(data_power_6, input_x)
    data_power_7_div_5040 = tbe.vmuls(data_power_7, taylor_seventh_order_param)

    # calculate first order tayloy plus one section : 1 + x
    res_first_taylor = tbe.vadds(input_x, tvm.const(Constant.SCALAR_ONE, "float32"))
    res_second_taylor = tbe.vadd(res_first_taylor, data_power_2_div_2)
    res_third_taylor = tbe.vadd(res_second_taylor, data_power_3_div_6)
    res_fourth_taylor = tbe.vadd(res_third_taylor, data_power_4_div_24)
    res_fifth_taylor = tbe.vadd(res_fourth_taylor, data_power_5_div_120)
    res_sixth_taylor = tbe.vadd(res_fifth_taylor, data_power_6_div_720)
    res = tbe.vadd(res_sixth_taylor, data_power_7_div_5040)

    return res


def _newton_exp_iter(input_x, input_y, input_z, dtype_change):
    """
    do element-wise Newton compute
    z(n+1) = z(n) - (e^(z(n)*x(n)^-1) - y(n))/x(n)^-1*e^(z(n)*x(n)^-1)

    Parameters:
    ----------
    input_x: TVM tensor, the placeholder of input_x
    input_y: TVM tensor, the placeholder of input_y
    input_z: start value of Newton iteration

    Returns : A Tensor. Has the same type as input_z.
    -------
    """
    # Newton begin: `z(n+1) = z(n) - x(n) + x(n)*y(n)*e^(-z(n)*x(n)^-1)`
    input_x_mul = tbe.vmuls(input_x, tvm.const(Constant.SCALAR_NEG_ONE, "float32"))
    newton_exp = tbe.vadd(input_x_mul, input_z)
    input_xy = tbe.vmul(input_x, input_y)
    input_x_rec = tbe.vrec(input_x, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
    input_x_res = tbe.vmuls(input_x_rec, tvm.const(Constant.SCALAR_NEG_ONE, "float32"))
    input_z_mul = tbe.vmul(input_x_res, input_z)
    if not tbe_platform.api_check_support("tbe.dsl.vexp", "float32") and dtype_change == "float32":
        input_z_mul_16 = tbe.cast_to(input_z_mul, "float16")
        input_z_exp_16 = input_z_exp = tbe.vexp(input_z_mul_16)
        input_z_exp = tbe.cast_to(input_z_exp_16, "float32")
    else:
        input_z_exp = tbe.vexp(input_z_mul)

    input_z_res = tbe.vmul(input_z_exp, input_xy)
    newton_exp = tbe.vadd(newton_exp, input_z_res)

    return newton_exp


def _newton_taylor_iter(input_x, input_y, input_z):
    """
    do element-wise Newton compute
    z(n+1) = z(n) - (e^(z(n)*x(n)^-1) - y(n))/x(n)^-1*e^(z(n)*x(n)^-1)

    Parameters:
    ----------
    input_x: TVM tensor, the placeholder of input_x
    input_y: TVM tensor, the placeholder of input_y
    input_z: start value of Newton iteration

    Returns : A Tensor. Has the same type as input_z.
    -------
    """
    # Newton begin: `z(n+1) = z(n) - x(n) + x(n)*y(n)*e^(-z(n)*x(n)^-1)`
    input_x_mul = tbe.vmuls(input_x, tvm.const(Constant.SCALAR_NEG_ONE, "float32"))
    newton_taylor = tbe.vadd(input_x_mul, input_z)
    input_xy = tbe.vmul(input_x, input_y)
    input_x_rec = tbe.vrec(input_x, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
    input_x_res = tbe.vmuls(input_x_rec, tvm.const(Constant.SCALAR_NEG_ONE, "float32"))
    input_z_mul = tbe.vmul(input_x_res, input_z)
    input_z_taylor = _exp_taylor_compute(input_z_mul)
    input_z_res = tbe.vmul(input_z_taylor, input_xy)
    newton_taylor = tbe.vadd(newton_taylor, input_z_res)

    return newton_taylor


def _newton_exp_xlogy(input_x, input_y, output_z, dtype_change):
    """
    do element-wise Newton compute
    z(n+1) = z(n) - (e^(z(n)*x(n)^-1) - y(n))/x(n)^-1*e^(z(n)*x(n)^-1)

    Parameters:
    ----------
    input_x: TVM tensor, the placeholder of input_x
    input_y: TVM tensor, the placeholder of input_y
    output_z: TVM tensor, start value of xlogy's Newton iteration

    Returns : A Tensor. Has the same type as output_z.
    -------
    """
    for _ in range(2):
        output_z = _newton_exp_iter(input_x, input_y, output_z, dtype_change)
    return output_z


def _newton_taylor_xlogy(input_x, input_y, output_z):
    """
    do element-wise Newton compute
    z(n+1) = z(n) - (e^(z(n)*x(n)^-1) - y(n))/x(n)^-1*e^(z(n)*x(n)^-1)

    Parameters:
    ----------
    input_x: TVM tensor, the placeholder of input_x
    input_y: TVM tensor, the placeholder of input_y
    output_z: TVM tensor, start value of xlogy's Newton iteration

    Returns : A Tensor. Has the same type as output_z.
    -------
    """
    for _ in range(2):
        output_z = _newton_taylor_iter(input_x, input_y, output_z)
    return output_z


@register_operator("Xlogy")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def xlogy(input_x, input_y, output_z, kernel_name="xlogy"):
    """
    algorithm: xlogy
    calculating data's xlogy, res = 0 if x == 0 else x*log(y)

    Parameters
    ----------
    input_x: dict
        dict of input_x, include keys(shape and dtype)
    input_y: dict
        dict of input_y, include keys(shape and dtype)
    output_z: dict
        dict info of output_z
    kernel_name: str
        kernel name, default value is "xlogy"

    Returns
    -------
    None
    """
    dtype = input_x.get("dtype")
    dtype_y = input_y.get("dtype")
    input_dtype = dtype.lower()
    input_dtype_y = dtype_y.lower()
    check_list = ("bfloat16", "float16", "float32")
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")
    para_check.check_dtype(input_dtype_y, check_list, param_name="input_y")
    if dtype != dtype_y:
        error_manager_vector.raise_err_inputs_dtype_not_equal("Xlogy", "input_x", "input_y",
                                                              str(dtype), str(dtype_y))

    ins = classify([input_x, input_y], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_input_x, _input_y) in ins:
        with tbe.compute():
            # shape
            shape_x1, shape_x2 = shape_util.variable_shape([_input_x, _input_y])
            # mul_compute
            data_x1 = tvm.placeholder(shape_x1, dtype=dtype, name="data_x1")
            data_x2 = tvm.placeholder(shape_x2, dtype=dtype_y, name="data_x2")
            res = xlogy_compute(data_x1, data_x2, output_z, kernel_name)
            tensors.append((data_x1, data_x2, res))
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
