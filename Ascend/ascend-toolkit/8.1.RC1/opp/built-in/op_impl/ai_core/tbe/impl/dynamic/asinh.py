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
asinh

Op_description :
Computes inverse hyperbolic sine of x element-wise

# asinh(
#   input_x,
#   output_y,
#   kernel_name="cce_asinh")

Supportive_dtype_format :
['float16', 'bfloat16', 'float32']
['ALL']

Constraint :
[1] All : shape size limit is 2147483648.

"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util import util_compute

from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpTbeImplMode
from impl.util.util_soc_common import after_v200


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # Log threshold
    CONST_LOG_THRESHOLD_1 = 0.6666666666666667
    CONST_LOG_THRESHOLD_2 = 0.3333333333333333

    # Log value
    LOG_FOUR_THREE = 0.28768207245178085
    LOG_FIVE_THREE = 0.5108256237659907

    # const value
    CONST_NEG_ONE = -1
    CONST_ZERO = 0
    CONST_ONE = 1
    CONST_TWO = 2

    CONST_THREE_FOUR = 0.75

    CONST_DOT_SIX = 0.6
    FLOAT_16_MAX = 32768

    CONST_LOG_TWO = 0.6931472
    CONST_SQRT_FP32_MAX = 1.8446743e19

    CONST_ZERO_VALUE = 1e-45
    CONST_ASINH_S_VALUE = 3.4028235e34
    CONST_COMPUTE_MIN_VALUE = 6.93147180559945286227e-01
    CONST_BRANCH_CONDITION = 0.00024414063


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,too-many-locals,invalid-name
def asinh_compute_mini(input_x, output_y, kernel_name="asinh"):
    """
    algrithm: asinh(x) = log(x + sqrt(x^2 + 1))

    Parameters
    ----------
    input_x: the placeholder of data input

    output_y : the dict of output

    kernel_name : cce kernel name, default value is "asinh"

    Returns
    -------
    res : result of asinh

    """

    min_fp16 = 2 ** (-24)
    inp_dtype = input_x.dtype.lower()
    shape = input_x.shape
    has_improve_precision = False
    if inp_dtype == "float16" and \
            tbe_platform.api_check_support("tbe.dsl.vrec", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        has_improve_precision = True

    input_x1 = tbe.vabs(input_x)
    # to fix bug for input data is 0.0
    input_x1 = tbe.vadds(input_x1, min_fp16)
    data_1_x = tbe.vrec(input_x1, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
    data_1_x_square = tbe.vmul(data_1_x, data_1_x)
    data_1_x_square = tbe.vadds(data_1_x_square, tvm.const(Constant.CONST_ONE, "float32"))
    data_s_1_sqrt = _newton_sqrt(data_1_x_square, inp_dtype)
    data_res = tbe.vmul(data_s_1_sqrt, input_x1)
    data_res = tbe.vadd(input_x1, data_res)
    result = _log_taylor(data_res, shape)
    res_neg = tbe.vmuls(result, tvm.const(Constant.CONST_NEG_ONE, inp_dtype))

    if input_x.dtype == result.dtype and tbe_platform.api_check_support("tbe.dsl.vcmpsel", input_x.dtype):
        res = tbe.vcmpsel(
            input_x,
            tvm.const(Constant.CONST_ZERO, input_x.dtype),
            'le',
            res_neg,
            result)
    else:
        const_zero_tensor = tbe.broadcast(tvm.const(Constant.CONST_ZERO, input_x.dtype), shape)
        compare_one = tbe.vcmp(input_x, const_zero_tensor, "le")
        res = tbe.vsel(compare_one, res_neg, result)

    if has_improve_precision:
        res = tbe.cast_to(res, "float16")
    else:
        res = tbe.cast_to(res, "float32")

    return res


@register_operator_compute("Asinh", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def asinh_compute_cloud(input_x, output_y, kernel_name="asinh"):
    """
    algrithm: asinh(x) = log(x + sqrt(x^2 + 1))

    Parameters
    ----------
    input_x: the placeholder of data input

    output_y : the dict of output

    kernel_name : cce kernel name, default value is "asinh"

    Returns
    -------
    res : result of asinh

    """
    if after_v200():
        return asinh_compute_cloud_new_formula(input_x, output_y)

    inp_dtype = input_x.dtype.lower()
    has_improve_precision = False
    check_support_flag = False
    if inp_dtype == "float16" and \
            tbe_platform.api_check_support("tbe.dsl.vlog", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        has_improve_precision = True
        inp_dtype = "float32"

    if inp_dtype == "float32" and \
            not tbe_platform.api_check_support("tbe.dsl.vlog", "float32"):
        input_x = tbe.cast_to(input_x, "float16")
        check_support_flag = True
        inp_dtype = "float16"

    data_abs = tbe.vabs(input_x)
    data_x_square = tbe.vmul(data_abs, data_abs)
    data_add = tbe.vadds(data_x_square, tvm.const(Constant.CONST_ONE, inp_dtype))
    data_s_1_sqrt = tbe.vsqrt(data_add)
    data_res = tbe.vadd(data_s_1_sqrt, data_abs)
    result = tbe.vlog(data_res)
    res = tbe.vmul(result, util_compute.sign(input_x))
    
    if has_improve_precision:
        res = tbe.cast_to(res, "float16")
    if check_support_flag is True:
        res = tbe.cast_to(res, "float32")

    return res


def asinh_compute_cloud_new_formula(input_x, output_y):
    """
    Parameters
    ----------
    input_x: the placeholder of data input

    output_y : the dict of output

    Returns
    -------
    res : result of asinh

    """

    input_dtype = input_x.dtype.lower()
    has_improve_precision = False
    check_support_flag = False
    if input_dtype == "float16" and \
            tbe_platform.api_check_support("tbe.dsl.vlog", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
        has_improve_precision = True
        input_dtype = "float32"

    if input_dtype == "float32" and \
            not tbe_platform.api_check_support("tbe.dsl.vlog", "float32"):
        input_x = tbe.cast_to(input_x, "float16")
        check_support_flag = True
        input_dtype = "float16"

    data_a = tbe.vabs(input_x)
    const_one_tensor = tbe.broadcast(tvm.const(Constant.CONST_ONE, input_dtype), input_x.shape)
    data_b = tbe.vdiv(const_one_tensor, data_a)
    data_b2 = tbe.vmul(data_b, data_b)

    result_1 = data_a

    data_b2_add_1 = tbe.vadds(data_b2, tvm.const(Constant.CONST_ONE, input_dtype))
    data_b2_sqrt = tbe.vsqrt(data_b2_add_1)
    data_b2_sqrt_add_b = tbe.vadd(data_b2_sqrt, data_b)
    data_div = tbe.vdiv(data_a, data_b2_sqrt_add_b)
    data_r = tbe.vadd(data_a, data_div)
    data_u = tbe.vadds(data_r, tvm.const(Constant.CONST_ONE, input_dtype))
    data_u_sub_1 = tbe.vadds(data_u, -1.0)
    data_s = tbe.vmaxs(data_u_sub_1, tvm.const(Constant.CONST_ZERO_VALUE, input_dtype))
    data_s = tbe.vmins(data_s, tvm.const(Constant.CONST_ASINH_S_VALUE, input_dtype))
    data_u_log = tbe.vlog(data_u)
    data_u_mul_r = tbe.vmul(data_u_log, data_r)
    data_r = tbe.vdiv(data_u_mul_r, data_s)
    data_a_log = tbe.vlog(data_a)
    data_s = tbe.vadds(data_a_log, Constant.CONST_COMPUTE_MIN_VALUE)
    data_s = tbe.vadd(data_s, data_b2)
    result_2 = tbe.vmin(data_r, data_s)
    
    cmp_m = tbe.vcmp(data_a, tvm.const(Constant.CONST_BRANCH_CONDITION, input_dtype), "lt", "bool")
    output = tbe.vsel(cmp_m, result_1, result_2)
    neg_output = tbe.vmuls(output, tvm.const(-1.0, input_dtype))

    cmp_x = tbe.vcmp(input_x, tvm.const(0.0, input_dtype), "ge", "bool")
    res = tbe.vsel(cmp_x, output, neg_output)

    if has_improve_precision:
        res = tbe.cast_to(res, "float16")
    if check_support_flag is True:
        res = tbe.cast_to(res, "float32")

    return res


def _newton_iter(data, data_x0, dtype):
    """
    algrithm: x(n+1) = 1/2 ( x(n) + a/x(n))

    Parameters
    ----------
    data: input tensor that we want to calculate sqrt

    data_x0 : input tensor of an approximate value of sqrt

    dtype : the type of tensor

    Returns
    -------
    data_newton : result of newton iter

    """
    # Newton begin:
    const_newton_factor = 0.5
    data_newton = tbe.vrec(data, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
    data_newton = tbe.vmul(data_x0, data_newton)
    data_newton = tbe.vadd(data_newton, data)
    data_newton = tbe.vmuls(data_newton, tvm.const(const_newton_factor, dtype))
    # Newton end
    return data_newton


def _newton_sqrt(data, dtype):
    """
    use three times to calculate sqrt

    Parameters
    ----------
    data: input tensor that we want to calculate sqrt

    dtype : the type of tensor

    Returns
    -------
    data_sqrt : return of sqrt

    """
    data_sqrt = tbe.vrsqrt(data)
    data_sqrt = tbe.vrec(data_sqrt, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
    data_sqrt = _newton_iter(data_sqrt, data, dtype)
    data_sqrt = _newton_iter(data_sqrt, data, dtype)
    data_sqrt = _newton_iter(data_sqrt, data, dtype)
    return data_sqrt


def _log_taylor(data_x, shape):
    """
    use taylor expansion to calculate log

    Parameters
    ----------
    data_x: input tensor that we want to calculate sqrt

    dtype : the type of tensor

    Returns
    -------
    res :  return of log

    """
    data = tbe.vadds(data_x, tvm.const(Constant.CONST_NEG_ONE, "float32"))
    data_1 = tbe.vadds(
        data,
        tvm.const(Constant.CONST_NEG_ONE * Constant.CONST_LOG_THRESHOLD_1, "float32"))
    if tbe_platform.api_check_support("te.lang.cce.vcmpsel", "float32"):
        data_sel = tbe.vcmpsel(
            data,
            tvm.const(Constant.CONST_LOG_THRESHOLD_1, data.dtype),
            'ge',
            tbe.vmuls(data_1, tvm.const(Constant.CONST_DOT_SIX, "float32")),
            data)
        data_sel = tbe.cast_to(data_sel, "float32")
        data_2 = tbe.vadds(
            data_sel,
            tvm.const(Constant.CONST_NEG_ONE * Constant.CONST_LOG_THRESHOLD_2, "float32"))
        data_vmuls = tbe.vmuls(
            data_2,
            tvm.const(Constant.CONST_THREE_FOUR, "float32"))
        data_sel_1 = tbe.vcmpsel(
            data_sel,
            tvm.const(Constant.CONST_LOG_THRESHOLD_2, data_sel.dtype),
            'ge',
            data_vmuls,
            data_sel)
        data_sel_1 = tbe.cast_to(data_sel_1, "float32")
        taylor = _taylor_compute(data_sel_1)
        # add log(4/3)
        res = tbe.vcmpsel(
            data_sel,
            tvm.const(Constant.CONST_LOG_THRESHOLD_2, data_sel.dtype),
            'ge',
            tbe.vadds(taylor, tvm.const(Constant.LOG_FOUR_THREE, "float32")),
            taylor)
        res = tbe.cast_to(res, "float32")
        # add log(5/3)
        data = tbe.cast_to(data, "float32")
        res = tbe.vcmpsel(
            data,
            tvm.const(Constant.CONST_LOG_THRESHOLD_1, data.dtype),
            'ge',
            tbe.vadds(taylor, tvm.const(Constant.LOG_FIVE_THREE, "float32")),
            res)
    else:
        threshold_1 = tbe.broadcast(
            tvm.const(Constant.CONST_LOG_THRESHOLD_1, "float32"), shape)
        index_1 = tbe.vcmp(data, threshold_1, 'ge')
        data_sel = tbe.vsel(
            index_1,
            tbe.vmuls(data_1, tvm.const(Constant.CONST_DOT_SIX, "float32")),
            data)
        data_sel = tbe.cast_to(data_sel, "float32")

        threshold_2 = tbe.broadcast(
            tvm.const(Constant.CONST_LOG_THRESHOLD_2, "float32"), shape)
        index_2 = tbe.vcmp(data_sel, threshold_2, 'ge')
        data_2 = tbe.vadds(
            data_sel,
            tvm.const(Constant.CONST_NEG_ONE * Constant.CONST_LOG_THRESHOLD_2, "float32"))
        data_vmuls = tbe.vmuls(
            data_2,
            tvm.const(Constant.CONST_THREE_FOUR, "float32"))
        data_sel = tbe.vsel(index_2, data_vmuls, data_sel)
        data_sel = tbe.cast_to(data_sel, "float32")
        taylor = _taylor_compute(data_sel)
        # add log(4/3)
        res = tbe.vsel(
            index_2,
            tbe.vadds(taylor, tvm.const(Constant.LOG_FOUR_THREE, "float32")),
            taylor)
        res = tbe.cast_to(res, "float32")
        # add log(5/3)
        res = tbe.vsel(index_1, tbe.vadds(taylor, tvm.const(Constant.LOG_FIVE_THREE, "float32")), res)
    res = tbe.cast_to(res, "float32")
    # d: vlog:
    res = _log_compute(data_x, res, shape)

    return res


def _taylor_compute(data):
    """
    algrithm: log(x) = ((((0.2x - 0.25)x + 0.33333)x - 0.5)x + 1)x

    Parameters
    ----------
    data: input tensor that we want to calculate log

    Returns
    -------
    None

    """
    const_newton_factor_neg = -0.5
    const_one_three = 0.3333333333333333
    const_one_five = 0.2
    const_one_four_neg = -0.25
    # 0.2x - 0.25
    taylor_five = tbe.vmuls(data, tvm.const(const_one_five, "float32"))
    taylor_four_1 = tbe.vadds(taylor_five, tvm.const(const_one_four_neg, "float32"))
    # (0.2x - 0.25)x + 0.33333
    taylor_four_2 = tbe.vmul(taylor_four_1, data)
    taylor_three_1 = tbe.vadds(taylor_four_2, tvm.const(const_one_three, "float32"))
    # ((0.2x - 0.25)x + 0.33333)x - 0.5
    taylor_three_2 = tbe.vmul(taylor_three_1, data)
    taylor_two_1 = tbe.vadds(
        taylor_three_2,
        tvm.const(const_newton_factor_neg, "float32"))
    # (((0.2x - 0.25)x + 0.33333)x - 0.5)x+1
    taylor_two_2 = tbe.vmul(taylor_two_1, data)
    taylor_one = tbe.vadds(taylor_two_2, tvm.const(Constant.CONST_ONE, "float32"))
    # ((((0.2x - 0.25)x + 0.33333)x - 0.5)x + 1)x
    taylor = tbe.vmul(taylor_one, data)

    return taylor


def _log_compute(data_x, res, shape):
    """
    when data > 2, use vlog directly
    when data > 32768, float16 will overflow, use log(x/2.5)+log(2.5)

    Parameters
    ----------
    data: input tensor that we want to calculate log

    Returns
    -------
    res : return of log

    """
    log_five_two = 0.916290731874155
    const_five_two = 0.4
    # if data > 2, use vlog
    if data_x.dtype == res.dtype and tbe_platform.api_check_support("tbe.dsl.vcmpsel", data_x.dtype):
        res = tbe.vcmpsel(
            data_x,
            tvm.const(Constant.CONST_TWO, data_x.dtype),
            'ge',
            tbe.vlog(data_x),
            res)
    else:
        threshold_3 = tbe.broadcast(tvm.const(Constant.CONST_TWO, "float32"), shape)
        index_3 = tbe.vcmp(data_x, threshold_3, 'ge')
        res = tbe.vsel(index_3, tbe.vlog(data_x), res)

    # if data > 32768, use log(x/2.5)+log(2.5)
    overflow_value = tbe.vmuls(data_x, const_five_two)
    res_overflow = tbe.vadds(
        tbe.vlog(overflow_value), log_five_two)
    if data_x.dtype == res.dtype and tbe_platform.api_check_support("tbe.dsl.vcmpsel", data_x.dtype):
        res = tbe.vcmpsel(
            data_x,
            tvm.const(Constant.FLOAT_16_MAX, data_x.dtype),
            'ge',
            res_overflow,
            res)
    else:
        float_16_max_tensor = tbe.broadcast(
            tvm.const(Constant.FLOAT_16_MAX, "float32"), shape)
        index_4 = tbe.vcmp(data_x, float_16_max_tensor, 'ge')
        res = tbe.vsel(index_4, res_overflow, res)
    res = tbe.cast_to(res, "float32")

    return res


@register_operator("Asinh")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def asinh(input_x, output_y, kernel_name="asinh"):
    """
    algrithm: asinh(x) = log(x + sqrt(x^2 + 1))

    Parameters
    ----------
    input_x: the dict of input_x, only support float16, bfloat16, float32

    output_y : the dict of output_y

    kernel_name : cce kernel name, default value is "asinh"

    Returns
    -------
    None

    """
    
    shape_input = input_x.get("shape")
    dtype_input = input_x.get("dtype")

    para_check.check_shape(shape_input, param_name="input_x")

    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(dtype_input, check_list, param_name="input_x")

    schedules, tensors = [], []
    ins = classify([input_x], OpPatternMode.ELEWISE)
    for (_input_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_input_x])
            data_input = tvm.placeholder(x_shape[0], dtype=dtype_input,
                                         name="data_input")
            res = asinh_compute_cloud(data_input, output_y, kernel_name)
            tensors.append([data_input, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {
        "name": kernel_name,
        "tensor_list": tensors,
        "bool_storage_as_1bit": False
    }

    tbe.build(schedules, config)
