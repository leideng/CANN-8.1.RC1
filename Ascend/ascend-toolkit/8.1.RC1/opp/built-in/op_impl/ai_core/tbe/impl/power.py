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
power
"""
# 'pylint: disable=redefined-outer-name
import math
import functools
from tbe.dsl.api import auto_schedule
from tbe.dsl.api import build
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=too-many-locals,redefined-argument-from-local
def isclose(valuex, valuey, rel_tol=1e-08, abs_tol=0.0):
    """
    determines whether the values of two floating-point numbers are close or equal
    """
    return math.isclose(valuex, valuey, rel_tol=rel_tol, abs_tol=abs_tol)


# 'pylint: disable=unused-argument,too-many-locals
# 'pylint: disable=unrecognized-inline-option,unused-argument
def positive_compute(base, power, version, input_dtype):
    """
    calculate power for positive elements of base tensor

    Parameters
    ----------
    base: the base tensor
    power: attr power
    version: the product version
    input_dtype: dtype of input

    Returns
    ----------
    res: the result tensor
    """

    base_cast = base

    if input_dtype == "float16" and \
            tbe_platform.api_check_support("te.lang.cce.vexp", "float32") and \
            tbe_platform.api_check_support("te.lang.cce.vlog", "float32"):
        base_cast = tbe.cast_to(base, "float32")

    log_val = tbe.vlog(base_cast)
    mul_val = tbe.vmuls(log_val, power)
    exp_val = tbe.vexp(mul_val)

    if exp_val.dtype.lower() != input_dtype:
        exp_val = tbe.cast_to(exp_val, input_dtype)

    return exp_val


def negtive_compute(base, power, nan_values, version, input_dtype):
    """
    calculate power for negative elements of base tensor

    Parameters
    ----------
    base: the base tensor
    power: attr power
    nan_values: a tensor with nan values
    version: the product version
    input_dtype: dtype of input

    Returns
    ----------
    res: the result tensor
    """

    if float(power).is_integer():
        base_cast = base

        if input_dtype == "float16" and \
                tbe_platform.api_check_support("te.lang.cce.vexp", "float32") and \
                tbe_platform.api_check_support("te.lang.cce.vlog", "float32"):
            base_cast = tbe.cast_to(base, "float32")

        sign_value = math.pow(-1, power)
        abs_base_value = tbe.vabs(base_cast)
        log_value = tbe.vlog(abs_base_value)
        mul_value = tbe.vmuls(log_value, power)
        exp_value = tbe.vexp(mul_value)
        res = tbe.vmuls(exp_value, sign_value)

        if res.dtype.lower() != input_dtype:
            res = tbe.cast_to(res, input_dtype)

        return res

    return nan_values


def zero_compute(power, nan_values, zero_values):
    """
    calculate power for zero elements of base tensor

    Parameters
    ----------
    power: attr power
    nan_values: a tensor with nan values
    zero_values: a tensor with zero values

    Returns
    ----------
    res: the result tensor
    """

    if power > 0.0:
        return zero_values

    return nan_values


def power_scalar(input_x, base, power):
    """
    calculate power when attr scale is 0.0 and attr power is not

    Parameters
    ----------
    input_x: placeholder of input
    base: the base value, equals attr shift
    power: attr power

    Returns
    ----------
    res: the result when attr scale is 0.0 and attr power is not
    """

    tmp_zero = tbe.vmuls(input_x, 0)
    ones = tbe.vadds(tmp_zero, 1)
    zeros = tmp_zero

    if base > 0.0:
        res = tbe.vmuls(ones, math.pow(base, power))
        return res
    if base < 0.0:
        if float(power).is_integer():
            res = tbe.vmuls(ones, math.pow(base, power))
            return res

        # `return abnormal value`
        res = tbe.vrec(zeros)
        return res

    if power > 0:
        return zeros

    # `return abnormal value`
    res = tbe.vrec(zeros)

    return res


def zero_diff_scale_compute(input_x, shift, power):
    """
    calculate power when power*scale is 0.0

    Parameters
    ----------
    input_x: placeholder of input
    shift: attr shift
    power: attr power

    Returns
    ----------
    res: the result when power*scale is 0.0
    """

    if isclose(power, 0.0):
        tmp_zero = tbe.vmuls(input_x, 0)
        res = tbe.vadds(tmp_zero, 1)
        return res

    res = power_scalar(input_x, shift, power)

    return res


# 'pylint: disable=locally-disabled,unused-argument,too-many-arguments
# 'pylint: disable=too-many-locals
@register_operator_compute("power", op_mode="static", support_fusion=True)
def power_compute(input_x, output_y, power=1.0, scale=1.0, shift=0.0, kernel_name="power"):
    """
    calculate power according to different cases

    Parameters
    ----------
    input_x: placeholder of input
    power: attr power
    scale: attr scale
    shift: attr shift

    Returns
    ----------
    res: result of power
    """

    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    input_dtype = input_x.dtype.lower()

    diff_scale = power * scale

    if isclose(diff_scale, 0.0):
        res = zero_diff_scale_compute(input_x, shift, power)
        return res

    shift_scaled_x = tbe.vmuls(input_x, scale)
    shift_scaled_x = tbe.vadds(shift_scaled_x, shift)

    tmp_zero = tbe.vmuls(input_x, 0)
    zeros = tmp_zero

    nan_value = tbe.vrec(zeros)

    if isclose(power, 1.0):
        res = shift_scaled_x
        return res
    if isclose(power, 2.0):
        res = tbe.vmul(shift_scaled_x, shift_scaled_x)
        return res
    if isclose(power, 3.0):
        res = tbe.vmul(shift_scaled_x, shift_scaled_x)
        res = tbe.vmul(res, shift_scaled_x)
        return res

    positive_pow_val = \
        positive_compute(shift_scaled_x, power, cce_product, input_dtype)
    negative_pow_val = \
        negtive_compute(shift_scaled_x, power,
                        nan_value, cce_product, input_dtype)
    zero_pow_val = zero_compute(power, nan_value, zeros)

    res = tbe.vcmpsel(shift_scaled_x, zeros, 'gt', positive_pow_val, negative_pow_val)
    res = tbe.vcmpsel(shift_scaled_x, zeros, 'eq', zero_pow_val, res)

    return res


# 'pylint: disable=redefined-outer-name, too-many-arguments, unused-variable
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def power(input_x, output_y, power=1.0, scale=1.0, shift=0.0, kernel_name="power"):
    """
    calculate power of input tensor according to
    y = (x * scale + shift) ** power

    Parameters
    ----------
    input_x: dict of input, include shape and
    dtype, dtype support float16, float32
    output_y: dict of output, include shape and
    dtype, dtype support float16, float32
    power: attr power, default value is 1.0
    scale: attr scale, default value is 1.0
    shift: attr shift, default value is 0.0
    kernel_name: cce kernel name, default value is "power"

    Returns
    ----------
    None
    """

    shape = input_x.get("shape")
    input_dtype = input_x.get("dtype").lower()

    para_check.check_shape(shape, param_name="x")
    type_tuple = ("float16", "float32")
    para_check.check_dtype(input_dtype, type_tuple, param_name="x")

    fuseshape = [1]
    fuseshape[0] = functools.reduce(lambda x, y: x * y, shape)

    data_input = tvm.placeholder(fuseshape, name="data_input", dtype=input_dtype)

    cur_cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if cur_cce_product in ("Ascend310", "Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        if input_dtype == "float32":
            error_manager_vector.raise_err_input_dtype_not_supported("power", "input_x", "float16", str(input_dtype))

        res = power_compute(data_input, output_y, power, scale, shift, kernel_name)
    else:
        res = power_compute(data_input, output_y, power, scale, shift, kernel_name)

    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_input, res], "print_ir": True}

    build(sch, config)
