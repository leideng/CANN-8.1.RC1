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
power
"""
# 'pylint: disable=redefined-outer-name
import math
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import OpTbeImplMode


# 'pylint: disable=too-many-locals,redefined-argument-from-local
def isclose(valuex, valuey, rel_tol=1e-08, abs_tol=0.0):
    """
    determines whether the values of two floating-point numbers are close or equal
    """
    return math.isclose(valuex, valuey, rel_tol=rel_tol, abs_tol=abs_tol)


# 'pylint: disable=unused-argument,too-many-locals
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

    base_d_cast = base

    if input_dtype == "float16" and \
            tbe_platform.api_check_support("te.lang.cce.vexp", "float32") and \
            tbe_platform.api_check_support("te.lang.cce.vlog", "float32"):
        base_d_cast = tbe.cast_to(base, "float32")

    log_d_val = tbe.vlog(base_d_cast)
    mul_d_val = tbe.vmuls(log_d_val, power)
    exp_d_val = tbe.vexp(mul_d_val)

    if exp_d_val.dtype.lower() != input_dtype:
        exp_d_val = tbe.cast_to(exp_d_val, input_dtype)

    return exp_d_val


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
        base_d_cast = base

        if input_dtype == "float16" and \
                tbe_platform.api_check_support("te.lang.cce.vexp", "float32") and \
                tbe_platform.api_check_support("te.lang.cce.vlog", "float32"):
            base_d_cast = tbe.cast_to(base, "float32")

        sign_value = math.pow(-1, power)
        abs_base_d_value = tbe.vabs(base_d_cast)
        log_d_value = tbe.vlog(abs_base_d_value)
        mul_d_value = tbe.vmuls(log_d_value, power)
        exp_d_value = tbe.vexp(mul_d_value)
        res_d = tbe.vmuls(exp_d_value, sign_value)

        if res_d.dtype.lower() != input_dtype:
            res_d = tbe.cast_to(res_d, input_dtype)

        return res_d

    return nan_values


def zero_compute(power, nan_values_d, zero_values):
    """
    calculate power for zero elements of base tensor

    Parameters
    ----------
    power: attr power
    nan_values_d: a tensor with nan values
    zero_values: a tensor with zero values

    Returns
    ----------
    res: the result tensor
    """

    if power > 0.0:
        return zero_values

    return nan_values_d


def power_scalar(input_x_d, base, power):
    """
    calculate power when attr scale is 0.0 and attr power is not

    Parameters
    ----------
    input_x_d: placeholder of input
    base: the base value, equals attr shift
    power: attr power

    Returns
    ----------
    res: the result when attr scale is 0.0 and attr power is not
    """

    tmp_zero_d = tbe.vmuls(input_x_d, 0)
    ones = tbe.vadds(tmp_zero_d, 1)
    zeros = tmp_zero_d

    if base > 0.0:
        res_d = tbe.vmuls(ones, math.pow(base, power))
        return res_d
    if base < 0.0:
        if float(power).is_integer():
            res_d = tbe.vmuls(ones, math.pow(base, power))
            return res_d

        # abnormal value
        res_d = tbe.vrec(zeros, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
        return res_d

    if power > 0:
        return zeros

    # abnormal value
    res_d = tbe.vrec(zeros, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)

    return res_d


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
        tmp_zero_d = tbe.vmuls(input_x, 0)
        res_d = tbe.vadds(tmp_zero_d, 1)
        return res_d

    res_d = power_scalar(input_x, shift, power)

    return res_d


# 'pylint: disable=locally-disabled,unused-argument,too-many-arguments
# 'pylint: disable=too-many-locals
@register_operator_compute("Power", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def power_compute(input_x, output_y, power=1.0, scale=1.0,
                  shift=0.0, kernel_name="power"):
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

    nan_value = tbe.vrec(zeros, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)

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

    res = tbe.vcmpsel(shift_scaled_x, zeros,
                      'gt', positive_pow_val, negative_pow_val)
    res = tbe.vcmpsel(shift_scaled_x, zeros,
                      'eq', zero_pow_val, res)

    return res


# 'pylint:disable=redefined-argument-from-local
@register_operator("Power")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def power(input_x, output_y, power=1.0, scale=1.0,
          shift=0.0, kernel_name="power"):
    """
    calculate power of input tensor according to
    y = (x * scale + shift) ** power

    Parameters
    ----------
    input_x: dict of input, include shape and
    dtype, dtype support float16, float32,bfloat16
    output_y: dict of output, include shape and
    dtype, dtype support float16, float32,bfloat16
    power: attr power, default value is 1.0
    scale: attr scale, default value is 1.0
    shift: attr shift, default value is 0.0
    kernel_name: cce kernel name, default value is "power"

    Returns
    ----------
    None
    """
    input_dtype = input_x.get("dtype").lower()
    type_tuple = ("bfloat16", "float16", "float32", "int32")
    para_check.check_dtype(input_dtype, type_tuple, param_name="x")
    ins = classify([input_x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    cur_cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    for (input_x,) in ins:
        with tbe.compute():
            # shape
            x_shape = shape_util.variable_shape([input_x])
            data_input = tvm.placeholder(x_shape[0], name="data_input",
                                         dtype=input_dtype)
            if cur_cce_product in ("Ascend310", "Hi3796CV300ES", "Hi3796CV300CS"):
                if input_dtype == "float32":
                    error_manager_vector.raise_err_input_dtype_not_supported("power", "input_x",
                                                                             "float16", str(input_dtype))

                res = power_compute(data_input, output_y, power, scale, shift, kernel_name)
            else:
                res = power_compute(data_input, output_y, power, scale, shift, kernel_name)

            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {
        "name": kernel_name,
        "tensor_list": tensors,
        "print_ir": True
    }

    tbe.build(schedules, config)
