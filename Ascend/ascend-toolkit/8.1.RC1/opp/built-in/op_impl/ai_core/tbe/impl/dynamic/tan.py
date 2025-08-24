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
dynamic tan
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # define a string name of "bfloat16"
    BFLOAT_16 = "bfloat16"
    # define a string name of "float16"
    FLOAT_16 = "float16"
    # define a string name of "float32"
    FLOAT_32 = "float32"
    # define a string name of "int32"
    INT_32 = "int32"
    # define the PI
    PI = 3.14159265
    # define the expansion order of Tan series
    TAN_EXPANSION_ORDER = 5
    # define the number of times using the tan2x formula
    TAN_2X_TIMES = 6
    # define the number of x div pi
    PI_FOR_X_TODIV = 0.3183098733425140380859375
    # define the number of x div pi
    KPI_FIRS_PI_MULS = 0.0009670257568359375
    # define the PI for compute_v2
    PI_V2 = 3.140625
    # define the number of down of pi_div
    PI_DOWN = 1.57079637050628662109375
    PI_DOWN_NEG = -1.57079637050628662109375
    # kpi_2
    KPI_TWI_PI_MULS = 6.2771141529083251953125e-7
    PI_RESDOWN_ADDS = 0.00000004371139000189375
    PI_RESDOWN_ADDS_NEG = -0.00000004371139000189375
    # kpi_3 kpi_4
    KPI_THIR_PI_MULS = 1.21644916362129151821136474609375e-10
    KPI_FOR_PI_MULS = -1.0291767438275201129727065563201904296875e-13
    # define the number of tan_compute_v2
    TAN_RES_MULIT_SCA = 0.0698520831551998762793
    TAN_RES_ADDICT_UP = -6.8711573651634203789
    TAN_2ADDS = 61.20362572811089435388
    TAN_3ADDS = -24.8048928861126769186219


def _tan_expand(input_x):
    """
    calculating `tan x = x + x^3/3 + 2*x^5/15 + 17*x^7/315 +`
                        `62*x^9/2835 + 1382*x^11/155925...(|x|<pi/2)`
    """
    # Taylor expansion coefficient
    factors = [1 / 3, 2 / 15, 17 / 315, 62 / 2835, 1382 / 155925]

    input_x_power = tbe.vmul(input_x, input_x)
    iter_value = input_x
    res = input_x

    for i, _ in enumerate(range(Constant.TAN_EXPANSION_ORDER)):
        iter_value = tbe.vmuls(
            tbe.vmul(input_x_power, iter_value), factors[i])
        res = tbe.vadd(res, iter_value)

    return res


def _tan_2x_multi(input_x, times):
    """
    calculating tan x by calculating tan (x/2^times) and
    using formula tan 2x = 2*tan x/(1-tan x*tan x) multiple times
    """
    # calculate tan (x/2^times)
    input_x_divide = tbe.vmuls(input_x, 1.0 / (2.0 ** times))
    res = _tan_expand(input_x_divide)

    while times != 0:
        # `using double angle formula: tan 2x = 2*tan x/(1-tan x*tan x)`

        res_denominator = tbe.vmuls(res, 2.0)
        tanx_square = tbe.vmul(res, res)
        res_numerator = tbe.vadds(tbe.vmuls(tanx_square, -1.0), 1.0)
        res = tbe.vdiv(res_denominator, res_numerator)
        times = times - 1
    return res


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
@register_operator_compute("Tan", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def tan_compute(x, y, kernel_name="tan"):
    """
    algorithm: tan
    calculating tan x using _tan_2x_multi

    Parameters
    ----------
    x : TVM tensor
        the placeholders of x
    y: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "tan"

    Returns
    -------
    res: TVM tensor
        the result of tan(x)
    """
    dtype = x.dtype

    has_improve_precision = False
    cast_dtype = Constant.FLOAT_16
    if tbe_platform.api_check_support("te.lang.cce.vdiv", "float32"):
        has_improve_precision = True
        cast_dtype = Constant.FLOAT_32

    # cast to type float32 when type is float16 or int32
    if dtype in (Constant.FLOAT_16, Constant.INT_32):
        if has_improve_precision:
            x = tbe.cast_to(x, Constant.FLOAT_32)

    # adjust x to [-pi/2,pi/2] using x = x-round(x/pi)*pi
    round_pi_div = tbe.round(tbe.vmuls(x, tvm.const(1.0 / Constant.PI, cast_dtype)))
    if has_improve_precision:
        round_pi_div = tbe.cast_to(round_pi_div, Constant.FLOAT_32)
    input_x = tbe.vsub(x, tbe.vmuls(round_pi_div, tvm.const(Constant.PI, cast_dtype)))

    res = _tan_2x_multi(input_x, Constant.TAN_2X_TIMES)

    # cast the dtype to original dtype
    if dtype in (Constant.FLOAT_16, Constant.INT_32):
        if has_improve_precision:
            res = tbe.cast_to(res, dtype)

    return res


# 'pylint: disable=locally-disabled,unused-argument,invalid-name,too-many-locals
@register_operator_compute("Tan", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def tan_compute_v2(x, y, kernel_name="tan"):
    """
    algorithm: tan
    calculating tan x using _tan_2x_multi

    Parameters
    ----------
    x : TVM tensor
        the placeholders of x
    y: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "tan"

    Returns
    -------
    res: TVM tensor
        the result of tan(x)
    """
    dtype = x.dtype

    has_improve_precision = False
    cast_dtype = Constant.FLOAT_16
    if tbe_platform.api_check_support("te.lang.cce.vdiv", "float32"):
        has_improve_precision = True
        cast_dtype = Constant.FLOAT_32

    # cast to type float32 when type is float16 or int32
    if dtype in (Constant.FLOAT_16, Constant.INT_32):
        if has_improve_precision:
            x = tbe.cast_to(x, Constant.FLOAT_32)

    # adjust x to [-pi/2,pi/2] using x = x-round(x/pi)*pi
    round_pi_div = tbe.round(tbe.vmuls(x, tvm.const(Constant.PI_FOR_X_TODIV, cast_dtype)))
    if has_improve_precision:
        round_pi_div = tbe.cast_to(round_pi_div, Constant.FLOAT_32)
    kpi_0 = tbe.vmuls(round_pi_div, tvm.const(Constant.PI_V2, cast_dtype))
    input_x = tbe.vsub(x, kpi_0)

    kpi_1 = tbe.vmuls(round_pi_div, tvm.const(Constant.KPI_FIRS_PI_MULS, cast_dtype))
    input_x = tbe.vsub(input_x, kpi_1)
    res_down1 = tbe.vadds(input_x, Constant.PI_DOWN)
    res_down2 = tbe.vadds(input_x, Constant.PI_DOWN_NEG)

    kpi_2 = tbe.vmuls(round_pi_div, tvm.const(Constant.KPI_TWI_PI_MULS, cast_dtype))
    input_x = tbe.vsub(input_x, kpi_2)
    res_down1 = tbe.vsub(res_down1, kpi_2)
    res_down2 = tbe.vsub(res_down2, kpi_2)
    res_down1 = tbe.vadds(res_down1, Constant.PI_RESDOWN_ADDS_NEG)
    res_down2 = tbe.vadds(res_down2, Constant.PI_RESDOWN_ADDS)

    kpi_3 = tbe.vmuls(round_pi_div, tvm.const(Constant.KPI_THIR_PI_MULS, cast_dtype))
    input_x = tbe.vsub(input_x, kpi_3)
    res_down1 = tbe.vsub(res_down1, kpi_3)
    res_down2 = tbe.vsub(res_down2, kpi_3)

    kpi_4 = tbe.vmuls(round_pi_div, tvm.const(Constant.KPI_FOR_PI_MULS, cast_dtype))
    input_x = tbe.vsub(input_x, kpi_4)
    res_down1 = tbe.vsub(res_down1, kpi_4)
    res_down2 = tbe.vsub(res_down2, kpi_4)

    x2 = tbe.vmul(input_x, input_x)
    res_up = tbe.vmuls(x2, Constant.TAN_RES_MULIT_SCA)
    res_up = tbe.vadds(res_up, Constant.TAN_RES_ADDICT_UP)
    res_up = tbe.vmul(res_up, x2)
    res_up = tbe.vadds(res_up, Constant.TAN_2ADDS)
    res_up = tbe.vmul(res_up, input_x)

    res_down = tbe.vadds(x2, Constant.TAN_3ADDS)
    res_down = tbe.vmul(res_down, res_down1)
    res_down = tbe.vmul(res_down, res_down2)
    res = tbe.vdiv(res_up, res_down)

    # cast the dtype to original dtype
    if dtype in (Constant.FLOAT_16, Constant.INT_32):
        if has_improve_precision:
            res = tbe.cast_to(res, dtype)

    return res


@register_operator("Tan")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME
                            )
def tan(input_x, output_y, kernel_name="tan"):
    """
    algorithm: tan
    calculating tan x = x + x^3/3 + 2*x^5/5 + 17*x^7/315 +
                        62*x^9/2835 + 1382*x^11/155925...(|x|<pi/2)

    Parameters
    ----------
    x: dict
        dict with keys(shape and dtype) of input
    y: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "tan"

    Returns
    -------
    None
    """
    dtype_input = input_x.get("dtype").lower()
    check_list = (Constant.BFLOAT_16, Constant.FLOAT_16, Constant.FLOAT_32, Constant.INT_32)
    para_check.check_dtype(dtype_input, check_list, param_name="input_x")
    ins = classify([input_x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_input_x,) in ins:
        with tbe.compute():
            shape_x = shape_util.variable_shape([_input_x])[0]
            data_input = tvm.placeholder(shape_x, name="data_input", dtype=dtype_input)
            res = tan_compute_v2(data_input, output_y, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensors
    }
    tbe.build(schedules, config)
