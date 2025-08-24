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
asin

Op_description :
Computes acos of x element-wise

# asin(
#   x,
#   y,
#   kernel_name="cce_asin")

Supportive_dtype_format :
['bfloat16', 'float16', 'float32']
['ALL']

Constraint :
[1] All : shape size limit is 2147483648.
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util import util_compute
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    NUM_ONE = 1.0
    NEG_NUM_ONE = -1.0

    HALF_PI = 1.5707963267948966192313216916398

    BOUNDARY_1 = 0.70710678118654752440084436210485

    # Taylor coefficient
    COEF = (1.0,
        0.16666666666666666666666666666667,
        0.075,
        0.04464285714285714285714285714286,
        0.03038194444444444444444444444444,
        0.02237215909090909090909090909091,
        0.01735276442307692307692307692308,
        0.01396484375)

    # TAYLOR COUNT
    TAYLOR_COUNT = 7


def _taylor_compute(data_x, x_square=None):
    """
    do arcsinx compute use the 15th order taylor expansion when 0 <= x <= BOUNDARY_1
    asin(x) = x + 1/6*x^3 + 3/40*x^5 + 5/112*x^7 + ... + 13!!/(14!!*15)*x^15

    Parameters:
    ----------
    data_x : the placeholder of data input

    x_square : the placeholder of the square of data_x

    Returns : A Tensor. Has the same type as data.
    -------
    """

    if x_square is None:
        x_square = tbe.vmul(data_x, data_x)

    res = tbe.vmuls(x_square, tvm.const(Constant.COEF[Constant.TAYLOR_COUNT],
                                        x_square.dtype))
    for temp in reversed(range(Constant.TAYLOR_COUNT)):
        res = tbe.vadds(res, tvm.const(Constant.COEF[temp], x_square.dtype))
        if temp == 0:
            res = tbe.vmul(res, data_x)
        else:
            res = tbe.vmul(x_square, res)

    return res


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name,too-many-locals
@register_operator_compute("Asin", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def asin_compute(x, y, kernel_name="asin"):
    """
    do element-wise asin compute
    asin(x) = | arcsin(sqrt(1-x^2)) - HALF_PI, x belongs to (-1, -2^(-0.5))
              | the 15th order taylor expansion, x belongs to (-2^(-0.5), 2^(-0.5))
              | HALF_PI - arcsin(sqrt(1-x^2)), x belongs to (2^(-0.5), 1)

    Parameters:
    ----------
    x: the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name, default value is "cce_asin"

    Returns : A Tensor. Has the same type as data_input.
    -------
    """

    shape = x.shape
    dtype = x.dtype

    # Change dtype to float32
    if dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vadd", "float32"):
        x = tbe.cast_to(x, "float32")

    # Sign mask
    sign = util_compute.sign(x)

    # All positive
    x = tbe.vmul(x, sign)

    # x belongs to (0, 2^(-0.5))
    if tbe_platform.api_check_support("tbe.dsl.vmins", x.dtype):
        choice_1 = tbe.vmins(x, tvm.const(Constant.BOUNDARY_1, x.dtype))
    else:
        boundary_mask1 = tbe.broadcast(tvm.const(Constant.BOUNDARY_1, x.dtype), shape)
        choice_1 = tbe.vmin(x, boundary_mask1)

    if tbe_platform.api_check_support("tbe.dsl.vsubs", choice_1.dtype):
        choice_1 = tbe.vsubs(choice_1, tvm.const(Constant.BOUNDARY_1, choice_1.dtype))
    else:
        boundary_mask1 = tbe.broadcast(tvm.const(Constant.BOUNDARY_1, choice_1.dtype), shape)
        choice_1 = tbe.vsub(choice_1, boundary_mask1)

    choice_dtype = choice_1.dtype
    if choice_dtype != "float16" and tbe_platform.get_soc_spec("SHORT_SOC_VERSION") == "Ascend310":
        choice_1 = tbe.cast_to(choice_1, "float16")
    choice_1 = tbe.floor(choice_1)
    choice_1 = tbe.cast_to(choice_1, choice_dtype)
    choice_1 = tbe.vmuls(choice_1, Constant.NEG_NUM_ONE)

    res_1 = _taylor_compute(x)
    res_1 = tbe.vmul(res_1, choice_1)

    # x belongs to (2^(-0.5), 1)
    choice_2 = tbe.vmuls(choice_1, tvm.const(Constant.NEG_NUM_ONE, x.dtype))
    choice_2 = tbe.vadds(choice_2, tvm.const(Constant.NUM_ONE, x.dtype))

    res_2 = tbe.vmul(x, x)
    res_2 = tbe.vmuls(res_2, tvm.const(Constant.NEG_NUM_ONE, x.dtype))
    res_2 = tbe.vadds(res_2, tvm.const(Constant.NUM_ONE, x.dtype))
    res_2_sqrt = tbe.vsqrt(res_2)

    res_2 = _taylor_compute(res_2_sqrt, res_2)

    res_2 = tbe.vmuls(res_2, tvm.const(Constant.NEG_NUM_ONE, x.dtype))
    res_2 = tbe.vadds(res_2, tvm.const(Constant.HALF_PI, x.dtype))
    res_2 = tbe.vmul(res_2, choice_2)

    # Restore sign
    res_1 = tbe.vadd(res_1, res_2)
    res_1 = tbe.vmul(res_1, sign)

    # Restore dtype
    if dtype == "float16":
        res_1 = tbe.cast_to(res_1, "float16")

    return res_1


@register_operator("Asin")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def asin(x, y, kernel_name="asin"):
    """
    ----------
    asin(x) = | arcsin(sqrt(1-x^2)) - HALF_PI, x belongs to (-1, 2^(-0.5))
              | the 15th order taylor expansion, x belongs to (-2^(-0.5), 2^(-0.5))
              | HALF_PI - arcsin(sqrt(1-x^2)), x belongs to (2^(-0.5), 1)

    Parameters:
    ----------
    x : the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name, default value is "asin"

    Returns : None
    -------
    """
    shape_input = x.get("shape")
    x_dtype = x.get("dtype").lower()
    y_dtype = y.get("dtype").lower()

    para_check.check_shape(shape_input, param_name="x")

    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(x_dtype, check_list, param_name="x")
    if x_dtype != y_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal('asin', 'x_dtype', 'y_dtype', str(x_dtype), str(y_dtype))

    ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for (_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])

            data_input = tvm.placeholder(x_shape[0], dtype=x_dtype,
                                         name="data_input")

            res = asin_compute(data_input, y, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)

        schedules.append(sch)

    config = {"name": kernel_name,
              "print_ir": False,
              "tensor_list": tensors,
              "bool_storage_as_1bit": False}

    tbe.build(schedules, config)
