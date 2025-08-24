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
acos

Op_description :
Computes acos of x element-wise

# acos(
#   x,
#   y,
#   kernel_name='cce_acos')

Supportive_dtype_format :
['bfloat16', 'float16', 'float32']
['ND', 'NCHW', 'NHWC', 'NC1HWC0']

Constraint :
[1] All : shape size limit is 2147483648.

"""
from impl.util.platform_adapter import tbe_platform
from impl.util import util_compute
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_soc_common import after_v200
from impl.util.platform_adapter import OpTbeImplMode


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
    do arcsinx compute use the 15th order taylor expansion when
     0 <= x <= BOUNDARY_1
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

    res = tbe.vmuls(x_square, tvm.const(Constant.COEF[Constant.TAYLOR_COUNT], x_square.dtype))
    for temp in reversed(range(Constant.TAYLOR_COUNT)):
        res = tbe.vadds(res, tvm.const(Constant.COEF[temp], x_square.dtype))
        if temp == 0:
            res = tbe.vmul(res, data_x)
        else:
            res = tbe.vmul(x_square, res)

    return res


# 'pylint: disable=too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals
# 'pylint: disable=unused-argument,invalid-name
@register_operator_compute("Acos", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def acos_compute(x, y, kernel_name="acos"):
    """
    do element-wise acos compute using asin op
    acos(x) = HALF_PI - asin(x)

    asin(x) = | arcsin(sqrt(1-x^2)) - HALF_PI, x belongs to (-1, -2^(-0.5))
              | the 15th order taylor expansion, x belongs to (-2^(-0.5),
              | 2^(-0.5))
              | HALF_PI - arcsin(sqrt(1-x^2)), x belongs to (2^(-0.5), 1)

    Parameters:
    ----------
    x: the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name, default value is "acos"

    Returns : A Tensor. Has the same type as x.
    -------
    """

    shape = x.shape
    dtype = x.dtype

    has_improve_precision = False
    # Change dtype to float32
    if dtype == "float16" and \
            tbe_platform.api_check_support("te.lang.cce.vadd", "float32"):
        x = tbe.cast_to(x, "float32")
        has_improve_precision = True

    # Sign mask
    sign = util_compute.sign(x)

    # store origin input
    input_x = x

    # All positive
    x = tbe.vmul(x, sign)

    # x belongs to (0, 2^(-0.5))
    if tbe_platform.api_check_support("te.lang.cce.vmins", x.dtype):
        choice_1 = tbe.vmins(x, tvm.const(Constant.BOUNDARY_1, x.dtype))
    else:
        boundary_mask1 = tbe.broadcast(tvm.const(Constant.BOUNDARY_1, x.dtype), shape)
        choice_1 = tbe.vmin(x, boundary_mask1)

    if tbe_platform.api_check_support("te.lang.cce.vsubs", choice_1.dtype):
        choice_1 = tbe.vsubs(choice_1, tvm.const(Constant.BOUNDARY_1, choice_1.dtype))
    else:
        boundary_mask1 = tbe.broadcast(tvm.const(Constant.BOUNDARY_1, choice_1.dtype), shape)
        choice_1 = tbe.vsub(choice_1, boundary_mask1)

    choice_1 = tbe.vmuls(tbe.floor(choice_1), Constant.NEG_NUM_ONE)
    choice_1 = tbe.cast_to(choice_1, x.dtype)
    res_1 = _taylor_compute(x)
    res_1 = tbe.vmul(res_1, choice_1)

    # x belongs to (2^(-0.5), 1)
    choice_2 = tbe.vmuls(choice_1, tvm.const(Constant.NEG_NUM_ONE, x.dtype))
    choice_2 = tbe.vadds(choice_2, tvm.const(Constant.NUM_ONE, x.dtype))

    res_2 = tbe.vmul(x, x)
    res_2 = tbe.vmuls(res_2, tvm.const(Constant.NEG_NUM_ONE, x.dtype))
    res_2 = tbe.vadds(res_2, tvm.const(Constant.NUM_ONE, x.dtype))
    res_2_sqrt = tbe.vsqrt(res_2, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)

    res_2 = _taylor_compute(res_2_sqrt, res_2)
    res_2 = tbe.vmuls(res_2, tvm.const(Constant.NEG_NUM_ONE, x.dtype))
    res_2 = tbe.vadds(res_2, tvm.const(Constant.HALF_PI, x.dtype))
    res_2 = tbe.vmul(res_2, choice_2)

    # Restore sign of asin
    res_1 = tbe.vadd(res_1, res_2)
    res_1 = tbe.vmul(res_1, sign)
    res_1 = tbe.vmuls(res_1, tvm.const(Constant.NEG_NUM_ONE, x.dtype))
    res = tbe.vadds(res_1, tvm.const(Constant.HALF_PI, x.dtype))

    # Restore dtype
    if has_improve_precision:
        res = tbe.cast_to(res, "float16")

    if after_v200():
        # edge computing
        nan_val = tbe.broadcast(tvm.const(float("nan"), dtype), shape)
        mask_ge = tbe.vcmp(input_x, -1, 'ge')
        res_neg = tbe.vsel(mask_ge, res, nan_val)
        mask_le = tbe.vcmp(input_x, 1, 'le')
        res = tbe.vsel(mask_le, res_neg, nan_val)

    return res


@register_operator("Acos")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def acos(x, y, kernel_name="acos"):
    """
    ----------
    acos(x) = HALF_PI - asin(x)

    Parameters:
    ----------
    x : the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name, default value is "acos"

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
        error_detail = "dtype of x and y should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "x", "y", error_detail)

    ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for (_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])

            data_input = tvm.placeholder(x_shape[0], dtype=x_dtype, name="data_input")

            res = acos_compute(data_input, y, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)

        schedules.append(sch)

    config = {"name": kernel_name,
              "print_ir": False,
              "tensor_list": tensors,
              "bool_storage_as_1bit": False}

    tbe.build(schedules, config)
