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
atan2
"""
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute
from impl.util import util_compute


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    the class for constant.
    """
    CONST_POS_ONE = 1.0
    CONST_NA_ONE = -1.0
    CONST_PI = 3.1415926535897932384626433832795
    CONST_PI_BY_TWO = 1.5707963267948966192313216916398
    CONST_PI_BY_FOUR = 0.78539816339744830961566084581988
    CONST_PI_BY_EIGHT = 0.39269908169872415480783042290994
    CONST_ITERTOR = 6
    CONST_ITERTOR2 = 4
    TAN_PI_BY_EIGHT = 0.4142135623730950
    TAN_PI_BY_EIGHT_NA = -0.4142135623730950

    CONST_ZERO = 0
    CONST_ONE = 1

    TAYLOR = (1.0, -1.0 / 3, 1.0 / 5, -1.0 / 7, 1.0 / 9, -1.0 / 11, 1.0 / 13)


# 'pylint: disable=too-many-locals,unused-variable
def _do_taylor(input_data):
    """
    Algorithm:
        if x > 0 and x < tan(pi/8):
            atan(x) = x - x^3/3 + x^5/5 - x^7/7 ...
        elif x > tan(pi/8) and x < tan(pi/4):
            atan(x) = atan(y) + atan((x-y)/(1+xy))

    ----------------------------------
    Parameters:

        input_data: Input data.

    ----------------------------------
    Returns:

        A Tensor of atan(x).

    """

    denominator_data = tbe.vmuls(input_data, Constant.TAN_PI_BY_EIGHT)
    denominator_data = tbe.vadds(denominator_data, Constant.CONST_POS_ONE)
    molecule = tbe.vadds(input_data, Constant.TAN_PI_BY_EIGHT_NA)
    data = tbe.vdiv(molecule, denominator_data)
    data = tbe.vabs(data)

    square_data = tbe.vmul(data, data)
    res = tbe.vmuls(square_data, Constant.TAYLOR[Constant.CONST_ITERTOR])
    res = tbe.vadds(res, Constant.TAYLOR[Constant.CONST_ITERTOR - 1])
    for i in reversed(range(Constant.CONST_ITERTOR - 1)):
        res = tbe.vmul(res, square_data)
        res = tbe.vadds(res, Constant.TAYLOR[i])
    res = tbe.vmul(res, data)
    res = tbe.vadds(res, Constant.CONST_PI_BY_EIGHT)

    square_data = tbe.vmul(input_data, input_data)
    res2 = tbe.vmuls(square_data, Constant.TAYLOR[Constant.CONST_ITERTOR2])
    res2 = tbe.vadds(res2, Constant.TAYLOR[Constant.CONST_ITERTOR2 - 1])
    for i in reversed(range(Constant.CONST_ITERTOR2 - 1)):
        res2 = tbe.vmul(res2, square_data)
        res2 = tbe.vadds(res2, Constant.TAYLOR[i])
    res2 = tbe.vmul(res2, input_data)

    res = tbe.vmin(res, res2)

    return res


def _atan_compute(input_x):
    """
    Algorithm: atan

    ----------------------------------
    Parameters:

        input_x: Input data.

    ----------------------------------
    Returns:

        A Tensor of atan(x).

    """

    shape = input_x.shape
    dtype = input_x.dtype
    if dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vadd", "float32"):
        input_x = tbe.cast_to(input_x, "float32")
    abs_data = tbe.vabs(input_x)

    tensor_one = tbe.broadcast(tvm.const(Constant.CONST_POS_ONE, input_x.dtype), shape)

    abs_data_sub_one = tbe.vsub(abs_data, tensor_one)
    abs_data_add_one = tbe.vadd(abs_data, tensor_one)
    abs_data2 = tbe.vdiv(abs_data_sub_one, abs_data_add_one)
    abs_data2 = tbe.vabs(abs_data2)

    # calucate data less than one
    res = _do_taylor(abs_data)
    # calucate data more than one
    res_mt_one = _do_taylor(abs_data2)
    res_mt_one = tbe.vadds(res_mt_one, Constant.CONST_PI_BY_FOUR)

    res = tbe.vmin(res, res_mt_one)

    sign_mask = util_compute.sign(input_x)
    res = tbe.vmul(res, sign_mask)

    if dtype == "float16":
        res = tbe.cast_to(res, "float16")
    return res


def _init_atan2_mask(data_y, data_x):
    """
    Algorithm: atan2

    ----------------------------------
    Parameters:

        data_y: the y of atan2(y, x)

        data_x: the x of atan2(y, x)
    ----------------------------------
    Returns:

        mask: the mask of x's and y's value

    """

    shape_input = data_y.shape
    dtype_input = data_y.dtype

    tensor_one = tbe.broadcast(tvm.const(Constant.CONST_POS_ONE, dtype_input), shape_input)
    tensor_zero = tbe.broadcast(tvm.const(Constant.CONST_ZERO, dtype_input), shape_input)
    tensor_na_one = tbe.vmuls(tensor_one, tvm.const(Constant.CONST_NA_ONE, dtype_input))

    y_me_zero = tbe.vsel(tbe.vcmp(data_y, tensor_zero, 'ge'), tensor_one, tensor_na_one)
    x_lt_zero_y_mask = tbe.vsel(
        tbe.vcmp(data_x, tensor_zero, 'lt'), y_me_zero, tensor_zero)

    y_cmp_zero = tbe.vsel(tbe.vcmp(data_y, tensor_zero, 'ge'), tensor_one, tensor_na_one)

    mask = (x_lt_zero_y_mask, y_cmp_zero)
    return mask


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
@register_operator_compute("atan2", op_mode="static", support_fusion=True)
def atan2_compute(y, x, output_dict, kernel_name="atan2"):
    """
    Algorithm: atan2
    ----------------------------------
    Parameters:

        y: Input data y.

        x: Input data x.

        kernel_name: cce kernel name, default value is "atan2"
    ----------------------------------
    Returns:

        A Tensor of atan2(x).

    """

    shape_y = y.shape
    dtype_y = y.dtype
    shape_x = x.shape

    shape_y = shape_util.shape_to_list(shape_y)
    shape_x = shape_util.shape_to_list(shape_x)
    shape_y, shape_x, shape_broadcast = shape_util.broadcast_shapes(shape_y,
                                                                    shape_x,
                                                                    param_name_input1="x1",
                                                                    param_name_input2="x2")
    y = tbe.broadcast(y, shape_broadcast)
    x = tbe.broadcast(x, shape_broadcast)

    if dtype_y == "float16" and tbe_platform.api_check_support("tbe.dsl.vadd", "float32"):
        y = tbe.cast_to(y, "float32")
        x = tbe.cast_to(x, "float32")

    mask = _init_atan2_mask(y, x)

    # caculate the atan(y/x) when x > 0
    res = tbe.vdiv(y, x)
    res = _atan_compute(res)

    y_cmp_zero = tbe.vmuls(mask[Constant.CONST_ONE], tvm.const(Constant.CONST_PI_BY_TWO, y.dtype))
    res_x_lt_zero = tbe.vmuls(mask[Constant.CONST_ZERO], tvm.const(Constant.CONST_PI, y.dtype))

    if x.dtype == res.dtype and tbe_platform.api_check_support("tbe.dsl.vcmpsel", x.dtype):
        res = tbe.vcmpsel(x, tvm.const(Constant.CONST_ZERO, x.dtype), 'eq', y_cmp_zero, res)
    else:
        tensor_zero = tbe.broadcast(tvm.const(Constant.CONST_ZERO, x.dtype), shape_broadcast)
        x_equal_zero = tbe.vcmp(x, tensor_zero, 'eq')
        res = tbe.vsel(x_equal_zero, y_cmp_zero, res)

    res = tbe.vadd(res, res_x_lt_zero)

    if dtype_y == "float16":
        res = tbe.cast_to(res, "float16")
    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def atan2(x1, x2, y, kernel_name="atan2"):
    """
    Algorithm: arctan2
        arctan2(y, x) = arctan(y/x)
    ----------------------------------
    Parameters:

        x1: the dict of input data x1, only support float16, float32.

        x2: the dict of input data x2, only support float16, float32.

        y: the dict of output

        kernel_name: default value is "atan2".
    ----------------------------------
    Returns:
        None
    """

    y_shape = x1.get("shape")
    x_shape = x2.get("shape")

    y_dtype = x1.get("dtype")
    x_dtype = x2.get("dtype")

    para_check.check_shape(y_shape, param_name="x1")
    para_check.check_shape(x_shape, param_name="x2")

    shape_y, shape_x, shape_max = shape_util.broadcast_shapes(
        y_shape, x_shape, param_name_input1="x1", param_name_input2="x2")

    check_list = ("float16", "float32")
    para_check.check_dtype(y_dtype, check_list, param_name="x1")
    para_check.check_dtype(x_dtype, check_list, param_name="x2")
    if y_dtype.lower() != x_dtype.lower():
        error_detail = "dtype of x1 and x2 should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "x1", "x2", error_detail)
    shape_y, shape_x = shape_util.refine_shapes_for_broadcast(shape_y, shape_x)
    input_y = tvm.placeholder(shape_y, dtype=y_dtype.lower(), name="input_y")
    input_x = tvm.placeholder(shape_x, dtype=x_dtype.lower(), name="input_x")

    res = atan2_compute(input_y, input_x, y, kernel_name)
    res = tbe.cast_to(res, x_dtype.lower())
    with tvm.target.cce():
        auto_sch = auto_schedule(res)

    config = {
        "name": kernel_name,
        "tensor_list": (input_y, input_x, res),
        "print_ir": False,
        "bool_storage_as_1bit": False
    }

    build(auto_sch, config)
