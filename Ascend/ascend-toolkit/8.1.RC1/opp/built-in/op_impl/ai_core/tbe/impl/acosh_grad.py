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
acosh_grad
"""
import operator

import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=too-few-public-methods,not-use-list-comprehension
class Constant:
    """
    Constant in this class
    """
    NUM_ONE = 1
    NUM_TWO = 2
    NUM_REPEAT = 0.125

    TAYLOR_SECOND_ORDER_PARAM = 0.1666666666666666666666666666666666
    TAYLOR_THIRD_ORDER_PARAM = 0.0083333333333333333333333333333333
    TAYLOR_FOURTH_ORDER_PARAM = 0.0001984126984126984126984126984126


def _taylor_sinh_compute(input_data):
    """
    do taylor sinh compute
    Parameters:
    ----------------
    input_data: input tensor
    return: sinh result
    ----------------
    """

    # x^2 / 7!
    data_power_2 = tbe.vmul(input_data, input_data)
    data_power_res = tbe.vmuls(
        data_power_2,
        tvm.const(Constant.TAYLOR_FOURTH_ORDER_PARAM, input_data.dtype))

    # 1/5! + x^2 / 7!
    data_power_res = tbe.vadds(
        data_power_res,
        tvm.const(Constant.TAYLOR_THIRD_ORDER_PARAM, input_data.dtype))

    # 1/3! + x^2( 1/5! + x^2/7!)
    data_power_res = tbe.vmul(data_power_res, data_power_2)
    data_power_res = tbe.vadds(
        data_power_res,
        tvm.const(Constant.TAYLOR_SECOND_ORDER_PARAM, input_data.dtype))

    # 1 + x^2( 1/3! + x^2(1/5! + x^2/7!))
    data_power_res = tbe.vmul(data_power_res, data_power_2)

    data_power_res = tbe.vadds(data_power_res, tvm.const(Constant.NUM_ONE, input_data.dtype))

    # x * (1 + x^2( 1/3! + x^2(1/5! + x^2/7!)))
    data_power_res = tbe.vmul(data_power_res, input_data)
    return data_power_res


def _sinh_repeat_with_sqrt(data):
    """
    do sinh convert compute with sqrt
    Calculate f(2x) = 2f(x) * (f(x)^2 + 1)^0.5
    Parameters:
    ----------------
    data: input tensor
    return: sinh repeat result
    ----------------
    """

    data_square = tbe.vmul(data, data)
    data_square = tbe.vadds(data_square, tvm.const(Constant.NUM_ONE, data.dtype))

    data_square = tbe.vsqrt(data_square, 1)

    data_square = tbe.vmul(data_square, data)
    data_square = tbe.vmuls(data_square, tvm.const(Constant.NUM_TWO, data.dtype))

    return data_square


# 'pylint: disable=unused-argument
@register_operator_compute("acosh_grad", op_mode="static", support_fusion=True)
def acosh_grad_compute(y, dy, z, kernel_name="acos_grad"):
    """
    do acosh_grad compute
    Parameters:
    ----------------
    y: input tensor y
    dy: input tensor dy
    z: output dict
    kernel_name: cce kernel name, default value is "acosh_grad"
    return: dy * (1 / sinh(y))
    ----------------
    """

    dtype = y.dtype
    dtype_1 = dtype
    if dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vadd", "float32"):
        y = tbe.cast_to(y, "float32")
        dy = tbe.cast_to(dy, "float32")
        dtype = "float32"

    data_y = tbe.vmuls(y, tvm.const(Constant.NUM_REPEAT, dtype))
    sinh_value_0 = _taylor_sinh_compute(data_y)
    sinh_value_1 = _sinh_repeat_with_sqrt(sinh_value_0)
    sinh_value_2 = _sinh_repeat_with_sqrt(sinh_value_1)
    data_sinh = _sinh_repeat_with_sqrt(sinh_value_2)

    res = tbe.vdiv(dy, data_sinh)

    if dtype_1 == "float16":
        res = tbe.cast_to(res, "float16")
    return res


# 'pylint: disable=too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def acosh_grad(y, dy, z, kernel_name="acosh_grad"):
    """
    do element-wise acosh_grad operation between two input tensors
    Parameters:
    ----------
    y : dict of y, include shape and dtype, dtype support float16, float32
    dy : dict of dy, include shape and dtype, dtype support float16, float32
    z : dict of z
    kernel_name : cce kernel name, default value is "acosh_grad"
    -------
    """

    # get the shape and dtype for input_1,input_2
    shape_y = y.get("shape")
    shape_dy = dy.get("shape")
    dtype = y.get("dtype")
    dtype_dy = dy.get("dtype")

    para_check.check_shape(shape_y, param_name="y")
    para_check.check_shape(shape_dy, param_name="dy")
    shape_y, _ = shape_util.refine_shape_axes(shape_y, [])
    shape_dy, _ = shape_util.refine_shape_axes(shape_dy, [])

    if not operator.eq(shape_y, shape_dy):
        error_detail = "shape of y and dy should be same"
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "y", "dy", error_detail)
    shape_y, _ = shape_util.refine_shape_axes(shape_y, [])
    shape_dy, _ = shape_util.refine_shape_axes(shape_dy, [])

    # raise runtimeerror if the input paras are invalid
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype, check_list, param_name="y")
    para_check.check_dtype(dtype_dy, check_list, param_name="dy")
    dtype = dtype.lower()
    dtype_dy = dtype_dy.lower()

    if dtype != dtype_dy:
        error_detail = "dtype of y and dy should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "y", "dy", error_detail)

    data_y = tvm.placeholder(shape_y, dtype=dtype, name="data1")
    data_dy = tvm.placeholder(shape_dy, dtype=dtype_dy, name="data2")

    res = acosh_grad_compute(data_y, data_dy, z, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": (data_y, data_dy, res)}
    tbe.cce_build_code(sch, config)
