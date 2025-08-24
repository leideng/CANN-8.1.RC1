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
bessel_i0e

Op_description :
Computes the Bessel i0e function of `x` element-wise

# bessel_i0e(
#   x,
#   y,
#   kernel_name="bessel_i0e")

Supportive_dtype_format :
['float16', 'float32']
['ALL']

Constraint :
[1] All : shape size limit is 2147483648.
"""
import te.lang.cce as tbe
from impl.util.platform_adapter import build
from impl.util.platform_adapter import auto_schedule
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=too-few-public-methods, not-use-list-comprehension
class Constant:
    """
    The class for constant
    """
    # const value
    ITR_BEFORE = (1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.0360768, 0.0045813)
    ITR_AFTER = (0.39894228, 0.01328592, 0.00225319, -0.00157565, 0.00916281,
                -0.02057706, 0.02635537, -0.01647633, 0.00392377)
    LEN_BEFORE = 7
    LEN_AFTER = 9
    CONST_LIMIT = 15.0 / 4


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name,too-many-locals,
@register_operator_compute("bessel_i0e", op_mode="static", support_fusion=True)
def bessel_i0e_compute(x, y, kernel_name="bessel_i0e"):
    """
    Algrithm:
    I0 = 1 + ( (z/2) / (1!) )^2 + ((z/2)^2 / (2!))^2 + ... + ((z/2)^n / (n!)) ^2
    I0e = I0 / exp(x)

    t = x / 3.75
    I0(x) = e^-|x|(1 + 3.5156229t^2 + 3.0899424t^4 + 1.2067492t^6 + 0.2659732t^8
            + 0.0360768t^10 + 0.0045813t^12)), |x| <= 3.75
    I0(x) = (1 / sqrt(|x|))*(0.39894228 + 0.01328592t^-1 + 0.00225319t^-2 + -0.00157565t^-3
            + 0.00916281t^-4 + -0.02057706t^-5 + 0.02635537t^-6 + -0.01647633t^-7
            + 0.00392377t^-8), |x| >= 3.75

    Parameters
    ----------
    x: the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name, default value is "bessel_i0e"

    Returns
    -------
    A tensor. Has the same type as x.

    """

    shape_input = x.shape
    dtype_input = x.dtype

    # chose the type of data in begin
    if dtype_input == "float16" and tbe_platform.api_check_support("tbe.dsl.vadd", "float32"):
        x = tbe.cast_to(x, "float32")
    abs_data = tbe.vabs(x)

    # compute bessel_i0e for data in (-3.75, 3.75)
    broad_const_limit = tbe.broadcast(tvm.const(Constant.CONST_LIMIT, x.dtype), shape_input)
    before_abs_data = tbe.vmin(abs_data, broad_const_limit)
    data = tbe.vdiv(before_abs_data, broad_const_limit)
    square_data = tbe.vmul(data, data)

    before_res = tbe.vmuls(square_data, tvm.const(Constant.ITR_BEFORE[Constant.LEN_BEFORE - 1]))
    before_res = tbe.vadds(before_res, Constant.ITR_BEFORE[Constant.LEN_BEFORE - 2])
    for index in reversed(range(Constant.LEN_BEFORE - 2)):
        before_res = tbe.vmul(before_res, square_data)
        before_res = tbe.vadds(before_res, Constant.ITR_BEFORE[index])

    exp_data = tbe.vexp(before_abs_data)
    before_res = tbe.vdiv(before_res, exp_data)

    # compute bessel_i0e for data in other domain
    data = tbe.vdiv(broad_const_limit, abs_data)

    after_res = tbe.vmuls(data, tvm.const(Constant.ITR_AFTER[Constant.LEN_AFTER - 1]))
    after_res = tbe.vadds(after_res, Constant.ITR_AFTER[Constant.LEN_AFTER - 2])
    for index in reversed(range(Constant.LEN_AFTER - 2)):
        after_res = tbe.vmul(after_res, data)
        after_res = tbe.vadds(after_res, Constant.ITR_AFTER[index])

    sqrt_data = tbe.vsqrt(abs_data, 1)

    after_res = tbe.vdiv(after_res, sqrt_data)
    after_res = tbe.vmin(before_res, after_res)

    # chose the type of data in end
    if dtype_input == "float16":
        after_res = tbe.cast_to(after_res, "float16")

    return after_res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def bessel_i0e(x, y, kernel_name="bessel_i0e"):
    """
    Computes the Bessel i0e function of x element-wise.

    Parameters
    ----------
    x: the dict of input, only support float16, float32

    y : the dict of output

    kernel_name : cce kernel name, default value is "bessel_i0e"

    Returns
    -------
    None
    """

    shape_input = x.get("shape")
    dtype_input = x.get("dtype")

    para_check.check_shape(shape_input, param_name="x")
    shape_input, _ = shape_util.refine_shape_axes(shape_input, [])

    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_input, check_list, param_name="x")

    input_dtype = dtype_input.lower()
    data = tvm.placeholder(shape_input, dtype=input_dtype, name="data_input")

    res = bessel_i0e_compute(data, y, kernel_name)

    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {"name": kernel_name,
              "print_ir": False,
              "tensor_list": (data, res)}
    build(sch, config)
