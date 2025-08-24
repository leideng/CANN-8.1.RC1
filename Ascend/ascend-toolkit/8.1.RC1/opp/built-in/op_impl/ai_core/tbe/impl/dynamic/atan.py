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
atan

Op_description :
Computes the trignometric inverse tangent of x element-wise

# atan(
#   x,
#   y,
#   kernel_name="cce_atan")

Supportive_dtype_format :
['bfloat16', 'float16', 'float32']
['ALL']

Constraint :
[1] All : shape size limit is 2147483648.

"""
from impl.util import util_compute
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    CONST_POS_ONE = 1.0
    CONST_ITERTOR = 6
    CONST_ITERTOR2 = 4
    TAN_PI_BY_EIGHT = 0.4142135623730950
    TAYLOR = (1.0, -1.0 / 3, 1.0 / 5, -1.0 / 7, 1.0 / 9, -1.0 / 11, 1.0 / 13)


def _do_taylor(input_data):
    """
    Algorithm:
        if x > 0 and x < tan(pi/8):
            atan(x) = x - x^3/3 + x^5/5 - x^7/7 ...
        elif x > tan(pi/8) and x < tan(pi/4):
            atan(x) = atan(y) + atan((x-y)/(1+xy))

    ----------------------------------
    Parameters:

    input_data: Input data

    ----------------------------------
    Returns:

        A Tensor of atan(x).

    """

    const_pi_by_eight = 0.39269908169872415480783042290994
    shape_input = input_data.shape
    dtype_input = input_data.dtype

    tensor_offset = tbe.broadcast(tvm.const(Constant.TAN_PI_BY_EIGHT, dtype_input), shape_input)
    denominator_data = tbe.vmuls(input_data, Constant.TAN_PI_BY_EIGHT)
    denominator_data = tbe.vadds(denominator_data, Constant.CONST_POS_ONE)
    molecule = tbe.vsub(input_data, tensor_offset)
    data = tbe.vdiv(molecule, denominator_data)
    data = tbe.vabs(data)

    square_data = tbe.vmul(data, data)
    res = tbe.broadcast(tvm.const(Constant.TAYLOR[Constant.CONST_ITERTOR], dtype_input), shape_input)
    for i in reversed(range(Constant.CONST_ITERTOR)):
        res = tbe.vmul(res, square_data)
        res = tbe.vadds(res, Constant.TAYLOR[i])
    res = tbe.vmul(res, data)
    res = tbe.vadds(res, const_pi_by_eight)

    square_data = tbe.vmul(input_data, input_data)
    res2 = tbe.broadcast(tvm.const(Constant.TAYLOR[Constant.CONST_ITERTOR2], dtype_input), shape_input)
    for i in reversed(range(Constant.CONST_ITERTOR2)):
        res2 = tbe.vmul(res2, square_data)
        res2 = tbe.vadds(res2, Constant.TAYLOR[i])
    res2 = tbe.vmul(res2, input_data)

    res = tbe.vmin(res, res2)

    return res


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
@register_operator_compute("Atan", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def atan_compute(x, y, kernel_name="atan"):
    """
    Algorithm: atan

    ----------------------------------
    Parameters:

    x: Input data

    y : the dict of output

    kernel_name: cce kernel name, default value is "atan"

    ----------------------------------
    Returns:

        A Tensor of atan(x).

    """

    const_pi_by_four = 0.78539816339744830961566084581988
    dtype = x.dtype
    shape = x.shape

    if dtype == "float16" and tbe_platform.api_check_support("te.lang.cce.vadd", "float32"):
        x = tbe.cast_to(x, "float32")

    # when x's value is too large the first caculator of _do_taylor will be overflow, and we expect
    #   the relative accuracy to be 6.1e-0.5
    # So we can use 'tan(pi/2 - 6.1e-0.5)' to get '16394', used to replace large value in x which grater than '16394'.
    max_input_value = 16394
    min_input_value = -max_input_value
    x = tbe.vmaxs(tbe.vmins(x, max_input_value), min_input_value)

    abs_data = tbe.vabs(x)

    tensor_one = tbe.broadcast(tvm.const(Constant.CONST_POS_ONE, x.dtype), shape)

    abs_data_sub_one = tbe.vsub(abs_data, tensor_one)
    abs_data_add_one = tbe.vadd(abs_data, tensor_one)
    abs_data2 = tbe.vdiv(abs_data_sub_one, abs_data_add_one)
    abs_data2 = tbe.vabs(abs_data2)

    # calucate data less than one
    res = _do_taylor(abs_data)
    # calucate data more than one
    res_mt_one = _do_taylor(abs_data2)
    res_mt_one = tbe.vadds(res_mt_one, const_pi_by_four)

    res = tbe.vmin(res, res_mt_one)

    sign_mask = util_compute.sign(x)
    res = tbe.vmul(res, sign_mask)

    if dtype == "float16":
        res = tbe.cast_to(res, "float16")
    return res


@register_operator("Atan")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def atan(x, y, kernel_name="atan"):
    """
    Algorithm: atan

    ----------------------------------
    Parameters:

    x: the dict of input data, only support bfloat16, float16, float32.

    y: the dict of output

    kernel_name: cce kernel name, default value is "atan".

    ----------------------------------
    Returns:

        None

    """
    dtype = x.get("dtype").lower()
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(dtype, check_list, param_name="x")
    ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (ins_x,) in ins:
        with tbe.compute():
            shape_x = shape_util.variable_shape([ins_x])[0]
            input_data = tvm.placeholder(shape_x, name="input_data", dtype=dtype)
            res = atan_compute(input_data, y, kernel_name)
            res = tbe.cast_to(res, dtype)
            tensors.append([input_data, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "print_ir": False, "tensor_list": tensors}
    tbe.build(schedules, config)
