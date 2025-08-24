# Copyright 2021 Huawei Technologies Co.,  Ltd
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

Op_description :
Computes arctangent of y/x element-wise, respecting signs of the arguments

# atan2(
#   x1,
#   x2,
#   y,
#   kernel_name="atan2")

Supportive_dtype_format :
['float16', 'float32']
['ALL']

Constraint :
[1] All : 'y' and 'x' must have the same type and shape.
[2] All : shape size limit is 2147483648.

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


# 'pylint: disable=too-few-public-methods
class Constant(object):
    """
    The class for constant
    """
    CONST_POS_ONE = 1.0
    CONST_ITERTOR = 6
    CONST_ITERTOR2 = 4
    CONST_ZERO = 0
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

    const_pi_by_eight = 0.39269908169872415480783042290994
    tan_pi_by_eight = 0.4142135623730950
    tan_pi_by_eight_na = -0.4142135623730950
    denominator_data = tbe.vmuls(input_data, tan_pi_by_eight)
    denominator_data = tbe.vadds(denominator_data, Constant.CONST_POS_ONE)
    molecule = tbe.vadds(input_data, tan_pi_by_eight_na)
    data = tbe.vdiv(molecule, denominator_data)
    data = tbe.vabs(data)

    square_data = tbe.vmul(data, data)
    res = tbe.vmuls(square_data, Constant.TAYLOR[Constant.CONST_ITERTOR])
    res = tbe.vadds(res, Constant.TAYLOR[Constant.CONST_ITERTOR - 1])
    for i in reversed(range(Constant.CONST_ITERTOR - 1)):
        res = tbe.vmul(res, square_data)
        res = tbe.vadds(res, Constant.TAYLOR[i])
    res = tbe.vmul(res, data)
    res = tbe.vadds(res, const_pi_by_eight)

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

    const_pi_by_four = 0.78539816339744830961566084581988
    shape = input_x.shape
    dtype = input_x.dtype
    if dtype == "float16" and \
            tbe_platform.api_check_support("te.lang.cce.vadd", "float32"):
        input_x = tbe.cast_to(input_x, "float32")

    # when x's value is too large the first caculator of _do_taylor will be overflow.
    # when epsilon is 0.0000001, the approximate value of `tan(pi/2 - 0.0000001)` is 10000000(fp32)
    # the maximum value of fp16(65504.0) is used when the data type is fp16.
    if input_x.dtype == "float32":
        max_input_value = 10000000.0
    else:
        max_input_value = 65504.0
    min_input_value = -max_input_value
    input_x = tbe.vmaxs(tbe.vmins(input_x, max_input_value), min_input_value)

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
    res_mt_one = tbe.vadds(res_mt_one, const_pi_by_four)

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

    const_na_one = -1.0
    shape_input = data_y.shape
    dtype_input = data_y.dtype

    tensor_one = tbe.broadcast(tvm.const(Constant.CONST_POS_ONE, dtype_input), shape_input)
    tensor_zero = tbe.broadcast(tvm.const(Constant.CONST_ZERO, dtype_input), shape_input)
    tensor_na_one = tbe.vmuls(tensor_one, tvm.const(const_na_one, dtype_input))

    y_me_zero = tbe.vcmpsel(data_y, tensor_zero, 'ge', tensor_one, tensor_na_one)
    x_lt_zero_y_mask = tbe.vcmpsel(data_x, tensor_zero, 'lt', y_me_zero, tensor_zero)

    y_cmp_zero = tbe.vcmpsel(data_y, tensor_zero, 'ge', tensor_one, tensor_na_one)

    mask = (x_lt_zero_y_mask, y_cmp_zero)
    return mask


# 'pylint: disable=locally-disabled,invalid-name,too-many-statements
def corner_case_post_process(y, x, res):
    """
    atan2(+-0, -0) returns +-pi;
    atan2(+-0, 0) returns +-0;
    atan2(+-0, x) returns +-pi for x < 0;
    atan2(+-0, x) returns +-0 for x > 0;
    atan2(y, +-0) returns -pi/2 for y < 0;
    atan2(y, +-0) returns pi/2 for y > 0;
    atan2(+-y, -inf) returns +-pi for finite y > 0;
    atan2(+-y, inf) returns +-0 for finite y > 0;
    atan2(+-inf, x) returns +-pi/2 for finite x;
    atan2(+-inf, -inf) returns +-3pi/4;
    atan2(+-inf, inf) returns +-pi/4;
    """
    xshape = x.shape
    xdtype = x.dtype.lower()
    tensor_zero = tbe.broadcast(tvm.const(0.0, xdtype), xshape)
    tensor_one = tbe.broadcast(tvm.const(1.0, xdtype), xshape)
    tensor_inf = tbe.broadcast(tvm.const(float("inf"), xdtype), xshape)
    tensor_ne_inf = tbe.broadcast(tvm.const(float("-inf"), xdtype), xshape)

    const_pi = tvm.const(3.1415926535897932384626433832795, xdtype)
    ne_const_pi = tvm.const(-3.1415926535897932384626433832795, xdtype)
    const_pi_by_two = tvm.const(1.5707963267948966192313216916398, xdtype)
    ne_const_pi_by_two = tvm.const(-1.5707963267948966192313216916398, xdtype)
    const_pi_by_four = tvm.const(0.78539816339744830961566084581988, xdtype)
    ne_const_pi_by_four = tvm.const(-0.78539816339744830961566084581988, xdtype)
    const_pi_by_three_quarters = tvm.const(2.356194490192345, xdtype)
    ne_const_pi_by_three_quarters = tvm.const(-2.356194490192345, xdtype)

    y_zero_temp = tbe.vcmp(y, tensor_zero, "eq")
    y_zero_index = tbe.vsel(y_zero_temp, tensor_one, tensor_zero)
    y_signbit_one = tbe.vsignbit(y)
    y_negative_zero = tbe.vmul(y_signbit_one, y_zero_index)
    y_positive_zero = tbe.vsub(y_zero_index, y_negative_zero)

    x_signbit_one = tbe.vsignbit(x)
    x_signbit_zero = tbe.vsub(tensor_one, x_signbit_one)

    y_pos_zero_x_sign_one = tbe.vmul(y_positive_zero, x_signbit_one)
    res_temp = tbe.vcmp(y_pos_zero_x_sign_one, tensor_one, "eq")
    res = tbe.vsel(res_temp, const_pi, res)

    y_pos_zero_x_sign_zero = tbe.vmul(y_positive_zero, x_signbit_zero)
    res_temp = tbe.vcmp(y_pos_zero_x_sign_zero, tensor_one, "eq")
    res = tbe.vsel(res_temp, tensor_zero, res)

    y_neg_zero_x_sign_one = tbe.vmul(y_negative_zero, x_signbit_one)
    res_temp = tbe.vcmp(y_neg_zero_x_sign_one, tensor_one, "eq")
    res = tbe.vsel(res_temp, ne_const_pi, res)

    y_neg_zero_x_sign_zero = tbe.vmul(y_negative_zero, x_signbit_zero)
    res_temp = tbe.vcmp(y_neg_zero_x_sign_zero, tensor_one, "eq")
    res = tbe.vsel(res_temp, tvm.const(-0.0, xdtype), res)

    neg_y_mask = tbe.vcmp(y, tensor_zero, "lt")
    neg_y = tbe.vsel(neg_y_mask, tensor_one, tensor_zero)
    neg_y_mask_2 = tbe.vcmp(y, tensor_ne_inf, "gt")
    neg_y_gt_inf = tbe.vsel(neg_y_mask_2, neg_y, tensor_zero)
    x_inf = tbe.vcmp(x, tensor_inf, "eq")
    neg_y_gt_inf_x_inf = tbe.vsel(x_inf, neg_y_gt_inf, tensor_zero)
    res_temp = tbe.vcmp(neg_y_gt_inf_x_inf, tensor_one, "eq")
    res = tbe.vsel(res_temp, tvm.const(-0.0, xdtype), res)

    y_inf_mask = tbe.vcmp(y, tensor_inf, "eq")
    y_inf = tbe.vsel(y_inf_mask, tensor_one, tensor_zero)
    y_ne_inf_mask = tbe.vcmp(y, tensor_ne_inf, "eq")
    y_ne_inf = tbe.vsel(y_ne_inf_mask, tensor_one, tensor_zero)

    y_inf_x_inf = tbe.vsel(x_inf, y_inf, tensor_zero)
    res_temp = tbe.vcmp(y_inf_x_inf, tensor_one, "eq")
    res = tbe.vsel(res_temp, const_pi_by_four, res)

    y_ne_inf_x_inf = tbe.vsel(x_inf, y_ne_inf, tensor_zero)
    res_temp = tbe.vcmp(y_ne_inf_x_inf, tensor_one, "eq")
    res = tbe.vsel(res_temp, ne_const_pi_by_four, res)

    x_ne_inf = tbe.vcmp(x, tensor_ne_inf, "eq")
    y_inf_x_ne_inf = tbe.vsel(x_ne_inf, y_inf, tensor_zero)
    res_temp = tbe.vcmp(y_inf_x_ne_inf, tensor_one, "eq")
    res = tbe.vsel(res_temp, const_pi_by_three_quarters, res)

    y_ne_inf_x_ne_inf = tbe.vsel(x_ne_inf, y_ne_inf, tensor_zero)
    res_temp = tbe.vcmp(y_ne_inf_x_ne_inf, tensor_one, "eq")
    res = tbe.vsel(res_temp, ne_const_pi_by_three_quarters, res)

    x_abs = tbe.vabs(x)
    x_lt_inf = tbe.vcmp(x_abs, tensor_inf, "lt")
    y_inf_x_lt_inf = tbe.vsel(x_lt_inf, y_inf, tensor_zero)
    res_temp = tbe.vcmp(y_inf_x_lt_inf, tensor_one, "eq")
    res = tbe.vsel(res_temp, const_pi_by_two, res)

    y_ne_inf_x_lt_inf = tbe.vsel(x_lt_inf, y_ne_inf, tensor_zero)
    res_temp = tbe.vcmp(y_ne_inf_x_lt_inf, tensor_one, "eq")
    res = tbe.vsel(res_temp, ne_const_pi_by_two, res)

    return res


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
@register_operator_compute("Atan2", op_mode="dynamic", support_fusion=True, support_bfp16=True)
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

    const_pi = 3.1415926535897932384626433832795
    const_pi_by_two = 1.5707963267948966192313216916398
    const_one = 1
    dtype_y = y.dtype

    shape_y, shape_x, shape_broadcast = shape_util.broadcast_shapes(y.shape,
                                                                    x.shape,
                                                                    param_name_input1="x1",
                                                                    param_name_input2="x2")
    y = tbe.broadcast(y, shape_broadcast)
    x = tbe.broadcast(x, shape_broadcast)

    if dtype_y == "float16" and \
            tbe_platform.api_check_support("te.lang.cce.vadd", "float32"):
        y = tbe.cast_to(y, "float32")
        x = tbe.cast_to(x, "float32")

    mask = _init_atan2_mask(y, x)

    # caculate the atan(y/x) when x > 0
    res = tbe.vdiv(y, x)
    res = _atan_compute(res)

    y_cmp_zero = tbe.vmuls(mask[const_one], tvm.const(const_pi_by_two, y.dtype))
    res_x_lt_zero = tbe.vmuls(mask[Constant.CONST_ZERO], tvm.const(const_pi, y.dtype))

    if x.dtype == res.dtype and \
            tbe_platform.api_check_support("te.lang.cce.vcmpsel", x.dtype):
        res = tbe.vcmpsel(x, tvm.const(Constant.CONST_ZERO, x.dtype), 'eq', y_cmp_zero, res)
    else:
        tensor_zero = tbe.broadcast(tvm.const(Constant.CONST_ZERO, x.dtype), shape_broadcast)
        x_equal_zero = tbe.vcmp(x, tensor_zero, 'eq')
        res = tbe.vsel(x_equal_zero, y_cmp_zero, res)

    res = tbe.vadd(res, res_x_lt_zero)

    if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ("Ascend910B", "Ascend910_93", "Ascend310B"):
        res = corner_case_post_process(y, x, res)
        is_y_not_nan = tbe.vcmp(y, y, "eq", "bit")
        res = tbe.vsel(is_y_not_nan, res, y)
        is_x_not_nan = tbe.vcmp(x, x, "eq", "bit")
        res = tbe.vsel(is_x_not_nan, res, x)

    if dtype_y == "float16":
        res = tbe.cast_to(res, "float16")
    return res


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
@register_operator("Atan2")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def atan2(x1, x2, y, kernel_name="atan2"):
    """
    Algorithm: arctan2
        arctan2(y, x) = arctan(y/x)
    ----------------------------------
    Parameters:

        x1: the dict of input data x1, only support float16, float32, bfloat16.

        x2: the dict of input data x2, only support float16, float32, bfloat16.

        y: the dict of output

        kernel_name: default value is "atan2".
    ----------------------------------
    Returns:
        None
    """

    y_dtype = x1.get("dtype").lower()
    x_dtype = x2.get("dtype").lower()
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(y_dtype, check_list, param_name="x1")
    para_check.check_dtype(x_dtype, check_list, param_name="x2")
    ins = classify([x1, x2], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (ins_x1, ins_x2) in ins:
        with tbe.compute():
            shape_y, shape_x = shape_util.variable_shape([ins_x1, ins_x2])
            input_y = tvm.placeholder(shape_y, dtype=y_dtype, name="input_y")
            input_x = tvm.placeholder(shape_x, dtype=x_dtype, name="input_x")
            res = atan2_compute(input_y, input_x, y, kernel_name)
            res = tbe.cast_to(res, x_dtype)
            tensors.append([input_y, input_x, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name,
              "tensor_list": tensors,
              "print_ir": False,
              "build_args": {"status_check": False},
              "bool_storage_as_1bit": False}
    tbe.build(schedules, config)
