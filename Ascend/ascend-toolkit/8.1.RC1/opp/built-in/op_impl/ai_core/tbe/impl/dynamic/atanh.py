# Copyright 2021 hHuawei Technologies Co., Ltd
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
atanh

Op_description :
Computes inverse hyperbolic tangent of x element-wise

# atanh(
#   x,
#   y,
#   kernel_name="atanh_cce")

Supportive_dtype_format :
['float16', 'bfloat16', 'float32']
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
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpTbeImplMode
from impl.util.util_soc_common import after_v200


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
@register_operator_compute("Atanh", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def atanh_compute(x, y, kernel_name="atanh"):
    """
    Algrithm : atanh(x) = 0.5 * log((1 + x) / (1 - x)) if abs(x) < 1

    Parameters
    ----------
    x: the placeholder of data input

    y : the dict of output

    kernel_name : cce kernel name

    Returns
    -------
    res : result of atanh
    """

    inp_dtype = x.dtype
    shape = x.shape

    if inp_dtype == "float16" and tbe_platform.api_check_support("te.lang.cce.vadd", "float32") \
            and tbe_platform.api_check_support("te.lang.cce.vlog", "float32"):
        x = tbe.cast_to(x, "float32")

    if after_v200():
        data_res = _compute_v2(x, shape)
    else:
        data_res = _compute(x, shape)

    if inp_dtype == "float16":
        data_res = tbe.cast_to(data_res, "float16")
    else:
        data_res = tbe.cast_to(data_res, "float32")

    return data_res


def _compute(data_input, shape):
    """
    Algrithm: atanh(x) = 0.5*log((1+x)/(1-x))

    Parameters
    ----------
    data_input: the placeholder of data input

    shape: the shape of data_input

    Returns
    -------
    data_res :  return of atanh
    """

    # const value
    const_half = 0.5
    const_one = 1
    const_neg_one = -1
    data_1_sum_x = tbe.vadds(data_input, tvm.const(const_one, data_input.dtype))
    data_sub_x = tbe.vmuls(data_input, tvm.const(const_neg_one, data_input.dtype))
    data_1_sub_x = tbe.vadds(data_sub_x, tvm.const(const_one, data_input.dtype))
    data_x_mul = tbe.vdiv(data_1_sum_x, data_1_sub_x)
    data_x_log = tbe.vlog(data_x_mul, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
    data_res = tbe.vmuls(data_x_log, tvm.const(const_half, data_input.dtype))

    return data_res


def _compute_v2(data_input, shape):
    """
    Algrithm: atanh(x) apply Segmented Function Implementation

    Parameters
    ----------
    data_input: the placeholder of data input

    shape: the shape of data_input

    Returns
    -------
    data_res :  return of atanh
    """

    # const value
    const_pos_one = 1
    const_two = 2
    const_neg_one = -1
    const_pi = 3.4028235e34
    const_zero = 0

    input_dtype = data_input.dtype

    data_input_abs = tbe.vabs(data_input)
    data_input_neg = tbe.vmuls(data_input_abs, tvm.const(const_neg_one, input_dtype))
    data_input_add = tbe.vadds(data_input_neg, tvm.const(const_pos_one, input_dtype))
    data_input_div = tbe.vdiv(data_input_abs, data_input_add)
    data_input_mul = tbe.vmuls(data_input_div, tvm.const(const_two, input_dtype))
    data_input_sum = tbe.vadds(data_input_mul, tvm.const(const_pos_one, input_dtype))

    data_x_add = tbe.vadds(data_input_sum, tvm.const(const_neg_one, input_dtype))
    data_x_min = tbe.vmins(data_x_add, tvm.const(const_pi, input_dtype))
    data_x_log = tbe.vlog(data_input_sum, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
    data_x_mul = tbe.vmul(data_x_log, data_input_div)
    data_x_div = tbe.vdiv(data_x_mul, data_x_min)

    is_equal_one = tbe.vcmp(data_input_sum, const_pos_one, "eq", "bool")
    data_vsel = tbe.vsel(is_equal_one, data_input_abs, data_x_div)

    data_vsel_neg = tbe.vmuls(data_vsel, tvm.const(const_neg_one, input_dtype))
    is_greater_than_zero = tbe.vcmp(data_input, const_zero, "ge", "bool")
    data_res = tbe.vsel(is_greater_than_zero, data_vsel, data_vsel_neg)

    return data_res


@register_operator("Atanh")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def atanh(x, y, kernel_name="atanh"):
    """
    Algrithm: atanh(x) = atanh

    Parameters
    ----------
    Algorithm: atanh

    Parameters:

    x: the dict of input data, only support float16, bfloat16, float32.

    y: the dict of output

    kernel_name: cce kernel name, default value is "atanh".

    Returns
    -------
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
            input_data = tvm.placeholder(shape_x, dtype, "input_data")
            res = atanh_compute(input_data, y, kernel_name)
            tensors.append([input_data, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors, "bool-stoage_as_1bit": False}
    tbe.build(schedules, config)
