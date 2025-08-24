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
acosh

Op_description :
Computes inverse hyperbolic cosine of x element-wise

# acosh(
#   input_data,
#   output_res,
#   kernel_name="cce_acosh")

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
from impl.util import util_soc_common


# 'pylint: disable=too-few-public-methods,not-use-list-comprehension
class Constant:
    """
    Constant in this class
    """
    CONST_ONE = 1.0
    CONST_NEG_ONE = -1.0
    CONST_LOG_TWO = 0.6931472
    CONST_SQRT_FP16_MAX = 255.9
    CONST_SQRT_FP32_MAX = 1.8446743e19
    CONST_COMPARE_VALUE_MAX = 3.4028235e34
    CONST_COMPARE_VALUE_MIN = 1e-45
    CONST_LOG_ADD_VALUE = 6.93147180559945286227e-01


def acosh_compute_v1(input_data):
    data = input_data
    input_dtype = data.dtype
    in_type = data.dtype
    has_improve_precision = False

    if input_dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vadd", "float32"):
        data = tbe.cast_to(data, "float32")
        input_dtype = "float32"
        has_improve_precision = True
    
    neg_one = tvm.const(Constant.CONST_NEG_ONE, input_dtype)
    one = tvm.const(Constant.CONST_ONE, input_dtype)
    data_t = tbe.vadds(data, neg_one)

    data_t_two = tbe.vadd(data_t, data_t)
    data_r = tbe.vmul(data_t, data_t)
    data_r = tbe.vadd(data_r, data_t_two)
    data_r = tbe.vsqrt(data_r, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
    data_r = tbe.vadd(data_t, data_r)

    data_u = tbe.vadds(data_r, one)
    data_s = tbe.vadds(data_u, neg_one)

    min_value = tvm.const(Constant.CONST_COMPARE_VALUE_MIN, input_dtype)
    max_value = tvm.const(Constant.CONST_COMPARE_VALUE_MAX, input_dtype)
    data_s = tbe.vmaxs(data_s, min_value)
    data_s = tbe.vmins(data_s, max_value)

    res = tbe.vlog(data_u, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
    res = tbe.vmul(res, data_r)
    res = tbe.vdiv(res, data_s)
    data_s1 = tbe.vlog(data, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
    data_s1 = tbe.vadds(data_s1, tvm.const(Constant.CONST_LOG_ADD_VALUE, input_dtype))
    res = tbe.vmin(res, data_s1)
    if has_improve_precision:
        res = tbe.cast_to(res, in_type)
    return res


def acosh_compute_revise(input_data, ori_res):
    if not util_soc_common.after_v200():
        return ori_res

    data = input_data
    input_dtype = data.dtype.lower()

    vlog_x = tbe.vlog(data)
    vlog_2 = tvm.const(Constant.CONST_LOG_TWO, dtype=input_dtype)
    overflow_res = tbe.vadds(vlog_x, vlog_2)

    neg_one = tvm.const(Constant.CONST_NEG_ONE, input_dtype)
    nan_val = tvm.const(float('nan'), input_dtype)
    if input_dtype == 'float16':
        sqrt_max_val = tvm.const(Constant.CONST_SQRT_FP16_MAX, dtype="float16")
    else:
        sqrt_max_val = tvm.const(Constant.CONST_SQRT_FP32_MAX, dtype="float32")

    act_res = tbe.vcmpsel(input_data, neg_one, 'lt', nan_val, ori_res)
    act_res = tbe.vcmpsel(input_data, sqrt_max_val, 'ge', overflow_res, act_res)

    return act_res


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
@register_operator_compute("Acosh", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def acosh_compute(input_data, output_res, kernel_name="acosh"):
    """
    do element-wise acosh compute
    f(x) = log(x+sqrt(x^2-1)),  for all inputs

    Parameters:
    ----------
    input_data: the placeholder of data input

    output_res : the dict of output

    kernel_name : cce kernel name, default value is "acosh"

    Returns : A Tensor. Has the same type as input_data.
    -------
    """
    if util_soc_common.after_v200():
        return acosh_compute_v1(input_data)

    data = input_data
    const_neg_one = -1.0

    input_dtype = data.dtype.lower()
    if input_dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vadd", "float32"):
        data = tbe.cast_to(data, "float32")

    res = tbe.vmul(data, data)
    res = tbe.vadds(res, tvm.const(const_neg_one, data.dtype))
    res = tbe.vsqrt(res, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
    res = tbe.vadd(res, data)
    if res.dtype == "float32" and not tbe_platform.api_check_support("tbe.dsl.vlog", "float32"):
        res = tbe.cast_to(res, "float16")
    res = tbe.vlog(res, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)

    res = tbe.cast_to(res, input_dtype)

    res = acosh_compute_revise(input_data, res)

    return res


@register_operator("Acosh")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def acosh(input_data, output_res, kernel_name="acosh"):
    """
    calculating data's acosh,y= log(x+sqrt(x^(2)-1))

    Parameters
    ----------
    input_data: the dict of input, only support float16, bfloat16, float32

    output_res : the dict of output

    kernel_name : cce kernel name, default value is "cce_acosh"

    Returns
    -------
    None

    """
    input_dtype = input_data.get("dtype").lower()
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(input_dtype, check_list, param_name="input_data")
    schedules, tensors = [], []
    ins = classify([input_data], OpPatternMode.ELEWISE)
    for (_input_data,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_input_data])[0]
            data_input = tvm.placeholder(x_shape, dtype=input_dtype, name="data_input")
            res = acosh_compute(data_input, output_res, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "print_ir": False, "tensor_list": tensors, "bool_storage_as_1bit": False}
    tbe.build(schedules, config)
