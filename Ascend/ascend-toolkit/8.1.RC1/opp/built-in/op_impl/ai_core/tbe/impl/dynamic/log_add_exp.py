# Copyright 2020 Huawei Technologies Co., Ltd
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
log_add_exp
"""
import math
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import OpImplMode
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_soc_common import after_v200


def isclose(valuex, valuey, rel_tol=1e-08, abs_tol=0.0):
    """
    determines whether the values of two floating-point numbers are close or equal
    """
    return math.isclose(valuex, valuey, rel_tol=rel_tol, abs_tol=abs_tol)

# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals,invalid-name


@register_operator_compute("LogAddExp", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def log_add_exp_compute(x1, x2, y, base=-1.0, scale=1.0, shift=0.0, kernel_name="log_add_exp"):
    """
    algorithm: log_add_exp
        calculating the logarithm of the sum of exponentiations of the data
    if base == -1:
        diff = -1 * abs(x1 - x2)
        y = max(x1, x2) + ln(1 + exp(diff))
    else base > 0:
        diff = shift + scale * (-1 * abs(x1 - x2))
        y = max(x1, x2) + ln(1 + exp(diff * ln(base))) / ln(base)

    Parameters
    ----------
    x1 : TVM tensor, the placeholder of x1
    x2 : TVM tensor, the placeholder of x2
    y : dict, shape and dtype of output, should be same shape and dtype as input
    base: (optional, default -1 for a value of e) the base gamma
    scale: (optional, default 1) the scale alpha
    shift: (optional, default 0) the shift beta
    kernel_name : str, kernel name, default value is "log_add_exp"

    Returns
    -------
    output tensor
    """
    _, _, shape_max = shape_util.broadcast_shapes(x1.shape,
                                                  x2.shape,
                                                  param_name_input1="x1",
                                                  param_name_input2="x2")
    input_x1 = tbe.broadcast(x1, shape_max)
    input_x2 = tbe.broadcast(x2, shape_max)
    input_max = tbe.vmax(input_x1, input_x2)

    if after_v200():
        input_min = tbe.vmin(input_x1, input_x2)
        is_input_inf1 = tbe.vcmp(input_min, float("inf"), "eq")
        is_input_inf2 = tbe.vcmp(input_max, float("-inf"), "eq")
        input_vsub_nan = tbe.vsub(input_x1, input_x2)
        input_vsub_nan = tbe.vsel(is_input_inf1, 0, input_vsub_nan)
        input_vsub = tbe.vsel(is_input_inf2, 0, input_vsub_nan)
    else:
        input_vsub = tbe.vsub(input_x1, input_x2)
    input_vsub_vabs = tbe.vabs(input_vsub)
    neg_const = tvm.const(-1.0, dtype=input_vsub.dtype)
    input_diff = tbe.vmuls(input_vsub_vabs, neg_const)

    if isclose(scale, 1.0) and isclose(shift, 0.0):
        input_diff_vadds = input_diff
    else:
        scale_const = tvm.const(scale, dtype=input_diff.dtype)
        shift_const = tvm.const(shift, dtype=input_diff.dtype)
        input_diff_vmuls = tbe.vmuls(input_diff, scale_const)
        input_diff_vadds = tbe.vadds(input_diff_vmuls, shift_const)

    if base > 0:
        log_base = math.log(base)
        base_const = tvm.const(log_base, dtype=input_diff.dtype)
        input_diff_vadds = tbe.vmuls(input_diff_vadds, base_const)

    exp_res = tbe.vexp(input_diff_vadds)

    one_const = tvm.const(1.0, dtype=exp_res.dtype)
    add_res = tbe.vadds(exp_res, one_const)

    log_res = tbe.vlog(add_res)

    if base > 0:
        base_scale = 1.0 / log_base
        log_res = tbe.vmuls(log_res, tvm.const(
            base_scale, dtype=log_res.dtype))

    res = tbe.vadd(input_max, log_res)

    return res


@register_operator("LogAddExp")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def log_add_exp(x1, x2, y, base=-1.0, scale=1.0, shift=0.0, kernel_name="log_add_exp"):
    """
    algorithm: log_add_exp
        calculating the logarithm of the sum of exponentiations of the data
    if base == -1:
        y = ln(exp(shift + scale * x1) + exp(shift + scale * x2))
    else base > 0:
        y = ln(exp(shift + scale * x1) * ln(base) + exp(shift + scale * x2) * ln(base)) / ln(base)

    Parameters
    ----------
    x1 : dict, shape and dtype of input, only support float16, float32
    x2 : dict, shape and dtype of input, only support float16, float32
    y : dict, shape and dtype of output, should be same shape and dtype as input
    base: (optional, default -1 for a value of e) the base gamma
    scale: (optional, default 1) the scale alpha
    shift: (optional, default 0) the shift beta
    kernel_name : str, kernel name, default value is "log_add_exp"

    Returns
    -------
    None
    """
    dtype_x1 = x1.get("dtype").lower()
    dtype_x2 = x2.get("dtype").lower()
    check_tuple = ("float16", "float32", "bfloat16")
    para_check.check_dtype(dtype_x1, check_tuple, param_name="x1")
    para_check.check_dtype(dtype_x2, check_tuple, param_name="x2")
    if dtype_x1 != dtype_x2:
        error_manager_vector.raise_err_inputs_dtype_not_equal(
            "log_add_exp", "input_x1", "input_x2", str(dtype_x1), str(dtype_x2))
    if base <= 0 and (not isclose(base, -1.0)):
        expect_value = "strictly positive or -1"
        real_value = "base < 0 or base notequal with -1"
        error_manager_vector.raise_err_input_value_invalid(
            kernel_name, "base", expect_value, real_value)

    ins = classify([x1, x2], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_x1, _x2) in ins:
        with tbe.compute():
            shape_x1, shape_x2 = shape_util.variable_shape([_x1, _x2])
            data_x1 = tvm.placeholder(shape_x1, name="data_x1", dtype=dtype_x1)
            data_x2 = tvm.placeholder(shape_x2, name="data_x2", dtype=dtype_x2)
            res = log_add_exp_compute(
                data_x1, data_x2, y, base, scale, shift, kernel_name)

            tensors.append((data_x1, data_x2, res))
        with tvm.target.cce():
            schedule = tbe.auto_schedule(res)
        schedules.append(schedule)

    config = {"print_ir": False, "need_build": False, "name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
