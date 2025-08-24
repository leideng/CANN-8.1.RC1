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
dynamic softplus
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant.
    """
    SCALAR_ONE = 1
    NEG_LN_2 = - 0.69314718055994530941723212145818
    NEG_ONE = -1


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals
# 'pylint: disable=invalid-name
@register_operator_compute("Softplus", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def softplus_compute(input_x, y, kernel_name="softplus"):
    """
    Compute for softplus.
    The compute: "log(exp(x) + 1)".

    Parameters
    ----------
    input_x: TVM tensor
        data of input.
        source data type, support "bfloat16", "float16", "float32".
    y: TVM tensor
        data of output.
    kernel_name: str
        kernel name, default value is "softplus".

    Returns
    -------
    res: TVM tensor
        output data and has the same type as `features`.
    """
    dtype = input_x.dtype
    if dtype == "float16" and \
            tbe_platform.api_check_support("te.lang.cce.vexp",
                                           "float32"):
        input_x = tbe.cast_to(input_x, "float32")

    positive_part = tbe.vmaxs(input_x, tvm.const(0, dtype="float32"))
    negative_part = tbe.vmins(input_x, tvm.const(0, dtype="float32"))
    # calculate positive part softplus
    pos_to_neg = tbe.vmuls(positive_part, tvm.const(Constant.NEG_ONE, dtype="float32"))
    # dtype is float32 and tbe_platform not support float32
    if dtype == "float32" and not tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        positive_part = tbe.cast_to(positive_part, "float16")
        pos_to_neg = tbe.cast_to(pos_to_neg, "float16")
        negative_part = tbe.cast_to(negative_part, "float16")
        exp_pos = tbe.vexp(pos_to_neg)
        exp_add_one = tbe.vadds(exp_pos, Constant.SCALAR_ONE)
        log_pos = tbe.vlog(exp_add_one)
        res_positive = tbe.vadd(log_pos, positive_part)
        # calculate positive part softplus
        exp_neg = tbe.vexp(negative_part)
        add_one = tbe.vadds(exp_neg, Constant.SCALAR_ONE)
        res_negative = tbe.vlog(add_one)
    else:
        exp_pos = tbe.vexp(pos_to_neg)
        exp_add_one = tbe.vadds(exp_pos, Constant.SCALAR_ONE)
        log_pos = tbe.vlog(exp_add_one)
        res_positive = tbe.vadd(log_pos, positive_part)
        # calculate positive part softplus
        exp_neg = tbe.vexp(negative_part)
        add_one = tbe.vadds(exp_neg, Constant.SCALAR_ONE)
        res_negative = tbe.vlog(add_one)
    res_positive = tbe.cast_to(res_positive, "float32")
    res_negative = tbe.cast_to(res_negative, "float32")
    res_tmp = tbe.vadd(res_positive, res_negative)
    res = tbe.vadds(res_tmp, Constant.NEG_LN_2)

    if dtype == "float16":
        res = tbe.cast_to(res, "float16")

    return res


@register_operator("Softplus")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def softplus(x, y, kernel_name="softplus"):
    """
    Compute for softplus.

    Parameters
    ----------
    x: dict
        data of input.
        source data type, support "bfloat16", "float16", "float32".
    y: dict
        data of output.
    kernel_name: str
        kernel name, default value is "softplus".

    Returns
    -------
    None
    """
    x_dtype = x.get("dtype").lower()
    check_list = ("bfloat16", "float16", "float32")
    para_check.check_dtype(x_dtype, check_list, param_name="x")

    ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (input_x,) in ins:
        # op compute
        with tbe.compute():
            x_shape = shape_util.variable_shape([input_x])
            input_data = tvm.placeholder(x_shape[0], name="input_data", dtype=x_dtype)

            res = softplus_compute(input_data, y, kernel_name)
            tensors.append([input_data, res])
        # target auto schedule
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        # append schedule 2 schedules
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
