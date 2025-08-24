# Copyright 2022 Huawei Technologies Co., Ltd
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
log_sigmoid
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from .log1p import log1p_compute


# 'pylint: disable=invalid-name,unused-argument
@register_operator_compute("LogSigmoid", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def log_sigmoid_compute(x, y, kernel_name='log_sigmoid'):
    """
    log-sigmoid function compute
    :param x: the shape and dtype of input tensor
    :param y: the shape and dtype of output tensor
    :param kernel_name: cce kernel name, default value is 'log_sigmoid'
    :return: value of log_sigmoid
    """
    dtype_x = x.dtype

    cloud_flag = tbe_platform.api_check_support("te.lang.cce.vexp", "float32")
    if dtype_x == "float16":
        if cloud_flag:
            x = tbe.cast_to(x, "float32")
            zeros = tbe.broadcast(tvm.const(0, "float32"), x.shape)
        else:
            zeros = tbe.broadcast(tvm.const(0, dtype_x), x.shape)
    else:
        if cloud_flag:
            zeros = tbe.broadcast(tvm.const(0, dtype_x), x.shape)
        else:
            x = tbe.cast_to(x, "float16")
            zeros = tbe.broadcast(tvm.const(0, "float16"), x.shape)

    # log_sigmoid compute(x is positive)
    tempt_muls_pos = tbe.vmuls(x, tvm.const(-1, "float32"))  # -x
    tempt_exp_pos = tbe.vexp(tempt_muls_pos)  # e^(-x)
    tempt_log_pos = log1p_compute(tempt_exp_pos, y)  # ln(1+e^(-x))
    res_pos = tbe.vmuls(tempt_log_pos, tvm.const(-1, "float32"))  # -ln(1+e^(-x))
    # log_sigmoid compute(x is negative)
    tempt_exp_neg = tbe.vexp(x)  # e^(x)
    tempt_log_neg = log1p_compute(tempt_exp_neg, y)  # ln(1+e^(x))
    res_neg = tbe.vsub(x, tempt_log_neg)  # x - ln(1+e^(x))

    res = tbe.vcmpsel(x, zeros, "le", res_neg, res_pos)

    if dtype_x == "float16" and cloud_flag:
        res = tbe.cast_to(res, "float16")
    if dtype_x == "float32" and not cloud_flag:
        res = tbe.cast_to(res, "float32")

    return res


@register_operator("LogSigmoid")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def log_sigmoid(x, y, kernel_name='log_sigmoid'):
    """
    log-sigmoid function compute
    :param x: the shape and dtype of input tensor
    :param y: the shape and dtype of output tensor
    :param kernel_name: cce kernel name, default value is 'log_sigmoid'
    :return: value of log_sigmoid
    """

    # obtain operator information
    input_dtype = x.get("dtype").lower()
    check_list = ("float16", "float32", "bfloat16")

    # obtain operator information
    para_check.check_dtype(input_dtype, check_list, param_name="x")
    para_check.check_kernel_name(kernel_name)

    ins = classify([x], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])
            data_input = tvm.placeholder(x_shape[0], name="data_input",
                                         dtype=input_dtype)
            res = log_sigmoid_compute(data_input, y, kernel_name)

            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
