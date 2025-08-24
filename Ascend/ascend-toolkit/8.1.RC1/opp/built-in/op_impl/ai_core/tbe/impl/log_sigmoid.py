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
log_sigmoid
"""
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=unused-argument,too-many-locals
@register_operator_compute("log_sigmoid", op_mode="static", support_fusion=True)
def log_sigmoid_compute(x, y, kernel_name='log_sigmoid'):
    """
    log-sigmoid function compute
    :param x: the shape and dtype of input tensor
    :param y: the shape and dtype of output tensor
    :param kernel_name: cce kernel name, default value is 'log_sigmoid'
    :return: value of log_sigmoid
    """
    shape_x = shape_util.shape_to_list(x.shape)
    dtype_x = x.dtype

    cloud_flag = tbe_platform.api_check_support("tbe.dsl.vexp", "float32")
    if dtype_x == "float16":
        if cloud_flag:
            x = tbe.cast_to(x, "float32")
            zeros = tbe.broadcast(tvm.const(0, "float32"), shape_x)
        else:
            zeros = tbe.broadcast(tvm.const(0, dtype_x), shape_x)
    else:
        if cloud_flag:
            zeros = tbe.broadcast(tvm.const(0, dtype_x), shape_x)
        else:
            x = tbe.cast_to(x, "float16")
            zeros = tbe.broadcast(tvm.const(0, "float16"), shape_x)

    # log_sigmoid compute(x is positive)
    tempt1 = tbe.vmuls(x, tvm.const(-1, "float32"))         # -x
    tempt2 = tbe.vexp(tempt1)                               # e^(-x)
    tempt3 = tbe.vadds(tempt2, tvm.const(1, "float32"))     # 1+e^(-x)
    tempt4 = tbe.vlog(tempt3)                               # ln(1+e^(-x))
    res_pos = tbe.vmuls(tempt4, tvm.const(-1, "float32"))   # -ln(1+e^(-x))
    # log_sigmoid compute(x is negative)
    tempt5 = tbe.vexp(x)                                    # e^(x)
    tempt6 = tbe.vadds(tempt5, tvm.const(1, "float32"))     # 1+e^(x)
    tempt7 = tbe.vlog(tempt6)                               # ln(1+e^(x))
    res_neg = tbe.vsub(x, tempt7)                           # x - ln(1+e^(x))

    res = tbe.vcmpsel(x, zeros, "le", res_neg, res_pos)

    if dtype_x == "float16" and cloud_flag:
        res = tbe.cast_to(res, "float16")
    if dtype_x == "float32" and not cloud_flag:
        res = tbe.cast_to(res, "float32")

    return res


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
    x_shape = x.get("shape")
    x_dtype = x.get("dtype").lower()
    check_list = ("float16", "float32")

    # operator check
    para_check.check_shape_rule(x_shape)
    para_check.check_shape_size(x_shape)
    para_check.check_dtype_rule(x_dtype, check_list)
    para_check.check_kernel_name(kernel_name)

    # tensor placeholder
    data_x = tvm.placeholder(x_shape, name='data_x', dtype=x_dtype)

    # log sigmoid compute function
    res = log_sigmoid_compute(data_x, y, kernel_name)

    # auto schedule
    with tvm.target.cce():
        schedule = auto_schedule(res)

    # compile configuration
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": (data_x, res)}
    build(schedule, config)
