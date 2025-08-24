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
log_sigmoid_grad
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_platform
from impl.util import util_soc_common


# 'pylint: disable=too-many-arguments,too-many-locals
# 'pylint: disable=pointless-string-statement,no-else-return,unused-argument,invalid-name
def log_sigmoid_grad_compute_v2(grads, features, dtype_grads, shape_list):
    """
    log_sigmoid backward compute

    Parameters:
    -----------
    grads: the shape and dtype of tensor from previous layer
    features: the shape and dtype of input
    dtype_grads: the dtype of grads tensor
    shape_list: the broadcast shape of tensor inputs

    Returns: gradient of log_sigmoid
    -----------
    """
    grads = tbe.cast_to(grads, "float32")
    features = tbe.cast_to(features, "float32")

    # `exp(-abs(in))`
    abs_in = tbe.vabs(features)
    neg_in = tbe.vmuls(abs_in, tvm.const(-1, "float32"))
    exp_in = tbe.vexp(neg_in)

    # divide by zero protection
    z_addsone = tbe.vadds(exp_in, tvm.const(1, "float32"))
    z_div = tbe.vdiv(exp_in, z_addsone)

    # select 1/0 by compare features with 0
    ones = tbe.broadcast(tvm.const(1, "float32"), shape_list[2])
    z_sub = tbe.vsub(ones, z_div)
    zeros = tbe.broadcast(tvm.const(0, "float32"), shape_list[2])
    mask = tbe.vcmp(features, zeros, "lt", mode="bit")
    res = tbe.vsel(mask, z_sub, z_div)
    res = tbe.vmul(res, grads)

    if dtype_grads == "float16":
        res = tbe.cast_to(res, "float16")

    return res


# 'pylint: disable=too-many-arguments,too-many-locals
# 'pylint: disable=pointless-string-statement,no-else-return,unused-argument,invalid-name
@register_operator_compute("LogSigmoidGrad", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def log_sigmoid_grad_compute(grads, features, backprops, kernel_name='log_sigmoid_grad'):
    """
    log-sigmoid backward compute
    :param grads: the shape and dtype of tensor from previous layer
    :param features: the shape and dtype of input
    :param backprops: the shape and dtype of output
    :param kernel_name: cce kernel name, default value is 'log_sigmoid_grad'
    :return: gradient of log_sigmoid
    """
    # input tensor broadcast
    dtype_grads = grads.dtype.lower()
    features_dtype = features.dtype.lower()
    shape_list = shape_util.broadcast_shapes(grads.shape, features.shape)
    grads = tbe.broadcast(grads, shape_list[2])
    features = tbe.broadcast(features, shape_list[2])

    if util_soc_common.after_v200():
        return log_sigmoid_grad_compute_v2(grads, features, dtype_grads, shape_list)

    cloud_flag = tbe_platform.api_check_support("te.lang.cce.vexp", "float32")
    if dtype_grads == "float16":
        if cloud_flag:
            grads = tbe.cast_to(grads, "float32")
            features = tbe.cast_to(features, "float32")
            zeros = tbe.broadcast(tvm.const(0, "float32"), shape_list[2])
        else:
            zeros = tbe.broadcast(tvm.const(0, dtype_grads), shape_list[2])
    else:
        if cloud_flag:
            zeros = tbe.broadcast(tvm.const(0, dtype_grads), shape_list[2])
        else:
            grads = tbe.cast_to(grads, "float16")
            features = tbe.cast_to(features, "float16")
            zeros = tbe.broadcast(tvm.const(0, "float16"), shape_list[2])

    # log_sigmoid_backward compute
    tmp_exp = tbe.vexp(features)  # e^x
    tmp_add = tbe.vadds(tmp_exp, tvm.const(1, "float32"))  # 1+e^x
    tmp_log = tbe.vlog(tmp_add)  # ln(1+e^x)
    tmp_muls = tbe.vmuls(tmp_log, tvm.const(-1, "float32"))  # -ln(1+e^x)
    tmp_result = tbe.vexp(tmp_muls)  # e^(-ln(1+e^x))

    res = tbe.vmul(tmp_result, grads)

    if dtype_grads == "float16" and cloud_flag:
        res = tbe.cast_to(res, "float16")
    if dtype_grads == "float32" and not cloud_flag:
        res = tbe.cast_to(res, "float32")

    return res


# 'pylint: disable=unused-argument
@register_operator("LogSigmoidGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def log_sigmoid_grad(grads, features, backprops, kernel_name='log_sigmoid_grad'):
    """
    log-sigmoid backward compute
    :param grads: the shape and dtype of tensor from previous layer
    :param features: the shape and dtype of input
    :param backprops: the shape and dtype of output
    :param kernel_name: cce kernel name, default value is 'log_sigmoid_grad'
    :return: gradient of log_sigmoid
    """
    # obtain operator information
    dtype_grads = grads.get("dtype").lower()
    dtype_features = features.get("dtype").lower()
    check_list = ("float16", "float32", "bfloat16")

    # operator check
    para_check.check_dtype_rule(dtype_grads, check_list)
    para_check.check_dtype_rule(dtype_features, check_list)
    para_check.check_kernel_name(kernel_name)

    """
    operator compute, invoke log_sigmoid_grad_compute
    """
    ins = classify([grads, features], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (input_grads, input_features) in ins:
        with tbe.compute():
            shape_grads, shape_features = shape_util.variable_shape([input_grads, input_features])
            data_grads = tvm.placeholder(shape_grads, name="data_grads", dtype=dtype_grads)
            data_features = tvm.placeholder(shape_features, name="data_features", dtype=dtype_features)
            res = log_sigmoid_grad_compute(data_grads, data_features, backprops, kernel_name)
            tensors.append([data_grads, data_features, res])

        """
        auto schedule
        """
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # compile configuration
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
