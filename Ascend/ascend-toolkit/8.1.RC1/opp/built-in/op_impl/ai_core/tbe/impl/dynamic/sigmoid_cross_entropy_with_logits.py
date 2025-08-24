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
sigmoid_cross_entropy_with_logits
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpImplMode


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals
@register_operator_compute("SigmoidCrossEntropyWithLogits", op_mode="dynamic", support_fusion=True)
def sigmoid_cross_entropy_with_logits_compute(predict, target, loss, kernel_name="sigmoid_cross_entropy_with_logits"):
    """
    calculating data
          z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
        = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
        = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
        = (1 - z) * x + log(1 + exp(-x))
        = x - x * z + log(1 + exp(-x))

        For x < 0, to avoid overflow in exp(-x), we reformulate the above
          x - x * z + log(1 + exp(-x))
        = log(exp(x)) - x * z + log(1 + exp(-x))
        = - x * z + log(1 + exp(x))

        max(x, 0) - x * z + log(1 + exp(-abs(x)))
    Parameters
    ----------
    predict : TVM tensor
        the placeholder of predict
    target : TVM tensor
        the placeholder of target
    loss : dict
        dict of loss, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "sigmoid_cross_entropy_with_logits"

    Returns
    -------
    output tensor
    """
    predict_dtype = predict.dtype
    target_dtype = target.dtype
    if predict_dtype in ("float16", "bfloat16") and tbe_platform.api_check_support("tbe.dsl.vsub", "float32"):
        predict = tbe.cast_to(predict, "float32")
    if target_dtype in ("float16", "bfloat16") and tbe_platform.api_check_support("tbe.dsl.vmul", "float32"):
        target = tbe.cast_to(target, "float32")

    dtype_predict = predict.dtype

    const_zero = tvm.const(0, dtype=dtype_predict)
    max_predict_zero = tbe.vmaxs(predict, const_zero)

    abs_predict = tbe.vabs(predict)
    const_zero_broadcast = tbe.broadcast(const_zero, predict.shape)
    reverse_abs_predict = tbe.vsub(const_zero_broadcast, abs_predict)

    if dtype_predict == "float32" and \
            not tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        reverse_abs_predict = tbe.cast_to(reverse_abs_predict, "float16")

    vexp_predict = tbe.vexp(reverse_abs_predict)

    if dtype_predict == "float32" and \
            not tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        vexp_predict = tbe.cast_to(vexp_predict, "float32")

    const_one = tvm.const(1, dtype=dtype_predict)
    vadds_res = tbe.vadds(vexp_predict, const_one)

    if dtype_predict == "float32" and \
            not tbe_platform.api_check_support("te.lang.cce.vlog", "float32"):
        vadds_res = tbe.cast_to(vadds_res, "float16")

    vlog_res = tbe.vlog(vadds_res, OpImplMode.HIGH_PRECISION)

    if dtype_predict == "float32" and \
            not tbe_platform.api_check_support("te.lang.cce.vlog", "float32"):
        vlog_res = tbe.cast_to(vlog_res, "float32")

    vmul_res = tbe.vmul(predict, target)
    res = tbe.vsub(vlog_res, vmul_res)
    loss = tbe.vadd(res, max_predict_zero)

    if predict_dtype == "float16":
        loss = tbe.cast_to(loss, "float16")
    if predict_dtype == "bfloat16":
        loss = tbe.round(loss, "bfloat16")

    return loss


@register_operator("SigmoidCrossEntropyWithLogits")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def sigmoid_cross_entropy_with_logits(predict, target, loss, kernel_name="sigmoid_cross_entropy_with_logits"):
    """
    calculating data:
        calculating sigmoid cross entropy given logits
        `predict` and `target` must have the same type and shape.

    Parameters
    ----------
    predict : dict
        shape and dtype of predict
    target : dict
        shape and dtype of target
    loss : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "sigmoid_cross_entropy_with_logits"

    Returns
    -------
    None
    """

    dtype_predict = predict.get("dtype")
    input_dtype_predict = dtype_predict.lower()

    dtype_target = target.get("dtype")
    input_dtype_target = dtype_target.lower()

    check_list = ("bfloat16", "float16", "float32")
    para_check.check_dtype(input_dtype_predict, check_list, param_name="predict")
    para_check.check_dtype(input_dtype_target, check_list, param_name="target")
    ins = classify([predict, target], OpPatternMode.ELEWISE)

    schedules, tensors = [], []
    for (x1, x2) in ins:
        with tbe.compute():
            shape_predict, shape_target = shape_util.variable_shape([x1, x2])
            data_predict = tvm.placeholder(shape_predict,
                                           name="data_predict",
                                           dtype=input_dtype_predict)
            data_target = tvm.placeholder(shape_target,
                                          name="data_target",
                                          dtype=input_dtype_target)
            loss = sigmoid_cross_entropy_with_logits_compute(data_predict,
                                                             data_target,
                                                             loss,
                                                             kernel_name)

            tensors.append([data_predict, data_target, loss])
        with tvm.target.cce():
            sch = tbe.auto_schedule(loss)
        schedules.append(sch)
    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
