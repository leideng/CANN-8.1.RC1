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
# 'pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name
"""
sigmoid_cross_entropy_with_logits_grad
"""

from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpPatternMode


@register_operator_compute("SigmoidCrossEntropyWithLogitsGrad",
                           op_mode="dynamic", support_fusion=True, support_bfp16=True)
def sigmoid_cross_entropy_with_logits_grad_compute(
        predict,
        target,
        dout,
        gradient,
        kernel_name):
    """
    calculating sigmoid_cross_entropy_with_logits_grad_compute

    Parameters
    ----------
    predict : TVM tensor
        the output of previous layer
    target : TVM tensor
        label
    dout : TVM tensor
        last gradient
    gradient : TVM tensor
        result after compute
    Returns
    -------
    output tensor
    """
    dtype = predict.dtype
    if dtype == "float16" and tbe_platform.api_check_support(
            "tbe.dsl.vmul", "float32"):
        predict = tbe.cast_to(predict, "float32")
        target = tbe.cast_to(target, "float32")
        dout = tbe.cast_to(dout, "float32")

    # e^x
    val1 = tbe.vexp(predict)
    # 1 + e^x
    val2 = tbe.vadds(val1, tvm.const(1, dtype="float32"))

    val3 = tbe.vdiv(val1, val2)
    # -target
    val4 = tbe.vmuls(target, tvm.const(-1, dtype="float32"))

    val5 = tbe.vadd(val3, val4)

    result = tbe.vmul(val5, dout)

    if dtype == "float16":
        result = tbe.cast_to(result, dtype)
    return result


@register_operator("SigmoidCrossEntropyWithLogitsGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def sigmoid_cross_entropy_with_logits_grad(predict,
                                           target,
                                           dout,
                                           gradient,
                                           kernel_name="sigmoid_cross_entropy_with_logits_grad"):
    """
    calculating data

    Parameters
    ----------
    predict : dict
        the output of previous layer
    target : dict
        label
    dout : dict
        last gradient
    gradient : dict
        result after compute
    kernel_name : str
        kernel name, default value is "sigmoid_cross_entropy_with_logits_grad"

    Returns
    -------
    None
    """
    check_list = ("bfloat16", "float16", "float32")

    predict_dtype = predict.get("dtype")
    gradient_dtype = gradient.get("dtype").lower()
    predict_dtype_lower = predict_dtype.lower()

    para_check.check_dtype(gradient_dtype, check_list, param_name="gradient")
    para_check.check_dtype(predict_dtype_lower, check_list, param_name="predict")

    target_dtype = target.get("dtype")
    target_dtype_lower = target_dtype.lower()
    para_check.check_dtype(target_dtype_lower, check_list, param_name="target")

    dout_dtype = dout.get("dtype")
    dout_dtype_lower = dout_dtype.lower()
    para_check.check_dtype(dout_dtype_lower, check_list, param_name="dout")

    ins = classify([predict, target, dout], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_predict, _target, _dout) in ins:
        with tbe.compute():
            predict_shape, target_shape, dout_shape = shape_util.variable_shape([_predict, _target, _dout])

            predict_data_input = tvm.placeholder(
                predict_shape, name="predict_data_input", dtype=predict_dtype_lower)
            target_data_input = tvm.placeholder(
                target_shape, name="target_data_input", dtype=target_dtype_lower)
            dout_data_input = tvm.placeholder(
                dout_shape, name="dout_data_input", dtype=dout_dtype_lower)

            res = sigmoid_cross_entropy_with_logits_grad_compute(
                predict_data_input, target_data_input, dout_data_input, gradient,
                kernel_name)
            tensors.append([predict_data_input, target_data_input, dout_data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    # build
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
