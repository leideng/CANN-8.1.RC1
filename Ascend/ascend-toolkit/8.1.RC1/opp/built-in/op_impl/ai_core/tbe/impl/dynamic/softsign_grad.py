# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
softsign_grad dynamic
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,unused-argument
@register_operator_compute("SoftsignGrad", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def softsign_grad_compute(gradients, features, output, kernel_name="softsign_grad"):
    """
    calculate the backpropagation of relu operation
    output = gradients / (1 + abs(features)) ** 2.

    Parameters
    ----------
    gradients: TVM tensor
        input tensor of grad
    features: TVM tensor
        input tensor of relu output
    output: dict
        output dict of relu grad
    kernel_name: str
        cce kernel name, default value is "softsign_grad"

    Returns
    -------
    res: TVM tensor
        the result of softsign_grad_compute
    """
    dtype = gradients.dtype
    estimate = dtype != "float32"
    if estimate:
        gradients = tbe.cast_to(gradients, "float32")
        features = tbe.cast_to(features, "float32")

    features_abs = tbe.vabs(features)
    features_add = tbe.vadds(features_abs, 1)
    features_mul = tbe.vmul(features_add, features_add)
    result = tbe.vdiv(gradients, features_mul)
    if estimate:
        result = tbe.cast_to(result, "float16")

    return result


@register_operator("SoftsignGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def softsign_grad(gradients, features, output, kernel_name="softsign_grad"):
    """
    calculate the backpropagation of relu operation
    output = gradients / (1 + abs(features)) ** 2.
    support dtype:bfloat16,float16,float32,int32,int8,uint8.
    input not support broadcast.

    Parameters
    ----------
    gradients: dict
        the backpropagated gradients to the corresponding relu operation
    features: dict
        the features passed as output of relu operation
    output: dict
        the output of relu back propagation
    kernel_name: str
        cce kernel name, default value is "softsign_grad"

    Returns
    -------
    None
    """
    g_dtype = gradients.get("dtype").lower()
    x_dtype = features.get("dtype").lower()
    check_list = ("bfloat16", "float16", "float32")
    para_check.check_dtype(g_dtype, check_list, param_name="gradients")
    para_check.check_dtype(x_dtype, check_list, param_name="features")
    para_check.check_elewise_shape_range([gradients, features], support_broadcast=True)
    if g_dtype != x_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "gradients", "features", g_dtype,
                                                              x_dtype)
    ins = classify([gradients, features], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_g, _x) in ins:
        with tbe.compute():
            g_shape, x_shape = shape_util.variable_shape([_g, _x])
            tensor_g = tvm.placeholder(g_shape, g_dtype, "tensor_g")
            tensor_x = tvm.placeholder(x_shape, x_dtype, "tensor_x")
            res = softsign_grad_compute(tensor_g, tensor_x, output, kernel_name)
            tensors.append((tensor_g, tensor_x, res))
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
