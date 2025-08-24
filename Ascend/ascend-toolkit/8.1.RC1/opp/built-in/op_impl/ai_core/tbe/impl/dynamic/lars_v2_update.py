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
dynamic lars_v2_update
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import error_manager_vector
from impl.util.util_attr_common import OpAttr
from impl.util.util_attr_common import get_attr_by_cls


class LarsV2UpdateAttrInfo:
    """
    define attr info
    """
    ATTR_HYPERPARAM = OpAttr(0, "hyperparam", "Float", 0.001)
    ATTR_EPSILON = OpAttr(1, "epsilon", "Float", 1e-5)


# 'pylint: disable=too-many-arguments,unused-argument,too-many-locals
@register_operator_compute("LarsV2Update", op_mode="dynamic", support_fusion=True)
def lars_v2_update_compute(inputs_data, hyperparam, epsilon, use_clip, out, kernel_name="lars_v2_update"):
    """
    lars_update compute

    Parameters:
    ----------
    inputs_data: list
        the placeholders of input data
    hyperparam: float
        default value is 0.001
    epsilon: float
        default value is 1e-5
    use_clip: bool
        default value is "False".
    out: dict
        output contains shape and dtype attributes.
    kernel_name : str
        kernel name, default value is "lars_update"

    Returns:
    None
    """
    weight, grad, weight_s, grad_s, weight_decay, learning_rate = inputs_data
    weight_dtype = weight.dtype
    weight_norm = tbe.vsqrt(weight_s)
    grad_norm = tbe.vsqrt(grad_s)

    hyperparam = get_attr_by_cls(hyperparam, LarsV2UpdateAttrInfo.ATTR_HYPERPARAM, weight_dtype)
    epsilon = get_attr_by_cls(epsilon, LarsV2UpdateAttrInfo.ATTR_EPSILON, weight_dtype)
    coeff_weight_norm = tbe.vmuls(weight_norm, hyperparam)
    weight_norm_decay = tbe.vmul(weight_norm, weight_decay)
    weight_grad_norm = tbe.vadd(weight_norm_decay, grad_norm)
    norm_res = tbe.vadds(weight_grad_norm, epsilon)
    vdiv_support = tbe_platform.api_check_support("te.lang.cce.vdiv", "float32")
    if not vdiv_support:
        coeff_weight_norm = tbe.cast_to(coeff_weight_norm, "float16")
        norm_res = tbe.cast_to(norm_res, "float16")
        coeff = tbe.vdiv(coeff_weight_norm, norm_res)
        coeff = tbe.cast_to(coeff, "float32")
    coeff = tbe.vdiv(coeff_weight_norm, norm_res)

    if use_clip:
        coeff_clip = tbe.vdiv(coeff, learning_rate)
        coff_max = tbe.vmins(coeff_clip, tvm.const(1, dtype=weight.dtype))
        clip_coff = tbe.vmaxs(coff_max, tvm.const(0, dtype=weight.dtype))
        coeff_broadcast = tbe.broadcast(clip_coff, weight.shape)
    else:
        coeff_broadcast = tbe.broadcast(coeff, weight.shape)

    weight_decay_broadcast = tbe.broadcast(weight_decay, weight.shape)
    weight_weight_decay = tbe.vmul(weight, weight_decay_broadcast)
    weight_grad = tbe.vadd(weight_weight_decay, grad)

    out = tbe.vmul(weight_grad, coeff_broadcast)

    return out


# 'pylint: disable=too-many-arguments,unused-argument,too-many-locals
@register_operator("LarsV2Update")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def lars_v2_update(weight,
                   grad,
                   weight_s,
                   grad_s,
                   weight_decay,
                   learning_rate,
                   out,
                   hyperparam=0.001,
                   epsilon=1e-5,
                   use_clip=False,
                   kernel_name="lars_v2_update"):
    """
    the opreator's compute
    hyper_weight_norm = hyperparam * sqrt(weight_s)
    grad_weight_norm = sqrt(grad_s) + weight_decay*sqrt(weight_s) + epsilon
    grad_weight = grad + weight_decay * weight

    if use_clip == True:
        coeff = hyper_weight_norm / grad_weight_norm
        coeff = min(coeff / learning_rate, 1)
        coeff = max(coeff, 0)
    else:
        coeff = hyper_weight_norm / grad_weight_norm

    grad_new = coeff * grad_weight
    Parameters:
    ----------
    weight: dict
        input tensor contains shape and dtype attributes.
        only support float32.
    grad: dict
        input tensor contains shape and dtype attributes.
        Must have the same dtype and shape as 'weight'.
    weight_s: dict
        input tensor contains shape and dtype attributes.
        Must have the same dtype as 'weight'.
    grad_s: dict
        input tensor contains shape and dtype attributes.
        Must have the same dtype as 'weight'.
    weight_decay: dict
        input tensor contains shape and dtype attributes.
        Must have the same dtype as 'weight'.
    learning_rate: dict
        input tensor contains shape and dtype attributes.
        Must have the same dtype as 'weight'.
    out: dict
        output tensor contains shape and dtype attributes.
        Must have the same dtype and shape  as 'weight'.
    hyperparam: float
        default value is 0.001
    epsilon: float
        default value is 1e-5
    use_clip: bool
        default value is "False".
    kernel_name : str
        kernel name, default value is "lars_update"

    Returns:
    None
    """

    check_list = ("float32")
    inputs = [weight, grad, weight_s, grad_s, weight_decay, learning_rate]

    weight_dtype = weight.get("dtype")
    grad_dtype = grad.get("dtype")
    if grad_dtype != weight_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, 'weight', 'grad', weight_dtype, grad_dtype)

    for input_val in inputs:
        input_dtype = input_val.get("dtype").lower()
        para_check.check_dtype(input_dtype, check_list)

    shape_scaler = [1]
    data_weights = tvm.placeholder(shape_scaler, name="data_weights", dtype=weight_dtype)
    data_grads = tvm.placeholder(shape_scaler, name="data_grads", dtype=weight_dtype)
    data_weight_decay = tvm.placeholder(shape_scaler, name="data_weight_decay", dtype=weight_dtype)
    data_learning_rate = tvm.placeholder(shape_scaler, name="data_lr", dtype=weight_dtype)

    ins = classify([weight, grad], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for (_weight, _grad) in ins:
        with tbe.compute():
            shape_weight, shape_grad = shape_util.variable_shape([_weight, _grad])
            data_weight = tvm.placeholder(shape_weight, name="data_weight", dtype=weight_dtype)
            data_grad = tvm.placeholder(shape_grad, name="data_grad", dtype=weight_dtype)
            inputs_place_holders = [data_weight, data_grad, data_weights, data_grads,
                                    data_weight_decay, data_learning_rate]
            res = lars_v2_update_compute(inputs_place_holders, hyperparam, epsilon, use_clip, out, kernel_name)
            tensors.append(inputs_place_holders + [res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
