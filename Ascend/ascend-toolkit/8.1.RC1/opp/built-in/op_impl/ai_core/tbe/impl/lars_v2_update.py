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
lars_v2_update
"""
import functools
import operator
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_build
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import build_config


# 'pylint: disable=too-many-arguments,unused-argument,too-many-locals
@register_operator_compute("lars_v2_update", op_mode="static", support_fusion=True)
def lars_v2_update_compute(inputs_data, hyperparam, epsilon, use_clip, out, kernel_name="lars"):
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

    weight_norm = tbe.vsqrt(weight_s)
    grad_norm = tbe.vsqrt(grad_s)

    coeff_weight_norm = tbe.vmuls(weight_norm, hyperparam)
    weight_norm_decay = tbe.vmul(weight_norm, weight_decay)
    weight_grad_norm = tbe.vadd(weight_norm_decay, grad_norm)
    norm_res = tbe.vadds(weight_grad_norm, epsilon)
    coeff = tbe.vdiv(coeff_weight_norm, norm_res)

    if use_clip:
        coeff_clip = tbe.vdiv(coeff, learning_rate)
        coff_max = tbe.vmins(coeff_clip, tvm.const(1, dtype=weight.dtype))
        clip_coff = tbe.vmaxs(coff_max, tvm.const(0, dtype=weight.dtype))
        coeff_broadcast = tbe.broadcast(clip_coff, weight.shape)
    else:
        coeff_broadcast = tbe.broadcast(coeff, weight.shape)

    weight_decay_broadcast = tbe.broadcast(weight_decay, weight.shape)
    if weight.dtype.lower() != weight_decay_broadcast.dtype.lower():
        weight = tbe.cast_to(weight, weight_decay_broadcast.dtype.lower())
    weight_weight_decay = tbe.vmul(weight, weight_decay_broadcast)
    if weight_decay_broadcast.dtype.lower() != grad.dtype.lower():
        grad = tbe.cast_to(grad, weight_weight_decay.dtype.lower())
    weight_grad = tbe.vadd(weight_weight_decay, grad)

    out = tbe.vmul(weight_grad, coeff_broadcast)

    return out


# 'pylint: disable=too-many-arguments,unused-argument,too-many-locals
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
                   kernel_name="lars_update"):
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

    check_list = ("float16", "float32")
    inputs = [weight, grad, weight_s, grad_s, weight_decay, learning_rate]

    weight_shape = weight.get("shape")
    grad_shape = grad.get("shape")
    weight_dtype = weight.get("dtype")
    grad_dtype = grad.get("dtype")
    if list(weight_shape) != list(grad_shape):
        error_manager_vector.raise_err_inputs_shape_not_equal(kernel_name, 'weight', 'grad', weight_shape, grad_shape,
                                                              weight_shape)

    if grad_dtype != weight_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, 'weight', 'grad', weight_dtype, grad_dtype)

    vdiv_support = tbe_platform.api_check_support("tbe.dsl.vdiv", "float32")
    if not vdiv_support:
        new_check_list = list(check_list)
        new_check_list.remove("float32")
        para_check.check_dtype(weight_dtype, new_check_list, param_name="weight")

    input_place_holders = []
    for i, input_val in enumerate(inputs):
        input_dtype = input_val.get("dtype").lower()
        input_shape = input_val.get("shape")
        para_check.check_shape(input_shape)
        para_check.check_dtype(input_dtype, check_list)
        shape_one_dim = (functools.reduce(operator.mul, input_shape), )
        input_place_holders.append(tvm.placeholder(shape_one_dim, name="input_data_%d" % i, dtype=input_dtype))

    res = lars_v2_update_compute(input_place_holders, hyperparam, epsilon, use_clip, out, kernel_name)

    with tvm.target.cce():
        schedule = auto_schedule(res)

    data = input_place_holders
    data.append(res)

    with build_config(dummy_placeholder=True):
        tvm.build(schedule, data, "cce", name=kernel_name)
