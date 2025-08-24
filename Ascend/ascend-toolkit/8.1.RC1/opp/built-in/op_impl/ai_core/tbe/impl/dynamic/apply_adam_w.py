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
apply_adam_w
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,invalid-name,too-many-arguments
# 'pylint: disable=too-many-locals,unused-argument
def _gt_compute(grad, maximize):
    """
    gt compute
    """
    input_dtype = grad.dtype
    if maximize:
        symbol_one = tvm.const(-1, dtype=input_dtype)
    else:
        symbol_one = tvm.const(1, dtype=input_dtype)
    gt = tbe.vmuls(grad, symbol_one)

    return gt


def _output_m_compute(m, beta1, gt):
    """
    output 1, m_out compute.
    `m_out = m * beta1 - (beta1 + (-1)) * gt`
    """
    input_dtype = m.dtype
    shape_m = shape_util.shape_to_list(m.shape)

    s_neg_one = tvm.const(-1, dtype=input_dtype)
    beta1_broad = tbe.broadcast(beta1, shape_m)

    # `formula; m * beta1`
    vmul_m_beta1 = tbe.vmul(m, beta1_broad)

    # `formula; beta1 + (-1)`
    vsub_beta1_1 = tbe.vadds(beta1_broad, s_neg_one)

    # `formula; (beta1 + (-1)) * gt`
    vmul_beta1_1_gt = tbe.vmul(vsub_beta1_1, gt)

    # `formula; m * beta1 - (beta1 + (-1)) * gt`
    m_out = tbe.vsub(vmul_m_beta1, vmul_beta1_1_gt)

    return m_out


def _output_v_compute(v, beta2, gt):
    """
    output 2, v_out compute.
    `v_out = v * beta2 - (beta2 + (-1)) * gt * gt`
    """
    input_dtype = v.dtype
    shape_v = shape_util.shape_to_list(v.shape)

    s_neg_one = tvm.const(-1, dtype=input_dtype)
    beta2_broad = tbe.broadcast(beta2, shape_v)

    # `formula; v * beta2`
    vmul_v_beta2 = tbe.vmul(v, beta2_broad)

    # `formula; beta2 + (-1)`
    vsub_beta2_1 = tbe.vadds(beta2_broad, s_neg_one)

    # `formula; (beta2 + (-1)) * gt * gt`
    vmul_beta2_1_gt = tbe.vmul(vsub_beta2_1, gt)
    vmul_beta2_1_gt_gt = tbe.vmul(vmul_beta2_1_gt, gt)

    # `formula; v * beta2 - (beta2 + (-1)) * gt * gt`
    v_out = tbe.vsub(vmul_v_beta2, vmul_beta2_1_gt_gt)

    return v_out


def _var_t_compute(var, lr_broad, weight_decay):
    """
    var_t compute.
    `var_t = var * (1 + (-lr * weight_decay))`
    """
    input_dtype = var.dtype
    shape_var = shape_util.shape_to_list(var.shape)

    s_one = tvm.const(1, dtype=input_dtype)
    s_neg_one = tvm.const(-1, dtype=input_dtype)
    weight_decay_broad = tbe.broadcast(weight_decay, shape_var)

    # `formula; -lr * weight_decay`
    vmul_lr_broad_neg = tbe.vmuls(lr_broad, s_neg_one)
    vmul_lr_weight_decay_neg = tbe.vmul(vmul_lr_broad_neg, weight_decay_broad)

    # `formula; 1 + (-lr * weight_decay)`
    var_t_right = tbe.vadds(vmul_lr_weight_decay_neg, s_one)

    # `formula; var * (1 + (-lr * weight_decay))`
    var_t = tbe.vmul(var, var_t_right)

    return var_t


@register_operator_compute("ApplyAdamW", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def apply_adam_w_compute(var, m, v, beta1_power, beta2_power, lr, weight_decay, beta1, beta2, epsilon, grad,
                         max_grad_norm, var_out, m_out, v_out, amsgrad=False, maximize=False,
                         kernel_name="apply_adam_w"):
    """
    apply_adam_w compute.

    if maximize:
        gt = -grad
    else:
        gt = grad
    m_out = m * beta1 - (beta1 + (-1)) * gt
    v_out = v * beta2 - (beta2 + (-1)) * gt * gt

    var_t = var * (1 + (-lr * weight_decay))
    beta1_power_out = beta1_power * beta1
    beta2_power_out = beta2_power * beta2
    if amsgrad:
        max_grad_norm_out = max(max_grad_norm, v_out)
        denom = sqrt(-max_grad_norm_out / (beta2_power_out + (-1))) + epsilon
    else:
        denom = sqrt(-v_out / (beta2_power_out + (-1))) + epsilon
    var_out = var_t + ((lr / (beta1_power_out + (-1))) * (m_out / denom))

    Parameters
    ----------
    var: TVM tensor
         the input tensor of var
    m: TVM tensor
         the input tensor of m
    v: TVM tensor
         the input tensor of v
    beta1_power: TVM tensor
         the input tensor of beta1_power
    beta2_power: TVM tensor
         the input tensor of beta2_power
    lr: TVM tensor
         the input tensor of lr
    weight_decay: TVM tensor
         the input tensor of weight_decay
    beta1: TVM tensor
         the input tensor of beta1
    beta2: TVM tensor
         the input tensor of beta2
    epsilon: TVM tensor
         the input tensor of epsilon
    grad: TVM tensor
         the input tensor of grad
    max_grad_norm: TVM tensor
         the input tensor of max_grad_norm
    var_out: dict
        output tensor contains shape and dtype attributes.
        Data type and shape are same as var.
    m_out: dict
        output tensor contains shape and dtype attributes.
        Data type and shape are same as var.
    v_out: dict
        output tensor contains shape and dtype attributes.
        Data type and shape are same as var.
    amsgrad: bool
        default value is "False".
    maximize: bool
        default value is "False".
    kernel_name : str
        kernel name, default value is "apply_adam_w"

    Returns
    -------
    output tensor
    """
    shape_var = shape_util.shape_to_list(var.shape)
    input_dtype = var.dtype
    s_neg_one = tvm.const(-1, dtype=input_dtype)
    epsilon_broad = tbe.broadcast(epsilon, shape_var)

    gt = _gt_compute(grad, maximize)
    data_m_out = _output_m_compute(m, beta1, gt)
    data_v_out = _output_v_compute(v, beta2, gt)

    lr_broad = tbe.broadcast(lr, shape_var)
    var_t = _var_t_compute(var, lr_broad, weight_decay)

    # `formula; beta1_power_out = beta1_power * beta1`
    # `formula; beta2_power_out = beta2_power * beta2`
    data_beta1_power_out = tbe.vmul(beta1_power, beta1)
    data_beta2_power_out = tbe.vmul(beta2_power, beta2)

    # `formula; beta2_power_out + (-1)`
    beta2_power_out_sub_1 = tbe.vadds(data_beta2_power_out, s_neg_one)
    beta2_power_out_sub_1_broad = tbe.broadcast(beta2_power_out_sub_1, shape_var)
    # `formula; beta1_power_out + (-1)`
    beta1_power_out_sub_1 = tbe.vadds(data_beta1_power_out, s_neg_one)

    data_max_grad_norm_out = None
    if amsgrad:
        # `formula; max_grad_norm_out = max(max_grad_norm, v_out)`
        data_max_grad_norm_out = tbe.vmax(max_grad_norm, data_v_out)
        # `formula; -1 * max_grad_norm_out`
        v_t_left = tbe.vmuls(data_max_grad_norm_out, s_neg_one)
    else:
        # `formula; -1 * v_out`
        v_t_left = tbe.vmuls(data_v_out, s_neg_one)

    # `formula; denom = sqrt(v_t_left / (beta2_power_out + (-1))) + epsilon`
    v_t = tbe.vdiv(v_t_left, beta2_power_out_sub_1_broad)
    sqrt_v_t = tbe.vsqrt(v_t)
    denom = tbe.vadd(sqrt_v_t, epsilon_broad)

    # `formula; lr / (beta1_power_out + (-1))`
    step_size = tbe.vdiv(lr, beta1_power_out_sub_1)
    step_size_broad = tbe.broadcast(step_size, shape_var)

    # `formula; m_out / denom`
    m_out_div_demo = tbe.vdiv(data_m_out, denom)

    # `formula; (lr / (beta1_power_out + (-1))) * (m_out / denom)`
    m_out_div_demo_mul_step_size = tbe.vmul(step_size_broad, m_out_div_demo)

    # `formula; var_t + ((lr / (beta1_power_out + (-1))) * (m_out / denom))`
    data_var_out = tbe.vadd(var_t, m_out_div_demo_mul_step_size)

    res = [data_var_out, data_m_out, data_v_out]

    return res


@register_operator("ApplyAdamW")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def apply_adam_w(var, m, v, beta1_power, beta2_power, lr, weight_decay, beta1, beta2, epsilon, grad, max_grad_norm,
                 var_out, m_out, v_out, amsgrad=False, maximize=False, kernel_name="apply_adam_w"):
    """
    algorithm: apply_adam_w.

    Parameters:
    ----------
    var: dict
        input tensor contains shape and dtype attributes.
        only support float16, float32, bfloat16.
    m: dict
        input tensor contains shape and dtype attributes.
        Data type and shape are same as var.
    v: dict
        input tensor contains shape and dtype attributes.
        Data type and shape are same as var.
    beta1_power: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as var. Shape is [1].
    beta2_power: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as var. Shape is [1].
    lr: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as var. Shape is [1].
    weight_decay: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as var. Shape is [1].
    beta1: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as var. Shape is [1].
    beta2: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as var. Shape is [1].
    epsilon: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as var. Shape is [1].
    grad: dict
        input tensor contains shape and dtype attributes.
        Data type and shape are same as var.
    max_grad_norm: dict
        input tensor contains shape and dtype attributes.
        Data type and shape are same as var.
    var_out: dict
        output tensor contains shape and dtype attributes.
        Data type and shape are same as var.
    m_out: dict
        output tensor contains shape and dtype attributes.
        Data type and shape are same as var.
    v_out: dict
        output tensor contains shape and dtype attributes.
        Data type and shape are same as var.
    amsgrad: bool
        default value is "False".
    maximize: bool
        default value is "False".
    kernel_name : str
        kernel name, default value is "apply_adam_w"

    Returns
    -------
    None
    """
    compute_type = var.get("dtype").lower()
    shape_scalar = [1]
    data_beta1_power = tvm.placeholder(shape_scalar, name="data_beta1_power", dtype=compute_type)
    data_beta2_power = tvm.placeholder(shape_scalar, name="data_beta2_power", dtype=compute_type)
    data_lr = tvm.placeholder(shape_scalar, name="data_lr", dtype=compute_type)
    data_weight_decay = tvm.placeholder(shape_scalar, name="data_weight_decay", dtype=compute_type)
    data_beta1 = tvm.placeholder(shape_scalar, name="data_beta1", dtype=compute_type)
    data_beta2 = tvm.placeholder(shape_scalar, name="data_beta2", dtype=compute_type)
    data_epsilon = tvm.placeholder(shape_scalar, name="data_epsilon", dtype=compute_type)

    inputs_scalar_data = [data_beta1_power, data_beta2_power, data_lr, data_weight_decay,
                          data_beta1, data_beta2, data_epsilon]

    schedules, tensors = [], []

    inputs_dict = [var, m, v, grad]
    if amsgrad:
        inputs_dict = inputs_dict + [max_grad_norm]

    ins = classify(inputs_dict, OpPatternMode.ELEWISE)

    for inputs_list in ins:
        with tbe.compute():
            shape_list = shape_util.variable_shape(inputs_list)
            data_var = tvm.placeholder(shape_list[0], name="data_var", dtype=compute_type)
            data_m = tvm.placeholder(shape_list[1], name="data_m", dtype=compute_type)
            data_v = tvm.placeholder(shape_list[2], name="data_v", dtype=compute_type)
            data_grad = tvm.placeholder(shape_list[3], name="data_grad", dtype=compute_type)
            data_max_grad_norm = None
            if amsgrad:
                data_max_grad_norm = tvm.placeholder(shape_list[4], name="data_max_grad_norm", dtype=compute_type)

            inputs_data = [data_var, data_m, data_v] + inputs_scalar_data + [data_grad]
            if amsgrad:
                inputs_data = inputs_data + [data_max_grad_norm]

            res = apply_adam_w_compute(data_var, data_m, data_v, data_beta1_power, data_beta2_power, data_lr,
                                       data_weight_decay,  data_beta1, data_beta2, data_epsilon, data_grad,
                                       data_max_grad_norm, var_out, m_out, v_out, amsgrad, maximize, kernel_name)

            tensors.append(inputs_data + list(res))

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
