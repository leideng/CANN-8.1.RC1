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
apply_adam_d
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpTbeImplMode
from impl.util.util_compute import only_static_support


# 'pylint: disable=invalid-name,too-many-arguments,too-many-locals
# 'pylint: disable=unused-argument
def _output_m_compute(m, beta1_broad, grad):
    """
    _output_m_compute
    """
    input_dtype = m.dtype

    sneg_one = tvm.const(-1, dtype=input_dtype)

    # `formula; beta1 -1`
    vsub_beta1_1 = tbe.vadds(beta1_broad, sneg_one)

    # `formula; m - grad`
    vsub_m_grad = tbe.vsub(m, grad)

    # `formula; (beta1 - 1) * (m - grad)`
    vmul_m = tbe.vmul(vsub_beta1_1, vsub_m_grad)

    # `formula; m_t = m + (beta1 - 1) * (m - grad)`
    m_t = tbe.vadd(m, vmul_m)

    return m_t


def _output_v_compute(v, beta2, grad):
    """_output_v_compute
    do compute v_t = v + (1 - beta2)*(grad*grad -v)
    """
    input_dtype = v.dtype
    shape_m_grad = shape_util.shape_to_list(v.shape)
    sneg_one = tvm.const(-1, dtype=input_dtype)

    # `formula; broadcast beta2 to vector`
    beta2_broad = tbe.broadcast(beta2, shape_m_grad)

    # `formula; beta2 - 1`
    vsub_beta2_1 = tbe.vadds(beta2_broad, sneg_one)

    # `formula; grad * grad`
    vmul_grad_grad = tbe.vmul(grad, grad)

    # `formula; (v - grad*grad)`
    vsub_v_grad = tbe.vsub(v, vmul_grad_grad)

    # `formula; (beta2 -1) * (v - grad * grad)`
    vmul_grad = tbe.vmul(vsub_beta2_1, vsub_v_grad)

    # `formula; v_t = v + (beta2 - 1) * (v - grad * grad)`
    v_t = tbe.vadd(v, vmul_grad)

    return v_t


def _inner_eps_add_sqrt_vt_compute(epsilon, v_t):
    """
    (epsilon + sqrt(v_t) )
    """
    # `formula; sqrt(v_t)`
    sqrt_vt = tbe.vsqrt(v_t, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)

    # `formula; broadcast epsilon  to vector`
    compute_shape = shape_util.shape_to_list(v_t.shape)
    epsilon_broad = tbe.broadcast(epsilon, compute_shape)

    # `formula; epsilon + sqrt(v_t)`
    v_add_sqrt_v = tbe.vadd(sqrt_vt, epsilon_broad)

    return v_add_sqrt_v


def _inner_lr_compute(lr, beta2_power, beta1_power, compute_shape):
    """
    _inner_lr_compute
    #lr_t = learning_rate * (sqrt(1-beta2_power)) / (1 - beta1_power)
    """
    input_dtype = lr.dtype
    s_one = tvm.const(1, dtype=input_dtype)
    s_neg_one = tvm.const(-1, dtype=input_dtype)

    # `formula; (1 - beta2_power)`
    v_neg_beta2_power = tbe.vmuls(beta2_power, s_neg_one)
    v_add_beta2_power = tbe.vadds(v_neg_beta2_power, s_one)

    # `formula; sqrt(1 - beta2_power)`
    v_sqrt_beta2_power = tbe.vsqrt(v_add_beta2_power, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)

    # `formula; (1 - beta1_power)`
    v_neg_beta1_power = tbe.vmuls(beta1_power, s_neg_one)
    v_add_beta1_power = tbe.vadds(v_neg_beta1_power, s_one)

    # `formula; learning_rate * (sqrt(1-beta2_power)`
    res = tbe.vmul(lr, v_sqrt_beta2_power)

    # `formula; learning_rate*(sqrt(1-beta2_power))/(1-beta1_power)`
    res = tbe.vdiv(res, v_add_beta1_power)
    return tbe.broadcast(res, compute_shape)


def _output_var_t_compute_use_nesterov(var, lr_t, m_t, beta1_broad, grad, epsilon, v_t):
    """
    _output_var_t_compute_use_nesterov
    # var_t = var - lr_t * (m_t * beta1 + (1 - beta1) * grad) / (epsilon + sqrt(v_t))
    # var_t = var - lr_t * (m_t * beta1 + (1 - beta1) * grad) / (epsilon + sqrt(v_t))
    """
    input_dtype = var.dtype
    compute_shape = shape_util.shape_to_list(var.shape)
    s_one = tvm.const(1, dtype=input_dtype)
    s_neg_one = tvm.const(-1, dtype=input_dtype)

    # `formula; m_t * beta1`
    v_muls_mt_beta1 = tbe.vmul(m_t, beta1_broad)

    # `formula; 1 -beta1`
    v_neg_beta1 = tbe.vmuls(beta1_broad, s_neg_one)
    vsub_1_beta1 = tbe.vadds(v_neg_beta1, s_one)

    # `formula; (1-beta1)* grad`
    v_mul_grad = tbe.vmul(vsub_1_beta1, grad)

    # `formula; (m_t*beta1 + (1 - beta1)*grad)`
    v_div_left = tbe.vadd(v_muls_mt_beta1, v_mul_grad)

    # `formula; lr_t * (m_t*beta1 + (1 - beta1) * grad)`
    # broadcast lr_t to vector
    lrt_broad = tbe.broadcast(lr_t, compute_shape)
    v_mul_left = tbe.vmul(lrt_broad, v_div_left)

    # `formula; (epsilon + sqrt(v_t))`
    v_add_sqrt_v = _inner_eps_add_sqrt_vt_compute(epsilon, v_t)

    # `formula; lr_t * (m_t*beta1 + (1-beta1)*grad / (epsilon + sqrt(v_t))`
    v_div_res = tbe.vdiv(v_mul_left, v_add_sqrt_v)

    # `formula; var - lr_t * (m_t*beta1 + (1-beta1)*grad) / (epsilon + sqrt(v_t))`
    v_t = tbe.vsub(var, v_div_res)

    return v_t


# `var_t = var - lr_t * m_t / (epsilon + sqrt(v_t))`
def _output_var_t_compute(var, lr_t, m_t, epsilon, v_t):
    """
    _output_var_t_compute
    `var_t = var - lr_t * m_t / (epsilon + sqrt(v_t))`
    """
    # `formula; lr_t * m_t`
    v_mul_left = tbe.vmul(lr_t, m_t)

    # `formula; (epsilon + sqrt(v_t))`
    v_add_sqrt_v = _inner_eps_add_sqrt_vt_compute(epsilon, v_t)

    # `formula; lr_t * m_t /(epsilon + sqrt(v_t))`
    v_div_res = tbe.vdiv(v_mul_left, v_add_sqrt_v)

    # `formula; var - lr_t * m_t / (epsilon + sqrt(v_t))`
    v_t = tbe.vsub(var, v_div_res)

    return v_t


@register_operator_compute("ApplyAdamD", op_mode="dynamic", support_fusion=only_static_support, support_bfp16=True)
def apply_adam_d_compute(var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, var_out, m_out, v_out,
                         use_nesterov, kernel_name="apply_adam_d"):
    """
    the opreator's compute
    lr_t = learning_rate*(sqrt(1-beta2_power)) / (1-beta1_power)
    m_t = m + (1-beta1)*(grad-m)
    v_t = v + (1-beta2)*(grad*grad-v)
    if use_nesterov == True:
        var_t = var - lr_t*(m_t*beta1 + (1-beta1)*grad) / (epsilon + sqrt(v_t))
    else:
        vat_t = var - lr_t*m_t / (epsilon + sqrt(v_t))
    Parameters:
    ----------
    var: dict
        input tensor contains shape and dtype attributes.
        only support float16, float32.
    m: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    v: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    beta1_power: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    beta2_power: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    lr: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    beta1: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    beta2: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    epsilon: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    grad: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    var_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    m_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'm'.
    v_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'v'.
    use_nesterov: bool
        default value is "False".
    kernel_name : str
        kernel name, default value is "apply_adam_d"

    Returns:
    None
    """

    shape_m_grad = m.shape
    beta1_broad = tbe.broadcast(beta1, shape_m_grad)
    m_t = _output_m_compute(m, beta1_broad, grad)
    v_t = _output_v_compute(v, beta2, grad)

    compute_shape = shape_util.shape_to_list(m.shape)
    lr_r = _inner_lr_compute(lr, beta2_power, beta1_power, compute_shape)

    if use_nesterov is True:
        var_t = _output_var_t_compute_use_nesterov(var, lr_r, m_t, beta1_broad, grad, epsilon, v_t)
    else:
        var_t = _output_var_t_compute(var, lr_r, m_t, epsilon, v_t)

    res = [var_t, m_t, v_t]
    return res


@register_operator("ApplyAdamD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def apply_adam_d(var,
                 m,
                 v,
                 beta1_power,
                 beta2_power,
                 lr,
                 beta1,
                 beta2,
                 epsilon,
                 grad,
                 var_out,
                 m_out,
                 v_out,
                 use_locking=False,
                 use_nesterov=False,
                 kernel_name="apply_adam_d"):
    """
    the opreator's compute
    lr_t = learning_rate*(sqrt(1-beta2_power)) / (1-beta1_power)
    m_t = m + (1-beta1)*(grad-m)
    v_t = v + (1-beta2)*(grad*grad-v)
    if use_nesterov == True:
        var_t = var - lr_t*(m_t*beta1 + (1-beta1)*grad) / (epsilon + sqrt(v_t))
    else:
        vat_t = var - lr_t*m_t / (epsilon + sqrt(v_t))
    Parameters:
    ----------
    var: dict
        input tensor contains shape and dtype attributes.
        only support bfloat16, float16, float32.
    m: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    v: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    beta1_power: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    beta2_power: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    lr: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    beta1: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    beta2: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    epsilon: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    grad: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    var_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    m_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'm'.
    v_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'v'.
    use_locking: bool
        default value is "False".
    use_nesterov: bool
        default value is "False".
    kernel_name : str
        kernel name, default value is "apply_adam_d"

    Returns:
    None
    """
    input_m_type = m.get("dtype").lower()
    compute_type = input_m_type
    shape_scalar = [1]
    data_beta1_power = tvm.placeholder(shape_scalar, name="data_beta1_power", dtype=compute_type)
    data_beta2_power = tvm.placeholder(shape_scalar, name="data_beta2_power", dtype=compute_type)
    data_lr = tvm.placeholder(shape_scalar, name="data_lr", dtype=compute_type)
    data_beta1 = tvm.placeholder(shape_scalar, name="data_beta1", dtype=compute_type)
    data_beta2 = tvm.placeholder(shape_scalar, name="data_beta2", dtype=compute_type)
    data_epsilon = tvm.placeholder(shape_scalar, name="data_epsilon", dtype=compute_type)

    ins = classify([var, m, v, grad], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for (_var, _m, _v, _grad) in ins:
        with tbe.compute():
            shape_var, shape_m, shape_v, shape_grad = shape_util.variable_shape([_var, _m, _v, _grad])
            data_var = tvm.placeholder(shape_var, name="data_var", dtype=compute_type)
            data_m = tvm.placeholder(shape_m, name="data_m", dtype=compute_type)
            data_v = tvm.placeholder(shape_v, name="data_v", dtype=compute_type)
            data_grad = tvm.placeholder(shape_grad, name="data_grad", dtype=compute_type)

        res = apply_adam_d_compute(data_var, data_m, data_v, data_beta1_power, data_beta2_power, data_lr, data_beta1,
                                   data_beta2, data_epsilon, data_grad, var_out, m_out, v_out, use_nesterov,
                                   kernel_name)

        tensors.append([
            data_var, data_m, data_v, data_beta1_power, data_beta2_power, data_lr, data_beta1, data_beta2, data_epsilon,
            data_grad
        ] + list(res))

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
