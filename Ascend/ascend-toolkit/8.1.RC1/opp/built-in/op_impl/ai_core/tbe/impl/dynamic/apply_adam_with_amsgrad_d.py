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
apply_adam_with_amsgrad_d
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_attr_common import OpAttr
from impl.util.util_attr_common import get_attr_by_cls


class ApplyAdamWithAmsgradDAttrInfo:
    """
    define attr info
    """
    ATTR_BETA1 = OpAttr(0, "beta1", "Float")
    ATTR_BETA2 = OpAttr(1, "beta2", "Float")
    ATTR_EPSILON = OpAttr(2, "epsilon", "Float")


# 'pylint: disable=too-many-arguments,invalid-name,too-many-locals,unused-argument
@register_operator_compute("ApplyAdamWithAmsgradD", op_mode="dynamic", support_fusion=True)
def apply_adam_with_amsgrad_d_compute(var,
                                      m,
                                      v,
                                      vhat,
                                      beta1_power,
                                      beta2_power,
                                      lr,
                                      beta1,
                                      beta2,
                                      epsilon,
                                      grad,
                                      kernel_name="apply_adam_with_amsgrad_d"):
    """
    the operator's compute
    :param var: weight, placeholder
    :param m: moment, placeholder
    :param v: moment, placeholder
    :param vhat: vhat, placeholder
    :param beta1_power: beta1_power, placeholder
    :param beta2_power: beta2_power, placeholder
    :param lr: learning rate, placeholder
    :param beta1: beta1, const
    :param beta2: beta2, const
    :param epsilon: epsilon, const
    :param grad: grad, placeholder
    """
    num_one = 1.0
    num_n_one = -1.0
    inp_dtype = var.dtype
    # check the instruction supports or not
    vmul_support = tbe_platform.api_check_support("tbe.dsl.vmul", "float32")
    if inp_dtype == "float32" and not vmul_support:
        error_manager_vector.raise_err_input_dtype_not_supported(kernel_name, 'var', [], inp_dtype)

    one = tvm.const(num_one, "float32")
    neg_one = tvm.const(num_n_one, "float32")
    beta1 = get_attr_by_cls(beta1, ApplyAdamWithAmsgradDAttrInfo.ATTR_BETA1, "float32")
    beta2 = get_attr_by_cls(beta2, ApplyAdamWithAmsgradDAttrInfo.ATTR_BETA2, "float32")
    epsilon = get_attr_by_cls(epsilon, ApplyAdamWithAmsgradDAttrInfo.ATTR_EPSILON, "float32")

    beta1_power = tbe.broadcast(beta1_power, var.shape)
    beta2_power = tbe.broadcast(beta2_power, var.shape)
    lr = tbe.broadcast(lr, var.shape)

    # update lr
    beta1_power_neg = tbe.vmuls(beta1_power, neg_one)
    beta2_power_neg = tbe.vmuls(beta2_power, neg_one)
    beta1_power_tmp = tbe.vadds(beta1_power_neg, one)
    beta2_power_tmp = tbe.vadds(beta2_power_neg, one)
    beta_sqrt = tbe.vsqrt(beta2_power_tmp)
    lr_sqrt = tbe.vmul(lr, beta_sqrt)
    lr_t = tbe.vdiv(lr_sqrt, beta1_power_tmp)

    # update m
    m_mul = tbe.vmuls(m, beta1)
    beta1_negadd = beta1 * neg_one + one
    m_grad = tbe.vmuls(grad, beta1_negadd)
    m_t = tbe.vadd(m_mul, m_grad)

    # update v
    beta2_t = tbe.vmuls(v, beta2)
    beta2_negadd = beta2 * neg_one + one
    grad_pow = tbe.vmul(grad, grad)
    beta2_grad = tbe.vmuls(grad_pow, beta2_negadd)
    v_t = tbe.vadd(beta2_t, beta2_grad)

    # update vhat
    vhat_t = tbe.vmax(vhat, v_t)

    # update var
    var_m = tbe.vmul(lr_t, m_t)
    var_sqrt = tbe.vsqrt(vhat_t)
    var_epsilon = tbe.vadds(var_sqrt, epsilon)
    var_div = tbe.vdiv(var_m, var_epsilon)
    var_t = tbe.vsub(var, var_div)

    return [var_t, m_t, v_t, vhat_t]


# 'pylint: disable=too-many-arguments,unused-argument,too-many-locals,unbalanced-tuple-unpacking
@register_operator("ApplyAdamWithAmsgradD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def apply_adam_with_amsgrad_d(var,
                              m,
                              v,
                              vhat,
                              beta1_power,
                              beta2_power,
                              lr,
                              grad,
                              var_output,
                              m_output,
                              v_output,
                              vhat_output,
                              beta1,
                              beta2,
                              epsilon,
                              use_locking=False,
                              kernel_name="apply_adam_with_amsgrad_d"):
    """
    Update '*var' according to the Adam algorithm.

    lr_t := {learning_rate} * sqrt{1 - beta_2^t} / (1 - beta_1^t)

    m_t := beta_1 * m_{t-1} + (1 - beta_1) * g

    v_t := beta_2 * v_{t-1} + (1 - beta_2) * g * g

    vhat_t := max{vhat_{t-1}, v_t}

    variable := variable - lr_t * m_t / (sqrt{vhat_t} + epsilon)

    Parameters
    ----------
    var : dict of tensor var, include shape and dtype

    m : dict of tensor m, include shape and dtype

    v: dict of tensor v, include shape and dtype

    vhat : dict of tensor vhat, include shape and dtype

    beta1_power: dict of beta1_power, include shape and dtype.

    beta2_power: dict of beta2_power, include shape and dtype.

    lr: dict of lr, include shape and dtype.

    grad: dict of grad, include shape and dtype.

    var_output: dict of update var.

    m_output: dict of update m.

    v_output: dict of update v.

    vhat_output: dict of update vhat.

    beta1: scalar, attr in D. Must have the same dtype as var.

    beta2: scalar, attr in D. Must have the same dtype as var.

    epsilon: scalar, attr in D. Must have the same dtype as var.

    use_locking: An optional `bool`. Defaults to `False`. If `True`,
    updating of the var, m, and v tensors will be protected.

    kernel_name : kernel name, default value is "apply_adam_with_amsgrad_d"

    Returns
    -------
    None
    """
    compute_type = var.get("dtype").lower()
    shape_scalar = [1]
    data_lr = tvm.placeholder(shape_scalar, name="data_lr", dtype=compute_type)
    data_beta1_power = tvm.placeholder(shape_scalar, name="data_beta1_power", dtype=compute_type)
    data_beta2_power = tvm.placeholder(shape_scalar, name="data_beta2_power", dtype=compute_type)

    ins = classify([var, m, v, vhat, grad], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for(_var, _m, _v, _vhat, _grad) in ins:
        with tbe.compute():
            shape_var, shape_m, shape_v, shape_vhat, shape_grad = \
                shape_util.variable_shape([_var, _m, _v, _vhat, _grad])

            data_var = tvm.placeholder(shape_var, name="data_var", dtype=compute_type)
            data_m = tvm.placeholder(shape_m, name="data_m", dtype=compute_type)
            data_v = tvm.placeholder(shape_v, name="data_v", dtype=compute_type)
            data_vhat = tvm.placeholder(shape_vhat, name="data_vhat", dtype=compute_type)
            data_grad = tvm.placeholder(shape_grad, name="data_grad", dtype=compute_type)

            res = apply_adam_with_amsgrad_d_compute(data_var, data_m, data_v,
                                                    data_vhat, data_beta1_power,
                                                    data_beta2_power, data_lr, beta1, beta2,
                                                    epsilon, data_grad, kernel_name)

            tensors.append([data_var, data_m, data_v, data_vhat, data_beta1_power,
                            data_beta2_power, data_lr, data_grad] + res)

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {
        "name": kernel_name,
        "tensor_list": tensors
    }

    tbe.build(schedules, config)
