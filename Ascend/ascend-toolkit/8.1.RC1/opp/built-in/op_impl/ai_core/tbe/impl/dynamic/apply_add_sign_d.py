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
this file achieved the apply_add_sign_d which is a optimizer operator
to update weight, this file contains compute and schedule.
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpPatternMode


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    CONST_ZERO = 0
    CONST_ONE = 1
    CONST_ONE_NEG = -1


# 'pylint: disable=locally-disabled,too-many-arguments
# 'pylint: disable=unused-argument,invalid-name,too-many-locals
@register_operator_compute("ApplyAddSignD", op_mode="dynamic", support_fusion=True)
def apply_add_sign_d_compute(var,
                             m,
                             lr,
                             alpha,
                             sign_decay,
                             beta,
                             grad,
                             var_output,
                             m_output,
                             kernel_name='apply_add_sign_d'):
    """
    Update '*var' according to the AddSign update.

    m_t <- beta1 * m_{t-1} + (1 - beta1) * g
    update <- (alpha + sign_decay * sign(g) *sign(m)) * g
    variable <- variable - lr_t * update

    Parameters:
    ----------
    var: the dict of var, support float16, float32
    m: the dict of m, support float16, float32
    lr: the dict of lr, support float16, float32
    alpha: the dict of alpha, support float16, float32
    sign_decay: the dict of sign_decay, support float16, float32
    beta: the dict of beta, support float16, float32
    grad: the dict of grad, support float16, float32

    Returns
    -------
    the new value of var and m
    the output
    """
    output_data, m_output_data = _compute_process(
        var, m, tbe.broadcast(lr, var.shape),
        tbe.broadcast(alpha, var.shape),
        tbe.broadcast(sign_decay, var.shape),
        tbe.broadcast(beta, var.shape), grad)

    return output_data, m_output_data


def _compute_process(var, m, lr_broad, alpha_broad, sign_decay_broad,
                     beta_broad, grad):
    """
    calculate
    m_t <- beta1 * m_{t-1} + (1 - beta1) * g
    update <- (alpha + sign_decay * sign(g) *sign(m)) * g
    variable <- variable - lr_t * update

    Parameters:
    ----------
    var: the dict of var, support float16, float32
    m: the dict of m, support float16, float32
    lr: the dict of lr, support float16, float32
    alpha: the dict of alpha, support float16, float32
    sign_decay: the dict of sign_decay, support float16, float32
    beta: the dict of beta, support float16, float32
    grad: the dict of grad, support float16, float32

    Returns
    -------
    the new value of var and out
    the output
    """
    m_out = _update_m(m, beta_broad, grad)
    sign_gm = tbe.vmul(_sign_compute(grad), _sign_compute(m_out))
    decay_gm = tbe.vmul(sign_gm, sign_decay_broad)
    var_out = _update_var(decay_gm, alpha_broad, lr_broad, grad, var)

    output_data = tbe.vadds(var_out, tvm.const(Constant.CONST_ZERO, "float32"))
    m_output_data = tbe.vadds(m_out, tvm.const(Constant.CONST_ZERO, "float32"))

    return output_data, m_output_data


def _update_m(m_old, beta_broad, grad):
    """
    calculate m = m * beta + grad * (1 - beta)

    Parameters:
    ----------
    m_old: the value of initial m
    beta_broad: the value of beta_broad
    grad: the value of grad

    Returns
    -------
    the new value of m
    """
    m_beta = tbe.vmul(m_old, beta_broad)
    beta_neg = tbe.vmuls(beta_broad, tvm.const(Constant.CONST_ONE_NEG,
                                               "float32"))
    beta_1 = tbe.vadds(beta_neg, tvm.const(Constant.CONST_ONE, "float32"))
    grad_beta_gs = tbe.vmul(grad, beta_1)
    m_out = tbe.vadd(m_beta, grad_beta_gs)

    return m_out


def _update_var(decay_gm, alpha_broad, lr_broad, grad, var):
    """
    calculate var = var - lr * (alpha + sign_decay * sign_gm) * grad

    Parameters:
    ----------
    decay_gm: the value of decay_gm
    alpha_broad: the dict of alpha_broad
    lr_broad: the dict of lr_broad
    grad: the dict of grad
    var: the value of var

    Returns
    -------
    the new value of var
    """
    decay_gm_alpha = tbe.vadd(decay_gm, alpha_broad)
    res = tbe.vmul(decay_gm_alpha, lr_broad)
    res = tbe.vmul(res, grad)
    res_neg = tbe.vmuls(res, tvm.const(Constant.CONST_ONE_NEG, "float32"))
    var_out = tbe.vadd(var, res_neg)

    return var_out


def _sign_compute(input_data):
    """
    this compute is for sign output
    """
    input_dtype = input_data.dtype
    input_x = tbe.broadcast(tvm.const(Constant.CONST_ONE, input_dtype), input_data.shape)
    input_y = tbe.broadcast(tvm.const(Constant.CONST_ZERO, input_dtype), input_data.shape)
    input_z = tbe.broadcast(tvm.const(Constant.CONST_ONE_NEG, input_dtype), input_data.shape)
    res1 = tbe.vcmpsel(input_data, tvm.const(Constant.CONST_ZERO, input_dtype), "gt", input_x, input_y)
    res2 = tbe.vcmpsel(input_data, tvm.const(Constant.CONST_ZERO, input_dtype), "lt", input_z, input_y)
    res = tbe.vadd(res1, res2)

    return res


@register_operator("ApplyAddSignD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def apply_add_sign_d(var,
                     m,
                     lr,
                     alpha,
                     sign_decay,
                     beta,
                     grad,
                     var_out,
                     m_out,
                     kernel_name="apply_add_sign_d"):
    """
    Update '*var' according to the AddSign update.

    m_t <- beta1 * m_{t-1} + (1 - beta1) * g
    update <- (alpha + sign_decay * sign(g) *sign(m)) * g
    variable <- variable - lr_t * update

    Parameters:
    ----------
    var: the dict of var, support float16, float32
    m: the dict of m, support float16, float32
    lr: the dict of lr, support float16, float32
    alpha: the dict of alpha, support float16, float32
    sign_decay: the dict of sign_decay, support float16, float32
    beta: the dict of beta, support float16, float32
    grad: the dict of grad, support float16, float32
    var_out: the dict of var output data
    m_out: the dict of m output data
    otherwise the behavior is undefined, but may exhibit less contention.
    kernel_name : cce kernel name, default value is "apply_add_sign_d"

    Returns
    -------
    None
    """

    check_list = ('float16', 'float32')
    var_dtype = var.get("dtype").lower()
    compute_type = var_dtype
    para_check.check_dtype(compute_type, check_list, param_name="var")

    ins = classify([var, m, grad], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_var, _m, _grad) in ins:
        with tbe.compute():
            shape_var, shape_m, shape_grad = shape_util.variable_shape([_var, _m, _grad])

            data_var = tvm.placeholder(shape_var, name="data_var", dtype=compute_type)
            data_m = tvm.placeholder(shape_m, name="data_m", dtype=compute_type)
            data_grad = tvm.placeholder(shape_grad, name="data_grad", dtype=compute_type)

            data_lr = tvm.placeholder([1], dtype=compute_type, name="data_lr")
            data_alpha = tvm.placeholder([1], dtype=compute_type, name="data_alpha")
            data_sign_decay = tvm.placeholder([1], dtype=compute_type, name="data_sign_decay")
            data_beta = tvm.placeholder([1], dtype=compute_type, name="data_beta")

            var_output, m_output = \
                apply_add_sign_d_compute(data_var, data_m, data_lr, data_alpha, data_sign_decay,
                                         data_beta, data_grad, var_out, m_out, kernel_name)

            tensors.append([data_var, data_m, data_lr, data_alpha, data_sign_decay,
                            data_beta, data_grad, var_output, m_output])

        with tvm.target.cce():
            sch = tbe.auto_schedule([var_output, m_output])
        schedules.append(sch)

    config = {
        "name": kernel_name,
        "tensor_list": tensors
    }

    tbe.build(schedules, config)
