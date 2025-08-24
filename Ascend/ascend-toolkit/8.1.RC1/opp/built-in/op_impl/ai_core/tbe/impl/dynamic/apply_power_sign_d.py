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
dynamic apply_power_sign

Op_description :
Update '*var' according to the AddSign update.

# apply_power_sign_d(var,
#   m,
#   lr,
#   logbase,
#   sign_decay,
#   beta,
#   grad,
#   out,
#   kernel_name='cce_apply_power_sign')

Supportive_dtype_format :
['int32', 'int8', 'uint8', 'float32', 'float16']
['ND', 'NCHW', 'NHWC', 'NC1HWC0']

Constraint :
[1] All : the input tensors must have the same shape and type.
[2] All : shape size limit is 2147483648.
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util import util_compute


# 'pylint: disable=locally-disabled,invalid-name
# 'pylint: disable=too-many-locals
def _compute_m_t(m, beta, grad):
    beta_tmp = tbe.vmul(m, beta)
    beta_na = tbe.vmuls(beta, tvm.const(-1.0, beta.dtype))
    beta_na = tbe.vadds(beta_na, tvm.const(1.0, beta_na.dtype))
    beta_sub_tmp = tbe.vmul(grad, beta_na)
    m_t = tbe.vadd(beta_tmp, beta_sub_tmp)

    return m_t


def _compute_update(logbase, sign_decay, sign_gm, grad):
    vmul_tmp = tbe.vmul(sign_gm, sign_decay)
    vmul_tmp = tbe.vmul(vmul_tmp, logbase)
    exp_tmp = tbe.vexp(vmul_tmp)
    update = tbe.vmul(exp_tmp, grad)

    return update


def _compute_var(var, lr, update):
    lt_tmp = tbe.vmul(update, lr)
    var_t = tbe.vsub(var, lt_tmp)

    return var_t


def _compute_process(input_list):
    """
    Calculate the vat_t, m_t

    Returns
    ----------
    var_t, m_t
    """
    var, m, lr, logbase, sign_decay, beta, grad = input_list[0], input_list[1], \
                                                  input_list[2], input_list[3], \
                                                  input_list[4], input_list[5], \
                                                  input_list[6]
    m_t = _compute_m_t(m, beta, grad)
    sign_gm = tbe.vmul(util_compute.sign(m_t), util_compute.sign(grad))
    update = _compute_update(logbase, sign_decay, sign_gm, grad)
    var_t = _compute_var(var, lr, update)

    return var_t, m_t


# 'pylint: disable=locally-disabled, too-many-arguments, unused-argument
@register_operator_compute("ApplyPowerSignD", op_mode="dynamic", support_fusion=True)
def apply_power_sign_d_compute(var,
                               m,
                               lr,
                               logbase,
                               sign_decay,
                               beta,
                               grad,
                               var_out,
                               m_out,
                               kernel_name="apply_power_sign_d"):
    """
    Calculate the algorithm

    Parameters:
    ----------
    var : mutable tensor var
    m : mutable tensor m
    lr : scalar lr
    logbase : scalar logbase
    sign_decay : scalar sign_decay
    beta : scalar beta
    grad : mutable tensor grad
    var_out : var output
    m_out : m output

    Algorithm :
    ----------
    m_t <- beta * m_{t-1} + (1 - beta) * grad
    update <- exp(logbase * sign_decay * sign(grad) * sign(m_t)) * grad
    variable <- variable - lr_t * update

    Returns
    ----------
    out_var, out_m
    """

    dtype = var.dtype
    if dtype == "float16" and \
            tbe_platform.api_check_support("te.lang.cce.vadd", "float32"):
        var = tbe.cast_to(var, "float32")
        m = tbe.cast_to(m, "float32")
        lr = tbe.cast_to(lr, "float32")
        logbase = tbe.cast_to(logbase, "float32")
        sign_decay = tbe.cast_to(sign_decay, "float32")
        beta = tbe.cast_to(beta, "float32")
        grad = tbe.cast_to(grad, "float32")

    compute_shape = shape_util.shape_to_list(var.shape)
    lr = tbe.broadcast(lr, compute_shape)
    logbase = tbe.broadcast(logbase, compute_shape)
    sign_decay = tbe.broadcast(sign_decay, compute_shape)
    beta = tbe.broadcast(beta, compute_shape)

    input_list = [var, m, lr, logbase, sign_decay, beta, grad]
    var_t, m_t = _compute_process(input_list)

    if dtype == "float16":
        var_t = tbe.cast_to(var_t, "float16")
        m_t = tbe.cast_to(m_t, "float16")

    return var_t, m_t


@register_operator("ApplyPowerSignD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def apply_power_sign_d(var,
                       m,
                       lr,
                       logbase,
                       sign_decay,
                       beta,
                       grad,
                       var_out,
                       m_out,
                       kernel_name="apply_power_sign_d"):
    """
    Update '*var' according to the AddSign update

    Parameters:
    ----------
    var: dict of Variable, only support float16, float32
    m : dict of input_grad, only support float16, float32
    lr : dict of lr, only support float16, float32
    logbase : dict of logbase, only support float16, float32
    sign_decay : dict of sign_decay, only support float16, float32
    grad : dict of grad, only support float16, float32
    beta : dict of beta, only support float16, float32
    var_out : dict of output, only support float16, float32
    m_out : dict of output, only support float16, float32
    kernel_name : cce kernel name, default value is apply_power_sign

    Algorithm :
    ----------
    m_t <- beta * m_{t-1} + (1 - beta) * grad
    update <- exp(logbase * sign_decay * sign(grad) * sign(m_t)) * grad
    variable <- variable - lr_t * update


    Returns
    ----------
    None
    """

    dtype = var.get('dtype').lower()
    check_list = ('float16', 'float32')
    para_check.check_dtype(dtype, check_list, param_name="var")

    shape_scalar = [1]
    data_lr = tvm.placeholder(shape_scalar, name="ipnut_lr", dtype=dtype)
    data_logbase = tvm.placeholder(shape_scalar, name="ipnut_logbase", dtype=dtype)
    data_sign_decay = tvm.placeholder(shape_scalar, name="ipnut_sign_decay", dtype=dtype)
    data_beta = tvm.placeholder(shape_scalar, name="ipnut_beta", dtype=dtype)

    ins = classify([var, m, grad], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_var, _m, _grad) in ins:
        with tbe.compute():
            shape_var, shape_m, shape_grad = shape_util.variable_shape([_var, _m, _grad])
            data_var = tvm.placeholder(shape_var, name="ipnut_var", dtype=dtype)
            data_m = tvm.placeholder(shape_m, name="ipnut_m", dtype=dtype)
            data_grad = tvm.placeholder(shape_grad, name="ipnut_grad", dtype=dtype)
            res = apply_power_sign_d_compute(data_var, data_m, data_lr,
                                             data_logbase, data_sign_decay,
                                             data_beta, data_grad, var_out,
                                             m_out, kernel_name)

            tensor_list = [data_var, data_m, data_lr,
                           data_logbase, data_sign_decay,
                           data_beta, data_grad] + list(res)
            tensors.append(tensor_list)
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
