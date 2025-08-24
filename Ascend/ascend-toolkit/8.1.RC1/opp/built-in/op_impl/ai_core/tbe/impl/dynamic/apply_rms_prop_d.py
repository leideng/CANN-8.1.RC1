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
this file achieved the apply_rms_prop which is a optimizer operator to update
weight, this file contains compute and schedule.
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_attr_common import OpAttr
from impl.util.util_attr_common import get_attr_by_cls


class ApplyRMSPropDAttrInfo:
    """
    define attr info
    """
    ATTR_RHO = OpAttr(0, "rho", "Float")
    ATTR_MOMENTUM = OpAttr(1, "momentum", "Float")
    ATTR_EPSILON = OpAttr(2, "epsilon", "Float")


# 'pylint: disable=too-many-arguments,invalid-name,too-many-locals
# 'pylint: disable=unused-argument
def rho_gs_compute(rho, grad_square_ms, dtype):
    """
    get the rho_gs
    """
    if rho is None:
        rho_var = get_attr_by_cls(rho, ApplyRMSPropDAttrInfo.ATTR_RHO, dtype)
        rho_gs = tbe.vmuls(grad_square_ms, rho_var)
        rho_gs = tbe.vsub(grad_square_ms, rho_gs)
    else:
        rho_gs = tbe.vmuls(grad_square_ms, tvm.const(1.0 - rho, dtype))
    return rho_gs


@register_operator_compute("ApplyRMSPropD", op_mode="dynamic", support_fusion=True)
def apply_rms_prop_d_compute(var,
                             ms,
                             mom,
                             lr,
                             grad,
                             out_var,
                             out_ms,
                             out_mom,
                             rho,
                             momentum,
                             epsilon,
                             kernel_name="apply_rms_prop_d"):
    """
    the operator's compute
    :param var: weight, placeholder, float32
    :param ms: mean square, placeholder, float32
    :param mom: momentum, placeholder, float32
    :param lr: learning rate, placeholder, float32
    :param grad: gradient, placeholder, float32
    :param out_var: updated of var
    :param out_ms: updated of ms
    :param out_mom: updated of mom
    :param rho: const, float32
    :param momentum: const, float32
    :param epsilon: const, float32
    :return: out_var, out_ms, out_mom
    """
    grad_square = tbe.vmul(grad, grad)
    grad_square_ms = tbe.vsub(grad_square, ms)

    rho_gs = rho_gs_compute(rho, grad_square_ms, grad.dtype)
    ms_t = tbe.vadd(ms, rho_gs)

    momentum_var = get_attr_by_cls(momentum, ApplyRMSPropDAttrInfo.ATTR_MOMENTUM, mom.dtype)
    m_mom = tbe.vmuls(mom, momentum_var)
    lr_brc = tbe.broadcast(lr, grad.shape)
    lr_grad = tbe.vmul(grad, lr_brc)

    epsilon_var = get_attr_by_cls(epsilon, ApplyRMSPropDAttrInfo.ATTR_EPSILON, ms.dtype)
    e_ms = tbe.vadds(ms_t, epsilon_var)

    sqrt_ms = tbe.vsqrt(e_ms)
    tmp_grad = tbe.vdiv(lr_grad, sqrt_ms)
    mom_t = tbe.vadd(m_mom, tmp_grad)
    var_t = tbe.vsub(var, mom_t)

    res = [var_t, ms_t, mom_t]

    return res


# 'pylint: disable=too-many-arguments,unused-argument,unbalanced-tuple-unpacking
@register_operator("ApplyRMSPropD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_FLOAT,
                            para_check.OPTION_ATTR_FLOAT, para_check.KERNEL_NAME)
def apply_rms_prop_d(var,
                     ms,
                     mom,
                     lr,
                     grad,
                     out_var,
                     out_ms,
                     out_mom,
                     rho,
                     momentum,
                     epsilon,
                     kernel_name="apply_rms_prop_d"):
    """
    Update '*var' according to the RMSProp algorithm.
    mean_square = decay * mean_square + (1-decay) * gradient ** 2
    Delta = learning_rate * gradient / sqrt(mean_square + epsilon)
    ms_{t} <- rho * ms_{t-1} + (1-rho) * grad * grad
    mom_{t} <- momentum * mom_{t-1} + learning_rate * grad / sqrt(ms_{t} + epsilon)
    var_{t} <- var_{t-1} - mom_{t}
    shape of learning_rate is (1,)

    Parameters:
    ----------
    var: dict of tensor var, include shape and dtype, dtype support float32.
    ms: dict of tensor ms(mean_square), include shape and dtype, dtype support
        float32.
    mom: dict of tensor mom, include shape and dtype, dtype support float32.
    lr: dict of scalar lr(learning rate), include shape and dtype, dtype support float32
    grad: dict of tensor grad, include shape and dtype, dtype support float32.
    out_var: dict of updated var.
    out_ms: dict of updated ms.
    out_mom: dict of updated mom.
    rho: scalar rho(decay rate), attr in d. Must have the same dtype as var.
    momentum: scalar momentum, attr in d. Must have the same dtype as var.
    epsilon: scalar epsilon, attr in d. Must have the same dtype as var.
    kernel_name : cce kernel name, default value is "apply_rms_prop".

    Returns
    -------
    None
    """

    var_dtype = var.get("dtype").lower()
    check_list = ("float32",)
    para_check.check_dtype(var_dtype, check_list, param_name="var")

    shape_scalar = [1]
    compute_dtype = var_dtype
    data_lr = tvm.placeholder(shape_scalar, name="data_alpha", dtype=compute_dtype)

    ins = classify([var, ms, mom, grad], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_var, _ms, _mom, _grad) in ins:
        with tbe.compute():
            shape_var, shape_ms, shape_mom, shape_grad = shape_util.variable_shape([_var, _ms, _mom, _grad])
            data_var = tvm.placeholder(shape_var, name="data_var", dtype=compute_dtype)
            data_ms = tvm.placeholder(shape_ms, name="data_ms", dtype=compute_dtype)
            data_mom = tvm.placeholder(shape_mom, name="data_mom", dtype=compute_dtype)
            data_grad = tvm.placeholder(shape_grad, name="data_grad", dtype=compute_dtype)
            res = apply_rms_prop_d_compute(data_var, data_ms, data_mom,
                                           data_lr, data_grad, out_var,
                                           out_ms, out_mom, rho,
                                           momentum, epsilon)
            tensor_list = [data_var, data_ms, data_mom, data_lr, data_grad] + list(res)
            tensors.append(tensor_list)
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
