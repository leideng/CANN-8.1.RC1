# Copyright 2019 Huawei Technologies Co., Ltd
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
import functools
import operator
from tbe.dsl.api import auto_schedule
from tbe.dsl.api import build
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=too-many-arguments,invalid-name,too-many-locals
# 'pylint: disable=unused-argument
@register_operator_compute('apply_rms_prop_d', op_mode="static", support_fusion=True)
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
    rho_gs = tbe.vmuls(grad_square_ms, tvm.const(1.0 - rho, grad.dtype))
    ms_t = tbe.vadd(ms, rho_gs)

    m_mom = tbe.vmuls(mom, tvm.const(momentum, mom.dtype))

    lr_brc = tbe.broadcast(lr, grad.shape)
    lr_grad = tbe.vmul(grad, lr_brc)

    e_ms = tbe.vadds(ms_t, tvm.const(epsilon, ms.dtype))
    sqrt_ms = tbe.vsqrt(e_ms)
    tmp_grad = tbe.vdiv(lr_grad, sqrt_ms)
    mom_t = tbe.vadd(m_mom, tmp_grad)

    var_t = tbe.vsub(var, mom_t)

    return var_t, ms_t, mom_t


def _get_placeholder(dict_list, name_list):
    list_placeholder = []
    var_shape = []
    for var, name in zip(dict_list, name_list):
        shape = var.get('shape')
        dtype = var.get('dtype').lower()
        if name == 'var':
            var_shape = list(shape)
        if name == 'ms' and var_shape != list(shape):
            error_detail = "the shape of var and ms must be equal"
            error_manager_vector.raise_err_two_input_shape_invalid("apply_rms_prop_d", "var", "ms", error_detail)
        if name == 'mom' and var_shape != list(shape):
            error_detail = "the shape of var and mom must be equal"
            error_manager_vector.raise_err_two_input_shape_invalid("apply_rms_prop_d", "var", "mom", error_detail)
        if name == 'grad' and var_shape != list(shape):
            error_detail = "the shape of var and grad must be equal"
            error_manager_vector.raise_err_two_input_shape_invalid("apply_rms_prop_d", "var", "grad", error_detail)
        if name == 'lr' and shape[0] != 1:
            error_detail = "the shape of lr must be scalar"
            error_manager_vector.raise_err_input_shape_invalid("apply_rms_prop_d", "lr", error_detail)

        para_check.check_dtype(dtype, ('float32', ), param_name="var")
        para_check.check_shape(shape)
        shape_refine = (functools.reduce(operator.mul, shape), )
        list_placeholder.append(tvm.placeholder(shape=shape_refine, name=name, dtype=dtype))
    return list_placeholder


# 'pylint: disable=too-many-arguments,unused-argument,unbalanced-tuple-unpacking
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_FLOAT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.REQUIRED_ATTR_FLOAT, para_check.KERNEL_NAME)
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
    mom_{t} <- momentum * mom_{t-1} + learning_rate * grad / sqrt(ms_{t} +
            epsilon)
    var_{t} <- var_{t-1} - mom_{t}
    shape of learning_rate is (1,)

    Parameters:
    ----------
    var: dict of tensor var, include shape and dtype, dtype support float32.

    ms: dict of tensor ms(mean_square), include shape and dtype, dtype support
        float32.

    mom: dict of tensor mom, include shape and dtype, dtype support float32.

    lr: dict of scalar lr(learning rate), include shape and dtype, dtype
        support float32

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

    input_name_list = ['var', 'ms', 'mom', 'lr', 'grad']
    var, ms, mom, lr, grad = _get_placeholder([var, ms, mom, lr, grad], input_name_list)

    # compute
    out_var, out_ms, out_mom = apply_rms_prop_d_compute(var, ms, mom, lr, grad, out_var, out_ms, out_mom, rho, momentum,
                                                        epsilon)

    outs = [out_var, out_ms, out_mom]
    build_list = [var, ms, mom, lr, grad, out_var, out_ms, out_mom]

    with tvm.target.cce():
        sch = auto_schedule(outs)
    config = {"name": kernel_name, "tensor_list": build_list}
    build(sch, config)
