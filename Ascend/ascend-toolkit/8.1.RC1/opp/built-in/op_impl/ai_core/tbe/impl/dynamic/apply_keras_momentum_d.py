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
dynamic apply_keras_momentum_d
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
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=too-many-arguments,invalid-name,too-many-locals
@register_operator_compute("ApplyKerasMomentumD", op_mode="dynamic", support_fusion=True)
def apply_keras_momentum_d_compute(var,
                                   accum,
                                   lr,
                                   grad,
                                   momentum,
                                   out_var,
                                   out_accum,
                                   use_nesterov,
                                   kernel_name="apply_keras_momentum_d"):
    """
    the operator's compute
    :param var: weight, placeholder
    :param accum: accum, placeholder
    :param lr: learning rate, placeholder
    :param grad: gradient, placeholder
    :param momentum: nesterov momentum, placeholder
    :param out_var: updated of var
    :param out_accum: updated of accum
    :param use_nesterov: bool
    :return: out_var, out_accum
    """
    inp_dtype = var.dtype
    # check the instruction supports or not
    vmul_support = tbe_platform.api_check_support("te.lang.cce.vmul", "float32")
    if inp_dtype == "float32" and not vmul_support:
        error_manager_vector.raise_err_input_dtype_not_supported(kernel_name, 'var', [], inp_dtype)

    compute_shape = shape_util.shape_to_list(var.shape)

    lr = tbe.broadcast(lr, compute_shape)
    momentum = tbe.broadcast(momentum, compute_shape)

    # update var and accum according to the momentum scheme
    # `accum = accum * momentum - grad * lr`
    accum_momen = tbe.vmul(accum, momentum)
    grad_lr = tbe.vmul(grad, lr)
    out_accum = tbe.vsub(accum_momen, grad_lr)

    # `var = var + accum * momentum - grad * lr`
    if use_nesterov:
        accum_momen2 = tbe.vmul(out_accum, momentum)
        add_var_am = tbe.vadd(var, accum_momen2)
        out_var = tbe.vsub(add_var_am, grad_lr)
    # `var = var + accum`
    else:
        out_var = tbe.vadd(var, out_accum)

    res = [out_var, out_accum]

    return res


# 'pylint: disable=unused-argument
@register_operator("ApplyKerasMomentumD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def apply_keras_momentum_d(var,
                           accum,
                           lr,
                           grad,
                           momentum,
                           out_var,
                           out_accum,
                           use_locking=False,
                           use_nesterov=False,
                           kernel_name="apply_keras_momentum_d"):
    """
    Update '*var' according to the momentum scheme.
    accum = accum * momentum - grad * lr
    if use_nesterov is True:
        var = var + accum * momentum - grad * lr
    else:
        var = var + accum
    Parameters
    ----------
    var : dict of tensor var, include shape and dtype.
    accum : dict of tensor accum, include shape and dtype.
    lr: dict of scalar lr(learning rate), include shape and dtype.
    grad: dict of tensor grad, include shape and dtype.
    momentum: dict of scalar, include shape and dtype.
    out_var: dict of updated var.
    out_accum: dict of updated accum.
    use_locking: bool, default value is "False",
                 if "True", var will be updated by using Nesterov momentum.
    use_nesterov: bool, default value is "False".
    kernel_name :  kernel name, default value is "apply_keras_momentum_d"

    Returns
    -------
    None
    """
    var_type = var.get("dtype").lower()
    compute_type = var_type

    shape_scalar = [1]
    data_lr = tvm.placeholder(shape_scalar, name="data_lr", dtype=compute_type)
    data_momentum = tvm.placeholder(shape_scalar, name="data_momentum", dtype=compute_type)

    ins = classify([var, accum, grad], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_var, _accum, _grad) in ins:
        with tbe.compute():
            shape_var, shape_accum, shape_grad = shape_util.variable_shape([_var, _accum, _grad])
            data_var = tvm.placeholder(shape_var, name="data_var", dtype=compute_type)
            data_accum = tvm.placeholder(shape_accum, name="data_accum", dtype=compute_type)
            data_grad = tvm.placeholder(shape_grad, name="data_grad", dtype=compute_type)
            res = apply_keras_momentum_d_compute(data_var, data_accum,
                                                 data_lr, data_grad,
                                                 data_momentum, out_var,
                                                 out_accum, use_nesterov)
            tensor_list = [data_var, data_accum, data_lr, data_grad, data_momentum] + list(res)
            tensors.append(tensor_list)
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
