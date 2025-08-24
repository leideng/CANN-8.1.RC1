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
This file achieved the apply_centered_rms_prop_d which is a optimizer operator
to update weight.

apply_centered_rms_prop_d

Op_description :
Update '*var' according to the centered RMSProp algorithm.
Update '*mg' according to the centered RMSProp algorithm.
Update '*ms' according to the centered RMSProp algorithm.
Update '*mom' according to the centered RMSProp algorithm.

# apply_centered_rms_prop_d(var,
#   mg,
#   ms,
#   mom,
#   lr,
#   rho,
#   momentum,
#   epsilon,
#   grad,
#   var_out,
#   mg_out,
#   ms_out,
#   mom_out,
#   kernel_name='apply_centered_rms_prop_d')

Supportive_dtype_format :
['int32', 'int8', 'uint8', 'float32', 'float16']
['ND', 'NCHW', 'NHWC', 'NC1HWC0']

Constraint :
[1] All : the input tensors must have the same shape and type.
[2] All : shape size limit is 2147483648.
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name,too-many-locals
@register_operator_compute("ApplyCenteredRMSPropD", op_mode="dynamic", support_fusion=True)
def apply_centered_rms_prop_d_compute(var,
                                      mg,
                                      ms,
                                      mom,
                                      lr,
                                      rho,
                                      momentum,
                                      epsilon,
                                      grad,
                                      var_out,
                                      mg_out,
                                      ms_out,
                                      mom_out,
                                      kernel_name="apply_centered_rms_prop_d"):
    """
    Update '*var' according to the centered RMSProp algorithm.

    mean_square = decay * mean_square + (1-decay) * gradient ** 2
    mean_grad = decay * mean_grad + (1-decay) * gradient
    Delta = learning_rate*gradient/sqrt(mean_square+epsilon-mean_grad**2)
    mg_{t} <- rho * mg_{t-1} + (1-rho) * grad
    ms_{t} <- rho * ms_{t-1} + (1-rho) * grad * grad
    mom_{t} <- momentum*mom_{t-1}+lr*grad/sqrt(ms_{t}-mg{t}*mg{t}+epsilon)
    var_{t} <- var_{t-1} - mom_{t}

    Parameters:
    ----------
    var: dict of tensor var, include shape and dtype,
        dtype support float16 and float32.
    mg: dict of tensor mg(mean_grad), include shape and dtype,
        dtype support float16 and float32.
    ms: dict of tensor ms(mean_square), include shape and dtype,
        dtype support float16 and float32.
    mom: dict of tensor mom, include shape and dtype,
        dtype support float16 and float32.
    lr: dict of scalar lr(learning rate). Must have the same dtype as var.
    rho: dict of scalar rho(decay rate). Must have the same dtype as var.
    momentum: dict of scalar momentum. Must have the same dtype as var.
    epsilon: dict of scalar epsilon. Must have the same dtype as var.
    grad: dict of tensor grad. Must have the same dtype as var.
    out: dict of output out.
    kernel_name : cce kernel name, default value is "apply_centered_rms_prop_d".

    Returns
    -------
    None
    """

    inp_dtype = var.dtype
    if inp_dtype == "float16" and \
            tbe_platform.api_check_support("te.lang.cce.vadd", "float32"):
        var = tbe.cast_to(var, "float32")
        mg = tbe.cast_to(mg, "float32")
        ms = tbe.cast_to(ms, "float32")
        mom = tbe.cast_to(mom, "float32")
        lr = tbe.cast_to(lr, "float32")
        rho = tbe.cast_to(rho, "float32")
        momentum = tbe.cast_to(momentum, "float32")
        epsilon = tbe.cast_to(epsilon, "float32")
        grad = tbe.cast_to(grad, "float32")

    compute_shape = shape_util.shape_to_list(var.shape)
    lr = tbe.broadcast(lr, compute_shape)
    rho = tbe.broadcast(rho, compute_shape)
    momentum = tbe.broadcast(momentum, compute_shape)
    epsilon = tbe.broadcast(epsilon, compute_shape)
    grad = tbe.broadcast(grad, compute_shape)

    tensor_one_rho = tbe.vmuls(rho, -1.0)
    tensor_one_rho = tbe.vadds(tensor_one_rho, 1.0)

    mg_rho = tbe.vmul(mg, rho)

    rhs = tbe.vmul(grad, tensor_one_rho)
    out_mg = tbe.vadd(mg_rho, rhs)

    ms_rho = tbe.vmul(ms, rho)
    rhs = tbe.vmul(grad, grad)

    rhs = tbe.vmul(rhs, tensor_one_rho)
    out_ms = tbe.vadd(ms_rho, rhs)

    lhs_mom = tbe.vmul(mom, momentum)

    lr_grad = tbe.vmul(grad, lr)
    rhs = tbe.vmul(out_mg, out_mg)
    rhs = tbe.vsub(out_ms, rhs)

    rhs_eps = tbe.vadd(rhs, epsilon)
    rhs_eps = tbe.vsqrt(rhs_eps)
    rhs_eps = tbe.vdiv(lr_grad, rhs_eps)
    out_mom = tbe.vadd(lhs_mom, rhs_eps)

    out_var = tbe.vsub(var, out_mom)

    if inp_dtype == "float16":
        out_var = tbe.cast_to(out_var, "float16")
        out_mg = tbe.cast_to(out_mg, "float16")
        out_ms = tbe.cast_to(out_ms, "float16")
        out_mom = tbe.cast_to(out_mom, "float16")

    res = [out_var, out_mg, out_ms, out_mom]

    return res


@register_operator("ApplyCenteredRMSPropD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def apply_centered_rms_prop_d(var,
                              mg,
                              ms,
                              mom,
                              lr,
                              rho,
                              momentum,
                              epsilon,
                              grad,
                              var_out,
                              mg_out,
                              ms_out,
                              mom_out,
                              kernel_name="apply_centered_rms_prop_d"):
    """
    Update '*var' according to the centered RMSProp algorithm.

    mean_square = decay * mean_square + (1-decay) * gradient ** 2
    mean_grad = decay * mean_grad + (1-decay) * gradient
    Delta = learning_rate*gradient/sqrt(mean_square+epsilon-mean_grad**2)
    mg_{t} <- rho * mg_{t-1} + (1-rho) * grad
    ms_{t} <- rho * ms_{t-1} + (1-rho) * grad * grad
    mom_{t} <- momentum*mom_{t-1}+lr*grad/sqrt(ms_{t}-mg{t}*mg{t}+epsilon)
    var_{t} <- var_{t-1} - mom_{t}

    Parameters:
    ----------
    var: dict of tensor var, include shape and dtype,
        dtype support float16 and float32.
    mg: dict of tensor mg(mean_grad), include shape and dtype,
        dtype support float16 and float32.
    ms: dict of tensor ms(mean_square), include shape and dtype,
        dtype support float16 and float32.
    mom: dict of tensor mom, include shape and dtype,
        dtype support float16 and float32.
    lr: dict of scalar lr(learning rate). Must have the same dtype as var.
    rho: dict of scalar rho(decay rate). Must have the same dtype as var.
    momentum: dict of scalar momentum. Must have the same dtype as var.
    epsilon: dict of scalar epsilon. Must have the same dtype as var.
    grad: dict of tensor grad. Must have the same dtype as var.
    var_out: the dict of var output, only support float16, float32
    mg_out: the dict of mg output, only support float16, float32
    ms_out: the dict of ms output, only support float16, float32
    mom_out: the dict of mom output, only support float16, float32
    kernel_name : cce kernel name, default value is "apply_centered_rms_prop_d".
    Returns
    -------
    None
    """

    var_type = var.get("dtype").lower()
    check_list = ('float16', 'float32')
    para_check.check_dtype(var_type, check_list, param_name="var")

    compute_type = var_type

    shape_scalar = [1]
    data_lr = tvm.placeholder(shape_scalar, name="data_lr", dtype=compute_type)
    data_rho = tvm.placeholder(shape_scalar, name="data_rho", dtype=compute_type)
    data_momentum = tvm.placeholder(shape_scalar, name="data_momentum", dtype=compute_type)
    data_epsilon = tvm.placeholder(shape_scalar, name="data_epsilon", dtype=compute_type)

    ins = classify([var, mg, ms, mom, grad], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_var, _mg, _ms, _mom, _grad) in ins:
        with tbe.compute():
            shape_var, shape_mg, shape_ms, shape_mom, shape_grad = shape_util.variable_shape(
                [_var, _mg, _ms, _mom, _grad])
            data_var = tvm.placeholder(shape_var, name="data_var", dtype=compute_type)
            data_mg = tvm.placeholder(shape_mg, name="data_mg", dtype=compute_type)
            data_ms = tvm.placeholder(shape_ms, name="data_ms", dtype=compute_type)
            data_mom = tvm.placeholder(shape_mom, name="data_mom", dtype=compute_type)
            data_grad = tvm.placeholder(shape_grad, name="data_grad", dtype=compute_type)

            res = apply_centered_rms_prop_d_compute(data_var, data_mg, data_ms,
                                                    data_mom, data_lr, data_rho,
                                                    data_momentum, data_epsilon,
                                                    data_grad, var_out, mg_out,
                                                    ms_out, mom_out)

            tensors_list = [data_var, data_mg, data_ms, data_mom, data_lr, data_rho,
                            data_momentum, data_epsilon, data_grad] + list(res)
            tensors.append(tensors_list)

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
