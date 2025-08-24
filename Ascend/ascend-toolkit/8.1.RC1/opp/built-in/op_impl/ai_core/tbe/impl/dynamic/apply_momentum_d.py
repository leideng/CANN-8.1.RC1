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
dynamic apply_momentum_d
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.apply_momentum_d import get_op_support_info as apply_momentum_d_get_op_support_info


def get_op_support_info(var,
                        accum,
                        lr,
                        grad,
                        momentum,
                        var_out,
                        accum_out,
                        use_nesterov=False,
                        kernel_name="apply_momentum_d"):
    """
    get_op_support_info
    """
    return apply_momentum_d_get_op_support_info(var, accum, lr, grad, momentum, var_out, accum_out,
                                                use_nesterov, kernel_name)


# 'pylint: disable=locally-disabled,too-many-arguments,too-many-locals
# 'pylint: disable=locally-disabled,unused-argument,invalid-name,unused-variable
@register_operator_compute("ApplyMomentumD", op_mode="dynamic", support_fusion=True)
def apply_momentum_compute_d(var,
                             accum,
                             lr,
                             grad,
                             momentum,
                             var_out,
                             accum_out,
                             use_nesterov,
                             kernel_name='apply_momentum_d'):
    """
    Update '*var' according to the ApplyMomentum algorithm.

    accum = accum * momentum + grad
    if use_nesterov is True:
        var -= grad * lr + accum * momentum * lr
    else:
        var -= accum * lr

    Parameters:
    ----------
    var : mutable tensor var.

    accum: mutable tensor accum.

    lr : scalar lr.

    grad : tensor grad.

    momentum : scalar momentum.

    var_out : the dict of output var.

    accum_out : the dict of output accum.

    use_nesterov: bool. If true, use nesterov computing grad,
        default value is False.

    kernel_name : cce kernel name, default value is "apply_momentum_d".

    Returns:
    -------
    None
    """

    # cast to float32 for higher accuracy
    dtype = var.dtype
    if dtype == "float16" and tbe_platform.api_check_support("te.lang.cce.vadd", "float32"):
        var = tbe.cast_to(var, "float32")
        accum = tbe.cast_to(accum, "float32")
        lr = tbe.cast_to(lr, "float32")
        grad = tbe.cast_to(grad, "float32")
        momentum = tbe.cast_to(momentum, "float32")

    compute_shape = shape_util.shape_to_list(var.shape)

    beta1_momentum = momentum[0]
    beta1_lr = lr[0]

    # `accum_delta = accum * momentum`
    accum_delta = tbe.vmuls(accum, beta1_momentum)

    # `accum_t =  accum_delta + grad`
    accum_t = tbe.vadd(accum_delta, grad)

    # update var
    if use_nesterov:
        var_delta = tbe.vmuls(grad, beta1_lr)
        var_delta_2 = tbe.vmuls(accum_t, beta1_momentum)
        var_delta_2 = tbe.vmuls(var_delta_2, beta1_lr)
        var_delta = tbe.vadd(var_delta, var_delta_2)
        var_t = tbe.vsub(var, var_delta)
    else:
        var_delta = tbe.vmuls(accum_t, beta1_lr)
        var_t = tbe.vsub(var, var_delta)

    if dtype == "float16":
        var_t = tbe.cast_to(var_t, "float16")
        accum_t = tbe.cast_to(accum_t, "float16")

    res = [var_t, accum_t]
    return res


@register_operator("ApplyMomentumD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def apply_momentum_d(var,
                     accum,
                     lr,
                     grad,
                     momentum,
                     var_out,
                     accum_out,
                     use_nesterov=False,
                     kernel_name="apply_momentum_d"):
    """
    Update '*var' according to the ApplyMomentum algorithm.

    accum = accum * momentum + grad
    if use_nesterov is True:
        var -= gard * lr + accum * momentum * lr
    else:
        var -= accum * lr

    Parameters:
    ----------
    var : the dict of mutable tensor var, only support float16, float32.

    accum : the dict of mutable tensor accum.
        Must have the same data type as `var`.

    lr : the dict of scalar lr. Must have the same data type as `var`.

    grad : the dict of tensor grad. Must have the same data type as `var`.

    momentum : the dict of scalar momentum.
        Must have the same data type as `var`.

    var_out : the dict of output var.

    accum_out : the dict of output accum.

    use_nesterov: bool. If true, use nesterov computing grad,
        default value is False.

    kernel_name : cce kernel name, default value is "apply_momentum_d".

    Returns
    -------
    None
    """
    var_type = var.get("dtype").lower()
    accum_type = accum.get("dtype").lower()
    grad_type = grad.get("dtype").lower()
    lr_type = lr.get("dtype").lower()
    momentum_type = momentum.get("dtype").lower()

    if var_type != accum_type:
        error_detail = "type of var and accum should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "var", "accum", error_detail)

    if var_type != grad_type:
        error_detail = "type of var and grad should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "var", "grad", error_detail)

    if var_type != lr_type:
        error_detail = "type of var and lr should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "var", "lr", error_detail)

    if var_type != momentum_type:
        error_detail = "type of var and momentum should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "var", "momentum", error_detail)

    shape_lr = [1]
    shape_momentum = [1]
    compute_type = var.get("dtype").lower()

    # `shape_size = reduce(lambda x, y: x * y, shape_var[:])`
    # `compute_type = [shape_size, ]`
    ins = classify([var, accum, grad], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for (_var, _accum, _grad) in ins:
        with tbe.compute():
            shape_var, shape_accum, shape_grad = shape_util.variable_shape([_var, _accum, _grad])
            data_var = tvm.placeholder(shape_var, name="data_var", dtype=compute_type)
            data_accum = tvm.placeholder(shape_accum, name="data_accum", dtype=compute_type)
            data_lr = tvm.placeholder(shape_lr, name="data_lr", dtype=compute_type)
            data_grad = tvm.placeholder(shape_grad, name="data_grad", dtype=compute_type)
            data_momentum = tvm.placeholder(shape_momentum, name="data_momentum", dtype=compute_type)

            res = apply_momentum_compute_d(data_var, data_accum, data_lr, data_grad, data_momentum,
                                           var_out, accum_out, use_nesterov, kernel_name)

            tensor_list = [data_var, data_accum, data_lr, data_grad, data_momentum] + list(res)
            tensors.append(tensor_list)
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
