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
this file achieved the apply_adadelta_d which is a optimizer operator
to update weight
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


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    NUM_ZERO = 0.0


# 'pylint: disable=locally-disabled,too-many-arguments
# 'pylint: disable=unused-argument,invalid-name,too-many-locals
@register_operator_compute("ApplyAdadeltaD", op_mode="dynamic", support_fusion=True)
def apply_adadelta_d_compute(var,
                             accum,
                             accum_update,
                             lr,
                             rho,
                             epsilon,
                             grad,
                             var_out,
                             accum_out,
                             accum_update_out,
                             kernel_name='apply_adadelta_d'):
    """
    Update '*var' according to the adadelta scheme.

    accum = rho * accum + (1 - rho) * grad ** 2
    update = (update_accum + epsilon).sqrt() * (accum + epsilon).rsqrt()*grad
    update_accum = rho * update_accum + (1 - rho) * update.square();
    var -= update * lr;

    Parameters:
    ----------
    var : mutable tensor var.

    accum: mutable tensor accum.

    accum_update : mutable tensor accum_update.

    lr : scalar lr.

    rho : scalar rho.

    epsilon : scalar epsilon.

    grad : tensor grad.

    var_out : the dict of var output.

    accum_out : the dict of accum output.

    accum_update_out : the dict of accum_update output.

    kernel_name : cce kernel name, default value is "apply_adadelta_d".

    Returns:
    -------
    None
    """
    num_one = 1.0
    dtype = var.dtype
    has_improve_precision = False
    if dtype == "float16" and \
            tbe_platform.api_check_support("te.lang.cce.vsqrt", "float32"):
        has_improve_precision = True

    if dtype == "float16" and has_improve_precision:
        var = tbe.cast_to(var, "float32")
        accum = tbe.cast_to(accum, "float32")
        accum_update = tbe.cast_to(accum_update, "float32")
        lr = tbe.cast_to(lr, "float32")
        rho = tbe.cast_to(rho, "float32")
        epsilon = tbe.cast_to(epsilon, "float32")
        grad = tbe.cast_to(grad, "float32")

    scalar_one = tbe.broadcast(tvm.const(num_one, "float32"), rho.shape)
    tensor_rho_gs = tbe.vsub(scalar_one, rho)
    scalar_rho = rho[0]
    scalar_epsilon = epsilon[0]

    # step1: update accum
    rhs = tbe.vmuls(accum, scalar_rho)
    lhs = tbe.vmul(grad, grad)
    scalar_rho_gs = tensor_rho_gs[0]
    lhs = tbe.vmuls(lhs, scalar_rho_gs)
    accum_res = tbe.vadd(lhs, rhs)

    accum_update_orig = tbe.vadds(accum_update, Constant.NUM_ZERO)
    # step2
    rhs = tbe.vadds(accum_update_orig, scalar_epsilon)
    rhs = tbe.vsqrt(rhs)
    lhs = tbe.vadds(accum_res, scalar_epsilon)
    lhs = tbe.vsqrt(lhs)
    lhs = tbe.vdiv(grad, lhs)
    update = tbe.vmul(lhs, rhs)

    # step3: update var
    var_res = lr[0]
    var_res = tbe.vmuls(update, var_res)
    var_res = tbe.vsub(var, var_res)

    # step4: update accum_update
    rhs = tbe.vmuls(accum_update_orig, scalar_rho)
    lhs = tbe.vmul(update, update)
    lhs = tbe.vmuls(lhs, scalar_rho_gs)
    accum_update_res = tbe.vadd(lhs, rhs)

    # out
    output_data = tbe.vadds(var_res, Constant.NUM_ZERO)
    accum_output_data = tbe.vadds(accum_res, Constant.NUM_ZERO)
    accum_update_output_data = tbe.vadds(accum_update_res, Constant.NUM_ZERO)

    if dtype == "float16" and has_improve_precision:
        output_data = tbe.cast_to(output_data, "float16")
        accum_output_data = tbe.cast_to(accum_output_data, "float16")
        accum_update_output_data = \
            tbe.cast_to(accum_update_output_data, "float16")

    return [output_data, accum_output_data, accum_update_output_data]


@register_operator("ApplyAdadeltaD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def apply_adadelta_d(var,
                     accum,
                     accum_update,
                     lr,
                     rho,
                     epsilon,
                     grad,
                     var_out,
                     accum_out,
                     accum_update_out,
                     kernel_name="apply_adadelta_d"):
    """
    Update '*var' according to the adadelta scheme.

    accum = rho * accum + (1 - rho) * grad ** 2
    update = (update_accum + epsilon).sqrt() * (accum + epsilon).rsqrt() * grad
    update_accum = rho * update_accum + (1 - rho) * update.square();
    var -= update * lr;

    Parameters:
    ----------
    var: the dict of input, only support float16, float32

    accum: the dict of accum, only support float16, float32

    accum_update: the dict of accum_update, only support float16, float32

    lr: the dict of lr, only support float16, float32

    rho: the dict of rho, only support float16, float32

    epsilon: the dict of epsilon, only support float16, float32

    grad: the dict of grad, only support float16, float32

    var_out: the dict of var output data

    accum_out: the dict of accum output data

    accum_update_out: the dict of accum_update output data

    kernel_name : cce kernel name, default value is "apply_adadelta_d"

    Returns
    -------
    None
    """
    input_accum_type = accum.get("dtype").lower()
    compute_type = input_accum_type
    shape_scalar = [1]
    data_lr = tvm.placeholder(shape_scalar, name="data_lr", dtype=compute_type)
    data_rho = tvm.placeholder(shape_scalar, name="data_rho", dtype=compute_type)
    data_epsilon = tvm.placeholder(shape_scalar, name="data_epsilon", dtype=compute_type)

    ins = classify([var, accum, accum_update, grad], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for(d_var, d_accum, d_accum_update, d_grad) in ins:
        with tbe.compute():
            shape_var, shape_accum, shape_accum_update, shape_grad = \
                shape_util.variable_shape([d_var, d_accum, d_accum_update, d_grad])

            data_var = tvm.placeholder(shape_var, name="data_input_var", dtype=compute_type)
            data_accum = tvm.placeholder(shape_accum, name="data_input_accum", dtype=compute_type)
            data_accum_update = tvm.placeholder(shape_accum_update, name="data_input_accum_update",
                                                dtype=compute_type)
            data_grad = tvm.placeholder(shape_grad, name="data_input_grad", dtype=compute_type)

        res = apply_adadelta_d_compute(data_var, data_accum, data_accum_update,
                                       data_lr, data_rho, data_epsilon, data_grad,
                                       var_out, accum_out, accum_update_out, kernel_name)

        tensors.append([data_var, data_accum, data_accum_update,
                        data_lr, data_rho, data_epsilon, data_grad] + list(res))

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
