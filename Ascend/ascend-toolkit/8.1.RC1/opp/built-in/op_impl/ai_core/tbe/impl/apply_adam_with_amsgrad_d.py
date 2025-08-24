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
apply_adam_with_amsgrad_d
"""
import functools
import operator

from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
import te.lang.cce as tbe
from impl.util.platform_adapter import auto_schedule
from impl.util.platform_adapter import build
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=too-few-public-methods, not-use-list-comprehension
class Constant:
    """
    The class for constant
    """
    NUM_ONE = 1.0
    NUM_N_ONE = -1.0


# 'pylint: disable=too-many-arguments,invalid-name,too-many-locals,unused-argument
@register_operator_compute("apply_adam_with_amsgrad_d", op_mode="static", support_fusion=True)
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
    :param lr: learning rate, const
    :param beta1: beta1, const
    :param beta2: beta2, const
    :param epsilon: epsilon, const
    :param grad: grad, placeholder
    """
    inp_dtype = var.dtype
    # check the instruction supports or not
    vmul_support = tbe_platform.api_check_support("tbe.dsl.vmul", "float32")
    if inp_dtype == "float32" and not vmul_support:
        error_manager_vector.raise_err_input_dtype_not_supported(kernel_name, 'var', [], inp_dtype)

    one = tvm.const(Constant.NUM_ONE, "float32")
    neg_one = tvm.const(Constant.NUM_N_ONE, "float32")

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

    return var_t, m_t, v_t, vhat_t


def _check_para_and_getplaceholder(scalar_input, tensor_input, input_dict):
    check_list = ("float32", )
    var_shape = input_dict["var"].get("shape")
    var_dtype = input_dict["var"].get("dtype")
    list_placeholder = []
    for key, value in input_dict.items():
        shape = shape_util.scalar2tensor_one(value.get("shape"))
        para_check.check_shape(shape)
        if value in scalar_input:
            if not para_check.is_scalar(shape):
                error_detail = "the shape of " + key + " must be scalar"
                error_manager_vector.raise_err_input_shape_invalid("apply_adam_with_amsgrad_d", key, error_detail)
        if value in tensor_input:
            if shape != var_shape:
                error_detail = "the shape of " + key + " must be the same as the var"
                error_manager_vector.raise_err_two_input_shape_invalid("apply_adam_with_amsgrad_d", "var", key,
                                                                       error_detail)

        dtype = value.get("dtype").lower()
        para_check.check_dtype(dtype, check_list, param_name="var")
        if dtype != var_dtype:
            error_detail = "the dtype of " + key + " must be the same as the var"
            error_manager_vector.raise_err_two_input_dtype_invalid("apply_adam_with_amsgrad_d", "var", key,
                                                                   error_detail)

        shape_refine = (functools.reduce(operator.mul, shape), )
        list_placeholder.append(tvm.placeholder(shape=shape_refine, name=key, dtype=dtype))
    return list_placeholder


# 'pylint: disable=too-many-arguments,unused-argument,too-many-locals,unbalanced-tuple-unpacking
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.REQUIRED_ATTR_FLOAT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
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
    input_dict = {
        "var": var,
        "m": m,
        "v": v,
        "vhat": vhat,
        "beta1_power": beta1_power,
        "beta2_power": beta2_power,
        "lr": lr,
        "grad": grad
    }
    scalar_input = (lr, beta1_power, beta2_power, epsilon)
    tensor_input = (var, m, v, vhat, grad)
    var_input, m_input, v_input, vhat_input, beta1_power, beta2_power, lr_input, grad_input =\
         _check_para_and_getplaceholder(scalar_input, tensor_input, input_dict)

    var_output, m_output, v_output, vhat_output = apply_adam_with_amsgrad_d_compute(var_input, m_input, v_input,
                                                                                    vhat_input, beta1_power,
                                                                                    beta2_power, lr_input, beta1, beta2,
                                                                                    epsilon, grad_input, kernel_name)
    with tvm.target.cce():
        schedule = auto_schedule([var_output, m_output, v_output, vhat_output])

    config = {
        "name":
        kernel_name,
        "tensor_list": [
            var_input, m_input, v_input, vhat_input, beta1_power, beta2_power, lr_input, grad_input, var_output,
            m_output, v_output, vhat_output
        ]
    }

    build(schedule, config)
