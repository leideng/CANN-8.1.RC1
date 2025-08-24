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
apply_proximal_adagrad_d
"""
from impl.util import util_compute
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.apply_proximal_adagrad_d import op_select_format as apply_proximal_adagrad_d_op_select_format


def op_select_format(var, accum, lr, l1, l2, grad, var_out,
                     accum_out, use_locking=False,
                     kernel_name="apply_proximal_adagrad_d"):
    """
    Select format according to the following rules.
    lr l1 l2 only support ND,and var_out accum_out is same as var
    1.When the var, accum or grad's shape is -2, all inputs and outputs only ND format is supported
    2.Supports var accum grad ND, 5HD, FZ when the N and C axis is divisible by 16
    3.Supports var accum grad ND, 5HD when the N and C axis is divisible by 16
    4.In other cases, only ND is supported
    """
    return apply_proximal_adagrad_d_op_select_format(var, accum, lr, l1, l2, grad, var_out, accum_out,
                                                     use_locking, kernel_name)


# 'pylint: disable=too-many-locals,unused-argument,invalid-name,too-many-arguments
@register_operator_compute("ApplyProximalAdagradD", op_mode="dynamic", support_fusion=True)
def apply_proximal_adagrad_d_compute(var, accum, lr, l1, l2, grad, var_out,
                                     accum_out, use_locking=False,
                                     kernel_name="apply_proximal_adagrad_d"):
    """
    the operator's compute
    accum += grad * grad
    learning_rate = lr_broad * rsqrt(accum)
    prox_v = var - grad * learning_rate
    if l1 > 0 :
        var = sign(prox_v)/(1+learning_rate*l2)*max{|prox_v|-learning_rate*l1,0}
    else:
        var = prox_v / (1+l2*learning_rate)

    Parameters
    ----------
    var: dict
        input tensor contains shape and dtype attributes.
        only support float16, float32.
    accum: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    lr: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    l1: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    l2: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    grad: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    var_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    accum_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'accum'.
    use_locking: bool
        default value is "False"
    kernel_name: str
        kernel name, default value is "apply_proximal_adagrad_d"

    Returns:
        the value of out_var, accum_out, out_data
    """
    dtype = var.dtype
    has_improve_precision = False
    if dtype == "float16" and \
        tbe_platform.api_check_support("te.lang.cce.vsqrt", "float32"):
        var = tbe.cast_to(var, "float32")
        accum = tbe.cast_to(accum, "float32")
        lr = tbe.cast_to(lr, "float32")
        l1 = tbe.cast_to(l1, "float32")
        l2 = tbe.cast_to(l2, "float32")
        grad = tbe.cast_to(grad, "float32")
        has_improve_precision = True

    lr_broad = tbe.broadcast(lr, var.shape)
    l1_broad = tbe.broadcast(l1, var.shape)
    l2_broad = tbe.broadcast(l2, var.shape)

    grad_2 = tbe.vmul(grad, grad)
    accum_out = tbe.vadd(accum, grad_2)
    accum_sqrt = tbe.vsqrt(accum_out)
    learning_rate = tbe.vdiv(lr_broad, accum_sqrt)
    learning_rate_grad = tbe.vmul(grad, learning_rate)
    prox_v = tbe.vsub(var, learning_rate_grad)
    l2_lr = tbe.vmul(l2_broad, learning_rate)
    l2_lr_1 = tbe.vadds(l2_lr, tvm.const(1, "float32"))
    prox_v_abs = tbe.vabs(prox_v)
    prox_v_sign = util_compute.sign(prox_v)
    learning_rate_l1 = tbe.vmul(learning_rate, l1_broad)
    prox_v_l1 = tbe.vsub(prox_v_abs, learning_rate_l1)
    max_value = tbe.vmax(prox_v_l1, tbe.broadcast(
        tvm.const(0, "float32"), prox_v.shape))
    var_res = tbe.vmul(prox_v_sign, max_value)
    var_out = tbe.vdiv(var_res, l2_lr_1)

    if has_improve_precision:
        var_out = tbe.cast_to(var_out, "float16")
        accum_out = tbe.cast_to(accum_out, "float16")


    res = [var_out, accum_out]

    return res


@register_operator("ApplyProximalAdagradD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def apply_proximal_adagrad_d(var, accum, lr, l1, l2, grad, var_out,
                             accum_out, use_locking=False,
                             kernel_name="apply_proximal_adagrad_d"):
    """
    Update '*var' and '*accum' according to FOBOS with Adagrad learning rate.

    Parameters
    ----------
    var: dict
        input tensor contains shape and dtype attributes.
        only support float16, float32.
    accum: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    lr: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    l1: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    l2: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    grad: dict
        input tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    var_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'var'.
    accum_out: dict
        output tensor contains shape and dtype attributes.
        Must have the same type as 'accum'.
    use_locking: bool
        default value is "False"
    kernel_name: str
        kernel name, default value is "apply_proximal_adagrad_d"

    Returns:
    None
    """
    var_dtype = var.get("dtype").lower()
    check_list = ("float16", "float32")
    para_check.check_dtype(var_dtype, check_list, param_name="var")

    shape_scalar = [1]
    compute_dtype = var_dtype
    data_lr = tvm.placeholder(shape_scalar, name="data_lr", dtype=compute_dtype)
    data_l1 = tvm.placeholder(shape_scalar, name="data_l1", dtype=compute_dtype)
    data_l2 = tvm.placeholder(shape_scalar, name="data_l2", dtype=compute_dtype)

    ins = classify([var, accum, grad], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_var, _accum, _grad) in ins:
        with tbe.compute():
            shape_var, shape_accum, shape_grad = shape_util.variable_shape([_var, _accum, _grad])
            data_var = tvm.placeholder(shape_var, name="data_var", dtype=compute_dtype)
            data_accum = tvm.placeholder(shape_accum, name="data_delta", dtype=compute_dtype)
            data_grad = tvm.placeholder(shape_grad, name="data_grad", dtype=compute_dtype)
            res = apply_proximal_adagrad_d_compute(data_var, data_accum, data_lr,
                                                   data_l1, data_l2, data_grad,
                                                   var_out, accum_out, use_locking)
            tensor_list = [data_var, data_accum, data_lr, data_l1, data_l2, data_grad] + list(res)
            tensors.append(tensor_list)
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
