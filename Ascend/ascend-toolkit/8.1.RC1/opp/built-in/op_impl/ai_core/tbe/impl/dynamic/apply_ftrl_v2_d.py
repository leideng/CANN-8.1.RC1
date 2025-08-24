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
dynamic apply_ftrl_v2_d
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util import util_compute
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpImplMode
from impl.util.platform_adapter import OpTbeImplMode


def _pow(data_x, data_y):
    """
    Calculate power result for non-negative data.

    res = data_x^data_y
        = exp(data_y * ln(data_x)) if data_x >= 0

    Parameters:
    ----------
    data_x : base value of power operation.
    data_y: index value of power operation.

    Returns:
    -------
    power result of data_x^data_y
    """

    log_value = tbe.vlog(data_x, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
    mul_value = tbe.vmul(data_y, log_value)
    res = tbe.vexp(mul_value)

    return res


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=invalid-name,too-many-locals, too-many-statements
@register_operator_compute("ApplyFtrlV2D", op_mode="dynamic", support_fusion=True)
def apply_ftrl_v2_d_compute(var, accum, linear, grad, lr, l1, l2, l2_shrinkage,
                            lr_power, var_out, accum_out, linear_out,
                            kernel_name='apply_ftrl_v2_d'):
    """
    Update '*var' according to the Ftrl-proximal algorithm.

    grad_with_shrinkage = grad + 2 * l2_shrinkage * var
    accum_new = accum + grad * grad
    linear += grad_with_shrinkage -
        (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
    x = l1 * linear.sign - linear
    y = accum_new^(-lr_power) / lr + 2 * l2
    var = x / y if |linear| > l1 else 0.0
    accum = accum_new

    Parameters:
    ----------
    var : mutable tensor var.
    accum: mutable tensor accum.
    linear : mutable tensor linear.
    grad : tensor grad.
    lr : scalar lr.
    l1 : scalar l1.
    l2 : scalar l2.
    l2_shrinkage: scalar l2_shrinkage.
    lr_power : scalar lr_power.
    var_out : the dict of output var.
    accum_out : the dict of output accum.
    linear_out : the dict of output linear.
    kernel_name : cce kernel name, default value is "apply_ftrl_v2_d".

    Returns:
    -------
    the value of var_new, accum_new, linear_new, output_data
    """
    dtype = var.dtype
    # cast to float32 for higher accuracy
    has_improve_precision = False
    if dtype == "float16" and \
            tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        var_tmp = tbe.cast_to(var, "float32")
        accum_tmp = tbe.cast_to(accum, "float32")
        linear_tmp = tbe.cast_to(linear, "float32")
        grad = tbe.cast_to(grad, "float32")
        lr = tbe.cast_to(lr, "float32")
        l1 = tbe.cast_to(l1, "float32")
        l2 = tbe.cast_to(l2, "float32")
        l2_shrinkage = tbe.cast_to(l2_shrinkage, "float32")
        lr_power = tbe.cast_to(lr_power, "float32")
        has_improve_precision = True
    else:
        var_tmp = tbe.vadds(var, tvm.const(0.0, "float32"))
        accum_tmp = tbe.vadds(accum, tvm.const(0.0, "float32"))
        linear_tmp = tbe.vadds(linear, tvm.const(0.0, "float32"))

    # `1.grad_with_shrinkage = grad + 2 * l2_shrinkage * var`
    mul_value = tbe.vmuls(l2_shrinkage, tvm.const(2.0, "float32"))
    mul_value = tbe.broadcast(mul_value, var_tmp.shape)
    mul_value2 = tbe.vmul(mul_value, var_tmp)
    grad_with_shrinkage = tbe.vadd(grad, mul_value2)

    # `2.accum_new = accum + grad^2`
    gs = tbe.vmul(grad, grad)
    accum_new = tbe.vadd(accum_tmp, gs)

    # `3.accum_pow_sub = accum_new^(-lr_power)-accum^(-lr_power)`
    lr_power = tbe.vmuls(lr_power, tvm.const(-1.0, "float32"))
    lr_power = tbe.broadcast(lr_power, var_tmp.shape)
    accum_new_pow = _pow(accum_new, lr_power)
    accum_pow = _pow(accum_tmp, lr_power)
    accum_pow_sub = tbe.vsub(accum_new_pow, accum_pow)

    # `4.linear += grad_with_shrinkage - accum_pow_sub / lr * var`
    lr = tbe.broadcast(lr, var_tmp.shape)
    accum_pow_div = tbe.vdiv(accum_pow_sub, lr)
    accum_pow_mul = tbe.vmul(accum_pow_div, var_tmp)
    accum_pow = tbe.vsub(grad_with_shrinkage, accum_pow_mul)
    linear_new = tbe.vadd(linear_tmp, accum_pow)

    # `5.x_res = l1*linear.sign()-linear`
    l1 = tbe.broadcast(l1, var_tmp.shape)
    x_res = util_compute.sign(linear_new)
    x_res = tbe.vmul(x_res, l1)
    x_res = tbe.vsub(x_res, linear_new)

    # `6.y_res = accum_new^(-lr_power)/lr + 2*l2`
    l2 = tbe.vmuls(l2, tvm.const(2.0, "float32"))
    l2 = tbe.broadcast(l2, var_tmp.shape)
    y_res = tbe.vdiv(accum_new_pow, lr)
    y_res = tbe.vadd(y_res, l2)

    # `7.var = x_res / y_res if linear.abs > l1, else var = 0`
    x_res = tbe.vdiv(x_res, y_res)
    linear_abs = tbe.vabs(linear_new)
    zero_tensor = tbe.broadcast(tvm.const(0.0, "float32"), var_tmp.shape)
    var_new = tbe.vcmpsel(linear_abs, l1, 'gt', x_res, zero_tensor)

    # dtype after vsel is float16 at mini
    var_new = tbe.cast_to(var_new, "float32")

    if has_improve_precision:
        var_new = tbe.cast_to(var_new, "float16")
        accum_new = tbe.cast_to(accum_new, "float16")
        linear_new = tbe.cast_to(linear_new, "float16")
    res = [var_new, accum_new, linear_new]

    return res


@register_operator("ApplyFtrlV2D")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def apply_ftrl_v2_d(var, accum, linear, grad, lr, l1, l2, l2_shrinkage, lr_power,
                    var_out, accum_out, linear_out, use_locking=False,
                    kernel_name="apply_ftrl_v2_d"):
    """
    Update '*var' according to the Ftrl-proximal algorithm.

    grad_with_shrinkage = grad + 2 * l2_shrinkage * var
    accum_new = accum + grad * grad
    linear += grad_with_shrinkage -
        (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
    x = l1 * linear.sign - linear
    y = accum_new^(-lr_power) / lr + 2 * l2
    var = x / y if |linear| > l1 else 0.0
    accum = accum_new

    Parameters:
    ----------
    var : the dict of mutable tensor var, only support float16, float32
    accum : the dict of mutable tensor accum.
        Must have the same data type as `var`.
    linear : the dict of mutable tensor linear.
        Must have the same data type as `var`.
    grad : the dict of tensor grad. Must have the same data type as `var`.
    lr : the dict of scalar lr. Must have the same data type as `var`.
    l1 : the dict of scalar l1. Must have the same data type as `var`.
    l2 : the dict of scalar l2. Must have the same data type as `var`.
    l2_shrinkage: the dict of scalar l2_shrinkage.
        Must have the same data type as `var`.
    lr_power : the dict of scalar lr_power.
        Must have the same data type as `var`.
    var_out : the dict of output var.
    accum_out : the dict of output accum.
    linear_out : the dict of output linear.
    use_locking : optional attr, default value is False.
    kernel_name : cce kernel name, default value is "apply_ftrl_v2_d".

    Returns
    -------
    None
    """
    var_dtype = var.get("dtype").lower()
    compute_dtype = var_dtype
    shape_scalar = [1]
    data_lr = tvm.placeholder(shape_scalar, name="data_lr", dtype=compute_dtype)
    data_l1 = tvm.placeholder(shape_scalar, name="data_l1", dtype=compute_dtype)
    data_l2 = tvm.placeholder(shape_scalar, name="data_l2", dtype=compute_dtype)
    data_l2_shrinkage = tvm.placeholder(shape_scalar, name="data_l2_shrinkage", dtype=compute_dtype)
    data_lr_power = tvm.placeholder(shape_scalar, name="data_lr_power", dtype=compute_dtype)

    ins = classify([var, accum, linear, grad], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_var, _accum, _linear, _grad) in ins:
        with tbe.compute():
            shape_var, shape_accum, shape_linear, shape_grad = \
                shape_util.variable_shape([_var, _accum, _linear, _grad])
            data_var = tvm.placeholder(shape_var, name="data_var", dtype=compute_dtype)
            data_accum = tvm.placeholder(shape_accum, name="data_accum", dtype=compute_dtype)
            data_linear = tvm.placeholder(shape_linear, name="data_linear", dtype=compute_dtype)
            data_grad = tvm.placeholder(shape_grad, name="data_grad", dtype=compute_dtype)

            res = apply_ftrl_v2_d_compute(data_var, data_accum, data_linear,
                                          data_grad, data_lr, data_l1, data_l2,
                                          data_l2_shrinkage, data_lr_power,
                                          var_out, accum_out, linear_out)

            tensor_list = [data_var, data_accum, data_linear, data_grad, data_lr,
                           data_l1, data_l2, data_l2_shrinkage, data_lr_power] + list(res)
            tensors.append(tensor_list)
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
