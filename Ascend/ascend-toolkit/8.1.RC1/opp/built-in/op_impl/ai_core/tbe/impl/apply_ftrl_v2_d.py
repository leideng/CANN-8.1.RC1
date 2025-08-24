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
apply_ftrl_v2_d
"""
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util import util_apply_op_schedule
from impl.util import util_build
from impl.util import util_compute


# 'pylint: disable=too-few-public-methods, not-use-list-comprehension
class Constant:
    """
    The class for constant
    """
    NUM_ONE = 1.0
    NUM_TWO = 2.0
    NUM_ZERO = 0.0
    NUM_M_ONE = -1.0


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

    if not tbe_platform.api_check_support("tbe.dsl.vlog", "float32"):
        data_x = tbe.cast_to(data_x, "float16")
    log_value = tbe.vlog(data_x, priority_flag=1)
    data_y = tbe.cast_to(data_y, log_value.dtype)
    mul_value = tbe.vmul(data_y, log_value)
    res = tbe.vexp(mul_value)

    return res


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=invalid-name,too-many-locals, too-many-statements
@register_operator_compute("apply_ftrl_v2_d", op_mode="static", support_fusion=True)
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
    if dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
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
        var_tmp = tbe.vadds(var, tvm.const(Constant.NUM_ZERO, "float32"))
        accum_tmp = tbe.vadds(accum, tvm.const(Constant.NUM_ZERO, "float32"))
        linear_tmp = tbe.vadds(linear, tvm.const(Constant.NUM_ZERO, "float32"))

    # 1.cal grad_with_shrinkage
    mul_value = tbe.vmuls(l2_shrinkage, tvm.const(Constant.NUM_TWO, "float32"))
    mul_value = tbe.broadcast(mul_value, var_tmp.shape)
    mul_value2 = tbe.vmul(mul_value, var_tmp)
    grad_with_shrinkage = tbe.vadd(grad, mul_value2)

    # 2.cal accum_new
    gs = tbe.vmul(grad, grad)
    accum_new = tbe.vadd(accum_tmp, gs)

    # 3.cal accum_pow_sub
    lr_power = tbe.vmuls(lr_power, tvm.const(Constant.NUM_M_ONE, "float32"))
    lr_power = tbe.broadcast(lr_power, var_tmp.shape)
    accum_new_pow = _pow(accum_new, lr_power)
    accum_pow = _pow(accum_tmp, lr_power)
    accum_pow_sub = tbe.vsub(accum_new_pow, accum_pow)

    # 4.cal linear
    lr = tbe.broadcast(lr, var_tmp.shape)
    accum_pow_div = tbe.vdiv(accum_pow_sub, lr)
    accum_pow_mul = tbe.vmul(accum_pow_div, var_tmp)
    accum_pow = tbe.vsub(grad_with_shrinkage, accum_pow_mul)
    linear_new = tbe.vadd(linear_tmp, accum_pow)

    # 5.cal x_res
    l1 = tbe.broadcast(l1, var_tmp.shape)
    x_res = util_compute.sign(linear_new)
    x_res = tbe.vmul(x_res, l1)
    x_res = tbe.vsub(x_res, linear_new)

    # 6.cal y_res
    l2 = tbe.vmuls(l2, tvm.const(Constant.NUM_TWO, "float32"))
    l2 = tbe.broadcast(l2, var_tmp.shape)
    y_res = tbe.vdiv(accum_new_pow, lr)
    y_res = tbe.vadd(y_res, l2)

    # 7.cal var
    x_res = tbe.vdiv(x_res, y_res)
    linear_abs = tbe.vabs(linear_new)
    zero_tensor = tbe.broadcast(tvm.const(Constant.NUM_ZERO, "float32"), var_tmp.shape)
    var_sel = tbe.vcmp(linear_abs, l1, 'gt')
    var_new = tbe.vsel(var_sel, x_res, zero_tensor)

    # dtype after vsel is float16 at mini
    var_new = tbe.cast_to(var_new, "float32")

    if has_improve_precision:
        var_new = tbe.cast_to(var_new, "float16")
        accum_new = tbe.cast_to(accum_new, "float16")
        linear_new = tbe.cast_to(linear_new, "float16")

    # 8.cal output_var
    output_data = tbe.vadds(var_new, tvm.const(Constant.NUM_ZERO, var_new.dtype))
    accum_out_data = tbe.vadds(
        accum_new, tvm.const(Constant.NUM_ZERO, accum_new.dtype))
    linear_out_data = tbe.vadds(
        linear_new, tvm.const(Constant.NUM_ZERO, linear_new.dtype))

    def _compute(*index):
        return [var_new(*index), accum_new(*index), \
               linear_new(*index), output_data(*index), \
               accum_out_data(*index), linear_out_data(*index)]

    return tvm.compute(var.shape, _compute, name="outputs")


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
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
    input_dict = (var, accum, linear, grad, lr, l1, l2, l2_shrinkage, lr_power)

    args = util_apply_op_schedule.ApplyOpConfig.TensorArgs(input_dict, apply_ftrl_v2_d_compute,
                                                           [var_out, accum_out, linear_out], 15)
    name = util_apply_op_schedule.ApplyOpConfig.TensorName(all=('var', 'accum', 'linear', 'grad',
                                                                'lr', 'l1', 'l2',
                                                                'l2_shrinkage', 'lr_power'),
                                                           scalar=('lr', 'l1', 'l2',
                                                                   'l2_shrinkage', 'lr_power'),
                                                           reuse=('var', 'accum', 'linear'))
    options = util_apply_op_schedule.ApplyOpConfig.TensorOptions(build=util_build.set_bool_storage_config())
    util_apply_op_schedule.common_apply_op_process(util_apply_op_schedule.ApplyOpConfig(args, name, options),
                                                   kernel_name)
