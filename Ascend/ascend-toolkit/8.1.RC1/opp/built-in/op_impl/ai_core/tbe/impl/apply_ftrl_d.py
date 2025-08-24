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
apply_ftrl_d
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


# 'pylint: disable=unused-argument,unused-variable
def _pow(data, index, bound):
    """
    Calculate power result for non-negative data.

    res = data^index
        = exp(index * ln(data)) if data > 0
        = 1                    if data = 0, index = 0
        = 0                   if data = 0, index is not 0

    Parameters:
    ----------
    data : base value of power operation.

    index: index value of power operation.

    bound : computation bound for data.

    Returns:
    -------
    power result of data^index
    """

    log_value = tbe.vlog(data)
    base = tbe.vmul(log_value, index)
    res = tbe.vexp(base)

    return res


# 'pylint: disable=locally-disabled,too-many-arguments
# 'pylint: disable=unused-argument,invalid-name,too-many-locals
@register_operator_compute("apply_ftrl_d", op_mode="static", support_fusion=True)
def apply_ftrl_d_compute(var,
                         accum,
                         linear,
                         grad,
                         lr,
                         l1,
                         l2,
                         lr_power,
                         var_out,
                         accum_out,
                         linear_out,
                         kernel_name='apply_ftrl_d'):
    """
    Update '*var' according to the Ftrl-proximal algorithm.

    accum_new = accum + grad * grad
    linear += grad - (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
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

    lr_power : scalar lr_power.

    var_out : the dict of var output.

    accum_out : the dict of accum output.

    linear_out : the dict of linear output.

    kernel_name : cce kernel name, default value is "apply_ftrl_d" (optional).

    Returns:
    -------
    None
    """

    # cast to float32 for higher accuracy
    dtype = var.dtype
    has_improve_precision = False
    if dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vexp", "float32"):
        var_tmp = tbe.cast_to(var, "float32")
        accum_tmp = tbe.cast_to(accum, "float32")
        linear_tmp = tbe.cast_to(linear, "float32")
        grad = tbe.cast_to(grad, "float32")
        lr = tbe.cast_to(lr, "float32")
        l1 = tbe.cast_to(l1, "float32")
        l2 = tbe.cast_to(l2, "float32")
        lr_power = tbe.cast_to(lr_power, "float32")
        has_improve_precision = True
    else:
        var_tmp = tbe.vadds(var, tvm.const(Constant.NUM_ZERO, dtype))
        accum_tmp = tbe.vadds(accum, tvm.const(Constant.NUM_ZERO, dtype))
        linear_tmp = tbe.vadds(linear, tvm.const(Constant.NUM_ZERO, dtype))

    # broadcast scalar to appropriate shape
    zero_tensor = tbe.broadcast(tvm.const(Constant.NUM_ZERO, var_tmp.dtype), var.shape)
    lr = tbe.broadcast(lr, var.shape)
    l1 = tbe.broadcast(l1, var.shape)
    l2 = tbe.broadcast(l2, var.shape)
    lr_power = tbe.broadcast(lr_power, var.shape)

    # 1.cal accum_new
    gs = tbe.vmul(grad, grad)
    accum_new = tbe.vadd(accum_tmp, gs)

    # 2.cal linear
    lr_power = tbe.vmuls(lr_power, tvm.const(Constant.NUM_M_ONE, var_tmp.dtype))
    accum_new_p = _pow(accum_new, lr_power, zero_tensor)
    accum_p = _pow(accum_tmp, lr_power, zero_tensor)
    accum_p = tbe.vsub(accum_new_p, accum_p)

    accum_p = tbe.vdiv(accum_p, lr)
    accum_p = tbe.vmul(accum_p, var_tmp)
    accum_p = tbe.vsub(grad, accum_p)
    linear_t = tbe.vadd(linear_tmp, accum_p)

    # 3.cal x_res
    x_res = util_compute.sign(linear_t)
    x_res = tbe.vmul(x_res, l1)
    x_res = tbe.vsub(x_res, linear_t)

    # 4.cal y_res
    l2 = tbe.vmuls(l2, tvm.const(Constant.NUM_TWO, var_tmp.dtype))
    y_res = tbe.vdiv(accum_new_p, lr)
    y_res = tbe.vadd(y_res, l2)

    # 5.cal var
    x_res = tbe.vdiv(x_res, y_res)
    linear_abs = tbe.vabs(linear_t)
    var_sel = tbe.vcmp(linear_abs, l1, 'gt')
    var_t = tbe.vsel(var_sel, x_res, zero_tensor)

    # result of vsel is fp16, should cast to fp32
    var_t = tbe.cast_to(var_t, "float32")

    if has_improve_precision:
        var_t = tbe.cast_to(var_t, "float16")
        accum_new = tbe.cast_to(accum_new, "float16")
        linear_t = tbe.cast_to(linear_t, "float16")

    # 8.cal var_output_data
    var_output_data = tbe.vadds(var_t, tvm.const(Constant.NUM_ZERO, var_t.dtype))
    accum_output_data = tbe.vadds(accum_new, tvm.const(Constant.NUM_ZERO, accum_new.dtype))
    linear_output_data = tbe.vadds(linear_t, tvm.const(Constant.NUM_ZERO, linear_t.dtype))

    def _compute(*index):
        return [var_t(*index), accum_new(*index), linear_t(*index), var_output_data(
            *index), accum_output_data(*index), linear_output_data(*index)]

    return tvm.compute(var.shape, _compute, name="outputs")


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def apply_ftrl_d(var,
                 accum,
                 linear,
                 grad,
                 lr,
                 l1,
                 l2,
                 lr_power,
                 var_out,
                 accum_out,
                 linear_out,
                 kernel_name="apply_ftrl_d"):
    """
    Update '*var' according to the Ftrl-proximal algorithm.
    accum_new = accum + grad * grad
    linear += grad - (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
    quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
    var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
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

    lr_power : the dict of scalar lr_power.
        Must have the same data type as `var`.

    var_out: the dict of var output data.

    accum_out: the dict of accum output data.

    linear_out: the dict of linear output data

    kernel_name : cce kernel name, default value is "apply_ftrl_d".

    Returns
    -------
    None
    """

    input_dict = (var, accum, linear, grad, lr, l1, l2, lr_power)
    out = [var_out, accum_out, linear_out]
    args = util_apply_op_schedule.ApplyOpConfig.TensorArgs(input_dict, apply_ftrl_d_compute, out, 15)
    name = util_apply_op_schedule.ApplyOpConfig.TensorName(all=('var', 'accum', 'linear', 'grad',
                                                                'lr', 'l1', 'l2', 'lr_power'),
                                                           scalar=('lr', 'l1', 'l2', 'lr_power'),
                                                           reuse=('var', 'accum', 'linear'))
    options = util_apply_op_schedule.ApplyOpConfig.TensorOptions(build=util_build.set_bool_storage_config())

    util_apply_op_schedule.common_apply_op_process(util_apply_op_schedule.ApplyOpConfig(args, name, options),
                                                   kernel_name)
