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
apply_adagrad_da_d
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpImplMode
from impl.util import util_apply_op_schedule
from impl.util import util_build
from impl.util import util_compute


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class Const:
    """
    The class for constant.
    """
    # scalar in apply_adagrad_da_d
    NUM_ZERO = 0.0
    NUM_M_ONE = -1.0


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=invalid-name,too-many-locals
# 'pylint: disable=too-many-statements,wrong-import-order,ungrouped-imports
@register_operator_compute("ApplyAdagradDAD", op_mode="static", support_fusion=True)
def apply_adagrad_da_d_compute(var, gradient_accumulator,
                               gradient_squared_accumulator, grad,
                               lr, l1, l2, global_step, var_out,
                               gradient_accumulator_out,
                               gradient_squared_accumulator_out,
                               kernel_name='apply_adagrad_da_d'):
    """
    Update '*var' according to the Ftrl-proximal algorithm.

    grad_accum += grad
    grad_squared_accum += grad * grad
    tmp_val=sign(grad_accum) * max⁡{|grad_accum|-l1*global_step, 0}
        if l1>0 else grad_accum
    x_value = -1 * lr * tmp_val
    y_value = l2 * global_step * lr + sqrt(grad_squared_accum)
    var = x_value / y_value

    Parameters:
    ----------
    var : mutable tensor var.

    gradient_accumulator: mutable tensor gradient_accumulator.

    gradient_squared_accumulator : mutable tensor gradient_squared_accumulator.

    grad : tensor grad.

    lr : scalar lr.

    l1 : scalar l1.

    l2 : scalar l2.

    global_step : scalar global_step.

    var_out : the dict of output.

    gradient_accumulator_out : the dict of output.

    gradient_squared_accumulator_out : the dict of output.

    kernel_name : cce kernel name, default value is "apply_adagrad_da".

    Returns:
    -------
    None
    """
    # cast to float32 for higher accuracy
    dtype = var.dtype
    has_improve_precision = False
    cast_type = var.dtype
    if dtype == "float16" and \
            tbe_platform.api_check_support("tbe.dsl.vsqrt",
                                                    "float32"):
        cast_type = "float32"
        has_improve_precision = True

    if dtype == "float16":
        if has_improve_precision:
            var_tmp = tbe.cast_to(var, "float32")
            var_tmp = tbe.vmuls(var_tmp, tvm.const(Const.NUM_ZERO, "float32"))
            grad_accum_tmp = tbe.cast_to(gradient_accumulator, "float32")
            grad_sq_accum_tmp = tbe.cast_to(
                gradient_squared_accumulator, "float32")
            grad = tbe.cast_to(grad, "float32")
            lr = tbe.cast_to(lr, "float32")
            l1 = tbe.cast_to(l1, "float32")
            l2 = tbe.cast_to(l2, "float32")
        else:
            var_tmp = tbe.vmuls(var, tvm.const(Const.NUM_ZERO, "float16"))
            grad_accum_tmp = tbe.vadds(gradient_accumulator, tvm.const(Const.NUM_ZERO, "float16"))
            grad_sq_accum_tmp = tbe.vadds(gradient_squared_accumulator, tvm.const(Const.NUM_ZERO, "float16"))
    else:
        var_tmp = tbe.vmuls(var, tvm.const(Const.NUM_ZERO, "float32"))
        grad_accum_tmp = tbe.vadds(gradient_accumulator, tvm.const(Const.NUM_ZERO, "float32"))
        grad_sq_accum_tmp = tbe.vadds(gradient_squared_accumulator, tvm.const(Const.NUM_ZERO, "float32"))

    global_step = tbe.cast_to(global_step, cast_type)

    # 1.grad_accum += grad
    gradient_accum_new = tbe.vadd(grad_accum_tmp, grad)

    # 2.grad_squared_accum += grad * grad
    gs = tbe.vmul(grad, grad)
    gradient_squared_accum_new = tbe.vadd(grad_sq_accum_tmp, gs)

    # 3.tmp_val = sign(grad_accum) * max⁡{|grad_accum|-l1*global_step, 0}
    #     if l1>0 else grad_accum
    sign_val = util_compute.sign(gradient_accum_new)
    abs_val = tbe.vabs(gradient_accum_new)

    mul_val = tbe.vmul(global_step, l1)
    mul_val = tbe.broadcast(mul_val, var_tmp.shape)
    sub_val = tbe.vsub(abs_val, mul_val)
    zero_tensor = tbe.broadcast(tvm.const(Const.NUM_ZERO, cast_type), var_tmp.shape)
    max_val = tbe.vmax(sub_val, zero_tensor)
    tmp_val = tbe.vmul(sign_val, max_val)

    l1 = tbe.broadcast(l1, var_tmp.shape)
    l1_cmp = tbe.vcmp(l1, zero_tensor, "gt")
    tmp_val = tbe.vsel(l1_cmp, tmp_val, gradient_accum_new)

    # 4.x_value = -1 * lr * tmp_val
    x_value = tbe.vmuls(lr, tvm.const(Const.NUM_M_ONE, cast_type))
    x_value = tbe.broadcast(x_value, var_tmp.shape)
    x_value = tbe.vmul(x_value, tmp_val)

    # 5.y_value = l2 * global_step * lr + sqrt(grad_squared_accum)
    pro_val = tbe.vmul(l2, global_step)
    pro_val = tbe.vmul(pro_val, lr)
    pro_val = tbe.broadcast(pro_val, var_tmp.shape)
    sqrt_val = tbe.vsqrt(gradient_squared_accum_new, impl_mode = OpImplMode.HIGH_PRECISION)
    y_value = tbe.vadd(pro_val, sqrt_val)

    # 6.var = x_value / y_value
    var_t = tbe.vdiv(x_value, y_value)
    var_new = tbe.vadd(var_t, var_tmp)

    if dtype == "float16" and has_improve_precision:
        var_new = tbe.cast_to(var_new, "float16")
        gradient_accum_new = tbe.cast_to(
            gradient_accum_new, "float16")
        gradient_squared_accum_new = tbe.cast_to(
            gradient_squared_accum_new, "float16")

    # 7. output_data = var_new
    output_data = tbe.vadds(var_new, tvm.const(Const.NUM_ZERO, var_new.dtype))
    res1_data = tbe.vadds(gradient_accum_new, tvm.const(Const.NUM_ZERO, var_new.dtype))
    res2_data = tbe.vadds(gradient_squared_accum_new, tvm.const(Const.NUM_ZERO, var_new.dtype))

    def _compute(*index):
        return [var_new(*index), gradient_accum_new(*index), \
               gradient_squared_accum_new(*index), output_data(*index),\
               res1_data(*index), res2_data(*index)]

    return tvm.compute(var.shape, _compute, name="outputs")


# 'pylint: disable=locally-disabled,unused-argument,too-many-arguments
# 'pylint: disable=too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def apply_adagrad_da_d(var, gradient_accumulator,
                       gradient_squared_accumulator, grad,
                       lr, l1, l2, global_step, var_out,
                       gradient_accumulator_out,
                       gradient_squared_accumulator_out,
                       use_locking=False, kernel_name='apply_adagrad_da_d'):
    """
    Update '*var' according to the Ftrl-proximal algorithm.

    grad_accum += grad
    grad_squared_accum += grad * grad
    tmp_val=sign(grad_accum) * max⁡{|grad_accum|-l1*global_step, 0}
        if l1>0 else grad_accum
    x_value = -1 * lr * tmp_val
    y_value = l2 * global_step * lr + sqrt(grad_squared_accum)
    var = x_value / y_value

    Parameters:
    ----------
    var : the dict of mutable tensor var, only support float16, float32

    gradient_accumulator:
        the dict of mutable tensor gradient_accumulator,
        Must have the same data type as `var`.

    gradient_squared_accumulator :
        the dict of mutable tensor gradient_squared_accumulator,
        Must have the same data type as `var`.

    grad : the dict of tensor grad. Must have the same data type as `var`.

    lr : the dict of scalar lr. Must have the same data type as `var`.

    l1 : the dict of scalar l1. Must have the same data type as `var`.

    l2 : the dict of scalar l2. Must have the same data type as `var`.

    global_step : the dict of scalar global_step, only support int32.

    var_out : the dict of output.

    gradient_accumulator_out : the dict of output.

    gradient_squared_accumulator_out : the dict of output.

    use_locking : optional attr, default value is False.

    kernel_name : cce kernel name, default value is "apply_adagrad_da".

    Returns:
    -------
    None
    """
    # check dtype same
    stype_dict = (var, gradient_accumulator, gradient_squared_accumulator, grad, lr, l1, l2)
    normalized_dtype_list = [None] * len(stype_dict)
    for i, d in enumerate(stype_dict):
        dtype = d.get('dtype')
        normalized_dtype_list[i] = dtype.lower()
    if any(elem != normalized_dtype_list[0] for elem in normalized_dtype_list):
        rule_desc = "All input data types must be the same"
        param_value = "%s,%s,%s,%s,%s,%s,%s" % (var.get('dtype'), gradient_accumulator.get('dtype'),
                                              gradient_squared_accumulator.get('dtype'), grad.get('dtype'),
                                              lr.get('dtype'), l1.get('dtype'), l2.get('dtype'))
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule_desc,
            "var,gradient_accumulator,gradient_squared_accumulator,grad,lr,l1,l2", param_value)

    # check global_step dtype
    dtype = global_step.get("dtype").lower()
    para_check.check_dtype(dtype, ("int32",), param_name="global_step")

    input_dict = (var, gradient_accumulator, gradient_squared_accumulator, grad,
                  lr, l1, l2, global_step)
    args = util_apply_op_schedule.ApplyOpConfig.TensorArgs(input_dict, apply_adagrad_da_d_compute,
                                                           [var_out, gradient_accumulator_out,
                                                           gradient_squared_accumulator_out], 15)
    name = util_apply_op_schedule.ApplyOpConfig.TensorName(all=('var', 'gradient_accumulator',
                                                                'gradient_squared_accumulator', 'grad',
                                                                'lr', 'l1', 'l2', 'global_step'),
                                                           scalar=('lr', 'l1', 'l2', 'global_step'),
                                                           reuse=('var', 'gradient_accumulator',
                                                                  'gradient_squared_accumulator'))
    options = util_apply_op_schedule.ApplyOpConfig.TensorOptions(
        build=util_build.set_bool_storage_config(),
        dtype=('float16', 'float32', 'int32'))
    util_apply_op_schedule.common_apply_op_process(util_apply_op_schedule.ApplyOpConfig(args, name, options),
                                                   kernel_name, same_flag=False)
