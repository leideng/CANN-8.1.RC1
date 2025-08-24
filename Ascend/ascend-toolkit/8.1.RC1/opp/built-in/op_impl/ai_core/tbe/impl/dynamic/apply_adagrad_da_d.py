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
# 'pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name,too-many-arguments
"""
apply_adagrad_da_d
"""
from impl.util import util_compute
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import OpTbeImplMode


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # scalar in apply_adagrad_da_d
    NUM_ZERO = 0.0
    NUM_M_ONE = -1.0

# 'pylint: disable=too-many-statements
@register_operator_compute("ApplyAdagradDAD", op_mode="dynamic", support_fusion=True)
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
    tmp_val=sign(grad_accum) * max(| grad_accum | -l1 * global_step, 0)
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
            var_tmp = tbe.vmuls(var_tmp, tvm.const(Constant.NUM_ZERO, "float32"))
            grad_accum_tmp = tbe.cast_to(gradient_accumulator, "float32")
            grad_sq_accum_tmp = tbe.cast_to(
                gradient_squared_accumulator, "float32")
            grad = tbe.cast_to(grad, "float32")
            lr = tbe.cast_to(lr, "float32")
            l1 = tbe.cast_to(l1, "float32")
            l2 = tbe.cast_to(l2, "float32")
        else:
            var_tmp = tbe.vmuls(var, tvm.const(Constant.NUM_ZERO, "float16"))
            grad_accum_tmp = tbe.vadds(gradient_accumulator, tvm.const(Constant.NUM_ZERO, "float16"))
            grad_sq_accum_tmp = tbe.vadds(gradient_squared_accumulator, tvm.const(Constant.NUM_ZERO, "float16"))
    else:
        var_tmp = tbe.vmuls(var, tvm.const(Constant.NUM_ZERO, "float32"))
        grad_accum_tmp = tbe.vadds(gradient_accumulator, tvm.const(Constant.NUM_ZERO, "float32"))
        grad_sq_accum_tmp = tbe.vadds(gradient_squared_accumulator, tvm.const(Constant.NUM_ZERO, "float32"))

    global_step = tbe.cast_to(global_step, cast_type)

    gradient_accum_new = tbe.vadd(grad_accum_tmp, grad)

    gs = tbe.vmul(grad, grad)
    gradient_squared_accum_new = tbe.vadd(grad_sq_accum_tmp, gs)

    # tmp_val value is sign(grad_accum) * max(| grad_accum | -l1 * global_step, 0)
    sign_val = util_compute.sign(gradient_accum_new)
    abs_val = tbe.vabs(gradient_accum_new)

    mul_val = tbe.vmul(global_step, l1)
    mul_val = tbe.vmuls(mul_val, tvm.const(Constant.NUM_M_ONE, cast_type))
    sub_val = tbe.vadds(abs_val, mul_val[0])

    zero_tensor = tbe.broadcast(tvm.const(Constant.NUM_ZERO, cast_type), var_tmp.shape)
    max_val = tbe.vmax(sub_val, zero_tensor)
    tmp_val = tbe.vmul(sign_val, max_val)

    l1 = tbe.broadcast(l1, var_tmp.shape)
    tmp_val = tbe.vcmpsel(l1, zero_tensor, "gt", tmp_val, gradient_accum_new)

    # x_value is -1 * lr * tmp_val
    x_value = tbe.vmuls(lr, tvm.const(Constant.NUM_M_ONE, cast_type))
    x_value = tbe.vmuls(tmp_val, x_value[0])

    # `y_value is l2 * global_step * lr + sqrt(grad_squared_accum)`
    pro_val = tbe.vmul(l2, global_step)
    pro_val = tbe.vmul(pro_val, lr)

    sqrt_val = tbe.vsqrt(gradient_squared_accum_new, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
    y_value = tbe.vadds(sqrt_val, pro_val[0])

    # var  value is x_value / y_value
    var_t = tbe.vdiv(x_value, y_value)
    var_new = tbe.vadd(var_t, var_tmp)

    if dtype == "float16" and has_improve_precision:
        var_new = tbe.cast_to(var_new, "float16")
        gradient_accum_new = tbe.cast_to(
            gradient_accum_new, "float16")
        gradient_squared_accum_new = tbe.cast_to(
            gradient_squared_accum_new, "float16")

    # output_data value is  var_new
    output_data = tbe.vadds(var_new, tvm.const(Constant.NUM_ZERO, var_new.dtype))
    res1_data = tbe.vadds(gradient_accum_new, tvm.const(Constant.NUM_ZERO, var_new.dtype))
    res2_data = tbe.vadds(gradient_squared_accum_new, tvm.const(Constant.NUM_ZERO, var_new.dtype))

    res = [output_data, res1_data, res2_data]
    return res


@register_operator("ApplyAdagradDAD")
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
    tmp_val=sign(grad_accum) * max(| grad_accum | -l1 * global_step, 0)
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

    check_list = ('float16', 'float32')
    var_dtype = var.get("dtype").lower()
    para_check.check_dtype(var_dtype, check_list, param_name="var")

    global_step_dtype = global_step.get("dtype").lower()
    para_check.check_dtype(global_step_dtype, ("int32",), param_name="global_step")

    shape_util.compare_tensor_dict_key(var, gradient_accumulator, "dtype")
    shape_util.compare_tensor_dict_key(var, gradient_squared_accumulator, "dtype")
    shape_util.compare_tensor_dict_key(var, grad, "dtype")

    ins = classify([var, gradient_accumulator, gradient_squared_accumulator, grad], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (data1, data2, data3, data4) in ins:
        with tbe.compute():
            var_shape, gradient_accumulator_shape, gradient_squared_accumulator_shape, grad_shape = \
                    shape_util.variable_shape([data1, data2, data3, data4])

            var_data = tvm.placeholder(var_shape, dtype=var_dtype, name="var")

            gradient_accumulator_data = \
                tvm.placeholder(gradient_accumulator_shape, dtype=var_dtype,
                                name="gradient_accumulator_data")

            gradient_squared_accumulator_data = \
                tvm.placeholder(gradient_squared_accumulator_shape, dtype=var_dtype,
                                name="gradient_squared_accumulator_data")

            grad_data = tvm.placeholder(grad_shape, dtype=var_dtype, name="grad_data")

            global_step_data = tvm.placeholder([1], dtype=global_step_dtype, name="global_step_data")
            lr_data = tvm.placeholder([1], dtype=var_dtype, name="lr_data")
            l1_data = tvm.placeholder([1], dtype=var_dtype, name="l1_data")
            l2_data = tvm.placeholder([1], dtype=var_dtype, name="l2_data")



            res = apply_adagrad_da_d_compute(var_data, gradient_accumulator_data,
                                             gradient_squared_accumulator_data, grad_data,
                                             lr_data, l1_data, l2_data, global_step_data, var_out,
                                             gradient_accumulator_out,
                                             gradient_squared_accumulator_out,
                                             kernel_name)

        tensors.append([var_data, gradient_accumulator_data, gradient_squared_accumulator_data,
                        grad_data, lr_data, l1_data, l2_data, global_step_data] + list(res))

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
