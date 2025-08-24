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
# 'pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name
"""
this file achieved the apply_adagrad_d which is a optimizer operator
to update weight, this file contains compute and schedule.

apply_adagrad_d

Op_description :
Update '*var' according to the Adagrad algorithm.

# apply_adagrad_d(var,
#   accum,
#   lr,
#   grad,
#   var_out,
#   accum_out,
#   update_slots,
#   kernel_name='apply_adagrad_d')

Supportive_dtype_format :
['int32', 'int8', 'uint8', 'float32', 'float16']
['ND', 'NCHW', 'NHWC', 'NC1HWC0']

Constraint :
[1] All : the input tensors must have the same shape and type.
[2] All : shape size limit is 2147483648.
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpPatternMode


# 'pylint: disable=too-many-arguments
@register_operator_compute("ApplyAdagradD", op_mode="dynamic", support_fusion=True)
def apply_adagrad_d_compute(var,
                            accum,
                            lr,
                            grad,
                            var_out,
                            accum_out,
                            update_slots,
                            kernel_name="apply_adagrad_d"):
    """
    Update '*var' according to the Adagrad algorithm.

    accum += grad ** 2
    var -= lr * grad / accum.sqrt()

    Parameters:
    ----------
    var: the dict of var, only support float16, float32

    accum: the dict of accum, only support float16, float32

    lr: the dict of lr, only support float16, float32

    grad: the dict of grad, only support float16, float32

    var_out: the dict of var output, only support float16, float32

    accum_out: the dict of accum output, only support float16, float32

    update_slots: An optional 'bool'. Defaults to 'True',
        If True, the accum tensor will be updated;
        otherwise the option is False, the accum tensor will not be update.

    kernel_name : cce kernel name, default value is "apply_adagrad".

    Returns
    -------
    None
    """
    num_zero = 0.0
    input_dtype = var.dtype

    if input_dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vadd", "float32"):
        var = tbe.cast_to(var, "float32")
        accum = tbe.cast_to(accum, "float32")
        lr = tbe.cast_to(lr, "float32")
        grad = tbe.cast_to(grad, "float32")
    beta1_lr = lr[0]

    if update_slots is True:
        grad_square = tbe.vmul(grad, grad)
        accum = tbe.vadd(accum, grad_square)
    elif input_dtype == 'float32':
        accum = tbe.vadds(accum, tvm.const(num_zero, "float32"))
    lr_grad = tbe.vmuls(grad, beta1_lr)
    sqrtdata = tbe.vsqrt(accum)
    update = tbe.vdiv(lr_grad, sqrtdata)
    var = tbe.vsub(var, update)
    res1 = tbe.vadds(var, tvm.const(0.0, dtype="float32"))
    res2 = tbe.vadds(accum, tvm.const(0.0, dtype="float32"))
    if input_dtype == "float16":
        res1 = tbe.cast_to(res1, "float16")
        res2 = tbe.cast_to(res2, "float16")

    return [res1, res2]


# 'pylint: disable=too-many-arguments
@register_operator("ApplyAdagradD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def apply_adagrad_d(var,
                    accum,
                    lr,
                    grad,
                    var_out,
                    accum_out,
                    update_slots=True,
                    kernel_name="apply_adagrad_d"):
    """
    Update '*var' according to the Adagrad algorithm.

    accum += grad ** 2
    var -= lr * grad / accum.sqrt()

    Parameters:
    ----------
    var: the dict of var, only support float16, float32

    accum: the dict of accum, only support float16, float32

    lr: the dict of lr, only support float16, float32

    grad: the dict of grad, only support float16, float32

    var_out: the dict of var output, only support float16, float32

    accum_out: the dict of accum output, only support float16, float32

    update_slots: An optional 'bool'. Defaults to 'True',
        If True, the accum tensor will be updated;
        otherwise the option is False, the accum tensor will not be update.

    kernel_name : cce kernel name, default value is "apply_adagrad".

    Returns
    -------
    None
    """

    check_list = ('float16', 'float32')
    var_dtype = var.get("dtype").lower()
    para_check.check_dtype(var_dtype, check_list, param_name="var")

    shape_util.compare_tensor_dict_key(var, lr, "dtype")
    shape_util.compare_tensor_dict_key(accum, grad, "dtype")

    ins = classify([var, accum, grad], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_var, _accum, _grad) in ins:
        with tbe.compute():
            var_shape, accum_shape, grad_shape = shape_util.variable_shape([_var, _accum, _grad])

            var_data = tvm.placeholder(var_shape, dtype=var_dtype, name="var")

            accum_data = tvm.placeholder(accum_shape, dtype=var_dtype, name="accum_data")

            lr_data = tvm.placeholder([1], dtype=var_dtype, name="lr_data")

            grad_data = tvm.placeholder(grad_shape, dtype=var_dtype, name="grad_data")

            res = apply_adagrad_d_compute(var_data, accum_data, lr_data, grad_data, var_out,
                                          accum_out, update_slots, kernel_name)

        tensors.append([var_data, accum_data, lr_data, grad_data] + list(res))
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
