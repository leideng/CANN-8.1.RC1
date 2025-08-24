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
apply_adagrad_d
"""
import te.lang.cce as tbe
from impl.util.platform_adapter import auto_schedule
from impl.util.platform_adapter import build
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class Const:
    """
    The class for constant.
    """
    NUM_ZERO = 0.0


# 'pylint: disable=too-many-locals, too-many-arguments
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
    dtype = var.get('dtype')
    para_check.check_dtype(dtype, check_list, param_name="var")
    var_shape = var.get("shape")
    lr_shape = lr.get("shape")
    var_dtype = var.get("dtype")
    if len(lr_shape) != 1 or int(lr_shape[0]) != 1:
        param_name = 'len(lr_shape), int(lr_shape[0])'
        error_manager_vector.raise_err_input_value_invalid("apply_adagrad_d",
            param_name, "1", len(lr_shape) + "," + int(lr_shape[0]))

    shape_util.compare_tensor_dict_key(var, accum, "shape")
    shape_util.compare_tensor_dict_key(accum, grad, "shape")
    shape_util.compare_tensor_dict_key(var, lr, "dtype")
    shape_util.compare_tensor_dict_key(accum, grad, "dtype")

    shape_list = shape_util.broadcast_shapes(var_shape, lr_shape, param_name_input1="input_x",
                                             param_name_input2="input_y")
    reshape_x, reshape_y = shape_util.refine_shapes_for_broadcast(shape_list[0], shape_list[1])

    var_data = tvm.placeholder(reshape_x, dtype=var_dtype, name="var")
    accum_data = tvm.placeholder(reshape_x, dtype=var_dtype, name="accum_data")
    lr_data = tvm.placeholder(reshape_y, dtype=var_dtype, name="lr_data")
    grad_data = tvm.placeholder(reshape_x, dtype=var_dtype, name="grad_data")
    res = apply_adagrad_d_compute(var_data, accum_data, lr_data, grad_data, var_out,
                                  accum_out, update_slots, kernel_name)
    inputlist = [var_data, accum_data, lr_data, grad_data]
    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": list(inputlist) + list(res)}

    build(sch, config)


# 'pylint: disable=locally-disabled, too-many-arguments, unused-argument
# 'pylint: disable=too-many-locals, invalid-name
@register_operator_compute("apply_adagrad_d", op_mode="static", support_fusion=True)
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
    shape_list = shape_util.broadcast_shapes(
        shape_util.shape_to_list(var.shape),
        shape_util.shape_to_list(lr.shape),
        param_name_input1="input_var", param_name_input2="input_lr")

    input_dtype = var.dtype

    if input_dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vadd", "float32"):
        var = tbe.cast_to(var, "float32")
        accum = tbe.cast_to(accum, "float32")
        lr = tbe.cast_to(lr, "float32")
        grad = tbe.cast_to(grad, "float32")
    lr = tbe.broadcast(lr, shape_list[2])

    if update_slots is True:
        grad_square = tbe.vmul(grad, grad)
        accum = tbe.vadd(accum, grad_square)
    elif input_dtype == 'float32':
        accum = tbe.vadds(accum, tvm.const(Const.NUM_ZERO, "float32"))
    lr_grad = tbe.vmul(grad, lr)
    sqrtdata = tbe.vsqrt(accum)
    update = tbe.vdiv(lr_grad, sqrtdata)
    var = tbe.vsub(var, update)
    res1 = tbe.vadds(var, tvm.const(0.0, dtype="float32"))
    res2 = tbe.vadds(accum, tvm.const(0.0, dtype="float32"))
    if input_dtype == "float16":
        res1 = tbe.cast_to(res1, "float16")
        res2 = tbe.cast_to(res2, "float16")

    return [res1, res2]
