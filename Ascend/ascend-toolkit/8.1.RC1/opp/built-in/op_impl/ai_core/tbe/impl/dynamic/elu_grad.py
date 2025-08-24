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
dynamic elu_grad

Op_description :
do element-wise elu operation.

# elu_grad(
#   grads,
#   activations,
#   y,
#   kernel_name='cce_elu_grad')

Supportive_dtype_format :
["float16", "float32"]
['ND', 'NCHW', 'NHWC', 'NC1HWC0']

Constraint :
[1] All : `grads` and `activations` must have the same shape and type.
[2] All : shape size limit is 2147483648.
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpPatternMode
from impl.util.util_soc_common import after_v200


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name,too-many-locals
@register_operator_compute("EluGrad", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def elu_grad_compute(grads, activations, y, kernel_name="elu_grad"):
    """
    elu_grad_compute
    f(x) = vmul(add(min(activation, 0), 1), gradient)

    Parameters:
    ----------
    data_gradient : the placeholder of gradient data

    data_activation : the placeholder of activation data

    data_output : the dict of output

    kernel_name : cce kernel name, default value is "elu_grad"

    Returns : A Tensor. Has the same type as data_gradient.
    -------
    """

    num_zero = 0.0
    num_one = 1.0
    dtype = grads.dtype.lower()
    dtype_present = grads.dtype.lower()
    shape = grads.shape

    if dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vadd", "float32"):
        dtype_present = "float32"
        grads = tbe.cast_to(grads, dtype_present)
        activations = tbe.cast_to(activations, dtype_present)

    if after_v200() and dtype_present == "float32":
        add_res = tbe.vadds(activations, num_one)
        mask_temp = tbe.vcmp(activations, num_zero, "le", "bit")
        res_mask = tbe.vsel(mask_temp, add_res, num_one)
        res = tbe.vmul(res_mask, grads)
    elif tbe_platform.api_check_support("tbe.dsl.vmins", "float32"):
        min_res = tbe.vmins(activations, num_zero)
        add_res = tbe.vadds(min_res, num_one)
        res = tbe.vmul(add_res, grads)
    else:
        input_border = tvm.const(num_zero, grads.dtype)
        scalar_param_one = tvm.const(num_one, grads.dtype)
        tensor_input_border = tbe.broadcast(input_border, shape)
        tensor_scalar_param_one = tbe.broadcast(scalar_param_one, shape)

        min_res = tbe.vmin(activations, tensor_input_border)
        add_res = tbe.vadd(min_res, tensor_scalar_param_one)
        res = tbe.vmul(add_res, grads)

    if dtype != dtype_present:
        res = tbe.cast_to(res, dtype)

    return res


# 'pylint: disable=invalid-name
@register_operator("EluGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def elu_grad(grads, activations, y, kernel_name="elu_grad"):
    """
    do element-wise elu_grad operation

    Parameters:
    ----------
    grads: the dict of gradient input, only support bfloat16, float16, float32

    activations: the dict of activation input, only support float16, float32
                dtype of grads and activations should be same

    y : the dict of output

    kernel_name : cce kernel name, default value is "cce_elu_grad"

    Returns
    -------
    None
    """
    dtype_gradient = grads.get("dtype").lower()
    dtype_activation = activations.get("dtype").lower()
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(dtype_gradient, check_list, param_name="grads")
    para_check.check_dtype(dtype_activation, check_list, param_name="activations")

    if dtype_gradient != dtype_activation:
        error_detail = "dtype of grads and activations should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "grads", "activations", error_detail)

    ins = classify([grads, activations], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_grads, _activations) in ins:
        with tbe.compute():
            # shape
            shape_x1, shape_x2 = shape_util.variable_shape([_grads, _activations])
            # mul_compute
            data_x1 = tvm.placeholder(shape_x1, dtype=dtype_gradient, name="data_x1")
            data_x2 = tvm.placeholder(shape_x2, dtype=dtype_activation, name="data_x2")
            res = elu_grad_compute(data_x1, data_x2, y, kernel_name)
            tensors.append((data_x1, data_x2, res))
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

        # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
