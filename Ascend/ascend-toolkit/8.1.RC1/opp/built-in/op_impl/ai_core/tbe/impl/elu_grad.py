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
elu_grad

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
import operator

import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name,too-many-locals
@register_operator_compute("elu_grad", op_mode="static", support_fusion=True)
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

    dtype = grads.dtype
    shape = grads.shape

    if dtype.lower() == "float16" and \
            tbe_platform.api_check_support("tbe.dsl.vadd", "float32"):
        grads = tbe.cast_to(grads, "float32")
        activations = tbe.cast_to(activations, "float32")

    if tbe_platform.api_check_support("tbe.dsl.vmins", "float32"):
        min_res = tbe.vmins(activations, 0.0)
        add_res = tbe.vadds(min_res, 1.0)
        res = tbe.vmul(add_res, grads)
    else:
        input_border = tvm.const(0.0, grads.dtype)
        scalar_param_one = tvm.const(1.0, grads.dtype)
        tensor_input_border = tbe.broadcast(input_border, shape)
        tensor_scalar_param_one = tbe.broadcast(scalar_param_one, shape)

        min_res = tbe.vmin(activations, tensor_input_border)
        add_res = tbe.vadd(min_res, tensor_scalar_param_one)
        res = tbe.vmul(add_res, grads)

    if dtype.lower() == "float16":
        res = tbe.cast_to(res, "float16")

    return res


# 'pylint: disable=invalid-name
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def elu_grad(grads, activations, y, kernel_name="elu_grad"):
    """
    do element-wise elu_grad operation

    Parameters:
    ----------
    grads: the dict of gradient input, only support float16, float32

    activations: the dict of activation input, only support float16, float32

    y : the dict of output

    kernel_name : cce kernel name, default value is "cce_elu_grad"

    Returns
    -------
    None
    """

    shape_gradient = grads.get("shape")
    shape_activation = activations.get("shape")
    dtype_gradient = grads.get("dtype")
    dtype_activation = activations.get("dtype")


    para_check.check_shape(shape_gradient, param_name="grads")
    para_check.check_shape(shape_activation, param_name="activations")
    if not operator.eq(shape_gradient, shape_activation):
        error_detail = "shape of grads and activations should be same"
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "grads", "activations", error_detail)
    shape_gradient, _ = shape_util.refine_shape_axes(shape_gradient, [])
    shape_activation, _ = shape_util.refine_shape_axes(shape_activation, [])

    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_gradient, check_list, param_name="grads")
    para_check.check_dtype(dtype_activation, check_list, param_name="activations")
    if dtype_gradient.lower() != dtype_activation.lower():
        error_detail = "dtype of grads and activations should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "grads", "activations", error_detail)

    dtype = dtype_gradient.lower()
    data_gradient = tvm.placeholder(shape_gradient, dtype=dtype, name="data_gradient")
    data_activation = tvm.placeholder(shape_activation, dtype=dtype, name="data_activation")
    res = elu_grad_compute(data_gradient, data_activation, y, kernel_name)

    with tvm.target.cce():
        auto_sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "print_ir": False,
              "tensor_list": [data_gradient, data_activation, res]}
    tbe.cce_build_code(auto_sch, config)
