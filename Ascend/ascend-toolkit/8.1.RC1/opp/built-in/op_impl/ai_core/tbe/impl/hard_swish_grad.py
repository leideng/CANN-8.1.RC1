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
hard_swish_grad
"""
import functools
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import build
from impl.util.platform_adapter import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant.
    """
    CONST_THREE = 3.0
    CONST_HALF = 0.5


# 'pylint: disable=too-many-arguments,unused-argument
@register_operator_compute("hard_swish_grad", op_mode="static", support_fusion=True)
def hard_swish_grad_compute(input_grad, input_x, output_y, kernel_name="hard_swish_grad"):
    """
    Parameters
    ----------
    input_grad : TVM tensor
        the placeholder of input_grad
    input_x : TVM tensor
        the placeholder of input_x
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is "hard_swish_grad"

    Returns
    ------
    compute result of hard_swish_grad
    result = input_grad * input_x_grad
    input_x_grad = (relu6(x+3) + x*(relu6'(x+3)))/6
    input_x_grad = 0             if x < -3,
    input_x_grad = x / 3 + 0.5   if -3 <= x <= 3,
    input_x_grad = 1             if x > 3
    """
    input_dtype = input_x.dtype
    three_tensor = tbe.broadcast(tvm.const(Constant.CONST_THREE, input_dtype), input_x.shape)
    output_x = tbe.vdiv(input_x, three_tensor)
    output_x = tbe.vadds(output_x, tvm.const(Constant.CONST_HALF, input_dtype))
    output_x = tbe.vcmpsel(input_x, -Constant.CONST_THREE, 'lt', 0, output_x)
    output_x = tbe.vcmpsel(input_x, Constant.CONST_THREE, 'gt', 1, output_x)
    return tbe.vmul(input_grad, output_x)


# 'pylint: disable=too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def hard_swish_grad(input_grad, input_x, output_y, kernel_name="hard_swish_grad"):
    """
    Parameters
    ----------
    input_grad : dict
        shape and dtype of input_grad
    input_x : dict
        shape and dtype of input_x
    output_y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is "hard_swish_grad"

    Returns
    ------
    None
    """
    # check input shape
    shape_x = input_x.get("shape")
    shape_grad = input_grad.get("shape")
    para_check.check_shape(shape_x, param_name="input_x")
    para_check.check_shape(shape_grad, param_name="input_grad")
    if list(shape_x) != list(shape_grad):
        error_detail = "The shape of two input parameters are not match for dynamic hard_swish_grad."
        error_manager_vector.raise_err_two_input_shape_invalid(
            "hard_swish_grad", "input_x", "input_grad", error_detail)

    # check input tensor data_type and kernel_name
    check_list = ("float16", "float32")
    input_dtype = input_x.get("dtype").lower()
    grad_dtype = input_grad.get("dtype").lower()
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")
    if input_dtype != grad_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "input_x", "input_grad",
                                                              input_dtype, grad_dtype)

    shape_x = [functools.reduce(lambda x, y: x * y, shape_x[:])]
    input_data_orginal = tvm.placeholder(
        shape_x, name="input_data", dtype=input_dtype)
    input_grad = tvm.placeholder(shape_x, name="input_grad", dtype=grad_dtype)

    final_res = hard_swish_grad_compute(
        input_grad, input_data_orginal, output_y, kernel_name="hard_swish_grad")
    with tvm.target.cce():
        auto_sch = auto_schedule(final_res)

    config = {"name": kernel_name, "tensor_list": (
        input_grad, input_data_orginal, final_res)}

    build(auto_sch, config)
