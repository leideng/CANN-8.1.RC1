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
dynamic hard_swish_grad
"""
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=invalid-name,too-many-locals,redefined-argument-from-local
@register_operator_compute("HardSwishGrad", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def hard_swish_grad_compute(input_grad, input_x, output_y, kernel_name="hardswish_grad"):
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
    const_half = 0.5
    const_one_in_three = 0.33333334
    const_three = 3.0
    input_dtype = input_x.dtype
    if input_dtype == "float16":
        input_x = tbe.cast_to(input_x, "float32")
        input_grad = tbe.cast_to(input_grad, "float32")
    nan_res = tbe.vcmp(input_x, input_x, 'eq')
    output_x = tbe.vmuls(input_x, tvm.const(const_one_in_three, "float32"))
    output_x = tbe.vadds(output_x, tvm.const(const_half, "float32"))
    output_x = tbe.vcmpsel(input_x, -const_three, 'lt', 0, output_x)
    output_x = tbe.vcmpsel(input_x, const_three, 'gt', 1, output_x)
    if input_dtype == "float16":
        res = tbe.vmul(input_grad, output_x)
        return tbe.cast_to(res, input_dtype)
    result = tbe.vmul(input_grad, output_x)
    result = tbe.vsel(nan_res, result, tvm.const(1, input_dtype))
    return result


@register_operator("HardSwishGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
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
    # check input shape range
    para_check.check_elewise_shape_range([input_x, input_grad], support_broadcast=False)

    # check input tensor data_type and kernel_name
    check_list = ("float16", "float32", "bfloat16")
    input_dtype = input_x.get("dtype").lower()
    grad_dtype = input_grad.get("dtype").lower()
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")
    if input_dtype != grad_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "input_x", "input_grad",
                                                              input_dtype, grad_dtype)
    ins = classify([input_grad, input_x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (dx, sig) in ins:
        with tbe.compute():
            shape_dx, shape_sig = shape_util.variable_shape([dx, sig])
            tensor_sig = tvm.placeholder(shape_sig, input_dtype, "tensor_x")
            tensor_dx = tvm.placeholder(shape_dx, grad_dtype, "tensor_dx")
            res = hard_swish_grad_compute(tensor_dx, tensor_sig, output_y, kernel_name)
            tensors.append([tensor_dx, tensor_sig, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
