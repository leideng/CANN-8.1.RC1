# Copyright 2020 Huawei Technologies Co., Ltd
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
relu6_grad
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals,redefined-argument-from-local
def relu6_grad_compute_v2(input_grad, input_x):
    """
    Parameters
    ----------
    input_grad : TVM tensor
        the placeholder of input_grad
    input_x : TVM tensor
        the placeholder of input_x

    Returns
    ------
    compute result of relu6grad for Ascend910B and Ascend910_93
    """
    xdtype = input_x.dtype
    _, _, data_shape = shape_util.broadcast_shapes(input_x.shape, input_grad.shape,
                                                param_name_input1="input_x",
                                                param_name_input2="input_grad")
    input_x = tbe.broadcast(input_x, data_shape)
    input_grad = tbe.broadcast(input_grad, data_shape)

    tensor_zero = tbe.broadcast(tvm.const(0.0, xdtype), data_shape)
    tensor_one = tvm.const(1.0, xdtype)
    tensor_six = tvm.const(6.0, xdtype)

    temp_mask = tbe.vcmp(input_x, tensor_six, "lt", "bit")
    res_temp = tbe.vsel(temp_mask, tensor_one, tensor_zero)
    temp_mask_2 = tbe.vcmp(input_x, tensor_zero, "gt", "bit")
    res_temp_2 = tbe.vsel(temp_mask_2, res_temp, tensor_zero)
    temp_mask_3 = tbe.vcmp(res_temp_2, tensor_one, "eq", "bit")
    res = tbe.vsel(temp_mask_3, input_grad, tensor_zero)

    return res


# 'pylint: disable=too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals,redefined-argument-from-local
@register_operator_compute("Relu6Grad", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def relu6_grad_compute(input_grad, input_x, output_y, kernel_name="relu6_grad"):
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
        cce kernel name, default value is "relu6_grad"

    Returns
    ------
    compute result of relu6grad
    """
    if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") in ("Ascend910B", "Ascend910_93"):
        res = relu6_grad_compute_v2(input_grad, input_x)
        return res

    # `input_x<=6 and input_x>=0`
    # get min between input and 6
    min_positive_6 = tbe.vmins(input_x, 6)
    # get max between input and 0
    max_zero_min_6 = tbe.vmaxs(min_positive_6, 0)

    x_sub_6 = tbe.vadds(max_zero_min_6, -6)
    x_mul_x_6 = tbe.vmul(max_zero_min_6, x_sub_6)

    input_dtype = input_x.dtype
    if input_dtype == "float16":
        # algrithm : `Y = X*(X-6)*1024/(X*(X-6)*1024+ESP_MIN)`
        # for float16, add a small number which value is 1.18e-7, so that the divisor is not equal to 0, and for
        # accuracy, multiply by a number which value is 1024.
        x_mul_x_6_big = tbe.vmuls(x_mul_x_6, 1024)
        y_add_espmin = tbe.vadds(x_mul_x_6_big, 1.18e-7)
        y_y_esp_min = tbe.vdiv(x_mul_x_6_big, y_add_espmin)
    if input_dtype == "float32":
        # algrithm : `Y = X*(X-6)/(X*(X-6)+ESP_MIN)`
        # for float32, add a small number which value is 1.18e-38, so that the divisor is not equal to 0.
        y_add_espmin = tbe.vadds(x_mul_x_6, 1.18e-38)
        y_y_esp_min = tbe.vdiv(x_mul_x_6, y_add_espmin)

    _, _, y_shape = shape_util.broadcast_shapes(y_y_esp_min.shape, input_grad.shape,
                                                param_name_input1="y_y_esp_min",
                                                param_name_input2="input_grad")

    input1 = tbe.broadcast(y_y_esp_min, y_shape)
    input2 = tbe.broadcast(input_grad, y_shape)

    final_res = tbe.vmul(input1, input2)

    return final_res


# 'pylint: disable=too-many-locals
@register_operator("Relu6Grad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def relu6_grad(input_grad, input_x, output_y, kernel_name="relu6_grad"):
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
        cce kernel name, default value is "relu6_grad"

    Returns
    ------
    None
    """
    # check input shape
    g_dtype = input_grad.get("dtype").lower()
    x_dtype = input_x.get("dtype").lower()
    check_list = ("bfloat16", "float16", "float32")
    para_check.check_dtype(g_dtype, check_list, param_name="input_g")
    para_check.check_dtype(x_dtype, check_list, param_name="input_x")
    if x_dtype == "float32" and not tbe_platform.api_check_support("te.lang.cce.vmuls", "float32"):
        error_manager_vector.raise_err_input_dtype_not_supported(kernel_name, "relu6_grad",
                                                                 "float16 while input dtype is float32", x_dtype)

    if g_dtype != x_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "input_grad", "input_x",
                                                              g_dtype, x_dtype)

    ins = classify([input_grad, input_x], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (input_grad, input_x) in ins:
        with tbe.compute():
            g_shape, x_shape = shape_util.variable_shape([input_grad, input_x])
            tensor_g = tvm.placeholder(g_shape, g_dtype, "tensor_g")
            tensor_x = tvm.placeholder(x_shape, x_dtype, "tensor_x")
            res = relu6_grad_compute(tensor_g, tensor_x, output_y, kernel_name)
            tensors.append((tensor_g, tensor_x, res))
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
