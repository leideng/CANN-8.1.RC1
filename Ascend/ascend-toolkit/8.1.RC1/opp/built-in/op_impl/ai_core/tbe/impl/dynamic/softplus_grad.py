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
dynamic softplus_grad
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.dynamic.sigmoid import sigmoid_compute


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals
# 'pylint: disable=unused-variable
@register_operator_compute("SoftplusGrad", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def softplus_grad_compute(input_gradients, input_features, output_backprops,
                          kernel_name="softplus_grad"):
    """
    Computes softplus gradients for a softplus operation.
    The gradients: "dy * exp(x) / (1 + exp(x))".

    Parameters
    ----------
    input_gradients: TVM tensor
        The backpropagated gradients to the corresponding softplus operation.
    input_features: TVM tensor
        The input_features passed as input to the corresponding softplus operation.
        source data type support "bfloat16", "float16", "float32", "int32", "int8", "uint8".
    output_backprops: dict
        data of output.
    kernel_name: str
        kernel name, default value is "softplus_grad".

    Returns
    -------
    res: TVM tensor
        output tensor. has the same type as "input_gradients".
    """
    dtype = input_gradients.dtype

    shape_list = shape_util.broadcast_shapes(input_gradients.shape, input_features.shape)
    input_gradients = tbe.broadcast(input_gradients, shape_list[2])
    input_features = tbe.broadcast(input_features, shape_list[2])

    if dtype != "float32":
        input_gradients = tbe.cast_to(input_gradients, "float32")
        input_features = tbe.cast_to(input_features, "float32")

    res_sigmoid = sigmoid_compute(input_features, output_backprops)
    res_tmp = tbe.vmul(input_gradients, res_sigmoid)

    if dtype == "float16":
        res = tbe.cast_to(res_tmp, "float16")
    elif dtype in ("int32", "int8", "uint8"):
        data_zero = tbe.broadcast(tvm.const(0, "float16"), shape_list[2], "float16")
        res_min = tbe.vmin(res_tmp, data_zero)
        res_max = tbe.vmax(res_tmp, data_zero)
        res_max_int = tbe.floor(res_max)
        res_min_int = tbe.ceil(res_min)
        res = tbe.vadd(res_max_int, res_min_int)
    else:
        res = res_tmp

    if dtype == "int8":
        res = tbe.cast_to(res, "int8")
    elif dtype == "uint8":
        res = tbe.cast_to(res, "uint8")

    return res


@register_operator("SoftplusGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def softplus_grad(input_gradients, input_features, output_backprops,
                  kernel_name="softplus_grad"):
    """
    Computes softplus gradients for a softplus operation.
    The gradients: "dy * exp(x) / (1 + exp(x))".

    Parameters
    ----------
    input_gradients: dict
        The backpropagated gradients to the corresponding softplus operation.
    input_features: dict
        The input_features passed as input to the corresponding softplus operation.
        source data type support "bfloat16", "float16", "float32", "int32", "int8", "uint8".
    output_backprops: dict
        data of output.
    kernel_name: str
        kernel name, default value is "softplus_grad".

    Returns
    -------
    None
    """

    dtype_dy = input_gradients.get("dtype").lower()
    dtype_x = input_features.get("dtype").lower()
    check_list = ("bfloat16", "float16", "float32", "int32", "int8", "uint8")
    para_check.check_dtype(dtype_dy, check_list, param_name="input_g")
    para_check.check_dtype(dtype_x, check_list, param_name="input_x")
    if dtype_dy != dtype_x:
        error_detail = "Dtype of tensor input_gradients and input_features must be same!"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "input_gradients", \
                                                               "input_features", error_detail)

    ins = classify([input_gradients, input_features], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (input_g, input_x) in ins:
        with tbe.compute():
            g_shape, x_shape = shape_util.variable_shape([input_g, input_x])
            g_shape, x_shape = shape_util.refine_shapes_for_broadcast(g_shape, x_shape)
            tensor_g = tvm.placeholder(g_shape, dtype_dy, "tensor_g")
            tensor_x = tvm.placeholder(x_shape, dtype_x, "tensor_x")
            res = softplus_grad_compute(tensor_g, tensor_x, output_backprops, kernel_name)
            tensors.append([tensor_g, tensor_x, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {
        "name": kernel_name,
        "tensor_list": tensors}
    tbe.build(schedules, config)
