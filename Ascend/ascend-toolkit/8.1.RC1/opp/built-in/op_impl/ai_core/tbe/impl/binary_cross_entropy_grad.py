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
binary_cross_entropy_grad
"""
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=locally-disabled,unused-argument
# 'pylint: disable=too-many-arguments,invalid-name,too-many-locals
@register_operator_compute("binary_cross_entropy_grad", op_mode="static", support_fusion=True)
def binary_cross_entropy_grad_compute(
        x,
        y,
        grad_output,
        weight,
        output,
        reduction,
        kernel_name):
    """
    calculating binary_cross_entropy_grad_compute

    Parameters
    ----------
    x : TVM tensor
        the output of previous layer
    y : TVM tensor
        label
    grad_output : TVM tensor
        last gradient
    weight : None or TVM tensor
        weight for bce
    output : TVM tensor
        result after compute
    reduction : string
        reduce type of bceloss
    kernel_name : str
        kernel name

    Returns
    -------
    output tensor
    """
    shape = shape_util.shape_to_list(x.shape)
    dtype = x.dtype
    support = tbe_platform.api_check_support(
        "tbe.dsl.vmul", "float32")
    if dtype == "float16" and \
            tbe_platform.api_check_support("tbe.dsl.vmul", "float32"):
        x = tbe.cast_to(x, "float32")
        y = tbe.cast_to(y, "float32")
        grad_output = tbe.cast_to(grad_output, "float32")
        if weight is not None:
            weight = tbe.cast_to(weight, "float32")

    # if grad_output is scaler will boradcast to predict tensor
    # else not changed
    grad_output = tbe.broadcast(grad_output, shape)

    val1 = tbe.vsub(x, y)
    scalar_one = 1
    scalar_eps = 1e-12
    scalar_negtive_one = -1
    if support is True:
        minus_predict = tbe.vmuls(
            x, tvm.const(scalar_negtive_one, dtype="float32"))

        val2_tmp = tbe.vadds(
            minus_predict, tvm.const(scalar_one, dtype="float32"))
        val2 = tbe.vmul(x, val2_tmp)
        val2 = tbe.vmaxs(val2, tvm.const(scalar_eps, dtype="float32"))
    else:
        minus_predict = tbe.vmuls(
            x, tvm.const(scalar_negtive_one, dtype="float16"))

        val2_tmp = tbe.vadds(
            minus_predict, tvm.const(scalar_one, dtype="float16"))
        val2 = tbe.vmul(x, val2_tmp)
        val2 = tbe.vmaxs(val2, tvm.const(scalar_eps, dtype="float16"))
    result = tbe.vdiv(val1, val2)
    if weight is not None:
        result = tbe.vmul(weight, result)
    result = tbe.vmul(grad_output, result)

    if reduction == "mean":
        reduce_elts = 1.0
        for i in shape:
            reduce_elts *= i
        result = tbe.vmuls(result, reduce_elts**(-1))

    if dtype == "float16":
        result = tbe.cast_to(result, dtype)

    return result


# 'pylint: disable=invalid-name,too-many-locals,too-many-statements
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def binary_cross_entropy_grad(
        x,
        y,
        grad_output,
        weight,
        output,
        reduction="mean",
        kernel_name="binary_cross_entropy_grad"):
    """
    calculating data

    Parameters
    ----------
    x : dict
        the predict of previous layer shape and dtype
    y : dict
        target label
    grad_output : dict
        last gradient/dout, if scalar, reshape first
    weight : None or TVM tensor
        weight for bce
    output : dict
        result gradient after compute
    reduction : string
        reduce type of bceloss, must be "none", "sum" or "mean"
    kernel_name : str
        kernel name, default value is "binary_cross_entropy_grad"

    Returns
    -------
    None
    """
    predict_shape = x.get("shape")
    predict_dtype = x.get("dtype")
    predict_dtype_lower = predict_dtype.lower()
    para_check.check_shape(predict_shape, param_name="x")
    shape_size = para_check.check_tensor_shape_size(predict_shape)
    predict_data_input = tvm.placeholder(
        [shape_size], name="predict_data_input", dtype=predict_dtype_lower)

    target_shape = y.get("shape")
    target_dtype = y.get("dtype")
    target_dtype_lower = target_dtype.lower()
    para_check.check_shape(target_shape, param_name="y")
    shape_size = para_check.check_tensor_shape_size(target_shape)
    target_data_input = tvm.placeholder(
        [shape_size], name="target_data_input", dtype=target_dtype_lower)

    dout_shape = grad_output.get("shape")
    dout_dtype = grad_output.get("dtype")
    dout_dtype_lower = dout_dtype.lower()
    para_check.check_shape(dout_shape, param_name="grad_output")

    # if dout is scaler get the boardcast shape, else not chenged
    _, dout_shape, _ = shape_util.broadcast_shapes(target_shape, dout_shape, param_name_input1="y",
                                                   param_name_input2="grad_output")
    shape_size = para_check.check_tensor_shape_size(dout_shape)
    dout_data_input = tvm.placeholder(
        [shape_size], name="dout_data_input", dtype=dout_dtype_lower)

    check_list = ("float16", "float32")
    para_check.check_dtype(predict_dtype_lower, check_list, param_name="x")

    if predict_shape != target_shape:
        error_detail = "shape of x and y should be same"
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "x", "y", error_detail)

    if not (predict_dtype_lower == target_dtype_lower and
            predict_dtype_lower == dout_dtype_lower):
        error_detail = "dtype of x and y should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "x", "y", error_detail)

    weight_data_input = None
    if weight is not None:
        weight_shape = weight.get("shape")
        weight_dtype = weight.get("dtype")
        weight_dtype_lower = weight_dtype.lower()
        para_check.check_shape(weight_shape, param_name="weight")
        shape_size = para_check.check_tensor_shape_size(weight_shape)
        weight_data_input = tvm.placeholder(
            [shape_size], name="weight_data_input", dtype=weight_dtype_lower)

        if predict_shape != weight_shape:
            error_detail = "shape of x and weight should be same"
            error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "x", "weight", error_detail)
        if predict_dtype_lower != weight_dtype_lower:
            error_detail = "dtype of x and weight should be same"
            error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "x", "weight", error_detail)
    if reduction not in ("mean", "sum", "none"):
        rule_desc = "reduction type should in mean/sum/none"
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule_desc, "reduction", reduction)

    res = binary_cross_entropy_grad_compute(
        predict_data_input, target_data_input,
        dout_data_input, weight_data_input, output,
        reduction, kernel_name)

    with tvm.target.cce():
        sch = auto_schedule(res)

    if weight is not None:
        config = {
            "print_ir": False,
            "name":
                kernel_name,
            "tensor_list": [
                predict_data_input, target_data_input,
                dout_data_input, weight_data_input, res
            ]
        }
    else:
        config = {
            "print_ir": False,
            "name":
                kernel_name,
            "tensor_list": [
                predict_data_input, target_data_input, dout_data_input, res
            ]
        }

    build(sch, config)
