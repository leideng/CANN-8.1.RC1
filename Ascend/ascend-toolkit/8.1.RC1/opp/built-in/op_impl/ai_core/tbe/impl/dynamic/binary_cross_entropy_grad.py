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
binary_cross_entropy_grad
"""
import math
from impl.util.util_common import is_unknown_rank_input
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_context
from impl.util.util_compute import only_static_support


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # `define a scalar, value = 1`
    SCALAR_ONE = 1
    # `define a scalar, value = -1`
    SCALAR_NEGTIVE_ONE = -1
    # `define a scalar, value = -1`
    SCALAR_EPS = 1e-12


# 'pylint: disable=locally-disabled,unused-argument
# 'pylint: disable=too-many-arguments,invalid-name,too-many-locals
@register_operator_compute("BinaryCrossEntropyGrad", op_mode="dynamic", support_fusion=only_static_support,
                           support_bfp16=True)
def binary_cross_entropy_grad_compute(x, y, grad_output, weight, output,
                                      reduction, kernel_name):
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
    x_shape = shape_util.shape_to_list(x.shape)
    x_shape, grad_shape, shape_max = shape_util.unify_broadcast_shapes([x.shape, grad_output.shape])
    x = tbe.broadcast(x, shape_max)
    y = tbe.broadcast(y, shape_max)
    grad_output = tbe.broadcast(grad_output, shape_max)
    if weight is not None:
        weight = tbe.broadcast(weight, shape_max)
    dtype = x.dtype
    support = tbe_platform.api_check_support("tbe.dsl.vmul", "float32")
    if dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.vmul", "float32"):
        x = tbe.cast_to(x, "float32")
        y = tbe.cast_to(y, "float32")
        grad_output = tbe.cast_to(grad_output, "float32")
        if weight is not None:
            weight = tbe.cast_to(weight, "float32")
    calc_dtype = x.dtype

    val1 = tbe.vsub(x, y)
    if support is True:
        minus_predict = tbe.vmuls(x, tvm.const(Constant.SCALAR_NEGTIVE_ONE, dtype="float32"))

        val2_tmp = tbe.vadds(minus_predict, tvm.const(Constant.SCALAR_ONE, dtype="float32"))
        val2 = tbe.vmul(x, val2_tmp)
        val2 = tbe.vmaxs(val2, tvm.const(Constant.SCALAR_EPS, dtype="float32"))
    else:
        minus_predict = tbe.vmuls(
            x, tvm.const(Constant.SCALAR_NEGTIVE_ONE, dtype="float16"))

        val2_tmp = tbe.vadds(
            minus_predict, tvm.const(Constant.SCALAR_ONE, dtype="float16"))
        val2 = tbe.vmul(x, val2_tmp)
        val2 = tbe.vmaxs(val2, tvm.const(Constant.SCALAR_EPS, dtype="float16"))
    result = tbe.vdiv(val1, val2)
    if weight is not None:
        result = tbe.vmul(weight, result)
    result = tbe.vmul(grad_output, result)

    if reduction == "mean":
        reduce_elts = 1.0
        for i in x_shape:
            reduce_elts *= i
        if isinstance(reduce_elts, float):
            cof = reduce_elts if math.isclose(reduce_elts, 0.0) else reduce_elts ** (-1)
            cof = tvm.const(cof, dtype=calc_dtype)
        else:
            cof = tbe.var("cof", dtype=calc_dtype)
            if calc_dtype == "float16":
                tbe.var("cof_empty", dtype=calc_dtype)
            tbe_context.get_context().add_compile_info("reduce_mean_cof_dtype", calc_dtype)
        result = tbe.vmuls(result, cof)

    if dtype == "float16":
        result = tbe.cast_to(result, dtype)

    return result


# 'pylint: disable=invalid-name,too-many-locals,too-many-statements,unused-variable
@register_operator("BinaryCrossEntropyGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def binary_cross_entropy_grad(x, y, grad_output, weight, output,
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
    if reduction not in ("mean", "sum", "none"):
        rule_desc = "reduction type should in mean/sum/none"
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule_desc, "reduction", reduction)

    predict_shape = x.get("shape")
    predict_dtype_lower = x.get("dtype").lower()
    target_dtype_lower = y.get("dtype").lower()

    dout_shape = grad_output.get("shape")
    dout_dtype_lower = grad_output.get("dtype").lower()

    # if dout is scaler get the boardcast shape, else not chenged
    if is_unknown_rank_input([x, grad_output]):
        x, grad_output = [x, x] if is_unknown_rank_input(x) else [grad_output, grad_output]
    else:
        dif_len = len(predict_shape) - len(dout_shape)
        if dif_len > 0:
            grad_output["shape"] = [1] * dif_len + list(grad_output["shape"])
            grad_output["range"] = [(1, 1)] * dif_len + list(grad_output["range"])

    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(predict_dtype_lower, check_list, param_name="x")

    if not (predict_dtype_lower == target_dtype_lower and
            predict_dtype_lower == dout_dtype_lower):
        error_detail = "dtype of x and y should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "x", "y", error_detail)

    if weight is not None:
        weight_dtype_lower = weight.get("dtype").lower()

        if predict_dtype_lower != weight_dtype_lower:
            error_detail = "dtype of x and weight should be same"
            error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "x", "weight", error_detail)

    schedules, tensors = [], []
    if weight is not None:
        ins = classify([x, grad_output], OpPatternMode.ELEWISE_WITH_BROADCAST)
        for (_predict_shape, _dout_shape) in ins:
            with tbe.compute():
                predict_shape, dout_shape = shape_util.variable_shape([_predict_shape, _dout_shape])
                dout_data_input = tvm.placeholder(dout_shape, name="dout_data_input",
                                                  dtype=predict_dtype_lower)
                predict_data_input = tvm.placeholder(predict_shape, name="predict_data_input",
                                                     dtype=predict_dtype_lower)
                target_data_input = tvm.placeholder(predict_shape, name="target_data_input",
                                                    dtype=predict_dtype_lower)
                weight_data_input = tvm.placeholder(predict_shape, name="weight_data_input",
                                                    dtype=predict_dtype_lower)
                res = binary_cross_entropy_grad_compute(predict_data_input, target_data_input,
                                                        dout_data_input, weight_data_input, output,
                                                        reduction, kernel_name)
                tensors.append([predict_data_input, target_data_input,
                                dout_data_input, weight_data_input, res])
            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)
    else:
        ins = classify([x, grad_output], OpPatternMode.ELEWISE_WITH_BROADCAST)
        for (_predict_shape, _dout_shape) in ins:
            with tbe.compute():
                predict_shape, dout_shape = shape_util.variable_shape([_predict_shape, _dout_shape])
                weight_data_input = None
                dout_data_input = tvm.placeholder(dout_shape, name="dout_data_input",
                                                  dtype=predict_dtype_lower)
                predict_data_input = tvm.placeholder(predict_shape, name="predict_data_input",
                                                     dtype=predict_dtype_lower)
                target_data_input = tvm.placeholder(predict_shape, name="target_data_input",
                                                    dtype=predict_dtype_lower)
                res = binary_cross_entropy_grad_compute(predict_data_input, target_data_input,
                                                        dout_data_input, weight_data_input, output,
                                                        reduction, kernel_name)
                tensors.append([predict_data_input, target_data_input, dout_data_input, res])
            with tvm.target.cce():
                sch = tbe.auto_schedule(res)
            schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
