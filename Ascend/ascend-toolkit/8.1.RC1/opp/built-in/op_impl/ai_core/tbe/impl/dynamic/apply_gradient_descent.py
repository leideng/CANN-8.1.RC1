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
dynamic apply_gradient_descent

Op_description :
Update var by subtracting alpha * delta from it.

# apply_gradient_descent(var,
#   alpha,
#   delta,
#   out,
#   kernel_name='apply_gradient_descent')

Supportive_dtype_format :
['int32', 'int8', 'uint8', 'float32', 'float16']
['ND', 'NCHW', 'NHWC', 'NC1HWC0']

Constraint :
[1] All : the input tensors must have the same shape and type.
[2] All : shape size limit is 2147483648.
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
@register_operator_compute("ApplyGradientDescent", op_mode="dynamic", support_fusion=True)
def apply_gradient_descent_compute(var,
                                   alpha,
                                   delta,
                                   out,
                                   kernel_name="apply_gradient_descent"):
    """
    compute out_var = var - alpha * delta

    Parameters:
    ----------
    var: the placeholder of var.
    alpha : the placeholder of alpha.
    delta : the placeholder of delta.
    out : the dict of output.
    kernel_name :  cce kernel name, default value is "apply_gradient_descent".

    Returns
    -------
    out
    """

    alpha = tbe.broadcast(alpha, var.shape)
    var_change = tbe.vmul(delta, alpha)
    reuse_var = tbe.vsub(var, var_change)

    return reuse_var


# 'pylint: disable=too-many-locals
@register_operator("ApplyGradientDescent")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def apply_gradient_descent(var,
                           alpha,
                           delta,
                           out,
                           kernel_name="apply_gradient_descent"):
    """
    Update var by subtracting alpha * delta from it.

    var_{t} = var_{t-1} - alpha * delta

    Parameters:
    ----------
    var : dict of input_var, include shape and dtype,
        dtype support float16, float32.
    alpha : dict of input_alpha, include shape and dtype,
        dtype support float16, float32.
        Must have the same type as 'var', Must have the shape(1,).
    delta : dict of input_delta, include shape and dtype,
        dtype support float16, float32.
        Must have the same shape and dtype as input_var.
    out : dict of output, include shape and dtype.
    kernel_name : cce kernel name, default value is "apply_gradient_descent".

    Returns
    -------
    None
    """

    var_dtype = var.get("dtype").lower()
    alpha_dtype = alpha.get("dtype").lower()
    delta_dtype = delta.get("dtype").lower()

    check_list = ("float16", "float32")

    para_check.check_dtype(var_dtype, check_list, param_name="var")
    para_check.check_dtype(alpha_dtype, check_list, param_name="alpha")
    para_check.check_dtype(delta_dtype, check_list, param_name="delta")

    if var_dtype != alpha_dtype:
        error_detail = "type of var and alpha should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "var",
                                                               "alpha", error_detail)

    if var_dtype != delta_dtype:
        error_detail = "type of var and delta should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "var",
                                                               "delta", error_detail)

    shape_scalar = [1]
    compute_dtype = var_dtype
    data_alpha = tvm.placeholder(shape_scalar, name="data_alpha", dtype=compute_dtype)

    ins = classify([var, delta], OpPatternMode.ELEWISE)

    schedules, tensors = [], []
    for (_var, _delta) in ins:
        with tbe.compute():
            shape_var, shape_delta = shape_util.variable_shape([_var, _delta])
            data_var = tvm.placeholder(shape_var, name="data_var", dtype=compute_dtype)
            data_delta = tvm.placeholder(shape_delta, name="data_delta", dtype=compute_dtype)
            res = apply_gradient_descent_compute(data_var, data_alpha, data_delta, out)
            tensors.append([data_var, data_alpha, data_delta, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
