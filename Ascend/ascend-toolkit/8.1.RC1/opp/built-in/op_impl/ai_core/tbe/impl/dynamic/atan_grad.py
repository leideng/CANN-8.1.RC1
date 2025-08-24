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
atan_grad

Op_description :
Computes gradients for Atan operation

# atan_grad(
#   y,
#   dy,
#   z,
#   kernel_name="cce_atan_grad")

Supportive_dtype_format :
['bfloat16', 'float16', 'float32']
['ALL']

Constraint :
[1] All : 'y' and 'dy' must have the same type and shape.
[2] All : shape size limit is 2147483648.
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=unused-argument,invalid-name,too-many-locals
@register_operator_compute("AtanGrad", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def atan_grad_compute(y, dy, z, kernel_name="atan_grad"):
    """
    Calculation for backward gradient

    Parameters:
    ----------
    y: the placeholder of input data
    dy: the placeholder of input dy
    output_z : dict of output
    kernel_name : cce kernel name, default value is atan_grad

    Algorithm :
    ----------
        res = 1/(1+y^2)*dy

    Returns
    ----------
    result res
    """

    const_one = 1
    scalar_one = tvm.const(const_one, "float32")
    dtype = y.dtype

    if dtype == "float16" and tbe_platform.api_check_support("te.lang.cce.vadd", "float32"):
        y = tbe.cast_to(y, "float32")
        dy = tbe.cast_to(dy, "float32")

    data_square = tbe.vmul(y, y)
    sum_tmp = tbe.vadds(data_square, scalar_one)
    res = tbe.vdiv(dy, sum_tmp)

    if dtype == "float16":
        res = tbe.cast_to(res, "float16")

    return res


@register_operator("AtanGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def atan_grad(y, dy, z, kernel_name="atan_grad"):
    """
    Gradient calculation for atan(x)

    Parameters:
    ----------
    y : dict of y, include shape and dtype, dtype support bfloat16, float16, float32
    dy : dict of dy, include shape and dtype, dtype support bfloat16, float16, float32
    z : dict of output, include shape and dtype
    kernel_name : cce kernel name, default value is atan_grad

    Algorithm :
    ----------
    forward :
        y = atan(x)
    backward gradient :
        de/dx = dy/dx*de/dy = 1/(1+x^2)*grad

    Returns
    ----------
    None
    """
    dtype = y.get("dtype").lower()
    dtype_grad = dy.get("dtype").lower()
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(dtype, check_list, param_name="y")
    para_check.check_dtype(dtype_grad, check_list, param_name="dy")
    ins = classify([y, dy], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (ins_y, ins_dy) in ins:
        with tbe.compute():
            shape_y, shape_dy = shape_util.variable_shape([ins_y, ins_dy])
            data_input = tvm.placeholder(shape_y, name="input_data", dtype=dtype)
            grad = tvm.placeholder(shape_dy, name="input_grad", dtype=dtype_grad)
            res = atan_grad_compute(data_input, grad, z, kernel_name)
            tensors.append([data_input, grad, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
