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
dynamic acos_grad

Op_description :
Computes gradients for Acos operation

# acos_grad(
#   y,
#   dy,
#   z,
#   kernel_name="acos_grad")

Supportive_dtype_format :
['bfloat16', 'float16', 'float32']
['ALL']

Constraint :
[1] All : 'y' and 'dy' must have the same type and shape.
[2] All : shape size limit is 2147483648.
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import OpTbeImplMode


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # newton eqation is x1 = x0(3-a*(x0^2))/2
    NUM_MINUS_ONE = -1


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,too-many-locals,invalid-name
@register_operator_compute("AcosGrad", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def acos_grad_compute(y, dy, z, kernel_name="acos_grad"):
    """
    do acos_grad compute with sqrt and div
    Parameters:
    ----------------
    y: input tensor y
    dy: input tensor dy
    z: output dict
    kernel_name: cce kernel name, default value is "acos_grad"
    return: dy * (- 1 / (1 - data_y^2)^1/2)
    ----------------
    """

    dtype = y.dtype
    dtype_1 = dtype
    num_one = 1
    if dtype == "float16" and \
            tbe_platform.api_check_support("tbe.dsl.vadd", "float32"):
        y = tbe.cast_to(y, "float32")
        dy = tbe.cast_to(dy, "float32")
        dtype = "float32"

    data1_square = tbe.vmul(y, y)
    data1_square = tbe.vmuls(data1_square, tvm.const(Constant.NUM_MINUS_ONE, dtype=dtype))
    data1_square = tbe.vadds(data1_square, tvm.const(num_one, dtype=dtype))

    data1_reciprocal = tbe.vsqrt(data1_square, impl_mode=OpTbeImplMode.TBE_HIGH_PRECISION)
    data1_reciprocal = tbe.vdiv(dy, data1_reciprocal)
    res = tbe.vmuls(data1_reciprocal, tvm.const(Constant.NUM_MINUS_ONE, dtype=dtype))

    if dtype_1 == "float16":
        res = tbe.cast_to(res, "float16")
    return res


@register_operator("AcosGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def acos_grad(y, dy, z, kernel_name="acos_grad"):
    """
    do element-wise acos_grad operation between two input tensors

    Parameters:
    ----------
    y : dict of y, include shape and dtype, dtype support float16, bfloat16, float32

    dy : dict of dy, include shape and dtype, dtype support float16, bfloat16, float32

    z : dict of z, include shape and dtype, dtype support float16, bfloat16, float32

    kernel_name : cce kernel name, default value is "acos_grad"
    -------
    """

    # get the shape and dtype for input_1,input_2
    dtype = y.get("dtype").lower()
    dtype1 = dy.get("dtype").lower()

    # raise runtimeerror if the input paras are invalid
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(dtype, check_list, param_name="y")
    para_check.check_dtype(dtype1, check_list, param_name="dy")

    if dtype != dtype1:
        error_detail = "dtype of y and dy should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "y", "dy", error_detail)

    ins = classify([y, dy], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for (_y, _dy) in ins:
        with tbe.compute():
            # shape
            shape_x1, shape_x2 = shape_util.variable_shape([_y, _dy])
            # mul_compute
            data_x1 = tvm.placeholder(shape_x1, dtype=dtype, name="data_x1")
            data_x2 = tvm.placeholder(shape_x2, dtype=dtype1, name="data_x2")
            res = acos_grad_compute(data_x1, data_x2, z, kernel_name)
            tensors.append((data_x1, data_x2, res))
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
