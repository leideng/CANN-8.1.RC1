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
inv_grad
"""
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,unused-argument
@register_operator_compute("InvGrad", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def inv_grad_compute(input_y, input_dy, output_z, kernel_name="inv_grad"):
    """
    compute inv_grad

    Parameters
    ----------
    input_y: TVM tensor
        the placeholder of input y
    input_dy: TVM tensor
        the placeholder of input dy
    output_z: TVM tensor
        shape and dtype of output
    kernel_name: str
        kernel name, default value is "inv_grad"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    scalar_negative_one = -1
    dtype = input_y.dtype

    inv_const = tvm.const(scalar_negative_one, dtype=dtype)
    has_improve_precision = False
    if dtype in ("float16", "int8"):
        if tbe_platform.api_check_support("tbe.dsl.vmuls", "float32"):
            inv_const = tvm.const(scalar_negative_one, dtype="float32")
            input_y = tbe.cast_to(input_y, "float32")
            input_dy = tbe.cast_to(input_dy, "float32")
            has_improve_precision = True
        const_res = tbe.vmuls(input_dy, inv_const)
    elif dtype in ("int32",):
        inv_const = tbe.broadcast(inv_const, input_y.shape, "int32")
        const_res = tbe.vmul(inv_const, input_dy)
    else:
        const_res = tbe.vmuls(input_dy, inv_const)
    vmul_res = tbe.vmul(const_res, input_y)
    res = tbe.vmul(vmul_res, input_y)

    if has_improve_precision:
        res = tbe.cast_to(res, dtype, f1628IntegerFlag=True)

    return res


# 'pylint: disable=too-many-locals
@register_operator("InvGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def inv_grad(input_y, input_dy, output_z, kernel_name="inv_grad"):
    """
    algorithm: inv_grad
    calculating data's reciprocal grad,dx = -1*dy*y*y, where `y = 1/x`, and `dy`
    is the corresponding input gradient.

    Parameters
    ----------
    input_y: dict
        shape and dtype of input_y, only support bfloat16, float16, float32, int32, int8
    input_dy: dict
        shape and dtype of input_dy, should be same shape and type as input_y
    output_z: dict
        shape and dtype of output, should be same shape and type as input_y
    kernel_name: str
        kernel name, default value is "inv_grad"

    Returns
    -------
    None
    """
    dtype_input_y = input_y.get("dtype")
    dtype_input_dy = input_dy.get("dtype")

    dtype_input_y = dtype_input_y.lower()
    dtype_input_dy = dtype_input_dy.lower()

    if dtype_input_dy != dtype_input_y:
        error_detail = "dtype of input_y and input_dy must be the same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "input_y", \
                                                               "input_dy", error_detail)

    check_list = ("float16", "float32", "int32", "int8", "bfloat16")

    para_check.check_dtype(dtype_input_y, check_list, param_name="y")
    para_check.check_dtype(dtype_input_dy, check_list, param_name="dy")

    ins = classify([input_y, input_dy], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for (data1, data2) in ins:
        with tbe.compute():
            shape_y, shape_dy = shape_util.variable_shape([data1, data2])

            date_y = tvm.placeholder(shape_y, dtype=dtype_input_y, name="date_y")
            date_dy = tvm.placeholder(shape_dy, dtype=dtype_input_dy, name="date_dy")

            res = inv_grad_compute(date_y, date_dy, output_z, kernel_name)

            tensors.append([date_y, date_dy, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    # build
    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
