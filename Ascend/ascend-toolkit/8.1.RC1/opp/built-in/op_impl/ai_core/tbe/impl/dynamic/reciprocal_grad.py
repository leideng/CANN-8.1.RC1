# Copyright 2021 Huawei Technologies Co., Ltd;
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
# See the License for the specific language governing permissions and;
# limitations under the License.
# ============================================================================
"""
reciprocal_grad
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,unused-argument
# 'pylint: disable=too-many-locals
@register_operator_compute("ReciprocalGrad", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def reciprocal_grad_compute(input_y, input_dy, output_data,
                            kernel_name="reciprocal_grad"):
    """
    compute reciprocal_grad

    Parameters
    ----------
    input_y: TVM tensor
        the placeholder of input y
    input_dy: TVM tensor
        the placeholder of input dy
    output_data: TVM tensor
        shape and dtype of output
    kernel_name: str
        kernel name, default value is "reciprocal_grad"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    dtype = input_y.dtype

    reciprocal_const = tvm.const(-1, dtype=dtype)
    is_cast = False

    if dtype in ("int32",):
        reciprocal_const = tbe.broadcast(reciprocal_const, input_y.shape, "int32")
        const_res = tbe.vmul(reciprocal_const, input_y)
    if dtype == "float32" and \
            tbe_platform.api_check_support("te.lang.cce.vmuls", "float32"):
        const_res = tbe.vmuls(input_y, reciprocal_const)
    if dtype in ("float16", "int8") and \
            tbe_platform.api_check_support("te.lang.cce.vmuls", "float32"):
        is_cast = True
        reciprocal_const = tvm.const(-1, dtype="float32")
        input_y = tbe.cast_to(input_y, "float32")
        input_dy = tbe.cast_to(input_dy, "float32")
        const_res = tbe.vmuls(input_y, reciprocal_const)
    if dtype != "float32" and \
            not tbe_platform.api_check_support("te.lang.cce.vmuls", "float32"):
        const_res = tbe.vmuls(input_y, reciprocal_const)
    vmul_res = tbe.vmul(const_res, input_y)
    res = tbe.vmul(vmul_res, input_dy)

    if is_cast:
        res = tbe.cast_to(res, dtype, f1628IntegerFlag=True)

    return res


@register_operator("ReciprocalGrad")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def reciprocal_grad(input_y, input_dy, output_data,
                    kernel_name="reciprocal_grad"):
    """
    algorithm: reciprocal_grad
    calculating data's reciprocal grad,dx = -1*dy*y*y,
    where `y = 1/x`, and `dy`
    is the corresponding input gradient.

    Parameters
    ----------
    input_y: dict
        shape and dtype of input_y, only support bfloat16, float16, float32, int32, int8
    input_dy: dict
        shape and dtype of input_dy, should be same shape and type as input_y
    output_data: dict
        shape and dtype of output, should be same shape and type as input_y
    kernel_name: str
        kernel name, default value is "reciprocal_grad"

    Returns
    -------
    None
    """

    dtype_y = input_y.get("dtype").lower()
    dtype_dy = input_dy.get("dtype").lower()
    check_list = ("float16", "float32", "int32", "int8", "bfloat16")
    para_check.check_dtype(dtype_y, check_list, param_name="input_y")
    schedules, tensors = [], []
    ins = classify([input_y, input_dy], OpPatternMode.ELEWISE)
    for (_input_y, _input_dy) in ins:
        with tbe.compute():
            shape_y, shape_dy = shape_util.variable_shape([_input_y, _input_dy])
            data_y = tvm.placeholder(shape_y, name="data_y", dtype=dtype_y)
            data_dy = tvm.placeholder(shape_dy, name="data_dy", dtype=dtype_dy)
            res = reciprocal_grad_compute(data_y, data_dy, output_data, kernel_name)
            tensors.append([data_y, data_dy, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
