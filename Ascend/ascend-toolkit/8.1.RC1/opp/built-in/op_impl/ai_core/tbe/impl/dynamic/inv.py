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
inv
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import register_operator


# 'pylint: disable=locally-disabled,unused-argument
@register_operator_compute("Inv", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def inv_compute(input_x, output_y, kernel_name="inv"):
    """
    compute inv

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: TVM tensor
        the placeholder of output data
    kernel_name: str
        kernel name, default value is "inv"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    scalar_one = 1

    dtype = input_x.dtype

    temp_const = tvm.const(scalar_one, dtype=dtype)

    if dtype == "int32":
        input_x = tbe.cast_to(input_x, "float32")
        temp_tensor = tbe.broadcast(temp_const, input_x.shape, "float32")
        res = tbe.vdiv(temp_tensor, input_x)
    else:
        temp_tensor = tbe.broadcast(temp_const, input_x.shape, dtype)

    res = tbe.vdiv(temp_tensor, input_x)

    if dtype == "int32":
        res = tbe.cast_to(res, "int32")

    return res


@register_operator("Inv")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def inv(input_x, output_y, kernel_name="inv"):
    """
    algorithm: inv
    calculating data's reciprocal, y = 1 / x

    Parameters
    ----------
    input_x: dict
        shape and dtype of input, only support bfloat16, float16, float32, int32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        kernel name, default value is "inv"

    Returns
    -------
    None
    """
    input_dtype = input_x.get("dtype").lower()
    check_list = ("float16", "float32", "int32", "bfloat16")
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")

    schedules, tensors = [], []
    ins = classify([input_x], OpPatternMode.ELEWISE)
    for (_input_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_input_x])
            data_input = tvm.placeholder(x_shape[0], dtype=input_dtype,
                                         name="data_input")
            res = inv_compute(data_input, output_y, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors,
              "bool_storage_as_1bit": False}
    tbe.build(schedules, config)
