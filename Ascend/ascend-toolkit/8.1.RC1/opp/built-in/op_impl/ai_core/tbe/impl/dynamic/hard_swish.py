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
dynamic hard_swish
`f(x) = min(max(0,x+3), 6) * x / 6`
"""
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    # constants used in the equation
    CONST_THREE = 3
    CONST_SIX = 6
    CONST_ONE_IN_SIX = 1 / 6


# 'pylint: disable=unused-argument,too-many-locals,invalid-name
@register_operator_compute("HardSwish", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def hard_swish_compute(input_x, output_y, kernel_name="hard_swish"):
    """
    compute of hard_swish

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of first input data
    output_y: dict
        shape and dtype of output,should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is "hard_swish"

    Returns
    -------
    compute result of hard_swish
    """
    dtype = input_x.dtype
    input_x_fp32 = tbe.cast_to(input_x, "float32") if dtype == "float16" else input_x
    input_add3 = tbe.vadds(input_x_fp32, tvm.const(Constant.CONST_THREE, "float32"))
    max_res = tbe.vmaxs(input_add3, tvm.const(0, "float32"))
    relu6_res = tbe.vmins(max_res, tvm.const(Constant.CONST_SIX, "float32"))
    relu6_res_ov6 = tbe.vmuls(relu6_res, tvm.const(Constant.CONST_ONE_IN_SIX, "float32"))
    res_fp32 = tbe.vmul(input_x_fp32, relu6_res_ov6)
    return tbe.cast_to(res_fp32, "float16") if dtype == "float16" else res_fp32


@register_operator("HardSwish")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def hard_swish(input_x, output_y, kernel_name="hard_swish"):
    """
       f(x)= 0(x <= -3)
       f(x)= x(x >= 3)
       f(x)= x(x+3)/6(otherwise)
    Parameters
    ----------
    input_x : dict
        shape and dtype of input_x
    output_y : dict
        shape and dtype of output_y, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is "hard_swish"

    Returns
    ------
    None
    """

    input_dtype = input_x.get("dtype").lower()
    # check input tensor data_type
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")

    schedules, tensors = [], []
    ins = classify([input_x], OpPatternMode.ELEWISE)
    for (_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])
            data_input = tvm.placeholder(x_shape[0], dtype=input_dtype, name="data_input")
            res = hard_swish_compute(data_input, output_y, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors,
              "bool_storage_as_1bit": False}
    tbe.build(schedules, config)
