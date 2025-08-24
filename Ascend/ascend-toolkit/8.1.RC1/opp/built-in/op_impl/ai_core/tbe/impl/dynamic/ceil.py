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
dynamic ceil
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_platform


# 'pylint: disable=unused-argument,too-many-locals,invalid-name
@register_operator_compute("Ceil", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def ceil_compute(input_x, output_x, kernel_name="ceil"):
    """
    ceil compute
    calculating element-wise smallest integer not less than input_x

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    output_y: dict
        dict with keys(shape and dtype) of output_y
    kernel_name: str
        kernel name, default value is "ceil"

    Returns
    -------
    res: TVM tensor
        the result of ceil(input_x)
    """
    dtype_x = input_x.dtype.lower()
    if not tbe_platform.api_check_support("tbe.dsl.ceil", dtype_x):
        input_x = tbe.cast_to(input_x, "float16")
    
    if tbe_platform.api_check_support("tbe.dsl.ceil", "f322f32"):
        x_f32 = tbe.cast_to(input_x, "float32")
        res = tbe.ceil(x_f32, "float32")
    else:
        res = tbe.ceil(input_x)
    res = tbe.cast_to(res, dtype_x)
    
    return res


@register_operator("Ceil")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def ceil(input_x, output_x, kernel_name="ceil"):
    """
     algorithm: ceil
    calculating element-wise smallest integer not less than input_x,
    the type of input_x is bfloat16, float16 or float32

    Parameters
    ----------
    input_x: dict
        dict with keys(shape and dtype) of input
    output_y: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "ceil"

    Returns
    -------
    None
    """
    x_dtype = input_x.get("dtype").lower()
    check_list = ("float16", "float32", "bfloat16")
    para_check.check_dtype(x_dtype, check_list, param_name="input_x")

    ins = classify([input_x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_input_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_input_x])
            input_data = tvm.placeholder(x_shape[0], name="input_data", dtype=x_dtype)
            res = ceil_compute(input_data, output_x, kernel_name)
            tensors.append([input_data, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
