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
floor
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,unused-argument
@register_operator_compute("Floor", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def floor_compute(input_x, output_y, kernel_name="floor"):
    """
    floor compute
    calculating element-wise largest integer not greater than input_x

    Parameters
    ----------
    input_x : TVM tensor
        the placeholder of input_x
    output_y : dict
        dict with keys(shape and dtype) of output
    kernel_name : str
        kernel name, default value is "floor"

    Returns
    -------
    res : TVM tensor
        the result of floor(input_x)
    """
    dtype_x = input_x.dtype.lower()
    if not tbe_platform.api_check_support("tbe.dsl.floor", dtype_x):
        input_x = tbe.cast_to(input_x, "float16")
        
    if tbe_platform.api_check_support("tbe.dsl.floor", "f322f32"):
        x_f32 = tbe.cast_to(input_x, "float32")
        res = tbe.floor(x_f32, "float32")
    else:
        res = tbe.floor(input_x)
    res = tbe.cast_to(res, dtype_x)

    return res


@register_operator("Floor")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def floor(input_x, output_y, kernel_name="floor"):
    """
    algorithm: floor
    calculation element-wise largest integer not greater than input_x,
    the type of input_x is bfloat16 or float16 or float32

    Parameters
    ----------
    input_x : dict
        dict with keys(shape and dtype) of input
    output_y : dict
        dict with keys(shape and dtype) of output
    kernel_name : str
        kernel_name, default value is "floor"

    Returns
    -------
    None
    """
    input_dtype = input_x.get("dtype").lower()
    check_list = ("bfloat16", "float16", "float32")
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")

    schedules, tensors = [], []
    ins = classify([input_x], OpPatternMode.ELEWISE)
    for (classify_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([classify_x])[0]
            data_input = tvm.placeholder(x_shape, dtype=input_dtype, name="data_input")
            res = floor_compute(data_input, output_y, kernel_name)
            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors, "bool_storage_as_1bit": False}

    tbe.build(schedules, config)
