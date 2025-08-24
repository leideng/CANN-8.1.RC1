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
rint
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
@register_operator_compute("Rint", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def rint_compute(input_x, output_y, kernel_name="rint"):
    """
    rint compute
    calculating rint(x):
    returns the integer nearest to x by element-wise
    If the result is between two representable values,
     the even number should be used.
    For example:
    x :    [0.9, 2.5, 2.3, 1.5, -4.5]
    res : [ 1.0, 2.0, 2.0, 2.0, -4.0 ]

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input_x
    output_y: dict
        dict with keys(shape and dtype) of output_y
    kernel_name: str
        kernel name, default value is "rint"

    Returns
    -------
    res: TVM tensor
        the result of rint compute
    """
    dtype = input_x.dtype
    if not tbe_platform.api_check_support("tbe.dsl.round", dtype):
        input_x = tbe.cast_to(input_x, "float16")
    if tbe_platform.api_check_support("tbe.dsl.round", "f322f32"):
        x_f32 = tbe.cast_to(input_x, "float32")
        res = tbe.round(x_f32, "float32")
    else:
        res = tbe.round(input_x)
    res = tbe.cast_to(res, dtype)

    return res


@register_operator("Rint")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def rint(input_x, output_y, kernel_name="rint"):
    """
    algorithm: rint
    calculating rint(x):
    returns the integer nearest to x by element-wise
    If the result is between two representable values,
     the even number should be used.
    For example:
    x :    [0.9, 2.5, 2.3, 1.5, -4.5]
    res : [ 1.0, 2.0, 2.0, 2.0, -4.0 ]

    Parameters
    ----------
    input_x: dict
        dict with keys(shape and dtype) of input_x
    output_y: dict
        dict with keys(shape and dtype) of output_y
    kernel_name: str
        kernel name, default value is "rint"

    Returns
    -------
    None
    """
    dtype = input_x.get("dtype").lower()
    check_list = ("bfloat16", "float16", "float32")
    para_check.check_dtype(dtype.lower(), check_list, param_name="input_x")

    ins = classify([input_x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []

    for (_input_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_input_x])
            data_x = tvm.placeholder(x_shape[0], dtype=dtype, name="data_x")
            res = rint_compute(data_x, output_y, kernel_name)

            tensors.append([data_x, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
