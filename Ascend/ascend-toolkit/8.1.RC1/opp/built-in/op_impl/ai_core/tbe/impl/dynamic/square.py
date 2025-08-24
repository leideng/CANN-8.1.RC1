# Copyright 2020 Huawei Technologies Co., Ltd
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
dynamic square
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode


# 'pylint: disable=unused-argument,redefined-argument-from-local
@register_operator_compute("Square", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def square_compute(input_x, output_y, kernel_name="square"):
    """
    algorithm: square
    calculating data's square,y= x*x

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is square

    Returns
    -------
    res : tvm.tensor
        the result of square
    """
    api_check = tbe_platform.api_check_support("tbe.dsl.vmul",
                                               "float32")
    input_dtype = input_x.dtype
    if not(api_check) and input_dtype == "float32":
        input_x = tbe.cast_to(input_x, "float16")
        res = tbe.vmul(input_x, input_x)
        res = tbe.cast_to(res, "float32")
    else:
        res = tbe.vmul(input_x, input_x)

    return res


@register_operator("Square")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def square(input_x, output, kernel_name="square"):
    """
    algorithm: square
    calculating data's square,y= x*x

    Parameters
    ----------
    input_x : dict
        shape and dtype of input, only support bfloat16, float16, float32, int32, int64
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "square"

    Returns
    -------
    None
    """

    # check dtype
    x_dtype = input_x.get("dtype").lower()
    check_list = ("float16", "float32", "int32", "bfloat16", "int64")
    para_check.check_dtype(x_dtype, check_list, param_name="input_x")

    ins = classify([input_x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (input_x,) in ins:
        with tbe.compute():
            # shape
            x_shape = shape_util.variable_shape([input_x])
            # square_compute
            data_x = tvm.placeholder(x_shape[0], x_dtype, name="data_x")
            res = square_compute(data_x, output, kernel_name)

            tensors.append((data_x, res))
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
