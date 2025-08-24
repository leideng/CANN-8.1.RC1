# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
real
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=invalid-name,unused-argument
@register_operator_compute("Real", op_mode="dynamic")
def real_compute(input, output, Tout=None, kernel_name="real"):
    """
    algorithm: real

    Parameters
    ----------
    input: TVM tensor
        the placeholder of input
    output: dict
        dict info of output
    kernel_name: str
        kernel name, default value is "real"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    res = tbe.real(input)
    return res


# 'pylint: disable=redefined-builtin
@register_operator("Real")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
def real(input, output, Tout=None, kernel_name="real"):
    """
    algorithm: real

    calculating data's real

    Parameters
    ----------
    input : dict
        shape and dtype of input, only support bfloat16, float16, float32, int32
    output: dict
        shape and dtype of output, Extracting Real Numbers from Complex Numbers.
    kernel_name : str
        cce kernel name, default value is real

    Returns
    -------
    None
    """
    dtype_input = input.get("dtype").lower()
    check_list = ("float16", "float32", "complex32", "complex64")
    para_check.check_dtype(dtype_input, check_list, param_name="input")

    ins = classify([input], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (_x, ) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([_x])
            data_input = tvm.placeholder(x_shape[0], name="data_input", dtype=dtype_input)
            res = real_compute(data_input, output, None, kernel_name)

            tensors.append([data_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
