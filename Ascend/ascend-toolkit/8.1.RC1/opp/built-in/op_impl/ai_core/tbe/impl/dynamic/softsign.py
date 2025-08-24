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
dynamic softsign
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


# 'pylint: disable=unused-argument,too-many-locals
# 'pylint: disable=invalid-name
@register_operator_compute("Softsign", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def softsign_compute(input_x, y, kernel_name="softsign"):
    """
    Computes for softsign.
    The compute: "x / (abs(x) + 1)".

    Parameters
    ----------
    input_x: TVM tensor
        data of input.
        source data type, support "bfloat16", "float16", "float32".
    y: dict
        data of output.
    kernel_name: str
        kernel name, default value is "softsign".

    Returns
    -------
    res: TVM tensor
        output tensor. has the same type as `input_x`.
    """
    dtype = input_x.dtype
    if dtype == "float16" and \
            tbe_platform.api_check_support("te.lang.cce.vmul",
                                           "float32"):
        input_x = tbe.cast_to(input_x, "float32")

    data_abs = tbe.vabs(input_x)
    data_add = tbe.vadds(data_abs, 1)
    data_rec = tbe.vrec(data_add, "high_precision")
    res = tbe.vmul(input_x, data_rec)

    if dtype == "float16":
        res = tbe.cast_to(res, "float16")

    return res


@register_operator("Softsign")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def softsign(x, y, kernel_name="softsign"):
    """
    Computes for softsign.

    Parameters
    ----------
    x: dict
        data of input.
        source data type, support "bfloat16", "float16", "float32".
    y: dict
        data of output.
    kernel_name : str
        kernel name, default value is "softsign".

    Returns
    -------
    None
    """
    x_dtype = x.get("dtype").lower()
    check_list = ("bfloat16", "float16", "float32")
    para_check.check_dtype(x_dtype, check_list, param_name="x")

    ins = classify([x], OpPatternMode.ELEWISE)
    schedules, tensors = [], []
    for (input_x,) in ins:
        with tbe.compute():
            x_shape = shape_util.variable_shape([input_x])
            input_data = tvm.placeholder(x_shape[0], name="input_data", dtype=x_dtype)
            res = softsign_compute(input_data, y, kernel_name)
            tensors.append([input_data, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    # build
    config = {"name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
