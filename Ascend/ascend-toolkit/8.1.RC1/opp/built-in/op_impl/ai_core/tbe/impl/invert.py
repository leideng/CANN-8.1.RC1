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
invert
"""
import functools

import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check


# 'pylint: disable=locally-disabled,unused-argument
@register_operator_compute("invert", op_mode="static", support_fusion=True)
def invert_compute(input_x, output_y, kernel_name="invert"):
    """Flips all bits elementwise.

    Parameters
    ----------
    input_x: TVM tensor
        input tensor.
    output_y: dict
        the dict of output tensor.
    kernel_name: str
        cce kernel name, default value is "invert".

    Returns
    -------
    res: TVM tensor
        output tensor, has the same shape and type as input tensor.
    """
    res = tbe.vnot(input_x)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def invert(input_x, output_y, kernel_name="invert"):
    """Flips all bits elementwise.

    Parameters
    ----------
    input_x: dict
        the dict of input tensor.
        Must be one of the following types: `int16`, `uint16`.
    output_y: dict
        the dict of output tensor.
    kernel_name: str
        cce kernel name, default value is "invert".

    Returns
    -------
    None.
    """
    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype")
    dtype_x_lower = dtype_x.lower()
    check_list = ("int16", "uint16")

    para_check.check_shape(shape_x, param_name="input_x")
    para_check.check_dtype(dtype_x_lower, check_list, param_name="input_x")

    shape_x = (functools.reduce(lambda x, y: x * y, shape_x[:]),)
    data_x = tvm.placeholder(shape_x, name="data_x", dtype=dtype_x_lower)
    res = invert_compute(data_x, output_y, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name, "tensor_list": [data_x, res]}
    tbe.cce_build_code(sch, config)
