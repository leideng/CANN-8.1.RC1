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
softsign
"""
import functools

from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import shape_util
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm


# 'pylint: disable=locally-disabled,unused-argument,too-many-locals
# 'pylint: disable=invalid-name
@register_operator_compute("softsign", op_mode="static", support_fusion=True)
def softsign_compute(input_x, y, kernel_name="softsign"):
    """
    Computes for softsign.
    The compute: "x / (abs(x) + 1)".

    Parameters
    ----------
    input_x: TVM tensor
        data of input.
        source data type, support "float16", "float32".
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
    if dtype == "float16" and tbe_platform.api_check_support("te.lang.cce.vmul", "float32"):
        input_x = tbe.cast_to(input_x, "float32")

    data_abs = tbe.vabs(input_x)
    data_add = tbe.vadds(data_abs, 1)
    data_rec = tbe.vrec(data_add)
    res = tbe.vmul(input_x, data_rec)

    if dtype == "float16":
        res = tbe.cast_to(res, "float16")

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def softsign(x, y, kernel_name="softsign"):
    """
    Computes for softsign.

    Parameters
    ----------
    x: dict
        data of input.
        source data type, support "float16", "float32".
    y: dict
        data of output.
    kernel_name : str
        kernel name, default value is "softsign".

    Returns
    -------
    None
    """
    shape_input = x.get("shape")
    dtype_input = x.get("dtype")

    para_check.check_shape(shape_input, param_name="x")

    check_list = ("float16", "float32")
    para_check.check_dtype(dtype_input.lower(), check_list, param_name="x")

    shape = shape_util.shape_refine(shape_input)
    shape_x = (functools.reduce(lambda x, y: x*y, shape[:]),)
    input_dtype = dtype_input.lower()
    data = tvm.placeholder(shape_x, name="data", dtype=input_dtype)

    res = softsign_compute(data, y, kernel_name)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data, res]}

    tbe.cce_build_code(sch, config)
