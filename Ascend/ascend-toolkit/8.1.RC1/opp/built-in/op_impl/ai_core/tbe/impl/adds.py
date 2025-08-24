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
adds
"""
import te.lang.cce as tbe
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check


# 'pylint: disable=invalid-name,unused-argument
@register_operator_compute("adds", op_mode="static", support_fusion=True)
def adds_compute(x, y, value, kernel_name="adds"):
    """
    calculating data

    Parameters
    ----------
    x : TVM tensor
        the placeholder of input
    y : dict
        dict of output
    value : a number of float or int
    kernel_name : str
        kernel name, default value is "adds"

    Returns
    -------
    res: TVM tensor
        the calculation results
    """
    res = tbe.vadds(x, value)
    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_FLOAT,
                            para_check.KERNEL_NAME)
def adds(x, y, value, kernel_name="adds"):
    """
    calculating data

    Parameters
    ----------
    x : dict
        shape and dtype of input
    y : dict
        shape and dtype of output, should be same shape and type as input
    value: a number of float
    kernel_name : str
        kernel name, default value is "adds"

    Returns
    -------
    None
    """
    shape = x.get("shape")
    dtype = x.get("dtype").lower()

    check_list = ("float16", "float32", "int32")
    para_check.check_dtype(dtype, check_list, param_name="x")

    scalar = tvm.const(value, dtype=dtype)
    data_input = tvm.placeholder(shape, name="data_input", dtype=dtype)
    res = adds_compute(data_input, y, scalar)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": [data_input, res]
    }
    tbe.build(sch, config)
