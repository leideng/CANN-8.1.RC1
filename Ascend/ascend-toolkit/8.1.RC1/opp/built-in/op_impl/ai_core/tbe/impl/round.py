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
round
"""
import functools

import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils import para_check
from tbe import tvm


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
@tbe_platform.fusion_manager.fusion_manager.register("round")
def round_compute(x, y, decimals=0, kernel_name="round"):
    """
    calculating data round, round to the nearst,tie to the even

    Parameters
    ----------
    x: TVM tensor
    the placeholder of input data
    y: dict
    shape and dtype of output, should be same shape and type as input
    kernel_name: str
    cce kernel name, default value is round

    Returns
    -------
    result: TVM tensor
    the result of round
    """
    dtype = x.dtype
    if dtype == "int32":
        input_data_one = tbe.broadcast(tvm.const(0, dtype), x.shape, dtype)
        result = tbe.vadd(x, input_data_one)
        return result

    result = tbe.round(x)
    result = tbe.cast_to(result, dtype)

    return result


# 'pylint: disable=locally-disabled,redefined-builtin
@para_check.check_op_params(
    para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME
)
def round(x, y, decimals=0, kernel_name="round"):
    """
    algorithm: round
    calculating data round, round to the nearst,tie to the even

    Parameters
    ----------
    x : dict
    shape and dtype of input, only support float16,float32,int32
    y: dict
    shape and dtype of output, should be same shape and type as input
    kernel_name : str
    cce kernel name, default value is round

    Returns
    -------
    None
    """
    input_shape = x.get("shape")
    input_dtype = x.get("dtype").lower()

    para_check.check_shape(input_shape, param_name="x")
    para_check.check_dtype(input_dtype, ("float16", "float32", "int32"), param_name="x")

    up_shape = [1]
    up_shape[0] = functools.reduce(lambda x, y: x * y, input_shape[:])

    input_data = tvm.placeholder(up_shape, name="input_data",
                                 dtype=input_dtype)
    result = round_compute(input_data, y, decimals, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(result)

    config = {"name": kernel_name,
              "tensor_list": [input_data, result]}

    tbe.cce_build_code(sch, config)
