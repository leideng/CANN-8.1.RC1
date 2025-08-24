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
rint
"""
import functools

import te.lang.cce as tbe
import te.platform as tbe_platform
from te.utils import para_check
from tbe import tvm


# 'pylint: disable=locally-disabled,unused-argument
@tbe_platform.fusion_manager.fusion_manager.register("rint")
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
    res = tbe.round(input_x)
    res = tbe.cast_to(res, input_x.dtype)

    return res


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
    shape_x = input_x.get("shape")
    dtype = input_x.get("dtype")

    para_check.check_shape(shape_x, param_name="input_x")

    check_list = ("float16", "float32")
    para_check.check_dtype(dtype.lower(), check_list, param_name="input_x")
    fuseshape = [1]
    fuseshape[0] = functools.reduce(lambda x, y: x*y, shape_x)
    data_x = tvm.placeholder(fuseshape, dtype=dtype.lower(), name="data")
    res = rint_compute(data_x, output_y, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_x, res]}
    tbe.cce_build_code(sch, config)
