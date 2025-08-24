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
in_training_reduce_v2
"""

from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tuple_sum
from impl.dynamic.in_training_reduce_v2 import op_select_format as in_op_select_format


# 'pylint: disable=unused-argument,invalid-name
# 'pylint: disable=too-many-locals, too-many-statements,redefined-builtin
def op_select_format(x, sum, square_sum, kernel_name="in_training_reduce_v2"):
    """
    select format dynamically
    """
    return in_op_select_format(x, sum, square_sum, kernel_name)


def in_training_reduce_compute(x, sum, square_sum, format_x, kernel_name="in_training_reduce_v2"):
    """
    DSL description of the instancenorm operator's mathematical calculation process

    Parameters
    ----------
    x: TVM tensor
        the placeholder of input x
    sum: dict
        shape and dtype of input sum
    square_sum: dict
        shape and dtype of input square_sum
    kernel_name: str
        cce kernel name, default value is "in_training_reduce_v2"

    Returns
    -------
    res_tuple: tuple
        (sum_x, square_sum_x)
    """
    if format_x in ("NDC1HWC0",):  # only support NDC1HWC0 and NC1HWC0
        axis = [1, 3, 4]
    else:
        axis = [2, 3]

    if x.dtype == "float16":
        x = tbe.cast_to(x, "float32")

    square_x = tbe.vmul(x, x)
    sum_x, square_sum_x = tuple_sum([x, square_x], axis, True)

    return sum_x, square_sum_x


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def in_training_reduce_v2(x, sum, square_sum, kernel_name="in_training_reduce_v2"):
    """
    instancenorm operator interface implementation

    Parameters
    ----------
    x: dict
        shape and dtype of input x, only support float16, float32
    sum: dict
        shape and dtype of input sum, only support float32
    square_sum: dict
        shape and dtype of input square_sum, only support float32
    kernel_name: str
        cce kernel name, default value is "in_training_reduce_v2"

    Returns
    -------
    None
    """

    shape_x = x.get("shape")
    dtype_x = x.get("dtype")
    format_x = x.get("format")
    para_check.check_shape(shape_x, param_name="x")
    para_check.check_dtype(dtype_x.lower(), ("float16", "float32"), param_name="x")

    data_x = tvm.placeholder(shape_x, name="data_x", dtype=dtype_x.lower())

    sum_x, square_sum_x = in_training_reduce_compute(data_x, sum, square_sum, format_x, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule([sum_x, square_sum_x])

    config = {"name": kernel_name, "tensor_list": [data_x, sum_x, square_sum_x]}
    tbe.build(sch, config)
