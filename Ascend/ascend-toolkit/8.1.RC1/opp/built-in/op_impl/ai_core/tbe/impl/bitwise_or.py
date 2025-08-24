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
bitwise_or
"""
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals,invalid-name
@register_operator_compute("bitwise_or", op_mode="static", support_fusion=True)
def bitwise_or_compute(placeholders, shape_x, shape_y):
    """
    calculating data's element_or, c = a | b

    Parameters
    ----------
    placeholders : tuple of data
    shape_x: list of int
            shape of input_x
    shape_y: list of int
            shape of input_y

    Returns
    -------
    res : z of the data's bitwise_or
    """
    data_x = placeholders[0]
    data_y = placeholders[1]
    shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x,
                                                              shape_y,
                                                              param_name_input1="x1",
                                                              param_name_input2="x2")
    data_x_broadcast = tbe.broadcast(data_x, shape_max)
    data_y_broadcast = tbe.broadcast(data_y, shape_max)
    res = tbe.vor(data_x_broadcast, data_y_broadcast)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def bitwise_or(x1, x2, y, kernel_name="bitwise_or",):
    """
    algorithm: bitwise_or
    calculating data's bitwise_or, c = a | b

    Parameters
    ----------
    x1: dict
              shape and dtype of data_1
    x2: dict
              shape and dtype of data_2
    y: dict
              shape and dtype of y
    kernel_name : string
                  cce kernel name, default value is "bitwise_or"

    Returns
    -------
    None
    """
    shape_x = x1.get("shape")
    shape_y = x2.get("shape")
    dtype_x = x1.get("dtype")
    dtype_y = x2.get("dtype")

    para_check.check_shape(shape_x, param_name="x1")
    para_check.check_shape(shape_y, param_name="x2")

    check_tuple = ("int16", "uint16", "int32")
    input_data_type = dtype_x.lower()
    para_check.check_dtype(input_data_type, check_tuple, param_name="x1")

    if dtype_x != dtype_y:
        error_detail = "dtype of x1 and x2 should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "x1", "x2", error_detail)

    shape_x, shape_y, _ = shape_util.broadcast_shapes(shape_x,
                                                      shape_y,
                                                      param_name_input1="x1",
                                                      param_name_input2="x2")
    shape_x, shape_y = shape_util.refine_shapes_for_broadcast(shape_x, shape_y)

    if input_data_type == "int32":
        input_data_type = "int16"
        shape_x.append(2)
        shape_y.append(2)

    data_x = tvm.placeholder(shape_x, dtype=input_data_type, name="data_x")
    data_y = tvm.placeholder(shape_y, dtype=input_data_type, name="data_y")
    res = bitwise_or_compute((data_x, data_y), shape_x, shape_y)
    y = {'shape': res.shape, 'dtype': input_data_type}

    with tvm.target.cce():
        schedule = auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": (data_x, data_y, res)}

    build(schedule, config)
