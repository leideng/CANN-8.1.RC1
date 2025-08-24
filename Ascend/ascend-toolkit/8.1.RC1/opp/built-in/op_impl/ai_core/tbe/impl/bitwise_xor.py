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
bitwise_xor
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
# 'pylint: disable=invalid-name,too-many-locals
@register_operator_compute("bitwise_xor", op_mode="static", support_fusion=True)
def bitwise_xor_compute(x1, x2, y, kernel_name="bitwise_xor"):
    """
    calculating data's bitwise xor
    (x&y)|!(x|y)

    Parameters
    ----------
    x1 : tvm tensor
              input data x
    x2 : tvm tensor
              input data y
    y : dict
               the shape and dtype of the tensor
    kernel_name : string
                  cce kernel name, default value is "bitwise_and"

    Returns
    -------
    result : y of the data's bitwise xor
    """
    shape_x = shape_util.shape_to_list(x1.shape)
    shape_y = shape_util.shape_to_list(x2.shape)
    shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x,
                                                              shape_y,
                                                              param_name_input1="x1",
                                                              param_name_input2="x2")

    data_x = tbe.broadcast(x1, shape_max)
    data_y = tbe.broadcast(x2, shape_max)

    data_and = tbe.vand(data_x, data_y)
    data_not = tbe.vnot(data_and)
    data_or = tbe.vor(data_x, data_y)
    result = tbe.vand(data_or, data_not)

    return result


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def bitwise_xor(x1, x2, y, kernel_name="bitwise_xor"):
    """
    algorithm: bitwise_xor
    calculating: gradient of bitwise_xor

    Parameters
    ----------
    x1 : dict
              the shape and dtype of the tensor x1
    x2 : dict
              the shape and dtype of the tensor x2
    y :  dict
              the shape and dtype of the tensor y
    kernel_name : string
                  cce kernel name, default value is "bitwise_xor"
    Returns
    -------
    None
    """
    shape_x = x1.get("shape")
    shape_y = x2.get("shape")
    dtype_x = x1.get("dtype").lower()
    dtype_y = x2.get("dtype").lower()

    para_check.check_shape(shape_x, param_name="x1")
    para_check.check_shape(shape_y, param_name="x2")

    check_tuple = ("int16", "uint16", "int32")
    input_data_type = dtype_x.lower()
    para_check.check_dtype(input_data_type, check_tuple, param_name="x1")

    if dtype_x != dtype_y:
        error_detail = "dtype of x1 and x2 should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "x1", "x2", error_detail)

    shape_x, shape_y, _ = shape_util.broadcast_shapes(shape_x, shape_y,
                                                      param_name_input1="x1",
                                                      param_name_input2="x2")
    shape_x, shape_y = shape_util.refine_shapes_for_broadcast(shape_x, shape_y)

    if input_data_type == "int32":
        input_data_type = "int16"
        shape_x.append(2)
        shape_y.append(2)

    data_x = tvm.placeholder(shape_x, dtype=input_data_type, name="data_x")
    data_y = tvm.placeholder(shape_y, dtype=input_data_type, name="data_y")

    result = bitwise_xor_compute(data_x, data_y, y, kernel_name)

    with tvm.target.cce():
        sch = auto_schedule(result)

    config = {
        "name": kernel_name,
        "tensor_list": [data_x, data_y, result]}
    build(sch, config)
