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
logical_and
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
@register_operator_compute("logical_and", op_mode="static", support_fusion=True)
def logical_and_compute(x1, x2, y, kernel_name="logical_and"):
    """
    calculating data

    Parameters
    ----------
    x1 : TVM tensor
        the placeholder of x1
    x2: TVM tensor
        the placeholder of x2
    y : dict
        dict of output_y, include keys(shape and dtype)
    kernel_name : str
        kernel name, default value is "logical_and"

    Returns
    -------
    output tensor
    """
    shape_x = shape_util.shape_to_list(x1.shape)
    shape_y = shape_util.shape_to_list(x2.shape)
    _, _, shape_max = shape_util.broadcast_shapes(shape_x, shape_y,
                                                  param_name_input1="x1", param_name_input2="x2")

    x1 = tbe.cast_to(x1, "float16")
    x2 = tbe.cast_to(x2, "float16")

    data_x = tbe.broadcast(x1, shape_max)
    data_y = tbe.broadcast(x2, shape_max)

    res = tbe.vmul(data_x, data_y)

    res = tbe.cast_to(res, "int8", True)

    return res


# @register_operator("LogicalAnd")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def logical_and(x1, x2, y, kernel_name="logical_and"):
    """
    calculating data

    Parameters
    ----------
    x1 : dict
        shape and dtype of input, only support float16, float32
    x2 : dict
        shape and dtype of input, only support float16, float32
    y : dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        kernel name, default value is "logical_and"

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

    if dtype_x != dtype_y:
        error_detail = "dtype of x1 and x2 must be the same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "x1", "x2", error_detail)

    input_data_type = dtype_x.lower()
    check_tuple = ("int8",)
    para_check.check_dtype(input_data_type, check_tuple, param_name="x1")

    shape_x, shape_y, _ = shape_util.broadcast_shapes(shape_x, shape_y,
                                                      param_name_input1="x1", param_name_input2="x2")
    shape_x, shape_y = shape_util.refine_shapes_for_broadcast(shape_x, shape_y)
    data_x = tvm.placeholder(shape_x, dtype=dtype_x, name="data_x")
    data_y = tvm.placeholder(shape_y, dtype=dtype_y, name="data_y")

    res = logical_and_compute(data_x, data_y, y, kernel_name)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": (data_x, data_y, res)}

    tbe.build(sch, config)
