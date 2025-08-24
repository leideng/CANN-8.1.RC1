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
squared_difference
"""
import te.lang.cce as tbe
from te.utils import para_check
from te.utils import shape_util
import te.platform as tbe_platform
from tbe import tvm
from te.utils.error_manager import error_manager_vector

SHAPE_SIZE_LIMIT = 2147483648


# 'pylint: disable=locally-disabled,too-many-locals,unused-argument
# 'pylint: disable=invalid-name,unused-variable
@tbe_platform.fusion_manager.fusion_manager.register("squared_difference")
def squared_difference_compute(data_x, data_y, y, kernel_name="squared_difference"):
    """
    squared_difference compute
    calculating data's squared_difference

    Parameters
    ----------
    data_x: TVM tensor
        the placeholder of data_x
    data_y: TVM tensor
        the placeholder of data_y
    y: dict
        dict with keys(shape and dtype) of output
    kernel_name: str
        kernel name, default value is "squared_difference"

    Returns
    -------
    res: TVM tensor
        the result of squared_difference compute
    """
    input_data1 = shape_util.shape_to_list(data_x.shape)
    input_data2 = shape_util.shape_to_list(data_y.shape)
    shape_list = shape_util.broadcast_shapes(input_data1, input_data2,
                                             param_name_input1="data_x",
                                             param_name_input2="data_y")
    data_x_broad = tbe.broadcast(data_x, shape_list[2])
    data_y_broad = tbe.broadcast(data_y, shape_list[2])

    data_sub = tbe.vsub(data_x_broad, data_y_broad)
    res = tbe.vmul(data_sub, data_sub)

    return res


# 'pylint: disable=locally-disabled,too-many-locals,invalid-name
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def squared_difference(x1, x2, y, kernel_name="squared_difference"):
    """
    algorithm: squared_difference

    calculating data's tf_squared_difference,y= (x - y) * (x - y)

    Parameters
    ----------
    x2 : dict
        shape and dtype of y input, only support float16, float32
    input_dy : dict
        shape and dtype of dy input, only support float16, float32
    output_x: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is squared_difference

    Returns
    -------
    None
    """
    shape_x = x1.get("shape")
    shape_y = x2.get("shape")
    para_check.check_shape(shape_x, param_name="x1")
    para_check.check_shape(shape_y, param_name="x2")

    check_list = ["float16", "float32", "int32"]
    dtype = x1.get("dtype").lower()
    para_check.check_dtype(dtype, check_list, param_name="x1")

    shape_x, shape_y, shape_max = \
        shape_util.broadcast_shapes(shape_x, shape_y, param_name_input1="x1", param_name_input2="x2")

    shape_x, shape_y = shape_util.refine_shapes_for_broadcast(shape_x, shape_y)
    data_x = tvm.placeholder(shape_x, dtype=dtype, name="data_x")
    data_y = tvm.placeholder(shape_y, dtype=dtype, name="data_y")

    res = squared_difference_compute(data_x, data_y, y, kernel_name)
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data_x, data_y, res]}

    tbe.cce_build_code(sch, config)
