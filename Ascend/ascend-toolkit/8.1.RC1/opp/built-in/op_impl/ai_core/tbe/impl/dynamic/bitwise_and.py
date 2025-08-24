# Copyright 2021 Huawei Technologies Co., Ltd
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
bitwise_and
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=unused-argument,invalid-name
@register_operator_compute("BitwiseAnd", op_mode="dynamic", support_fusion=True)
def bitwise_and_compute(x1, x2, y, kernel_name="bitwise_and"):
    """
    calculating data's bitwise and
    res = x & y

    Parameters
    ----------
    x1 : tvm tensor
              input data x1
    x2 : tvm tensor
              input data x2
    y : dict
               the shape and dtype of the tensor y
    kernel_name : string
                  cce kernel name, default value is "bitwise_and"

    Returns
    -------
    res : output of the data's bitwise and
    """
    shape_x, shape_y, shape_max = shape_util.broadcast_shapes(x1.shape,
                                                              x2.shape,
                                                              param_name_input1="x1",
                                                              param_name_input2="x2")

    data_x = tbe.broadcast(x1, shape_max)
    data_y = tbe.broadcast(x2, shape_max)

    res = tbe.vand(data_x, data_y)

    return res


# 'pylint: disable=too-many-locals
@register_operator("BitwiseAnd")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def bitwise_and(x1, x2, y, kernel_name="bitwise_and"):
    """
    algorithm: bitwise_and
    computes the bitwise and of `x1` and `x2`

    Parameters
    ----------
    x1 : dict
              the shape and dtype of the tensor x1, only support int16, uint16, int32, int64
    x2 : dict
              the shape and dtype of the tensor x2, only support int16, uint16, int32, int64
    y : dict
              the shape and dtype of the tensor y, only support int16, uint16, int32, int64
    kernel_name : string
                  cce kernel name, default value is "bitwise_and"

    Returns
    -------
    None
    """
    check_list = ["int16", "uint16", "int32", "int64"]
    dtype_x = x1.get("dtype").lower()
    dtype_y = x2.get("dtype").lower()
    para_check.check_dtype(dtype_x, check_list, param_name="x1")
    para_check.check_dtype(dtype_y, check_list, param_name="x2")
    if dtype_x != dtype_y:
        error_detail = "dtype of x1 and x2 should be same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "x1", "x2", error_detail)

    schedules, tensors = [], []
    ins = classify([x1, x2], OpPatternMode.ELEWISE_WITH_BROADCAST)
    for (_x, _y) in ins:
        with tbe.compute():
            x_shape, y_shape = shape_util.variable_shape([_x, _y])
            data_x = tvm.placeholder(x_shape, name="data_x", dtype=dtype_x)
            data_y = tvm.placeholder(y_shape, name="data_y", dtype=dtype_x)
            res = bitwise_and_compute(data_x, data_y, y, kernel_name)
            tensors.append([data_x, data_y, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {
        "name": kernel_name,
        "tensor_list": tensors
    }
    tbe.build(schedules, config)
