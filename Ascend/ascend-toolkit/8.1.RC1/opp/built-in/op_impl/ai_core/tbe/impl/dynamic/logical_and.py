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
logical_and
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=too-many-locals,invalid-name
@register_operator_compute("LogicalAnd", op_mode="dynamic", support_fusion=True)
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
    _, _, shape_max = shape_util.broadcast_shapes(x1.shape,
                                                  x2.shape,
                                                  param_name_input1="x1",
                                                  param_name_input2="x2")
    x1 = tbe.cast_to(x1, "float16")
    x2 = tbe.cast_to(x2, "float16")

    data_x = tbe.broadcast(x1, shape_max)
    data_y = tbe.broadcast(x2, shape_max)

    res = tbe.vmul(data_x, data_y)

    res = tbe.cast_to(res, "int8", True)

    return res


@register_operator("LogicalAnd")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
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
    dtype_x1 = x1.get("dtype").lower()
    dtype_x2 = x2.get("dtype").lower()
    if dtype_x1 == "bool" or dtype_x2 == "bool":
        dtype_x1 = "int8"
        dtype_x2 = "int8"

    check_tuple = ("int8",)
    para_check.check_dtype(dtype_x1, check_tuple, param_name="x1")
    para_check.check_dtype(dtype_x2, check_tuple, param_name="x2")

    ins = classify([x1, x2], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_x1, _x2) in ins:
        with tbe.compute():
            shape_x1, shape_x2 = shape_util.variable_shape([_x1, _x2])
            data_x1 = tvm.placeholder(shape_x1, name="data_x1", dtype=dtype_x1)
            data_x2 = tvm.placeholder(shape_x2, name="data_x2", dtype=dtype_x2)
            res = logical_and_compute(data_x1, data_x2, y, kernel_name)

            tensors.append((data_x1, data_x2, res))
        with tvm.target.cce():
            schedule = tbe.auto_schedule(res)
        schedules.append(schedule)

    config = {"print_ir": False, "need_build": False, "name": kernel_name,
              "tensor_list": tensors}
    tbe.build(schedules, config)
