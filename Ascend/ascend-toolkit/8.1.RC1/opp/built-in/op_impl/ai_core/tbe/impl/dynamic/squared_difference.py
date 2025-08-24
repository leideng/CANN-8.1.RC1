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
squared_difference
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=locally-disabled,too-many-locals,unused-argument
# 'pylint: disable=invalid-name,unused-variable
@register_operator_compute("SquaredDifference", op_mode="dynamic", support_fusion=True, support_bfp16=True)
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
    shape_list = shape_util.broadcast_shapes(data_x.shape, data_y.shape,
                                             param_name_input1="data_x",
                                             param_name_input2="data_y")
    data_x_broad = tbe.broadcast(data_x, shape_list[2])
    data_y_broad = tbe.broadcast(data_y, shape_list[2])

    data_sub = tbe.vsub(data_x_broad, data_y_broad)
    res = tbe.vmul(data_sub, data_sub)

    return res


# 'pylint: disable=locally-disabled,too-many-locals,invalid-name
@register_operator("SquaredDifference")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def squared_difference(x1, x2, y, kernel_name="squared_difference"):
    """
    algorithm: squared_difference

    calculating data's tf_squared_difference,y= (x - y) * (x - y)

    Parameters
    ----------
    x1 : dict
        shape and dtype of x1 input, only support bfloat16, float16, float32
    x2 : dict
        shape and dtype of x2 input, only support bfloat16, float16, float32
    y: dict
        shape and dtype of y, should be same shape and type as input
    kernel_name : str
        cce kernel name, default value is squared_difference

    Returns
    -------
    None
    """
    check_list = ["bfloat16", "float16", "float32", "int32", "int64"]
    x1_dtype = x1.get("dtype").lower()
    x2_dtype = x2.get("dtype").lower()
    para_check.check_dtype(x1_dtype, check_list, param_name="x1")
    para_check.check_dtype(x2_dtype, check_list, param_name="x2")
    if x1_dtype != x2_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "x1", "x2",
                                                              x1_dtype, x2_dtype)

    ins = classify([x1, x2], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_x1, _x2) in ins:
        with tbe.compute():
            x1_shape, x2_shape = shape_util.variable_shape([_x1, _x2])
            tensor_x1 = tvm.placeholder(x1_shape, x1_dtype, "tensor_x1")
            tensor_x2 = tvm.placeholder(x2_shape, x2_dtype, "tensor_x2")

            res = squared_difference_compute(tensor_x1, tensor_x2, y, kernel_name)
            tensors.append([tensor_x1, tensor_x2, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)
    # build
    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
