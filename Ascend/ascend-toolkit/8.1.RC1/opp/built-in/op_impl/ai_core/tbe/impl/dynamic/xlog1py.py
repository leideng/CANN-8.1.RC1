# Copyright (c) Huawei Technologies Co., Ltd.2022. All rights reserved.
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
dynamic xlog1py
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import error_manager_vector
from impl.dynamic.log1p import log1p_compute


@register_operator_compute("Xlog1py", op_mode="dynamic", support_fusion=True, support_bfp16=True)
def xlog1py_compute(x, y, z, kernel_name="xlog1py"):
    """
    calculating data's xlog1py, z = x * log1p(y + 1)

    Parameters
    ----------
    x: TVM tensor
        the placeholder of x
    y: TVM tensor
        the placeholder of y
    z: dict
        shape and dtype of output, should be broadcast shape and type as input
    kernel_name: str
        cce kernel name, default value is "xlog1py"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    shape_list = shape_util.broadcast_shapes(
        x.shape,
        y.shape,
        param_name_input1="input_x",
        param_name_input2="input_y"
    )

    shape = shape_list[2]
    x_dtype = x.dtype.lower()
    if x_dtype in ("float16",):
        x = tbe.cast_to(x, "float32")
        y = tbe.cast_to(y, "float32")

    y = tbe.broadcast(y, shape)
    data_log1p = log1p_compute(y, z)
    data_x_broad = tbe.broadcast(x, shape)
    res = tbe.vmul(data_x_broad, data_log1p)

    if x_dtype != res.dtype:
        res = tbe.cast_to(res, "float16")

    return res


# 'pylint: disable=too-many-locals
@register_operator("Xlog1py")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def xlog1py(x, y, z, kernel_name="xlog1py"):
    """
    algorithm: xlog1py
    calculating data's xlog1py, z = x * log1p(y + 1)

    Parameters
    ----------
    x: dict
       including shape, dtype and range, only support bfloat16, float16, float32
    y: dict
       including shape, dtype and range, only support bfloat16, float16, float32
    z: dict
       shape should be broadcast shape of input, and type equals to input
    kernel_name: str
       cce kernel name, default value is "xlog1py"

    Returns
    -------
    None
    """

    # check input tensor data_type
    x_dtype = x.get("dtype").lower()
    y_dtype = y.get("dtype").lower()
    check_list = ("bfloat16", "float16", "float32")
    para_check.check_dtype(x_dtype, check_list, param_name="x")
    para_check.check_dtype(y_dtype, check_list, param_name="y")
    
    para_check.check_elewise_shape_range([x, y], support_broadcast=True)
    if x_dtype != y_dtype:
        error_manager_vector.raise_err_inputs_dtype_not_equal("xlog1py", "x", "y",
                                                              str(x_dtype), str(y_dtype))

    ins = classify([x, y], OpPatternMode.ELEWISE_WITH_BROADCAST)
    schedules, tensors = [], []
    for (_input_x, _input_y) in ins:
        with tbe.compute():
            shape_x, shape_y = shape_util.variable_shape([_input_x, _input_y])
            x_data = tvm.placeholder(shape_x, name="data_1", dtype=x_dtype)
            y_data = tvm.placeholder(shape_y, name="data_2", dtype=y_dtype)
            res = xlog1py_compute(x_data, y_data, z, kernel_name)

            tensors.append((x_data, y_data, res))
        with tvm.target.cce():
            schedule = tbe.auto_schedule(res)
        schedules.append(schedule)

    config = {"print_ir": False, "name": kernel_name,
              "tensor_list": tensors}

    tbe.build(schedules, config)
