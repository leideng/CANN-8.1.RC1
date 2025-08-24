# Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.
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
dynamic diag_part_d
"""
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute


# 'pylint: disable=too-few-public-methods,not-use-list-comprehension
class Constant:
    """
    The class for constant
    """
    # 'define a VALUE, value = 2
    VALUE_TWO = 2


# 'pylint: disable=locally-disabled,unused-argument,invalid-name,no-member
@register_operator_compute("DiagPartD", op_mode="dynamic", support_fusion=True)
def diag_part_d_compute(x, assist, sum_dims, y, kernel_name="diag_part_d"):
    """
    compute for diag_part_d

    Parameters
    ----------
    x: TVM tensor
        the placeholder of input diagonal
    assist: TVM tensor
        the placeholder of input help
    y: dict
        dict of y
    kernel_name: str
        cce kernel name, default value is "diag_part_d"

    Returns
    -------
    res: TVM tensor
        the result of compute
    """
    shape_x = shape_util.shape_to_list(x.shape)
    dtype_x = x.dtype

    res_vmul = tbe.vmul(x, assist)
    has_improve_precision = False
    if dtype_x == "int32" and tbe_platform.api_check_support("tbe.dsl.reduce_sum", "float32"):
        res_vmul = tbe.cast_to(res_vmul, "float32")
        has_improve_precision = True

    res = tbe.reduce_sum(res_vmul, sum_dims)
    if has_improve_precision:
        res = tbe.cast_to(res, "int32")
    return res


# 'pylint: disable=too-many-locals
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def diag_part_d(x, assist, y, kernel_name="diag_part_d"):
    """
    Returns the batched diagonal part of a batched tensor

    Parameters
    ----------
    x: dict
        dict of x, include keys(shape and dtype)
    assist: dict
        dict of help Matrix, Its Diagonal Line value is 1 else value is 0
    y: dict
        dict of output
    kernel_name: str
        cce kernel name, default value is "diag_part_d"

    Returns
    -------
    None
    """
    shape_x = x.get("shape")
    dtype_x = x.get("dtype").lower()
    shape_assist = assist.get("shape")
    dtype_assist = assist.get("dtype").lower()
    shape_y = y.get("shape")

    para_check.check_shape(shape_x, param_name="x")
    para_check.check_shape(shape_assist, param_name="assist")

    check_list = ("float16", "float32", "int32")
    para_check.check_dtype(dtype_x, check_list, param_name="x")
    para_check.check_dtype(dtype_assist, check_list, param_name="assist")
    if dtype_assist != dtype_x:
        error_detail = "dtype of x and assist must be the same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "x", "assist", error_detail)

    assist["rel_pos_to_reduce"] = "before"
    sum_dims = []
    len_output = len(shape_x) // Constant.VALUE_TWO
    for dims in range(len_output):
        sum_dims.append(dims + len_output)
    input_axis = {"shape": [len(sum_dims)], "value": sum_dims, "rel_pos_to_reduce": "axis"}

    ins = classify([x, assist, input_axis], OpPatternMode.REDUCE, {"keepdims": False})
    schedules, tensors = [], []
    for (_x, _assist, _axes) in ins:
        with tbe.compute():
            shape_x, shape_assist = shape_util.variable_shape([_x, _assist, _axes], op_mode="reduce")[0:2]
            x_input = tvm.placeholder(shape_x, name="x_input", dtype=dtype_x)
            assist_input = tvm.placeholder(shape_assist, name="assist_input", dtype=dtype_assist)

            res = diag_part_d_compute(x_input, assist_input, _axes.get("value"), y, kernel_name=kernel_name)
            tensors.append([x_input, assist_input, res])
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)

