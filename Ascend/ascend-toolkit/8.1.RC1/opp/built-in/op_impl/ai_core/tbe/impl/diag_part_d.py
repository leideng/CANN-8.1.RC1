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
diag_part_d
"""
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
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
@register_operator_compute("diag_part_d", op_mode="static", support_fusion=True)
def diag_part_d_compute(x, assist, y, kernel_name="diag_part_d"):
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
    sum_dims = []
    len_output = len(shape_x) // Constant.VALUE_TWO
    for dims in range(len_output):
        sum_dims.append(dims + len_output)

    has_improve_precision = False
    if dtype_x == "int32" and \
            tbe_platform.api_check_support("tbe.dsl.reduce_sum", "float32"):
        res_vmul = tbe.cast_to(res_vmul, "float32")
        has_improve_precision = True

    res = tbe.sum(res_vmul, sum_dims)
    if has_improve_precision:
        res = tbe.cast_to_round(res, "int32")
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
    dtype_x = x.get("dtype")
    shape_assist = assist.get("shape")
    dtype_assist = assist.get("dtype")
    shape_y = y.get("shape")

    para_check.check_shape(shape_x, param_name="x")
    para_check.check_shape(shape_assist, param_name="assist")

    if len(shape_x) not in (2, 4, 6, 8):
        error_detail = "Input x of rank 2,4,6,8 are supported!"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "x", error_detail)
    if list(shape_x) != list(shape_assist):
        error_detail = "the shape of x and assist must be equal!"
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "x", \
                                                               "assist", error_detail)
    len_shape_out = len(shape_x) // Constant.VALUE_TWO
    for i in range(len_shape_out):
        if shape_x[i] != shape_x[i + len_shape_out]:
            error_detail = "the shape of input x is not supported!"
            error_manager_vector.raise_err_input_shape_invalid(kernel_name, "x", error_detail)
    if list(shape_x) != list(shape_y + shape_y):
        error_detail = "the shape of output y is not supported!"
        error_manager_vector.raise_err_input_shape_invalid(kernel_name, "y", error_detail)
    if list(shape_x) != list(shape_assist):
        error_detail = "the shape of x and assist must be equal!"
        error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "x", \
                                                               "assist", error_detail)

    check_list = ("float16", "float32", "int32")
    dtype_x = dtype_x.lower()
    para_check.check_dtype(dtype_x, check_list, param_name="x")
    dtype_assist = dtype_assist.lower()
    para_check.check_dtype(dtype_assist, check_list, param_name="assist")
    if dtype_assist != dtype_x:
        error_detail = "dtype of x and assist must be the same"
        error_manager_vector.raise_err_two_input_dtype_invalid(kernel_name, "x", "assist", error_detail)

    data_x = tvm.placeholder(shape_x, name="data_x",
                             dtype=dtype_x)
    data_assist = tvm.placeholder(shape_assist, name="data_assist",
                                  dtype=dtype_assist)

    res = diag_part_d_compute(data_x, data_assist, y,
                              kernel_name)
    with tvm.target.cce():
        sch = auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [data_x, data_assist, res]}
    build(sch, config)
