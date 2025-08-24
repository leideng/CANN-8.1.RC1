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
bias_add_grad
"""
from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util import util_common
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=unused-argument
# 'pylint: disable=invalid-name,unnecessary-comprehension
def get_op_support_info(x, y, data_format, kernel_name="bias_add_grad"):
    """
    get_op_support_info
    """
    format_x = x.get("format").upper()
    if format_x == "FRACTAL_NZ":
        axis_split_matrix = [[SplitInput([0, [0], [-1], [-1]]), SplitOutput([0, [0]])]]

    else:
        axis_split_matrix = None
    axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name,too-many-branches
def bias_add_grad_compute_nz(x, y, data_format, kernel_name="bias_add_grad"):
    """
    Reduce a tensor on last dimension in axis based on sum.

    Parameters:
    ----------
    x: TVM tensor
        the placeholder of y input data ,dataformat = "NZ"
    y: dict
        shape and dtype of output, should be same shape and type as input
    data_format: str
        'NCHW' or 'NHWC'
    kernel_name : str
        cce kernel name, default value is bias_add_grad

    Returns
    -------
    TVM tensor by bias add grad
    """
    dtype = x.dtype
    y_dtype = y.get("dtype").lower()
    shape = shape_util.shape_to_list(x.shape)
    shape_list = []
    if dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.reduce_sum", "float32"):
        x = tbe.cast_to(x, "float32")

    if data_format == "NCHW":
        if len(shape) == 4:
            for i in range(-1 * len(shape), 0):
                if i not in (-1, -4):
                    shape_list += [i + len(shape)]
        elif len(shape) == 5:
            for i in range(-1 * len(shape), 0):
                if i not in (-2, -3):
                    shape_list += [i + len(shape)]
        else:
            shape_list.append(0)
            for i in range(2, len(shape)):
                shape_list = shape_list + [i]
    else:
        if len(shape) < 4:
            error_manager_vector.raise_err_specific_reson("bias_add_grad", "cce_bias_add_grad_nz_2_nhwc \
                                                          only support shape larger than 4D")
        for i in range(-1 * len(shape), 0):
            if i not in (-1, -4):
                shape_list += [i + len(shape)]

    result = tbe.sum(x, shape_list)
    result = tbe.cast_to(result, y_dtype)

    return result


def bias_add_grad_compute_special_format(x, y, data_format, kernel_name="bias_add_grad"):
    """
    Reduce a tensor based on sum for format is [FRACTAL_Z, FRACTAL_Z_3D, 5HD, 6HD]

    Parameters:
    ----------
    x: TVM tensor
        the placeholder of y input data
    y: dict
        shape and dtype of output, should be same shape and type as input
    data_format: str
        'NCHW' or 'NHWC'
    kernel_name : str
        cce kernel name, default value is bias_add_grad

    Returns
    -------
    TVM tensor by bias add grad
    """
    dtype = x.dtype
    y_dtype = y.get("dtype").lower()
    shape_list = []
    if dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.reduce_sum", "float32"):
        x = tbe.cast_to(x, "float32")

    if data_format == "FRACTAL_Z":
        # mean format is FRACTAL_Z, shape is C1HWNiNoC0
        shape_list = [1, 2, 3, 4]
    elif data_format == "FRACTAL_Z_3D":
        # mean format is FRACTAL_Z_3D, shape is DC1HWNiNoC0
        shape_list = [0, 2, 3, 4, 5]
    elif data_format == "NC1HWC0":
        # mean format is NC1HWC0, shape is NC1HWC0
        shape_list = [0, 2, 3]
    elif data_format == "NDC1HWC0":
        # mean format is NDC1HWC0, shape is NDC1HWC0
        shape_list = [0, 1, 3, 4]

    result = tbe.sum(x, shape_list)
    result = tbe.cast_to(result, y_dtype)

    return result


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
@register_operator_compute("bias_add_grad", op_mode="static", support_fusion=True)
def bias_add_grad_compute(x, y, data_format, kernel_name="bias_add_grad"):
    """
    Reduce a tensor on last dimension in axis based on sum.

    Parameters:
    ----------
    x: TVM tensor
        the placeholder of y input data
    y: dict
        shape and dtype of output, should be same shape and type as input
    data_format: str
        'NCHW' or 'NHWC'
    kernel_name : str
        cce kernel name, default value is bias_add_grad

    Returns
    -------
    TVM tensor by bias add grad
    """
    dtype = x.dtype
    y_dtype = y.get("dtype").lower()
    shape = shape_util.shape_to_list(x.shape)

    if dtype == "float16":
        x = tbe.cast_to(x, "float32")

    if data_format in ("NCHW", "NCDHW"):
        shape_list = [0]
        for i in range(2, len(shape)):
            shape_list = shape_list + [i]
        result = tbe.sum(x, shape_list)
    else:
        if len(shape) < 2:
            error_manager_vector.raise_err_specific_reson("bias_add_grad", "cce_bias_add_grad \
                                                          only support shape larger than 2D")
        result = tbe.sum(x, list(range(len(shape) - 1)))

    result = tbe.cast_to(result, y_dtype)

    return result


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_STR, para_check.KERNEL_NAME)
def bias_add_grad(x, y, data_format, kernel_name="bias_add_grad"):
    """
    Reduce a tensor on last dimension in axis based on sum.

    Parameters:
    ----------
    x : dict
        shape and dtype of input, only support float16, float32
    y: dict
        shape and dtype of output, should be same shape and type as input
    data_format: str
        'NCHW' or 'NHWC'
    kernel_name : str
        cce kernel name, default value is bias_add_grad
    Returns
    -------
    None
    """
    x = util_common.update_shape_base_other_format(x)
    y = util_common.update_shape_base_other_format(y)
    shape = x.get("shape")
    para_check.check_shape(shape, param_name="x")
    dtype = x.get("dtype").lower()
    data_format = data_format.upper()
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype, check_list, param_name="x")
    data_format_tuple = ("NCHW", "NHWC", "NDHWC", "NCDHW")
    input_data_format = x.get("format").upper()

    if data_format not in data_format_tuple:
        error_manager_vector.raise_err_input_format_invalid("bias_add_grad", "data_format",
                                                            "NCHW, NHWC, NDHWC, NCDHW", str(data_format))

    data = tvm.placeholder(shape, dtype, name="data")

    if input_data_format == "FRACTAL_NZ":
        result = bias_add_grad_compute_nz(data, y, data_format, kernel_name)
    elif input_data_format in ("FRACTAL_Z", "FRACTAL_Z_3D", "NC1HWC0", "NDC1HWC0"):
        result = bias_add_grad_compute_special_format(data, y, input_data_format, kernel_name)
    else:
        result = bias_add_grad_compute(data, y, data_format, kernel_name)

    with tvm.target.cce():
        sch = auto_schedule(result)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data, result]}
    build(sch, config)
