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
data_format_dim_map
"""

from tbe.dsl.api import build
from tbe.dsl.api import auto_schedule
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.platform_adapter import error_manager_vector
from impl.util.util_select_op_base import get_op_cal_info


# 'pylint: disable=too-few-public-methods,not-use-list-comprehension
class Constant:
    """
    Constant
    """
    MOD_RHS = 4
    QUARTER = 0.25


# 'pylint: disable = unused-argument,invalid-name
def get_op_support_info(x,
                        y,
                        src_format="NHWC",
                        dst_format="NCHW",
                        kernel_name="data_format_dim_map"):
    """
    get_op_support_info
    """
    format_x = x.get("format").upper()
    shape_x_len = len(x.get("shape"))
    if format_x in ("ND", "NCHW", "NHWC"):
        axis_split_matrix = []
        for i in range(0, shape_x_len):
            split_0 = [SplitInput([0, [i], [-1], [-1]]), SplitOutput([0, [i]])]
            axis_split_matrix.append(split_0)
        axis_reduce_list = None

    elif format_x == "NC1HWC0":
        axis_split_matrix = [
            [SplitInput([0, [0], [-1], [-1]]), SplitOutput([0, [0]])],
            [SplitInput([0, [2], [-1], [-1]]), SplitOutput([0, [2]])],
            [SplitInput([0, [3], [-1], [-1]]), SplitOutput([0, [3]])]
        ]
        axis_reduce_list = None

    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


def _data_format_dim_map_mod(data):
    """
    mod function based on TF

    Parameters
    ----------
    data: the shape of input, only support int32.

    Returns
    -------
    data mod by 4
    """

    data = tbe.cast_to(data, 'float16')
    data = tbe.vadds(data, Constant.MOD_RHS)
    data_div_4 = tbe.vmuls(data, Constant.QUARTER)
    data_floor = tbe.floor(data_div_4)
    data_floor = tbe.cast_to(data_floor, 'float16')
    data_mul_4 = tbe.vmuls(data_floor, Constant.MOD_RHS)
    data_mod = tbe.vsub(data, data_mul_4)
    return data_mod


def _dimension_index(data_mod, ind):
    """
    dimension index function

    Parameters
    ----------
    data_mod: the data after modulo
    ind: mapping of index

    Returns
    -------
    dimension index
    """

    is_zero = tbe.vcmp(data_mod, 0., 'eq')
    is_one = tbe.vcmp(data_mod, 1., 'eq')
    is_two = tbe.vcmp(data_mod, 2., 'eq')
    return tbe.cast_to(tbe.vsel(is_zero, ind[0], \
                                tbe.vsel(is_one, ind[1], \
                                         tbe.vsel(is_two, ind[2], ind[3]))), \
                       "int32")


# 'pylint: disable=locally-disabled,unused-argument,invalid-name
@register_operator_compute("data_format_dim_map", op_mode="static", support_fusion=True)
def _data_format_dim_map_compute(x,
                                 y,
                                 src_format='NHWC',
                                 dst_format='NCHW',
                                 kernel_name='data_format_dim_map'):
    """
    Parameters
    ----------
    x: the dict of input, only support int32.
    y : the dict of y, reserved parameter, not used now.
    src_format : the original type of x, default value is "NHWC" (optional).
    dst_format : the aim type of x, default value is "NCHW" (optional).
    kernel_name : cce kernel name, default value is "data_format_dim_map" (optional).

    Returns
    -------
    Tensor after dataformatdimmap compute
    """

    data_mod = _data_format_dim_map_mod(x)
    src_format = src_format.upper()
    dst_format = dst_format.upper()

    ind = [0] * len(src_format)
    for i, src in enumerate(src_format):
        for j, dst in enumerate(dst_format):
            if src == dst:
                ind[i] = j
                break

    return _dimension_index(data_mod, ind)


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_STR,
                            para_check.KERNEL_NAME)
def data_format_dim_map(x,
                        y,
                        src_format="NHWC",
                        dst_format="NCHW",
                        kernel_name="data_format_dim_map"):
    """
    Returns the dimension index in the destination data format given the one in.

    Parameters
    ----------
    x : A Tensor with each element as a dimension index in source data format.
        Must be the following types: `int32`. Must be in the range [-4, 4).
    y : Shape and dtype of y, reserved parameter, not used now.
    src_format : An optional `string`. Defaults to `"NHWC"`. source data format.
    dst_format : An optional `string`. Defaults to `"NCHW"`. destination data format.
    kernel_name : CCE kernel name, default value is "data_format_dim_map" (optional).

    Returns
    -------
    None
    """

    shape_input = x.get("shape")
    dtype_input = x.get("dtype")

    # check kernel name, shape, size, dtype
    para_check.check_shape(shape_input, param_name="x")
    shape_input, _ = shape_util.refine_shape_axes(shape_input, [])
    check_list = ("int32", )
    dtype_input = dtype_input.lower()
    para_check.check_dtype(dtype_input, check_list, param_name="x")

    # check length of format
    if len(src_format) != 4:
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "src_format", \
                                                           "length 4", len(src_format))

    if len(dst_format) != 4:
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "dst_format", \
                                                           "length 4", len(dst_format))
    # get data and compute
    data_input = tvm.placeholder(shape_input,
                                 dtype=dtype_input,
                                 name="data_input")
    res = _data_format_dim_map_compute(data_input, y, src_format, dst_format,
                                       kernel_name)

    with tvm.target.cce():
        sch = auto_schedule(res)
    config = {
        "name": kernel_name,
        "print_ir": False,
        "tensor_list": (data_input, res),
        "bool_storage_as_1bit": False
    }
    build(sch, config)
