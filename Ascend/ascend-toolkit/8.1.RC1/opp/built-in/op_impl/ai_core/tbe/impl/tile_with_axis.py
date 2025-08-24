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
tile_with_axis
"""
# 'pylint: disable=import-error
import te.lang.cce as tbe
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import error_manager_vector
from impl.dynamic.tile_with_axis import get_op_support_info as static_get_op_support_info
from impl.dynamic.tile_with_axis import op_select_format as static_op_select_format


# 'pylint: disable = unused-argument
def get_op_support_info(input_x, output_y, tiles, axis=1, kernel_name="tile_with_axis"):
    """
    get_op_support_info
    """
    return static_get_op_support_info(input_x, output_y, tiles, axis, kernel_name)


# 'pylint: disable=unused-argument
def op_select_format(input_x, output_y, tiles, axis=1, kernel_name="tile_with_axis"):
    """
    select format dynamically
    """
    return static_op_select_format(input_x, output_y, tiles, axis, kernel_name)


@register_operator_compute("tile_with_axis", op_mode="static", support_fusion=True)
def tile_with_axis_compute(data, shape_y):
    """TVM calculation process, used for fusion operation.

    Parameters
    ----------
    data: list of placeholders.
        Input data.
    shape_y: tuple or list.
        The shape of output.

    Returns
    -------
    res the compute results
    """
    res = tbe.broadcast(data, shape_y)

    return res


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_INT, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def tile_with_axis(input_x, output_y, tiles, axis=1, kernel_name="tile_with_axis"):
    """
    algorithm: tile.
    Expanding the input tensor according to a specified dimension,
    and the expansion multiple is specified by the tiles param.
    For example, tiling [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11,
    12]]], which shape is (2, 3, 2), by axis:1 and tiles:2 produces
    [[[1, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6]],
    [[7, 8], [9, 10], [11, 12], [7, 8], [9, 10], [11, 12]]]
    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same type as input
    axis: int
         The index of the axis to tile
    tiles: int
        The number of copies (tiles) of the blob to output.
    kernel_name : str
        kernel name, default value is "tile_with_axis"

    Returns
    -------
    tik_instance
    """

    axis, shape_x, shape_y, dtype_x = check_param(input_x, output_y, tiles,
                                                  axis, kernel_name)

    input_data = tvm.placeholder(shape_x, name="input_data", dtype=dtype_x)

    if tiles > 1:
        res = tile_with_axis_compute(input_data, shape_y)
    else:
        zero_data = tvm.const(0, dtype=dtype_x)
        res = tbe.vadds(input_data, zero_data)

    with tvm.target.cce():
        sch = tbe.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [input_data, res]}

    tbe.cce_build_code(sch, config)


# 'pylint: disable=too-many-locals,too-many-branches,too-many-statements
def check_param_range(param_name, min_value, max_value, real_value, op_name='tile_with_axis'):
    """
    check_param_range
    """
    error_manager_vector.raise_err_input_param_range_invalid("tile_with_axis", param_name,
                                                             str(min_value), str(max_value), str(real_value))


# 'pylint: disable=too-many-locals,too-many-branches,too-many-statements
def check_param(input_x, output_y, tiles, axis, kernel_name):
    """
    Check the input parameter

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_y : dict
        shape and dtype of output, should be same type as input
    axis: int
         The index of the axis to tile
    tiles: int
        The number of copies (tiles) of the blob to output.
    kernel_name : str
        kernel name, default value is "tile_with_axis"

    Returns
    ----------
    axis: int
         The index of the axis to tile which is adjusted to positive
    """
    shape_x = input_x.get("shape")
    dtype_x = input_x.get("dtype").lower()
    shape_y = output_y.get("shape")
    dtype_y = output_y.get("dtype").lower()

    para_check.check_shape(shape_x, param_name="input_x")
    para_check.check_shape(shape_y, param_name="input_y")

    check_list = ["int8", "int16", "int32", "uint8", "uint16",
                  "uint32", "float16", "float32"]

    para_check.check_dtype(dtype_x, check_list, param_name="input_x")
    para_check.check_dtype(dtype_y, check_list, param_name="input_x")

    if dtype_x != dtype_y:
        error_manager_vector.raise_err_inputs_dtype_not_equal("tile_with_axis", "x", "y",
                                                              str(dtype_x), str(dtype_y))

    if tiles <= 0:
        check_param_range('tiles', 1, 'inf', tiles)

    shape_x_len = len(shape_x)

    # check for 5HD
    input_format = input_x.get("format")
    if input_format == "NC1HWC0":
        shape_x_ori = input_x.get("ori_shape")
        ori_format = input_x.get("ori_format")
        length_x_ori = len(shape_x_ori)

        if ori_format not in ("NCHW", "NHWC"):
            error_manager_vector.raise_err_specific_reson("tile_with_axis", "input_x's ori_format \
                                                          is invalid for 5D Tensor")
        if shape_x_len != 5:
            error_manager_vector.raise_err_specific_reson("tile_with_axis", "input_x's shape is \
                                                          invalid for 5D Tensor")
        if length_x_ori != 4:
            error_manager_vector.raise_err_specific_reson("tile_with_axis", "input_x's ori_shape \
                                                          is invalid for 5D Tensor")
        axis = shape_util.axis_check(length_x_ori, axis)
        axis = shape_util.axis_transform_5d(axis, ori_format)
        if axis in (1, 4):
            error_manager_vector.raise_err_specific_reson("tile_with_axis", "axis is invalid for 5D Tensor")
    else:
        if axis >= shape_x_len or axis < -shape_x_len:
            check_param_range('axis', -shape_x_len, shape_x_len-1, axis)

        if axis < 0:
            axis += shape_x_len

    shape_y_expected = [0] * shape_x_len
    shape_y_expected[0:shape_x_len] = shape_x[0:shape_x_len]
    shape_y_expected[axis] *= tiles

    if not check_same_shape(shape_y, shape_y_expected):
        error_manager_vector.raise_err_input_value_invalid("tile_with_axis", "output_y", "the parameter[shape_y] \
                                                           should be [{}],  but actually is \
                                                           [{}].".format(str(shape_y_expected), str(shape_y)))

    shape_x_adapt = []
    shape_y_adapt = []
    for i in range(shape_x_len):
        if i == axis:
            shape_x_adapt.append(1)
            shape_y_adapt.append(tiles)
            if shape_x[i] == 1:
                continue
        shape_x_adapt.append(shape_x[i])
        shape_y_adapt.append(shape_x[i])

    return [axis, shape_x_adapt, shape_y_adapt, dtype_x]


def check_same_shape(shape_x, shape_y):
    """
    check shape_x is the same shape as shape_y

    Parameters
    ----------
    shape_x: a tuple or list
    shape_y: a tuple or list

    Returns
    -------
    boolean: True, if the same shape otherwise False
    """
    shape_x_len = len(shape_x)
    shape_y_len = len(shape_y)

    if shape_x_len != shape_y_len:
        return False

    for i in range(shape_x_len):
        if shape_x[i] != shape_y[i]:
            return False

    return True
