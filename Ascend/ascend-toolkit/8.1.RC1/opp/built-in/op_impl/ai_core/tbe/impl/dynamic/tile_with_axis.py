# Copyright 2021Huawei Technologies Co., Ltd
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
from impl.util.util_common import is_unknown_rank_input
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import error_manager_vector
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info
from impl.util.util_compute import only_static_support


# 'pylint: disable = unused-argument
def get_op_support_info(input_x, output_y, tiles, axis=1, kernel_name="tile_with_axis"):
    """
    get_op_support_info
    """
    shape_x_len = len(input_x.get("shape"))
    format_x = input_x.get("format").upper()
    if axis < 0:
        axis += shape_x_len
    if format_x in ("NC1HWC0", "ND"):
        axis_split_matrix = []
        for i in range(0, shape_x_len - 1):
            if i != axis:
                split_0 = [SplitInput([0, [i], [-1], [-1]]), SplitOutput([0, [i]])]
                axis_split_matrix.append(split_0)
        axis_reduce_list = None

    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


# 'pylint: disable=unused-argument
def op_select_format(input_x, output_y, tiles, axis=1, kernel_name="tile_with_axis"):
    """
    select format dynamically
    """
    ori_format = input_x.get("ori_format")
    ori_shape = input_x.get("ori_shape")

    if not is_unknown_rank_input(input_x) and ori_shape:
        axis = shape_util.axis_check(len(ori_shape), axis)

    cce_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")

    # for 5hd, axis is only valid for n,h,w
    if ((ori_format == "NHWC" and axis != 3) or (ori_format == "NCHW" and axis != 1)) and \
            len(ori_shape) == 4:
        # NC1HWC0+ND
        if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
            # fp16
            input0 = gen_param(classify="input0",
                               name="x",
                               datatype="float16,int8,int32,uint8,"
                               "float16,int8,int32,uint8",
                               format="ND,ND,ND,ND,"
                               "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0")
            output0 = gen_param(classify="output0",
                                name="y",
                                datatype="float16,int8,int32,uint8,"
                                "float16,int8,int32,uint8",
                                format="ND,ND,ND,ND,"
                                "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0")
        else:
            # fp16/fp32
            input0 = gen_param(classify="input0",
                               name="x",
                               datatype="bfloat16,float16,float32,int8,int32,uint8,"
                               "bfloat16,float16,float32,int8,int32,uint8",
                               format="ND,ND,ND,ND,ND,ND,"
                               "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0")
            output0 = gen_param(classify="output0",
                                name="y",
                                datatype="bfloat16,float16,float32,int8,int32,uint8,"
                                "bfloat16,float16,float32,int8,int32,uint8",
                                format="ND,ND,ND,ND,ND,ND,"
                                "NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0,NC1HWC0")
    else:
        # ND
        if cce_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
            # fp16
            input0 = gen_param(classify="input0", name="x", datatype="float16,int8,int32,uint8", format="ND,ND,ND,ND")
            output0 = gen_param(classify="output0", name="y", datatype="float16,int8,int32,uint8", format="ND,ND,ND,ND")
        else:
            # fp16/fp32
            input0 = gen_param(classify="input0",
                               name="x",
                               datatype="bfloat16,float16,float32,int8,int32,uint8",
                               format="ND,ND,ND,ND,ND,ND")
            output0 = gen_param(classify="output0",
                                name="y",
                                datatype="bfloat16,float16,float32,int8,int32,uint8",
                                format="ND,ND,ND,ND,ND,ND")

    param_list = [input0, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def check_axis(input_x, shape_x_len, axis):
    input_format = input_x.get("format")
    if input_format == "NC1HWC0":
        shape_x_ori = input_x.get("ori_shape")
        ori_format = input_x.get("ori_format")
        length_x_ori = len(shape_x_ori)
        if ori_format not in ("NCHW", "NHWC"):
            error_manager_vector.raise_err_specific_reson(
                "tile_with_axis", "input_x's ori_format is invalid for 5D Tensor")
        if shape_x_len != 5:
            error_manager_vector.raise_err_specific_reson(
                "tile_with_axis", "input_x's shape is invalid for 5D Tensor")
        if length_x_ori != 4:
            error_manager_vector.raise_err_specific_reson(
                "tile_with_axis", "input_x's ori_shape is invalid for 5D Tensor")
        axis = shape_util.axis_check(length_x_ori, axis)
        axis = shape_util.axis_transform_5d(axis, ori_format)
        if axis in (1, 4):
            error_manager_vector.raise_err_specific_reson("tile_with_axis", "axis is invalid for 5D Tensor")
    else:
        if axis >= shape_x_len or axis < -shape_x_len:
            error_manager_vector.raise_err_input_value_invalid("tile_with_axis", "axis",
                                                               "in range of [ {} , {} ]".format(-shape_x_len, \
                                                               shape_x_len - 1), str(axis))
        if axis < 0:
            axis += shape_x_len

    return axis


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=C0103
@register_operator_compute("TileWithAxis", op_mode="dynamic", support_fusion=only_static_support)
def tile_with_axis_compute(x, shape_y):
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
    src_dtype = x.dtype.lower()
    if src_dtype == "int8" or src_dtype == "uint8":
        x = tbe.cast_to(x, "float16")
    res = tbe.broadcast(x, shape_y)
    if src_dtype == "int8" or src_dtype == "uint8":
        res = tbe.cast_to(res, src_dtype)
    return res


# 'pylint: disable=too-many-locals,too-many-branches,too-many-statements
# 'pylint: disable=C0103
@register_operator("TileWithAxis")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.KERNEL_NAME)
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
    shape_x = list(input_x.get("shape"))
    output_x = input_x.copy()
    output_x_shape = shape_x.copy()
    is_unknown_rank = is_unknown_rank_input(input_x)

    # check dtype
    dtype_x = input_x.get("dtype").lower()
    check_list = ["int8", "int32", "uint8", "float16", "bfloat16", "float32"]
    para_check.check_dtype(dtype_x, check_list, param_name="input_x")

    # check tiles
    if not is_unknown_rank and tiles is not None and tiles <= 0:
        error_manager_vector.raise_err_input_value_invalid("tile_with_axis", "tiles", "more than 1", str(tiles))

    # check shape for 5HD
    if not is_unknown_rank:
        shape_x_len = len(shape_x)
        axis = check_axis(input_x, shape_x_len, axis)
    
    if not is_unknown_rank:
        shape_range_x = list(input_x.get("range"))
        output_x_range = list(output_x.get("range"))

        if shape_x[axis] == 1:
            output_x_shape[axis] = tiles
            output_x_range[axis] = [tiles, tiles]
        else:
            shape_x.insert(axis, 1)
            shape_range_x.insert(axis, [1, 1])
            output_x_shape.insert(axis, tiles)
            output_x_range.insert(axis, [tiles, tiles])


        input_x["shape"] = shape_x
        input_x["range"] = shape_range_x
        output_x["shape"] = output_x_shape
        output_x["range"] = output_x_range

    # classify
    schedules, tensors = [], []

    extra_params = {"disable_optimization": True}
    ins = classify([input_x, output_x], OpPatternMode.ELEWISE_WITH_BROADCAST, extra_params)

    for (_x, _y) in ins:
        with tbe.compute():
            shape_x, shape_y = shape_util.variable_shape([_x, _y])
            data_x = tvm.placeholder(shape_x, name="data_x", dtype=dtype_x)
            res = tile_with_axis_compute(data_x, shape_y)
            tensors.append([data_x, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}

    tbe.build(schedules, config)
