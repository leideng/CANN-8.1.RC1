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
square_sum_v1
"""
from impl.util.util_common import is_unknown_rank_input
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import OpImplMode
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.util.util_common import update_axis_for_other_format
from impl.util.platform_adapter import tbe_context


# 'pylint: disable = unused-argument
def get_op_support_info(input_x, output1, axis, keep_dims=True, kernel_name="square_sum_v1"):
    """
    get_op_support_info
    """
    x_shape = shape_util.shape_to_list(input_x.get("shape"))
    axis_d = []
    axis_split = []
    for i, _ in enumerate(x_shape):
        axis_d.append(i)
    x_format = input_x.get("format").upper()
    if axis is None:
        axis = []
    for i in axis_d:
        if i not in axis:
            axis_split.append(i)
    if x_format == "ND":
        if keep_dims:
            axis_split_matrix = []
            for i in axis_split:
                split_0 = [SplitInput([0, [i], [-1], [-1]]), SplitOutput([0, [i]])]
                axis_split_matrix.append(split_0)
            axis_reduce_list = None
        else:
            axis_split_matrix = None
            axis_reduce_list = None
    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


def op_select_format(input_x, output1, axis, keep_dims, kernel_name="square_sum_v1"):
    """
    select format dynamically
    op_select_format support desc:
    1. input_format always support 'ND'
    2. when ori_format is 'HWCN', input_format support 'FRACTAL_Z' or 'FRACTAL_NZ' in compile_static process
        for example:
            ori:
                input_x              shape = [5,5,16,16]           format = 'HWCN'
                output1              shape = []                    format = 'ND'
            format transformer:
                input_x              shape = [25,1,16,16]          format = 'FRACTAL_Z'
                output1              shape = []                    format = 'ND'
            ---------------------------------------------------------------------------
            ori:
                input_x              shape = [16,16]               format = 'ND'
                output1              shape = []                    format = 'ND'
            format transformer:
                input_x              shape = [1,1,16,16]          format = 'FRACTAL_NZ'
                output1              shape = []                    format = 'ND'

    """
    x_dtype = "float16, float"
    format_input = "ND, ND"
    format_output = "ND, ND"
    bfp16_support = tbe_platform.intrinsic_check_support("Intrinsic_vconv", "bf162f32")
    if bfp16_support:
        x_dtype = x_dtype + ", bfloat16"
        format_input = format_input + ", ND"
        format_output = format_output + ", ND"
    shape_ori = input_x.get("ori_shape")
    format_ori = input_x.get("ori_format")
    if axis == []:
        axis = [i for i in range(len(shape_ori))]
    if axis is not None:
        if format_ori in ("HWCN",) and len(shape_ori) == 4 and shape_ori[-1] % 16 == 0 and \
            shape_ori[-2] % 16 == 0 and list(axis) == [0, 1, 2, 3]:
            x_dtype = "float16, float, float16, float"
            format_input = "ND, ND, FRACTAL_Z, FRACTAL_Z"
            format_output = "ND, ND, ND, ND"
        if len(shape_ori) >= 2 and shape_ori[-1] % 16 == 0 and shape_ori[-2] % 16 == 0 and \
        list(axis) == [i for i in range(len(shape_ori))]:
            x_dtype = x_dtype + ", float16, float"
            format_input = format_input + ", FRACTAL_NZ, FRACTAL_NZ"
            format_output = format_output + ", ND, ND"
        if bfp16_support:
            x_dtype = x_dtype + ", bfloat16, bfloat16"
            format_input = format_input + ", ND, FRACTAL_NZ"
            format_output = format_output + ", ND, ND"
    input0 = gen_param(classify="input0", name="input_x", datatype=x_dtype, format=format_input,
                       unknownshape_format=format_input)
    output0 = gen_param(classify="output0", name="output1", datatype=x_dtype, format=format_output,
                        unknownshape_format=format_output)

    param_list = [input0, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


def get_new_format_axis(ori_shape, ori_axis, input_format, ori_format):
    """
    convert nd_axis to new_format_axis
    """
    axis_new = []

    # when input_format is FRACTAL_Z, ori_axis is [0, 1, 2, 3], so reduce all is applied, axis convert will not be done.
    if input_format == "FRACTAL_Z":
        axis_new = list(ori_axis)
    else:
        for i in ori_axis:
            new_axis = update_axis_for_other_format(ori_shape, i, input_format, ori_format, True)
            if isinstance(new_axis, int):
                axis_new.append(new_axis)
            else:
                for j in new_axis:
                    axis_new.append(j)

    return axis_new


def support_fusion_condition():
    """
    check ub fusion support
    """
    inputs = tbe_context.op_context.get_context().get_op_info()[0].inputs
    atts = tbe_context.op_context.get_context().get_op_info()[0].attrs
    axis_value = atts[0].get("value")
    shape = inputs[0].get("shape")
    axis_all = [i for i in range(len(shape))]
    if axis_all == list(axis_value):
        return False
    if inputs[0].get("format") in ["FRACTAL_Z", "FRACTAL_NZ"]:
        return False
    return True


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument,invalid-name
# 'pylint: disable=locally-disabled,redefined-builtin,too-many-locals,unused-variable
def reduce_sum_d_compute(x, y, axis=None, keepdims=None, kernel_name="reduce_sum_d"):
    """
    redusce_sum_d compute

    Parameters:
    ----------
    x: TVM tensor
        input tensor.
    y: dict
        the dict of output tensor.
    axis: int, list, tuple or NONETYPE
        the axis for reduce.
    keepdims: bool or NONETYPE
        if true, retains reduced dimensions with length 1.
    kernel_name: str
        cce kernel name, default value is "reduce_sum_d".

    Returns
    -------
    result: TVM tensor
        output tensor, has the same shape and type as input tensor.
    """
    x_dtype = x.dtype

    if x_dtype == "float16" and tbe_platform.api_check_support("tbe.dsl.reduce_sum", "float32"):
        x = tbe.cast_to(x, "float32")
    res_sum = tbe.reduce_sum(x, axis=axis, keepdims=keepdims)
    result = tbe.cast_to(res_sum, x_dtype)

    return result


def square_compute(input_x, output_y, kernel_name="square"):
    """
    algorithm: square
    calculating data's square,y= x*x

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    kernel_name: str
        cce kernel name, default value is square

    Returns
    -------
    result_square : tvm.tensor
        the result of square
    """
    result_square = tbe.vmul(input_x, input_x)

    return result_square


@register_operator_compute("SquareSumV1", op_mode="dynamic",
                            support_fusion=support_fusion_condition, support_bfp16=True)
def square_sum_v1_compute(input_x, output1, axis, keep_dims, kernel_name="square_sum_v1"):
    """
    calculating data

    Parameters
    ----------
    Input and output of fusion graph

    Returns
    -------
    output tensor
    """

    data_square = square_compute(input_x, {}, kernel_name)
    data_sum = reduce_sum_d_compute(data_square, {}, axis, keepdims=keep_dims, kernel_name=kernel_name)

    return data_sum


@register_operator("SquareSumV1")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_LIST_INT, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME, para_check.OPTION_ATTR_STR)
def square_sum_v1(input_x, output1, axis, keep_dims=True, kernel_name="square_sum_v1",
                  impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    calculating data

    Parameters
    ----------
    Input and output of fusion graph

    Returns
    -------
    None
    """
    x_shape = input_x.get("shape")
    shape_ori = input_x.get("ori_shape")
    x_dtype = input_x.get("dtype").lower()
    format_x = input_x.get("format")
    format_ori = input_x.get("ori_format")

    # In binary, the reduce template uses the unkonw mode
    if axis is None:
        axis_input = {"shape": [-1], "rel_pos_to_reduce": "axis"}
    else:
        if axis == []:
            axis = list(range(len(x_shape)))
        elif format_x in ["FRACTAL_NZ", "FRACTAL_Z"]:
            axis = get_new_format_axis(shape_ori, axis, format_x, format_ori)
        axis_input = {"shape": [len(axis)], "value": axis, "rel_pos_to_reduce": "axis"}
    input_x["rel_pos_to_reduce"] = "before"

    ins = classify([input_x, axis_input], OpPatternMode.REDUCE,
                   {"keepdims": keep_dims is True, "ignore_fractal_format": False})
    schedules, tensors = [], []

    for (_input_x, _axis) in ins:
        with tbe.compute():
            shape_new = shape_util.variable_shape([_input_x, _axis], op_mode="reduce")[0]
            data_input = tvm.placeholder(shape_new, name="data_input", dtype=x_dtype)

            data_input.op.attrs["format"] = format_x
            res = square_sum_v1_compute(data_input, output1, _axis.get("value"),
                                        keep_dims, kernel_name)
            tensors.append([data_input, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}

    tbe.build(schedules, config)
