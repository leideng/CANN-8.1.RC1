# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
layer_norm_v3
"""
from tbe import tvm
from tbe.common.utils.errormgr import error_manager_vector
from impl.util import util_select_op_base
from impl.util.util_common import is_unknown_rank_input
from impl.util.util_common import is_unknown
from impl.util.util_attr_common import LayerNormAttrInfo
from impl.util.util_attr_common import get_attr_by_cls
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import OpPatternMode
from impl.util.norm_pattern_adapter import NormPattern
from impl import constant_util as constant


# 'pylint: disable=unused-argument,too-many-lines,invalid-name
# 'pylint: disable=too-many-arguments,too-many-locals,unused-variable
# 'pylint: disable=huawei-too-many-arguments
def get_op_support_info(input_x,
                        input_gamma,
                        input_beta,
                        output_y,
                        output_mean,
                        output_rstd,
                        begin_norm_axis,
                        begin_params_axis,
                        epsilon=1e-5,
                        kernel_name="layer_norm_v3",
                        impl_mode="high_performance"):
    """
    get_op_support_info
    """
    is_unknown_rank = is_unknown_rank_input((input_x, input_gamma, input_beta))
    format_x = input_x.get("format").upper()
    shape_x = input_x.get("shape")
    ori_shape_x = input_x.get("ori_shape")
    axis_split_matrix = []

    if not is_unknown_rank and format_x in ("ND", "NCHW", "NHWC", "NC1HWC0"):
        begin_norm_axis = shape_util.axis_check(len(shape_x), begin_norm_axis)
        begin_params_axis = shape_util.axis_check(len(shape_x), begin_params_axis)
        if begin_params_axis == 0:
            for i in range(begin_norm_axis):
                split_0 = [
                    SplitInput([0, [i], [-1], [-1]], [1, [i], [-1], [-1]], [2, [i], [-1], [-1]]),
                    SplitOutput([0, [i]], [1, [i]], [2, [i]])
                ]
                axis_split_matrix.append(split_0)
        else:
            if begin_norm_axis <= begin_params_axis:
                for i in range(begin_norm_axis):
                    split_0 = [SplitInput([0, [i], [-1], [-1]]), SplitOutput([0, [i]], [1, [i]], [2, [i]])]
                    axis_split_matrix.append(split_0)
            else:
                for i in range(begin_params_axis):
                    split_0 = [SplitInput([0, [i], [-1], [-1]]), SplitOutput([0, [i]], [1, [i]], [2, [i]])]
                    axis_split_matrix.append(split_0)

    elif not is_unknown((input_x, input_gamma, input_beta)) and format_x == "FRACTAL_NZ":
        index_list = tuple(index for index, _ in enumerate(ori_shape_x))
        start_axis = min(begin_norm_axis, begin_params_axis)

        no_split_axis = index_list[start_axis:]
        no_split_axis = to_frac_z_axis(ori_shape_x, no_split_axis)
        for i in range(len(shape_x)):
            if i not in no_split_axis:
                split_0 = [SplitInput([0, [i], [-1], [-1]]),
                           SplitOutput([0, [i]], [1, [i]], [2, [i]])]
                axis_split_matrix.append(split_0)

    else:
        axis_split_matrix = None
    axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


def _division_sixteen(input_x, begin_norm_axis):
    """
    division_sixteen
    """
    shape = input_x.get("ori_shape")
    shape = shape_util.scalar2tensor_one(shape)

    if len(shape) < 2:
        if shape[-1] == 0:
            error_detail = "value of shape_x is illegal"
            error_manager_vector.raise_err_input_shape_invalid("layer_norm_v3", "input_x", error_detail)
        return False

    # Format of NZ is not supported when the size of reduce axis is larger than 4096
    reduce_axis_threshold = 4096
    if shape[-1] >= reduce_axis_threshold:
        return False

    if shape[-1] == 0 or shape[-2] == 0:
        error_detail = "value of shape_x is illegal"
        error_manager_vector.raise_err_input_shape_invalid("layer_norm_v3", "input_x", error_detail)

    if shape[-2] % constant.C0_SIZE == 0:
        if is_unknown(input_x):
            if shape[-1] % constant.C0_SIZE == 0:
                return True
        else:
            is_reduce_last = begin_norm_axis in (-1, len(shape) - 1)
            if shape[-1] % constant.C0_SIZE == 0:
                return True
    return False


# 'pylint: disable=huawei-too-many-arguments
def op_select_format(input_x,
                     input_gamma,
                     input_beta,
                     output_y,
                     output_mean,
                     output_rstd,
                     begin_norm_axis,
                     begin_params_axis,
                     kernel_name="layer_norm_v3"):
    """
    select format dynamically
    """
    shape_gamma = input_gamma.get("ori_shape")
    shape_gamma = shape_util.scalar2tensor_one(shape_gamma)

    # can not support Nz + ND
    if begin_params_axis == 0:
        if len(shape_gamma) >= 2 or (not _division_sixteen(input_x, begin_norm_axis)):
            input0 = util_select_op_base.gen_param(classify="input0",
                                                   name="x",
                                                   datatype="float16,float16,float16,float16,"
                                                            "float,float,float,float,"
                                                            "bfloat16,bfloat16,bfloat16,bfloat16,"
                                                            "float16,float16,float16,float16,"
                                                            "bfloat16,bfloat16,bfloat16,bfloat16",
                                                   format="NCHW,NC1HWC0,NHWC,ND,"
                                                          "NCHW,NC1HWC0,NHWC,ND,"
                                                          "NCHW,NC1HWC0,NHWC,ND,"
                                                          "NCHW,NC1HWC0,NHWC,ND,"
                                                          "NCHW,NC1HWC0,NHWC,ND")

            input1 = util_select_op_base.gen_param(classify="input1",
                                                   name="gamma",
                                                   datatype="float16,float16,float16,float16,"
                                                            "float,float,float,float,"
                                                            "bfloat16,bfloat16,bfloat16,bfloat16,"
                                                            "float,float,float,float,"
                                                            "float,float,float,float",
                                                   format="NCHW,NC1HWC0,NHWC,ND,"
                                                          "NCHW,NC1HWC0,NHWC,ND,"
                                                          "NCHW,NC1HWC0,NHWC,ND,"
                                                          "NCHW,NC1HWC0,NHWC,ND,"
                                                          "NCHW,NC1HWC0,NHWC,ND")

            input2 = util_select_op_base.gen_param(classify="input2",
                                                   name="beta",
                                                   datatype="float16,float16,float16,float16,"
                                                            "float,float,float,float,"
                                                            "bfloat16,bfloat16,bfloat16,bfloat16,"
                                                            "float,float,float,float,"
                                                            "float,float,float,float",
                                                   format="NCHW,NC1HWC0,NHWC,ND,"
                                                          "NCHW,NC1HWC0,NHWC,ND,"
                                                          "NCHW,NC1HWC0,NHWC,ND,"
                                                          "NCHW,NC1HWC0,NHWC,ND,"
                                                          "NCHW,NC1HWC0,NHWC,ND")

            output0 = util_select_op_base.gen_param(classify="output0",
                                                    name="y",
                                                    datatype="float16,float16,float16,float16,"
                                                             "float,float,float,float,"
                                                             "bfloat16,bfloat16,bfloat16,bfloat16,"
                                                             "float16,float16,float16,float16,"
                                                             "bfloat16,bfloat16,bfloat16,bfloat16",
                                                    format="NCHW,NC1HWC0,NHWC,ND,"
                                                           "NCHW,NC1HWC0,NHWC,ND,"
                                                           "NCHW,NC1HWC0,NHWC,ND,"
                                                           "NCHW,NC1HWC0,NHWC,ND,"
                                                           "NCHW,NC1HWC0,NHWC,ND")

            output1 = util_select_op_base.gen_param(classify="output1",
                                                    name="mean",
                                                    datatype="float16,float16,float16,float16,"
                                                             "float,float,float,float,"
                                                             "bfloat16,bfloat16,bfloat16,bfloat16,"
                                                             "float,float,float,float,"
                                                             "float,float,float,float",
                                                    format="NCHW,NC1HWC0,NHWC,ND,"
                                                           "NCHW,NC1HWC0,NHWC,ND,"
                                                           "NCHW,NC1HWC0,NHWC,ND,"
                                                           "NCHW,NC1HWC0,NHWC,ND,"
                                                           "NCHW,NC1HWC0,NHWC,ND")

            output2 = util_select_op_base.gen_param(classify="output2",
                                                    name="rstd",
                                                    datatype="float16,float16,float16,float16,"
                                                             "float,float,float,float,"
                                                             "bfloat16,bfloat16,bfloat16,bfloat16,"
                                                             "float,float,float,float,"
                                                             "float,float,float,float",
                                                    format="NCHW,NC1HWC0,NHWC,ND,"
                                                           "NCHW,NC1HWC0,NHWC,ND,"
                                                           "NCHW,NC1HWC0,NHWC,ND,"
                                                           "NCHW,NC1HWC0,NHWC,ND,"
                                                           "NCHW,NC1HWC0,NHWC,ND")
        else:
            input0 = util_select_op_base.gen_param(classify="input0",
                                                   name="x",
                                                   datatype="float16,float16,float16,float16,float16,"
                                                            "float,float,float,float,float,"
                                                            "bfloat16,bfloat16,bfloat16,bfloat16,bfloat16",
                                                   format="FRACTAL_NZ,NCHW,NC1HWC0,NHWC,ND,"
                                                          "FRACTAL_NZ,NCHW,NC1HWC0,NHWC,ND,"
                                                          "FRACTAL_NZ,NCHW,NC1HWC0,NHWC,ND")

            input1 = util_select_op_base.gen_param(classify="input1",
                                                   name="gamma",
                                                   datatype="float16,float16,float16,float16,float16,"
                                                            "float,float,float,float,float,"
                                                            "bfloat16,bfloat16,bfloat16,bfloat16,bfloat16",
                                                   format="ND,NCHW,NC1HWC0,NHWC,ND,"
                                                          "ND,NCHW,NC1HWC0,NHWC,ND,"
                                                          "ND,NCHW,NC1HWC0,NHWC,ND")

            input2 = util_select_op_base.gen_param(classify="input2",
                                                   name="beta",
                                                   datatype="float16,float16,float16,float16,float16,"
                                                            "float,float,float,float,float,"
                                                            "bfloat16,bfloat16,bfloat16,bfloat16,bfloat16",
                                                   format="ND,NCHW,NC1HWC0,NHWC,ND,"
                                                          "ND,NCHW,NC1HWC0,NHWC,ND,"
                                                          "ND,NCHW,NC1HWC0,NHWC,ND")

            output0 = util_select_op_base.gen_param(classify="output0",
                                                    name="y",
                                                    datatype="float16,float16,float16,float16,float16,"
                                                             "float,float,float,float,float,"
                                                             "bfloat16,bfloat16,bfloat16,bfloat16,bfloat16",
                                                    format="FRACTAL_NZ,NCHW,NC1HWC0,NHWC,ND,"
                                                           "FRACTAL_NZ,NCHW,NC1HWC0,NHWC,ND,"
                                                           "FRACTAL_NZ,NCHW,NC1HWC0,NHWC,ND")

            output1 = util_select_op_base.gen_param(classify="output1",
                                                    name="mean",
                                                    datatype="float16,float16,float16,float16,float16,"
                                                             "float,float,float,float,float,"
                                                             "bfloat16,bfloat16,bfloat16,bfloat16,bfloat16",
                                                    format="ND,NCHW,NC1HWC0,NHWC,ND,"
                                                           "ND,NCHW,NC1HWC0,NHWC,ND,"
                                                           "ND,NCHW,NC1HWC0,NHWC,ND")

            output2 = util_select_op_base.gen_param(classify="output2",
                                                    name="rstd",
                                                    datatype="float16,float16,float16,float16,float16,"
                                                             "float,float,float,float,float,"
                                                             "bfloat16,bfloat16,bfloat16,bfloat16,bfloat16",
                                                    format="ND,NCHW,NC1HWC0,NHWC,ND,"
                                                           "ND,NCHW,NC1HWC0,NHWC,ND,"
                                                           "ND,NCHW,NC1HWC0,NHWC,ND")
    else:
        if len(shape_gamma) >= 2 or (not _division_sixteen(input_x, begin_norm_axis)):
            input0 = util_select_op_base.gen_param(classify="input0",
                                                   name="x",
                                                   datatype="float16,float16,float16,"
                                                            "float,float,float,"
                                                            "bfloat16,bfloat16,bfloat16,"
                                                            "float16,float16,float16,"
                                                            "bfloat16,bfloat16,bfloat16",
                                                   format="NCHW,NHWC,ND,"
                                                          "NCHW,NHWC,ND,"
                                                          "NCHW,NHWC,ND,"
                                                          "NCHW,NHWC,ND,"
                                                          "NCHW,NHWC,ND")

            input1 = util_select_op_base.gen_param(classify="input1",
                                                   name="gamma",
                                                   datatype="float16,float16,float16,"
                                                            "float,float,float,"
                                                            "bfloat16,bfloat16,bfloat16,"
                                                            "float,float,float,"
                                                            "float,float,float",
                                                   format="NCHW,NHWC,ND,"
                                                          "NCHW,NHWC,ND,"
                                                          "NCHW,NHWC,ND,"
                                                          "NCHW,NHWC,ND,"
                                                          "NCHW,NHWC,ND")

            input2 = util_select_op_base.gen_param(classify="input2",
                                                   name="beta",
                                                   datatype="float16,float16,float16,"
                                                            "float,float,float,"
                                                            "bfloat16,bfloat16,bfloat16,"
                                                            "float,float,float,"
                                                            "float,float,float",
                                                   format="NCHW,NHWC,ND,"
                                                          "NCHW,NHWC,ND,"
                                                          "NCHW,NHWC,ND,"
                                                          "NCHW,NHWC,ND,"
                                                          "NCHW,NHWC,ND")

            output0 = util_select_op_base.gen_param(classify="output0",
                                                    name="y",
                                                    datatype="float16,float16,float16,"
                                                             "float,float,float,"
                                                             "bfloat16,bfloat16,bfloat16,"
                                                             "float16,float16,float16,"
                                                             "bfloat16,bfloat16,bfloat16",
                                                    format="NCHW,NHWC,ND,"
                                                           "NCHW,NHWC,ND,"
                                                           "NCHW,NHWC,ND,"
                                                           "NCHW,NHWC,ND,"
                                                           "NCHW,NHWC,ND")

            output1 = util_select_op_base.gen_param(classify="output1",
                                                    name="mean",
                                                    datatype="float16,float16,float16,"
                                                             "float,float,float,"
                                                             "bfloat16,bfloat16,bfloat16,"
                                                             "float,float,float,"
                                                             "float,float,float",
                                                    format="NCHW,NHWC,ND,"
                                                           "NCHW,NHWC,ND,"
                                                           "NCHW,NHWC,ND,"
                                                           "NCHW,NHWC,ND,"
                                                           "NCHW,NHWC,ND")

            output2 = util_select_op_base.gen_param(classify="output2",
                                                    name="rstd",
                                                    datatype="float16,float16,float16,"
                                                             "float,float,float,"
                                                             "bfloat16,bfloat16,bfloat16,"
                                                             "float,float,float,"
                                                             "float,float,float",
                                                    format="NCHW,NHWC,ND,"
                                                           "NCHW,NHWC,ND,"
                                                           "NCHW,NHWC,ND,"
                                                           "NCHW,NHWC,ND,"
                                                           "NCHW,NHWC,ND")
        else:
            input0 = util_select_op_base.gen_param(classify="input0",
                                                   name="x",
                                                   datatype="float16,float16,float16,float16,"
                                                            "float,float,float,float,"
                                                            "bfloat16,bfloat16,bfloat16,bfloat16",
                                                   format="FRACTAL_NZ,NCHW,NHWC,ND,"
                                                          "FRACTAL_NZ,NCHW,NHWC,ND,"
                                                          "FRACTAL_NZ,NCHW,NHWC,ND")

            input1 = util_select_op_base.gen_param(classify="input1",
                                                   name="gamma",
                                                   datatype="float16,float16,float16,float16,"
                                                            "float,float,float,float,"
                                                            "bfloat16,bfloat16,bfloat16,bfloat16",
                                                   format="ND,NCHW,NHWC,ND,"
                                                          "ND,NCHW,NHWC,ND,"
                                                          "ND,NCHW,NHWC,ND")

            input2 = util_select_op_base.gen_param(classify="input2",
                                                   name="beta",
                                                   datatype="float16,float16,float16,float16,"
                                                            "float,float,float,float,"
                                                            "bfloat16,bfloat16,bfloat16,bfloat16",
                                                   format="ND,NCHW,NHWC,ND,"
                                                          "ND,NCHW,NHWC,ND,"
                                                          "ND,NCHW,NHWC,ND")

            output0 = util_select_op_base.gen_param(classify="output0",
                                                    name="y",
                                                    datatype="float16,float16,float16,float16,"
                                                             "float,float,float,float,"
                                                             "bfloat16,bfloat16,bfloat16,bfloat16",
                                                    format="FRACTAL_NZ,NCHW,NHWC,ND,"
                                                           "FRACTAL_NZ,NCHW,NHWC,ND,"
                                                           "FRACTAL_NZ,NCHW,NHWC,ND")

            output1 = util_select_op_base.gen_param(classify="output1",
                                                    name="mean",
                                                    datatype="float16,float16,float16,float16,"
                                                             "float,float,float,float,"
                                                             "bfloat16,bfloat16,bfloat16,bfloat16",
                                                    format="ND,NCHW,NHWC,ND,"
                                                           "ND,NCHW,NHWC,ND,"
                                                           "ND,NCHW,NHWC,ND")

            output2 = util_select_op_base.gen_param(classify="output2",
                                                    name="rstd",
                                                    datatype="float16,float16,float16,float16,"
                                                             "float,float,float,float,"
                                                             "bfloat16,bfloat16,bfloat16,bfloat16",
                                                    format="ND,NCHW,NHWC,ND,"
                                                           "ND,NCHW,NHWC,ND,"
                                                           "ND,NCHW,NHWC,ND")

    param_list = [input0, input1, input2, output0, output1, output2]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def to_frac_z_axis(ori_shape, ori_axis):
    """
    judge the format is fractal NZ

    Parameters
    ----------
    ori_shape: list or tuple
        original shape of input
    ori_axis: list or tuple
        original axis of original shape to operate

    Returns
    -------
    output: list
        axis of the fractal Nz shape
    """

    frac_z_axis = list(ori_axis)
    shape_len = len(ori_shape)
    axis_count = len(frac_z_axis)
    axis_negative_1 = shape_len - 1
    axis_negative_2 = shape_len - 2
    for i in range(axis_count):
        axis_index = (frac_z_axis[i] + shape_len) % shape_len
        if axis_index == axis_negative_1:
            if frac_z_axis[i] > shape_len - 2:
                frac_z_axis[i] = axis_index - 1
                frac_z_axis.append(axis_index + 2)
        elif axis_index == axis_negative_2:
            frac_z_axis[i] = axis_index + 1
            frac_z_axis.append(axis_index + 2)
        else:
            frac_z_axis[i] = axis_index
    return frac_z_axis


def _broadcast_nz(tensor, shape):
    """
    broadcast_nz
    """
    broadcast_axes = []
    src_shape = shape_util.shape_to_list(tensor.shape)
    for i, _ in enumerate(shape):
        if shape[i] != src_shape[i]:
            broadcast_axes.append(i)
    if len(broadcast_axes) == 2 and \
            broadcast_axes[1] - broadcast_axes[0] != 1 and \
            broadcast_axes[1] + 1 == len(shape):
        temp_shape = src_shape[:-1] + [shape[-1]]
        tensor = tbe.broadcast(tensor, temp_shape)
    tensor = tbe.broadcast(tensor, shape)
    return tensor


# 'pylint: disable=huawei-too-many-arguments
def layer_norm_v3_compute_nz(input_x, input_gamma, input_beta, output_y, output_mean, output_rstd,
                             reduce_axis, begin_params_axis, epsilon, kernel_name="layer_norm_v3",
                             impl_mode="high_performance"):
    """
    DSL description of the layernorm operator's mathematical calculation process

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of x input data
    input_gamma: TVM tensor
        the placeholder of gamma input data
    input_beta: TVM tensor
        the placeholder of beta input data
    output_y: dict
        shape and dtype of output_y
    output_mean: dict
        shape and dtype of output_mean
    output_rstd: dict
        shape and dtype of output_rstd
    reduce_axis: list
      the reduce  axis of  shape
    begin_params_axis: int
      The first parameter (beta, gamma) dimension: scale
      and centering parameters will have dimensions
      `begin_params_axis : rank(inputs)` and will be broadcast with the
      normalized inputs accordingly.
    epsilon: float,
      Minimum positive number greater than 0
    kernel_name: str
        cce kernel name, default value is "layer_norm_v3"

    Returns
    -------
    res_tuple: tuple
        (result, mean, rstd)
    """
    shape_x = shape_util.shape_to_list(input_x.shape)
    dtype = input_x.dtype.lower()
    ori_dtype = dtype
    cast_dtype = dtype
    cast_dtype_precision = dtype
    is_cast = False
    is_support_vexp = tbe_platform.api_check_support("te.lang.cce.vexp", "float32")
    tbe_context.get_context().add_compile_info("is_support_vexp", is_support_vexp)
    is_fp16_promote = dtype == "float16" and impl_mode != "keep_fp16" and \
        ((is_support_vexp and impl_mode == "high_performance") or impl_mode == "high_precision")
    is_option_out = output_mean is None and output_rstd is None
    if dtype == "bfloat16" or is_fp16_promote:
        cast_dtype = "float32"
        cast_dtype_precision = "float32"
        input_x = tbe.cast_to(input_x, "float32")
        input_gamma = tbe.cast_to(input_gamma, "float32")
        input_beta = tbe.cast_to(input_beta, "float32")
        is_cast = True

    # Calculate the scaling ratio of the average
    reduce_elts = 1.0
    for i in reduce_axis:
        reduce_elts *= shape_x[i]
    if isinstance(reduce_elts, float):
        mean_cofs = reduce_elts**(-1)
        mean_cof = tvm.const(mean_cofs, dtype=cast_dtype)
    else:
        mean_cof = tbe.var("mean_cof", dtype=cast_dtype)
        operation.add_compile_info("reduce_mean_cof_dtype", cast_dtype)

    # DSL description of the mean calculation process
    if impl_mode != "keep_fp16":
        mean_muls = tbe.vmuls(input_x, mean_cof)
        mean = tbe.reduce_sum(mean_muls, axis=reduce_axis, keepdims=True)

        if not is_option_out and is_cast and ori_dtype == "bfloat16":
            mean_16 = tbe.round(mean, ori_dtype)
            mean = tbe.cast_to(mean_16, "float32")

        if not is_option_out and is_cast and ori_dtype == "float16":
            mean_16 = tbe.cast_to(mean, ori_dtype)
            mean = tbe.cast_to(mean_16, "float32")
    else:
        x_sum = tbe.reduce_sum(input_x, axis=reduce_axis, keepdims=True)
        mean = tbe.vmuls(x_sum, mean_cof)
        if not is_option_out and is_cast and ori_dtype == "bfloat16":
            mean_16 = tbe.round(mean, ori_dtype)
            mean = tbe.cast_to(mean_16, "float32")

    # DSL description of the variance calculation process
    mean_variance_broadcast = _broadcast_nz(mean, shape_x)
    variance_sub = tbe.vsub(input_x, mean_variance_broadcast)
    variance_mul = tbe.vmul(variance_sub, variance_sub)
    if impl_mode != "keep_fp16":
        variance_muls = tbe.vmuls(variance_mul, mean_cof)
        variance = tbe.reduce_sum(variance_muls, axis=reduce_axis, keepdims=True)
    else:
        variance_sum = tbe.reduce_sum(variance_mul, axis=reduce_axis, keepdims=True)
        variance = tbe.vmuls(variance_sum, mean_cof)

    # DSL description of the normalize calculation process
    if impl_mode == "keep_fp16":
        epsilon_scalar = get_attr_by_cls(epsilon, LayerNormAttrInfo.ATTR_EPSILON, "float16")
        normalize_add = tbe.vadds(variance, epsilon_scalar)
        normalize_log = tbe.vlog(normalize_add)
        normalize_log_mul = tbe.vmuls(normalize_log, tvm.const(-0.5, dtype="float16"))
        rstd = tbe.vexp(normalize_log_mul)
        if not is_option_out and is_cast and ori_dtype == "bfloat16":
            rstd_16 = tbe.round(rstd, ori_dtype)
            rstd = tbe.cast_to(rstd_16, "float32")
    else:
        if impl_mode == "high_performance" and is_support_vexp:
            epsilon_scalar = get_attr_by_cls(epsilon, LayerNormAttrInfo.ATTR_EPSILON, cast_dtype)
            normalize_add = tbe.vadds(variance, epsilon_scalar)
            normalize_log = tbe.vlog(normalize_add)
            normalize_log_mul = tbe.vmuls(normalize_log, tvm.const(-0.5, dtype=cast_dtype))
            rstd = tbe.vexp(normalize_log_mul)
        else:
            epsilon_scalar = get_attr_by_cls(epsilon, LayerNormAttrInfo.ATTR_EPSILON, cast_dtype_precision)
            normalize_add = tbe.vadds(variance, epsilon_scalar)
            normalize_sqrt = tbe.vsqrt(normalize_add)
            tensor_one = tbe.broadcast(tvm.const(1, cast_dtype_precision), normalize_sqrt.shape)
            rstd = tbe.vdiv(tensor_one, normalize_sqrt)

        if not is_option_out and is_cast and ori_dtype == "bfloat16":
            rstd_16 = tbe.round(rstd, ori_dtype)
            rstd = tbe.cast_to(rstd_16, "float32")

        if not is_option_out and is_cast and ori_dtype == "float16":
            rstd_16 = tbe.cast_to(rstd, ori_dtype)
            rstd = tbe.cast_to(rstd_16, "float32")

    normalize_rstd_broadcast = _broadcast_nz(rstd, shape_x)
    normalize_mul = tbe.vmul(variance_sub, normalize_rstd_broadcast)

    # DSL description of the scale and translate calculation process
    if begin_params_axis == 0:
        scale_mul = tbe.vmul(normalize_mul, input_gamma)
        res = tbe.vadd(scale_mul, input_beta)
    else:
        gamma_broadcast = _broadcast_nz(input_gamma, shape_x)
        beta_broadcast = _broadcast_nz(input_beta, shape_x)
        scale_mul = tbe.vmul(normalize_mul, gamma_broadcast)
        res = tbe.vadd(scale_mul, beta_broadcast)

    if not is_option_out and is_cast and ori_dtype == "bfloat16":
        res = tbe.round(res, ori_dtype)
        return res, mean_16, rstd_16

    if not is_option_out and is_cast and ori_dtype == "float16":
        res = tbe.cast_to(res, ori_dtype)
        return res, mean_16, rstd_16

    return res, mean, rstd


# 'pylint: disable=huawei-too-many-arguments
@register_operator_compute("LayerNormV3", op_mode="dynamic", support_fusion=True)
def layer_norm_v3_auto_fusion_compute(input_x,
                                      input_gamma,
                                      input_beta,
                                      output_y,
                                      output_mean,
                                      output_rstd,
                                      begin_norm_axis,
                                      begin_params_axis,
                                      epsilon=1e-5,
                                      kernel_name="layer_norm_v3",
                                      impl_mode="high_performance"):
    """
    layernormv3 compute for auto fusions
    """
    shape_x = shape_util.shape_to_list(input_x.shape)
    indices = tuple(range(len(shape_x)))
    reduce_axis = indices[begin_norm_axis:]
    res, mean, variance = layer_norm_v3_compute(input_x, input_gamma, input_beta, output_y, output_mean, output_rstd,
                                                reduce_axis, begin_params_axis, epsilon, kernel_name, impl_mode)

    return res, mean, variance


# 'pylint: disable=huawei-too-many-arguments
def layer_norm_v3_compute(input_x, input_gamma, input_beta, output_y, output_mean, output_rstd,
                          reduce_axis, begin_params_axis, epsilon, kernel_name="layer_norm_v3",
                          impl_mode="high_performance"):
    """
    DSL description of the layernorm operator's mathematical calculation process

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of x input data
    input_gamma: TVM tensor
        the placeholder of gamma input data
    input_beta: TVM tensor
        the placeholder of beta input data
    output_y: dict
        shape and dtype of output_y
    output_mean: dict
        shape and dtype of output_mean
    output_rstd: dict
        shape and dtype of output_rstd
    reduce_axis: list
      the reduce axis
    begin_params_axis: int
      The first parameter (beta, gamma) dimension: scale
      and centering parameters will have dimensions
      `begin_params_axis : rank(inputs)` and will be broadcast with the
      normalized inputs accordingly.
    epsilon: float,
      Minimum positive number greater than 0
    kernel_name: str
        cce kernel name, default value is "layer_norm_v3"

    Returns
    -------
    res_tuple: tuple
        (result, mean, rstd)
    """
    shape_x = shape_util.shape_to_list(input_x.shape)
    dtype = input_x.dtype.lower()
    gamma_dtype = input_gamma.dtype.lower()
    beta_dtype = input_beta.dtype.lower()
    ori_dtype = dtype
    cast_dtype = dtype
    cast_dtype_precision = dtype
    is_cast = False
    is_support_vexp = tbe_platform.api_check_support("te.lang.cce.vexp", "float32")
    is_mix_dtype = dtype in ("float16", "bfloat16") and gamma_dtype == "float32" and beta_dtype == "float32"
    is_fp16_promote = dtype == "float16" and impl_mode != "keep_fp16" and \
        ((is_support_vexp and impl_mode == "high_performance") or impl_mode == "high_precision")
    is_option_out = output_mean is None and output_rstd is None
    if dtype == "bfloat16" or is_mix_dtype or is_fp16_promote:
        input_x = tbe.cast_to(input_x, "float32")
        input_gamma = tbe.cast_to(input_gamma, "float32")
        input_beta = tbe.cast_to(input_beta, "float32")
        cast_dtype = "float32"
        cast_dtype_precision = "float32"
        # mix dtype no use is_cast for keep all float32 calculation of mean
        if not is_mix_dtype:
            is_cast = True

    reduce_elts = 1.0
    for i in reduce_axis:
        reduce_elts *= shape_x[i]
    if isinstance(reduce_elts, float):
        mean_cofs = reduce_elts**(-1)
        mean_cof = tvm.const(mean_cofs, dtype=cast_dtype)
        need_fake_data = tbe_context.get_context() and tbe_context.get_context().get_op_mode() == "dynamic" and \
            operation.get_compile_info() and operation.get_compile_info().get("reduce_mean_cof_dtype")
        if need_fake_data:
            _ = tbe.var("fake_mean_cof", dtype=cast_dtype)
    else:
        mean_cof = tbe.var("mean_cof", dtype=cast_dtype)
        operation.add_compile_info("reduce_mean_cof_dtype", cast_dtype)

    # DSL description of the mean calculation process
    if impl_mode != "keep_fp16":
        mean_muls = tbe.vmuls(input_x, mean_cof)
        mean = tbe.reduce_sum(mean_muls, axis=reduce_axis, keepdims=True)
        if not is_option_out and is_cast and ori_dtype == "bfloat16":
            mean_16 = tbe.round(mean, ori_dtype)
            mean = tbe.cast_to(mean_16, "float32")

        if not is_option_out and is_cast and ori_dtype == "float16":
            mean_16 = tbe.cast_to(mean, ori_dtype)
            mean = tbe.cast_to(mean_16, "float32")
    else:
        x_sum = tbe.reduce_sum(input_x, axis=reduce_axis, keepdims=True)
        mean = tbe.vmuls(x_sum, mean_cof)
        if not is_option_out and is_cast and ori_dtype == "bfloat16":
            mean_16 = tbe.round(mean, ori_dtype)
            mean = tbe.cast_to(mean_16, "float32")

    # DSL description of the variance calculation process
    mean_variance_broadcast = tbe.broadcast(mean, input_x.shape)
    variance_sub = tbe.vsub(input_x, mean_variance_broadcast)
    variance_mul = tbe.vmul(variance_sub, variance_sub)
    if impl_mode != "keep_fp16":
        variance_muls = tbe.vmuls(variance_mul, mean_cof)
        variance = tbe.reduce_sum(variance_muls, axis=reduce_axis, keepdims=True)
    else:
        variance_sum = tbe.reduce_sum(variance_mul, axis=reduce_axis, keepdims=True)
        variance = tbe.vmuls(variance_sum, mean_cof)

    # DSL description of the normalize calculation process
    if impl_mode == "keep_fp16":
        epsilon_scalar = get_attr_by_cls(epsilon, LayerNormAttrInfo.ATTR_EPSILON, "float16")
        normalize_add = tbe.vadds(variance, epsilon_scalar)
        normalize_log = tbe.vlog(normalize_add)
        normalize_log_mul = tbe.vmuls(normalize_log, tvm.const(-0.5, dtype="float16"))
        rstd = tbe.vexp(normalize_log_mul)
        if not is_option_out and is_cast and ori_dtype == "bfloat16":
            rstd_16 = tbe.round(rstd, ori_dtype)
            rstd = tbe.cast_to(rstd_16, "float32")
    else:
        cur_cce_product = tbe_platform.get_soc_spec(tbe_platform.SHORT_SOC_VERSION)
        if (impl_mode == "high_performance" and is_support_vexp) or cur_cce_product in (tbe_platform.ASCEND_910,):
            epsilon_scalar = get_attr_by_cls(epsilon, LayerNormAttrInfo.ATTR_EPSILON, cast_dtype)
            normalize_add = tbe.vadds(variance, epsilon_scalar)
            normalize_log = tbe.vlog(normalize_add)
            normalize_log_mul = tbe.vmuls(normalize_log, tvm.const(-0.5, dtype=cast_dtype))
            rstd = tbe.vexp(normalize_log_mul)
        else:
            epsilon_scalar = get_attr_by_cls(epsilon, LayerNormAttrInfo.ATTR_EPSILON, cast_dtype_precision)
            normalize_add = tbe.vadds(variance, epsilon_scalar)
            normalize_sqrt = tbe.vsqrt(normalize_add)
            tensor_one = tbe.broadcast(tvm.const(1, cast_dtype_precision), normalize_sqrt.shape)
            rstd = tbe.vdiv(tensor_one, normalize_sqrt)

        if not is_option_out and is_cast and ori_dtype == "bfloat16":
            rstd_16 = tbe.round(rstd, ori_dtype)
            rstd = tbe.cast_to(rstd_16, "float32")

        if not is_option_out and is_cast and ori_dtype == "float16":
            rstd_16 = tbe.cast_to(rstd, ori_dtype)
            rstd = tbe.cast_to(rstd_16, "float32")

    normalize_rstd_broadcast = tbe.broadcast(rstd, input_x.shape)
    normalize_mul = tbe.vmul(variance_sub, normalize_rstd_broadcast)

    # DSL description of the scale and translate calculation process
    if begin_params_axis == 0:
        scale_mul = tbe.vmul(normalize_mul, input_gamma)
        res = tbe.vadd(scale_mul, input_beta)
    else:
        gamma_broadcast = tbe.broadcast(input_gamma, input_x.shape)
        beta_broadcast = tbe.broadcast(input_beta, input_x.shape)
        scale_mul = tbe.vmul(normalize_mul, gamma_broadcast)
        res = tbe.vadd(scale_mul, beta_broadcast)

    if is_cast or is_mix_dtype:
        if ori_dtype == "bfloat16":
            res = tbe.round(res, ori_dtype)
        else:
            res = tbe.cast_to(res, "float16")
        if is_mix_dtype or is_option_out:
            return res, mean, rstd
        return res, mean_16, rstd_16

    return res, mean, rstd


# 'pylint: disable=huawei-too-many-arguments
@register_operator("LayerNormV3")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_OUTPUT, para_check.OPTION_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_FLOAT,
                            para_check.KERNEL_NAME, para_check.OPTION_ATTR_STR)
def layer_norm_v3(input_x,
                  input_gamma,
                  input_beta,
                  output_y,
                  output_mean,
                  output_rstd,
                  begin_norm_axis,
                  begin_params_axis,
                  epsilon=1e-5,
                  kernel_name="layer_norm_v3",
                  impl_mode="high_performance"):
    """
    layernorm operator interface implementation
    calculating: x, gamma, beta
        mean  = np.mean(x, reduce_axis, keepdims=True)
        variance = np.mean(np.power((x - mean),2), reduce_axis, keepdims=True)
        result = gamma*((x - mean) / np.sqrt(variance + 0.001)) + beta

    Parameters
    ----------
    input_x : dict
        shape and dtype of input x, only support float16, float32, bfloat16
    input_gamma: dict
        shape and dtype of input gamma, only support float16, float32, bfloat16
    input_beta: dict
        shape and dtype of input beta, only support float16, float32, bfloat16
    output_y: dict
        shape and dtype of output, only support float16, float32, bfloat16
    output_mean: dict
        shape and dtype of output_mean, only support float16, float32, bfloat16
    output_rstd: dict
        shape and dtype of output_rstd, only support float16, float32, bfloat16
    begin_norm_axis: int
      The first normalization dimension: normalization will be
      performed along dimensions `begin_norm_axis : rank(inputs)`
    begin_params_axis: int
      The first parameter (beta, gamma) dimension: scale
      and centering parameters will have dimensions
      `begin_params_axis : rank(inputs)` and will be broadcast with the
      normalized inputs accordingly.
    epsilon: float,
      Minimum positive number greater than 0
    kernel_name: str
        cce kernel name, default value is "layer_norm_v3"

    Returns
    -------
    None
    """
    input_format = input_x.get("format").upper()
    check_list = ("float16", "float32", "bfloat16")
    dtype = input_x.get("dtype").lower()
    dtype_gamma = input_gamma.get("dtype").lower()
    dtype_beta = input_gamma.get("dtype").lower()
    para_check.check_dtype(dtype, check_list, param_name="input_x")
    para_check.check_dtype(dtype_gamma, check_list, param_name="input_gamma")
    para_check.check_dtype(dtype_beta, check_list, param_name="input_beta")

    is_invalid_1 = output_mean is None and output_rstd is not None
    is_invalid_2 = output_mean is not None and output_rstd is None

    if is_invalid_1 or is_invalid_2:
        rule_desc = "output_mean and output_rstd just support all None or all required"
        param_value = "%s, %s" % (output_mean, output_rstd)
        error_manager_vector.raise_err_check_params_rules(kernel_name, rule_desc,
                                                          "output_mean, output_rstd", param_value)

    extra_params = dict()
    if is_unknown_rank_input(
        (input_x, input_gamma, input_beta)) or begin_norm_axis is None or begin_params_axis is None:
        # input is -2 case or axis is None, will use reduce unkonw mode
        reduce_axis = NormPattern.REDUCE_UNKNOWN_MODE
        broadcast_axis = NormPattern.BROADCAST_UNKNOWN_MODE
        extra_params.update(NormPattern.REDUCE_AFTER_TYPE)
        operation.add_compile_info("unknown_mode", True)
        extra_params.update({"broadcast_axes_type": {1: "opposite_reduce", 2: "opposite_reduce"}})
    else:
        shape_x = list(input_x.get("shape"))
        ori_shape_x = list(input_x.get("ori_shape"))
        input_gamma_format = input_gamma.get("format").upper()
        input_beta_format = input_beta.get("format").upper()

        range_x = list(input_x.get("range"))
        shape_gamma = list(input_gamma.get("shape"))
        shape_beta = list(input_beta.get("shape"))
        range_gamma = list(input_gamma.get("range"))
        range_beta = list(input_beta.get("range"))

        if input_format == "FRACTAL_NZ":
            begin_norm_axis = shape_util.axis_check(len(ori_shape_x), begin_norm_axis)
            begin_params_axis = shape_util.axis_check(len(ori_shape_x), begin_params_axis)

            if input_gamma_format == "FRACTAL_NZ" or input_beta_format == "FRACTAL_NZ":
                error_detail = "gamma and beta not support Nz in bert"
                error_manager_vector.raise_err_two_input_format_invalid(kernel_name, "input_gamma", "input_beta",
                                                                        error_detail)
            if shape_gamma != shape_beta:
                error_detail = "gamma and beta's shape must be same."
                error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "input_gamma", "input_beta",
                                                                        error_detail)
            if ori_shape_x[begin_params_axis:] != shape_gamma:
                error_detail = "x or gamma or begin_params_axis is wrong."
                error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "x", "input_gamma",
                                                                        error_detail)
            if len(shape_gamma) > 1:
                error_detail = "shape of gamma or beta only support 1D in bert"
                error_manager_vector.raise_err_input_shape_invalid(kernel_name, "input_gamma", error_detail)

            if begin_params_axis != 0:
                for i in range(begin_params_axis):
                    shape_gamma.insert(i, 1)
                    range_gamma.insert(i, (1, 1))
            shape_gamma[-2] = shape_x[-4]
            shape_gamma[-1] = 1
            shape_gamma.append(1)
            shape_gamma.append(shape_x[-1])
            shape_beta = shape_gamma
            range_gamma[-2] = range_x[-4]
            range_gamma[-1] = 1
            range_gamma.append(1)
            range_gamma.append(range_x[-1])
            range_beta = range_gamma
            index_list = tuple(range(len(ori_shape_x)))
            ori_reduce_axis = index_list[begin_norm_axis:]
            reduce_axis = to_frac_z_axis(ori_shape_x, ori_reduce_axis)
            ori_broadcast_axis = index_list[:begin_params_axis]
            broadcast_axis = to_frac_z_axis(ori_shape_x, ori_broadcast_axis)
        else:
            begin_norm_axis = shape_util.axis_check(len(shape_x), begin_norm_axis)
            begin_params_axis = shape_util.axis_check(len(shape_x), begin_params_axis)

            if shape_gamma != shape_beta:
                error_detail = "gamma and beta's shape must be same."
                error_manager_vector.raise_err_two_input_shape_invalid(kernel_name, "input_gamma", "input_beta",
                                                                        error_detail)
            index_list = tuple(range(len(shape_x)))
            reduce_axis = index_list[begin_norm_axis:]
            broadcast_axis = index_list[:begin_params_axis]

        input_gamma["shape"] = tuple(shape_gamma)
        input_beta["shape"] = tuple(shape_beta)
        input_gamma["range"] = tuple(range_gamma)
        input_beta["range"] = tuple(range_beta)

    extra_params.update({"input_shape_type": [0, 1, 1]})
    extra_params.update({"same_input_shape_group": [[1, 2]]})
    extra_params.update({"compile_broadcast_axes": {1: broadcast_axis, 2: broadcast_axis}})
    ins = classify([input_x, input_gamma, input_beta, reduce_axis], OpPatternMode.NORM, extra_params)

    schedules, tensors = [], []
    for (dy_shape_x, dy_shape_gamma, dy_shape_beta, dy_reduce_axis) in ins:
        with tbe.compute():
            x_var, gamma_var, beta_var = shape_util.variable_shape([dy_shape_x, dy_shape_gamma, dy_shape_beta],
                                                                    op_mode="norm")
            data_x = tvm.placeholder(x_var, name="x", dtype=dtype)
            data_gamma = tvm.placeholder(gamma_var, name="gamma", dtype=dtype_gamma)
            data_beta = tvm.placeholder(beta_var, name="beta", dtype=dtype_beta)

            if input_format == "FRACTAL_NZ":
                res, mean, rstd = layer_norm_v3_compute_nz(data_x, data_gamma, data_beta, output_y, output_mean,
                                                           output_rstd, dy_reduce_axis, begin_params_axis,
                                                           epsilon, kernel_name, impl_mode)
            else:
                res, mean, rstd = layer_norm_v3_compute(data_x, data_gamma, data_beta, output_y, output_mean,
                                                        output_rstd, dy_reduce_axis, begin_params_axis,
                                                        epsilon, kernel_name, impl_mode)
            if output_mean is None and output_rstd is None:
                tensors.append([data_x, data_gamma, data_beta, res])
            else:
                tensors.append([data_x, data_gamma, data_beta, res, mean, rstd])
        with tvm.target.cce():
            if output_mean is None and output_rstd is None:
                sch = tbe.auto_schedule(res)
            else:
                sch = tbe.auto_schedule([res, mean, rstd])

        schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}

    tbe.build(schedules, config)
