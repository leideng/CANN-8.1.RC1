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
softmax_v2
"""
# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=invalid-name,unused-variable,too-many-locals
# 'pylint: disable=too-many-statements,unnecessary-lambda
# 'pylint: disable=unidiomatic-typecheck,ungrouped-imports
# 'pylint: disable=too-many-lines,too-many-branches
from __future__ import absolute_import

import math

from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_context
from impl.util import util_frac_z as fz
from impl.util import util_select_op_base
import impl.dynamic as dyn_impl
import te.lang.cce
import te.platform.cce_params as cce
from tbe import tvm
from tbe.common import platform as tbe_platform
from tbe.common.buildcfg import build_config
from te.platform.fusion_manager import fusion_manager
from te.utils import para_check
from te.utils.error_manager import error_manager_vector


# General limitation of the size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648
# 1/4 UB size

UB_SIZE_LIMIT = \
    tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
UB_SIZE_LIMIT = UB_SIZE_LIMIT / 4


# 'pylint: disable = unused-argument
def get_op_support_info(input_x, output_y, axis=-1, half_to_float=False, kernel_name="softmax_v2"):
    format_x = input_x.get("format")
    origin_format_x = input_x.get("ori_format")
    dims_x = len(input_x.get("shape"))

    if not hasattr(axis, 'index'):
        new_axis = axis
    else:
        new_axis = axis[0]
    if format_x == "NC1HWC0":
        if origin_format_x == "NCHW":
            if new_axis in (0, -4):
                axis_split_matrix = [
                    [util_select_op_base.SplitInput([0, [2], [-1], [-1]]), util_select_op_base.SplitOutput([0, [2]])],
                    [util_select_op_base.SplitInput([0, [3], [-1], [-1]]), util_select_op_base.SplitOutput([0, [3]])]]
            if new_axis in (1, -3):
                axis_split_matrix = [
                    [util_select_op_base.SplitInput([0, [0], [-1], [-1]]), util_select_op_base.SplitOutput([0, [0]])],
                    [util_select_op_base.SplitInput([0, [2], [-1], [-1]]), util_select_op_base.SplitOutput([0, [2]])],
                    [util_select_op_base.SplitInput([0, [3], [-1], [-1]]), util_select_op_base.SplitOutput([0, [3]])]]
            if new_axis in (2, -2):
                axis_split_matrix = [
                    [util_select_op_base.SplitInput([0, [0], [-1], [-1]]), util_select_op_base.SplitOutput([0, [0]])],
                    [util_select_op_base.SplitInput([0, [3], [-1], [-1]]), util_select_op_base.SplitOutput([0, [3]])]]
            if new_axis in (3, -1):
                axis_split_matrix = [
                    [util_select_op_base.SplitInput([0, [0], [-1], [-1]]), util_select_op_base.SplitOutput([0, [0]])],
                    [util_select_op_base.SplitInput([0, [2], [-1], [-1]]), util_select_op_base.SplitOutput([0, [2]])]]
        elif origin_format_x == "NHWC":
            if new_axis in (0, -4):
                axis_split_matrix = [
                    [util_select_op_base.SplitInput([0, [2], [-1], [-1]]), util_select_op_base.SplitOutput([0, [2]])],
                    [util_select_op_base.SplitInput([0, [3], [-1], [-1]]), util_select_op_base.SplitOutput([0, [3]])]]
            if new_axis in (1, -3):
                axis_split_matrix = [
                    [util_select_op_base.SplitInput([0, [0], [-1], [-1]]), util_select_op_base.SplitOutput([0, [0]])],
                    [util_select_op_base.SplitInput([0, [3], [-1], [-1]]), util_select_op_base.SplitOutput([0, [3]])]]
            if new_axis in (2, -2):
                axis_split_matrix = [
                    [util_select_op_base.SplitInput([0, [0], [-1], [-1]]), util_select_op_base.SplitOutput([0, [0]])],
                    [util_select_op_base.SplitInput([0, [2], [-1], [-1]]), util_select_op_base.SplitOutput([0, [2]])]]
            if new_axis in (3, -1):
                axis_split_matrix = [
                    [util_select_op_base.SplitInput([0, [0], [-1], [-1]]), util_select_op_base.SplitOutput([0, [0]])],
                    [util_select_op_base.SplitInput([0, [2], [-1], [-1]]), util_select_op_base.SplitOutput([0, [2]])],
                    [util_select_op_base.SplitInput([0, [3], [-1], [-1]]), util_select_op_base.SplitOutput([0, [3]])]]
        else:
            axis_split_matrix = None
        axis_reduce_list = None
    elif format_x == "FRACTAL_NZ":
        if new_axis in (dims_x - 3, -1):
            axis_split_matrix = []
            for i in range(dims_x):
                if i not in (dims_x - 1, dims_x - 4):
                    split_0 = [util_select_op_base.SplitInput([0, [i], [-1], [-1]]),
                               util_select_op_base.SplitOutput([0, [i]])]
                    axis_split_matrix.append(split_0)
        else:
            axis_split_matrix = []
            for i in range(dims_x):
                if i in (dims_x - 1, dims_x - 4):
                    split_0 = [util_select_op_base.SplitInput([0, [i], [-1], [-1]]),
                               util_select_op_base.SplitOutput([0, [i]])]
                    axis_split_matrix.append(split_0)
        axis_reduce_list = None
    elif format_x == "ND":
        if new_axis < 0:
            new_axis = new_axis + len(input_x.get("shape"))
        axis_split_matrix = []
        for i in range(dims_x):
            if i != new_axis:
                split_0 = [util_select_op_base.SplitInput([0, [i], [-1], [-1]]),
                           util_select_op_base.SplitOutput([0, [i]])]
                axis_split_matrix.append(split_0)
        axis_reduce_list = None
    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


def select_nd_to_5d(dtype, shape_x_ori, axis):
    """
    select nd to 5d
    """
    length_x_ori = len(shape_x_ori)
    if not isinstance(axis, int):
        axis = list(axis)
    else:
        axis = [axis]
    nd_to_5d = 0
    if ((dtype == "float16" and shape_x_ori[-1] % 16 != 0)
        or (dtype == "float32" and shape_x_ori[-1] % 8 !=
            0)) and length_x_ori in (3, 4):
        if (axis[0] == -1 and len(axis) == 1):
            nd_to_5d = 1
        else:
            nd_to_5d = 0

    return nd_to_5d


def check_axis_is_last(shape_x_ori, axis):
    """
    check axis is last
    """
    length_x_ori = len(shape_x_ori)
    if not isinstance(axis, int):
        axis = list(axis)
    else:
        axis = [axis]
    axis_is_last = 0

    if axis[0] == -1 or axis[0] == length_x_ori - 1:
        axis_is_last = 1
    return axis_is_last


def op_select_format(input_x, output_y, axis=-1, half_to_float=False, kernel_name="softmax_v2"):
    """
    select format dynamically \n
    1.when is dynamic softmax, the formats of x and y are the same and only support ND.

        example:
        original:
        x's Tensor(shape=(16, 16, 16), "ND")
        y's Tensor(shape=(16, 16, 16), "ND")

    2.when the lengths of x's shape and y's shape are the same and equal to 2,
    the formats of x and y are the same and are one of [FRACTAL_NZ,NC1HWC0,ND].

        example:
        original:
        x's Tensor(shape=(16, 16, 16, 16, 16), "FRACTAL_NZ")
        y's Tensor(shape=(16, 16, 16, 16, 16), "FRACTAL_NZ")

    3.when the lengths of x's shape and y's shape are the same and larger than 2,
    the formats of x and y are the same and are one of [FRACTAL_NZ,NC1HWC0,ND,NDC1HWC0].

        example:
        original:
        x's Tensor(shape=(16, 16, 16, 16, 16, 16), "NDC1HWC0")
        y's Tensor(shape=(16, 16, 16, 16, 16, 16), "NDC1HWC0")
    """
    shape_x_ori = shape_util.scalar2tensor_one(input_x.get("ori_shape"))
    length_x_ori = len(shape_x_ori)
    dtype = input_x.get("dtype").lower()
    ori_input_format = input_x.get("ori_format")
    tbe_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if length_x_ori == 2:
        if shape_x_ori[0] == 1:
            if tbe_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
                input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                        datatype="float16,float16",
                                                        format="NC1HWC0,ND")
                output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                        datatype="float16,float16",
                                                        format="NC1HWC0,ND")
            if tbe_product in ("Ascend610", "BS9SX1A", "Ascend310P",):
                input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                        datatype="float16,float16,float",
                                                        format="NC1HWC0,ND,ND")
                output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                        datatype="float16,float16,float",
                                                        format="NC1HWC0,ND,ND")
            if tbe_product in ("Ascend910", "Ascend310") or tbe_platform.api_check_support("tik.vcopy"):
                input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                        datatype="float16,float16,float,float",
                                                        format="NC1HWC0,ND,ND,NC1HWC0")
                output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                        datatype="float16,float16,float,float",
                                                        format="NC1HWC0,ND,ND,NC1HWC0")
        else:
            if tbe_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
                input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                        datatype="float16,float16,float16",
                                                        format="FRACTAL_NZ,NC1HWC0,ND")
                output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                        datatype="float16,float16,float16",
                                                        format="FRACTAL_NZ,NC1HWC0,ND")
            if tbe_product in ("Ascend610", "BS9SX1A", "Ascend310P",):
                input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                        datatype="float16,float16,float16,float",
                                                        format="FRACTAL_NZ,NC1HWC0,ND,ND")
                output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                        datatype="float16,float16,float16,float",
                                                        format="FRACTAL_NZ,NC1HWC0,ND,ND")
            if tbe_product in ("Ascend910", "Ascend310") or tbe_platform.api_check_support("tik.vcopy"):
                input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                        datatype="float16,float16,float16,float,float",
                                                        format="FRACTAL_NZ,NC1HWC0,ND,ND,NC1HWC0")
                output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                        datatype="float16,float16,float16,float,float",
                                                        format="FRACTAL_NZ,NC1HWC0,ND,ND,NC1HWC0")
    elif length_x_ori > 2 and (shape_x_ori[-1] % 16 != 0 or shape_x_ori[-2] % 16 != 0):
        if tbe_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
            input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                   datatype="float16,float16,float16",
                                                   format="NC1HWC0,NDC1HWC0,ND")
            output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                    datatype="float16,float16,float16",
                                                    format="NC1HWC0,NDC1HWC0,ND")
        if tbe_product in ("Ascend610", "BS9SX1A", "Ascend310P",):
            if _is_special_cases(shape_x_ori, 0):
                input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                       datatype="float16",
                                                       format="FRACTAL_NZ")
                output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                        datatype="float16",
                                                        format="FRACTAL_NZ")
            else:
                input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                       datatype="float16,float16,float16,float,float16,float",
                                                       format="NC1HWC0,NDC1HWC0,ND,ND,FRACTAL_NZ,FRACTAL_NZ")
                output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                        datatype="float16,float16,float16,float,float16,float",
                                                        format="NC1HWC0,NDC1HWC0,ND,ND,FRACTAL_NZ,FRACTAL_NZ")
        if tbe_product in ("Ascend910",) or tbe_platform.api_check_support("tik.vcopy"):
            if shape_x_ori[-2] < 5:
                input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                       datatype="float16,float16,float,float,float16,float",
                                                       format="NC1HWC0,ND,ND,NC1HWC0,NDC1HWC0,NDC1HWC0")
                output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                        datatype="float16,float16,float,float,float16,float",
                                                        format="NC1HWC0,ND,ND,NC1HWC0,NDC1HWC0,NDC1HWC0")
            else:
                input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                       datatype="float16,float16,float,float,"
                                                                "float16,float,float16,float",
                                                       format="NC1HWC0,ND,ND,NC1HWC0,"
                                                              "NDC1HWC0,NDC1HWC0,FRACTAL_NZ,FRACTAL_NZ")
                output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                        datatype="float16,float16,float,float,"
                                                                 "float16,float,float16,float",
                                                        format="NC1HWC0,ND,ND,NC1HWC0,"
                                                               "NDC1HWC0,NDC1HWC0,FRACTAL_NZ,FRACTAL_NZ")
        if tbe_product in ("Ascend310",):
            if select_nd_to_5d(dtype, shape_x_ori, axis):
                # Supplement dimensions to find the C-axis
                if len(shape_x_ori) < 4:
                    if isinstance(shape_x_ori, list):
                        shape_x_ori.insert(0, 1)
                    else:
                        shape_x_ori = (1,) + shape_x_ori
                if ori_input_format == "NCHW" and shape_x_ori[1] <= 16:
                    input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                           datatype="float16,float",
                                                           format="ND,ND")
                    output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                            datatype="float16,float",
                                                            format="ND,ND")
                else:
                    input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                           datatype="float16,float",
                                                           format="NC1HWC0,NC1HWC0")
                    output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                            datatype="float16,float",
                                                            format="NC1HWC0,NC1HWC0")
            else:
                if shape_x_ori[-2] < 4:
                    input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                           datatype="float16,float16,float,float,float16,float",
                                                           format="NC1HWC0,ND,ND,NC1HWC0,NDC1HWC0,NDC1HWC0")
                    output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                            datatype="float16,float16,float,float,float16,float",
                                                            format="NC1HWC0,ND,ND,NC1HWC0,NDC1HWC0,NDC1HWC0")
                else:
                    input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                           datatype="float16,float16,float,float,"
                                                                    "float16,float,float16,float",
                                                           format="NC1HWC0,ND,ND,NC1HWC0,NDC1HWC0,NDC1HWC0,"
                                                                  "FRACTAL_NZ,FRACTAL_NZ")
                    output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                            datatype="float16,float16,float,float,"
                                                                     "float16,float,float16,float",
                                                            format="NC1HWC0,ND,ND,NC1HWC0,NDC1HWC0,NDC1HWC0,"
                                                                   "FRACTAL_NZ,FRACTAL_NZ")
    else:
        if tbe_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
            input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                   datatype="float16,float16,float16,float16",
                                                   format="FRACTAL_NZ,NC1HWC0,ND,NDC1HWC0")
            output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                    datatype="float16,float16,float16,float16",
                                                    format="FRACTAL_NZ,NC1HWC0,ND,NDC1HWC0")
        if tbe_product in ("Ascend610", "BS9SX1A", "Ascend310P",):
            input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                   datatype="float16,float,float16,float16,float,float16",
                                                   format="FRACTAL_NZ,FRACTAL_NZ,NC1HWC0,ND,ND,NDC1HWC0")
            output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                    datatype="float16,float,float16,float16,float,float16",
                                                    format="FRACTAL_NZ,FRACTAL_NZ,NC1HWC0,ND,ND,NDC1HWC0")
        if tbe_product in ("Ascend910", "Ascend310") or tbe_platform.api_check_support("tik.vcopy"):
            input0 = \
                util_select_op_base.gen_param(classify="input0", name="x",
                                              datatype="float16,float,float16,float16,float,float,float16,float",
                                              format="FRACTAL_NZ,FRACTAL_NZ,NC1HWC0,ND,ND,NC1HWC0,NDC1HWC0,NDC1HWC0")
            output0 = \
                util_select_op_base.gen_param(classify="output0", name="y",
                                              datatype="float16,float,float16,float16,float,float,float16,float",
                                              format="FRACTAL_NZ,FRACTAL_NZ,NC1HWC0,ND,ND,NC1HWC0,NDC1HWC0,NDC1HWC0")

    param_list = [input0, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def _broadcast_nz(tensor, shape):
    broadcast_axes = []
    src_shape = te.lang.cce.util.shape_to_list(tensor.shape)
    for i, _ in enumerate(shape):
        if shape[i] != src_shape[i]:
            broadcast_axes.append(i)
    if len(broadcast_axes) == 2 and \
            broadcast_axes[1] - broadcast_axes[0] != 1 and \
            broadcast_axes[1] + 1 == len(shape):
        temp_shape = src_shape[:-1] + [shape[-1]]
        tensor = te.lang.cce.broadcast(tensor, temp_shape)
    tensor = te.lang.cce.broadcast(tensor, shape)
    return tensor


def is_white_shape(shape):
    """
    is_white_shape
    """
    white_list_shape = [[4096, 3, 49, 49], [1024, 6, 49, 49], [256, 12, 49, 49], [64, 24, 49, 49],
                        [128, 8, 12, 12], [262144, 4, 1, 4]]
    shape_t = list(shape)
    if shape_t in white_list_shape:
        return True
    return False


def _is_special_cases(input_shape, compare_type):
    tbe_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if tbe_product not in ("Ascend610", "BS9SX1A", "Ascend310P",):
        return False
    white_list_shape = [[8, 8732, 81], [16, 8732, 81], [96, 50, 50],
                        [192, 50, 50], [384, 50, 50], [768, 50, 50],
                        [192, 197, 197], [384, 197, 197], [96, 197, 197],
                        [512, 3, 49, 49], [1024, 3, 49, 49], [2048, 3, 49, 49], [4096, 3, 49, 49],
                        [512, 4, 49, 49], [1024, 4, 49, 49], [2048, 4, 49, 49], [4096, 4, 49, 49],
                        [128, 6, 49, 49], [256, 6, 49, 49], [512, 6, 49, 49],
                        [1024, 6, 49, 49], [2048, 6, 49, 49], [4096, 6, 49, 49],
                        [128, 8, 49, 49], [256, 8, 49, 49], [512, 8, 49, 49], [1024, 8, 49, 49],
                        [32, 12, 49, 49], [64, 12, 49, 49], [128, 12, 49, 49],
                        [256, 12, 49, 49], [512, 12, 49, 49], [1024, 12, 49, 49],
                        [32, 16, 49, 49], [64, 16, 49, 49], [128, 16, 49, 49], [256, 16, 49, 49],
                        [8, 24, 49, 49], [16, 24, 49, 49], [32, 24, 49, 49],
                        [64, 24, 49, 49], [128, 24, 49, 49], [256, 24, 49, 49],
                        [8, 32, 49, 49], [16, 32, 49, 49], [32, 32, 49, 49], [64, 32, 49, 49],
                        [8, 48, 49, 49], [16, 48, 49, 49], [32, 48, 49, 49], [64, 48, 49, 49]]
    shape_t = list(input_shape)
    if compare_type == 0:
        if shape_t in white_list_shape:
            return True
    else:
        shape_t_size = len(shape_t)
        for shape_w in white_list_shape:
            count = 0
            if shape_t_size == len(shape_w):
                for i in range(shape_t_size):
                    if shape_t[i].value == shape_w[i]:
                        count += 1
                        continue
                    break
                if count == shape_t_size:
                    return True

    return False


def check_supported(input_x, output_y, axis=-1, half_to_float=False, kernel_name="softmax_v2"):
    """
    Judge whether the current input specification supports
    """
    return True, ""


# 'pylint: disable=variable_type_changed
@fusion_manager.register("softmax_v2")
def softmax_v2_compute(input_x, output_y, axis=-1, half_to_float=False, kernel_name="softmax_v2",
                       impl_mode="high_performance"):
    """
    calculating data's softmax, produces an output tensor with shape
    the result equals the sum of the x power of e over the sum of
    the x power of e

    Parameters
    ----------
    input_x: TVM tensor
        the placeholder of input data
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    axis : int or list or tuple
        the data's axis, range == [-d, d-1]
    half_to_float: bool
        if it is true and input dtype is float16, output dtype should be float32
        otherwise, output dtype should be same as input dtype
    kernel_name: str
        cce kernel name, default value is softmax_v2

    Returns
    -------
    output: TVM tensor
        the result of softmax
    """

    dtype = input_x.dtype
    shape = te.lang.cce.util.shape_to_list(input_x.shape)
    axis = list(axis)
    vcmax_flag = False
    last_dim = len(input_x.shape) - 1

    for i in axis:
        if i in (-1, last_dim):
            vcmax_flag = True
    use_tail_block = False
    ori_shape = input_x.op.attrs["ori_shape"]
    if output_y.get("format") == "FRACTAL_NZ" and len(axis) == 2 and ori_shape[-1].value % 16 != 0:
        input_x = te.lang.cce.vadds(input_x, 0)
        with tvm.tag_scope("tail_block_pretreatment"):
            lambda_func = lambda *indice: tvm.const(-65000, input_x.dtype)
            temp = tvm.compute(input_x.shape, lambda_func, name="tail_block_pretreatment")
        input_x = te.lang.cce.vadd(input_x, temp)
        use_tail_block = True

    if dtype == "float32" and vcmax_flag and \
        not tbe_platform.api_check_support(
            "te.lang.cce.reduce_max", "float32"):
        data_max_input = te.lang.cce.cast_to(input_x, "float16")
        data_max_output = te.lang.cce.reduce_max(data_max_input,
                                                 axis=axis, keepdims=True)
        data_max = te.lang.cce.cast_to(data_max_output, "float32")
    elif dtype == "float32" and vcmax_flag and \
        tbe_platform.api_check_support(
            "te.lang.cce.reduce_max", "float32") and len(axis) == 1:
        data_max = te.lang.cce.reduce_max(input_x, axis=axis, keepdims=True, priority_flag=True)
    else:
        if impl_mode == "high_precision":
            data_max = te.lang.cce.reduce_max(input_x, axis=axis, keepdims=True, priority_flag=True)
        else:
            data_max = te.lang.cce.reduce_max(input_x, axis=axis, keepdims=True)

    data_max = _broadcast_nz(data_max, shape)
    data_subtrac = te.lang.cce.vsub(input_x, data_max)

    has_improve_precision = False
    if dtype == "float16" and tbe_platform.api_check_support(
            "te.lang.cce.vexp", "float32"):
        data_subtrac = te.lang.cce.cast_to(data_subtrac, "float32")
        has_improve_precision = True
    data_exp = te.lang.cce.vexp(data_subtrac)

    tbe_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if data_exp.dtype == "float16" and tbe_product in ("Ascend310",):
        data_exp = te.lang.cce.cast_to(data_exp, "float32")
        has_improve_precision = True
    data_expsum = te.lang.cce.sum(data_exp, axis, keepdims=True)

    if use_tail_block:
        if not _is_special_cases(ori_shape, 1):
            data_expsum = _broadcast_nz(data_expsum, shape)
            output = te.lang.cce.vdiv(data_exp, data_expsum)
        else:
            data_expsum = te.lang.cce.vrec(data_expsum, priority_flag=0)
            data_expsum = _broadcast_nz(data_expsum, shape)
            output = te.lang.cce.vmul(data_exp, data_expsum)
    elif (tbe_product in ("Ascend910", "Ascend610", "BS9SX1A", "Ascend310P") or
          tbe_platform.api_check_support("tik.vcopy")) and \
            output_y.get("format") == "FRACTAL_NZ" and dtype == "float16":
        data_expsum = te.lang.cce.vrec(data_expsum, priority_flag=0)
        data_expsum = _broadcast_nz(data_expsum, shape)
        output = te.lang.cce.vmul(data_exp, data_expsum)
    else:
        data_expsum = _broadcast_nz(data_expsum, shape)
        output = te.lang.cce.vdiv(data_exp, data_expsum)

    if has_improve_precision and dtype == "float16":
        output = te.lang.cce.cast_to(output, "float16")

    return output


def buffer_mapping(schedule, ops):
    """
    set buffer scope
    """
    if not ops:
        error_manager_vector.raise_err_specific_reson("softmax_v2", "operation list is empty")
    for i_op in ops:
        schedule[i_op].set_scope(cce.scope_ubuf)


def align(schedule, ops, pad_param, factor=16, offset=0):
    """
    determine if aligning needs to be enabled
    """
    if not ops:
        error_manager_vector.raise_err_specific_reson("softmax_v2", "operation list is empty")
    length = len(ops)
    if length <= 3:
        # no op need aligning
        return
    for i in range(0, length - 1):
        shape_len = len(ops[i].shape)
        if shape_len > 1:
            if ops[i].shape[1].value == 1 and pad_param[1] == 15:
                if ops[i].shape[4].value == 1:
                    schedule[ops[i]].storage_align(ops[i].op.axis[shape_len - 2],
                                                   factor, offset)
                if ops[i].op.name == "res_vonv_fp32_max":
                    schedule[ops[i]].storage_align(ops[i].op.axis[shape_len - 2],
                                                   factor, offset)
            else:
                if ops[i].shape[4].value == 1:
                    schedule[ops[i]].storage_align(ops[i].op.axis[shape_len - 3],
                                                   factor, offset)
                if ops[i].op.name == "res_vonv_fp32_max":
                    schedule[ops[i]].storage_align(ops[i].op.axis[shape_len - 3],
                                                   factor, offset)


def align_nz(schedule, ops, pad_param, factor=16, offset=0):
    """
    determine if aligning needs to be enabled
    """
    if not ops:
        error_manager_vector.raise_err_specific_reson("softmax_v2", "operation list is empty")
    length = len(ops)
    if length <= 3:
        # no op need aligning
        return
    for i in range(0, length-1):
        shape_len = len(ops[i].shape)
        if shape_len > 1:
            if ops[i].shape[0].value == 1 and pad_param[1] == 15:
                if ops[i].shape[3].value == 1:
                    schedule[ops[i]].storage_align(ops[i].op.axis[shape_len - 1],
                                                   factor, offset)
                if ops[i].op.name == "res_vonv_fp32_max":
                    schedule[ops[i]].storage_align(ops[i].op.axis[shape_len - 1],
                                                   factor, offset)
            else:
                if ops[i].shape[3].value == 1:
                    schedule[ops[i]].storage_align(ops[i].op.axis[shape_len - 2],
                                                   factor, offset)
                if ops[i].op.name == "res_vonv_fp32_max":
                    schedule[ops[i]].storage_align(ops[i].op.axis[shape_len - 2],
                                                   factor, offset)


def multicore_factor_calculate(shape):
    """
    the compute produce, calculate multicore information
    """
    if not shape:
        error_manager_vector.raise_err_specific_reson("softmax_v2", "input shape is empty")

    device_core_num = \
        tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

    split_axis = 0
    split_size = 0
    if shape[0] >= device_core_num:
        npart_n = device_core_num
        npart_h = 1
        npart_w = 1
        split_axis = 1
        split_size = math.ceil(shape[0] / device_core_num)
    elif device_core_num // shape[0] <= shape[2]:
        npart_n = shape[0]
        npart_h = device_core_num // shape[0]
        npart_w = 1
        split_axis = 2
        split_size = math.ceil(shape[2] / (device_core_num // shape[0]))
    elif device_core_num // shape[0] // shape[2] <= shape[3]:
        npart_n = shape[0]
        npart_h = shape[2]
        npart_w = (device_core_num // shape[0] // shape[2])
        split_axis = 3
        split_size = math.ceil(shape[3] / (device_core_num // shape[0] // shape[2]))
    else:
        npart_n = shape[0]
        npart_h = shape[2]
        npart_w = shape[3]
        split_axis = 4
        split_size = 1
    return_list = [npart_n, npart_h, npart_w, split_axis, split_size]
    return return_list


def multicore_factor_calculate_nz(shape):
    """
    the compute produce, calculate multicore information
    """
    if not shape:
        error_manager_vector.raise_err_specific_reson("softmax_v2", "input shape is empty")

    device_core_num = \
        tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

    split_axis = 0
    split_size = 0
    if shape[1] >= device_core_num:
        npart_n1 = device_core_num
        npart_n0 = 1
        split_axis = 1
        split_size = math.ceil(shape[1] / device_core_num)
    elif device_core_num // shape[1] <= shape[2]:
        npart_n1 = shape[1]
        npart_n0 = device_core_num // shape[1]
        split_axis = 2
        split_size = math.ceil(shape[2] / (device_core_num // shape[1]))
    else:
        npart_n1 = shape[1]
        npart_n0 = shape[2]
        split_axis = 3
        split_size = 1
    return_list = [npart_n1, npart_n0, split_axis, split_size]
    return return_list


def tiling_factor_calculate(shape, split_axis_0, split_size, use_fp32):
    """
    do tiling calculate
    """
    if not shape:
        error_manager_vector.raise_err_specific_reson("softmax_v2", "input shape is empty")

    if use_fp32:
        temp = UB_SIZE_LIMIT // (shape[1] * shape[4] * 4)
    else:
        temp = UB_SIZE_LIMIT // (shape[1] * shape[4] * 2)

    split_flag = False
    split_axis = 0
    split_factor = 0
    if split_axis_0 == 1:
        if temp >= split_size * shape[2] * shape[3]:
            # no split
            split_flag = False
        elif shape[2] * shape[3] <= temp < split_size * shape[2] * shape[3]:
            # split on n.inner
            split_flag = True
            split_axis = 0
            split_factor = int(temp // (shape[2] * shape[3]))
        elif shape[3] <= temp < shape[2] * shape[3]:
            # split on h
            split_flag = True
            split_axis = 2
            split_factor = int(temp // shape[3])
        elif temp < shape[3]:
            # split on w
            split_flag = True
            split_axis = 3
            split_factor = int(temp)
    if split_axis_0 == 2:
        if temp >= split_size * shape[3]:
            # no split
            split_flag = False
        elif shape[3] <= temp < shape[2] * shape[3]:
            # split on h
            split_flag = True
            split_axis = 2
            split_factor = int(temp // shape[3])
        elif temp < shape[3]:
            # split on w
            split_flag = True
            split_axis = 3
            split_factor = int(temp)
    if split_axis_0 == 3:
        if temp >= split_size:
            # no split
            split_flag = False
        else:
            # split on w
            split_flag = True
            split_axis = 3
            split_factor = int(temp)
    if split_axis_0 == 4:
        # no split
        split_flag = False

    return split_flag, split_axis, split_factor


def tiling_factor_calculate_nz(shape, split_axis_0, split_size, use_fp32):
    """
    do tiling calculate
    """
    if not shape:
        error_manager_vector.raise_err_specific_reson("softmax_v2", "input shape is empty")

    if use_fp32:
        temp = UB_SIZE_LIMIT // (shape[0] * shape[3] * 4)
    else:
        temp = UB_SIZE_LIMIT // (shape[0] * shape[3] * 2)

    split_flag = False
    split_axis = 0
    split_factor = 0
    if split_axis_0 == 1:
        if temp >= split_size * shape[2]:
            # no split
            split_flag = False
        elif shape[2] <= temp < split_size * shape[2]:
            # split on n.inner
            split_flag = True
            split_axis = 1
            split_factor = int(temp // shape[2])
        elif temp < shape[3]:
            # split on w
            split_flag = True
            split_axis = 2
            split_factor = int(temp)
    if split_axis_0 == 2:
        if temp >= split_size:
            # no split
            split_flag = False
        else:
            # split on w
            split_flag = True
            split_axis = 2
            split_factor = int(temp)
    if split_axis_0 == 3:
        # no split
        split_flag = False

    return split_flag, split_axis, split_factor


def ops_integrate(schedule, ops, axis):
    """
    determine if integrating needs to be enabled
    """
    if not ops:
        error_manager_vector.raise_err_specific_reson("softmax_v2", "operation list is empty")

    length = len(ops)
    if length < 2:
        # no op need integrating
        return
    integration_op = schedule[ops[-1]]
    for i in range(0, length - 1):
        schedule[ops[i]].compute_at(integration_op, axis)


def double_buf(schedule, ops):
    """
    determine if double buffer needs to be enabled
    """
    if not ops:
        error_manager_vector.raise_err_specific_reson("softmax_v2", "operation list is empty")

    length = len(ops)
    if length < 2:
        # no op need double buffer
        return
    for i in range(0, length - 1):
        schedule[ops[i]].double_buffer()


def emit_axis_collect(ops, pad_param, instructions, last_axis):
    """
    the compute produce, emit axis information
    """
    if not ops:
        error_manager_vector.raise_err_specific_reson("softmax_v2", "operation list is empty")
    if len(ops) != len(instructions):
        error_manager_vector.raise_err_specific_reson("softmax_v2",
                                                      "length of operations and instructions does not match")

    axis_list = []
    length = len(ops)
    for i in range(0, length - 1):
        if pad_param[1] == 15:
            axis_list += [ops[i].op.axis[1]]
        else:
            if instructions[i] == 'vector_adds' and \
                    ops[i].op.name == 'res_sub':
                axis_list += [ops[i].op.axis[1]]
            elif instructions[i] == 'vector_muls' and \
                    ops[i].op.name == 'res_mul':
                axis_list += [ops[i].op.axis[1]]
            else:
                axis_list += [ops[i].op.axis[0]]
    axis_list += [last_axis]
    return axis_list


def emit_nz_axis_collect(ops, pad_param, instructions, last_axis):
    """
    the compute produce, emit axis information
    """
    if not ops:
        error_manager_vector.raise_err_specific_reson("softmax_v2", "operation list is empty")
    if len(ops) != len(instructions):
        error_manager_vector.raise_err_specific_reson("softmax_v2",
                                                      "length of operations and instructions does not match")
    axis_list = []
    length = len(ops)
    for i in range(0, length-1):
        axis_list += [ops[i].op.axis[0]]
    axis_list += [last_axis]
    return axis_list


def axis_reorder(schedule, ops, instructions):
    """
    the compute produce, reorder axis
    """
    if not ops:
        error_manager_vector.raise_err_specific_reson("softmax_v2", "operation list is empty")
    if len(ops) != len(instructions):
        error_manager_vector.raise_err_specific_reson("softmax_v2",
                                                      "length of operations and instructions does not match")
    length = len(ops)
    for i in range(0, length):
        if instructions[i] == 'vector_adds' or instructions[i] == 'vector_muls':
            schedule[ops[i]].reorder(ops[i].op.axis[0], ops[i].op.axis[2],
                                     ops[i].op.axis[3], ops[i].op.axis[1],
                                     ops[i].op.axis[4])


def axis_reorder_nz(schedule, ops, instructions):
    """
    the compute produce, reorder axis
    """
    if not ops:
        error_manager_vector.raise_err_specific_reson("softmax_v2", "operation list is empty")
    if len(ops) != len(instructions):
        error_manager_vector.raise_err_specific_reson("softmax_v2",
                                                      "length of operations and instructions does not match")
    length = len(ops)
    for i in range(0, length):
        if instructions[i] == 'vector_adds' or instructions[i] == 'vector_muls':
            schedule[ops[i]].reorder(ops[i].op.axis[1], ops[i].op.axis[2],
                                     ops[i].op.axis[0], ops[i].op.axis[3])


def instructions_replace(schedule, ops, axes, instructions):
    """
    the compute produce, replace instructions
    """
    if not ops:
        error_manager_vector.raise_err_specific_reson("softmax_v2", "operation list is empty")
    if len(ops) != len(instructions):
        error_manager_vector.raise_err_specific_reson("softmax_v2",
                                                      "length of operations and instructions does not match")
    length = len(ops)
    for i in range(0, length):
        schedule[ops[i]].emit_insn(axes[i], instructions[i])


def check_isusefp32(shape, dtype):
    """
    check compute wheather ues fp32
    """
    use_fp32 = True
    if dtype == "float32":
        use_fp32 = True
        return use_fp32
    tbe_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if tbe_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        use_fp32 = False
        return use_fp32
    if shape[1] * shape[4] * 4 > UB_SIZE_LIMIT:
        use_fp32 = False
        return use_fp32
    return use_fp32


def check_nz_isusefp32(shape, dtype):
    """
    check nz format compute wheather ues fp32
    """
    use_fp32 = True
    if dtype == "float32":
        use_fp32 = True
        return use_fp32
    tbe_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if tbe_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        use_fp32 = False
        return use_fp32
    if shape[0] * shape[3] * 4 > UB_SIZE_LIMIT:
        use_fp32 = False
        return use_fp32
    return use_fp32


def compute_nopad_fp32(tensor_in, shape):
    """
    the compute produce, handling the scenes without padding
    """
    # preparing
    reduce_shape = (shape[0], 1, shape[2], shape[3], 1)
    op_list = []
    instruction_list = []

    # move data from gm to ub
    tensor_in_ub = \
        tvm.compute(shape, lambda *i: tensor_in(*i), name="tensor_in_ub")
    op_list += [tensor_in_ub]
    instruction_list += ['dma_copy']

    if tensor_in.dtype == "float32":
        # vcov fp32tofp16
        tensor_in_ub_fp16 = te.lang.cce.cast_to(tensor_in_ub, "float16")
        tensor_in_ub_fp16 = tvm.compute(shape,
                                        lambda *i: shape_util.cast(tensor_in_ub(*i), "float16"),
                                        name='res_vonv_fp16_tensor')
        op_list += [tensor_in_ub_fp16]
        instruction_list += ['vector_conv']

        # reduce max
        i = tvm.reduce_axis((0, shape[1]), "c1_max")
        j = tvm.reduce_axis((0, shape[4]), "c0_max")
        res_max = tvm.compute(reduce_shape,
                              lambda n, c1, h, w, c0:
                              tvm.max(tensor_in_ub_fp16[n, i, h, w, j],
                                      axis=[i, j]), name="res_max")
        op_list += [res_max]
        instruction_list += ['vector_reduce_max']

        res_max_fp32 = te.lang.cce.cast_to(res_max, "float32")
        res_max_fp32 = tvm.compute(shape,
                                   lambda *i: shape_util.cast(res_max(*i), "float32"),
                                   name='res_vonv_fp32_max')
        op_list += [res_max_fp32]
        instruction_list += ['vector_conv']

        # sub
        minus = tvm.const(-1, 'float32')
        res_minus = tvm.compute(reduce_shape,
                                lambda *i: minus * res_max_fp32(*i), name="res_minus")
        op_list += [res_minus]
        instruction_list += ['vector_muls']

        res_sub_fp32 = tvm.compute(shape, lambda n, c1, h, w, c0:
                                   tensor_in_ub[n, c1, h, w, c0] + res_minus[n, 0, h, w, 0], name="res_sub")
        op_list += [res_sub_fp32]
        instruction_list += ['vector_adds']

        res_sub = te.lang.cce.cast_to(res_sub_fp32, "float16")
        res_sub = tvm.compute(shape,
                              lambda *i: shape_util.cast(res_sub_fp32(*i), "float16"),
                              name='res_vonv_fp16_sub')
        op_list += [res_sub]
        instruction_list += ['vector_conv']

    else:
        # reduce max
        i = tvm.reduce_axis((0, shape[1]), "c1_max")
        j = tvm.reduce_axis((0, shape[4]), "c0_max")
        res_max = tvm.compute(reduce_shape,
                              lambda n, c1, h, w, c0:
                              tvm.max(tensor_in_ub[n, i, h, w, j], axis=[i, j]),
                              name="res_max")
        op_list += [res_max]
        instruction_list += ['vector_reduce_max']

        # sub
        minus = tvm.const(-1, 'float16')
        res_minus = tvm.compute(reduce_shape, lambda *i: minus * res_max(*i), name="res_minus")
        op_list += [res_minus]
        instruction_list += ['vector_muls']

        res_sub = tvm.compute(shape,
                              lambda n, c1, h, w, c0:
                              tensor_in_ub[n, c1, h, w, c0] + res_minus[n, 0, h, w, 0],
                              name="res_sub")
        op_list += [res_sub]
        instruction_list += ['vector_adds']

    if not tbe_platform.intrinsic_check_support("Intrinsic_exp",
                                                         "float32"):
        # exp
        res_exp = tvm.compute(shape, lambda *i: tvm.exp(res_sub(*i)), name="res_exp")
        op_list += [res_exp]
        instruction_list += ['vector_exp']

        # vcov fp16tofp32
        res_exp_fp32 = te.lang.cce.cast_to(res_exp, "float32")
        res_exp_fp32 = tvm.compute(shape,
                                   lambda *i: shape_util.cast(res_exp(*i), "float32"),
                                   name='res_vonv_fp32_exp')
        op_list += [res_exp_fp32]
        instruction_list += ['vector_conv']
    else:
        # vcov fp16tofp32
        res_sub_fp32 = te.lang.cce.cast_to(res_sub, "float32")
        res_sub_fp32 = tvm.compute(shape, lambda *i: shape_util.cast(res_sub(*i), "float32"),
                                   name='res_vonv_fp32_exp')
        op_list += [res_sub_fp32]
        instruction_list += ['vector_conv']

        # exp
        res_exp_fp32 = tvm.compute(shape, lambda *i: tvm.exp(res_sub_fp32(*i)), name="res_exp")
        op_list += [res_exp_fp32]
        instruction_list += ['vector_exp']

    # reduce sum
    ii = tvm.reduce_axis((0, shape[1]), "c1_sum")
    jj = tvm.reduce_axis((0, shape[4]), "c0_sum")
    res_sum = tvm.compute(reduce_shape,
                          lambda n, c1, h, w, c0:
                          tvm.sum(res_exp_fp32[n, ii, h, w, jj], axis=[ii, jj]),
                          name="res_sum")
    op_list += [res_sum]
    instruction_list += ['vector_reduce_sum']

    # rec
    res_rec = tvm.compute(reduce_shape, lambda *i: 1 / (res_sum(*i)), name="res_rec")
    op_list += [res_rec]
    instruction_list += ['vector_rec']

    # loop 1 do newton iteration
    # vmul
    res_mul_newton = tvm.compute(reduce_shape,
                                 lambda *i: res_rec(*i) * res_sum(*i), name="res_mul_newton")
    op_list += [res_mul_newton]
    instruction_list += ['vector_mul']

    # vmuls
    const1 = tvm.const(-1, 'float32')
    res_muls_newton = tvm.compute(reduce_shape,
                                  lambda *i: const1 * res_mul_newton(*i), name="res_muls_newton")
    op_list += [res_muls_newton]
    instruction_list += ['vector_muls']

    # vadds
    const2 = tvm.const(2, 'float32')
    res_adds_newton = tvm.compute(reduce_shape,
                                  lambda *i: const2 + res_muls_newton(*i), name="res_adds_newton")
    op_list += [res_adds_newton]
    instruction_list += ['vector_adds']

    # vmlu
    res_mul_newton = tvm.compute(reduce_shape,
                                 lambda *i: res_rec(*i) * res_adds_newton(*i),
                                 name="res_mul_newton")
    op_list += [res_mul_newton]
    instruction_list += ['vector_mul']

    # mul
    res_mul = tvm.compute(shape,
                          lambda n, c1, h, w, c0:
                          res_exp_fp32[n, c1, h, w, c0] * res_mul_newton[n, 0, h, w, 0],
                          name="res_mul")
    op_list += [res_mul]
    instruction_list += ['vector_muls']

    if tensor_in.dtype == "float16":
        res_mul_fp16 = te.lang.cce.cast_to(res_mul, "float16")
        res_mul_fp16 = tvm.compute(shape,
                                   lambda *i: shape_util.cast(res_mul(*i), "float16"),
                                   name='res_vonv_fp16')
        op_list += [res_mul_fp16]
        instruction_list += ['vector_conv']

        # move data from ub to gm
        res = tvm.compute(shape, lambda *i: res_mul_fp16(*i), name="result_out")
        op_list += [res]
        instruction_list += ['dma_copy']
    else:
        # move data from ub to gm
        res = tvm.compute(shape, lambda *i: res_mul(*i), name="result_out")
        op_list += [res]
        instruction_list += ['dma_copy']

    # schedule
    schedule = tvm.create_schedule([res.op])
    return schedule, op_list, instruction_list


def compute_nz_nopad_fp32(tensor_in, shape):
    """
    the compute produce, handling the scenes without padding
    """
    # preparing
    reduce_shape = (1, shape[1], shape[2], 1)
    op_list = []
    instruction_list = []

    # move data from gm to ub
    tensor_in_ub = \
        tvm.compute(shape, lambda *i: tensor_in(*i), name="tensor_in_ub")
    op_list += [tensor_in_ub]
    instruction_list += ['dma_copy']

    if tensor_in.dtype == "float32":
        # vcov fp32tofp16
        tensor_in_ub_fp16 = te.lang.cce.cast_to(tensor_in_ub, "float16")
        tensor_in_ub_fp16 = tvm.compute(shape,
                                        lambda *i: shape_util.cast(tensor_in_ub(*i), "float16"),
                                        name='res_vonv_fp16_tensor')
        op_list += [tensor_in_ub_fp16]
        instruction_list += ['vector_conv']

        # reduce max
        i = tvm.reduce_axis((0, shape[0]), "c1_max")
        j = tvm.reduce_axis((0, shape[3]), "c0_max")
        res_max = tvm.compute(reduce_shape,
                              lambda c1, n1, n0, c0:
                              tvm.max(tensor_in_ub_fp16[i, n1, n0, j],
                                      axis=[i, j]), name="res_max")
        op_list += [res_max]
        instruction_list += ['vector_reduce_max']

        res_max_fp32 = te.lang.cce.cast_to(res_max, "float32")
        res_max_fp32 = tvm.compute(shape,
                                   lambda *i: shape_util.cast(res_max(*i), "float32"),
                                   name='res_vonv_fp32_max')
        op_list += [res_max_fp32]
        instruction_list += ['vector_conv']

        # sub
        minus = tvm.const(-1, 'float32')
        res_minus = tvm.compute(reduce_shape,
                                lambda *i: minus * res_max_fp32(*i), name="res_minus")
        op_list += [res_minus]
        instruction_list += ['vector_muls']

        res_sub_fp32 = tvm.compute(shape, lambda c1, n1, n0, c0:
                                   tensor_in_ub[c1, n1, n0, c0] + res_minus[0, n1, n0, 0],
                                   name="res_sub")
        op_list += [res_sub_fp32]
        instruction_list += ['vector_adds']

        res_sub = te.lang.cce.cast_to(res_sub_fp32, "float16")
        res_sub = tvm.compute(shape,
                              lambda *i: shape_util.cast(res_sub_fp32(*i), "float16"),
                              name='res_vonv_fp16_sub')
        op_list += [res_sub]
        instruction_list += ['vector_conv']

    else:
        # reduce max
        i = tvm.reduce_axis((0, shape[0]), "c1_max")
        j = tvm.reduce_axis((0, shape[3]), "c0_max")
        res_max = tvm.compute(reduce_shape,
                              lambda c1, n1, n0, c0:
                              tvm.max(tensor_in_ub[i, n1, n0, j], axis=[i, j]),
                              name="res_max")
        op_list += [res_max]
        instruction_list += ['vector_reduce_max']

        # sub
        minus = tvm.const(-1, 'float16')
        res_minus = tvm.compute(reduce_shape, lambda *i: minus * res_max(*i), name="res_minus")
        op_list += [res_minus]
        instruction_list += ['vector_muls']

        res_sub = tvm.compute(shape,
                              lambda c1, n1, n0, c0:
                              tensor_in_ub[c1, n1, n0, c0] + res_minus[0, n1, n0, 0],
                              name="res_sub")
        op_list += [res_sub]
        instruction_list += ['vector_adds']

    if not tbe_platform.intrinsic_check_support("Intrinsic_exp",
                                                         "float32"):
        # exp
        res_exp = tvm.compute(shape, lambda *i: tvm.exp(res_sub(*i)), name="res_exp")
        op_list += [res_exp]
        instruction_list += ['vector_exp']

        # vcov fp16tofp32
        res_exp_fp32 = te.lang.cce.cast_to(res_exp, "float32")
        res_exp_fp32 = tvm.compute(shape,
                                   lambda *i: shape_util.cast(res_exp(*i), "float32"),
                                   name='res_vonv_fp32_exp')
        op_list += [res_exp_fp32]
        instruction_list += ['vector_conv']
    else:
        # vcov fp16tofp32
        res_sub_fp32 = te.lang.cce.cast_to(res_sub, "float32")
        res_sub_fp32 = tvm.compute(shape, lambda *i: shape_util.cast(res_sub(*i), "float32"),
                                   name='res_vonv_fp32_exp')
        op_list += [res_sub_fp32]
        instruction_list += ['vector_conv']

        # exp
        res_exp_fp32 = tvm.compute(shape, lambda *i: tvm.exp(res_sub_fp32(*i)), name="res_exp")
        op_list += [res_exp_fp32]
        instruction_list += ['vector_exp']

    # reduce sum
    ii = tvm.reduce_axis((0, shape[0]), "c1_sum")
    jj = tvm.reduce_axis((0, shape[3]), "c0_sum")
    res_sum = tvm.compute(reduce_shape,
                          lambda c1, n1, n0, c0:
                          tvm.sum(res_exp_fp32[ii, n1, n0, jj], axis=[ii, jj]),
                          name="res_sum")
    op_list += [res_sum]
    instruction_list += ['vector_reduce_sum']

    # rec
    res_rec = tvm.compute(reduce_shape, lambda *i: 1/(res_sum(*i)), name="res_rec")
    op_list += [res_rec]
    instruction_list += ['vector_rec']

    # loop 1  do newton iteration
    # vmlu
    res_mul_newton = tvm.compute(reduce_shape,
                                 lambda *i: res_rec(*i) * res_sum(*i), name="res_mul_newton")
    op_list += [res_mul_newton]
    instruction_list += ['vector_mul']

    # vmuls
    const1 = tvm.const(-1, 'float32')
    res_muls_newton = tvm.compute(reduce_shape,
                                  lambda *i: const1 * res_mul_newton(*i), name="res_muls_newton")
    op_list += [res_muls_newton]
    instruction_list += ['vector_muls']

    # vadds
    const2 = tvm.const(2, 'float32')
    res_adds_newton = tvm.compute(reduce_shape,
                                  lambda *i: const2 + res_muls_newton(*i), name="res_adds_newton")
    op_list += [res_adds_newton]
    instruction_list += ['vector_adds']

    # vmlu
    res_mul_newton = tvm.compute(reduce_shape,
                                 lambda *i: res_rec(*i) * res_adds_newton(*i),
                                 name="res_mul_newton")
    op_list += [res_mul_newton]
    instruction_list += ['vector_mul']

    # mul
    res_mul = tvm.compute(shape,
                          lambda c1, n1, n0, c0:
                          res_exp_fp32[c1, n1, n0, c0] * res_mul_newton[0, n1, n0, 0],
                          name="res_mul")
    op_list += [res_mul]
    instruction_list += ['vector_muls']

    if tensor_in.dtype == "float16":
        res_mul_fp16 = te.lang.cce.cast_to(res_mul, "float16")
        res_mul_fp16 = tvm.compute(shape,
                                   lambda *i: shape_util.cast(res_mul(*i), "float16"),
                                   name='res_vonv_fp16')
        op_list += [res_mul_fp16]
        instruction_list += ['vector_conv']

        # move data from ub to gm
        res = tvm.compute(shape, lambda *i: res_mul_fp16(*i), name="result_out")
        op_list += [res]
        instruction_list += ['dma_copy']
    else:
        # move data from ub to gm
        res = tvm.compute(shape, lambda *i: res_mul(*i), name="result_out")
        op_list += [res]
        instruction_list += ['dma_copy']

    # schedule
    schedule = tvm.create_schedule([res.op])
    return schedule, op_list, instruction_list


def compute_nopad(tensor_in, shape):
    """
    the compute produce, handling the scenes without padding
    """
    # preparing
    reduce_shape = (shape[0], 1, shape[2], shape[3], 1)
    op_list = []
    instruction_list = []

    # move data from gm to ub
    tensor_in_ub = \
        tvm.compute(shape, lambda *i: tensor_in(*i), name="tensor_in_ub")
    op_list += [tensor_in_ub]
    instruction_list += ['dma_copy']

    # reduce max
    i = tvm.reduce_axis((0, shape[1]), "c1_max")
    j = tvm.reduce_axis((0, shape[4]), "c0_max")
    res_max = tvm.compute(reduce_shape,
                          lambda n, c1, h, w, c0:
                          tvm.max(tensor_in_ub[n, i, h, w, j],
                                  axis=[i, j]), name="res_max")
    op_list += [res_max]
    instruction_list += ['vector_reduce_max']

    # sub
    minus = tvm.const(-1, 'float16')
    res_minus = tvm.compute(reduce_shape,
                            lambda *i:
                            minus * res_max(*i), name="res_minus")
    op_list += [res_minus]
    instruction_list += ['vector_muls']

    res_sub = tvm.compute(shape,
                          lambda n, c1, h, w, c0:
                          tensor_in_ub[n, c1, h, w, c0] + res_minus[n, 0, h, w, 0], name="res_sub")
    op_list += [res_sub]
    instruction_list += ['vector_adds']

    # exp
    res_exp = \
        tvm.compute(shape, lambda *i: tvm.exp(res_sub(*i)), name="res_exp")
    op_list += [res_exp]
    instruction_list += ['vector_exp']

    # reduce sum
    ii = tvm.reduce_axis((0, shape[1]), "c1_sum")
    jj = tvm.reduce_axis((0, shape[4]), "c0_sum")
    res_sum = tvm.compute(reduce_shape,
                          lambda n, c1, h, w, c0:
                          tvm.sum(res_exp[n, ii, h, w, jj], axis=[ii, jj]),
                          name="res_sum")
    op_list += [res_sum]
    instruction_list += ['vector_reduce_sum']

    # rec
    res_rec = tvm.compute(reduce_shape,
                          lambda *i: 1 / (res_sum(*i)), name="res_rec")
    op_list += [res_rec]
    instruction_list += ['vector_rec']

    # mul
    res_mul = tvm.compute(shape,
                          lambda n, c1, h, w, c0:
                          res_exp[n, c1, h, w, c0] * res_rec[n, 0, h, w, 0],
                          name="res_mul")
    op_list += [res_mul]
    instruction_list += ['vector_muls']

    # move data from ub to gm
    res = tvm.compute(shape, lambda *i: res_mul(*i), name="result_out")
    op_list += [res]
    instruction_list += ['dma_copy']

    # schedule
    schedule = tvm.create_schedule([res.op])
    return schedule, op_list, instruction_list


def compute_nz_nopad(tensor_in, shape):
    """
    the compute produce, handling the scenes without padding
    """
    # preparing
    reduce_shape = (1, shape[1], shape[2], 1)
    op_list = []
    instruction_list = []

    # move data from gm to ub
    tensor_in_ub = \
        tvm.compute(shape, lambda *i: tensor_in(*i), name="tensor_in_ub")
    op_list += [tensor_in_ub]
    instruction_list += ['dma_copy']

    # reduce max
    i = tvm.reduce_axis((0, shape[0]), "c1_max")
    j = tvm.reduce_axis((0, shape[3]), "c0_max")
    res_max = tvm.compute(reduce_shape,
                          lambda c1, n1, n0, c0:
                          tvm.max(tensor_in_ub[i, n1, n0, j],
                                  axis=[i, j]), name="res_max")
    op_list += [res_max]
    instruction_list += ['vector_reduce_max']

    # sub
    minus = tvm.const(-1, 'float16')
    res_minus = tvm.compute(reduce_shape,
                            lambda *i:
                            minus * res_max(*i), name="res_minus")
    op_list += [res_minus]
    instruction_list += ['vector_muls']

    res_sub = tvm.compute(shape,
                          lambda c1, n1, n0, c0: tensor_in_ub[c1, n1, n0, c0] + res_minus[0, n1, n0, 0], name="res_sub")
    op_list += [res_sub]
    instruction_list += ['vector_adds']

    # exp
    res_exp = \
        tvm.compute(shape, lambda *i: tvm.exp(res_sub(*i)), name="res_exp")
    op_list += [res_exp]
    instruction_list += ['vector_exp']

    # reduce sum
    ii = tvm.reduce_axis((0, shape[0]), "c1_sum")
    jj = tvm.reduce_axis((0, shape[3]), "c0_sum")
    res_sum = tvm.compute(reduce_shape,
                          lambda c1, n1, n0, c0:
                          tvm.sum(res_exp[ii, n1, n0, jj], axis=[ii, jj]),
                          name="res_sum")
    op_list += [res_sum]
    instruction_list += ['vector_reduce_sum']

    # rec
    res_rec = tvm.compute(reduce_shape,
                          lambda *i: 1 / (res_sum(*i)), name="res_rec")
    op_list += [res_rec]
    instruction_list += ['vector_rec']

    # mul
    res_mul = tvm.compute(shape,
                          lambda c1, n1, n0, c0:
                          res_exp[c1, n1, n0, c0] * res_rec[0, n1, n0, 0],
                          name="res_mul")
    op_list += [res_mul]
    instruction_list += ['vector_muls']

    # move data from ub to gm
    res = tvm.compute(shape, lambda *i: res_mul(*i), name="result_out")
    op_list += [res]
    instruction_list += ['dma_copy']

    # schedule
    schedule = tvm.create_schedule([res.op])
    return schedule, op_list, instruction_list


def compute_padding_fp32(tensor_in, shape, pad_param, impl_mode):
    """
    the compute produce, handling the scenes with padding
    """
    # preparing
    reduce_shape = (shape[0], 1, shape[2], shape[3], 1)
    pad_c1 = pad_param[0]
    pad_c0 = pad_param[1]
    op_list = []
    instruction_list = []
    # move data from gm to ub
    tensor_in_ub = tvm.compute(shape, lambda *i: tensor_in(*i), name="tensor_in_ub")
    op_list += [tensor_in_ub]
    instruction_list += ['dma_copy']

    if tensor_in.dtype == "float16":
        if shape[1] == pad_c1:
            # reduce max
            i = tvm.reduce_axis((0, shape[1]), "c1_max_pad")
            j = tvm.reduce_axis((0, shape[4] - pad_c0), "c0_max_pad")
            res_max = tvm.compute(reduce_shape,
                                  lambda n, c1, h, w, c0:
                                  tvm.max(tensor_in_ub[n, i, h, w, j], axis=[i, j]),
                                  name="res_max")
            op_list += [res_max]
            if pad_c0 != 15:
                instruction_list += ['vector_reduce_max']
            else:
                instruction_list += ['vector_auto']
        else:
            # reduce max
            i = tvm.reduce_axis((0, shape[1] - pad_c1), "c1_max")
            j = tvm.reduce_axis((0, shape[4]), "c0_max")
            gmax1 = tvm.compute(reduce_shape,
                                lambda n, c1, h, w, c0:
                                tvm.max(tensor_in_ub[n, i, h, w, j], axis=[i, j]),
                                name="gmax1")
            op_list += [gmax1]
            instruction_list += ['vector_reduce_max']

            i = tvm.reduce_axis((shape[1] - pad_c1, shape[1]), "c1_max_pad")
            j = tvm.reduce_axis((0, shape[4] - pad_c0), "c0_max_pad")
            gmax2 = tvm.compute(reduce_shape,
                                lambda n, c1, h, w, c0:
                                tvm.max(tensor_in_ub[n, i, h, w, j], axis=[i, j]),
                                name="gmax2")
            op_list += [gmax2]
            if pad_c0 != 15:
                instruction_list += ['vector_reduce_max']
            else:
                instruction_list += ['vector_auto']

            res_max = tvm.compute(reduce_shape,
                                  lambda *i: tvm.max(gmax1(*i), gmax2(*i)), name="res_max")
            op_list += [res_max]
            instruction_list += ['vector_max']
    else:
        # vcov fp32tofp16
        tensor_in_ub_fp16 = te.lang.cce.cast_to(tensor_in_ub, "float16")
        tensor_in_ub_fp16 = tvm.compute(shape,
                                        lambda *i: shape_util.cast(tensor_in_ub(*i), "float16"),
                                        name='res_vonv_fp16_tensor')
        op_list += [tensor_in_ub_fp16]
        instruction_list += ['vector_conv']

        if shape[1] == pad_c1:
            # reduce max
            i = tvm.reduce_axis((0, shape[1]), "c1_max_pad")
            j = tvm.reduce_axis((0, shape[4] - pad_c0), "c0_max_pad")
            res_max = tvm.compute(reduce_shape,
                                  lambda n, c1, h, w, c0:
                                  tvm.max(tensor_in_ub_fp16[n, i, h, w, j], axis=[i, j]),
                                  name="res_max")
            op_list += [res_max]
            instruction_list += ['vector_reduce_max']
        else:
            # reduce max
            i = tvm.reduce_axis((0, shape[1] - pad_c1), "c1_max")
            j = tvm.reduce_axis((0, shape[4]), "c0_max")
            gmax1 = tvm.compute(reduce_shape,
                                lambda n, c1, h, w, c0:
                                tvm.max(tensor_in_ub_fp16[n, i, h, w, j], axis=[i, j]),
                                name="gmax1")
            op_list += [gmax1]
            instruction_list += ['vector_reduce_max']

            i = tvm.reduce_axis((shape[1] - pad_c1, shape[1]), "c1_max_pad")
            j = tvm.reduce_axis((0, shape[4] - pad_c0), "c0_max_pad")
            gmax2 = tvm.compute(reduce_shape,
                                lambda n, c1, h, w, c0:
                                tvm.max(tensor_in_ub_fp16[n, i, h, w, j], axis=[i, j]),
                                name="gmax2")
            op_list += [gmax2]
            instruction_list += ['vector_reduce_max']

            res_max = tvm.compute(reduce_shape,
                                  lambda *i: tvm.max(gmax1(*i), gmax2(*i)),
                                  name="res_max")
            op_list += [res_max]
            instruction_list += ['vector_max']

    res_max_broadcast = te.lang.cce.broadcast(res_max, shape)
    op_list += [res_max_broadcast]
    instruction_list += ['vector_broadcast']

    # sub
    if tensor_in.dtype == "float32":
        res_sub = tvm.compute(shape,
                              lambda *i:
                              tensor_in_ub_fp16(*i) - res_max_broadcast(*i),
                              name="res_sub")
        op_list += [res_sub]
        instruction_list += ['vector_sub']
    else:
        res_sub = tvm.compute(shape,
                              lambda *i:
                              tensor_in_ub(*i) - res_max_broadcast(*i),
                              name="res_sub")
        op_list += [res_sub]
        instruction_list += ['vector_sub']

    if tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        # vcov fp16tofp32
        res_sub_fp32 = te.lang.cce.cast_to(res_sub, "float32")
        res_sub_fp32 = tvm.compute(shape,
                                   lambda *i: shape_util.cast(res_sub(*i), "float32"),
                                   name='res_vonv_exp')
        op_list += [res_sub_fp32]
        instruction_list += ['vector_conv']

        # exp
        res_exp_fp32 = tvm.compute(shape, lambda *i: tvm.exp(res_sub_fp32(*i)), name="res_exp")
        op_list += [res_exp_fp32]
        instruction_list += ['vector_exp']
    else:
        # exp
        res_exp = tvm.compute(shape, lambda *i: tvm.exp(res_sub(*i)), name="res_exp")
        op_list += [res_exp]
        instruction_list += ['vector_exp']

        # vcov fp16tofp32
        res_exp_fp32 = te.lang.cce.cast_to(res_exp, "float32")
        res_exp_fp32 = tvm.compute(shape,
                                   lambda *i: shape_util.cast(res_exp(*i), "float32"),
                                   name='res_vonv_exp')
        op_list += [res_exp_fp32]
        instruction_list += ['vector_conv']

    if shape[1] == pad_c1:
        if pad_c0 != 15:
            # reduce sum
            ii = tvm.reduce_axis((0, shape[1]), "c1_sum_pad")
            jj = tvm.reduce_axis((0, shape[4] - pad_c0), "c0_sum_pad")
            res_sum = tvm.compute(reduce_shape,
                                  lambda n, c1, h, w, c0:
                                  tvm.sum(res_exp_fp32[n, ii, h, w, jj], axis=[ii, jj]),
                                  name="res_sum")
            op_list += [res_sum]
            instruction_list += ['vector_reduce_sum']
        else:
            res_sum = tvm.compute(reduce_shape,
                                  lambda n, c1, h, w, c0:
                                  res_exp_fp32[n, shape[1] - 1, h, w, 0], name="res_sum")
            op_list += [res_sum]
            instruction_list += ['vector_auto']
    else:
        # reduce sum
        ii = tvm.reduce_axis((0, shape[1] - pad_c1), "c1_sum")
        jj = tvm.reduce_axis((0, shape[4]), "c0_sum")
        sum1 = tvm.compute(reduce_shape,
                           lambda n, c1, h, w, c0:
                           tvm.sum(res_exp_fp32[n, ii, h, w, jj], axis=[ii, jj]),
                           name="sum1")
        op_list += [sum1]
        instruction_list += ['vector_reduce_sum']

        if pad_c0 != 15:
            ii = tvm.reduce_axis((shape[1] - pad_c1, shape[1]), "c1_sum_pad")
            jj = tvm.reduce_axis((0, shape[4] - pad_c0), "c0_sum_pad")
            sum2 = tvm.compute(reduce_shape,
                               lambda n, c1, h, w, c0:
                               tvm.sum(res_exp_fp32[n, ii, h, w, jj],
                                       axis=[ii, jj]), name="sum2")
            op_list += [sum2]
            instruction_list += ['vector_reduce_sum']
        else:
            sum2 = tvm.compute(reduce_shape,
                               lambda n, c1, h, w, c0:
                               res_exp_fp32[n, shape[1] - 1, h, w, 0], name="sum2")
            op_list += [sum2]
            instruction_list += ['vector_auto']

        res_sum = tvm.compute(reduce_shape,
                              lambda *i: tvm.sum(sum1(*i), sum2(*i)), name="res_sum")
        op_list += [res_sum]
        instruction_list += ['vector_add']

    # judge the platform is mini or not
    if tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        res_mul_newton_broadcast = te.lang.cce.broadcast(res_sum, shape)
        op_list += [res_mul_newton_broadcast]
        instruction_list += ['vector_broadcast']

        res_mul = tvm.compute(shape,
                              lambda *i:
                              res_exp_fp32(*i) / res_mul_newton_broadcast(*i),
                              name="res_mul")
        op_list += [res_mul]
        instruction_list += ['vector_div']
    else:
        # rec
        res_rec = tvm.compute(reduce_shape, lambda *i: 1 / (res_sum(*i)), name="res_rec")
        op_list += [res_rec]
        instruction_list += ['vector_rec']

        if impl_mode == "high_performance":
            res_mul_newton_broadcast = te.lang.cce.broadcast(res_rec, shape)
            op_list += [res_mul_newton_broadcast]
            instruction_list += ['vector_broadcast']
        else:
            # loop 1
            # vmlu
            res_mul_newton = tvm.compute(reduce_shape,
                                         lambda *i: res_rec(*i) * res_sum(*i), name="res_mul_newton")
            op_list += [res_mul_newton]
            instruction_list += ['vector_mul']

            res_const2 = tvm.compute(reduce_shape,
                                     lambda *i: 2.0,
                                     name="res_const2")
            op_list += [res_const2]
            instruction_list += ['vector_auto']

            # vsub
            res_sub_newton = tvm.compute(reduce_shape,
                                         lambda *i: res_const2(*i) - res_mul_newton(*i),
                                         name="res_sub_newton")
            op_list += [res_sub_newton]
            instruction_list += ['vector_sub']

            # vmlu
            res_mul_newton = tvm.compute(reduce_shape,
                                         lambda *i: res_sub_newton(*i) * res_rec(*i),
                                         name="res_mul_newton")
            op_list += [res_mul_newton]
            instruction_list += ['vector_mul']

            res_mul_newton_broadcast = te.lang.cce.broadcast(res_mul_newton, shape)
            op_list += [res_mul_newton_broadcast]
            instruction_list += ['vector_broadcast']
        # mul
        res_mul = tvm.compute(shape,
                              lambda *i:
                              res_exp_fp32(*i) * res_mul_newton_broadcast(*i),
                              name="res_mul")
        op_list += [res_mul]
        instruction_list += ['vector_mul']

    if tensor_in.dtype == "float16":
        res_mul_fp16 = te.lang.cce.cast_to(res_mul, "float16")
        res_mul_fp16 = tvm.compute(shape,
                                   lambda *i: shape_util.cast(res_mul(*i), "float16"),
                                   name='res_vonv_fp16')
        op_list += [res_mul_fp16]
        instruction_list += ['vector_conv']

        # move data from ub to gm
        res = tvm.compute(shape, lambda *i: res_mul_fp16(*i), name="result_out")
        op_list += [res]
        instruction_list += ['dma_copy']
    else:
        # move data from ub to gm
        res = tvm.compute(shape, lambda *i: res_mul(*i), name="result_out")
        op_list += [res]
        instruction_list += ['dma_copy']

    # schedule
    schedule = tvm.create_schedule([res.op])
    return schedule, op_list, instruction_list


def compute_nz_padding_fp32(tensor_in, shape, pad_param):
    """
    the compute produce, handling the scenes with padding
    """
    # preparing
    reduce_shape = (1, shape[1], shape[2], 1)
    pad_c1 = pad_param[0]
    pad_c0 = pad_param[1]
    op_list = []
    instruction_list = []
    # move data from gm to ub
    tensor_in_ub = tvm.compute(shape, lambda *i: tensor_in(*i), name="tensor_in_ub")
    op_list += [tensor_in_ub]
    instruction_list += ['dma_copy']

    if tensor_in.dtype == "float16":
        if shape[0] == pad_c1:
            # reduce max
            i = tvm.reduce_axis((0, shape[0]), "c1_max_pad")
            j = tvm.reduce_axis((0, shape[3] - pad_c0), "c0_max_pad")
            res_max = tvm.compute(reduce_shape,
                                  lambda c1, n1, n0, c0:
                                  tvm.max(tensor_in_ub[i, n1, n0, j], axis=[i, j]),
                                  name="res_max")
            op_list += [res_max]
            if pad_c0 != 15:
                instruction_list += ['vector_reduce_max']
            else:
                instruction_list += ['vector_auto']
        else:
            # reduce max
            i = tvm.reduce_axis((0, shape[0] - pad_c1), "c1_max")
            j = tvm.reduce_axis((0, shape[3]), "c0_max")
            gmax1 = tvm.compute(reduce_shape,
                                lambda c1, n1, n0, c0:
                                tvm.max(tensor_in_ub[i, n1, n0, j], axis=[i, j]),
                                name="gmax1")
            op_list += [gmax1]
            instruction_list += ['vector_reduce_max']

            i = tvm.reduce_axis((shape[0] - pad_c1, shape[0]), "c1_max_pad")
            j = tvm.reduce_axis((0, shape[3] - pad_c0), "c0_max_pad")
            gmax2 = tvm.compute(reduce_shape,
                                lambda c1, n1, n0, c0:
                                tvm.max(tensor_in_ub[i, n1, n0, j], axis=[i, j]),
                                name="gmax2")
            op_list += [gmax2]
            if pad_c0 != 15:
                instruction_list += ['vector_reduce_max']
            else:
                instruction_list += ['vector_auto']

            res_max = tvm.compute(reduce_shape,
                                  lambda *i: tvm.max(gmax1(*i), gmax2(*i)), name="res_max")
            op_list += [res_max]
            instruction_list += ['vector_max']
    else:
        # vcov fp32tofp16
        tensor_in_ub_fp16 = te.lang.cce.cast_to(tensor_in_ub, "float16")
        tensor_in_ub_fp16 = tvm.compute(shape,
                                        lambda *i: shape_util.cast(tensor_in_ub(*i), "float16"),
                                        name='res_vonv_fp16_tensor')
        op_list += [tensor_in_ub_fp16]
        instruction_list += ['vector_conv']

        if shape[1] == pad_c1:
            # reduce max
            i = tvm.reduce_axis((0, shape[0]), "c1_max_pad")
            j = tvm.reduce_axis((0, shape[3] - pad_c0), "c0_max_pad")
            res_max = tvm.compute(reduce_shape,
                                  lambda c1, n1, n0, c0:
                                  tvm.max(tensor_in_ub_fp16[i, n1, n0, j], axis=[i, j]),
                                  name="res_max")
            op_list += [res_max]
            instruction_list += ['vector_reduce_max']
        else:
            # reduce max
            i = tvm.reduce_axis((0, shape[0] - pad_c1), "c1_max")
            j = tvm.reduce_axis((0, shape[3]), "c0_max")
            gmax1 = tvm.compute(reduce_shape,
                                lambda c1, n1, n0, c0:
                                tvm.max(tensor_in_ub_fp16[i, n1, n0, j], axis=[i, j]),
                                name="gmax1")
            op_list += [gmax1]
            instruction_list += ['vector_reduce_max']

            i = tvm.reduce_axis((shape[0] - pad_c1, shape[0]), "c1_max_pad")
            j = tvm.reduce_axis((0, shape[3] - pad_c0), "c0_max_pad")
            gmax2 = tvm.compute(reduce_shape,
                                lambda c1, n1, n0, c0:
                                tvm.max(tensor_in_ub_fp16[i, n1, n0, j], axis=[i, j]),
                                name="gmax2")
            op_list += [gmax2]
            instruction_list += ['vector_reduce_max']

            res_max = tvm.compute(reduce_shape,
                                  lambda *i: tvm.max(gmax1(*i), gmax2(*i)),
                                  name="res_max")
            op_list += [res_max]
            instruction_list += ['vector_max']

    # sub
    minus = tvm.const(-1, 'float16')
    res_minus = tvm.compute(reduce_shape, lambda *i: minus * res_max(*i), name="res_minus")
    op_list += [res_minus]
    instruction_list += ['vector_muls']

    if tensor_in.dtype == "float32":
        res_sub = tvm.compute(shape,
                              lambda c1, n1, n0, c0:
                              tensor_in_ub_fp16[c1, n1, n0, c0] + res_minus[0, n1, n0, 0],
                              name="res_sub")
        op_list += [res_sub]
        instruction_list += ['vector_adds']
    else:
        res_sub = tvm.compute(shape,
                              lambda c1, n1, n0, c0:
                              tensor_in_ub[c1, n1, n0, c0] + res_minus[0, n1, n0, 0],
                              name="res_sub")
        op_list += [res_sub]
        instruction_list += ['vector_adds']

    # exp
    res_exp = tvm.compute(shape, lambda *i: tvm.exp(res_sub(*i)), name="res_exp")
    op_list += [res_exp]
    instruction_list += ['vector_exp']

    # vcov fp16tofp32
    res_exp_fp32 = te.lang.cce.cast_to(res_exp, "float32")
    res_exp_fp32 = tvm.compute(shape,
                               lambda *i: shape_util.cast(res_exp(*i), "float32"),
                               name='res_vonv_exp')
    op_list += [res_exp_fp32]
    instruction_list += ['vector_conv']

    if shape[0] == pad_c1:
        if pad_c0 != 15:
            # reduce sum
            ii = tvm.reduce_axis((0, shape[0]), "c1_sum_pad")
            jj = tvm.reduce_axis((0, shape[3] - pad_c0), "c0_sum_pad")
            res_sum = tvm.compute(reduce_shape,
                                  lambda c1, n1, n0, c0:
                                  tvm.sum(res_exp_fp32[ii, n1, n0, jj], axis=[ii, jj]),
                                  name="res_sum")
            op_list += [res_sum]
            instruction_list += ['vector_reduce_sum']
        else:
            res_sum = tvm.compute(reduce_shape,
                                  lambda c1, n1, n0, c0:
                                  res_exp_fp32[shape[0] - 1, n1, n0, 0], name="res_sum")
            op_list += [res_sum]
            instruction_list += ['vector_auto']
    else:
        # reduce sum
        ii = tvm.reduce_axis((0, shape[0] - pad_c1), "c1_sum")
        jj = tvm.reduce_axis((0, shape[3]), "c0_sum")
        sum1 = tvm.compute(reduce_shape,
                           lambda c1, n1, n0, c0:
                           tvm.sum(res_exp_fp32[ii, n1, n0, jj], axis=[ii, jj]),
                           name="sum1")
        op_list += [sum1]
        instruction_list += ['vector_reduce_sum']

        if pad_c0 != 15:
            ii = tvm.reduce_axis((shape[0] - pad_c1, shape[0]), "c1_sum_pad")
            jj = tvm.reduce_axis((0, shape[3] - pad_c0), "c0_sum_pad")
            sum2 = tvm.compute(reduce_shape,
                               lambda c1, n1, n0, c0:
                               tvm.sum(res_exp_fp32[ii, n1, n0, jj],
                                       axis=[ii, jj]), name="sum2")
            op_list += [sum2]
            instruction_list += ['vector_reduce_sum']
        else:
            sum2 = tvm.compute(reduce_shape,
                               lambda c1, n1, n0, c0:
                               res_exp_fp32[shape[0] - 1, n1, n0, 0], name="sum2")
            op_list += [sum2]
            instruction_list += ['vector_auto']

        res_sum = tvm.compute(reduce_shape,
                              lambda *i: tvm.sum(sum1(*i), sum2(*i)), name="res_sum")
        op_list += [res_sum]
        instruction_list += ['vector_add']

    # rec
    res_rec = tvm.compute(reduce_shape, lambda *i: 1/(res_sum(*i)), name="res_rec")
    op_list += [res_rec]
    instruction_list += ['vector_rec']

    # loop 1
    # vmlu
    res_mul_newton = tvm.compute(reduce_shape,
                                 lambda *i: res_rec(*i) * res_sum(*i), name="res_mul_newton")
    op_list += [res_mul_newton]
    instruction_list += ['vector_mul']

    # vmuls
    const1 = tvm.const(-1, 'float32')
    res_muls_newton = tvm.compute(reduce_shape,
                                  lambda *i: const1 * res_mul_newton(*i),
                                  name="res_muls_newton")
    op_list += [res_muls_newton]
    instruction_list += ['vector_muls']

    # vadds
    const2 = tvm.const(2, 'float32')
    res_adds_newton = tvm.compute(reduce_shape,
                                  lambda *i: const2 + res_muls_newton(*i),
                                  name="res_adds_newton")
    op_list += [res_adds_newton]
    instruction_list += ['vector_adds']

    # vmlu
    res_mul_newton = tvm.compute(reduce_shape,
                                 lambda *i: res_adds_newton(*i) * res_rec(*i),
                                 name="res_mul_newton")
    op_list += [res_mul_newton]
    instruction_list += ['vector_mul']

    # mul
    res_mul = tvm.compute(shape,
                          lambda c1, n1, n0, c0:
                          res_exp_fp32[c1, n1, n0, c0] * res_mul_newton[0, n1, n0, 0],
                          name="res_mul")
    op_list += [res_mul]
    instruction_list += ['vector_muls']

    if tensor_in.dtype == "float16":
        res_mul_fp16 = te.lang.cce.cast_to(res_mul, "float16")
        res_mul_fp16 = tvm.compute(shape,
                                   lambda *i: shape_util.cast(res_mul(*i), "float16"),
                                   name='res_vonv_fp16')
        op_list += [res_mul_fp16]
        instruction_list += ['vector_conv']

        # move data from ub to gm
        res = tvm.compute(shape, lambda *i: res_mul_fp16(*i), name="result_out")
        op_list += [res]
        instruction_list += ['dma_copy']
    else:
        # move data from ub to gm
        res = tvm.compute(shape, lambda *i: res_mul(*i), name="result_out")
        op_list += [res]
        instruction_list += ['dma_copy']

    # schedule
    schedule = tvm.create_schedule([res.op])
    return schedule, op_list, instruction_list


def compute_padding(tensor_in, shape, pad_param):
    """
    the compute produce, handling the scenes with padding
    """
    # preparing
    reduce_shape = (shape[0], 1, shape[2], shape[3], 1)
    pad_c1 = pad_param[0]
    pad_c0 = pad_param[1]
    op_list = []
    instruction_list = []

    # move data from gm to ub
    tensor_in_ub = tvm.compute(shape,
                               lambda *i: tensor_in(*i), name="tensor_in_ub")
    op_list += [tensor_in_ub]
    instruction_list += ['dma_copy']

    if shape[1] == pad_c1:
        # reduce max
        i = tvm.reduce_axis((0, shape[1]), "c1_max_pad")
        j = tvm.reduce_axis((0, shape[4] - pad_c0), "c0_max_pad")
        res_max = tvm.compute(reduce_shape,
                              lambda n, c1, h, w, c0:
                              tvm.max(tensor_in_ub[n, i, h, w, j],
                                      axis=[i, j]), name="res_max")
        op_list += [res_max]
        if pad_c0 != 15:
            instruction_list += ['vector_reduce_max']
        else:
            instruction_list += ['vector_auto']
    else:
        # reduce max
        i = tvm.reduce_axis((0, shape[1] - pad_c1), "c1_max")
        j = tvm.reduce_axis((0, shape[4]), "c0_max")
        gmax1 = tvm.compute(reduce_shape,
                            lambda n, c1, h, w, c0:
                            tvm.max(tensor_in_ub[n, i, h, w, j],
                                    axis=[i, j]), name="gmax1")
        op_list += [gmax1]
        instruction_list += ['vector_reduce_max']

        i = tvm.reduce_axis((shape[1] - pad_c1, shape[1]), "c1_max_pad")
        j = tvm.reduce_axis((0, shape[4] - pad_c0), "c0_max_pad")
        gmax2 = tvm.compute(reduce_shape,
                            lambda n, c1, h, w, c0:
                            tvm.max(tensor_in_ub[n, i, h, w, j],
                                    axis=[i, j]), name="gmax2")
        op_list += [gmax2]
        if pad_c0 != 15:
            instruction_list += ['vector_reduce_max']
        else:
            instruction_list += ['vector_auto']

        res_max = tvm.compute(reduce_shape,
                              lambda *i: tvm.max(gmax1(*i), gmax2(*i)),
                              name="res_max")
        op_list += [res_max]
        instruction_list += ['vector_max']

    # sub
    minus = tvm.const(-1, 'float16')
    res_minus = tvm.compute(reduce_shape,
                            lambda *i: minus * res_max(*i),
                            name="res_minus")
    op_list += [res_minus]
    instruction_list += ['vector_muls']

    res_sub = tvm.compute(shape,
                          lambda n, c1, h, w, c0:
                          tensor_in_ub[n, c1, h, w, c0] +
                          res_minus[n, 0, h, w, 0], name="res_sub")
    op_list += [res_sub]
    instruction_list += ['vector_adds']

    # exp
    res_exp = tvm.compute(shape, lambda *i: tvm.exp(res_sub(*i)), name="res_exp")
    op_list += [res_exp]
    instruction_list += ['vector_exp']

    if shape[1] == pad_c1:
        if pad_c0 != 15:
            # reduce sum
            ii = tvm.reduce_axis((0, shape[1]), "c1_sum_pad")
            jj = tvm.reduce_axis((0, shape[4] - pad_c0), "c0_sum_pad")
            res_sum = tvm.compute(reduce_shape,
                                  lambda n, c1, h, w, c0:
                                  tvm.sum(res_exp[n, ii, h, w, jj],
                                          axis=[ii, jj]), name="res_sum")
            op_list += [res_sum]
            instruction_list += ['vector_reduce_sum']
        else:
            res_sum = tvm.compute(reduce_shape,
                                  lambda n, c1, h, w, c0:
                                  res_exp[n, shape[1] - 1, h, w, 0], name="res_sum")
            op_list += [res_sum]
            instruction_list += ['vector_auto']
    else:
        # reduce sum
        ii = tvm.reduce_axis((0, shape[1] - pad_c1), "c1_sum")
        jj = tvm.reduce_axis((0, shape[4]), "c0_sum")
        sum1 = tvm.compute(reduce_shape,
                           lambda n, c1, h, w, c0:
                           tvm.sum(res_exp[n, ii, h, w, jj],
                                   axis=[ii, jj]), name="sum1")
        op_list += [sum1]
        instruction_list += ['vector_reduce_sum']

        if pad_c0 != 15:
            ii = tvm.reduce_axis((shape[1] - pad_c1, shape[1]), "c1_sum_pad")
            jj = tvm.reduce_axis((0, shape[4] - pad_c0), "c0_sum_pad")
            sum2 = tvm.compute(reduce_shape,
                               lambda n, c1, h, w, c0:
                               tvm.sum(res_exp[n, ii, h, w, jj],
                                       axis=[ii, jj]), name="sum2")
            op_list += [sum2]
            instruction_list += ['vector_reduce_sum']
        else:
            sum2 = tvm.compute(reduce_shape,
                               lambda n, c1, h, w, c0:
                               res_exp[n, shape[1] - 1, h, w, 0], name="sum2")
            op_list += [sum2]
            instruction_list += ['vector_auto']

        res_sum = tvm.compute(reduce_shape,
                              lambda *i: tvm.sum(sum1(*i), sum2(*i)),
                              name="res_sum")
        op_list += [res_sum]
        instruction_list += ['vector_add']

    # rec
    res_rec = tvm.compute(reduce_shape,
                          lambda *i: 1 / (res_sum(*i)), name="res_rec")
    op_list += [res_rec]
    instruction_list += ['vector_rec']

    # mul
    res_mul = tvm.compute(shape,
                          lambda n, c1, h, w, c0:
                          res_exp[n, c1, h, w, c0] * res_rec[n, 0, h, w, 0],
                          name="res_mul")
    op_list += [res_mul]
    instruction_list += ['vector_muls']

    # move data from ub to gm
    res = tvm.compute(shape, lambda *i: res_mul(*i), name="result_out")
    op_list += [res]
    instruction_list += ['dma_copy']

    schedule = tvm.create_schedule([res.op])
    return schedule, op_list, instruction_list


def compute_nz_padding(tensor_in, shape, pad_param):
    """
    the compute produce, handling the scenes with padding
    """
    # preparing
    reduce_shape = (1, shape[1], shape[2], 1)
    pad_c1 = pad_param[0]
    pad_c0 = pad_param[1]
    op_list = []
    instruction_list = []

    # move data from gm to ub
    tensor_in_ub = tvm.compute(shape,
                               lambda *i: tensor_in(*i), name="tensor_in_ub")
    op_list += [tensor_in_ub]
    instruction_list += ['dma_copy']

    if shape[0] == pad_c1:
        # reduce max
        i = tvm.reduce_axis((0, shape[0]), "c1_max_pad")
        j = tvm.reduce_axis((0, shape[3] - pad_c0), "c0_max_pad")
        res_max = tvm.compute(reduce_shape,
                              lambda c1, n1, n0, c0:
                              tvm.max(tensor_in_ub[i, n1, n0, j],
                                      axis=[i, j]), name="res_max")
        op_list += [res_max]
        if pad_c0 != 15:
            instruction_list += ['vector_reduce_max']
        else:
            instruction_list += ['vector_auto']
    else:
        # reduce max
        i = tvm.reduce_axis((0, shape[0] - pad_c1), "c1_max")
        j = tvm.reduce_axis((0, shape[3]), "c0_max")
        gmax1 = tvm.compute(reduce_shape,
                            lambda c1, n1, n0, c0:
                            tvm.max(tensor_in_ub[i, n1, n0, j],
                                    axis=[i, j]), name="gmax1")
        op_list += [gmax1]
        instruction_list += ['vector_reduce_max']

        i = tvm.reduce_axis((shape[0] - pad_c1, shape[0]), "c1_max_pad")
        j = tvm.reduce_axis((0, shape[3] - pad_c0), "c0_max_pad")
        gmax2 = tvm.compute(reduce_shape,
                            lambda c1, n1, n0, c0:
                            tvm.max(tensor_in_ub[i, n1, n0, j],
                                    axis=[i, j]), name="gmax2")
        op_list += [gmax2]
        if pad_c0 != 15:
            instruction_list += ['vector_reduce_max']
        else:
            instruction_list += ['vector_auto']

        res_max = tvm.compute(reduce_shape,
                              lambda *i: tvm.max(gmax1(*i), gmax2(*i)),
                              name="res_max")
        op_list += [res_max]
        instruction_list += ['vector_max']

    # sub
    minus = tvm.const(-1, 'float16')
    res_minus = tvm.compute(reduce_shape,
                            lambda *i: minus * res_max(*i),
                            name="res_minus")
    op_list += [res_minus]
    instruction_list += ['vector_muls']

    res_sub = tvm.compute(shape,
                          lambda c1, n1, n0, c0: tensor_in_ub[c1, n1, n0, c0] + res_minus[0, n1, n0, 0], name="res_sub")
    op_list += [res_sub]
    instruction_list += ['vector_adds']

    # exp
    res_exp = tvm.compute(shape,
                          lambda *i: tvm.exp(res_sub(*i)), name="res_exp")
    op_list += [res_exp]
    instruction_list += ['vector_exp']

    if shape[0] == pad_c1:
        if pad_c0 != 15:
            # reduce sum
            ii = tvm.reduce_axis((0, shape[0]), "c1_sum_pad")
            jj = tvm.reduce_axis((0, shape[3] - pad_c0), "c0_sum_pad")
            res_sum = tvm.compute(reduce_shape,
                                  lambda c1, n1, n0, c0:
                                  tvm.sum(res_exp[ii, n1, n0, jj],
                                          axis=[ii, jj]), name="res_sum")
            op_list += [res_sum]
            instruction_list += ['vector_reduce_sum']
        else:
            res_sum = tvm.compute(reduce_shape,
                                  lambda c1, n1, n0, c0:
                                  res_exp[shape[0] - 1, n1, n0, 0], name="res_sum")
            op_list += [res_sum]
            instruction_list += ['vector_auto']
    else:
        # reduce sum
        ii = tvm.reduce_axis((0, shape[0] - pad_c1), "c1_sum")
        jj = tvm.reduce_axis((0, shape[3]), "c0_sum")
        sum1 = tvm.compute(reduce_shape,
                           lambda c1, n1, n0, c0:
                           tvm.sum(res_exp[ii, n1, n0, jj],
                                   axis=[ii, jj]), name="sum1")
        op_list += [sum1]
        instruction_list += ['vector_reduce_sum']

        if pad_c0 != 15:
            ii = tvm.reduce_axis((shape[0] - pad_c1, shape[0]), "c1_sum_pad")
            jj = tvm.reduce_axis((0, shape[3] - pad_c0), "c0_sum_pad")
            sum2 = tvm.compute(reduce_shape,
                               lambda c1, n1, n0, c0:
                               tvm.sum(res_exp[ii, n1, n0, jj],
                                       axis=[ii, jj]), name="sum2")
            op_list += [sum2]
            instruction_list += ['vector_reduce_sum']
        else:
            sum2 = tvm.compute(reduce_shape,
                               lambda c1, n1, n0, c0:
                               res_exp[shape[0] - 1, n1, n0, 0], name="sum2")
            op_list += [sum2]
            instruction_list += ['vector_auto']

        res_sum = tvm.compute(reduce_shape,
                              lambda *i: tvm.sum(sum1(*i), sum2(*i)),
                              name="res_sum")
        op_list += [res_sum]
        instruction_list += ['vector_add']

    # rec
    res_rec = tvm.compute(reduce_shape,
                          lambda *i: 1 / (res_sum(*i)), name="res_rec")
    op_list += [res_rec]
    instruction_list += ['vector_rec']

    # mul
    res_mul = tvm.compute(shape,
                          lambda c1, n1, n0, c0:
                          res_exp[c1, n1, n0, c0] * res_rec[0, n1, n0, 0],
                          name="res_mul")
    op_list += [res_mul]
    instruction_list += ['vector_muls']

    # move data from ub to gm
    res = tvm.compute(shape, lambda *i: res_mul(*i), name="result_out")
    op_list += [res]
    instruction_list += ['dma_copy']

    schedule = tvm.create_schedule([res.op])
    return schedule, op_list, instruction_list


def softmax_channel_calculate(shape, dtype, pad_flag, pad_param, kernel_name, impl_mode):
    """
    calculating data's softmax, produces an output tensor with shape
    the result equals the sum of the x power of e over the sum of
    the x power of e

    Parameters
    ----------
    shape : list
        shape of input tensor
    dtype: str
        the data dtype, support float16 and float32
    pad_flag : bool
        the flag using for indicating if there is padding
    pad_param : list
        padding info
    kernel_name : str
        cce kernel name, default value is "softmax_cce"
    impl_mode : str
        high_precision or high_performance for inference, default value is "high_performance".
        no need to add into ops_info file.

    Returns
    -------
    None
    """

    # compute & instructions
    tensor_in = tvm.placeholder(shape, name='tensor_in', dtype=dtype)

    use_fp32 = check_isusefp32(shape, dtype)
    if not pad_flag:
        if use_fp32:
            sch, op_list, instruction_list = compute_nopad_fp32(tensor_in, shape)
        else:
            sch, op_list, instruction_list = compute_nopad(tensor_in, shape)
    else:
        if use_fp32:
            sch, op_list, instruction_list = compute_padding_fp32(tensor_in, shape, pad_param, impl_mode)
        else:
            sch, op_list, instruction_list = compute_padding(tensor_in, shape, pad_param)
    res = op_list[-1]

    # schedule
    # storage align
    align_factor = shape[4]
    align(sch, op_list, pad_param, align_factor, 0)

    [npart_n, npart_h, npart_w, split_axis_0, split_size] = multicore_factor_calculate(shape)

    xno, xni = sch[res].split(res.op.axis[0], nparts=npart_n)
    xho, xhi = sch[res].split(res.op.axis[2], nparts=npart_h)
    xwo, xwi = sch[res].split(res.op.axis[3], nparts=npart_w)

    sch[res].reorder(xno, xho, xwo, xni, xhi, xwi, res.op.axis[1], res.op.axis[4])
    block_axis = sch[res].fuse(xno, xho, xwo)
    sch[res].bind(block_axis, tvm.thread_axis("blockIdx.x"))

    # tiling strategy
    split_flag, split_axis, split_factor = \
        tiling_factor_calculate(shape, split_axis_0, split_size, use_fp32)

    # need splitting on  h or w
    if split_flag:
        if split_axis == 0:
            xo, xi = sch[res].split(xni, factor=split_factor)
        elif split_axis == 2:
            xo, xi = sch[res].split(xhi, factor=split_factor)
        elif split_axis == 3:
            xo, xi = sch[res].split(xwi, factor=split_factor)

        # schedule optimize
        ops_integrate(sch, op_list, xo)

        # buffer mapping
        buffer_mapping(sch, op_list[:-1])

        # double buffer
        double_buf(sch, op_list)

        # instructions replace
        axis_list = emit_axis_collect(op_list, pad_param, instruction_list, xi)
        axis_reorder(sch, op_list, instruction_list)
        instructions_replace(sch, op_list, axis_list, instruction_list)

    # no split
    else:

        # schedule optimize
        if split_axis_0 == 1:
            ops_integrate(sch, op_list, block_axis)
        elif split_axis_0 == 2:
            ops_integrate(sch, op_list, xni)
        elif split_axis_0 in (3, 4):
            ops_integrate(sch, op_list, xhi)

        # buffer mapping
        buffer_mapping(sch, op_list[:-1])

        # double buffer
        double_buf(sch, op_list)

        # instructions replace
        if split_axis_0 == 1:
            axis_list = emit_axis_collect(op_list, pad_param, instruction_list, xni)
        elif split_axis_0 == 2:
            axis_list = emit_axis_collect(op_list, pad_param, instruction_list, xhi)
        elif split_axis_0 in (3, 4):
            axis_list = emit_axis_collect(op_list, pad_param, instruction_list, xwi)

        axis_reorder(sch, op_list, instruction_list)
        instructions_replace(sch, op_list, axis_list, instruction_list)

    with build_config():
        tvm.build(sch, [tensor_in, res], "cce", name=kernel_name)


def softmax_nz_channel_calculate(shape, dtype, pad_flag, pad_param, kernel_name):
    """
    calculating data's softmax nz format, produces an output tensor with shape
    the result equals the sum of the x power of e over the sum of
    the x power of e

    Parameters
    ----------
    shape : list
        shape of input tensor
    dtype: str
        the data dtype, support float16 and float32
    pad_flag : bool
        the flag using for indicating if there is padding
    pad_param : list
        padding info
    kernel_name : str
        cce kernel name, default value is "softmax_cce"
    Returns
    -------
    None
    """

    # compute & instructions
    tensor_in = tvm.placeholder(shape, name='tensor_in', dtype=dtype)
    use_fp32 = check_nz_isusefp32(shape, dtype)
    if not pad_flag:
        if use_fp32:
            sch, op_list, instruction_list = compute_nz_nopad_fp32(tensor_in, shape)
        else:
            sch, op_list, instruction_list = compute_nz_nopad(tensor_in, shape)
    else:
        if use_fp32:
            sch, op_list, instruction_list = compute_nz_padding_fp32(tensor_in, shape, pad_param)
        else:
            sch, op_list, instruction_list = compute_nz_padding(tensor_in, shape, pad_param)
    res = op_list[-1]

    # schedule
    # storage align
    align_factor = shape[3]
    align_nz(sch, op_list, pad_param, align_factor, 0)

    [npart_n1, npart_n0, split_axis_0, split_size] = multicore_factor_calculate_nz(shape)

    xn1o, xn1i = sch[res].split(res.op.axis[1], nparts=npart_n1)
    xn0o, xn0i = sch[res].split(res.op.axis[2], nparts=npart_n0)

    sch[res].reorder(xn1o, xn0o, xn1i, xn0i, res.op.axis[0], res.op.axis[3])
    block_axis = sch[res].fuse(xn1o, xn0o)
    sch[res].bind(block_axis, tvm.thread_axis("blockIdx.x"))

    # tiling strategy
    split_flag, split_axis, split_factor = \
        tiling_factor_calculate_nz(shape, split_axis_0, split_size, use_fp32)

    # need splitting on  h or w
    if split_flag:
        if split_axis == 1:
            xo, xi = sch[res].split(xn1i, factor=split_factor)
        elif split_axis == 2:
            xo, xi = sch[res].split(xn0i, factor=split_factor)

        # schedule optimize
        ops_integrate(sch, op_list, xo)

        # buffer mapping
        buffer_mapping(sch, op_list[:-1])

        # double buffer
        double_buf(sch, op_list)

        # instructions replace
        axis_list = emit_nz_axis_collect(op_list, pad_param, instruction_list, xi)
        axis_reorder_nz(sch, op_list, instruction_list)
        instructions_replace(sch, op_list, axis_list, instruction_list)

    # no split
    else:

        # schedule optimize
        if split_axis_0 == 1:
            ops_integrate(sch, op_list, block_axis)
        elif split_axis_0 == 2:
            ops_integrate(sch, op_list, xn1i)

        # buffer mapping
        buffer_mapping(sch, op_list[:-1])

        # double buffer
        double_buf(sch, op_list)

        # instructions replace
        if split_axis_0 == 1:
            axis_list = emit_nz_axis_collect(op_list, pad_param, instruction_list, xn1i)
        elif split_axis_0 in (2, 3):
            axis_list = emit_nz_axis_collect(op_list, pad_param, instruction_list, xn0i)

        axis_reorder_nz(sch, op_list, instruction_list)
        instructions_replace(sch, op_list, axis_list, instruction_list)

    with build_config():
        tvm.build(sch, [tensor_in, res], "cce", name=kernel_name)


def softmax_param_check(in_tensor, output_tensor, axis, kernel_name):
    """
    checking the parameter of softmax, and calculating the intermediate
    data using for compute and schedule

    Parameters
    ----------
    in_tensor : dict
        shape and dtype of input tensor, shape and dtype of original tensor,
        input shape only support NC1HWC0, original shape support NCHW and NHWC,
        dtype  support float16 and float,
    output_tensor: dict
        shape and dtype of output tensor, should be same as input
    axis : listint
       the data's axis using for softmax,
    kernel_name : str
        cce kernel name, default value is "softmax_cce"

    Returns
    -------
    shape and stype of input tensor
    parameters of padding on dimension C0
    """

    # param calculate
    in_shape = in_tensor['shape']
    in_dtype = in_tensor['dtype']
    ori_shape = in_tensor['ori_shape']
    input_format = in_tensor['format']
    ori_format = in_tensor['ori_format']
    out_dtype = output_tensor['dtype']

    # shape check, check length,min,max,size
    if input_format in ("NC1HWC0",):
        para_check.check_shape(in_shape, min_rank=5, max_rank=5, param_name="x")
        if len(ori_shape) == 3:
            ori_shape = list(ori_shape)
            ori_shape.insert(0, 1)
        para_check.check_shape(ori_shape, min_rank=4, max_rank=4, param_name="x")

    # type check
    in_dtype = in_dtype.lower()
    para_check.check_dtype(in_dtype, ("float16", "float32"), param_name="x")
    out_dtype = out_dtype.lower()
    para_check.check_dtype(out_dtype, ("float16", "float32"), param_name="y")

    # shape check
    in_shape_c1 = in_shape[1] if input_format in ("NC1HWC0",) else in_shape[2]
    in_shape_c0 = in_shape[4] if input_format in ("NC1HWC0",) else in_shape[5]
    if in_dtype == "float16" and in_shape_c1 * in_shape_c0 * 2 > UB_SIZE_LIMIT:
        error_manager_vector.raise_err_input_param_range_invalid("softmax_v2", "C", 0, UB_SIZE_LIMIT,
                                                                 in_shape_c1 * in_shape_c0 * 2)
    if in_dtype == "float32" and in_shape_c1 * in_shape_c0 * 4 > UB_SIZE_LIMIT:
        error_manager_vector.raise_err_input_param_range_invalid("softmax_v2", "C", 0, UB_SIZE_LIMIT,
                                                                 in_shape_c1 * in_shape_c0 * 4)
    # calc padding parameters
    padding = 0
    if "C" in ori_format:
        padding = in_shape_c1 * in_shape_c0 - ori_shape[ori_format.index("C")]
    else:
        para_check.check_format(in_tensor.get("ori_format"), ("NCHW", "NHWC", "NDHWC"), param_name="x")

    pad_param = []
    pad_flag = True
    if padding < 0:
        error_manager_vector.raise_err_specific_reson("softmax_v2",
                                                      "the shapes of input tensor and original tensor don't match")
    elif padding == 0:
        pad_flag = False
        pad_c1 = 0
        pad_c0 = 0
        pad_param = [pad_c1, pad_c0]
    else:
        pad_flag = True
        pad_c1 = (padding + 15) // 16
        pad_c0 = padding % 16
        pad_param = [pad_c1, pad_c0]

    if input_format in ("NDC1HWC0",):
        in_shape = list(in_shape)
        in_shape = [in_shape[0] * in_shape[1]] + in_shape[2:]
    return_list = [in_shape, in_dtype, pad_flag, pad_param]

    return return_list


def softmax_nz_param_check(in_tensor, output_tensor, axis, kernel_name):
    """
    checking the parameter of softmax nz format, and calculating the
    intermediate data using for compute and schedule
    """

    # param calculate
    in_shape = in_tensor['shape']
    in_dtype = in_tensor['dtype']
    ori_shape = in_tensor['ori_shape']
    out_dtype = output_tensor['dtype']

    # shape check, check length,min,max,size
    para_check.check_shape(in_shape, min_rank=4, max_rank=4, param_name="x")

    # type check
    in_dtype = in_dtype.lower()
    para_check.check_dtype(in_dtype, ("float16"), param_name="x")
    out_dtype = out_dtype.lower()
    para_check.check_dtype(out_dtype, ("float16"), param_name="y")

    if not hasattr(axis, 'index'):
        if axis not in (-1, 1):
            error_manager_vector.raise_err_specific_reson("softmax_v2", "the nz format only support last axis.")
    else:
        if axis[0] not in (-1, 1):
            error_manager_vector.raise_err_specific_reson("softmax_v2", "the nz format only support last axis.")

    # shape check
    if in_dtype == "float16" and in_shape[0] * in_shape[3] * 2 > UB_SIZE_LIMIT:
        error_manager_vector.raise_err_input_param_range_invalid("softmax_v2", "C", 0, UB_SIZE_LIMIT,
                                                                 in_shape[1] * in_shape[4] * 2)
    # calc padding parameters
    padding = in_shape[0] * in_shape[3] - ori_shape[1]
    pad_param = []
    if padding < 0:
        error_manager_vector.raise_err_specific_reson("softmax_v2",
                                                      "the shapes of input tensor and original tensor don't match")
    elif padding == 0:
        pad_flag = False
        pad_c1 = 0
        pad_c0 = 0
        pad_param = [pad_c1, pad_c0]
    else:
        pad_flag = True
        pad_c1 = (padding + 15) // 16
        pad_c0 = padding % 16
        pad_param = [pad_c1, pad_c0]
    return_list = [in_shape, in_dtype, pad_flag, pad_param]

    return return_list


def softmax_axis_check(origin_format, value):
    """
    checking the axis of softmax
    data using for compute and schedule

    Parameters
    ----------
    origin_format: string
        origin_format
    value : listint
        the data's axis using for softmax

    Returns
    -------
    axic_is_c : bool
        if the data's axis is c default value is False
    """
    axic_is_c = origin_format[value] == "C"

    return axic_is_c


def update_5hd_axis(origin_format, axis, input_format):
    """
    update the axis of 5hd format
    data using for compute and schedule
    """
    if hasattr(axis, 'index'):
        axis = axis[0]

    axis_str = origin_format[axis]
    offset_6hd = 1 if input_format == "NDC1HWC0" else 0

    dict_format_axis = {
        "N": 0,
        "C": 1 + offset_6hd,
        "H": 2 + offset_6hd,
        "W": 3 + offset_6hd,
        "D": 1
    }

    return [dict_format_axis[axis_str]]


# 'pylint: disable=variable_type_changed
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            (para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_LIST_INT),
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME,
                            para_check.OPTION_ATTR_STR)
def softmax_v2(input_x, output_y, axis=-1, half_to_float=False, kernel_name="softmax_v2",
               impl_mode="high_performance"):
    """
    algorithm: softmax
    calculating data's softmax, produces an output tensor with shape
    the result equals the sum of the x power of e over the sum of
    the x power of e

    Parameters
    ----------
    input_x : dict
        format: FORMAT_ND , NC1HWC0
               dtype: only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    axis : int or list or tuple
        the data's axis.
        format: FORMAT_ND, NC1HWC0
                range == [-d, d-1]
    half_to_float: bool
        if it is true and input dtype is float16, output dtype should be float32
        otherwise, output dtype should be same as input dtype
    kernel_name : str
        cce kernel name, default value is softmax_v2
    impl_mode: str.
        high_precision or high_performance for inference, default value is "high_performance".
        no need to add into ops_info file.

    Returns
    -------
    None
    """
    # get input_x format
    ori_shape = input_x.get("ori_shape")
    input_format = input_x.get("format")
    if input_format == "NC1HWC0":
        if len(ori_shape) == 2:
            new_ori_shape = [1, ori_shape[0], ori_shape[1], 1]
            input_x["ori_shape"] = new_ori_shape
            if not isinstance(axis, int):
                axis = list(axis)
            if not hasattr(axis, 'index'):
                if axis >= 0:
                    axis = axis + 1
                else:
                    axis = axis - 1
            else:
                if axis[0] >= 0:
                    axis[0] = axis[0] + 1
                else:
                    axis[0] = axis[0] - 1
        if len(ori_shape) == 3:
            new_ori_shape = [1, ori_shape[0], ori_shape[1], ori_shape[2]]
            input_x["ori_shape"] = new_ori_shape
            if not isinstance(axis, int):
                axis = list(axis)
            if not hasattr(axis, 'index'):
                if axis >= 0:
                    axis = axis + 1
            else:
                if axis[0] >= 0:
                    axis[0] = axis[0] + 1

    tbe_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    use_dynamic = True
    if input_format in ("NDC1HWC0",) and input_x.get("dtype").lower() == "float32":
        use_dynamic = False
    if _is_special_cases(ori_shape, 0):
        use_dynamic = False
    if tbe_product in ("Hi3796CV300ES", "Hi3796CV300CS", "SD3403"):
        use_dynamic = False
    if input_format in ("FRACTAL_NZ",) and \
        ((ori_shape[-1] % 16 == 0 and ori_shape[-2] % 16 == 0) or ori_shape[-1] in (21841,)):
        use_dynamic = False
    if input_format in ("FRACTAL_NZ",) and \
        input_x.get("dtype").lower() == "float32":
        use_dynamic = True

    if input_format in ("NC1HWC0", "NDC1HWC0", "FRACTAL_NZ") and use_dynamic:
        context = tbe_context.op_context.get_context()
        if context is not None:
            context.set_op_mode("static")
            dyn_impl.softmax_v2(input_x, output_y, axis, False, kernel_name, impl_mode)
        else:
            with tbe_context.op_context.OpContext("static"):
                dyn_impl.softmax_v2(input_x, output_y, axis, False, kernel_name, impl_mode)
        return

    range_x = []
    for dim in input_x.get("shape"):
        range_x.append((dim, dim))
    input_x["range"] = range_x

    if is_white_shape(input_x.get("shape")):
        context = tbe_context.op_context.get_context()
        if context is not None:
            context.set_op_mode("static")
            dyn_impl.softmax_v2(input_x, output_y, axis, False, kernel_name, impl_mode)
        else:
            with tbe_context.op_context.OpContext("static"):
                dyn_impl.softmax_v2(input_x, output_y, axis, False, kernel_name, impl_mode)
        return

    axic_is_c = False
    if input_format in ("NC1HWC0", "NDC1HWC0"):
        if not hasattr(axis, 'index'):
            axic_is_c = softmax_axis_check(input_x.get("ori_format"), axis)
        else:
            axic_is_c = softmax_axis_check(input_x.get("ori_format"), axis[0])
    if input_format == "FRACTAL_NZ" and len(input_x.get("ori_shape")) == 2 \
            and input_x['ori_shape'][1] % 16 != 0:
        [in_shape, in_dtype, pad_flag, pad_param] = \
            softmax_nz_param_check(input_x, output_y, axis, kernel_name)

        # compute & schedule & build
        softmax_nz_channel_calculate(in_shape, in_dtype, pad_flag,
                                     pad_param, kernel_name)
    elif input_format in ("NC1HWC0", "NDC1HWC0") and axic_is_c:
        # 5D format, using TVM primitive, UB fusion is not supported.
        # parameters check
        [in_shape, in_dtype, pad_flag, pad_param] = \
            softmax_param_check(input_x, output_y, axis, kernel_name)

        # compute & schedule & build
        softmax_channel_calculate(in_shape, in_dtype, pad_flag,
                                  pad_param, kernel_name, impl_mode)
    else:
        # ND format, using DSL, UB fusion is not supported.
        # compute & schedule & build
        shape = input_x.get("shape")
        dtype = input_x.get("dtype").lower()

        if not isinstance(axis, int):
            axis = list(axis)

        if input_format in ("NC1HWC0", "NDC1HWC0"):
            axis = update_5hd_axis(input_x.get("ori_format"), axis, input_format)

        para_check.check_shape(shape, param_name="x")
        para_check.check_dtype(dtype, ("float16", "float32"), param_name="x")

        if fz.is_frac_z(input_x):
            axis = fz.to_frac_z_axis(input_x.get("ori_shape"), axis)
        axis = shape_util.axis_check(len(shape), axis)

        shape, axis = shape_util.shape_refine(list(shape), axis)
        shape, axis = shape_util.simplify_axis_shape(shape, axis)

        attr = {"ori_shape": input_x.get("ori_shape")}
        data_input = tvm.placeholder(shape, dtype=dtype, name="data", attrs=attr)
        output = softmax_v2_compute(data_input, output_y, axis, half_to_float, kernel_name, impl_mode)
        with tvm.target.cce():
            result = te.lang.cce.auto_schedule(output)

        tensor_list = [data_input, output]

        config = {"print_ir": False,
                  "name": kernel_name,
                  "tensor_list": tensor_list}

        te.lang.cce.cce_build_code(result, config)
