# Copyright 2020 Huawei Technologies Co., Ltd
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
dynamic softmax_v2
"""
from functools import reduce
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import OpImplMode
from impl.util.util_common import check_op_impl_mode
from impl.util.norm_pattern_adapter import NormPattern
from impl.util import util_common
from impl.util import util_frac_z as fz
from impl.util import util_select_op_base


HIGH_PERFORMANCE_LV1 = "high_performance_lv1"


# 'pylint: disable=too-few-public-methods,not-use-list-comprehension
class Constant:
    """
    The class for constant
    """
    FP16_MAX = tvm.const(6.0e04, dtype="float16")
    FP32_MAX = tvm.const(3.4e38, dtype="float32")
    FP32_USE_FP16_MAX = tvm.const(6.0e04, dtype="float32")
    HIGH_PRECISION_FP16_MAX = tvm.const(6.55e04, dtype="float16")
    HIGH_PRECISION_FP32_MAX = tvm.const(3.4028235e38, dtype="float32")
    HIGH_PRECISION_FP32_USE_FP16_MAX = tvm.const(6.55e04, dtype="float32")


def get_op_support_info(input_x, output_y, axes=-1, half_to_float=False, kernel_name="softmax_v2"):
    format_x = input_x.get("format")
    origin_format_x = input_x.get("ori_format")
    dims_x = len(input_x.get("shape"))

    if not hasattr(axes, 'index'):
        new_axis = axes
    else:
        new_axis = axes[0]
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


# 'pylint: disable=unused-argument
def op_select_format(input_x, output_y, axes=-1, half_to_float=False, kernel_name="softmax_v2"):
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
    tbe_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")

    if util_common.is_unknown([input_x]) or axes is None:
        input0 = util_select_op_base.gen_param(classify="input0",
                                               name="x",
                                               datatype="bfloat16,float16,float32",
                                               format="ND,ND,ND",
                                               unknownshape_format="ND,ND,ND")
        if not half_to_float:
            output0 = util_select_op_base.gen_param(classify="output0",
                                                    name="y",
                                                    datatype="bfloat16,float16,float32",
                                                    format="ND,ND,ND",
                                                    unknownshape_format="ND,ND,ND")
        else:
            output0 = util_select_op_base.gen_param(classify="output0",
                                                    name="y",
                                                    datatype="bfloat16,float32,float32",
                                                    format="ND,ND,ND",
                                                    unknownshape_format="ND,ND,ND")
    else:
        list_axis = list(axes) if not isinstance(axes, int) else [axes]
        is_last_single_reduce = len(list_axis) == 1 and (list_axis[0] == length_x_ori - 1 or list_axis[0] == -1)
        if length_x_ori == 2:
            if shape_x_ori[0] == 1 or not is_last_single_reduce:
                if tbe_product in ("Ascend610", "BS9SX1A", "Ascend310P",):
                    input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                           datatype="float16,float16,float",
                                                           format="NC1HWC0,ND,ND")
                    if not half_to_float:
                        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                                datatype="float16,float16,float",
                                                                format="NC1HWC0,ND,ND")
                    else:
                        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                                datatype="float,float,float",
                                                                format="NC1HWC0,ND,ND")
                if tbe_product in ("Ascend910",) or tbe_platform.api_check_support("tik.vcopy"):
                    input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                           datatype="float16,float16,float,float,bfloat16,bfloat16",
                                                           format="NC1HWC0,ND,ND,NC1HWC0,ND,NC1HWC0")
                    if not half_to_float:
                        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                                datatype="float16,float16,float,"
                                                                         "float,bfloat16,bfloat16",
                                                                format="NC1HWC0,ND,ND,NC1HWC0,ND,NC1HWC0")
                    else:
                        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                                datatype="float,float,float,float,bfloat16,bfloat16",
                                                                format="NC1HWC0,ND,ND,NC1HWC0,ND,NC1HWC0")
            else:
                if tbe_product in ("Ascend610", "BS9SX1A", "Ascend310P",):
                    input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                           datatype="float16,float16,float16,float",
                                                           format="FRACTAL_NZ,NC1HWC0,ND,ND")
                    if not half_to_float:
                        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                                datatype="float16,float16,float16,float",
                                                                format="FRACTAL_NZ,NC1HWC0,ND,ND")
                    else:
                        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                                datatype="float,float,float,float",
                                                                format="FRACTAL_NZ,NC1HWC0,ND,ND")
                if tbe_product in ("Ascend910",) or tbe_platform.api_check_support("tik.vcopy"):
                    input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                           datatype="float16,float16,float16,float,float,"
                                                                    "bfloat16,bfloat16",
                                                           format="FRACTAL_NZ,NC1HWC0,ND,ND,NC1HWC0,"
                                                                  "ND,NC1HWC0")
                    if not half_to_float:
                        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                                datatype="float16,float16,float16,float,float,"
                                                                         "bfloat16,bfloat16",
                                                                format="FRACTAL_NZ,NC1HWC0,ND,ND,NC1HWC0,"
                                                                       "ND,NC1HWC0")
                    else:
                        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                                datatype="float,float,float,float,float,"
                                                                         "bfloat16,bfloat16",
                                                                format="FRACTAL_NZ,NC1HWC0,ND,ND,NC1HWC0,"
                                                                       "ND,NC1HWC0")
        elif length_x_ori > 2 and (shape_x_ori[-1] % 16 != 0 or shape_x_ori[-2] % 16 != 0):
            if tbe_product in ("Ascend610", "BS9SX1A", "Ascend310P",):
                if _is_ori_special_cases(shape_x_ori, 0) and is_last_single_reduce:
                    input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                           datatype="float16",
                                                           format="FRACTAL_NZ")
                    if not half_to_float:
                        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                                datatype="float16",
                                                                format="FRACTAL_NZ")
                    else:
                        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                                datatype="float",
                                                                format="FRACTAL_NZ")
                elif is_last_single_reduce:
                    input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                           datatype="float16,float16,float16,float,float16,float",
                                                           format="NC1HWC0,NDC1HWC0,ND,ND,FRACTAL_NZ,FRACTAL_NZ")
                    if not half_to_float:
                        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                                datatype="float16,float16,float16,float,float16,float",
                                                                format="NC1HWC0,NDC1HWC0,ND,ND,FRACTAL_NZ,FRACTAL_NZ")
                    else:
                        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                                datatype="float,float,float,float,float,float",
                                                                format="NC1HWC0,NDC1HWC0,ND,ND,FRACTAL_NZ,FRACTAL_NZ")
                else:
                    input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                           datatype="float16,float16,float16,float",
                                                           format="NC1HWC0,NDC1HWC0,ND,ND")
                    if not half_to_float:
                        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                                datatype="float16,float16,float16,float",
                                                                format="NC1HWC0,NDC1HWC0,ND,ND")
                    else:
                        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                                datatype="float,float,float,float",
                                                                format="NC1HWC0,NDC1HWC0,ND,ND")
            if tbe_product in ("Ascend910",) or tbe_platform.api_check_support("tik.vcopy"):
                if shape_x_ori[-2] < 5 or not is_last_single_reduce:
                    input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                           datatype="float16,float16,float,float,float16,float,"
                                                                    "bfloat16,bfloat16,bfloat16",
                                                           format="NC1HWC0,ND,ND,NC1HWC0,NDC1HWC0,NDC1HWC0,"
                                                                  "NC1HWC0,ND,NDC1HWC0")
                    if not half_to_float:
                        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                                datatype="float16,float16,float,float,float16,float,"
                                                                         "bfloat16,bfloat16,bfloat16",
                                                                format="NC1HWC0,ND,ND,NC1HWC0,NDC1HWC0,NDC1HWC0,"
                                                                       "NC1HWC0,ND,NDC1HWC0")
                    else:
                        output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                                datatype="float,float,float,float,float,float,"
                                                                         "bfloat16,bfloat16,bfloat16",
                                                                format="NC1HWC0,ND,ND,NC1HWC0,NDC1HWC0,NDC1HWC0,"
                                                                       "NC1HWC0,ND,NDC1HWC0")
                else:
                    input0 = util_select_op_base.gen_param(
                        classify="input0", name="x",
                        datatype="float16,float16,float,float,float16,float,float16,float,"
                                 "bfloat16,bfloat16,bfloat16,bfloat16",
                        format="NC1HWC0,ND,ND,NC1HWC0,NDC1HWC0,NDC1HWC0,FRACTAL_NZ,FRACTAL_NZ,"
                               "NC1HWC0,ND,NDC1HWC0,FRACTAL_NZ"
                    )
                    if not half_to_float:
                        output0 = util_select_op_base.gen_param(
                            classify="output0", name="y",
                            datatype="float16,float16,float,float,float16,float,float16,float,"
                                     "bfloat16,bfloat16,bfloat16,bfloat16",
                            format="NC1HWC0,ND,ND,NC1HWC0,NDC1HWC0,NDC1HWC0,FRACTAL_NZ,FRACTAL_NZ,"
                                   "NC1HWC0,ND,NDC1HWC0,FRACTAL_NZ"
                        )
                    else:
                        output0 = util_select_op_base.gen_param(
                            classify="output0", name="y",
                            datatype="float,float,float,float,float,float,float,float,"
                                     "bfloat16,bfloat16,bfloat16,bfloat16",
                            format="NC1HWC0,ND,ND,NC1HWC0,NDC1HWC0,NDC1HWC0,FRACTAL_NZ,FRACTAL_NZ,"
                                   "NC1HWC0,ND,NDC1HWC0,FRACTAL_NZ"
                        )
        else:
            if tbe_product in ("Ascend610", "BS9SX1A", "Ascend310P",):
                input0 = util_select_op_base.gen_param(classify="input0", name="x",
                                                        datatype="float16,float,float16,float16,float,float16",
                                                        format="FRACTAL_NZ,FRACTAL_NZ,NC1HWC0,ND,ND,NDC1HWC0")
                if not half_to_float:
                    output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                            datatype="float16,float,float16,float16,float,float16",
                                                            format="FRACTAL_NZ,FRACTAL_NZ,NC1HWC0,ND,ND,NDC1HWC0")
                else:
                    output0 = util_select_op_base.gen_param(classify="output0", name="y",
                                                            datatype="float,float,float,float,float,float",
                                                            format="FRACTAL_NZ,FRACTAL_NZ,NC1HWC0,ND,ND,NDC1HWC0")
            if tbe_product in ("Ascend910",) or tbe_platform.api_check_support("tik.vcopy"):
                input0 = util_select_op_base.gen_param(
                    classify="input0", name="x",
                    datatype="float16,float,float16,float16,float,float,float16,float,"
                             "bfloat16,bfloat16,bfloat16,bfloat16",
                    format="FRACTAL_NZ,FRACTAL_NZ,NC1HWC0,ND,ND,NC1HWC0,NDC1HWC0,NDC1HWC0,"
                           "NC1HWC0,ND,NDC1HWC0,FRACTAL_NZ"
                )
                if not half_to_float:
                    output0 = util_select_op_base.gen_param(
                        classify="output0", name="y",
                        datatype="float16,float,float16,float16,float,float,float16,float,"
                                 "bfloat16,bfloat16,bfloat16,bfloat16",
                        format="FRACTAL_NZ,FRACTAL_NZ,NC1HWC0,ND,ND,NC1HWC0,NDC1HWC0,NDC1HWC0,"
                               "NC1HWC0,ND,NDC1HWC0,FRACTAL_NZ"
                    )
                else:
                    output0 = util_select_op_base.gen_param(
                        classify="output0", name="y",
                        datatype="float,float,float,float,float,float,float,float,"
                                 "bfloat16,bfloat16,bfloat16,bfloat16",
                        format="FRACTAL_NZ,FRACTAL_NZ,NC1HWC0,ND,ND,NC1HWC0,NDC1HWC0,NDC1HWC0,"
                               "NC1HWC0,ND,NDC1HWC0,FRACTAL_NZ"
                    )

    param_list = [input0, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def _is_ori_special_cases(input_shape, compare_type):
    tbe_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if tbe_product not in ("Ascend610", "BS9SX1A", "Ascend310P",):
        return False
    white_list_shape = [[8, 8732, 81], [16, 8732, 81], [96, 50, 50],
                        [192, 50, 50], [384, 50, 50], [768, 50, 50], [64, 50, 50],
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
                    if shape_t[i] == shape_w[i]:
                        count += 1
                        continue
                    break
                if count == shape_t_size:
                    return True

    return False


def _is_special_cases(input_shape):
    white_list_shape = [[96, 4]]
    shape_t = list(input_shape)
    if shape_t in white_list_shape:
        return True
    return False


def check_is_axes_with_last(shape, axes):
    """
    check_is_axes_with_last
    """
    if len(axes) > 1:
        for i, _ in enumerate(axes):
            if axes[i] == len(shape) - 1:
                return True
    return False


def do_broadcast_process(shape, axes, data):
    """
    do broadcast process
    """
    if check_is_axes_with_last(shape, axes):
        tmp_shape = list(data.shape[:-1]) + [shape[-1]]
        data = tbe.broadcast(data, tmp_shape)
        data = tbe.broadcast(data, shape)
    else:
        data = tbe.broadcast(data, shape)
    return data


def cal_max_with_fp16(input_x, list_axis):
    """
    cal rduce_max with float16
    """
    data_max_input = tbe.cast_to(input_x, "float16")
    data_max_output = tbe.reduce_max(data_max_input, axis=list_axis, keepdims=True)
    data_max = tbe.cast_to(data_max_output, "float32")
    return data_max


def check_supported(input_x, output_y, axes=-1, half_to_float=False, kernel_name="softmax_v2",
                    impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    Judge whether the current input specification supports
    """
    return True, ""


# 'pylint:disable=too-many-locals,disable=too-many-statements,too-many-branches
@register_operator("SoftmaxV2")
@register_operator_compute("SoftmaxV2", op_mode="dynamic", support_fusion=True)
def softmax_v2_compute(input_x, output_y, axes=-1, half_to_float=False, kernel_name="softmax_v2",
                       impl_mode=OpImplMode.HIGH_PERFORMANCE):
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
    axes : int or list or tuple
        the data's axes, range == [-d, d-1]
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
    shape = shape_util.shape_to_list(input_x.shape)
    list_axis = list(axes)
    last_dim = len(input_x.shape) - 1

    attributes = input_x.op.attrs
    disable_fuse_axes = attributes["disable_fuse_axes"]
    input_format = attributes["format"]
    ori_format = attributes["ori_format"]
    is_bf16 = False
    ori_shape = shape_util.shape_to_list(attributes["ori_shape"])
    if dtype == "bfloat16" or (half_to_float and dtype == "float16"):
        if dtype == "bfloat16":
            is_bf16 = True
        dtype = "float32"
        input_x = tbe.cast_to(input_x, "float32")
    if dtype == "float32":
        max_const = Constant.HIGH_PRECISION_FP32_MAX if impl_mode == OpImplMode.HIGH_PRECISION else Constant.FP32_MAX
    else:
        max_const = Constant.HIGH_PRECISION_FP16_MAX if impl_mode == OpImplMode.HIGH_PRECISION else Constant.FP16_MAX
    vcmax_flag = False

    check_axis_list = [-1, last_dim]
    for i in list_axis:
        if i in check_axis_list:
            vcmax_flag = True

    is_use_value = False
    is_another_axes_use_value = False
    if len(list_axis) == 2:
        if input_format in ("NC1HWC0", "NDC1HWC0"):
            is_use_value = True
            idc_list = shape_util.shape_to_list(disable_fuse_axes)
            idx_c0 = idc_list[1]
            ori_format = ori_format.upper()
            c = ori_shape[ori_format.find('C')]
            c = tbe.var('c') if c == -1 else c
            pad_c = tvm.floormod(c - 1, shape[idx_c0]) + 1
        if input_format in ("FRACTAL_NZ",):
            is_use_value = True
            is_another_axes_use_value = True
            idc_list = shape_util.shape_to_list(disable_fuse_axes)
            idx_c1 = idc_list[0]
            idx_c0 = idc_list[1]
            c = -1
            if (idx_c0 - idx_c1) == 2:
                c = ori_shape[-1]
                another_c = ori_shape[-2]
            else:
                c = ori_shape[-2]
                another_c = ori_shape[-1]
            if c % 16 == 0:
                is_use_value = False
            if another_c % 16 == 0:
                is_another_axes_use_value = False
            c = tbe.var('c') if c == -1 else c
            pad_c = tvm.floormod(c - 1, shape[idx_c0]) + 1

    is_cal_max_with_fp16 = False
    if dtype == "float32":
        if vcmax_flag and not tbe_platform.api_check_support("te.lang.cce.reduce_max", "float32"):
            is_cal_max_with_fp16 = True
        elif input_format in ("NDC1HWC0") and is_use_value and impl_mode != OpImplMode.HIGH_PRECISION:
            is_cal_max_with_fp16 = True
        elif impl_mode == HIGH_PERFORMANCE_LV1:
            is_cal_max_with_fp16 = True
    if is_cal_max_with_fp16:
        if impl_mode == OpImplMode.HIGH_PRECISION:
            max_const = Constant.HIGH_PRECISION_FP32_USE_FP16_MAX
        else:
            max_const = Constant.FP32_USE_FP16_MAX
    if is_use_value:
        input_x = tbe.set_value(input_x, lambda *i: tvm.all(i[list_axis[0]] > shape[list_axis[0]] - 2, \
                                                            i[list_axis[1]] > pad_c - 1), -max_const)

    if is_cal_max_with_fp16:
        data_max = cal_max_with_fp16(input_x, list_axis)
    else:
        data_max = tbe.reduce_max(input_x, axis=list_axis, keepdims=True)

    data_max = do_broadcast_process(shape, axes, data_max)
    if is_use_value and impl_mode == OpImplMode.HIGH_PRECISION:
        scalar_zero = tvm.const(0.0, dtype=input_x.dtype)
        input_x = tbe.vadds(input_x, scalar_zero)
        input_x = tbe.set_value(input_x, lambda *i: tvm.all(i[list_axis[0]] > shape[list_axis[0]] - 2, \
                                                            i[list_axis[1]] > pad_c - 1), scalar_zero)
    data_subtrac = tbe.vsub(input_x, data_max)

    has_improve_precision = False
    if dtype == "float16" and tbe_platform.api_check_support("te.lang.cce.vexp", "float32"):
        data_subtrac = tbe.cast_to(data_subtrac, "float32")
        has_improve_precision = True

    tbe_product = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if data_subtrac.dtype == "float32" and tbe_product in ("Ascend310",):
        data_subtrac = tbe.cast_to(data_subtrac, "float16")
        data_exp = tbe.vexp(data_subtrac)
        data_exp = tbe.cast_to(data_exp, "float32")
    else:
        data_exp = tbe.vexp(data_subtrac)

    if data_exp.dtype == "float16" and tbe_product in ("Ascend310",):
        data_exp = tbe.cast_to(data_exp, "float32")
        has_improve_precision = True

    if is_use_value:
        data_exp = tbe.set_value(data_exp, lambda *i: tvm.all(i[list_axis[0]] > shape[list_axis[0]] - 2, \
                                                              i[list_axis[1]] > pad_c - 1), 0)
    data_expsum = tbe.reduce_sum(data_exp, list_axis, keepdims=True)

    if is_use_value and input_format in ("FRACTAL_NZ",):
        if not _is_ori_special_cases(ori_shape, 1):
            data_expsum = do_broadcast_process(shape, axes, data_expsum)
            output = tbe.vdiv(data_exp, data_expsum)
        else:
            data_expsum = tbe.vrec(data_expsum)
            data_expsum = do_broadcast_process(shape, axes, data_expsum)
            output = tbe.vmul(data_exp, data_expsum)
    elif (tbe_product in ("Ascend910", "Ascend610", "BS9SX1A", "Ascend310P") or
            tbe_platform.api_check_support("tik.vcopy")
       ) and output_y.get("format") == "FRACTAL_NZ" and dtype == "float16":
        if _is_special_cases(ori_shape):
            data_expsum = tbe.vrec(data_expsum, "high_precision")
        elif impl_mode == OpImplMode.HIGH_PRECISION:
            data_expsum = tbe.vrec(data_expsum, "high_precision")
        else:
            data_expsum = tbe.vrec(data_expsum)
        data_expsum = do_broadcast_process(shape, axes, data_expsum)
        output = tbe.vmul(data_exp, data_expsum)
    elif dtype == "float32" and input_format in ("NDC1HWC0") and is_use_value:
        data_expsum = tbe.vrec(data_expsum, "high_precision")
        data_expsum = do_broadcast_process(shape, axes, data_expsum)
        output = tbe.vmul(data_exp, data_expsum)
    elif tbe_product in ("Ascend610", "BS9SX1A", "Ascend310P"):
        tensor_one_broadcast = tbe.broadcast(tvm.const(1, data_expsum.dtype), data_expsum.shape)
        data_div = tbe.vdiv(tensor_one_broadcast, data_expsum)
        data_div_broadcast = do_broadcast_process(shape, axes, data_div)
        output = tbe.vmul(data_exp, data_div_broadcast)
    else:
        data_expsum = do_broadcast_process(shape, axes, data_expsum)
        output = tbe.vdiv(data_exp, data_expsum)

    if has_improve_precision and dtype == "float16":
        output = tbe.cast_to(output, "float16")

    if is_another_axes_use_value and input_format in ("FRACTAL_NZ",):
        output = tbe.set_value(output, lambda *i: i[-2] > ori_shape[-2] - 1, 0)

    if is_bf16:
        output = tbe.round(output, "bfloat16")

    return output


def update_5hd_axis(origin_format, list_axis, input_format):
    """
    update the axes of 5hd format
    data using for compute and schedule
    """
    if hasattr(list_axis, 'index'):
        list_axis = list_axis[0]

    axis_str = origin_format[list_axis]
    offset_6hd = 1 if input_format == "NDC1HWC0" else 0

    dict_format_axis = {
        "N": [0],
        "C": [1 + offset_6hd, 4 + offset_6hd],
        "H": [2 + offset_6hd],
        "W": [3 + offset_6hd],
        "D": [1]
    }

    return dict_format_axis.get(axis_str)


def check_axis_is_int(axes):
    """
    check axes wherther int
    """
    if not isinstance(axes, int):
        axes = list(axes)
    return axes


# 'pylint:disable=invalid-name,too-many-locals
@register_operator("SoftmaxV2")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            (para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_LIST_INT),
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME,
                            para_check.OPTION_ATTR_STR)
def softmax_v2(input_x, output_y, axes=-1, half_to_float=False, kernel_name="softmax_v2",
               impl_mode=OpImplMode.HIGH_PERFORMANCE):
    """
    algorithm: softmax
    calculating data's softmax, produces an output tensor with shape
    the result equals the sum of the x power of e over the sum of
    the x power of e

    Parameters
    ----------
    input_x : dict
        format: FORMAT_ND , NC1HWC0
               dtype: only support float16, float32, bfloat16
    output_y: dict
        shape and dtype of output, should be same shape and type as input
    axes : int or list or tuple
        the data's axes.
        format: FORMAT_ND, NC1HWC0
                range == [-d, d-1]
    half_to_float: bool
        if it is true and input dtype is float16, output dtype should be float32
        otherwise, output dtype should be same as input dtype
    kernel_name : str
        cce kernel name, default value is softmax_v2
    impl_mode: str.
        high_precision or high_performance for inference, default value is OpImplMode.HIGH_PERFORMANCE.
        no need to add into ops_info file.

    Returns
    -------
    None
    """
    check_op_impl_mode(impl_mode, [OpImplMode.HIGH_PERFORMANCE, OpImplMode.HIGH_PRECISION, HIGH_PERFORMANCE_LV1],
                                   kernel_name)

    shape = input_x.get("shape")
    dtype = input_x.get("dtype").lower()
    input_format = input_x.get("format")
    ori_format = input_x.get("ori_format")
    ori_shape = input_x.get("ori_shape")
    para_check.check_dtype(dtype, ("bfloat16", "float16", "float32"), param_name="x")
    para_check.check_shape(shape, param_name="x")

    if input_format == "NC1HWC0":
        if len(ori_shape) == 2:
            new_ori_shape = [1, ori_shape[0], ori_shape[1], 1]
            input_x["ori_shape"] = new_ori_shape
            axes = check_axis_is_int(axes)
            if not hasattr(axes, 'index'):
                axes = axes + 1 if axes >= 0 else axes - 1
            else:
                axes[0] = axes[0] + 1 if axes[0] >= 0 else axes[0] - 1
        if len(ori_shape) == 3:
            new_ori_shape = [1, ori_shape[0], ori_shape[1], ori_shape[2]]
            input_x["ori_shape"] = new_ori_shape
            axes = check_axis_is_int(axes)
            if not hasattr(axes, 'index'):
                if axes >= 0:
                    axes = axes + 1
            else:
                if axes[0] >= 0:
                    axes[0] = axes[0] + 1
        ori_shape = input_x.get("ori_shape")

    extra_params = dict()
    if axes is None:
        # when axes is None, it is binary case, go unknown axes schedule
        list_axis = NormPattern.REDUCE_UNKNOWN_MODE
        extra_params.update(NormPattern.REDUCE_SINGLE_TYPE)
        operation.add_compile_info(NormPattern.REDUCE_ATTR_IDX, 0)
        operation.add_compile_info(NormPattern.REDUCE_ATTR_NAME, "axes")
        operation.add_compile_info(NormPattern.REDUCE_ATTR_DTYPE, "ListInt")
    elif not isinstance(axes, int):
        list_axis = list(axes)
    else:
        list_axis = [axes]

    # only static op support special format, update axes for special format
    if not util_common.is_unknown(input_x):
        if input_format in ("NC1HWC0", "NDC1HWC0"):
            list_axis = update_5hd_axis(ori_format, list_axis, input_format)

        if fz.is_frac_z(input_x):
            list_axis = fz.to_frac_z_axis(ori_shape, list_axis)

        if input_format in ("NC1HWC0", "NDC1HWC0", "FRACTAL_NZ") and len(list_axis) == 2:
            extra_params.update({"disable_fuse_axes": [list_axis[0], list_axis[1]]})

    tensors = []
    schedules = []
    ins = classify([input_x, list_axis], OpPatternMode.NORM, extra_params)

    for idx, (x, reduce_axis) in enumerate(ins):
        with tbe.compute():
            disable_fuse_axes = []
            if "disable_fuse_axes" in extra_params:
                disable_fuse_axes = extra_params.get("disable_fuse_axes")[idx]
            shape_var_new = shape_util.variable_shape([x], op_mode="norm")[0]
            input_x = tvm.placeholder(shape_var_new,
                                      dtype=dtype,
                                      name="input_x",
                                      attrs={
                                          "ori_shape": ori_shape,
                                          "ori_format": ori_format,
                                          "format": input_format,
                                          "disable_fuse_axes": disable_fuse_axes
                                      })
            output = softmax_v2_compute(input_x, output_y, reduce_axis, half_to_float, kernel_name, impl_mode)
            tensors.append([input_x, output])

        with tvm.target.cce():
            sch = tbe.auto_schedule(output)
        schedules.append(sch)

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)
