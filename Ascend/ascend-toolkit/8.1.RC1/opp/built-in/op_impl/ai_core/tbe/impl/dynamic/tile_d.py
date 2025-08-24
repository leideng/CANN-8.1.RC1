# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
dynamic tile_d
"""
from impl.util import util_common
from impl.util import util_select_op_base
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator_compute
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info


# 'pylint: disable = unused-argument
def get_op_support_info(input_x, output_x, multiples, kernel_name="tile_d"):
    """
    get_op_support_info
    """
    shape_x = input_x.get("shape")
    shape_x = list(shape_x)
    multiples = list(multiples)
    format_x = input_x.get("format").upper()
    format_output = output_x.get("format").upper()
    if format_x == "ND":
        axis_split_matrix = []
        if len(shape_x) < len(multiples):
            len_error = len(multiples) - len(shape_x)
            shape_x = [1] * len_error + shape_x
        for i, shape_i in enumerate(shape_x):
            multiples_i = multiples[i]
            if multiples_i == 1 and shape_i != 1:
                split_0 = [SplitInput([0, [i], [-1], [-1]]), SplitOutput([0, [i]])]
                axis_split_matrix.append(split_0)
        axis_reduce_list = None
    elif format_x in ("NC1HWC0",):
        # cut N
        axis_split_matrix = [[SplitInput([0, [0], [-1], [-1]]), SplitOutput([0, [0]])]]
        axis_reduce_list = None
    elif format_x in ("NCHW", "NHWC") and format_output == "NC1HWC0":
        shape_x = shape_x + [1]
        if format_x == "NCHW":
            multiples = [multiples[0], multiples[1] // 16, multiples[2], multiples[3], 16]
        if format_x == "NHWC":
            multiples = [multiples[0], multiples[3] // 16, multiples[1], multiples[2], 16]
        if len(shape_x) < len(multiples):
            len_error = len(multiples) - len(shape_x)
            shape_x = [1] * len_error + shape_x
        axis_split_matrix = []
        for i, shape_i in enumerate(shape_x):
            multiples_i = multiples[i]
            if multiples_i == 1 and shape_i != 1:
                split_0 = [SplitInput([0, [i], [-1], [-1]]), SplitOutput([0, [i]])]
                axis_split_matrix.append(split_0)
        axis_reduce_list = None
    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


def op_select_format(input_x, output_x, multiples, kernel_name="tile_d"):
    """TileD: to do boradcast with multiples

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_x: dict
        dict of output.
    multiples : list or tuple.
        Number of the axis replicates.
    kernel_name : str
        kernel name, default value is "tile_d".

    Returns
    -------
    param_dynamic_in_json
    """
    input_shape = list(input_x.get("ori_shape"))
    input_shape = shape_util.scalar2tensor_one(input_shape)
    input_format = input_x.get("ori_format")

    # check whether support 4D to 5HD
    align_len = 16
    is_support_4d_to_5hd = False
    is_support_hd = False
    is_support_fz = False

    hd_support_format = \
        util_common.get_fused_format_str(["N", "D", "H", "W", "C"]) \
        + util_common.get_fused_format_str(["N", "H", "W", "C"])

    if input_format in hd_support_format and len(input_shape) == len(input_format) \
            and len(input_shape) == len(multiples):
        is_muti_c_align = multiples[input_format.index("C")] % align_len == 0
        is_shape_c_align = input_shape[input_format.index("C")] % align_len == 0
        is_shape_n_align = input_shape[input_format.index("N")] % align_len == 0
        # special 4d to 5hd
        if input_format in ("NCHW", "NHWC") and len(input_shape) == 4 and \
                list(input_shape[1:]) == [1, 1, 1] and is_muti_c_align:
            is_support_4d_to_5hd = True

        if is_shape_c_align or multiples[input_format.index("C")] == 1:
            is_support_hd = True

        if is_support_hd and (is_shape_n_align or multiples[input_format.index("N")] == 1) and \
                util_common.is_support_fractal_z_input(input_x):
            is_support_fz = True

    is_support_nz = False
    if len(input_shape) >= 2:
        # whether the -1 dim size align
        is_neg_one_dim_align = input_shape[-1] % align_len == 0
        # whether the -2 dim size align
        is_neg_two_dim_align = input_shape[-2] % align_len == 0

        if (is_neg_one_dim_align or multiples[-1] == 1) and (is_neg_two_dim_align or multiples[-2] == 1):
            is_support_nz = True

    # ND dtype
    dtype_base = ["float16", "float", "int32", "uint8"]
    dtype_list = ["float16", "float", "int32", "bool", "uint8"]
    bfp16_support = tbe_platform.intrinsic_check_support("Intrinsic_vconv", "bf162f32")
    if bfp16_support:
        dtype_base.append("bfloat16")
        dtype_list.append("bfloat16")

    # default supprt ND for dtype_base
    dtype_base_out = dtype_list.copy()
    format_base_out = ["ND"] * len(dtype_list)
    format_base_in = ["ND"] * len(dtype_list)
    if is_support_4d_to_5hd and not util_common.is_dynamic_input([input_x]):
        dtype_base_out = dtype_base_out + dtype_base + dtype_base
        format_base_in = format_base_in + ["NCHW"] * len(dtype_base) + ["NHWC"] * len(dtype_base)
        format_base_out = format_base_out + ["NC1HWC0"] * len(dtype_base) + ["NC1HWC0"] * len(dtype_base)
    if is_support_hd and not util_common.is_dynamic_input([input_x]):
        other_format = "NC1HWC0" if len(input_shape) == 4 else "NDC1HWC0"
        dtype_base_out = dtype_base_out + dtype_base
        format_base_in = format_base_in + [other_format] * len(dtype_base)
        format_base_out = format_base_out + [other_format] * len(dtype_base)
    if is_support_fz and not util_common.is_dynamic_input([input_x]):
        other_format = "FRACTAL_Z" if len(input_shape) == 4 else "FRACTAL_Z_3D"
        dtype_base_out = dtype_base_out + dtype_base
        format_base_in = format_base_in + [other_format] * len(dtype_base)
        format_base_out = format_base_out + [other_format] * len(dtype_base)
    if is_support_nz and not util_common.is_dynamic_input([input_x]):
        other_format = "FRACTAL_NZ"
        dtype_base_out = dtype_base_out + dtype_base
        format_base_in = format_base_in + [other_format] * len(dtype_base)
        format_base_out = format_base_out + [other_format] * len(dtype_base)

    dtype_str = ",".join(dtype_base_out)
    format_input_str = ",".join(format_base_in)
    format_output_str = ",".join(format_base_out)

    input0 = util_select_op_base.gen_param(classify="input0",
                                           name="x",
                                           datatype=dtype_str,
                                           format=format_input_str,
                                           unknownshape_format=format_input_str)
    output0 = util_select_op_base.gen_param(classify="output0",
                                            name="y",
                                            datatype=dtype_str,
                                            format=format_output_str,
                                            unknownshape_format=format_output_str)
    param_list = [input0, output0]
    param_dynamic_in_json = util_select_op_base.get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


def update_mutiples_with_format(input_foramt, input_ori_format, mutiples):
    """
    _update_mutiples_with_format
    """
    if input_foramt in ("FRACTAL_Z", "FRACTAL_Z_3D", "NDC1HWC0", "NC1HWC0"):
        # when NDC1HWC0 or NC1HWC0 will update C0 C1 for mutiples
        # ex: mutiples [NDCHW] ->  [NDCHW1]
        # when FRACTAL_Z or FRACTAL_Z_3D will update C0 C1 N0 N1for mutiples
        # ex: mutiples [NDCHW] ->  [DCHWN11]
        _multiples_dict = dict(zip(list(input_ori_format), mutiples))
        _multiples_dict["one"] = 1
        format_idx = []

        if input_foramt == "NDC1HWC0":
            format_idx = ("N", "D", "C", "H", "W", "one")
        elif input_foramt == "NC1HWC0":
            format_idx = ("N", "C", "H", "W", "one")
        elif input_foramt == "FRACTAL_Z":
            format_idx = ("C", "H", "W", "N", "one", "one")
        elif input_foramt == "FRACTAL_Z_3D":
            format_idx = ("D", "C", "H", "W", "N", "one", "one")

        mutiples = [_multiples_dict[key] for key in format_idx]

    if input_foramt in ("FRACTAL_NZ",):
        # when FRACTAL_NZ will update -1 -2 for mutiples
        # ex: mutiples [ABCD] ->  [ABDC11]
        mutiples = mutiples[:-2] + [mutiples[-1], mutiples[-2], 1, 1]

    return mutiples


def shape_mutiples_with_format(input_x, input_shape, input_format, output_format, multiples):
    if input_format in ("NCHW", "NHWC") and output_format in ("NC1HWC0",):
        # branch: 4D tile to 5HD ((N, 1, 1, 1) to (N, C1, H, W, C0)) and output C is 16 align
        # change input shape from (N, 1, 1, 1) to (N, 1, 1, 1, 1)
        input_shape = input_shape + [1]
        if input_format == "NCHW":
            # change multiples from (1, C, H, W) to (1, C1, H, W, C0)
            multiples = [multiples[0], multiples[1] // 16, multiples[2], multiples[3], 16]
        else:
            # change multiples from (1, H, W, C) to (1, C1, H, W, C0)
            multiples = [multiples[0], multiples[3] // 16, multiples[1], multiples[2], 16]
    elif input_format in ("FRACTAL_Z", "FRACTAL_Z_3D", "NDC1HWC0", "NC1HWC0", "FRACTAL_NZ"):
        input_ori_format = input_x.get("ori_format")
        multiples = update_mutiples_with_format(input_format, input_ori_format, multiples)

    return [input_shape, multiples]


def len_between_input_multiples(input_shape, input_range, multiples):
    if len(input_shape) > len(multiples):
        multiples = [1] * (len(input_shape) - len(multiples)) + list(multiples)
    if len(input_shape) < len(multiples):
        len_diff = len(multiples) - len(input_shape)
        input_shape = [1] * len_diff + input_shape
        input_range = [(1, 1)] * len_diff + input_range

    return [input_shape, input_range, multiples]


def shape_multiples_range_with_adapt(input_shape, multiples, input_range):
    shape_adapt = []
    multiples_adapt = []
    range_adapt = []
    multiples_align = []
    for shape_i, multiples_i, range_i in zip(input_shape, multiples, input_range):
        if multiples_i != 1 and shape_i != 1:
            shape_adapt.extend([1, shape_i])
            range_adapt.extend([(1, 1), range_i])
            multiples_adapt.extend([multiples_i, 1])
            multiples_align.extend([multiples_i, shape_i])
        else:
            shape_adapt.append(shape_i)
            range_adapt.append(range_i)
            multiples_adapt.append(multiples_i)
            multiples_align.append(multiples_i * shape_i)
    return [shape_adapt, range_adapt, multiples_adapt, multiples_align]


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
# 'pylint: disable=too-many-statements
@register_operator_compute("TileD", op_mode="dynamic", support_fusion=False)
def tile_d_compute(data, output_x, multiples, kernel_name="tile_d"):
    """TVM calculation process, used for fusion operation.

    Parameters
    ----------
    data: list of placeholders.
        Input data.
    output_x: dict.
        dict of output.
    multiples : list or tuple.
        Number of the axis replicates.
    kernel_name : str.
        Cce kernel name, default value is "tile_d".

    Returns
    -------
    res
    """
    src_dtype = data.dtype.lower()
    shape = shape_util.shape_to_list(data.shape)
    out_shape = []
    for shape_i, multiples_i in zip(shape, multiples):
        out_shape_i = shape_i * multiples_i
        out_shape.append(out_shape_i)
    if src_dtype == "int8" or src_dtype == "uint8":
        data = tbe.cast_to(data, "float16")
    res = tbe.broadcast(data, out_shape)
    if src_dtype == "int8" or src_dtype == "uint8":
        res = tbe.cast_to(res, src_dtype)

    return res


# 'pylint: disable=too-many-locals
@register_operator("TileD")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.KERNEL_NAME)
def tile_d(input_x, output_x, multiples, kernel_name="tile_d"):
    """algorithm: tile.
    The tile in tensorflow can multiple the shape of the given tensor.
    For example, tiling [a b c d] by [2] produces [a b c d a b c d].
    The tile op in TBE is different from tf.tile, tile of TBE use broadcast
    api, and only support that at least an axis in shape is 1.The '1' axis
    is to be multipled.
    For example, if shape = [51, 1] and multiples = [1, 77], after computation,
    the output shape will be [51, 77].
    Abnormal condition:
    1. The length of shape must be equal to or less than the shape of multiples.
    2. The type of kernel_name is not string.
    3. The shape is neither list nor tuple.
    4. The dtype is not float32, float16, or int32.
    5. All of the axises of the multiples is 1.

    Parameters
    ----------
    input_x : dict
        shape and dtype of input
    output_x: dict
        dict of output.
    multiples : list or tuple
        Number of the axis replicates.
    kernel_name : str.
        kernel name, default value is "tile_d".

    Returns
    -------
    None
    """
    # Check not all shape is 1
    origin_multiples = list(multiples)
    axis_not_multiple = 0
    for multiples_i in origin_multiples:
        if multiples_i == 1:
            axis_not_multiple += 1
    if axis_not_multiple == len(origin_multiples):
        error_manager_vector.raise_err_input_param_range_invalid("tile_d", "axis_not_multiple", "1",
                                                                 str(len(origin_multiples) - 1), str(axis_not_multiple))

    # Check support dtype
    input_dtype = input_x.get("dtype").lower()
    check_list = ("bfloat16", "float16", "float32", "int32", "int8", "bool", "uint8")
    para_check.check_dtype(input_dtype, check_list, param_name="input_x")
    if input_dtype == "bool":
        input_dtype = "int8"

    input_range = list(input_x.get("range"))
    input_shape = list(input_x.get("shape"))

    # Write unknown dim index into tiling_info
    tiling_info = []
    for idx, shape_i in enumerate(input_shape):
        if shape_i == -1:
            tiling_info.append(idx)
    tiling_info.insert(0, len(tiling_info))

    if not util_common.is_unknown(input_x):
        multiples = list(multiples)
        input_format = input_x.get("format")
        output_format = output_x.get("format")
        input_shape, multiples = shape_mutiples_with_format(input_x, \
                                 input_shape, input_format, output_format, multiples)
        input_range = util_common.gen_range(input_shape)

    # Check len between input and multiples, the multiples len must not be less than input len
    input_shape, input_range, multiples = len_between_input_multiples(input_shape, input_range, multiples)

    shape_adapt, range_adapt, multiples_adapt, \
            multiples_align = shape_multiples_range_with_adapt(input_shape, multiples, input_range)

    tiling_info.extend(shape_adapt)
    tiling_info.extend(multiples_align)

    input_x["shape"] = input_x["ori_shape"] = shape_adapt
    input_x["range"] = range_adapt

    extra_params = {"disable_optimization": True}
    ins = classify([input_x], OpPatternMode.ELEWISE_WITH_BROADCAST, extra_params)
    schedules, tensors = [], []
    for (_input_x,) in ins:
        with tbe.compute():
            shape = shape_util.variable_shape([_input_x])[0]
            data = tvm.placeholder(shape, name="data", dtype=input_dtype)
            res = tile_d_compute(data, output_x, multiples_adapt, kernel_name)
            tensors.append([data, res])

        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        schedules.append(sch)

    config = {"print_ir": False, "name": kernel_name, "tensor_list": tensors}

    tbe.build(schedules, config)

    tbe_context.get_context().add_compile_info("tiling_info", tiling_info)
