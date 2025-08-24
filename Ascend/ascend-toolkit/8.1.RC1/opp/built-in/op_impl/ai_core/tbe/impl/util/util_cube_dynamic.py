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
public function for cube dynamic
"""

from __future__ import absolute_import
import copy
import math
import warnings

from impl.util.platform_adapter import error_manager_cube as err_man
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
import impl.util.util_deconv_comm as comm
from tbe.common.platform import get_soc_spec
from tbe.common.platform import platform_info
from tbe.dsl.compute.conv2d_backprop_input_compute_util import CalL1Size
from te.platform import cce_params


# the bytes length of several dtype
BIT_RATIO_DICT = {"int32": 4, "float32": 4, "float16": 2,
                  "uint8": 1, "int8": 1, "uint4": 0.5, "int4": 0.5}
BLOCK_K_DICT = {"float16": 16, "float32": 8, "int8": 32, "uint8": 32, "int32": 8, "uint4": 64, "int4": 64,
                "bfloat16": 16}
N_DIM = 0
C_DIM = 1
H_DIM = 2
W_DIM = 3
H_DIM_2D = 0
W_DIM_2D = 1
RANGE_DIM_LEN = 2
FORMAT_HW_DIM = 2
FORMAT_NCHW_DIM = 4
FORMAT_NC1HWC0_DIM = 5
FIX_FLAG = 0
DYNAMIC_FLAG = -1
UNKNOWN_FLAG = -2
UNKNOWN_SHAPE = [-2]
BINARY_RANGE = ((1, None), (1, None), (1, None), (1, None))
DIM_TO_NAME = {0: "N", 2: "H", 3: "W"}
INPUT_SIZE_DEFAULT_SHAPE = [4]
_K_MIN_RANGE = 1
_K_MAX_RANGE = 4096
_K_DIM_SIZE = 5
MAX_N_FUZZ_BUILD = 2**31 - 1
MAX_HW_FUZZ_BUILD = 4096
GRADE_N = (0, 1, 3, 7, 15, 31, MAX_N_FUZZ_BUILD)
GRADE_H = (0, 3, 15, 31, 63, 127, 191, 255, 511, 767, 1023, 4096)
GRADE_W = (0, 3, 15, 31, 63, 127, 191, 255, 511, 767, 1023, 4096)
GRADE_D = (0, 3, 15, 31, 63, 127, 191, 255, 511, 767, 1023, 4096)
DYNAMIC_FMAP_W_MIN = 1
DYNAMIC_FMAP_W_MAX = 4096
CUBE_SIZE = 16
FP16_M = 16
FP16_K = 16
FP16_N = 16
FP16_SIZE = 2
MAPPING_FROMAT_TO_FIX_DIMS = {"NCHW": 4, "NHWC": 4, "NC1HWC0": 5}
DY_DX_ORI_FORMAT_SUPPORT_LIST = ["NCHW", "NHWC"]
OUT_MIN_LOWER = 1


def get_idx_shape_from_format(obj_format, obj_shape, dst_format="NDHWC"):
    """
    get index and shape from ele_format
    """
    idx_list = []
    dst_shape = []
    for dim in dst_format:
        dim_idx = obj_format.find(dim)
        idx_list.append(dim_idx)
        dim_shape = obj_shape[dim_idx]
        dst_shape.append(dim_shape)
    return idx_list, dst_shape


def ceil_div(x_1, x_2):
    """
    ceil divide for inputs
    """

    if x_1 is None:
        return x_1
    if x_2 == 0:
        err_man.raise_err_specific("conv2d", "division by zero")
    return (x_1 + x_2 - 1) // x_2


def align(x_1, x_2):
    """
    align up for  inputs
    """

    return ceil_div(x_1, x_2) * x_2


def lcm(x_1, x_2):
    """
    get the least common multiple
    """

    return (x_1 * x_2) // math.gcd(x_1, x_2)


def pos_from_format(ele_format):
    """
    get value from ele_format
    """
    pos_n = ele_format.find('N')
    pos_c = ele_format.find('C')
    pos_h = ele_format.find('H')
    pos_w = ele_format.find('W')
    pos = (pos_n, pos_c, pos_h, pos_w)
    return pos


def set_default_para():
    """
    set default parameter value
    """
    default_para = {}
    default_para["res_dtype"] = "float16"
    default_para["input_size"] = {"ori_shape": INPUT_SIZE_DEFAULT_SHAPE}
    return default_para


def modify_w_range_max(fmap: dict, infilter: dict, dedy: dict, strides: list, dilations: list,
                       data_format: str, op_type: str, is_gragh_mode: bool = False) -> dict:
    """
    modify w range max value
    """
    fmap_w = fmap.get("ori_shape")[fmap.get("ori_format").find("W")]
    filter_h = infilter.get("ori_shape")[infilter.get("ori_format").find("H")]
    filter_w = infilter.get("ori_shape")[infilter.get("ori_format").find("W")]
    dilation_h = dilations[data_format.find("H")]
    dilation_w = dilations[data_format.find("W")]
    filter_h_dilation = (filter_h - 1) * dilation_h + 1
    filter_w_dilation = (filter_w - 1) * dilation_w + 1
    dedy_h_max = dedy.get("ori_range")[dedy.get("ori_format").find("H")][1]
    dedy_h = dedy.get("ori_shape")[dedy.get("ori_format").find("H")]
    dedy_w = dedy.get("ori_shape")[dedy.get("ori_format").find("W")]
    stride_h = strides[data_format.find("H")]
    stride_w = strides[data_format.find("W")]
    out_backprop_dtype = dedy.get("dtype").lower()
    filter_dtype = infilter.get("dtype").lower()
    if is_gragh_mode:
        fmap_w = fmap.get("ori_range")[fmap.get("ori_format").find("W")][1]
        dedy_h = dedy.get("ori_range")[dedy.get("ori_format").find("H")][1]
        dedy_w = dedy.get("ori_range")[dedy.get("ori_format").find("W")][1]

    c0_size = cce_params.C0_SIZE
    c0_size_k = cce_params.CUBE_MKN[filter_dtype]['mac'][1]
    while dedy_h_max >= dedy_h:
        h_value_max = min(filter_h_dilation + 1, dedy_h_max * stride_h)
        l1_size = get_soc_spec("L1_SIZE")
        filter_l1_size = FP16_SIZE * FP16_N * FP16_K * filter_w * filter_h
        a_l1_size = l1_size - filter_l1_size
        w_value = a_l1_size // (h_value_max * c0_size_k * BIT_RATIO_DICT.get(out_backprop_dtype))
        w_max = w_value // stride_w
        is_single_point = False
        if w_max >= dedy_w:
            return {"dedy_h_max": dedy_h_max, "w_max": w_max,
                    "is_single_point": is_single_point}

        graph_mode_invalid = is_gragh_mode and dedy.get("ori_range")[dedy.get("ori_format").find("W")][1] != \
                                               dedy.get("ori_range")[dedy.get("ori_format").find("W")][0]
        if graph_mode_invalid:
            break
        if fmap_w % c0_size == 0:
            is_single_point = True
            h_value_max = min(filter_h_dilation, dedy_h_max * stride_h)
            w_value = a_l1_size // (h_value_max * c0_size_k * BIT_RATIO_DICT.get(out_backprop_dtype))
            w_max = w_value // stride_w
            if w_max >= dedy_w:
                w_max = dedy_w
                return {"dedy_h_max": dedy_h_max, "w_max": w_max,
                        "is_single_point": is_single_point}
            dedy_h_max = dedy_h_max - 1
            continue
        dedy_h_max = dedy_h_max - 1
    warnings.warn("{}, Input shape is too large, the minimum tiling may exceed L1_Buffer".format(op_type))
    return {"is_exceed_l1": True}


def modify_dy_w_range_max_opti(dedy: dict, infilter: dict, param_list: list,
                               op_type: str, is_gragh_mode: bool = False) -> bool:
    """
    modify dy_w range max opti
    """
    [strides, _, _, data_format] = param_list
    is_pass_check = True
    param_index_dict = {"conv2d_backprop_input": 2, "depthwise_conv2d_backprop_input": 2,
                        "conv2d_transpose": 1, "deconvolution": 0, "avg_pool_grad": 1}
    param_idx = param_index_dict.get(op_type, 0)
    filter_shape = infilter.get("ori_shape")
    filter_format = infilter.get("ori_format")
    _, _, pos_filter_h, pos_filter_w = pos_from_format(filter_format)
    if filter_shape[pos_filter_h] != 1 or filter_shape[pos_filter_w] != 1:
        return is_pass_check, dedy
    dedy_format = dedy.get("ori_format")
    dedy_shape = dedy.get("ori_shape")
    dedy_range = dedy.get("ori_range")
    _, _, _, pos_w = pos_from_format(dedy_format)
    _, _, pos_attr_h, pos_attr_w = pos_from_format(data_format)
    w_max = _K_MAX_RANGE // (strides[pos_attr_h] * strides[pos_attr_w])
    if dedy_shape[pos_w] > DYNAMIC_FLAG and w_max < dedy_shape[pos_w]:
        warnings.warn("{}, w of dedy is too large for opti scheme ,w can't larger than {}"
                      "actually is {}".format(op_type, str(w_max), str(dedy_shape[pos_w])))
        is_pass_check = False
        return is_pass_check, [{"result": "UNSUPPORTED", "reason": {"param_index": [param_idx],
                                "type": ["lower_limit"]}}]
    if w_max < dedy_range[pos_w][1]:
        if is_gragh_mode:
            warnings.warn("w_range_max of dedy is too large for opti scheme"
                        "w_range_max should not larger than {}, actually is {}"
                        .format(str(w_max), str(dedy_range[pos_w][1])))
            is_pass_check = False
            return is_pass_check, [{"result": "UNSUPPORTED", "reason": {"param_index": [param_idx],
                                    "type": ["upper_limit"]}}]
        warnings.warn("w_range_max of dedy is too large for opti scheme"
                      "w_range_max will be modified to {}, actually is {}"
                      .format(str(w_max), str(dedy_range[pos_w][1])))
        dedy_range[pos_w][1] = w_max
    return is_pass_check, dedy


def generalize_shape_and_range_for_binary_const(shape):
    """
    generalize shape and range for binary const(python, static shape)
    """
    shape_len = len(shape)
    return [[1, None]] * shape_len, [-1] * shape_len


def correct_conv2d_backprop_range_start(fmap_dict : dict, filter_dict : dict,  dilations : list,
    pads : list, data_format : str) -> dict:
    """
    correct fmap lower range for conv2d & conv3d.
    """
    input_range = list(map(list, fmap_dict.get("ori_range")))
    input_format = fmap_dict.get("ori_format")
    filter_shape = filter_dict.get("ori_shape")
    filter_format = filter_dict.get("ori_format")
    _, _, pos_h, pos_w = pos_from_format(input_format)
    _, _, pos_filter_h, pos_filter_w = pos_from_format(filter_format)
    _, _, pos_attr_h, pos_attr_w = pos_from_format(data_format)

    # support 3d
    if data_format.find("D") > 0:
        pad_head, pad_back, pad_top, pad_bottom, pad_left, pad_right = pads
        pos_d = input_format.find("D")
        pos_filter_d = filter_format.find("D")
        pos_attr_d = data_format.find("D")
        kd_dilation = dilations[pos_attr_d] * (filter_shape[pos_filter_d] - 1) + 1
        low_d = kd_dilation - pad_head - pad_back
        input_range[pos_d][0] = max(input_range[pos_d][0], low_d)
    else:
        pad_top, pad_bottom, pad_left, pad_right = pads

    kh_dilation = dilations[pos_attr_h] * (filter_shape[pos_filter_h] - 1) + 1
    kw_dilation = dilations[pos_attr_w] * (filter_shape[pos_filter_w] - 1) + 1
    low_h = kh_dilation - pad_top - pad_bottom
    low_w = kw_dilation - pad_left - pad_right
    input_range[pos_h][0] = max(input_range[pos_h][0], low_h)
    input_range[pos_w][0] = max(input_range[pos_w][0], low_w)

    fmap_dict["ori_range"] = input_range

    return fmap_dict


def generalize_shape_and_range(in_format, shape, ndims: int=None):
    """
    generalize shape and range for te_fusion(c++)

    ndims(int): with this param, this function can also generalize tensor when its in_format is ND and shape is [-2].
    """
    shape_len = len(shape)
    if list(shape) == UNKNOWN_SHAPE:
        shape_len = ndims if ndims else MAPPING_FROMAT_TO_FIX_DIMS.get(in_format, ndims)
        if shape_len is None:
            err_man.raise_err_specific("conv2d", "This format:{0} is not supported in genaralize_shape stage."\
                                       .format(in_format))
    return [[1, -1]] * shape_len, [-1] * shape_len


def generalize_shape_and_range_inplace(dict_tensor):
    generalized_range, generalized_shape = generalize_shape_and_range_for_binary_const(dict_tensor.get("shape"))
    dict_tensor["shape"] = generalized_shape
    _, generalized_shape = generalize_shape_and_range_for_binary_const(dict_tensor.get("ori_shape"))
    dict_tensor["ori_shape"] = generalized_shape
    dict_tensor["range"] = generalized_range


def get_single_range(grade : list, shape_value : int, op_type : str) -> list:
    """
    gen single range
    """

    if shape_value > grade[-1]:
        err_man.raise_err_specific_user(op_type,
                                        "input value {} is out of the range of {}"\
                                        .format(str(shape_value), str(grade[-1])))
    if shape_value <= 0:
        err_man.raise_err_specific_user(op_type,
                                        "ori_shape should be greater than 0, which is {}".format(shape_value))
    if shape_value == grade[-1]:
        return [grade[-2] + 1, grade[-1]]
    low, high = 1, 1
    for point in grade:
        if shape_value > point:
            low = point + 1
        if shape_value <= point:
            high = point
            break
    return [low, high]


def gen_conv_shape_range(input_d : dict, op_type : str, is_graph_mode=False) -> dict:
    """
    gen conv shape range
    """

    if is_graph_mode:
        return input_d
    input_shape = input_d.get("ori_shape")
    input_format = input_d.get("ori_format")
    pos_n, _, pos_h, pos_w = pos_from_format(input_format)
    # deal 3d
    pos_d = input_format.find("D") if len(input_format) == 5 else None
    input_range = [(input_shape[i], input_shape[i]) for i in range(len(input_shape))]
    grade_map = {pos_n: GRADE_N, pos_h: GRADE_H, pos_w: GRADE_W, pos_d: GRADE_D}
    for key, value in grade_map.items():
        if key is not None:
            new_range = get_single_range(value, input_shape[key], op_type)
            input_range[key] = new_range
    input_d["ori_range"] = input_range
    return input_d


def check_graph_mode(tensor: dict) -> bool:
    """
    check graph mode
    """
    # check graph mode or single mode in fuzzy compile
    if (list(tensor.get("ori_shape", [])) == UNKNOWN_SHAPE or DYNAMIC_FLAG in tensor.get("ori_shape") and
        "ori_range" in tensor.keys()):
        return True
    return False


def check_dynamic_mode(tensor: dict) -> bool:
    """
    check dynamic or not
    """
    if list(tensor.get("ori_shape")) == UNKNOWN_SHAPE:
        return UNKNOWN_FLAG
    if DYNAMIC_FLAG in tensor.get("ori_shape"):
        return DYNAMIC_FLAG
    return FIX_FLAG


def check_input_output_format_and_shape(input_list, op_type):
    # check dy and dx's ori_format, shape length
    for input_ele in input_list:
        if input_ele.get("ori_format") not in DY_DX_ORI_FORMAT_SUPPORT_LIST:
            warnings.warn("ori_format of {} is {}, only support {}".format(op_type,
                          str(input_ele.get("ori_format")), str(DY_DX_ORI_FORMAT_SUPPORT_LIST)))
            return [{"result": "UNSUPPORTED"}]
        if not isinstance(input_ele.get("ori_shape"), (list, tuple)) or \
            len(input_ele.get("ori_shape")) != FORMAT_NCHW_DIM:
            warnings.warn("ori_shape of {} is {}, only support {}".format(
                op_type, str(input_ele.get("ori_shape")), str(FORMAT_NCHW_DIM)))
            return [{"result": "UNSUPPORTED"}]
    return []


def _get_nchw_shape(input_x, dynamic_mode=FIX_FLAG):
    """
    get the n,c,h,w dims of the shape
    """
    input_format = input_x.get("ori_format")
    pos_n, pos_c, pos_h, pos_w = pos_from_format(input_format)
    if dynamic_mode == FIX_FLAG:
        input_shape = input_x.get("ori_shape")
        return [input_shape[pos_n], input_shape[pos_c], input_shape[pos_h], input_shape[pos_w]]
    else:
        input_range = input_x.get("ori_range")
        return [input_range[pos_n][1], input_range[pos_c][1], input_range[pos_h][1], input_range[pos_w][1]]


def check_l1_size(input_list, attr_list, dynamic_flag):
    # default size
    def _cal_size_binary(cal_size, tiling_dict):
        flag_dict = {"binary_flag": True}
        return cal_size.cal_l1_size(flag_dict, tiling_dict)

    def _cal_conv2d_split_w_size_binary(cal_size, tiling_dict):
        flag_dict = {"binary_flag": True, "conv2d_split_w_flag": True}
        return cal_size.cal_l1_size(flag_dict, tiling_dict)

    input_dict = {}
    dy, filters, bias, dx = input_list
    strides, pads, dilations, data_format, output_padding = attr_list
    n_pos, c_pos, h_pos, w_pos = pos_from_format(data_format)
    dy_shape_nchw = _get_nchw_shape(dy, dynamic_flag)
    filter_shape_nchw = _get_nchw_shape(filters)
    dx_shape_nchw = _get_nchw_shape(dx, dynamic_flag)
    input_dict["filter_shape_nchw"] = filter_shape_nchw
    dy_c0 = tbe_platform.CUBE_MKN.get(dy.get("dtype").lower())["mac"][1]
    dy_n, dy_c, dy_h, dy_w = dy_shape_nchw
    input_dict["dy_shape_nc1hwc0"] = [dy_n, ceil_div(dy_c, dy_c0), dy_h, dy_w, dy_c0]
    input_dict["dx_shape_nchw"] = dx_shape_nchw
    if bias is not None:
        input_dict["bias_tensor"] = bias
        input_dict["bias_dtype"] = bias.get("dtype").lower()
    input_dict["strides"] = strides[n_pos], strides[c_pos], strides[h_pos], strides[w_pos]
    input_dict["padding"] = pads[n_pos], pads[c_pos], pads[h_pos], pads[w_pos]
    input_dict["dilations"] = dilations[n_pos], dilations[c_pos], dilations[h_pos], dilations[w_pos]
    if output_padding:
        input_dict["output_padding"] = output_padding[n_pos], output_padding[c_pos], \
            output_padding[h_pos], output_padding[w_pos]
    input_dict["dy_dtype"] = dy.get("dtype").lower()
    input_dict["filter_dtype"] = filters.get("dtype").lower()
    m0, k_block_size, n0 = tbe_platform.CUBE_MKN.get(filters.get("dtype").lower())["mac"]
    k_al1 = 2 if filters.get("dtype").lower() == "float32" else 1
    k_bl1 = k_al1
    tiling_dict = {"m_al1": 1, "m": 1, "m0": m0, "n_bl1": 1, "n": 1, "n0": n0, "k_al1": k_al1,
                   "k_bl1": k_bl1, "k_block_size": k_block_size, "db_al1": 1, "db_bl1": 1, "db_bias_l1": 1}

    cal_size = CalL1Size(input_dict)
    split_h_al1_size, split_h_bl1_size, split_h_bias_l1_size = _cal_size_binary(cal_size, tiling_dict)
    split_w_al1_size, split_w_bl1_size, split_w_bias_l1_size = _cal_conv2d_split_w_size_binary(cal_size, tiling_dict)
    min_load_size = min((split_h_al1_size + split_h_bl1_size + split_h_bias_l1_size),
                  (split_w_al1_size + split_w_bl1_size + split_w_bias_l1_size))
    l1_size = platform_info.get_soc_spec("L1_SIZE")
    if min_load_size > l1_size:
        warnings.warn("Current load size is {}, L1 size is {}".format(min_load_size, l1_size))
        return [{"result": "UNSUPPORTED"}]
    return []


def check_generalize_config(generalize_config, op_type):
    """
    check generalize config is valid
    """
    if generalize_config is not None and generalize_config.get("mode") == "keep_rank":
        return True
    warnings.warn("the generalize_config of {} is not keep_rank".format(op_type))
    return False


def check_fuzz_input_output(input_x, dilations, op_type):
    """
    check the dtype and shape of input_x
    """
    support_format = ["NCHW", "NHWC"]
    for input_mem in input_x:
        tensor_format = input_mem.get("ori_format")
        ori_shape = input_mem.get("ori_shape")

        if tensor_format not in support_format:
            warnings.warn("ori_format of {} is {}, only support {}".format(op_type,
                          str(tensor_format), str(support_format)))
            return False
        if not ori_shape or len(ori_shape) != FORMAT_NCHW_DIM:
            warnings.warn("the ori_shape of {} is {}, only support {}d".format(op_type,
                           str(ori_shape), FORMAT_NCHW_DIM))
            return False


    return True


def check_fuzz_n_dim(input_x, dynamic_flag, op_type):
    """
    check the n dim of input x
    """
    tensor_format = input_x.get("ori_format")
    n_pos = tensor_format.find("N")
    input_index = 1 if op_type == "conv2d_transpose" else 0
    if dynamic_flag == DYNAMIC_FLAG:
        ori_range = input_x.get("ori_range")
        if ori_range[n_pos][0] > MAX_N_FUZZ_BUILD:
            return [{"result": "UNSUPPORTED", "reason": {"param_index": [input_index], "type": ["lower_limit"]}}]
        if ori_range[n_pos][1] is None or ori_range[n_pos][1] > MAX_N_FUZZ_BUILD:
            return [{"result": "UNSUPPORTED", "reason": {"param_index": [input_index], "type": ["upper_limit"]}}]
    else:
        ori_shape = input_x.get("ori_shape")
        if ori_shape[n_pos] > MAX_N_FUZZ_BUILD:
            return [{"result": "UNSUPPORTED"}]
    return []


def _get_nchw_dims(input_x, dynamic_mode=FIX_FLAG):
    """
    get the n,c,h,w dims of the shape
    """
    input_format = input_x.get("ori_format")
    pos_n, pos_c, pos_h, pos_w = pos_from_format(input_format)
    if dynamic_mode == FIX_FLAG:
        input_shape = input_x.get("ori_shape")
        return [input_shape[pos_n], input_shape[pos_c], input_shape[pos_h], input_shape[pos_w]]
    else:
        input_range = input_x.get("ori_range")
        return [input_range[pos_n], input_range[pos_c], input_range[pos_h], input_range[pos_w]]


def _cal_dx_pads(dedy, dedx, strides, data_format):
    """
    cal the dx pads in fix shape mode
    """
    _, _, dedx_h, dedx_w = _get_nchw_dims(dedx)
    _, _, dedy_h, dedy_w = _get_nchw_dims(dedy)
    stride_h, stride_w = strides[data_format.find("H")], strides[data_format.find("W")]
    # when padding is same
    if ceil_div(dedx_h, stride_h) == dedy_h and ceil_div(dedx_w, stride_w) == dedy_w:
        return "SAME"
    return "VALID"


def _cal_dy_hw_max(input_filter, strides, data_format):
    """
    cal the hw max in dx
    """
    if len(strides) == FORMAT_NCHW_DIM:
        stride_h, stride_w = strides[data_format.find("H")], strides[data_format.find("W")]
    else:
        stride_h, stride_w = strides
    _, _, filter_h, filter_w = _get_nchw_dims(input_filter)

    h_max = ceil_div(MAX_HW_FUZZ_BUILD - filter_h, stride_h)
    w_max = ceil_div(MAX_HW_FUZZ_BUILD - filter_w, stride_w)
    if filter_h == 1 and filter_w == 1:
        w_max = min(w_max, _K_MAX_RANGE // stride_h * stride_w)
    else:
        l1_size = get_soc_spec("L1_SIZE")
        filter_dtype = input_filter.get("dtype").lower()
        c0_size = cce_params.C0_SIZE
        c0_size_k = cce_params.CUBE_MKN[filter_dtype]['mac'][1]
        b_l1_size = filter_h * filter_w * c0_size * c0_size_k * comm.BIT_RATIO_DICT.get(filter_dtype)
        al1_h_value = filter_h + 1
        w_max_l1 = (l1_size - b_l1_size) // \
                   (al1_h_value * c0_size_k * comm.BIT_RATIO_DICT.get(filter_dtype)) // stride_w
        w_max = min(w_max_l1, w_max)
    return h_max, w_max


def check_fuzz_hw_dim(input_tensor, strides, data_format, dynamic_flag, op_type):
    """
    in dynamic mode, check the range is valid
    """
    dedy, _, input_filter = input_tensor
    h_max, w_max = _cal_dy_hw_max(input_filter, strides, data_format)
    tensor_format = dedy.get("ori_format")
    h_pos, w_pos = tensor_format.find("H"), tensor_format.find("W")
    input_index = 1 if op_type == "conv2d_transpose" else 0
    if dynamic_flag == DYNAMIC_FLAG:
        _, _, dedy_h_range, dedy_w_range = _get_nchw_dims(dedy, DYNAMIC_FLAG)
        # check h dim
        if dedy_h_range[0] > h_max:
            return [{"result": "UNSUPPORTED", "reason": {"param_index": [input_index], "type": ["lower_limit"]}}]
        if dedy_h_range[1] is None or dedy_h_range[1] > h_max:
            return [{"result": "UNSUPPORTED", "reason": {"param_index": [input_index], "type": ["upper_limit"]}}]
        if dedy_w_range[0] > w_max:
            return [{"result": "UNSUPPORTED", "reason": {"param_index": [input_index], "type": ["lower_limit"]}}]
        if dedy_w_range[1] is None or dedy_w_range[1] > w_max:
            return [{"result": "UNSUPPORTED", "reason": {"param_index": [input_index], "type": ["upper_limit"]}}]
    else:
        _, _, dedy_h, dedy_w = _get_nchw_dims(dedy)
        if dedy_h > h_max or dedy_w > w_max:
            return [{"result": "UNSUPPORTED"}]
        dedy = gen_conv_shape_range(dedy, op_type, False)
        dedy["ori_range"][h_pos][1] = min(h_max, dedy["ori_range"][h_pos][1])
        dedy["ori_range"][w_pos][1] = min(w_max, dedy["ori_range"][w_pos][1])
    return []


def cal_dedx_range(input_tensor, strides, data_format):
    """
    cal the output range of dx
    """
    input_size_range = [[1, None], [1, None], [1, None], [1, None]]
    pos_n, pos_c, pos_h, pos_w = pos_from_format(data_format)
    dedy, dedx, input_filter = input_tensor
    _, _, filter_h, filter_w = _get_nchw_dims(input_filter)
    _, dedx_c, _, _ = _get_nchw_dims(dedx)
    dedy_n_range, _, dedy_h_range, dedy_w_range = _get_nchw_dims(dedy, DYNAMIC_FLAG)
    stride_h, stride_w = strides[pos_h], strides[pos_w]
    input_size_range[pos_n] = dedy_n_range
    input_size_range[pos_c][0] = input_size_range[pos_c][1] = dedx_c
    pad_mode = _cal_dx_pads(dedy, dedx, strides, data_format)
    if pad_mode == "SAME":
        input_size_range[pos_h][0] = dedy_h_range[0]*stride_h + 1 - stride_h
        input_size_range[pos_w][0] = dedy_w_range[0]*stride_w + 1 - stride_w
        input_size_range[pos_h][1] = dedy_h_range[1]*stride_h + 1 - 1
        input_size_range[pos_w][1] = dedy_w_range[1]*stride_w + 1 - 1
    else:
        input_size_range[pos_h][0] = dedy_h_range[0]*stride_h + filter_h - stride_h
        input_size_range[pos_w][0] = dedy_w_range[0]*stride_w + filter_w - stride_w
        input_size_range[pos_h][1] = dedy_h_range[1]*stride_h + filter_h - 1
        input_size_range[pos_w][1] = dedy_w_range[1]*stride_w + filter_w - 1
    return input_size_range


def check_graph_range(tensor: dict, op_type: str, is_graph_mode=False) -> str:
    """
    check wether range of N dim exceed 2**31 -1 or H/W dim exceed 4096
    """
    if not is_graph_mode:
        return ""
    n_dim = tensor.get("ori_format").find("N")
    h_dim = tensor.get("ori_format").find("H")
    w_dim = tensor.get("ori_format").find("W")
    ori_range = tensor.get("ori_range")
    lower_limit_flag = (len(ori_range) != FORMAT_NCHW_DIM or ori_range[n_dim][0] > MAX_N_FUZZ_BUILD or
                        ori_range[h_dim][0] > _K_MAX_RANGE or ori_range[w_dim][0] > _K_MAX_RANGE)
    if lower_limit_flag:
        warnings.warn("{}, if lower range exceeds 4096(H/W dim) or {}(N dim) or len(ori_range) != {}, "
                        "it's lower limit".format(op_type, MAX_N_FUZZ_BUILD, FORMAT_NCHW_DIM))
        return "lower_limit"
    range_none_flag = (None in list(zip(*ori_range))[1])
    upper_limit_flag = (range_none_flag or ori_range[n_dim][1] > MAX_N_FUZZ_BUILD or
                        ori_range[h_dim][1] > _K_MAX_RANGE or ori_range[w_dim][1] > _K_MAX_RANGE)
    if upper_limit_flag:
        warnings.warn(f"In {op_type}, if upper range exceeds 4096(H/W dim) or {MAX_N_FUZZ_BUILD}(N dim) or is None")
        return "upper_limit"
    return ""


def get_input(x_out, filter_dilation, pads, stride):
    """
    get fmap H/W dim according dedy
    """
    if DYNAMIC_FLAG in pads:
        input_low = stride * (x_out - 1) + 1
        input_high = stride * x_out
    else:
        input_low = (x_out - 1) * stride + filter_dilation - pads[0] - pads[1]
        input_high = (x_out - 1) * stride + filter_dilation - pads[0] - pads[1] + stride - 1
    return input_low, input_high


def calc_max_fmap_w(x: dict, out_backprop: dict, y: dict, param_list: list, graph_flag: bool = False) -> list:
    """
    modify max grade point, when exceed L1 size

    Parameters
    ----------
    x: dict with keys(ori_shape, ori_format, shape, format, dtype, range)
        input feature map tensor.

    out_backprop: dict with keys(ori_shape, ori_format, shape, format, dtype, range)
        input weight tensor.

    y: dict with keys(ori_shape, ori_format, shape, format, dtype, range)
        output tensor, dtype must be assigned.

    strides: tuple/list of 4 integers
             filter move stride

    pads: tuple/list of 4 integers
          [pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 4 integers
        filter expand size of dilated conv2d_backprop_filter. Default to (1, 1, 1, 1).

    data_format: str
            An optional string from: "NHWC", "NCHW". Defaults to "NHWC".
            Specify the data format of the input and output data.

    graph_flag: bool.

    Returns
    -------
    modified dedx's w_range
    """
    strides, pads, dilations = param_list
    _, x_nchw = get_idx_shape_from_format(x.get("ori_format"), x.get("ori_shape"), "NCHW")
    _, dedy_nchw = get_idx_shape_from_format(out_backprop.get("ori_format"), out_backprop.get("ori_shape"), "NCHW")
    # depthwise_dw filter format may is HWCK, only get h/w dim shape
    _, dedw_hw = get_idx_shape_from_format(y.get("ori_format"), y.get("ori_shape"), "HW")
    _, x_range_nchw = get_idx_shape_from_format(x.get("ori_format"), x.get("ori_range"), "NCHW")

    x_h_range, x_w_range = x_range_nchw[2:]
    pad_up, pad_down, pad_left, pad_right = pads
    stride_h, stride_w = strides
    dilation_h, dilation_w = dilations[H_DIM], dilations[W_DIM]
    filter_h_dilation = (dedw_hw[0] - 1) * dilation_h + 1
    filter_w_dilation = (dedw_hw[1] - 1) * dilation_w + 1

    upper_fmap_h_padding = x_h_range[1] + pad_up + pad_down
    lower_fmap_h_padding = x_h_range[0] + pad_up + pad_down
    is_conv1d_situation = upper_fmap_h_padding == 1 and lower_fmap_h_padding == 1 and \
                          filter_h_dilation == 1 and stride_h == 1
    l1_size = tbe_platform.get_soc_spec("L1_SIZE")  # L1 size
    # CUBE_SIZE * CUBE_SIZE * 2 is al1_min_byte
    bl1_min_byte = l1_size - CUBE_SIZE * CUBE_SIZE * 2
    kl1_min = bl1_min_byte // ((filter_h_dilation + stride_h) * CUBE_SIZE * 2)
    kl1_min_devided_sixteen = bl1_min_byte // (filter_h_dilation * CUBE_SIZE * 2)
    max_dedy_devided_sixteen = (kl1_min_devided_sixteen + pad_left + \
                                pad_right - filter_w_dilation) // stride_w + 1
    dedx_w = x_nchw[W_DIM]
    real_dedy = dedy_nchw[W_DIM]
    dy_w_range = [real_dedy, real_dedy]
    if graph_flag:
        _, dedy_range_nchw = get_idx_shape_from_format(out_backprop.get("ori_format"),
                                                out_backprop.get("ori_range"), "NCHW")
        dy_w_range = dedy_range_nchw[W_DIM]
        dedx_w = x_w_range[1]
        real_dedy = dy_w_range[0]
    test_single_mode = (not graph_flag or (dy_w_range[0] == dy_w_range[1]))
    upper_limit_flag = (x_w_range[1] > kl1_min)
    lower_limit_flag = (x_w_range[0] > kl1_min)
    if not is_conv1d_situation:
        if kl1_min >= x_w_range[1]:
            pass
        elif dedx_w <= kl1_min:
            x_w_range = (x_w_range[0], kl1_min)
        # h_dim has no range
        elif test_single_mode and (real_dedy % CUBE_SIZE == 0 and real_dedy <= max_dedy_devided_sixteen):
            x_w_range_low, x_w_range_high = get_input(real_dedy, filter_w_dilation, pads[2:], stride_w)
            x_w_range_low = max(x_w_range_low, x_w_range[0])
            x_w_range_high = min(x_w_range_high, x_w_range[1])
            x_w_range = (x_w_range_low, x_w_range_high)
        else:
            upper_res_flag = graph_flag and upper_limit_flag and not lower_limit_flag
            if upper_res_flag:
                return False, "upper_limit"
            return False, "lower_limit"
    return True, x_w_range


def check_modify_w_range(input_list, param_list, op_type, dynamic_flag):
    """
    check L1 range
    """
    [input_grad, filter_grad, out_backprop] = input_list
    [strides, pads, dilations, data_format] = param_list
    _, [_, dedy_w] = get_idx_shape_from_format(out_backprop.get("ori_format"),
                                                    out_backprop.get("ori_shape"), "HW")
    _, [_, fmap_w] = get_idx_shape_from_format(input_grad.get("ori_format"),
                                                    input_grad.get("ori_shape"), "HW")
    _, [filter_h, filter_w] = get_idx_shape_from_format(filter_grad.get("ori_format"),
                                                        filter_grad.get("ori_shape"), "HW")

    _, [_, strides_w] = get_idx_shape_from_format(data_format, strides, "HW")
    _, [_, dedy_range_w] = get_idx_shape_from_format(out_backprop.get("ori_format"),
                                                     out_backprop.get("ori_range"), "HW")
    filter_h_dilations = (filter_h - 1) * dilations[data_format.find('H')] + 1
    filter_w_dilations = (filter_w - 1) * dilations[data_format.find('W')] + 1
    filter_dtype = input_grad.get("dtype")
    dedy_dtype = input_grad.get("dtype")
    if dynamic_flag:
        _, [_, fmap_range_w] = get_idx_shape_from_format(input_grad.get("ori_format"),
                                                         input_grad.get("ori_range"), "HW")
        dedy_w = dedy_range_w[0]
        fmap_w = fmap_range_w[0]
    else:
        paras = {"data_format": data_format, "pads": pads, "strides": strides}
        cube_para = CubeParaProcess(paras)
        new_pads = cube_para.correct_pads(input_grad, out_backprop, filter_grad)
        fmap_range_w_low, fmap_range_w_high = get_input(dedy_w, filter_w_dilations, new_pads[2:], strides_w)
        fmap_range_w = [min(fmap_range_w_low, fmap_w), max(fmap_range_w_high, fmap_w)]
    bl1_size = filter_h * filter_w * BLOCK_K_DICT.get(filter_dtype) * FP16_N * BIT_RATIO_DICT.get(filter_dtype)
    l1_size = get_soc_spec("L1_SIZE")
    al1_max_size = l1_size - bl1_size
    w_max = al1_max_size // (BIT_RATIO_DICT.get(dedy_dtype) * BLOCK_K_DICT.get(dedy_dtype) *
                             (filter_h_dilations + 1) * strides_w)
    # w_dim only support one point scene
    fmap_w_static = (not dynamic_flag or fmap_range_w[0] == fmap_range_w[1]) and fmap_w % FP16_M == 0 and w_max < dedy_w
    if fmap_w_static:
        w_max = al1_max_size // (BIT_RATIO_DICT.get(dedy_dtype) * BLOCK_K_DICT.get(dedy_dtype)
                                 * filter_h_dilations * strides_w)
    supports = "no_limit"
    if w_max < dedy_w:
        if not dynamic_flag:
            warnings.warn(f"In {op_type}, the shape of inputs can not be supported which will exceed L1.")
            supports = "unsupported"
        else:
            warnings.warn(f'In {op_type}, the lower limit of inputs range can not be supported which will exceed L1.')
            supports = "lower_limit"
    elif w_max < dedy_range_w[1] and dynamic_flag:
        warnings.warn(f"In {op_type}, the upper limit of input range exceed support range.")
        supports = "upper_limit"
    dedy_range_w = (dedy_range_w[0], min(w_max, dedy_range_w[1]))
    fmap_range_w = (fmap_w, fmap_w) if fmap_w_static else fmap_range_w
    return supports, dedy_range_w, fmap_range_w


def check_dynamic_range_lower(tensor_list):
    def _modify_lower(range_in):
        nonlocal zero_flag
        new_range = []
        for x in range_in:
            if len(x) > 0 and x[0] == 0:
                x = list(x)
                x[0] = 1
                zero_flag = True
            new_range.append(tuple(x))
        return tuple(new_range)

    zero_flag = False
    for tensor in tensor_list:
        if tensor.get("range"):
            tensor["range"] = _modify_lower(tensor.get("range"))

        if tensor.get("ori_range"):
            tensor["ori_range"] = _modify_lower(tensor.get("ori_range"))

    return zero_flag


def check_tensor_shape(tensor_dict):
    tensor_list = tensor_dict.get("tensor")
    value_list = tensor_dict.get("value")
    range_list = tensor_dict.get("range")

    for tensor, value, drange in zip(tensor_list, value_list, range_list):
        is_valid = tensor.get("range") and tensor.get("shape")
        if is_valid:
            tensor["range"] = tuple((drange[0], drange[1]) if dim == 0 else tuple(dim_range)
                                    for dim, dim_range in zip(tensor.get("shape"), tensor.get("range")))
        if tensor.get("shape"):
            tensor["shape"] = tuple(value if dim == 0 else dim for dim in tensor.get("shape"))

        is_valid = tensor.get("ori_range") and tensor.get("ori_shape")
        if is_valid:
            tensor["ori_range"] = tuple((drange[0], drange[1]) if dim == 0 else tuple(dim_range)
                                        for dim, dim_range in zip(tensor.get("ori_shape"), tensor.get("ori_range")))
        if tensor.get("ori_shape"):
            tensor["ori_shape"] = tuple(value if dim == 0 else dim for dim in tensor.get("ori_shape"))


def set_shape_and_range(tensor, dim, dim_value, dim_range):
    ori_shape, ori_format, ori_range = tensor.get("ori_shape"), tensor.get("ori_format"), tensor.get("ori_range")
    new_shape = list(ori_shape)
    new_range = list(ori_range) if ori_range else None
    for idx, f in enumerate(ori_format):
        if f == dim:
            new_shape[idx] = dim_value
            if new_range:
                new_range[idx] = tuple(dim_range)
    if new_range:
        tensor["ori_range"] = tuple(new_range)
    tensor["ori_shape"] = tuple(new_shape)
    shape = tensor.get("shape")
    in_format = tensor.get("format").replace("C1", "X").replace("C0", "Y") # replace c1 and c0 to x and y
    in_range = tensor.get("range")
    new_shape, new_range = list(shape), list(in_range)
    for idx, f in enumerate(in_format):
        if f == dim:
            new_shape[idx] = dim_value
            new_range[idx] = tuple(dim_range)
    tensor["range"] = tuple(new_range)
    tensor["shape"] = tuple(new_shape)


def is_empty_tensor_scene(tensor_list):
    for tensor in tensor_list:
        if tensor and tensor.get("shape") and 0 in tensor.get("shape"):
            return True
    return False


def correct_range(fmap, fmap_range, w_ori_shape, strides, dilations, pads, data_format, is_dx=False):
    def _get_output(x_in, k_size, pads, stride, dilations):
        return (x_in + pads[0] + pads[1] - dilations * (k_size - 1) - 1) // stride + 1

    def _update_range(dim_c, dim_idx, pad_idx, lower=None):
        if lower is None:
            lower = w_ori_shape[dim_idx] - pads[pad_idx] - pads[pad_idx + 1]
        upper = fmap_range[dim_idx][1]
        if upper:
            upper = max(upper, lower)
        set_shape_and_range(fmap, dim_c, -1, (lower, upper))

    def _is_special_case():
        # 2ddx or 2ddw no need to check special scene here
        if data_format in ("NCHW", "NHWC"):
            return False

        if is_dx:
            dedx_w_lower = fmap_range[dim_w][0]
            dedx_h_upper = fmap_range[dim_h][1]
            flag = dedx_w_lower <= 1 and dedx_h_upper != 1
        else:
            dedy_w_lower = _get_output(fmap_range[dim_w][0], w_ori_shape[dim_w], (pads[pad_w], pads[pad_w + 1]),
                                       strides[dim_w], dilations[dim_w])
            dedy_h_upper = None
            if fmap_range[dim_h][1]:
                dedy_h_upper = _get_output(fmap_range[dim_h][1], w_ori_shape[dim_h], (pads[pad_h], pads[pad_h + 1]),
                                           strides[dim_h], dilations[dim_h])
            flag = dedy_w_lower <= 1 and dedy_h_upper != 1
        return flag

    dim_d, dim_h, dim_w = None, data_format.find('H'), data_format.find('W')
    pad_h, pad_w = 0, 2
    if 'D' in data_format:
        dim_d = data_format.find('D')
        pad_d, pad_h, pad_w = 0, 2, 4

    if -1 in pads:
        if _is_special_case():
            lower = 2 if is_dx else strides[dim_w] + 1
            _update_range('W', dim_w, pad_w, lower=lower)
    else:
        if _is_special_case():
            lower = strides[dim_w] + w_ori_shape[dim_w] - pads[pad_w] - pads[pad_w + 1]
            if is_dx:
                lower = max(2, w_ori_shape[dim_w] - pads[pad_w] - pads[pad_w + 1])
            _update_range('W', dim_w, pad_w, lower=lower)
        else:
            if _get_output(fmap_range[dim_w][0], w_ori_shape[dim_w], (pads[pad_w], pads[pad_w + 1]),
                           strides[dim_w], dilations[dim_w]) <= 0:
                _update_range('W', dim_w, pad_w)
        if _get_output(fmap_range[dim_h][0], w_ori_shape[dim_h], (pads[pad_h], pads[pad_h + 1]),
                       strides[dim_h], dilations[dim_h]) <= 0:
            _update_range('H', dim_h, pad_h)
        if dim_d and _get_output(fmap_range[dim_d][0], w_ori_shape[dim_d], (pads[pad_d], pads[pad_d + 1]),
                                 strides[dim_d], dilations[dim_d]) <= 0:
            _update_range('D', dim_d, pad_d)


def validate_range(ranges : tuple):
    """
    valadate the range lower index
    """
    res = []
    for item in ranges:
        validated_range = max(item[0], 1), item[1]
        res.append(validated_range)
    return res


def check_binary_flag(fmap_range, weight_shape):
    """
    confirm whether it is in binary mode
    """
    if fmap_range is None:
        return True
    for _, range_value in enumerate(fmap_range):
        if range_value is None or None in range_value:
            return True
    if weight_shape is not None and DYNAMIC_FLAG in weight_shape:
        return True

    return False


def check_supported_mm_ub(input_x1, input_x2, bias, output_z):
    res = True, ""
    support_matmul_ub_to_ub = platform_info.intrinsic_check_support("Intrinsic_matmul_ub_to_ub")
    if not support_matmul_ub_to_ub:
        return res
    dtype = "dtype"
    input_x1_data_type = input_x1.get(dtype)
    input_x2_data_type = input_x2.get(dtype)
    bias_data_type = bias.get(dtype) if bias is not None else None
    output_z_data_type = output_z.get(dtype)

    is_valid_dtype = input_x1_data_type in ["int16", "int8"] and input_x2_data_type in ["int8"] and \
                     (bias_data_type is None or bias_data_type in ["int32"]) and output_z_data_type in ["int32"]

    if not is_valid_dtype:
        reason = "Not support the given input and output data type. input_x1: %s, input_x2: %s, bias: %s"\
                 ", output_z: %s." % (input_x1_data_type, input_x2_data_type, bias_data_type, output_z_data_type)
        res = False, reason
    return res


class CubeParaProcess:
    """
    class of param check and preprocess for dynamic cube ops
    """

    def __init__(self, paras):
        self.paras = paras
        self.groups = paras.get("groups")
        self.strides = paras.get("strides")
        self.pads = paras.get("pads")
        self.dilations = paras.get("dilations")
        self.data_format = paras.get("data_format")
        self.op_type = None
        self.binary_mode = False
        self.fusion_flag = False
        self.valid_paras = {
            "n_min": 1,
            "hw_min": 1,
            "nhw_min": 1,
            "hw_max": 4096,
            "valid_format": {"weights": ("NCHW", "NHWC", "HWCN"),
                             "input": ("NCHW", "NHWC"),
                             "output": ("NCHW", "NHWC")},
            "valid_dtype": ("float16", "int8", "int32", "float32")
        }
        self.dim_valid_dict = {
            N_DIM: (self.valid_paras.get("n_min"), None),
            H_DIM: (self.valid_paras.get("hw_min"), self.valid_paras.get("hw_max")),
            W_DIM: (self.valid_paras.get("hw_min"), self.valid_paras.get("hw_max"))
        }

    def check_support_valid(self, in_shape, filter_shape):
        """
        check whether dynamic shape is supported for cube ops
        """

        if self.groups != 1 and self.op_type not in (
            "conv2d", "conv2d_backprop_input", "depthwise_conv2d_backprop_input", "conv2d_transpose",
            "deconvolution", "avg_pool_grad"):
            err_man.raise_err_specific_user(
                self.op_type, "group != 1 is not supported yet in dynamic")
        if DYNAMIC_FLAG not in (in_shape[N_DIM], in_shape[H_DIM], in_shape[W_DIM]):
            err_man.raise_err_specific_user(
                self.op_type, "need at least one dimension in N/H/W is a variable.")
        if DYNAMIC_FLAG in filter_shape:
            err_man.raise_err_specific_user(
                self.op_type, "dynamic weight is not supported yet.")

    def check_unknown_scene(self, in_shape, out_shape, channel):
        """
        check if is unknown scene
        """
        if list(in_shape) == UNKNOWN_SHAPE and out_shape == [DYNAMIC_FLAG, channel, DYNAMIC_FLAG, DYNAMIC_FLAG]:
            return True
        return False

    def check_dynamic_channel_scene(self, in_shape, out_shape, channel):
        """
        check if valid dynamic channel scene
        """
        if out_shape[C_DIM] == DYNAMIC_FLAG:
            err_man.raise_err_specific_user(
                self.op_type, "out channel does not support -1.")
        if in_shape[C_DIM] == DYNAMIC_FLAG:
            in_shape[C_DIM] = channel

    def check_range_valid(self, shape, dynamic_range, name, in_format):
        """
        check if the range is valid
        """

        if self.binary_mode:
            return

        def _check_range(in_range, dim):
            if in_range:
                if not isinstance(in_range, (tuple, list)):
                    err_man.raise_err_specific_user(self.op_type, "type of range must be tuple or list.")
                valid_lower, valid_upper = self.dim_valid_dict.get(dim)
                if not (isinstance(in_range[0], int) and isinstance(in_range[1], int)):
                    err_man.raise_err_specific_user(self.op_type, "each dimension of range must be int.")
                if in_range[0] < valid_lower:
                    err_man.raise_err_attr_range_invalid(
                        self.op_type, [valid_lower, valid_upper], \
                            DIM_TO_NAME[dim] + " of " + name + " in_format " + in_format, in_range[0])
                if in_range[1]:
                    if valid_upper and in_range[1] > valid_upper:
                        err_man.raise_err_attr_range_invalid(
                            self.op_type, [valid_lower, valid_upper], \
                                DIM_TO_NAME[dim] + " of " + name + " in_format " + in_format, in_range[1])
                    if in_range[0] > in_range[1]:
                        err_man.raise_err_specific_user(self.op_type, "upper bound must be greater than lower bound.")

        for index, dim in enumerate(zip(shape, dynamic_range)):
            if dim[0] == DYNAMIC_FLAG:
                if not dim[1]:
                    err_man.raise_err_specific_user(self.op_type, "must specify range when shape is -1")
                if len(dim[1]) != RANGE_DIM_LEN:
                    err_man.raise_err_specific_user(self.op_type, "each dimension of range must be 2.")
                if dim[1][1]:
                    _check_range(dim[1], index)

    def check_para_dim(self, seq, seq_name):
        """
        check if the sequence is four-dimensional
        """

        if len(seq) != FORMAT_NCHW_DIM:
            err_man.raise_err_should_be_4d(self.op_type, seq_name)

    def check_format(self, param_format, param_name):
        """
        check if the format is valid
        """

        expect_formats = self.valid_paras.get("valid_format").get(param_name)
        if param_format not in expect_formats:
            err_man.raise_err_input_format_invalid(
                self.op_type, param_name, expect_formats, param_format)

    def check_input_dict(self, para, para_name, need_range):
        """
        check if the input dict is valid
        """

        if not isinstance(para, dict):
            err_man.raise_err_check_type(self.op_type, para_name, dict, type(para))
        if not para.get("ori_shape"):
            err_man.raise_err_specific_user(self.op_type, f"need to pass ori_shape in {para_name}")
        if not para.get("dtype"):
            err_man.raise_err_specific_user(self.op_type, f"need to pass dtype in {para_name}")
        if not para.get("ori_format"):
            err_man.raise_err_specific_user(self.op_type, f"need to pass ori_format in {para_name}")
        if list(para.get("ori_shape")) != UNKNOWN_SHAPE:
            if len(para.get("ori_shape")) != FORMAT_NCHW_DIM:
                err_man.raise_err_specific_user(self.op_type, "dim of fmap/out_backprop should be 4")
            for i in range(len(para.get("ori_shape"))):
                if not isinstance(para.get("ori_shape")[i], int):
                    err_man.raise_err_specific_user(self.op_type, "value of shape must be int")
                if para.get("ori_shape")[i] <= 0 and para.get("ori_shape")[i] != DYNAMIC_FLAG:
                    err_man.raise_err_specific_user(self.op_type, "value of shape must be -1 or >0")
            if need_range and not para.get("range"):
                err_man.raise_err_specific_user(self.op_type, f"need to pass range in {para_name}")

    def get_input_nchw(self, in_shape, in_format, in_range=()):
        """
        get input shape and range of nchw format
        """

        pos_n, pos_c, pos_h, pos_w = pos_from_format(in_format)
        in_shape = [in_shape[pos_n], in_shape[pos_c], in_shape[pos_h], in_shape[pos_w]]
        if in_range:
            if len(in_range) == FORMAT_NCHW_DIM:
                in_range = [in_range[pos_n], in_range[pos_c], in_range[pos_h], in_range[pos_w]]
            # range in NC1HWC0 format sometimes
            elif len(in_range) == FORMAT_NC1HWC0_DIM:
                in_range = [in_range[N_DIM], (in_shape[C_DIM], in_shape[C_DIM]), in_range[H_DIM], in_range[W_DIM]]
            else:
                err_man.raise_err_specific_user(self.op_type, "dimension of range should be 4 or 5.")
            for r in in_range:
                if not isinstance(r, (tuple, list)):
                    err_man.raise_err_specific_user(self.op_type, "each dim of range must be tuple or list.")
            return in_shape, [tuple(r) if r else r for r in in_range]
        return in_shape

    def get_attr_nchw(self, in_format):
        """
        get the input shape of nchw format
        """

        pos_n, pos_c, pos_h, pos_w = pos_from_format(in_format)
        self.dilations = [self.dilations[pos_n], self.dilations[pos_c],
                          self.dilations[pos_h], self.dilations[pos_w]]
        self.strides = [self.strides[pos_n], self.strides[pos_c],
                        self.strides[pos_h], self.strides[pos_w]]

    def get_output_range(self, w_shape, in_range, out_range=()):
        """
        calculate output range
        """

        def _get_output(x_in, k_size, pads, stride, dilation):
            if not x_in:
                return x_in
            return (x_in + pads[0] + pads[1] - dilation * (k_size - 1) - 1) // stride + 1

        def _get_lower_input(y_in, k_size, pads, stride, dilation):
            if not y_in:
                return y_in
            if DYNAMIC_FLAG in pads:
                return stride * (y_in - 1) + dilation * (k_size - 1) + 1
            else:
                return stride * (y_in - 1) + dilation * (k_size - 1) + 1 - pads[0] - pads[1]

        def _get_higher_input(y_in, k_size, pads, stride, dilation):
            return stride * (y_in - 1) + dilation * (k_size - 1) + 1 - pads[0] - pads[1] + stride - 1

        correct_range_flag = False
        new_in_range = copy.deepcopy(in_range)
        if DYNAMIC_FLAG in self.pads:
            out_h_lower = ceil_div(in_range[H_DIM][0], self.strides[H_DIM])
            out_h_upper = ceil_div(in_range[H_DIM][1], self.strides[H_DIM])
            out_w_lower = ceil_div(in_range[W_DIM][0], self.strides[W_DIM])
            out_w_upper = ceil_div(in_range[W_DIM][1], self.strides[W_DIM])
        else:
            out_h_lower = _get_output(in_range[H_DIM][0], w_shape[H_DIM],
                                      (self.pads[0], self.pads[1]), self.strides[H_DIM],
                                      self.dilations[H_DIM])
            out_h_upper = _get_output(in_range[H_DIM][1], w_shape[H_DIM],
                                      (self.pads[0], self.pads[1]), self.strides[H_DIM],
                                      self.dilations[H_DIM])
            out_w_lower = _get_output(in_range[W_DIM][0], w_shape[W_DIM],
                                      (self.pads[2], self.pads[3]), self.strides[W_DIM],
                                      self.dilations[W_DIM])
            out_w_upper = _get_output(in_range[W_DIM][1], w_shape[W_DIM],
                                      (self.pads[2], self.pads[3]), self.strides[W_DIM],
                                      self.dilations[W_DIM])
        out_limit_lower = max(self.valid_paras.get("hw_min"), OUT_MIN_LOWER)
        if out_h_lower < out_limit_lower:
            out_h_lower = out_limit_lower
            new_in_range[H_DIM] = (_get_lower_input(out_h_lower, w_shape[H_DIM], (self.pads[0], self.pads[1]), \
                self.strides[H_DIM], self.dilations[H_DIM]), new_in_range[H_DIM][1])
            correct_range_flag = True
            warnings.warn("The output calculated based on the lower limit of the input h " + \
                "range is less than 1, and the lower limit of the output h range is corrected " + \
                "as {}".format(out_h_lower))
        if out_h_upper and out_h_upper > self.valid_paras.get("hw_max"):
            out_h_upper = min(out_h_upper, self.valid_paras.get("hw_max"))
            new_in_range[H_DIM] = (new_in_range[H_DIM][0], _get_higher_input(out_h_upper, w_shape[H_DIM], \
                (self.pads[0], self.pads[1]), self.strides[H_DIM], self.dilations[H_DIM]))
            correct_range_flag = True
            warnings.warn("The output calculated based on the higher limit of the input h " + \
                "range is more than 4096, and the higher limit of the output h range is corrected " + \
                "as {}".format(out_h_upper))
        if out_w_lower < out_limit_lower:
            out_w_lower = out_limit_lower
            new_in_range[W_DIM] = (_get_lower_input(out_w_lower, w_shape[W_DIM], (self.pads[2], self.pads[3]), \
                self.strides[W_DIM], self.dilations[W_DIM]), new_in_range[W_DIM][1])
            correct_range_flag = True
            warnings.warn("The output calculated based on the lower limit of the input w " + \
                "range is less than 1, and the lower limit of the output w range is corrected " + \
                "as {}".format(out_w_lower))
        if out_w_upper and out_w_upper > self.valid_paras.get("hw_max"):
            out_w_upper = min(out_w_upper, self.valid_paras.get("hw_max"))
            new_in_range[W_DIM] = (new_in_range[W_DIM][0], _get_higher_input(out_w_upper, w_shape[W_DIM], \
                (self.pads[2], self.pads[3]), self.strides[W_DIM], self.dilations[W_DIM]))
            correct_range_flag = True
            warnings.warn("The output calculated based on the higher limit of the input w " + \
                "range is more than 4096, and the higher limit of the output w range is corrected " + \
                "as {}".format(out_w_upper))
        if out_h_upper and out_h_lower > out_h_upper:
            out_h_lower = out_h_upper
        if out_w_upper and out_w_lower > out_w_upper:
            out_w_lower = out_w_upper
        if out_range:
            return [out_range[N_DIM], out_range[C_DIM], (out_h_lower, out_h_upper), (out_w_lower, out_w_upper)]
        return [in_range[N_DIM], (w_shape[N_DIM], w_shape[N_DIM]),
                (out_h_lower, out_h_upper), (out_w_lower, out_w_upper)], correct_range_flag, new_in_range

    def check_pads(self, op_type):
        """
        check pad
        """
        if op_type == "deconvolution":
            if DYNAMIC_FLAG in self.pads:
                err_man.raise_err_specific_user(self.op_type,
                "not support -1 in pads for deconvolution.")
            if self.pads[0] != self.pads[1] or self.pads[2] != self.pads[3]:
                err_man.raise_err_specific_user(self.op_type,
                "value of pads for deconvolution should be [A, A, B, B].")

    def calc_pads(self, in_shape_nc1hwc0, w_shape):
        """
        calculate pads
        """

        pads = self.pads
        if DYNAMIC_FLAG in self.pads:
            # if load2d, return [0,0,0,0]
            if (self.op_type == "conv2d" and w_shape[H_DIM] * w_shape[W_DIM] == 1
                    and self.strides[H_DIM] * self.strides[W_DIM] == 1):
                pads = [0, 0, 0, 0]
            else:
                filter_h_dilation = (w_shape[H_DIM] - 1) * self.dilations[H_DIM] + 1
                filter_w_dilation = (w_shape[W_DIM] - 1) * self.dilations[W_DIM] + 1
                pad_h = (align(in_shape_nc1hwc0[H_DIM], self.strides[H_DIM]) -
                         self.strides[H_DIM] + filter_h_dilation - in_shape_nc1hwc0[H_DIM])
                pad_h = tvm.max(pad_h, 0)
                pad_up = pad_h // 2
                pad_down = pad_h - pad_up
                pad_w = (align(in_shape_nc1hwc0[W_DIM], self.strides[W_DIM]) -
                         self.strides[W_DIM] + filter_w_dilation - in_shape_nc1hwc0[W_DIM])
                pad_w = tvm.max(pad_w, 0)
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                pads = pad_up, pad_down, pad_left, pad_right
                pads = list(map(lambda x: int(x) if (isinstance(x, tvm.tir.IntImm)) else x, pads))
        self.pads = pads

    def round_channel(self, in_shape, w_shape, dtype, out_shape=()):
        """
        round up the channel dimension
        """

        if (self.op_type == "conv2d_backprop_input" and in_shape[C_DIM] != w_shape[N_DIM]
                and out_shape[C_DIM] != w_shape[C_DIM]):
            err_man.raise_err_scene_equal_limitation(self.op_type, "input feature map channel", "filter channel")

        block_size_k, block_size_n = tbe_platform.CUBE_MKN[dtype]['mac'][1:3]

        in_shape[C_DIM] = align(in_shape[C_DIM], block_size_k)
        if out_shape:
            w_shape[N_DIM] = align(in_shape[C_DIM], block_size_n)
            out_shape[C_DIM] = align(out_shape[C_DIM], block_size_k)
        else:
            w_shape[N_DIM] = align(w_shape[N_DIM], block_size_k)

    def set_group_para(self, in_shape, w_shape, w_dtype):
        """
        calculate paras for group
        """

        block_size_k, block_size_n = tbe_platform.CUBE_MKN[w_dtype]['mac'][1:3]
        cin_ori = in_shape[C_DIM] // self.groups
        cout_ori = w_shape[N_DIM] // self.groups
        cin_lcm = lcm(cin_ori, block_size_k) // cin_ori
        cout_lcm = lcm(cout_ori, block_size_n) // cout_ori
        enlarge = min(lcm(cin_lcm, cout_lcm), self.groups)
        c1_opt = math.ceil(cin_ori * enlarge / block_size_k)
        cout1_opt = math.ceil(cout_ori * enlarge / block_size_n)
        group_opt = math.ceil(self.groups / enlarge)

        return {"enlarge": enlarge, "c1_opt": c1_opt, "cout1_opt": cout1_opt, "group_opt": group_opt}

    def get_binary_mode(self, ori_format):
        """
        -------------------------------------------------------
        | binary mode |                                       |
        -------------------------------------------------------
        |      0      |  non-binary                           |
        |      1      |  binary without transdata fusion      |
        |      2      |  binary with transdata fusion of NCHW |
        |      3      |  binary with transdata fusion of NHWC |
        """
        if (self.dtype, self.y.get("dtype")) not in (("float16", "float16"), ("float32", "float32"),
                                                     ("bfloat16", "bfloat16")):
            return 0

        if not self.fusion_flag:
            return 1
        elif ori_format == "NCHW":
            return 2
        elif ori_format == "NHWC":
            return 3
        return 0

    def correct_pads(self, fmap, out_backprop, filters):
        """
        in fuzzy mode pads may real pads or is defaulted to [0, 0, 0, 0],
        check padding same or valid to correct pads,
        set pads to [-1, -1, -1, -1] while padding is not valid.
        """

        _, [out_backprop_h, out_backprop_w] = get_idx_shape_from_format(out_backprop.get("ori_format"),
                                                                         out_backprop.get("ori_shape"), "HW")
        _, [fmap_h, fmap_w] = get_idx_shape_from_format(fmap.get("ori_format"),
                                                         fmap.get("ori_shape"), "HW")
        _, [filter_h, filter_w] = get_idx_shape_from_format(filters.get("ori_format"),
                                                             filters.get("ori_shape"), "HW")
        stride_h = self.strides[self.data_format.find("H")]
        stride_w = self.strides[self.data_format.find("W")]
        need_correct_pads = list(self.pads) == [0, 0, 0, 0] and ((out_backprop_h - 1) * stride_h + filter_h > fmap_h
            or (out_backprop_w - 1) * stride_w + filter_w > fmap_w)
        if need_correct_pads:
            self.pads = [-1, -1, -1, -1]
        return self.pads


class Conv2dParaProcess(CubeParaProcess):
    """
    class of param check and preprocess for dynamic conv2d
    """

    def __init__(self, paras):
        def conver_tensor2dict(tensor, need_range):
            if tensor is None:
                return None
            tdict = {}
            tdict["ori_shape"] = []
            for i in tensor.op.attrs['ori_shape']:
                tdict["ori_shape"].append(i.value)
            tdict["dtype"] = tensor.dtype
            tdict["ori_format"] = tensor.op.attrs['ori_format'].value

            if need_range is True:
                tdict["range"] = []
                for one_range in tensor.op.attrs['range']:
                    range_list = []
                    for value in one_range:
                        range_list.append(value.value)
                    tdict["range"].append(range_list)
                if operation.get_te_var("batch_n"):
                    tdict["range"][N_DIM] = list(operation.get_te_var("batch_n").get_bound())
                if operation.get_te_var("fmap_h"):
                    tdict["range"][H_DIM] = list(operation.get_te_var("fmap_h").get_bound())
                if operation.get_te_var("fmap_w"):
                    tdict["range"][W_DIM] = list(operation.get_te_var("fmap_w").get_bound())

            return tdict

        super().__init__(paras)
        self.op_type = "conv2d"
        if isinstance(paras.get("inputs"), dict):
            self.is_tensor = False
            self.inputs = paras.get("inputs")
            self.weights = paras.get("weights")
            self.bias = paras.get("bias")
            self.dtype = paras.get("inputs").get("dtype")
        else:
            self.is_tensor = True
            self.input_tensor = paras.get("inputs")
            self.weights_tensor = paras.get("weights")
            self.bias_tensor = paras.get("bias")

            self.inputs = conver_tensor2dict(self.input_tensor, True)
            self.weights = conver_tensor2dict(self.weights_tensor, False)
            self.bias = conver_tensor2dict(self.bias_tensor, False)
            self.dtype = self.input_tensor.dtype

        self.outputs = paras.get("outputs")
        self.data_format = paras.get("data_format")

    def check_support_valid(self, in_shape, filter_shape):
        """
        check whether dynamic shape is supported for conv2d
        """

        super().check_support_valid(in_shape, filter_shape)
        if in_shape[C_DIM] == DYNAMIC_FLAG:
            err_man.raise_err_specific_user(
                self.op_type, "dynamic c dimension is not supported yet.")
        if self.paras.get("offset_w"):
            err_man.raise_err_specific_user(
                self.op_type, "offset_w is not supported in dynamic shape yet.")

    def correct_in_range(self, in_range_nchw, w_shape_nchw):
        """
        correct in_range when w_range=[1, None]
        """


        m_bit_ratio = {"float16": 2, "int8": 1}
        c0 = tbe_platform.CUBE_MKN[self.weights["dtype"]]["mac"][1]
        fmap_w_upper = in_range_nchw[W_DIM][1]
        new_in_range_nchw = list(in_range_nchw)

        if not fmap_w_upper:
            stride_h = self.strides[H_DIM]
            stride_w = self.strides[W_DIM]
            hk_dilation = (w_shape_nchw[H_DIM] - 1) * self.dilations[H_DIM] + 1
            wk_dilation = (w_shape_nchw[W_DIM] - 1) * self.dilations[W_DIM] + 1
            l1size_limit_upper = tbe_platform.get_soc_spec("L1_SIZE")
            w_left = DYNAMIC_FMAP_W_MIN
            w_right = DYNAMIC_FMAP_W_MAX
            current_w = DYNAMIC_FMAP_W_MAX
            while (w_right - w_left) != 1:
                if -1 in self.pads:
                    w_out = (current_w + stride_w - 1) // stride_w
                else:
                    w_out = math.floor((current_w - wk_dilation + self.pads[2] + self.pads[3]) / stride_w) + 1
                ho_num = math.floor(tbe_platform.CUBE_MKN[self.weights["dtype"]]["mac"][0] / w_out) + 2
                l1_m = ((ho_num - 1) * stride_h + hk_dilation) * current_w
                max_feature_map_l1 = c0 * l1_m * m_bit_ratio[self.weights["dtype"]]
                if max_feature_map_l1 > l1size_limit_upper:
                    w_right = current_w
                else:
                    w_left = current_w
                current_w = w_left + (w_right - w_left)//2

                if w_left == DYNAMIC_FMAP_W_MAX:
                    break

            cor_w_range = (1, w_left)
            new_in_range_nchw[W_DIM] = cor_w_range
            to_print = "conv2d fmap ori_range changed from {} to {}.".format(in_range_nchw, new_in_range_nchw)
            warnings.warn(to_print)

        return new_in_range_nchw

    def check_paras(self):
        """
        check original paras
        """
        self.check_input_dict(self.inputs, "inputs", True)
        self.check_input_dict(self.weights, "weights", False)
        para_check.check_dtype_rule(self.dtype, self.valid_paras.get("valid_dtype"))
        para_check.check_dtype_rule(self.weights.get("dtype"), self.valid_paras.get("valid_dtype"))
        para_check.check_dtype_rule(self.paras.get("outputs").get("dtype"), self.valid_paras.get("valid_dtype"))
        if self.dtype != self.weights.get("dtype"):
            err_man.raise_err_specific_user("conv2d", "in_dtype != w_dtype")
        self.check_format(self.data_format, "input")
        self.check_format(self.weights.get("ori_format"), "weights")
        if self.inputs.get("ori_format") != self.data_format:
            err_man.raise_err_specific_user("conv2d", "in_format != data_format")
        para_check.check_kernel_name(self.paras.get("kernel_name"))

        in_shape = list(self.inputs.get("ori_shape"))
        in_range = self.inputs.get("range")
        w_shape = list(self.weights.get("ori_shape"))
        outputs_shape = list(self.outputs.get("ori_shape"))
        self.check_para_dim(w_shape, "weights")
        self.check_para_dim(self.strides, "strides")
        self.check_para_dim(self.dilations, "dilations")
        self.check_para_dim(self.pads, "pads")
        w_shape_nchw = self.get_input_nchw(w_shape, self.weights.get("ori_format"))
        out_shape_nchw = self.get_input_nchw(outputs_shape, self.outputs.get("ori_format"))

        if self.check_unknown_scene(in_shape, out_shape_nchw, w_shape_nchw[N_DIM]):
            in_shape_nchw = [DYNAMIC_FLAG, w_shape_nchw[C_DIM], DYNAMIC_FLAG, DYNAMIC_FLAG]
            in_range_nchw = [(1, None), (w_shape_nchw[C_DIM], w_shape_nchw[C_DIM]), (1, None), (1, None)]
        else:
            self.check_para_dim(in_shape, "in_shape")
            in_shape_nchw, in_range_nchw = self.get_input_nchw(in_shape, self.data_format, in_range)
            if in_shape_nchw[1] == -1:
                in_shape_nchw[1] = w_shape_nchw[1]*self.groups
            self.check_range_valid(in_shape_nchw, in_range_nchw, "fmap", self.data_format)

        cor_in_range_nchw = self.correct_in_range(in_range_nchw, w_shape_nchw)
        self.check_support_valid(in_shape_nchw, w_shape_nchw)
        self.get_attr_nchw(self.data_format)
        y_range, correct_range_flag, new_in_range_nchw = self.get_output_range(w_shape_nchw, cor_in_range_nchw)
        self.check_range_valid(out_shape_nchw, y_range, "output", self.data_format)

        group_para = self.set_group_para(in_shape_nchw, w_shape_nchw, self.dtype)
        in_shape_nchw, w_shape_nchw, in_shape_nc1hwc0, w_shape_frac_z = self.__calc_shape(
            in_shape_nchw, w_shape_nchw, new_in_range_nchw, y_range, group_para)
        self.calc_pads(in_shape_nc1hwc0, w_shape_nchw)

        return {"in_shape_nc1hwc0": in_shape_nc1hwc0, "w_shape_frac_z": w_shape_frac_z,
                "w_shape": w_shape_nchw, "group_para": group_para,
                "correct_range_flag": correct_range_flag,
                "new_in_range": new_in_range_nchw}

    def config_paras(self):
        """
        config paras and placeholders
        """

        param = self.check_paras()
        if self.is_tensor is False:
            input_tensor = tvm.placeholder(param.get("in_shape_nc1hwc0"), name="Fmap", dtype=self.dtype)
            weight_tensor = tvm.placeholder(param.get("w_shape_frac_z"), name="Filter", dtype=self.dtype)
            if self.bias:
                bias_tensor = tvm.placeholder((param.get("w_shape")[N_DIM],), name="bias_tensor",
                    dtype=self.bias.get("dtype"))
            else:
                bias_tensor = None
        else:
            input_tensor = self.input_tensor
            weight_tensor = self.weights_tensor
            bias_tensor = self.bias_tensor

        return {"input_tensor": input_tensor, "weight_tensor": weight_tensor, "bias_tensor": bias_tensor,
                "w_shape": param.get("w_shape"), "in_shape_nc1hwc0": param.get("in_shape_nc1hwc0"),
                "w_shape_frac_z": param.get("w_shape_frac_z"), "group_para": param.get("group_para"),
                "correct_range_flag": param.get("correct_range_flag", False), "new_in_range": param.get("new_in_range")}

    def __calc_shape(self, in_shape, w_shape, in_range, y_range, group_para):
        """
        calculate shape for mmad
        """

        block_size_k, block_size_n = tbe_platform.CUBE_MKN[self.dtype]['mac'][1:3]
        in_shape[C_DIM] = align(in_shape[C_DIM], block_size_k)
        # filter channel should be equal input channel
        w_shape[C_DIM] = in_shape[C_DIM]

        in_shape_nc1hwc0 = [in_shape[N_DIM], in_shape[C_DIM] // block_size_k,
                            in_shape[H_DIM], in_shape[W_DIM], block_size_k]
        if not self.is_tensor:
            if in_shape_nc1hwc0[N_DIM] == DYNAMIC_FLAG:
                in_shape_nc1hwc0[N_DIM] = operation.var("batch_n", in_range[N_DIM])
                operation.add_exclude_bound_var(in_shape_nc1hwc0[N_DIM])
            if in_shape_nc1hwc0[H_DIM] == DYNAMIC_FLAG:
                in_shape_nc1hwc0[H_DIM] = operation.var("fmap_h", in_range[H_DIM])
                operation.add_exclude_bound_var(in_shape_nc1hwc0[H_DIM])
                operation.add_exclude_bound_var(operation.var("ho", y_range[H_DIM]))
            if in_shape_nc1hwc0[W_DIM] == DYNAMIC_FLAG:
                in_shape_nc1hwc0[W_DIM] = operation.var("fmap_w", in_range[W_DIM])
                operation.add_exclude_bound_var(in_shape_nc1hwc0[W_DIM])
                operation.add_exclude_bound_var(operation.var("wo", y_range[W_DIM]))
        else:
            if in_shape_nc1hwc0[N_DIM] == DYNAMIC_FLAG:
                in_shape_nc1hwc0[N_DIM] = self.input_tensor.shape[N_DIM]
            if in_shape_nc1hwc0[H_DIM] == DYNAMIC_FLAG:
                in_shape_nc1hwc0[H_DIM] = self.input_tensor.shape[H_DIM]
                operation.add_exclude_bound_var(operation.var("ho", y_range[H_DIM]))
            if in_shape_nc1hwc0[W_DIM] == DYNAMIC_FLAG:
                in_shape_nc1hwc0[W_DIM] = self.input_tensor.shape[W_DIM]
                operation.add_exclude_bound_var(operation.var("wo", y_range[W_DIM]))

        if self.paras.get("optim_dict").get("c0_optim_flg"):
            w_shape_frac_z = (ceil_div(4 * w_shape[H_DIM] * w_shape[W_DIM], block_size_k),
                              math.ceil(w_shape[N_DIM] / block_size_n), block_size_n, block_size_k)
        else:
            w_shape_frac_z = (group_para.get("group_opt") * group_para.get("c1_opt") * w_shape[H_DIM] * w_shape[W_DIM],
                              group_para.get("cout1_opt"), block_size_n, block_size_k)
        return in_shape, w_shape, in_shape_nc1hwc0, w_shape_frac_z


class Conv2dBackpropParaProcess(CubeParaProcess):
    """
    class of param check and preprocess for dynamic conv2d_backprop_input
    """

    def __init__(self, paras):
        super().__init__(paras)
        self.op_type = "conv2d_backprop_input"
        self.filters = paras.get("filters")
        self.out_backprop = paras.get("out_backprop")
        self.y = paras.get("y")
        self.data_format = paras.get("data_format")
        self.pads = paras.get("pads")
        if isinstance(self.filters, dict):
            self.dtype = paras.get("filters").get("dtype")
        else:
            self.dtype = self.filters.dtype
        if paras.get("input_size") is not None:
            self.input_size = paras.get("input_size")
        else:
            self.input_size = {"ori_shape": INPUT_SIZE_DEFAULT_SHAPE}
        self.split_axis_mode = paras.get("split_axis_mode", 0)
        self.pooling_mode = paras.get("pooling_mode")
        self.shape = {}
        self.range = {}
        self.attrs = {}
        self.tensors = {}
        self.valid_paras["valid_dtype"] = ("float16", "int8", "int32", "float32", "bfloat16")

    @staticmethod
    def _get_none_range_flag(dynamic_range):
        """
        Determine whether it is a None range scene
        """
        if not dynamic_range:
            return True
        for idx_range in dynamic_range:
            if not idx_range or not idx_range[0] or not idx_range[1]:
                return True
            if idx_range[1] > DYNAMIC_FMAP_W_MAX:
                return True
        return False

    @staticmethod
    def _define_tiling_var():
        """
        define tiling vars for binary mode
        """
        operation.var("group_dim")
        operation.var("batch_dim")
        operation.var("n_dim")
        operation.var("m_dim")
        operation.var("batch_single_core")
        operation.var("m_al1")
        operation.var("n_bl1")
        operation.var("k_aub")
        operation.var("m_aub")
        operation.var("wo_aub")
        operation.var("m_l0")
        operation.var("n_l0_div_ub")
        operation.var("n_ub")
        operation.var("k_l0")
        operation.var("min_kl1_div_kl0")
        operation.var("max_kl1_div_min_kl1")
        operation.var("k_div_max_kl1")
        operation.var("al1_bound")
        operation.var("bl1_bound")
        operation.var("aub_bound")
        operation.var("bias_table_bound")

    def check_paras(self):
        """
        check original paras
        """
        if isinstance(self.filters, dict):
            self.check_input_dict(self.filters, "filters", False)
            self.check_input_dict(self.out_backprop, "out_backprop", False)
            self.check_input_dict(self.y, "y", False)
            if UNKNOWN_FLAG in self.input_size.get("ori_shape") or DYNAMIC_FLAG in self.input_size.get("ori_shape"):
                err_man.raise_err_specific_user(
                    self.op_type, "dynamic shape not support input size's shape [-1] and [-2]")
            if self.dtype != self.out_backprop.get("dtype"):
                err_man.raise_err_specific_user(
                    "conv2d_backprop_input", "the dtype of filter and out_backprop are not the same.")
            self.check_format(self.filters.get("ori_format"), "weights")
            if self.out_backprop.get("ori_format") != self.data_format:
                err_man.raise_err_specific_user(
                    "conv2d_backprop_input", "the format of out_backprop and data_format are not the same.")
            weight_shape = self.filters.get("ori_shape")
        else:
            self.fusion_flag = True

        para_check.check_dtype_rule(self.dtype, self.valid_paras.get("valid_dtype"))
        para_check.check_dtype_rule(self.y.get("dtype"), self.valid_paras.get("valid_dtype"))
        self.check_format(self.data_format, "output")
        if self.y.get("ori_format") != self.data_format:
            err_man.raise_err_specific_user(
                "conv2d_backprop_input", "the ori_format of y and data_format are not the same.")
        para_check.check_kernel_name(self.paras.get("kernel_name"))
        self.check_para_dim(self.strides, "strides")
        self.check_para_dim(self.dilations, "dilations")
        self.check_para_dim(self.pads, "pads")
        if self.split_axis_mode not in [0, 1]:
            err_man.raise_err_specific_user(self.op_type,
                                            "The value of split_axis_mode can only be 0 (split_hw) or 1 (split_w).")
        # must in the last line
        self.binary_mode = self.get_binary_mode(self.data_format)

    def config_paras(self):
        """
        config paras and placeholders
        """
        self.check_paras()
        self._infer_shape_range_attrs()
        if self.fusion_flag:
            self.tensors = {
                "input_tensor": self.input_size,
                "dy_tensor": self.out_backprop,
                "filter_tensor": self.filters
            }
        else:
            self.tensors["input_tensor"] = tvm.placeholder([4], name="input_size", dtype="int32")
            self.tensors["dy_tensor"] = tvm.placeholder(self.shape.get("dy_shape_nc1hwc0"),
                                                        name="dedy", dtype=self.dtype)
            self.tensors["filter_tensor"] = tvm.placeholder(self.shape.get("filter_shape_frac_z"),
                                                            name="filter", dtype=self.dtype)

    def _calc_filter_shape_fz(self):
        """
        calculate filter's shape with frac_z format
        """
        group_para = self.attrs.get("group_para")
        filter_shape = self.shape.get("filter_shape_nchw")
        block_size_k, block_size_n = tbe_platform.CUBE_MKN[self.dtype]['mac'][1:3]

        if self.dtype == "int8":
            filter_shape_frac_z = (
                group_para["g_extend"] * group_para["dy_c1_extend"] * filter_shape[H_DIM] * filter_shape[W_DIM],
                group_para["dx_c1_extend"],
                block_size_n,
                block_size_k,
            )
        else:
            filter_shape_frac_z = (
                group_para["g_extend"] * group_para["dx_c1_extend"] * filter_shape[H_DIM] * filter_shape[W_DIM],
                group_para["dy_c1_extend"],
                block_size_n,
                block_size_k
            )
        self.shape["filter_shape_frac_z"] = filter_shape_frac_z

    def _define_attrs_var(self):
        """
        define attrs vars for binary mode
        """
        self.pads = (operation.var("padt"), operation.var("padb"),
                     operation.var("padl"), operation.var("padr"))
        _, _, pos_h, pos_w = pos_from_format(self.data_format)
        stride_h = operation.var("stride_h")
        stride_w = operation.var("stride_w")
        if self.strides[pos_h] != 1 or self.strides[pos_w] != 1:
            self.strides = (1, 1, stride_h, stride_w)
        dilation_h = operation.var("dilation_h")
        dilation_w = operation.var("dilation_w")
        self.dilations = (1, 1, dilation_h, dilation_w)
        operation.var("shape_up_modify")
        operation.var("shape_left_modify")
        operation.var("shape_down_modify")
        operation.var("shape_right_modify")
        operation.var("load3d_special")
        operation.var("pad_up_before")
        operation.var("pad_left_before")
        operation.var("pad_down_after")
        operation.var("pad_right_after")
        operation.var("bias_flag")
        operation.var("hf32_flag")

    def _infer_binary_shape_range_attrs(self):
        """
        define vars for binary mode
        """
        # support both Tensor input and dict input
        # NOTE always create variables in the order filter, out_backprop in shape_util.py
        if isinstance(self.out_backprop, dict):
            filter_ci1hw = operation.var("filter_ci1hw")
            dy_c1_extend = operation.var("filter_col")
            batch = operation.var("batch")
            dy_c1 = operation.var("dedy_c1")
            dedy_h = operation.var("dedy_h")
            dedy_w = operation.var("dedy_w")
            dedy_dtype = self.out_backprop.get("dtype")
            dy_c0 = BLOCK_K_DICT.get(dedy_dtype)
        else:
            # fusion condition
            # batch, co1, ho, wo
            batch, dy_c1, dedy_h, dedy_w, dy_c0 = self.out_backprop.shape

            dy_c1_extend = self.filters.shape[1]
        dx_c = operation.var("dx_c")
        dx_c1 = operation.var("dx_c1")
        dx_h = operation.var("dx_h")
        dx_w = operation.var("dx_w")
        kernel_h = operation.var("kernel_h")
        kernel_w = operation.var("kernel_w")
        g_extend = operation.var("g_extend")
        dx_c1_extend = operation.var("dx_c1_extend")

        self.attrs["group_para"] = {"g_extend": g_extend,
                                    "dx_c1_extend": dx_c1_extend,
                                    "dy_c1_extend": dy_c1_extend}
        dx_c0 = CUBE_SIZE
        if tbe_platform.intrinsic_check_support("Intrinsic_fix_pipe_l0c2out"):
            dx_c0 = BLOCK_K_DICT.get(self.y.get("dtype"))
        self.shape["dy_shape_nc1hwc0"] = (batch, dy_c1, dedy_h, dedy_w, dy_c0)
        self.shape["dy_shape_nchw"] = (batch, dy_c1 * dy_c0, dedy_h, dedy_w)
        self.shape["filter_shape_nchw"] = (dy_c1 * dy_c0, dx_c, kernel_h, kernel_w)
        self.shape["dx_shape_nchw"] = (batch, dx_c, dx_h, dx_w)
        self.shape["dx_shape_nc1hwc0"] = (batch, dx_c1, dx_h, dx_w, dx_c0)
        self._calc_filter_shape_fz()
        self._define_attrs_var()
        self._define_tiling_var()
        self.range["dy_range_nchw"] = BINARY_RANGE
        self.range["new_dx_range_nchw"] = BINARY_RANGE


    def _config_shape(self):
        """
        calculate shape for mad
        """

        dy_shape = self.shape.get("dy_shape_nchw")
        input_size = self.shape.get("dx_shape_nchw")
        dy_range = self.range.get("dy_range_nchw")
        input_range = self.range.get("dx_range_nchw")

        self.round_channel(dy_shape, self.shape.get("filter_shape_nchw"), self.dtype, input_size)
        block_size_k = tbe_platform.CUBE_MKN[self.dtype]['mac'][1]
        dy_shape_nc1hwc0 = [dy_shape[N_DIM], dy_shape[C_DIM] // block_size_k,
                            dy_shape[H_DIM], dy_shape[W_DIM], block_size_k]

        if input_size[N_DIM] == DYNAMIC_FLAG:
            dy_shape_nc1hwc0[N_DIM] = operation.var("batch_n", dy_range[N_DIM])
            input_size[N_DIM] = dy_shape_nc1hwc0[N_DIM]
            operation.add_exclude_bound_var(dy_shape_nc1hwc0[N_DIM])
        if input_size[H_DIM] == DYNAMIC_FLAG:
            dy_shape_nc1hwc0[H_DIM] = operation.var("dedy_h", dy_range[H_DIM])
            input_size[H_DIM] = operation.var("dx_h", input_range[H_DIM])
            operation.add_exclude_bound_var(dy_shape_nc1hwc0[H_DIM])
            operation.add_exclude_bound_var(input_size[H_DIM])
        if input_size[W_DIM] == DYNAMIC_FLAG:
            dy_shape_nc1hwc0[W_DIM] = operation.var("dedy_w", dy_range[W_DIM])
            input_size[W_DIM] = operation.var("dx_w", input_range[W_DIM])
            operation.add_exclude_bound_var(dy_shape_nc1hwc0[W_DIM])
            operation.add_exclude_bound_var(input_size[W_DIM])

        self.shape['dy_shape_nc1hwc0'] = dy_shape_nc1hwc0
        self.calc_pads(input_size, self.shape.get("filter_shape_nchw"))
        self._calc_filter_shape_fz()

    def _infer_shape_nchw(self):
        """
        get shape in format nchw
        """
        self.check_input_dict(self.y, "y", False)
        filter_shape = self.filters.get("ori_shape")
        dx_shape = self.y.get("ori_shape")
        self.check_para_dim(dx_shape, "input_size")
        self.check_para_dim(filter_shape, "filters")
        self.check_pads(self.op_type)
        filter_shape_nchw = self.get_input_nchw(filter_shape, self.filters.get("ori_format"))
        self.get_attr_nchw(self.data_format)
        dx_shape_nchw = self.get_input_nchw(dx_shape, self.data_format)

        return filter_shape_nchw, dx_shape_nchw, dx_shape

    def _infer_shape_range_attrs(self):
        """
        infer range from dx to dy
        """
        if self.binary_mode:
            self._infer_binary_shape_range_attrs()
            return

        dy_shape = list(self.out_backprop.get("ori_shape"))
        filter_shape_nchw, dx_shape_nchw, dx_shape = self._infer_shape_nchw()
        correct_range_flag = False
        if self.check_unknown_scene(dy_shape, dx_shape_nchw, filter_shape_nchw[C_DIM] * self.groups):
            dy_shape_nchw = [DYNAMIC_FLAG, filter_shape_nchw[N_DIM], DYNAMIC_FLAG, DYNAMIC_FLAG]
            dy_range_nchw = BINARY_RANGE
            dx_range_nchw = BINARY_RANGE
            self.check_support_valid(dy_shape_nchw, filter_shape_nchw)
        else:
            self.check_para_dim(dy_shape, "out_backprop_shape")
            self.check_input_dict(self.y, "y", True)
            dx_range = self.y.get("range")
            dy_shape_nchw = self.get_input_nchw(dy_shape, self.data_format)
            for idx, dim_value in enumerate(dx_shape_nchw):
                if dim_value == DYNAMIC_FLAG:
                    dy_shape_nchw[idx] = DYNAMIC_FLAG
            dx_shape_nchw, dx_range_nchw = self.get_input_nchw(dx_shape, self.data_format, dx_range)
            self.check_support_valid(dy_shape_nchw, filter_shape_nchw)
            self.check_dynamic_channel_scene(
                dy_shape_nchw, dx_shape_nchw, filter_shape_nchw[N_DIM])
        group_para = comm.calculate_group(
            dy_shape_nchw, dx_shape_nchw, filter_shape_nchw, self.groups, self.dtype, self.data_format)
        if dx_range_nchw != BINARY_RANGE:
            self.check_range_valid(dx_shape_nchw, dx_range_nchw, "input_size", self.data_format)
            dy_range_nchw, correct_range_flag,\
                dx_range_nchw = self.get_output_range(filter_shape_nchw, dx_range_nchw)
            output_range = copy.deepcopy(dy_range_nchw)
            if output_range[W_DIM][1]:
                if filter_shape_nchw[H_DIM] == 1 and filter_shape_nchw[W_DIM] == 1:
                    output_range[W_DIM] = (output_range[W_DIM][0],
                                        output_range[W_DIM][1] * self.strides[H_DIM] * self.strides[W_DIM])
            self.check_range_valid(dy_shape_nchw, output_range, "out_backprop", self.data_format)

        self.shape = {"dy_shape_nchw": dy_shape_nchw, "filter_shape_nchw": filter_shape_nchw,
                      "dx_shape_nchw": dx_shape_nchw}
        self.range = {"dy_range_nchw": dy_range_nchw, "dx_range_nchw": dx_range_nchw}
        self.attrs = {"group_para": group_para, "correct_range_flag": correct_range_flag}
        self._config_shape()


class Conv2dTransposeParaProcess(Conv2dBackpropParaProcess):
    """
    class of param check and preprocess for dynamic conv2d_transpose
    """

    def __init__(self, paras):
        super().__init__(paras)
        self.op_type = "conv2d_transpose"
        self.out_backprop = paras.get("x")
        self.filter = paras.get("filters")
        self.data_format = paras.get("data_format")
        self.fmap = paras.get("y")

    @staticmethod
    def _get_lower_input(y_in, k_size, pads, stride, dilation):
        if not y_in:
            return y_in
        return stride * (y_in - 1) + dilation * (k_size - 1) + 1 - pads[0] - pads[1]

    @staticmethod
    def _get_higher_input(y_in, k_size, pads, stride, dilation):
        if not y_in:
            return y_in
        return stride * (y_in - 1) + dilation * (k_size - 1) + 1 - pads[0] - pads[1] + stride - 1

    @staticmethod
    def _get_output(x_in, k_size, pads, stride, dilation):
        if not x_in:
            return x_in
        if DYNAMIC_FLAG in pads:
            return ceil_div(x_in, stride)
        return (x_in + pads[0] + pads[1] - dilation * (k_size - 1) - 1) // stride + 1

    def check_support_valid(self, in_shape, filter_shape):
        """
        check whether dynamic shape is supported for conv2d_transpose
        """
        super().check_support_valid(in_shape, filter_shape)
        if self.paras.get("offset_w"):
            err_man.raise_err_specific_user(
                self.op_type, "offset_w is not supported in dynamic shape yet.")
        if self.paras.get("offset_x") != 0:
            err_man.raise_err_specific_user(
                self.op_type, "offset_x is not supported in dynamic shape yet.")

    def get_input_range(self, w_shape, dy_range, dx_range=()):
        """
        calculate input range
        """
        new_dy_range = copy.deepcopy(dy_range)
        if DYNAMIC_FLAG in self.pads:
            dx_h_lower, dx_h_upper, h_correct_range_flag = self._get_input_range_by_dynamic_pad(
                new_dy_range, dy_range, w_shape, H_DIM)
            dx_w_lower, dx_w_upper, w_correct_range_flag = self._get_input_range_by_dynamic_pad(
                new_dy_range, dy_range, w_shape, W_DIM)
        else:
            dx_h_lower, dx_h_upper, h_correct_range_flag = self._get_input_range_by_static_pad(
                new_dy_range, dy_range, w_shape, H_DIM)
            dx_w_lower, dx_w_upper, w_correct_range_flag = self._get_input_range_by_static_pad(
                new_dy_range, dy_range, w_shape, W_DIM)

        correct_range_flag = h_correct_range_flag or w_correct_range_flag
        if dx_h_upper and dx_h_lower > dx_h_upper:
            dx_h_lower = dx_h_upper
        if dx_w_upper and dx_w_lower > dx_w_upper:
            dx_w_lower = dx_w_upper
        if dx_range:
            return [dx_range[N_DIM], dx_range[C_DIM], (dx_h_lower, dx_h_upper), (dx_w_lower, dx_w_upper)]
        return [dy_range[N_DIM], (w_shape[C_DIM], w_shape[C_DIM]),
                (dx_h_lower, dx_h_upper), (dx_w_lower,  dx_w_upper)], correct_range_flag, new_dy_range

    def infer_shape_and_range(self):
        """
        infer range from dy to dx
        """
        if self.binary_mode:
            super()._infer_binary_shape_range_attrs()
            return

        self.check_input_dict(self.out_backprop, "out_backprop", True)
        dy_shape = list(self.out_backprop.get("ori_shape"))
        dy_range = self.out_backprop.get("range")
        filter_shape = self.filters.get("ori_shape")
        dx_shape = self.y.get("ori_shape")
        self.check_para_dim(dx_shape, "input_size")
        self.check_para_dim(filter_shape, "filters")
        self.check_pads(self.op_type)
        filter_shape_nchw = self.get_input_nchw(filter_shape, self.filters.get("ori_format"))
        self.get_attr_nchw(self.data_format)
        dx_shape_nchw = self.get_input_nchw(dx_shape, self.data_format)

        if self.check_unknown_scene(dy_shape, dx_shape_nchw, filter_shape_nchw[C_DIM] * self.groups):
            dy_shape_nchw = [DYNAMIC_FLAG, filter_shape_nchw[N_DIM], DYNAMIC_FLAG, DYNAMIC_FLAG]
            dy_range_nchw = [(1, None), None, (1, None), (1, None)]
            dx_range_nchw = [(1, None), None, (1, None), (1, None)]
            self.check_support_valid(dy_shape_nchw, filter_shape_nchw)
            group_para = comm.calculate_group(
                dy_shape_nchw, dx_shape_nchw, filter_shape_nchw, self.groups, self.dtype, self.data_format)
        else:
            self.check_para_dim(dy_shape, "out_backprop_shape")
            dy_shape_nchw, dy_range_nchw = self.get_input_nchw(dy_shape, self.data_format, dy_range)
            output_range = copy.deepcopy(dy_range_nchw)
            self.check_support_valid(dy_shape_nchw, filter_shape_nchw)
            self.check_dynamic_channel_scene(
                dy_shape_nchw, dx_shape_nchw, filter_shape_nchw[N_DIM])
            group_para = comm.calculate_group(
                dy_shape_nchw, dx_shape_nchw, filter_shape_nchw, self.groups, self.dtype, self.data_format)
            self.check_range_valid(dy_shape_nchw, output_range, "out_backprop", self.data_format)
            if output_range[W_DIM][1]:
                if filter_shape_nchw[H_DIM] == 1 and filter_shape_nchw[W_DIM] == 1:
                    output_range[W_DIM] = (output_range[W_DIM][0],
                                           output_range[W_DIM][1] * self.strides[H_DIM] * self.strides[W_DIM])
            self.check_range_valid(dy_shape_nchw, output_range, "out_backprop", self.data_format)
        dx_range_nchw, correct_range_flag, dy_range_nchw = self.get_input_range(filter_shape_nchw, dy_range_nchw)

        self.check_range_valid(dx_shape_nchw, dx_range_nchw, "input_size", self.data_format)

        self.shape = {"dy_shape_nchw": dy_shape_nchw, "filter_shape_nchw": filter_shape_nchw,
                      "dx_shape_nchw": dx_shape_nchw}
        self.range = {"dy_range_nchw": dy_range_nchw, "dx_range_nchw": dx_range_nchw}
        self.attrs = {"group_para": group_para, "correct_range_flag": correct_range_flag}
        self._config_shape()

    def config_paras(self):
        """
        check original paras
        """
        super().check_paras()
        self.infer_shape_and_range()
        if self.fusion_flag:
            self.tensors = {
                "input_tensor": self.input_size,
                "x_tensor": self.out_backprop,
                "filter_tensor": self.filters
            }
        else:
            self.tensors["input_tensor"] = tvm.placeholder([4], name="input_size", dtype="int32")
            self.tensors["x_tensor"] = tvm.placeholder(self.shape.get("dy_shape_nc1hwc0"),
                                                       name="dedy",
                                                       dtype=self.dtype)
            self.tensors["filter_tensor"] = tvm.placeholder(self.shape.get("filter_shape_frac_z"),
                                                            name="filter",
                                                            dtype=self.dtype)
        if self.paras.get("bias"):
            input_channel = align(self.shape.get("dx_shape_nchw")[C_DIM], tbe_platform.CUBE_MKN[self.dtype]['mac'][2])
            bias_dtype = "float32" if self.y.get("dtype") == "bfloat16" else self.y.get("dtype")
            self.tensors["bias_tensor"] = tvm.placeholder((input_channel,),
                                                          name="tensor_bias", dtype=bias_dtype)

    def _get_input_range_by_dynamic_pad(self, new_dy_range, dy_range, w_shape, dim_idx):
        dim_name = "h"
        pads = (self.pads[0], self.pads[1])
        if dim_idx == W_DIM:
            dim_name = "w"
            pads = (self.pads[2], self.pads[3])

        correct_range_flag = False
        dx_lower = (dy_range[dim_idx][0] - 1) * self.strides[dim_idx] + 1
        if not dy_range[dim_idx][1]:
            dx_upper = dy_range[dim_idx][1]
        else:
            dx_upper = dy_range[dim_idx][1] * self.strides[dim_idx]
            if dx_upper > self.valid_paras.get("hw_max"):
                dx_upper = min(dx_upper, self.valid_paras.get("hw_max"))
                new_dy_range[dim_idx] = (new_dy_range[dim_idx][0],
                                         self._get_output(dx_upper, w_shape[dim_idx], pads, self.strides[dim_idx],
                                                          self.dilations[dim_idx]))
                correct_range_flag = True
                warnings.warn("The input calculated based on the upper limit of the output {} "
                              "range is more than 4096, and the upper limit of the input {} range is corrected "
                              "as {}".format(dim_name, dim_name, dx_upper))
        return dx_lower, dx_upper, correct_range_flag

    def _get_input_range_by_static_pad(self, new_dy_range, dy_range, w_shape, dim_idx):
        dim_name = "h"
        pads = (self.pads[0], self.pads[1])
        if dim_idx == W_DIM:
            dim_name = "w"
            pads = (self.pads[2], self.pads[3])

        correct_range_flag = False
        dx_lower = self._get_lower_input(dy_range[dim_idx][0], w_shape[dim_idx], pads, self.strides[dim_idx],
                                         self.dilations[dim_idx])
        if dx_lower < self.valid_paras.get("nhw_min"):
            dx_lower = max(dx_lower, self.valid_paras.get("nhw_min"))
            new_dy_range[dim_idx] = (self._get_output(dx_lower, w_shape[dim_idx], pads, self.strides[dim_idx],
                                                      self.dilations[dim_idx]), new_dy_range[dim_idx][1])
            correct_range_flag = True
            warnings.warn("The input calculated based on the lower limit of the output {} "
                          "range is less than 1, and the lower limit of the input {} range is corrected "
                          "as {}".format(dim_name, dim_name, dx_lower))
        if not dy_range[dim_idx][1]:
            dx_upper = dy_range[dim_idx][1]
        else:
            dx_upper = self._get_higher_input(dy_range[dim_idx][1], w_shape[dim_idx], pads, self.strides[dim_idx],
                                              self.dilations[dim_idx])
            if dx_upper > self.valid_paras.get("hw_max"):
                dx_upper = min(dx_upper, self.valid_paras.get("hw_max"))
                new_dy_range[dim_idx] = (new_dy_range[dim_idx][0],
                                         self._get_output(dx_upper, w_shape[dim_idx], pads, self.strides[dim_idx],
                                                          self.dilations[dim_idx]))
                correct_range_flag = True
                warnings.warn("The input calculated based on the upper limit of the output {} "
                              "range is more than 4096, and the upper limit of the input {} range is corrected "
                              "as {}".format(dim_name, dim_name, dx_upper))
        return dx_lower, dx_upper, correct_range_flag


class DeconvolutionParaProcess(Conv2dBackpropParaProcess):
    """
    class of param check and preprocess for dynamic deconvolution
    """

    def __init__(self, paras):
        super().__init__(paras)
        self.op_type = "deconvolution"
        self.out_backprop = paras.get("x")
        self.valid_paras = {
            "nhw_min": 1,
            "hw_max": 4096,
            "valid_format": {"weights": ("NCHW",),
                             "input": ("NCHW",),
                             "output": ("NCHW",)},
            "valid_dtype": ("float16", "float32",)
        }

    def check_support_valid(self, in_shape, w_shape):
        """
        check whether dynamic shape is supported for deconvolution
        """
        super().check_support_valid(in_shape, w_shape)
        if self.paras.get("offset_w"):
            err_man.raise_err_specific_user(
                self.op_type, "offset_w is not supported in dynamic shape yet.")
        if self.paras.get("bias"):
            err_man.raise_err_specific_user(
                self.op_type, "bias is not supported in dynamic shape yet.")
        if self.paras.get("offset_x") != 0:
            err_man.raise_err_specific_user(
                self.op_type, "offset_x is not supported in dynamic shape yet.")

    def get_input_range(self, w_shape, dy_range, dx_range=()):
        """
        calculate input range
        """

        def _get_input(y_in, k_size, pads, stride, dilation):
            if not y_in:
                return y_in
            return stride * (y_in - 1) + dilation * (k_size - 1) + 1 - pads[0] - pads[1]

        def _get_output(x_in, k_size, pads, stride, dilation):
            if not x_in:
                return x_in
            if DYNAMIC_FLAG in pads:
                return ceil_div(x_in, stride)
            return (x_in + pads[0] + pads[1] - dilation * (k_size - 1) - 1) // stride + 1

        correct_range_flag = False
        new_dy_range = copy.deepcopy(dy_range)

        dx_h_lower = _get_input(dy_range[H_DIM][0], w_shape[H_DIM],
                                  (self.pads[0], self.pads[1]), self.strides[H_DIM],
                                  self.dilations[H_DIM])
        if dx_h_lower < self.valid_paras.get("nhw_min"):
            dx_h_lower = max(dx_h_lower, self.valid_paras.get("nhw_min"))
            new_dy_range[H_DIM] = (_get_output(dx_h_lower, w_shape[H_DIM],
                                    (self.pads[0], self.pads[1]), self.strides[H_DIM],
                                    self.dilations[H_DIM]), new_dy_range[H_DIM][1])
            correct_range_flag = True
            warnings.warn("The input calculated based on the lower limit of the output h " + \
                "range is less than 1, and the lower limit of the input h range is corrected " + \
                "as {}".format(dx_h_lower))
        if not dy_range[H_DIM][1]:
            dx_h_upper = dy_range[H_DIM][1]
        else:
            dx_h_upper = _get_input(dy_range[H_DIM][1], w_shape[H_DIM],
                                      (self.pads[0], self.pads[1]), self.strides[H_DIM],
                                      self.dilations[H_DIM])
            if dx_h_upper > self.valid_paras.get("hw_max"):
                dx_h_upper = min(dx_h_upper, self.valid_paras.get("hw_max"))
                new_dy_range[H_DIM] = (new_dy_range[H_DIM][0], _get_output(dx_h_upper, w_shape[H_DIM],
                                        (self.pads[0], self.pads[1]), self.strides[H_DIM],
                                        self.dilations[H_DIM]))
                correct_range_flag = True
                warnings.warn("The input calculated based on the upper limit of the output h " + \
                    "range is more than 4096, and the upper limit of the input h range is corrected " + \
                    "as {}".format(dx_h_upper))
        dx_w_lower = _get_input(dy_range[W_DIM][0], w_shape[W_DIM],
                                  (self.pads[2], self.pads[3]), self.strides[W_DIM],
                                  self.dilations[W_DIM])
        if dx_w_lower < self.valid_paras.get("nhw_min"):
            dx_w_lower = max(dx_w_lower, self.valid_paras.get("nhw_min"))
            new_dy_range[W_DIM] = (_get_output(dx_w_lower, w_shape[W_DIM],
                                    (self.pads[2], self.pads[3]), self.strides[W_DIM],
                                    self.dilations[W_DIM]), new_dy_range[W_DIM][1])
            correct_range_flag = True
            warnings.warn("The input calculated based on the lower limit of the output w " + \
                "range is less than 1, and the lower limit of the input w range is corrected " + \
                "as {}".format(dx_w_lower))
        if not dy_range[W_DIM][1]:
            dx_w_upper = dy_range[W_DIM][1]
        else:
            dx_w_upper = _get_input(dy_range[W_DIM][1], w_shape[W_DIM],
                                      (self.pads[2], self.pads[3]), self.strides[W_DIM],
                                      self.dilations[W_DIM])
            if dx_w_upper > self.valid_paras.get("hw_max"):
                dx_w_upper = min(dx_w_upper, self.valid_paras.get("hw_max"))
                new_dy_range[W_DIM] = (new_dy_range[W_DIM][0], _get_output(dx_w_upper, w_shape[W_DIM],
                                        (self.pads[2], self.pads[3]), self.strides[W_DIM],
                                        self.dilations[W_DIM]))
                correct_range_flag = True
                warnings.warn("The input calculated based on the upper limit of the output w " + \
                    "range is more than 4096, and the upper limit of the input w range is corrected " + \
                    "as {}".format(dx_w_upper))
        if dx_h_upper and dx_h_lower > dx_h_upper:
            dx_h_lower = dx_h_upper
        if dx_w_upper and dx_w_lower > dx_w_upper:
            dx_w_lower = dx_w_upper
        if dx_range:
            return [dx_range[N_DIM], dx_range[C_DIM], (dx_h_lower, dx_h_upper), (dx_w_lower, dx_w_upper)]
        return [dy_range[N_DIM], (w_shape[C_DIM], w_shape[C_DIM]),
                (dx_h_lower, dx_h_upper), (dx_w_lower,  dx_w_upper)], correct_range_flag, new_dy_range

    def infer_shape_and_range(self):
        """
        infer range from dy to dx
        """

        self.check_input_dict(self.out_backprop, "out_backprop", True)

        dy_shape = list(self.out_backprop.get("ori_shape"))
        dy_range = self.out_backprop.get("range")
        filter_shape = self.filters.get("ori_shape")
        dx_shape = self.y.get("ori_shape")
        self.check_para_dim(dx_shape, "input_size")
        self.check_para_dim(filter_shape, "filters")
        self.check_pads(self.op_type)
        filter_shape_nchw = self.get_input_nchw(filter_shape, self.filters.get("ori_format"))
        self.get_attr_nchw(self.data_format)
        dx_shape_nchw = self.get_input_nchw(dx_shape, self.data_format)

        if self.check_unknown_scene(dy_shape, dx_shape_nchw, filter_shape_nchw[C_DIM] * self.groups):
            dy_shape_nchw = [DYNAMIC_FLAG, filter_shape_nchw[N_DIM], DYNAMIC_FLAG, DYNAMIC_FLAG]
            dy_range_nchw = [(1, None), None, (1, None), (1, None)]
            dx_range_nchw = [(1, None), None, (1, None), (1, None)]
            self.check_support_valid(dy_shape_nchw, filter_shape_nchw)
            group_para = comm.calculate_group(
                dy_shape_nchw, dx_shape_nchw, filter_shape_nchw, self.groups, self.dtype, self.data_format)
        else:
            self.check_para_dim(dy_shape, "out_backprop_shape")
            dy_shape_nchw, dy_range_nchw = self.get_input_nchw(dy_shape, self.data_format, dy_range)
            output_range = copy.deepcopy(dy_range_nchw)
            self.check_support_valid(dy_shape_nchw, filter_shape_nchw)
            self.check_dynamic_channel_scene(
                dy_shape_nchw, dx_shape_nchw, filter_shape_nchw[N_DIM])
            group_para = comm.calculate_group(
                dy_shape_nchw, dx_shape_nchw, filter_shape_nchw, self.groups, self.dtype, self.data_format)
            self.check_range_valid(dy_shape_nchw, output_range, "out_backprop", self.data_format)
            if output_range[W_DIM][1]:
                if filter_shape_nchw[H_DIM] == 1 and filter_shape_nchw[W_DIM] == 1:
                    output_range[W_DIM] = (output_range[W_DIM][0],
                                           output_range[W_DIM][1] * self.strides[H_DIM] * self.strides[W_DIM])
            self.check_range_valid(dy_shape_nchw, output_range, "out_backprop", self.data_format)
        dx_range_nchw, correct_range_flag, dy_range_nchw = self.get_input_range(filter_shape_nchw, dy_range_nchw)

        self.check_range_valid(dx_shape_nchw, dx_range_nchw, "input_size", self.data_format)

        self.shape = {
            "dy_shape_nchw": dy_shape_nchw,
            "filter_shape_nchw": filter_shape_nchw,
            "dx_shape_nchw": dx_shape_nchw
        }
        self.range = {"dy_range_nchw": dy_range_nchw, "dx_range_nchw": dx_range_nchw}
        self.attrs = {"group_para": group_para, "correct_range_flag": correct_range_flag}

    def config_paras(self):
        """
        check original para
        """
        if len(self.strides) != FORMAT_HW_DIM:
            err_man.raise_err_specific_user(
                self.op_type, "length of stride in deconvolution should be 2.")
        self.strides = [1, 1, self.strides[H_DIM_2D], self.strides[W_DIM_2D]]

        self.check_paras()
        self.infer_shape_and_range()
        self._config_shape()

        self.tensors["x_tensor"] = tvm.placeholder(self.shape.get("dy_shape_nc1hwc0"), name="dedy", dtype=self.dtype)
        self.tensors["filter_tensor"] = tvm.placeholder(self.shape.get("filter_shape_frac_z"),
                                                        name="filter", dtype=self.dtype)

    def check_paras(self):
        super().check_paras()
        self.binary_mode = 0


class DepthwiseConv2dBackpropParaProcess(Conv2dBackpropParaProcess):
    """
    class of param check and preprocess for dynamic depthwise_conv2d_backprop_input
    """

    def __init__(self, paras):
        super().__init__(paras)
        self.op_type = "depthwise_conv2d_backprop_input"
        self.y = paras.get("input_grad")
        self.is_unknow_scene = False

    def infer_shape_and_range(self):
        """
        infer range from dx to dy
        """
        self.check_input_dict(self.y, "y", False)

        dy_shape = self.out_backprop.get("ori_shape")
        filter_shape = self.filters.get("ori_shape")
        dx_shape = self.y.get("ori_shape")

        self.check_para_dim(dx_shape, "input_size")
        self.check_para_dim(filter_shape, "filters")
        self.check_pads(self.op_type)
        self.get_attr_nchw(self.data_format)

        filter_shape_kchw = self.get_input_nchw(filter_shape, self.filters.get("ori_format"))

        if filter_shape_kchw[C_DIM] != 1:
            err_man.raise_err_specific_user(self.op_type, "not supported K != 1 in dynamic now!")
        filter_shape_nchw =  [filter_shape_kchw[N_DIM] * filter_shape_kchw[C_DIM], 1] + filter_shape_kchw[2:]
        dx_shape_nchw = self.get_input_nchw(dx_shape, self.data_format)
        groups = dx_shape_nchw[C_DIM]

        if self.check_unknown_scene(dy_shape, dx_shape_nchw, filter_shape_kchw[N_DIM]):
            dy_shape_nchw = [DYNAMIC_FLAG, filter_shape_nchw[N_DIM], DYNAMIC_FLAG, DYNAMIC_FLAG]
            dy_range_nchw = [(1, None), None, (1, None), (1, None)]
            dx_range_nchw = [(1, None), None, (1, None), (1, None)]

            self.check_support_valid(dy_shape_nchw, filter_shape_nchw)
            group_para = comm.calculate_group(
                dy_shape_nchw, dx_shape_nchw, filter_shape_nchw, groups, self.dtype, self.data_format)
        else:
            self.check_para_dim(dy_shape, "out_backprop_shape")
            self.check_input_dict(self.y, "y", True)
            dx_range = self.y.get("range")
            dy_shape_nchw = self.get_input_nchw(dy_shape, self.data_format)
            dx_shape_nchw, dx_range_nchw = self.get_input_nchw(dx_shape, self.data_format, dx_range)
            self.check_support_valid(dy_shape_nchw, filter_shape_nchw)
            self.check_dynamic_channel_scene(
                dy_shape_nchw, dx_shape_nchw, filter_shape_nchw[N_DIM])
            group_para = comm.calculate_group(
                dy_shape_nchw, dx_shape_nchw, filter_shape_nchw, groups, self.dtype, self.data_format)
            self.check_range_valid(dx_shape_nchw, dx_range_nchw, "input_size", self.data_format)

        dy_range_nchw, correct_range_flag, dx_range_nchw = self.get_output_range(
            filter_shape_nchw, dx_range_nchw)

        output_range = copy.deepcopy(dy_range_nchw)
        if output_range[W_DIM][1]:
            if filter_shape_nchw[H_DIM] == 1 and filter_shape_nchw[W_DIM] == 1:
                output_range[W_DIM] = (output_range[W_DIM][0],
                                       output_range[W_DIM][1] * self.strides[H_DIM] * self.strides[W_DIM])
        self.check_range_valid(dy_shape_nchw, output_range, "out_backprop", self.data_format)

        self.shape = {"dy_shape_nchw": dy_shape_nchw, "filter_shape_nchw": filter_shape_nchw,
                      "dx_shape_nchw": dx_shape_nchw}
        self.range = {"dy_range_nchw": dy_range_nchw, "dx_range_nchw": dx_range_nchw}
        self.attrs = {"group_para": group_para, "correct_range_flag": correct_range_flag}

    def config_paras(self):
        """
        config paras and placeholders
        """
        self.check_paras()
        self.infer_shape_and_range()
        self._config_shape()
        self.tensors["input_tensor"] = tvm.placeholder([4], name="input_size", dtype="int32")
        self.tensors["dy_tensor"] = tvm.placeholder(self.shape.get("dy_shape_nc1hwc0"),
                                                    name="dedy", dtype=self.dtype)
        self.tensors["filter_tensor"] = tvm.placeholder(self.shape.get("filter_shape_frac_z"),
                                                        name="filter", dtype=self.dtype)

    def check_paras(self):
        super().check_paras()
        self.binary_mode = 0


class AvgPoolGradParaProcess(Conv2dBackpropParaProcess):
    """
    class of param check and preprocess for dynamic avg_pool_grad
    """

    def __init__(self, paras):
        super().__init__(paras)
        self.op_type = "avg_pool_grad"

    def check_paras(self):
        super().check_paras()
        self.binary_mode = 0


class Conv3dBackpropParaProcess():
    """
    class of param check and preprocess for dynamic conv3d_backprop_input
    """
    def __init__(self, para_dict, pad_mode):
        self.para_dict = para_dict
        self.pad_mode = pad_mode
        self.strides = para_dict.get("strides") # ndhwc
        self.pads = para_dict.get("pads")
        self.dilations = para_dict.get("dilations") # ndhwc
        self.groups = para_dict.get("groups")
        self.filter = para_dict.get("ori_tensors").get("filter")
        self.out_backprop = para_dict.get("ori_tensors").get("out_backprop")
        self.y = para_dict.get("ori_tensors").get("y")
        self.input_size = para_dict.get("ori_tensors").get("input_size")


    def get_dx_ori_range(self, dy_ori_range : list) -> list:
        """
        get dx_ori_range according to dy_ori_range
        """

        _, shape_filter_ndhwc = get_idx_shape_from_format(self.filter["ori_format"],
                                                           self.filter["ori_shape"])
        idx_y_ndhwc, _ = get_idx_shape_from_format(self.y["ori_format"],
                                                      self.y["ori_shape"])
        idx_out_backprop_ndhwc, _ = get_idx_shape_from_format(self.out_backprop["ori_format"],
                                                                                      self.out_backprop["ori_shape"])
        _, filter_d, filter_h, filter_w, filter_c = shape_filter_ndhwc
        idx_out_backprop_n, idx_out_backprop_d, idx_out_backprop_h, idx_out_backprop_w, _ = idx_out_backprop_ndhwc
        idx_y_n, idx_y_d, idx_y_h, idx_y_w, idx_y_c = idx_y_ndhwc
        stride_d, stride_h, stride_w = \
            self.strides[1], self.strides[2], self.strides[3]
        dilations_d, dilations_h, dilations_w = \
            self.dilations[1], self.dilations[2], self.dilations[3]
        pad_front, pad_back, pad_up, pad_down, pad_left, pad_right = \
            self.pads[0], self.pads[1], self.pads[2], self.pads[3], self.pads[4], self.pads[5]
        kdext = (filter_d - 1) * dilations_d + 1
        khext = (filter_h - 1) * dilations_h + 1
        kwext = (filter_w - 1) * dilations_w + 1

        dx_ori_range = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

        if len(dy_ori_range) == _K_DIM_SIZE:
            dx_ori_range[idx_y_n] = dy_ori_range[idx_out_backprop_n]
            attr_param_d = [stride_d, kdext, pad_front + pad_back]
            self._set_conv3dx_dim_range([idx_y_d, idx_out_backprop_d], attr_param_d, dx_ori_range, dy_ori_range)
            attr_param_h = [stride_h, khext, pad_up + pad_down]
            self._set_conv3dx_dim_range([idx_y_h, idx_out_backprop_h], attr_param_h, dx_ori_range, dy_ori_range)
            attr_param_w = [stride_w, kwext, pad_left + pad_right]
            self._set_conv3dx_dim_range([idx_y_w, idx_out_backprop_w], attr_param_w, dx_ori_range, dy_ori_range)
            dx_ori_range[idx_y_c] = (filter_c * self.groups, filter_c * self.groups)
        return dx_ori_range

    def _set_conv3dx_dim_range(self, pos : list, attr_param : list, dx_ori_range : list, dy_ori_range : list) -> None:
        """

        Parameters
        ----------
        pos: dx dy pos
        attr_param: shape of weight
        dx_ori_range: H direction padding
        dy_ori_range: W direction padding

        Returns
        -------
        None
        """


        dx_pos, dy_pos = pos
        stride, kernel, pad = attr_param[0], attr_param[1], attr_param[2]
        low, high = dy_ori_range[dy_pos][0], dy_ori_range[dy_pos][1]
        if self.pad_mode == "VAR":
            range_low = stride * (low - 1) + 1
            range_high = stride * high
        else:
            range_low = stride * (low - 1) + kernel - pad
            range_high = stride * (high - 1) + kernel - pad + stride - 1
        dx_ori_range[dx_pos] = (max(range_low, _K_MIN_RANGE), min(range_high, _K_MAX_RANGE))


def define_operation_var_of_dx(need_define_input_vars):
    # shape vars
    # NOTE always create variables in the order filter, out_backprop in shape_util.py
    if need_define_input_vars:
        # scenerio: single op
        # filter: ci1g*hk*wk, co1g, co0, ci0
        operation.var("filter_ci1hw")  # ci1g*hk*wk
        operation.var("filter_col")  # co1g
        # out_backprop: batch, co1, ho, wo, co0
        operation.var("batch")  # batch
        operation.var("dedy_c1") # co1
        operation.var("dedy_h")  # ho
        operation.var("dedy_w")  # wo
    # y shape: batch, ci1, hi, wi, ci0 ori_shape: batch, ci, hi, wi
    operation.var("dx_c")  # ci
    operation.var("dx_c1")  # ci1
    operation.var("dx_h")  # hi
    operation.var("dx_w")  # wi
    operation.var("kernel_h")  # hk
    operation.var("kernel_w")  # wk
    operation.var("g_extend")
    operation.var("dx_c1_extend")  # ci1g

    # attr var
    operation.var("padt")
    operation.var("padb")
    operation.var("padl")
    operation.var("padr")
    operation.var("stride_h")
    operation.var("stride_w")
    operation.var("dilation_h")
    operation.var("dilation_w")
    operation.var("shape_up_modify")
    operation.var("shape_left_modify")
    operation.var("shape_down_modify")
    operation.var("shape_right_modify")
    operation.var("load3d_special")
    operation.var("pad_up_before")
    operation.var("pad_left_before")
    operation.var("pad_down_after")
    operation.var("pad_right_after")
    operation.var("bias_flag")
    operation.var("hf32_flag")

    # tiling var
    operation.var("group_dim")
    operation.var("batch_dim")
    operation.var("n_dim")
    operation.var("m_dim")
    operation.var("batch_single_core")
    operation.var("m_al1")
    operation.var("n_bl1")
    operation.var("k_aub")
    operation.var("m_aub")
    operation.var("wo_aub")
    operation.var("m_l0")
    operation.var("n_l0_div_ub")
    operation.var("n_ub")
    operation.var("k_l0")
    operation.var("min_kl1_div_kl0")
    operation.var("max_kl1_div_min_kl1")
    operation.var("k_div_max_kl1")
    operation.var("al1_bound")
    operation.var("bl1_bound")
    operation.var("aub_bound")
    operation.var("bias_table_bound")