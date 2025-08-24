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
util_common
"""
import os
import json
import itertools
import math
from tbe.common.utils.errormgr import error_manager_util
from tbe import tvm
import tbe.dsl as tbe
from .platform_adapter import tbe_platform

PAD_MIN = 0
# the dim of most parameters in conv3d is 5
CONV3D_SHAPE_COMMON_DIM = 5
# The constant be used to modulus an uint8 data
UINT8_MOD_VAL = 256
# The constant be used to modulus a int8 data
INT8_MOD_VAL = 128
MININUM_NUM_FLOAT = 2 ** (-126)
SCALAR_MUL1_FP32 = 2 ** (50)
SCALAR_MUL2_FP32 = 2 ** (26)

# the bytes length of several dtype
BIT_RATIO_DICT = {"int32": 4, "float32": 4, "float16": 2,
                  "uint8": 1, "int8": 1, "uint4": 0.5, "int4": 0.5}

def ceil(x_1, x_2):
    """
    do ceiling division

    Parameters
    ----------
    x_1: int
    x_2: int
    Returns
    -------
    result
    """
    if x_2 == 0:
        dict_args = {
            'errCode': 'E62502',
            'first_operand': str(x_1),
            'second_operand': str(x_2),
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))
    return (x_1 + x_2 - 1) // x_2


def check_pads_value_3d(pads):
    """
    check_pads_value_3d
    """
    pad_head, pad_tail, pad_up, pad_down, pad_left, pad_right = pads
    if pad_head < PAD_MIN or pad_tail < PAD_MIN:
        dict_args = {
            'errCode': 'E60000',
            'param_name': 'pad D',
            'expected_value': 'non_negative vlaue',
            'input_value': 'pad_d[0] = {}, pad_d[1] = {}'.format(pad_head, pad_tail)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if pad_up < PAD_MIN or pad_down < PAD_MIN:
        dict_args = {
            'errCode': 'E60000',
            'param_name': 'pad H',
            'expected_value': 'non_negative vlaue',
            'input_value': 'pad_h[0] = {}, pad_h[1] = {}'.format(pad_up, pad_down)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))
    if pad_left < PAD_MIN or pad_right < PAD_MIN:
        dict_args = {
            'errCode': 'E60000',
            'param_name': 'pad W',
            'expected_value': 'non_negative vlaue',
            'input_value': 'pad_w[0] = {}, pad_w[1] = {}'.format(pad_left, pad_right)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))


def align(x_1, x_2):
    """
    align x_1 with x_2

    Parameters
    ----------
    x_1: int
    x_2: int
    Returns
    -------
    result
    """
    if x_2 == 0:
        dict_args = {
            'errCode': 'E62502',
            'first_operand': str(x_1),
            'second_operand': str(x_2),
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))
    return ((x_1 + x_2 - 1) // x_2) * x_2


def write_code(wkspace_dict, kernel_name):
    """
    write workspaces to json file

    """
    fname = tbe_platform.get_current_build_config + "/" + kernel_name + ".json"
    fname = os.path.realpath(fname)
    if fname.startswith(os.getcwd()):
        if os.path.exists(fname):
            with open(fname, "r") as f_var:
                load_dict = json.load(f_var)
            load_dict.update(wkspace_dict)
            with open(fname, "w") as f_var:
                json.dump(load_dict, f_var, sort_keys=True,
                          indent=4, separators=(',', ':'))


def lcm(param1, param2):
    """
    calculate least common multiple
    """
    temp = param1 * param2
    while param1 % param2 != 0:
        param1, param2 = param2, param1 % param2

    return temp // param2


def calculate_group(fmap_c, cout, groups, cout0, cin0):
    """
    calculate groups parameter
    """
    # check group
    if groups <= 0 or groups > fmap_c or groups > cout:
        dict_args = {
            'errCode': 'E60038',
            'desc': "Group must not be larger than x channel and filter channel"
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if fmap_c % groups != 0 or cout % groups != 0:
        dict_args = {
            'errCode': 'E60038',
            'desc': "Feature map's or filter's channel must be divisible by group"
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    mag_factor0 = lcm(fmap_c // groups, cin0) // (fmap_c // groups)
    mag_factor1 = lcm(cout // groups, cout0) // (cout // groups)
    mag_factor = min(lcm(mag_factor0, mag_factor1), groups)

    cin1_g = (mag_factor * fmap_c // groups + cin0 - 1) // cin0
    cout_g = (mag_factor * cout // groups + cout0 - 1) // cout0 * cout0

    group_dict = {"real_g": (groups + mag_factor - 1) // mag_factor,
                  "mag_factor": mag_factor,
                  "cin1_g": cin1_g,
                  "cout_g": cout_g,
                  "cin_ori": fmap_c,
                  "cout_ori": cout}

    return group_dict


def update_axis_for_other_format(ori_shape, axis, input_format, ori_format, reduce_mode=False):
    """
    update_axis_for_other_format: when format is changed, the axis will be updated
    """
    if input_format in ("NDC1HWC0", "NC1HWC0"):
        ori_shape_len = len(ori_shape) if -2 not in ori_shape else len(ori_format)
        axis = axis % ori_shape_len
        # ex: ori axis with N, axis = 0
        # ex: ori axis with D, axis = 1
        # ex: ori axis with C, axis = 1 (NC1HWC0) 2(NDC1HWC0)
        # ex: ori axis with H, axis = 2 (NC1HWC0) 3(NDC1HWC0)
        # ex: ori axis with W, axis = 3 (NC1HWC0) 4(NDC1HWC0)
        offset_6hd = 1 if input_format == "NDC1HWC0" else 0
        format_c_axis = 1 + offset_6hd if not reduce_mode else [1 + offset_6hd, 4 + offset_6hd]
        format_axis_map = {
            "N": 0,
            "C": format_c_axis,
            "H": 2 + offset_6hd,
            "W": 3 + offset_6hd,
            "D": 1
        }
        concat_dim_name = ori_format[axis]
        axis = format_axis_map[concat_dim_name]

    if input_format in ("FRACTAL_NZ",):
        axis = axis % len(ori_shape)
        # when FRACTAL_NZ, mean: [A, B, C, D] -> [A, B, ceil(D//16), ceil(C//16), 16, 16]
        # update axis as follow:
        # ex: ori axis with last one dim, axis = the dim of ceil(D//16)
        # ex: ori axis with last second dim, axis = the dim of ceil(C//16)
        if axis == len(ori_shape) - 1:
            axis = len(ori_shape) - 2 if not reduce_mode else [len(ori_shape) - 2, len(ori_shape) + 1]
        elif axis == len(ori_shape) - 2:
            axis = len(ori_shape) - 1 if not reduce_mode else [len(ori_shape) - 1, len(ori_shape) + 0]

    if input_format in ("FRACTAL_Z", "FRACTAL_Z_3D"):
        axis = axis % len(ori_shape)
        # when FRACTAL_Z, mean: C1HWNiNoC0
        # when FRACTAL_Z_3D, mean: DC1HWNiNoC0
        offset_3d = 1 if input_format == "FRACTAL_Z_3D" else 0
        format_c_axis = 0 + offset_3d if not reduce_mode else [0 + offset_3d, 5 + offset_3d]
        format_n_axis = 3 + offset_3d if not reduce_mode else [3 + offset_3d, 4 + offset_3d]
        format_axis_map = {
            "N": format_n_axis,
            "C": format_c_axis,
            "H": 1 + offset_3d,
            "W": 2 + offset_3d,
            "D": 0
        }
        concat_dim_name = ori_format[axis]
        axis = format_axis_map[concat_dim_name]

    return axis


def update_shape_base_other_format(input_dict):
    """
    update_axis_for_other_format: when format is changed, the axis will be updated
    """
    ori_shape = input_dict.get("ori_shape")
    ori_format = input_dict.get("ori_format")
    input_shape = input_dict.get("shape")
    input_format = input_dict.get("format")

    if input_format in ("FRACTAL_Z", "FRACTAL_Z_3D"):
        # when FRACTAL_Z, mean: C1HWNiNoC0
        # when FRACTAL_Z_3D, mean: DC1HWNiNoC0
        if len(input_shape) == 4:
            # fe will reshape the C1HWNiNoC0/DC1HWNiNoC0 to 4s = (C1HW)NiNoC0/(DC1HW)NiNoC0
            # now will reshape to 6d/7d = C1HWNiNoC0/DC1HWNiNoC0
            dict_zip_shape = dict(zip(list(ori_format), ori_shape))
            shape_h_dim = dict_zip_shape["H"]
            shape_w_dim = dict_zip_shape["W"]

            shape_c1_dim = input_shape[0] // (shape_h_dim * shape_w_dim)
            new_shape = [shape_c1_dim, shape_h_dim, shape_w_dim] + list(input_shape[1:])
            if input_format == "FRACTAL_Z_3D":
                shape_d_dim = dict_zip_shape["D"]
                shape_c1_dim = new_shape[0] // shape_d_dim
                new_shape = [shape_d_dim] + [shape_c1_dim, shape_h_dim, shape_w_dim] + list(input_shape[1:])

            input_dict["shape"] = new_shape

    return input_dict


# pylint: disable=too-many-locals
def update_shape_base_other_format_dynamic(input_dict):
    """
    update_axis_for_other_format_dynamic: when format is changed, the axis will be updated
    """
    ori_shape = input_dict.get("ori_shape")
    ori_format = input_dict.get("ori_format")
    input_shape = input_dict.get("shape")
    input_format = input_dict.get("format")
    input_range = input_dict.get("range")

    if input_format in ("FRACTAL_Z", "FRACTAL_Z_3D"):
        # when FRACTAL_Z, mean: C1HWNiNoC0
        # when FRACTAL_Z_3D, mean: DC1HWNiNoC0
        if len(input_shape) == 4:
            # fe will reshape the C1HWNiNoC0/DC1HWNiNoC0 to 4s = (C1HW)NiNoC0/(DC1HW)NiNoC0
            # now will reshape to 6d/7d = C1HWNiNoC0/DC1HWNiNoC0
            dict_zip_shape = dict(zip(list(ori_format), ori_shape))
            shape_h_dim = dict_zip_shape["H"]
            shape_w_dim = dict_zip_shape["W"]

            if shape_h_dim <= 0 or shape_w_dim <= 0 or input_shape[0] <= 0:
                shape_c1_dim = -1
                temp_range = [(1, None)]
                if shape_h_dim > 0 and shape_w_dim > 0:
                    upper = None if input_range[0][1] is None else int(
                        math.ceil(input_range[0][1] / (shape_h_dim * shape_w_dim)))
                    lower = 1 if int(math.floor(input_range[0][0] / (shape_h_dim * shape_w_dim))) == 0 else int(
                        math.floor(input_range[0][0] / (shape_h_dim * shape_w_dim)))
                    temp_range = [(lower, upper)]
            else:
                shape_c1_dim = input_shape[0] // (shape_h_dim * shape_w_dim)
                temp_range = [(shape_c1_dim, shape_c1_dim)]

            for dim in [shape_h_dim, shape_w_dim]:
                temp_range.append((1, None) if dim == -1 else (dim, dim))
            input_range = temp_range + list(input_range[1:])

            new_shape = [shape_c1_dim, shape_h_dim, shape_w_dim] + list(input_shape[1:])

            if input_format == "FRACTAL_Z_3D":
                shape_d_dim = dict_zip_shape["D"]
                if shape_d_dim <= 0 or new_shape[0] <= 0:
                    shape_c1_dim = -1
                    temp_range = [(1, None)]
                    if shape_d_dim > 0:
                        upper = None if input_range[0][1] is None else int(math.ceil(input_range[0][1] / shape_d_dim))
                        lower = 1 if int(math.floor(input_range[0][0] / shape_d_dim)) == 0 else int(
                            math.floor(input_range[0][0] /shape_d_dim))
                        temp_range = [(lower, upper)]
                else:
                    shape_c1_dim = new_shape[0] // shape_d_dim
                    temp_range = [(shape_c1_dim, shape_c1_dim)]

                temp_range.insert(0, (1, None) if shape_d_dim == -1 else (shape_d_dim, shape_d_dim))
                input_range = temp_range + input_range[1:]
                new_shape = [shape_d_dim] + [shape_c1_dim, shape_h_dim, shape_w_dim] + list(input_shape[1:])

            input_dict["shape"] = new_shape
            input_dict["range"] = input_range

    return input_dict


def get_fused_format_str(format_char_list):
    """
    get_fused_format from char
    ex:
        input  ["N", "C", "H"]
        putput ["NCH", "NHC", "CNH", "CHN", "HNC", "HCN"]
    """
    format_iter = itertools.permutations(format_char_list, len(format_char_list))
    format_char_list = list(format_iter)
    format_str_list = []
    for _, char_list in enumerate(format_char_list):
        format_str_list.append(''.join(list(char_list)))

    return format_str_list


# pylint: disable=invalid-name
def is_dynamic_input(_inputs):
    """
    is_dynamic_input: check whether the shape contain -1
        contain -1 return True else False

    Parameters
    ----------
    _inputs: list of dict/tuple of dict/dict

    Returns
    -------
    bool
    """
    if not isinstance(_inputs, list) and not isinstance(_inputs, tuple):
        _inputs = [_inputs]

    for _, _input in enumerate(_inputs):
        if -1 in _input.get("shape") or -1 in _input.get("ori_shape"):
            return True

    return False


def is_unknown_rank_input(input_list):
    """
    is_unknown_rank_input: check whether the shape contain -2
        contain -2 return True else False

    Parameters
    ----------
    input_list: list of dict/tuple of dict/dict

    Returns
    -------
    bool
    """
    if not isinstance(input_list, list) and not isinstance(input_list, tuple):
        input_list = [input_list]

    for _, _input in enumerate(input_list):
        if -2 in _input.get("shape") or -2 in _input.get("ori_shape"):
            return True

    return False


def is_unknown(_inputs):
    """
    is_unknown: check whether the shape contain -1 or -2
        contain -1 or -2 return True else False

    Parameters
    ----------
    input_list: list of dict/tuple of dict/dict

    Returns
    -------
    bool
    """
    if is_dynamic_input(_inputs):
        return True

    if is_unknown_rank_input(_inputs):
        return True

    return False


def high_precision_floor(x_float, is_high_precision=True):
    """
    Calculate the maximum integer less than x_float
    Parameters
    ----------
    x_float: a float tensor
    is_high_precision: a bool value
    Returns
    ----------
    a tensor
    """
    x_floor = tbe.floor(x_float)
    if x_float == "float16":
        is_high_precision = False
    if is_high_precision:
        cast_flag = tbe_platform.api_check_support("tik.vconv", "s322f32")
        if not cast_flag:
            # when the product not support f322s32, will cast to fp16 and to int32, will get error
            # ex: f32 value is 1.99998, cast int32 is 2, this step will reduce the error
            # step 1 int32 cast to fp32_new   2.0
            # step 2 int32_sub_fp32_value = f32_old - fp32_new
            # step 3 int32_sub_fp32_value = 0 when int32_sub_fp32_value >= 0
            #        int32_sub_fp32_value = 1 when int32_sub_fp32_value < 0
            # step 4 int32 - int32_sub_fp32_value
            x_floor_fp32 = tbe.cast_to(x_floor, "float32")
            error_fp32 = tbe.vsub(x_floor_fp32, x_float)
            error_fp32 = tbe.vmaxs(error_fp32, 0)
            error_fp32 = tbe.vmins(error_fp32, MININUM_NUM_FLOAT)
            error_fp32 = tbe.vmuls(error_fp32, SCALAR_MUL1_FP32)
            error_fp32 = tbe.vmuls(error_fp32, SCALAR_MUL1_FP32)
            error_fp32 = tbe.vmuls(error_fp32, SCALAR_MUL2_FP32)
            error_fp16 = tbe.cast_to(error_fp32, "float16")
            error_int32 = tbe.cast_to(error_fp16, "int32")
            x_floor = tbe.vsub(x_floor, error_int32)
    return x_floor


def tensor_mod_int(x, y):
    """
    Calculate x mod y
    Parameters
    ----------
    x: a tensor
    y: a interger number
    Returns
    ----------
    x mod y
    """
    y_rec = tvm.const(1 / float(y), dtype=x.dtype.lower())
    x_float = tbe.vmuls(x, y_rec)
    x_floor = high_precision_floor(x_float)
    x_int = tbe.vmuls(x_floor, y)
    x_int = tbe.cast_to(x_int, x.dtype.lower())
    x = tbe.vsub(x, x_int)
    return x


def uint8_int8_overflow_proc(x, x_dtype):
    """
    Calculate the right result when input is overflow and its dtype is uint8 or int8
    Parameters
    ----------
    x: a tensor
    x_dtype: a string, dtype of the x
    Returns
    ----------
    a tensor after process
    """
    if x_dtype == "uint8":
        x = tensor_mod_int(x, UINT8_MOD_VAL)
    elif x_dtype == "int8":
        x = tbe.vadds(x, INT8_MOD_VAL)
        x = tensor_mod_int(x, UINT8_MOD_VAL)
        neg_int8_mod_val = -1 * INT8_MOD_VAL
        x = tbe.vadds(x, neg_int8_mod_val)
    x = tbe.cast_to(x, x_dtype)
    return x


def floor_div_scalar(input_num, align_factor):
    """
    floor_div_scalar

    Parameters
    ----------
    input_num: Scalar or Int
    align_factor: Scalar or Int(must be > 0)

    Returns
    -------
    result
    """
    return input_num // align_factor


def ceil_div_scalar(input_num, align_factor):
    """
    ceil_div_scalar

    Parameters
    ----------
    input_num: Scalar or Int
    align_factor: Scalar or Int(must be > 0)

    Returns
    -------
    result
    """
    return (input_num + align_factor - 1) // align_factor


def div_align_scalar(input_num, align_factor, div_mode="ceil"):
    """
    div_align_scalar

    Parameters
    ----------
    input_num: Scalar or Int
    align_factor: Scalar or Int(must be > 0)
    div_mode: str, div mode, when ceil, ceil div to  align

    Returns
    -------
    result
    """
    if div_mode == "ceil":
        return ceil_div_scalar(input_num, align_factor) * align_factor

    return floor_div_scalar(input_num, align_factor) * align_factor


def cal_mini_l1_size_matmul(dtype_w):
    """
    cal the mini l1 size of matmul, batchmatmul and fc
    """
    block_reduce = tbe_platform.CUBE_MKN[dtype_w]["mac"][1]
    block_out = tbe_platform.CUBE_MKN[dtype_w]["mac"][2]
    mini_l1space = block_out * block_reduce * BIT_RATIO_DICT.get(dtype_w)
    return mini_l1space


def is_support_fractal_z_input(_input):
    """
    is_unknown:check whether the operator supports FRACTAL_Z
               return True else False

    Parameters
    ----------
    _input:dict

    Returns
    -------
    bool
    """
    groups = _input.get("sub_format")
    if groups is None or groups == 0:
        return False
    ori_format = _input.get("ori_format")
    shape = _input.get("ori_shape")
    if len(ori_format) != len(shape):
        return False
    n_dim = shape[ori_format.index("N")]
    support_format = get_fused_str(["N", "C", "H", "W"])

    if ori_format not in support_format or \
            (n_dim % groups != 0) or ((n_dim // groups) % 16 != 0):
        return False

    return True


def is_support_fractal_z_inputs(_inputs):
    """
    is_unknown:check whether the operator supports FRACTAL_Z
               return True else False

    Parameters
    ----------
    _inputs:list of dict/tuple of dict/dict

    Returns
    -------
    bool
    """
    groups_first = _inputs[0].get("sub_format")
    if groups_first is None:
        return False
    for _input in _inputs:
        is_support_fractal = is_support_fractal_z_input(_input)
        groups = _input.get("sub_format")
        if not is_support_fractal or groups_first != groups:
            return False

    return True


def get_fused_str(format_char_list):
    """
        get_fused_str for format
    """
    format_iter = itertools.permutations(format_char_list, len(format_char_list))
    format_char_list = list(format_iter)
    format_str_list = []
    for i, char_list in enumerate(format_char_list):
        format_str_list.append(''.join(list(char_list)))

    return format_str_list


def is_same_group(_inputs):
    """
    is_unknown:check whether the operator supports same group
               return True else False

    Parameters
    ----------
    _inputs:list of dict/tuple of dict/dict

    Returns
    -------
    bool
    """
    groups_first =  _inputs[0].get("sub_format")
    if groups_first is None:
        return False
    for _input in _inputs:
        groups = _input.get("sub_format")
        if groups_first != groups:
            return False
    return True

def check_load3d_w_out_1_support():
    """
    check if current soc version load3d instruction support w_out==1 or not
    only Ascend310 and Hi3796CS support w_out==1
    when fmap_w(with padding) == filters_w(after dilation)
    -------

    Returns
    -------
    True: support
    False: not support
    """
    soc_version = tbe_platform.get_soc_spec("SHORT_SOC_VERSION")
    if soc_version in ["Ascend310", "Hi3796CV300CS"]:
        return True
    return False
