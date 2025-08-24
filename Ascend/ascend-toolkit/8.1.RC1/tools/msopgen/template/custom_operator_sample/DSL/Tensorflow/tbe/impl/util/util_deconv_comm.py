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
deconv_comm
provide common function used by conv2d_backprop_input and deconvlution
"""
from __future__ import absolute_import

from .platform_adapter import tvm
from .platform_adapter import tbe_platform
from .platform_adapter import para_check
from .platform_adapter import error_manager_util as err_man
from .platform_adapter import error_manager_cube as err_man_cube
from .platform_adapter import shape_util


# the dim of shape in conv_backprop must be 4
CONV_BACKPROP_SHAPE_DIM = 4
# the dim of strides in conv_backprop must be 2
STRIDES_SHAPE_DIM = 2
# the dim of pads in conv_backprop must be 4
PADDING_SHAPE_DIM = 4

# fmapW must be in [1,4096]
FMAP_HW_MIN = 1
FMAP_W_MAX = 4096
# fmapH must be in [1,200000]
FMAP_H_MAX = 200000

# DeDyW must be in [1,4096]
DEDY_HW_MIN = 1
DEDY_W_MAX = 4096
# DeDyH must be in [1,200000]
DEDY_H_MAX = 200000

# filterH, filterW must be >= 1
FILTER_HW_MIN = 1

# conv1d situation support w not larger than 2^31-1
CONV1D_W_MAX = 2147483647

# stride must be >= 1
STRIDE_HW_MIN = 1

# pad must be in [0, 255]
PAD_MIN = 0
PAD_MAX = 255

# dilation must be in [1, 255]
DILATION_MIN = 1
DILATION_MAX = 255

# the max num of each axis of shape
DEFAULT_MAX_SHAPE_NUM = 1000000

# the bytes length of several dtype
BIT_RATIO_DICT = {"int32": 4, "float32": 4, "float16": 2,
                  "uint8": 1, "int8": 1, "uint4": 0.5, "int4": 0.5}
# same as (2**63-1)
DATA_SIZE_MAX = 9223372036854775807

# If pads is string , only support "SAME" or "VALID"
PADDING_SUPPORT = ('SAME', 'VALID')
# pads valid mode to be [0, 0, 0, 0]
PADDING_VAILD = [0, 0, 0, 0]

# If dim is dynamic, the value of dim is -1
DYNAMIC_FLAG = -1

# The dim of N/H/W is 0/2/3, in the format NCHW
N_DIM = 0
H_DIM = 2
W_DIM = 3


class GroupDictKeys:
    """
    The keys of group_dict
    """
    groups = "groups"
    g_extend = "g_extend"
    multiple_extend = "multiple_extend"
    dx_c1_extend = "dx_c1_extend"
    dy_c1_extend = "dy_c1_extend"
    dx_c_ori = "dx_c_ori"
    dy_c_ori = "dy_c_ori"
    filter_batch_ori = "filter_batch_ori"
    filter_c_ori = "filter_c_ori"
    filter_ori_format = "filter_ori_format"


def get_filter_shape(ori_format_filters, ori_shape_filters):
    """
    Get filter shape of NCHW from original shape
    :param ori_format_filters:
    :param ori_shape_filters:
    :return: filter shape of NCHW
    """
    if ori_format_filters == "NCHW":
        shape_filters = ori_shape_filters
    elif ori_format_filters == "NHWC":
        shape_filters = (ori_shape_filters[0],
                         ori_shape_filters[3],
                         ori_shape_filters[1],
                         ori_shape_filters[2])
    elif ori_format_filters == "HWCN":
        shape_filters = (ori_shape_filters[3],
                         ori_shape_filters[2],
                         ori_shape_filters[0],
                         ori_shape_filters[1])
    else:
        args_dict = {
            "errCode": "E60004",
            "param_name": "filter",
            "expected_format_list": "[NCHW,NHWC,HWCN]",
            "format": ori_format_filters
        }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
    return shape_filters


def exchange_filter_nc_axis(ori_format_filters, ori_shape_filters, groups):
    """
    Get filter shape of NCHW from original shape
    :param ori_format_filters:
    :param ori_shape_filters:
    :return: filter shape of exchange filter nc axis
    """
    if ori_format_filters == "NCHW":
        shape_filters = [ori_shape_filters[1],
                         ori_shape_filters[0],
                         ori_shape_filters[2],
                         ori_shape_filters[3]]
    elif ori_format_filters == "NHWC":
        shape_filters = [ori_shape_filters[3],
                         ori_shape_filters[1],
                         ori_shape_filters[2],
                         ori_shape_filters[0]]
    elif ori_format_filters == "HWCN":
        shape_filters = [ori_shape_filters[0],
                         ori_shape_filters[1],
                         ori_shape_filters[3],
                         ori_shape_filters[2]]
    else:
        args_dict = {
            "errCode": "E60004",
            "param_name": "filter",
            "expected_format_list": "[NCHW,NHWC,HWCN]",
            "format": ori_format_filters
        }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
    if shape_filters[1] % groups != 0:
        args_dict = {
            "errCode": "E60108",
            "reason": "batch of weight % groups must be 0",
        }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
    shape_filters[0], shape_filters[1] = ( shape_filters[0] * groups, shape_filters[1] // groups)
    return shape_filters


def align(x_1, x_2):
    """
    Get minimum y: y >= x_1 and y % x_2 == 0
    :param x_1:
    :param x_2:
    :return: minimum y: y >= x_1 and y % x_2 == 0
    """
    if x_2 == 0:
        args_dict = {
            "errCode": "E60114",
            "reason": "Division by zero",
            "value": "x_1 = {}, x_2 = {}".format(x_1, x_2)
        }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
    return (x_1 + x_2 - 1) // x_2 * x_2


def ceil(x_1, x_2):
    """
    Get (x_1 + x_2 - 1) // x_2
    :param x_1:
    :param x_2:
    :return: (x_1 + x_2 - 1) // x_2
    """
    if x_2 == 0:
        args_dict = {
            "errCode": "E60114",
            "reason": "Division by zero",
            "value": "x_1 = {}, x_2 = {}".format(x_1, x_2),
        }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
    return (x_1 + x_2 - 1) // x_2


def _lcm(param1, param2):
    """
    calculate least common multiple
    """
    temp = param1 * param2
    while param1 % param2 != 0:
        param1, param2 = param2, param1 % param2

    return temp // param2


def check_equal_rule(param1, param2, param1_name, param2_name):
    """
    param1 must equal to param2,
    otherwise raise RuntimeError
    """
    if param1 != param2:
        args_dict = {
                "errCode": "E60002",
                "attr_name": "shape",
                "param1_name": param1_name,
                "param1_value": param1,
                "param2_name": param2_name,
                "param2_value": param2
            }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))


def calculate_group(out_backprop, input_size, w_shape_nchw, groups, filter_dtype, filter_ori_format):
    """
    calculate groups Parameter
    """
    if out_backprop[1] % groups != 0:
        args_dict = {
            "errCode": "E60108",
            "reason": "channel of out_backprop % groups must be 0",
        }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
    if input_size[1] % groups != 0:
        args_dict = {
            "errCode": "E60108",
            "reason": "channel of y % groups must be 0",
        }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
    c0_size = tbe_platform.C0_SIZE
    c0_size_k = tbe_platform.CUBE_MKN[filter_dtype]['mac'][1]
    #groups in w's N and dx's C, so dx_c_ori is filter_c, dy_c_ori is filter_N/groups.
    check_equal_rule(out_backprop[1],
                     w_shape_nchw[0],
                     "channel of out_backprop",
                     "batch of filter")
    check_equal_rule(input_size[1],
                     w_shape_nchw[1] * groups,
                     "channel of y",
                     "channel of filter * groups")
    dx_c_ori = w_shape_nchw[1]
    dy_c_ori = w_shape_nchw[0] // groups
    filter_batch_ori = w_shape_nchw[0] // groups
    filter_c_ori = w_shape_nchw[1]

    dx_c_extend = _lcm(dx_c_ori, c0_size) // dx_c_ori
    dy_c_extend = _lcm(dy_c_ori, c0_size_k) // dy_c_ori
    multiple_extend = min(_lcm(dx_c_extend, dy_c_extend), groups)

    dx_c1_extend = ceil(multiple_extend * dx_c_ori, c0_size)
    dy_c1_extend = ceil(multiple_extend * dy_c_ori, c0_size_k)

    group_dict = {GroupDictKeys.g_extend: ceil(groups, multiple_extend),
                  GroupDictKeys.multiple_extend: multiple_extend,
                  GroupDictKeys.groups: groups,
                  GroupDictKeys.dx_c1_extend: dx_c1_extend,
                  GroupDictKeys.dy_c1_extend: dy_c1_extend,
                  GroupDictKeys.dx_c_ori: dx_c_ori,
                  GroupDictKeys.dy_c_ori: dy_c_ori,
                  GroupDictKeys.filter_batch_ori: filter_batch_ori,
                  GroupDictKeys.filter_c_ori: filter_c_ori,
                  GroupDictKeys.filter_ori_format: filter_ori_format
                  }
    return group_dict


def get_padlist(  # pylint: disable=too-many-locals
    pads,
    shape_res,
    strides,
    shape_filters,
    dilations):
    """
    Get pad list of int
    :param pads: "SAME" or "VALID" or list of int
    :param shape_res: shape of dx
    :param strides:
    :param shape_filters:
    :param dilations:
    :return: pad list of int
    """
    fmap_h, fmap_w = shape_res[2], shape_res[3]
    _, _, filter_h, filter_w = shape_filters
    stride_h, stride_w = strides
    _, _, dilation_h, dilation_w = dilations

    filter_h_dilation = (filter_h - 1) * dilation_h + 1
    filter_w_dilation = (filter_w - 1) * dilation_w + 1
    if pads == 'SAME':
        pad_h = \
            align(fmap_h, stride_h) - stride_h + filter_h_dilation - fmap_h
        pad_h = tvm.max(pad_h, 0)
        pad_up = pad_h // 2
        pad_down = pad_h - pad_up
        pad_w = \
            align(fmap_w, stride_w) - stride_w + filter_w_dilation - fmap_w
        pad_w = tvm.max(pad_w, 0)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pads = [pad_up, pad_down, pad_left, pad_right]
    elif pads == "VALID":
        pads = PADDING_VAILD
    pads = shape_util.shape_to_list(pads)
    return pads


def get_shape_out_backprop(ori_format_out_backprop, ori_shape_out_backprop):
    """
    Get out_backpro shape of NCHW from original shape
    :param ori_format_out_backprop:
    :param ori_shape_out_backprop:
    :return: out_backpro shape of NCHW
    """
    if ori_format_out_backprop == "NCHW":
        shape_out_backprop = ori_shape_out_backprop
    elif ori_format_out_backprop == "NHWC":
        shape_out_backprop = (ori_shape_out_backprop[0],
                              ori_shape_out_backprop[3],
                              ori_shape_out_backprop[1],
                              ori_shape_out_backprop[2])
    else:
        args_dict = {
            "errCode": "E60004",
            "param_name": "out_backprop",
            "expected_format_list": "[NCHW,NHWC]",
            "format": ori_format_out_backprop
        }
        raise RuntimeError(args_dict,
                           err_man.get_error_message(args_dict))
    return shape_out_backprop


def get_shape_dilation(ori_out_backprop_format, dilations):
    """
    Get result shape of NCHW from original shape
    :param ori_out_backprop_format:
    :param dilations:
    :return: result shape of NCHW
    """
    if ori_out_backprop_format == "NCHW":
        shape_dilations = dilations
    elif ori_out_backprop_format == "NHWC":
        shape_dilations = (dilations[0],
                           dilations[3],
                           dilations[1],
                           dilations[2])
    else:
        args_dict = {
            "errCode": "E60004",
            "param_name": "out_backprop",
            "expected_format_list": "[NCHW,NHWC]",
            "format": ori_out_backprop_format
        }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
    return shape_dilations


def get_shape_res(ori_format_res, ori_shape_res):
    """
    Get result shape of NCHW from original shape
    :param ori_format_res:
    :param ori_shape_res:
    :return: result shape of NCHW
    """
    if ori_format_res == "NCHW":
        shape_res = ori_shape_res
    elif ori_format_res == "NHWC":
        shape_res = (
            ori_shape_res[0], ori_shape_res[3],
            ori_shape_res[1], ori_shape_res[2])
    else:
        args_dict = {
            "errCode": "E60004",
            "param_name": "y",
            "expected_format_list": "[NCHW,NHWC]",
            "format": ori_format_res
        }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
    return shape_res


def check_attr_range(attr_name, attr_value, attr_min=None, attr_max=None):
    """
    check the parameter size
    """
    if attr_min is None and attr_max is None:
        return
    if attr_value < 0 or not isinstance(attr_value, int):
        return
    if attr_min is None:
        if attr_value > attr_max:
            args_dict = {
                "errCode": "E60114",
                "reason": "{} exceed max_value."
                          " max_value={}.".format(attr_name, attr_max),
                "value": "attr_value = {}".format(attr_value)
            }
            raise RuntimeError(args_dict,
                               err_man.get_error_message(args_dict))
    elif attr_max is None:
        if attr_value < attr_min:
            args_dict = {
                "errCode": "E60114",
                "reason": "{} less than min_value. "
                          "min_value={}.".format(attr_name, attr_min),
                "value": "attr_value = {}".format(attr_value)
            }
            raise RuntimeError(args_dict,
                               err_man.get_error_message(args_dict))
    elif attr_value < attr_min or attr_value > attr_max:
        args_dict = {
            "errCode": "E60011",
            "range": "[{},{}]".format(attr_min, attr_max),
            "attr_name": attr_name,
            "value": attr_value
        }
        raise RuntimeError(args_dict,
                           err_man.get_error_message(args_dict))


@para_check.check_input_type(  # pylint: disable=R0913, R0914, R0915
    (list, tuple), (list, tuple), (list, tuple),
    (list, tuple), (str, list, tuple), (list, tuple),
    str,
    str,
    str,
    str,
    (dict, para_check.NONE_TYPE),
    (dict, para_check.NONE_TYPE))
def check_conv2dbp_input_params(shape_filter, shape_out_backprop, input_sizes,
                                strides, pads, dilations,
                                filter_dtype,
                                out_backprop_dtype,
                                res_dtype,
                                kernel_name,
                                fusion_para=None,
                                group_dict=None
                                ):
    """
    The params check function of conv2d backprop input and deconvolution

    Parameters:
    ----------
    shape_filter : The shape of filter.
                   4-D with shape [batch, channels, height, weight].

    shape_out_backprop : The shape of gradients.
                         4-D with shape [batch, channels, height, weight].

    input_sizes : The shape of feature map.
                  4-D with shape [batch, channels, height, weight].

    strides : A list of ints. The stride of the sliding window.

    pads : "SAME"or"VALID" indicating the type of pads algorithm to use,
           or list.

    dilations : An optional list of ints. Default value is [1, 1, 1, 1].

    filter_dtype : The dtype of filter data. Default value is float16.

    out_backprop_dtype : The dtype of gradients data. Default value is float16.

    res_dtype : The dtype of result(De/Dx) data. Default value is float16.

    kernel_name : Cce kernel name. Default value is "conv2d_backprop_input_cce"

    fusion_para: the l1 fusion para

    group_dict : The paras of group_dict.

    Returns : All transformed params.
    ----------
    """

    def _check_64bits_limitation(attr_name, attr_value, dtype=None):
        if dtype is None:
            bit_ratio = BIT_RATIO_DICT.get("float16")
        else:
            bit_ratio = BIT_RATIO_DICT.get(dtype)
        if attr_value * bit_ratio > DATA_SIZE_MAX:
            args_dict = {
                "errCode": "E60020",
                "attr_name": attr_name
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))

    def _check_shape_relation():
        if group_dict.get(GroupDictKeys.dx_c_ori) != \
                group_dict.get(GroupDictKeys.filter_c_ori):
            args_dict = {
                "errCode": "E60002",
                "attr_name": "shape",
                "param1_name": "dx_c_ori",
                "param1_value": group_dict.get(GroupDictKeys.dx_c_ori),
                "param2_name": "filter_c_ori",
                "param2_value": group_dict.get(GroupDictKeys.filter_c_ori)
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
        if group_dict.get(GroupDictKeys.dy_c_ori) != \
                group_dict.get(GroupDictKeys.filter_batch_ori):
            args_dict = {
                "errCode": "E60002",
                "attr_name": "shape",
                "param1_name": "dy_c_ori",
                "param1_value": group_dict.get(GroupDictKeys.dy_c_ori),
                "param2_name": "filter_batch_ori",
                "param2_value": group_dict.get(GroupDictKeys.filter_batch_ori)
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
        if shape_out_backprop[N_DIM] != DYNAMIC_FLAG and fmap_batch != dedy_batch:
            args_dict = {
                "errCode": "E60002",
                "attr_name": "shape",
                "param1_name": "y's N",
                "param1_value": fmap_batch,
                "param2_name": "out_backprop'N",
                "param2_value": dedy_batch
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
        if DYNAMIC_FLAG not in shape_out_backprop and filter_h_dilation > fmap_h_padding:
            args_dict = {
                "errCode": "E60014",
                "h_of_x": fmap_h_padding,
                "h_of_filter": filter_h_dilation
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
        if DYNAMIC_FLAG not in shape_out_backprop and filter_w_dilation > fmap_w_padding:
            args_dict = {
                "errCode": "E60014",
                "h_of_x": fmap_w_padding,
                "h_of_filter": filter_w_dilation
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))

    def _check_l1_size_limit():
        def _l1fusion_size_limit(l1_size):
            l1fusion_l1_size = 0
            if list(pads)[::2] != [0, 0] or [filter_h, filter_w] != [1, 1]:
                if stride_h > 1 or stride_w > 1:
                    l1fusion_l1_size = l1_size
            return l1fusion_l1_size

        if shape_out_backprop[W_DIM] == DYNAMIC_FLAG:
            return
        c0_size = tbe_platform.C0_SIZE
        c0_size_k = tbe_platform.CUBE_MKN[filter_dtype]['mac'][1]
        w_value = dedy_w * stride_w
        h_value_max = 1
        if fmap_w % c0_size != 0:
            h_value_max += 1

        a_l1_size = h_value_max * w_value * c0_size_k * BIT_RATIO_DICT.get(out_backprop_dtype)
        if _is_conv1d_situation():
            load3d_stride = 1
            a_l1_m_length = (c0_size - 1) * load3d_stride + filter_w_dilation
            a_l1_size = a_l1_m_length *\
                        c0_size_k * BIT_RATIO_DICT.get(out_backprop_dtype)
        b_l1_size = filter_w * c0_size * c0_size_k * BIT_RATIO_DICT.get(filter_dtype)
        if fusion_para.get("l1_fusion_type") != -1:
            a_l1_size = _l1fusion_size_limit(a_l1_size)
        l1_size = tbe_platform.get_soc_spec("L1_SIZE")
        if (a_l1_size + b_l1_size) > l1_size:
            args_dict = {
                "errCode": "E60022",
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))

    def _check_pads():
        if isinstance(pads, (tuple, list)) \
                and len(pads) != CONV_BACKPROP_SHAPE_DIM:
            args_dict = {
                "errCode": "E60107",
                "param_name": "pads"
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))

        if isinstance(pads, str) and pads not in PADDING_SUPPORT:
            args_dict = {
                "errCode": "E60021",
                "expected_pad_mode": PADDING_SUPPORT,
                "actual_pad_mode": pads
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))

    def _is_conv1d_situation():
        if fmap_h_padding == 1 and filter_h_dilation == 1 and stride_h == 1:
            return True
        return False

    def _is_load3d_special():
        # limitation by chip:
        # load3d instruction not support out_w = 1
        # only Ascend310/Hi3796CS/SD3403 can support
        # in dx, out is fmap
        if tbe_platform.get_soc_spec("SHORT_SOC_VERSION") not in ["Ascend310", "Hi3796CV300CS", "SD3403"]  \
            and fmap_h != 1 \
            and fmap_w == 1:
            return True
        return False

    def _fusion_para_check(fusion_para):
        l1_fusion_type = fusion_para.get("l1_fusion_type")
        input_memory_type = fusion_para.get("input_memory_type")
        output_memory_type = fusion_para.get("output_memory_type")

        if l1_fusion_type == -1:
            if input_memory_type == 1:
                args_dict = {
                    "errCode": "E60109",
                    "input_memory_type": str(input_memory_type)
                }
                raise RuntimeError(args_dict,
                                   err_man.get_error_message(args_dict))
            if output_memory_type == 1:
                args_dict = {
                    "errCode": "E60110",
                    "output_memory_type": str(output_memory_type)
                }
                raise RuntimeError(args_dict,
                                   err_man.get_error_message(args_dict))

    def _modify_fusion_para(fusion_para):
        if fusion_para is None:
            fusion_para = {"input_memory_type": 0,
                           "output_memory_type": 0,
                           "l1_fusion_type": -1,
                           "fmap_l1_addr_flag": False,
                           "fmap_l1_valid_size": 0}
        return fusion_para

    def _check_shape_rule():
        if DYNAMIC_FLAG not in shape_out_backprop:
            para_check.check_shape_rule(shape_out_backprop,
                                        CONV_BACKPROP_SHAPE_DIM, CONV_BACKPROP_SHAPE_DIM,
                                        DEFAULT_MAX_SHAPE_NUM)
            para_check.check_shape_rule(input_sizes,
                                        CONV_BACKPROP_SHAPE_DIM, CONV_BACKPROP_SHAPE_DIM,
                                        DEFAULT_MAX_SHAPE_NUM)
        para_check.check_shape_rule(dilations, CONV_BACKPROP_SHAPE_DIM,
                                    CONV_BACKPROP_SHAPE_DIM, DEFAULT_MAX_SHAPE_NUM)
        para_check.check_shape_rule(strides, STRIDES_SHAPE_DIM, STRIDES_SHAPE_DIM,
                                    DEFAULT_MAX_SHAPE_NUM)

    def _change_hw_limitation(dedy_hw_min, fmap_hw_min,
                              dedy_w_max, fmap_w_max):
        if _is_load3d_special():
            dedy_hw_min = 2
            fmap_hw_min = 2

        # if conv1d situation, make sure w is in [1,CONV1D_W_MAX]
        if _is_conv1d_situation():
            dedy_w_max = CONV1D_W_MAX
            fmap_w_max = CONV1D_W_MAX
        return dedy_hw_min, fmap_hw_min, dedy_w_max, fmap_w_max

    def _need_change_hw():
        return fmap_w == 1 and filter_w == 1 and dedy_w == 1 and pad_left == 0 and pad_right == 0

    # First : Base check, Mainly required by interface appearance
    # util check
    para_check.check_kernel_name(kernel_name)
    para_check.check_shape_rule(shape_filter, CONV_BACKPROP_SHAPE_DIM,
                                CONV_BACKPROP_SHAPE_DIM, DEFAULT_MAX_SHAPE_NUM)

    _check_shape_rule()
    _check_pads()

    # dilations check
    dilation_n, dilation_c, dilation_h, dilation_w = dilations
    if dilation_n != 1 or dilation_c != 1:
        args_dict = {
            "errCode": "E60023",
            "dilation_n": dilation_n,
            "dilation_c": dilation_c
        }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))

    # dtype check
    valid_filter_dtype = ("float16", "int8")
    valid_dedy_dtype = ("float16", "int8")
    valid_res_dtype = ("float16", "float32", "int32")

    filter_dtype = filter_dtype.lower()
    out_backprop_dtype = out_backprop_dtype.lower()
    res_dtype = res_dtype.lower()
    para_check.check_dtype_rule(filter_dtype, valid_filter_dtype)
    para_check.check_dtype_rule(out_backprop_dtype, valid_dedy_dtype)
    para_check.check_dtype_rule(res_dtype, valid_res_dtype)

    # Second : Furture Check, Mainly required by SRS
    # the relation limits between shape
    shape_filter = list(shape_filter)
    shape_out_backprop = list(shape_out_backprop)
    input_sizes = list(input_sizes)
    strides = list(strides)
    fmap_batch, fmap_channel, fmap_h, fmap_w = input_sizes
    dedy_batch, dedy_channel, dedy_h, dedy_w = shape_out_backprop
    filter_batch, filter_channel, filter_h, filter_w = shape_filter
    stride_h, stride_w = strides

    filter_h_dilation = (filter_h - 1) * dilation_h + 1
    filter_w_dilation = (filter_w - 1) * dilation_w + 1

    pads = get_padlist(pads, input_sizes, strides, shape_filter, dilations)
    pad_up, pad_down, pad_left, pad_right = pads

    fmap_h_padding = fmap_h + pad_up + pad_down
    fmap_w_padding = fmap_w + pad_left + pad_right

    # special cases
    dedy_hw_min, fmap_hw_min = DEDY_HW_MIN, FMAP_HW_MIN
    dedy_h_max, fmap_h_max = DEDY_H_MAX, FMAP_H_MAX
    dedy_w_max, fmap_w_max = DEDY_W_MAX, FMAP_W_MAX

    fusion_para = _modify_fusion_para(fusion_para)
    shape_out_backprop = (dedy_batch, dedy_channel, dedy_h, dedy_w)

    # exchange h and w will not change date in memmory
    if _need_change_hw():
        input_sizes = (fmap_batch, fmap_channel, fmap_w, fmap_h)
        shape_out_backprop = (dedy_batch, dedy_channel, dedy_w, dedy_h)
        shape_filter = (filter_batch, filter_channel, filter_w, filter_h)
        strides = stride_w, stride_h
        stride_h, stride_w = stride_w, stride_h
        dilations = dilation_n, dilation_c, dilation_w, dilation_h
        fmap_h_padding, fmap_w_padding = fmap_w_padding, fmap_h_padding
        dedy_h, dedy_w = dedy_w, dedy_h
        fmap_h, fmap_w = fmap_w, fmap_h
        filter_h, filter_w = filter_w, filter_h
        filter_h_dilation, filter_w_dilation = filter_w_dilation,\
                                               filter_h_dilation
        pad_left, pad_right, pad_up, pad_down = pads
        pads = pad_up, pad_down, pad_left, pad_right

    dedy_hw_min, fmap_hw_min, dedy_w_max, fmap_w_max = \
        _change_hw_limitation(dedy_hw_min, fmap_hw_min, dedy_w_max, fmap_w_max)

    _check_shape_relation()

    # Dedy value limit
    check_attr_range("out_backprop's H after expands", dedy_h * stride_h,
                     dedy_hw_min, dedy_h_max)
    if filter_h == 1 and filter_w == 1:
        check_attr_range("out_backprop's W after expands",
                         dedy_w * stride_w * stride_h,
                         dedy_hw_min, dedy_w_max)
    else:
        check_attr_range("out_backprop's W after expands", dedy_w * stride_w,
                         dedy_hw_min, dedy_w_max)

    # filter value limit
    check_attr_range("filter's H", filter_h, FILTER_HW_MIN)
    check_attr_range("filter's W", filter_w, FILTER_HW_MIN)

    # Fmap value limit
    check_attr_range("y's H", fmap_h, fmap_hw_min, fmap_h_max)
    check_attr_range("y's W", fmap_w, fmap_hw_min, fmap_w_max)

    # stride value limit
    check_attr_range("stride's H", stride_h, STRIDE_HW_MIN)
    check_attr_range("stride's W", stride_w, STRIDE_HW_MIN)

    # dilation value limit
    check_attr_range("dilations's H", dilation_h, DILATION_MIN, DILATION_MAX)
    check_attr_range("dilations's W", dilation_w, DILATION_MIN, DILATION_MAX)

    # Third : value check, Mainly required by the convolution rule
    if DYNAMIC_FLAG not in shape_out_backprop and ((fmap_h - filter_h_dilation + pad_up + pad_down) // stride_h + 1) != dedy_h:
        err_man_cube.raise_err_specific_user("conv2d_backprop_input", "fmap_h not match dedy_h")
    if DYNAMIC_FLAG not in shape_out_backprop and ((fmap_w - filter_w_dilation + pad_left + pad_right) // stride_w + 1) != dedy_w:
        err_man_cube.raise_err_specific_user("conv2d_backprop_input", "fmap_w not match dedy_w")

    # Forth : L1 limitation, Mainly required by chip
    _check_l1_size_limit()

    # Fifth : check shape size, 64 bits limitation
    c0_size = tbe_platform.C0_SIZE
    if -1 not in shape_out_backprop:
        fmap_size = fmap_batch * align(fmap_channel, c0_size) * fmap_h * fmap_w
        dedy_size = dedy_batch * align(dedy_channel, c0_size) * dedy_h * dedy_w
        _check_64bits_limitation("fmap_size", fmap_size, dtype=res_dtype)
        _check_64bits_limitation("dedy_size", dedy_size, dtype=out_backprop_dtype)
    filter_size = align(filter_batch, c0_size) * align(filter_channel, c0_size) * filter_h * filter_w
    _check_64bits_limitation("filter_size", filter_size, dtype=filter_dtype)

    result = (
        shape_filter, shape_out_backprop, input_sizes, strides, pads,
        dilations, filter_dtype, out_backprop_dtype, res_dtype, kernel_name)

    _fusion_para_check(fusion_para)

    return result
