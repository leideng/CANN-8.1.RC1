#!/usr/bin/env python
# -*- coding:utf-8 -*-
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

from tbe.common.utils.errormgr import error_manager_cube
from tbe.tvm.tir import expr
from tbe import tvm
from tbe.common.platform import get_soc_spec
from tbe.common.utils.const import WEIGHT_SPARSE_4_2
from te.platform import cce_params
from te.utils.error_manager import error_manager_util as err_man
from te.utils.error_manager import error_manager_cube as err_man_cube
from te.lang.cce.te_compute.cube_util import shape_to_list
from impl.util import util_select_op_base
from impl.util.platform_adapter import error_manager
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_build

# filter H, filter W must be in [1, 255]
_FILTER_HW_MIN = 1
_FILTER_HW_MAX = 255

# the dim of shape in conv_backprop must be 4
CONV_BACKPROP_SHAPE_DIM = 4

# DeDyW must be in [1,4096]
DEDY_HW_MIN = 1
DEDY_W_MAX = 4096

# the bytes length of several dtype
BIT_RATIO_DICT = {"int32": 4, "float32": 4, "float16": 2,
                  "uint8": 1, "int8": 1, "uint4": 0.5, "int4": 0.5, "bfloat16": 2}
# same as (2**63-1)
DATA_SIZE_MAX = 9223372036854775807

# If dim is dynamic, the value of dim is -1
DYNAMIC_FLAG = -1

# The dim of N/H/W is 0/2/3, in the format NCHW
N_DIM = 0
H_DIM = 2
W_DIM = 3

# 2 means L1 enable
L1FUSION_INPUT_CTR = 2


OUT_BACKPROP_FORMAT_LIST = ["NCHW", "NHWC"]
FMAP_FORMAT_LIST = ["NCHW", "NHWC"]
WEIGHT_FORMAT_LIST = ["NCHW", "NHWC", "HWCN"]
DEPTHWISW_WEIGHT_FORMAT_LIST = ["HWCK", "HWCN", "NCHW"]

INTRINSIC_FIXPIPE_UNIT_LIST = "Intrinsic_fix_pipe_unit_list"


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


def check_data_format(data_format, support_format_list, param_name):
    """
    check data format
    :param data_format: str
    :param support_format_list: list str
    :param param_name: str
    """
    if data_format not in support_format_list:
        err_man_cube.raise_err_input_format_invalid(
            "Conv2dBackpropInput", param_name, support_format_list, data_format
        )


def check_value_min(val_name, val, val_min):
    if val <= val_min:
        args_dict = {
            "errCode": "E60114",
            "reason": "{} should be more than min_value. min_value={}.".format(val_name, val_min),
            "value": "{}".format(val)
        }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))


def check_shape_values(values_lst, tensor_name, is_shape_flag):
    lst_len = len(values_lst)
    for i in range(lst_len):
        if is_shape_flag:
            tmp_name = "the [{}]th axis of {}'s shape".format(i, tensor_name)
        else:
            tmp_name = "the [{}]th value of {}".format(i, tensor_name)
        check_value_min(tmp_name, values_lst[i], 0)


def update_ori_shape(input_tensor):
    # in split fusion, the ori_shape is before split while shape has been splited
    # use shape to update the ori_shape, shape format is NC1HWC0
    tensor_ori_shape = list(input_tensor.get("ori_shape"))
    tensor_shape = input_tensor.get("shape")
    tensor_ori_format = input_tensor.get("ori_format")
    n_dim_index = tensor_ori_format.find("N")
    h_dim_index = tensor_ori_format.find("H")
    w_dim_index = tensor_ori_format.find("W")
    tensor_ori_shape[n_dim_index] = tensor_shape[N_DIM]
    tensor_ori_shape[h_dim_index] = tensor_shape[H_DIM]
    tensor_ori_shape[w_dim_index] = tensor_shape[W_DIM]
    input_tensor["ori_shape"] = tensor_ori_shape


def get_nchw_shape(ori_format, ori_shape, support_format_list, param_name):
    """
    get nchw shape according input shape and format
    :param ori_format: str
    :param ori_shape: list or tuple
    :param support_format_list: list str
    :param param_name: str
    :return: shape of NCHW format
    """
    if isinstance(ori_format, expr.StringImm):
        ori_format = ori_format.value
    ori_shape = shape_to_list(ori_shape)
    check_data_format(ori_format, support_format_list, param_name)
    n_dim_index = ori_format.find("N")
    if n_dim_index < 0:
        n_dim_index = ori_format.find("K")
    c_dim_index = ori_format.find("C")
    h_dim_index = ori_format.find("H")
    w_dim_index = ori_format.find("W")
    shape_nchw = [ori_shape[n_dim_index], ori_shape[c_dim_index], ori_shape[h_dim_index], ori_shape[w_dim_index]]
    return shape_nchw


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
    c0_size = cce_params.C0_SIZE
    c0_size_k = tbe_platform.CUBE_MKN[filter_dtype]['mac'][1]
    if filter_dtype == "float32":
        c0_size_k, c0_size = c0_size, c0_size_k
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

    group_dict = {GroupDictKeys.g_extend: ceil(groups, multiple_extend),  # value of the attribute 'groups'
                  GroupDictKeys.multiple_extend: multiple_extend,
                  GroupDictKeys.groups: groups,
                  GroupDictKeys.dx_c1_extend: dx_c1_extend,
                  GroupDictKeys.dy_c1_extend: dy_c1_extend,
                  GroupDictKeys.dx_c_ori: dx_c_ori,  # in_channels / groups, where in_channels is the channel of dedx
                  GroupDictKeys.dy_c_ori: dy_c_ori,
                  GroupDictKeys.filter_batch_ori: filter_batch_ori,
                  GroupDictKeys.filter_c_ori: filter_c_ori,  # in_channels / groups
                  GroupDictKeys.filter_ori_format: filter_ori_format
                  }
    return group_dict


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


def need_exchange_hw_axis(shape_filter_nchw, shape_out_backprop_nchw, input_sizes_nchw, strides_hw, pads):
    """
    check for special case, whether need to exchange h and w
    """
    fmap_w = input_sizes_nchw[-1]
    _, _, dedy_h, dedy_w = shape_out_backprop_nchw
    _, _, filter_h, filter_w = shape_filter_nchw
    stride_h, stride_w = strides_hw
    _, _, pad_left, pad_right = pads

    need_change = False
    if fmap_w == 1 and filter_w == 1 and dedy_w == 1 and pad_left == 0 and pad_right == 0:
        if filter_h == 1:
            need_change = DEDY_HW_MIN <= dedy_h * stride_h * stride_w <= DEDY_W_MAX
        else:
            need_change = DEDY_HW_MIN <= dedy_h * stride_h <= DEDY_W_MAX
    return need_change


def swap_h_w_axes_in_shape(shape, src_format):
    if src_format not in ('FRACTAL_Z', 'NC1HWC0', 'NCHW', 'HWCN', 'NHWC'):
        error_manager_cube.raise_err_message_cube(f"swap h w axes with format({src_format}) is not supported.")

    if src_format == 'FRACTAL_Z':
        return shape

    if src_format == 'NC1HWC0':
        # NC1HWC0 -> NC1WHC0
        dst_shape = [x for x in shape]
        dst_shape[2], dst_shape[3] = dst_shape[3], dst_shape[2]
        return dst_shape

    # 'NCHW', 'HWCN', 'NHWC'
    idx_h = src_format.find('H')
    idx_w = src_format.find('W')
    dst_shape = [None] * len(shape)
    for idx in range(len(src_format)):
        if idx == idx_h:
            dst_shape[idx] = shape[idx_w]
        elif idx == idx_w:
            dst_shape[idx] = shape[idx_h]
        else:
            dst_shape[idx] = shape[idx]
    return dst_shape


def _check_fusion_para(fusion_para):
    if fusion_para is None:
        fusion_para = {"input_memory_type": 0,
                       "output_memory_type": 0,
                       "l1_fusion_type": -1,
                       "fmap_l1_addr_flag": False,
                       "fmap_l1_valid_size": 0}
    l1_fusion_type = fusion_para.get("l1_fusion_type")
    input_memory_type = fusion_para.get("input_memory_type")
    output_memory_type = fusion_para.get("output_memory_type")

    if l1_fusion_type == -1:
        if input_memory_type == 1:
            args_dict = {
                "errCode": "E60109",
                "input_memory_type": str(input_memory_type)
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
        if output_memory_type == 1:
            args_dict = {
                "errCode": "E60110",
                "output_memory_type": str(output_memory_type)
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
    return fusion_para


def check_conv2dbp_input_params(tensor_content, option_input, attrs_dict,
        op_type="conv2d_transpose", fusion_para=None):
    """
    check the input of shape and attr
    """
    fusion_para = _check_fusion_para(fusion_para)
    # check the attr
    _check_attr_limited(attrs_dict)
    # check the shape
    _check_shape_limited(tensor_content, attrs_dict, op_type)
    # check l1 size
    _check_l1_size(tensor_content, attrs_dict, fusion_para)
    # check the data size is great than int64
    _check_size_limited(tensor_content)
    # check bias size and dtype
    _check_bias_limit(tensor_content, option_input)


def _check_bias_limit(tensor_content, option_input):
    """
    check bias size and dtype
    """
    bias = option_input.get("bias")
    if bias is None:
        return
    if hasattr(bias, "op"):
        return
    bias_size = bias.get("ori_shape")[0]
    y_c = tensor_content.get("ori_nchw_shape")["fmap_dedx"][1]
    if bias_size != y_c:
        dict_args = {
            'errCode': 'E65007',
            'param1': 'bias_size',
            'param2': 'y_channel',
            'actual_value': '{}, {}'.format(bias_size, y_c)
        }
        raise RuntimeError(dict_args,
                            err_man.get_error_message(dict_args))
    bias_dtype = bias.get("dtype").lower()
    y_dtype = tensor_content.get("dtype")["fmap_dedx"]
    if bias_dtype == "float16" and y_dtype == "float32":
        dict_args = {
            'errCode': 'E65007',
            'reason': 'when y_dtype=float16, int32 or float32, bias_dtype should be equal to y_dtype',
            'param1': 'bias_dtype',
            'param2': 'y_dtype',
            'actual_value': '{}, {}'.format(bias_dtype, y_dtype)
        }
        raise RuntimeError(dict_args,
                            err_man.get_error_message(dict_args))


def get_tensor_content(tensor_dict, attrs_dict, op_type="conv2d_transpose", mode="op_calculate"):
    """
    get nchw shape of input, output
    """
    ori_nchw_shape_dict, dtype_dict, ori_format_dict = {}, {}, {}
    input_size = attrs_dict.get("input_size")
    groups = attrs_dict.get("groups", 1)
    # get nchw shape of input and output
    for tensor_name, tensor in tensor_dict.items():
        if hasattr(tensor, "op"):
            tensor_ori_shape = [i.value for i in tensor.op.attrs["ori_shape"]]
            tensor_shape = shape_to_list(tensor.shape)
            tensor_ori_format = tensor.op.attrs["ori_format"]
            tensor_dtype = tensor.dtype
        else:
            tensor_ori_shape = tensor.get("ori_shape")
            tensor_shape = tensor.get("shape")
            tensor_ori_format = tensor.get("ori_format")
            tensor_dtype = tensor.get("dtype")
        # the ori_shape length must be 4
        check_shape_values(tensor_ori_shape, tensor_name, True)
        para_check.check_shape_rule(tensor_ori_shape,
                                    CONV_BACKPROP_SHAPE_DIM,
                                    CONV_BACKPROP_SHAPE_DIM)
        # the output shape of y must equal with attr input_size
        if input_size is not None and tensor_name == "fmap_dedx":
            if not (all(i == 0 for i in input_size)) and list(input_size) != list(tensor_ori_shape):
                error_manager_cube.raise_err_param_should_be_equal(
                    "input_size", "ori_shape of output", input_size, tensor_ori_shape
                )
        if "fmap" in tensor_name:
            support_format_list = FMAP_FORMAT_LIST
        else:
            if op_type == "depthwise_conv2d_backprop_input":
                support_format_list = DEPTHWISW_WEIGHT_FORMAT_LIST
            else:
                support_format_list = WEIGHT_FORMAT_LIST
        tensor_ori_nchw_shape = get_nchw_shape(tensor_ori_format, tensor_ori_shape, support_format_list, tensor_name)
        if tensor_dtype in ("int4", "int8") and tensor_name == "weight" and mode != "check_supported":
            if tensor_ori_nchw_shape[0] % groups != 0:
                args_dict = {
                "errCode": "E60108",
                "reason": "batch of weight % groups must be 0",
                }
                raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
            tensor_ori_nchw_shape = [
                tensor_ori_nchw_shape[1] * groups,
                tensor_ori_nchw_shape[0] // groups,
                tensor_ori_nchw_shape[2],
                tensor_ori_nchw_shape[3]
            ]
        if tensor_name == "fmap_dedy" and mode != "check_supported":
            # in split fusion, the ori_shape is before split while shape has been splited
            tensor_ori_nchw_shape[N_DIM] = tensor_shape[N_DIM]
            tensor_ori_nchw_shape[H_DIM] = tensor_shape[H_DIM]
            tensor_ori_nchw_shape[W_DIM] = tensor_shape[W_DIM]
        ori_nchw_shape_dict[tensor_name] = list(tensor_ori_nchw_shape)
        dtype_dict[tensor_name] = tensor_dtype
        # the type of tensor_ori_format could be tvm.Runtime.String
        ori_format_dict[tensor_name] = str(tensor_ori_format)

    return {"ori_nchw_shape": ori_nchw_shape_dict, "dtype": dtype_dict, "ori_format": ori_format_dict}


def cal_attrs(tensor_content, attrs_dict, mode="op_calculate"):
    """
    cal the hw strides, nchw dilation, groups
    """
    strides = attrs_dict.get("strides")
    data_format = attrs_dict.get("data_format")
    dilations = attrs_dict.get("dilations")
    para_check.check_kernel_name(attrs_dict.get("kernel_name"))
    # get stride_h and stride_w
    if len(strides) == CONV_BACKPROP_SHAPE_DIM:
        if strides[data_format.find("N")] != 1 or strides[data_format.find("C")] != 1:
            args_dict = {
                "errCode": "E60114",
                "reason": "Stride N, C must be 1",
                "value": "actual N={}, C={}".format(strides[data_format.find("N")],
                                                    strides[data_format.find("C")])
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
        attrs_dict["strides"] = [strides[data_format.find("H")], strides[data_format.find("W")]]
    # get nchw dilation
    check_shape_values(dilations, "dilations", False)
    para_check.check_shape_rule(dilations, CONV_BACKPROP_SHAPE_DIM, CONV_BACKPROP_SHAPE_DIM)
    attrs_dict["dilations"] = get_nchw_shape(data_format, dilations, FMAP_FORMAT_LIST, "dialtion")
    # cal the group dict
    if mode == "op_calculate":
        c_in = tensor_content.get("ori_nchw_shape")["fmap_dedx"][1]
        attrs_dict["groups_dict"] = calculate_group(
            tensor_content.get("ori_nchw_shape")["fmap_dedy"],
            tensor_content.get("ori_nchw_shape")["fmap_dedx"],
            tensor_content.get("ori_nchw_shape")["weight"],
            attrs_dict.get("groups", c_in),
            tensor_content.get("dtype")["weight"],
            tensor_content.get("ori_format")["weight"]
        )


def exchange_hw_axis(tensor_content, attrs_dict, op_type):
    """
    exchange the h and axis when w is 1
    """
    if op_type == "depthwise_conv2d_backprop_input":
        return False
    need_change = need_exchange_hw_axis(
        tensor_content.get("ori_nchw_shape")["weight"],
        tensor_content.get("ori_nchw_shape")["fmap_dedy"],
        tensor_content.get("ori_nchw_shape")["fmap_dedx"],
        attrs_dict.get("strides"), attrs_dict.get("pads"))
    if need_change:
        attrs_dict["strides"] = [attrs_dict.get("strides")[x] for x in (1, 0)]
        attrs_dict["pads"] = [attrs_dict.get("pads")[x] for x in (2, 3, 0, 1)]
        attrs_dict["dilations"] = [attrs_dict.get("dilations")[x] for x in (0, 1, 3, 2)]
        if attrs_dict.get("output_padding") is not None:
            attrs_dict["output_padding"] = [attrs_dict.get("output_padding")[x] for x in (0, 1, 3, 2)]
        for tensor_name, nchw_shape in tensor_content.get("ori_nchw_shape").items():
            tensor_content["ori_nchw_shape"][tensor_name] = [nchw_shape[x] for x in (0, 1, 3, 2)]
    return need_change


def _get_value(obj, key, default=None):
    """
    get value from obj by key with default value
    obj supports type Tensor and dict
    """
    if isinstance(obj, tvm.Tensor):
        tensor_value = obj.op.attrs[key] if key in obj.op.attrs else default
        return tensor_value.value if hasattr(tensor_value, "value") else tensor_value
    return obj.get(key, default)


def _get_deconvolution_fusion_para(input_x, input_y=None):
    """
    get fusion para for L1 fusion in op deconvlution
    """
    input_memory_type = _get_value(input_x, "addr_type", 0)
    l1_fusion_type = _get_value(input_x, "L1_fusion_type", -1)
    fmap_l1_addr_flag = _get_value(input_x, "L1_addr_flag", False)
    fmap_l1_valid_size = _get_value(input_x, "L1_valid_size", 0)
    output_memory_type = (
        _get_value(input_y, "addr_type", 0) if input_y is not None else "fuse_flag"
    )
    l1_fusion_enable_flag = tbe_build.get_L1_info("L1_fusion_enabled")
    # 0, 1, 2 mean memory DDR, L1, L2
    if input_memory_type not in (0, 1, 2):
        args_dict = {
            "errCode": "E65008",
            "input_memory_type_range": "(MEMORY_DDR, MEMORY_L1, MEMORY_L2)",
            "input_memory_type": str(input_memory_type),
        }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
    # 0, 1, 2 mean memory DDR, L1, L2
    if input_y is not None and output_memory_type not in (0, 1, 2):
        args_dict = {
            "errCode": "E65009",
            "output_memory_type_range": "(MEMORY_DDR, MEMORY_L1, MEMORY_L2)",
            "output_memory_type": str(output_memory_type),
        }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))

    if not l1_fusion_enable_flag:
        input_memory_type = 0
        if input_y is not None:
            output_memory_type = 0
        l1_fusion_type = -1
        fmap_l1_addr_flag = False
        fmap_l1_valid_size = 0
    fusion_para = {
        "input_memory_type": input_memory_type,
        "output_memory_type": output_memory_type,
        "l1_fusion_type": l1_fusion_type,
        "fmap_l1_addr_flag": fmap_l1_addr_flag,
        "fmap_l1_valid_size": fmap_l1_valid_size,
    }
    return fusion_para


def conv2d_transpose_static_impl(tensor_dict, option_input, attrs_dict, op_type="conv2d_transpose"):
    """
    the impl for conv2d_transpose_d, conv2d_backprop_input_d, deconv, depth_conv2d_backprop_input_d
    """
    # get fusion_para
    if op_type == "deconvolution":
        fusion_para = _get_deconvolution_fusion_para(tensor_dict.get("fmap_dedy"), tensor_dict.get("fmap_dedx"))
    else:
        fusion_para = None
    # get the shape, dtype and format
    tensor_content = get_tensor_content(tensor_dict, attrs_dict, op_type)
    cal_attrs(tensor_content, attrs_dict)
    # check of the input, output and attrs
    if op_type != "depthwise_conv2d_backprop_input":
        check_conv2dbp_input_params(tensor_content, option_input, attrs_dict, op_type, fusion_para=fusion_para)
    else:
        # in depth wise, change the fmap dedx shape and filter shape
        c_in = tensor_content.get("ori_nchw_shape")["fmap_dedx"][1]
        cin_0 = tbe_platform.CUBE_MKN.get(tensor_content.get("dtype")["weight"]).get("mac")[1]
        tensor_content.get("ori_nchw_shape")["fmap_dedx"][1] = align(c_in, cin_0)
        tensor_content.get("ori_nchw_shape")["weight"][0] = 1
        tensor_content.get("ori_nchw_shape")["weight"][1] = align(c_in, cin_0)

    # get placeholder of input
    # change the h and w axis when fmap_w and weight_w both is 1
    fmap_dedy = tensor_dict.get("fmap_dedy")
    weight = tensor_dict.get("weight")
    shape_5hd_dedy = fmap_dedy.get("shape")
    if exchange_hw_axis(tensor_content, attrs_dict, op_type):
        shape_5hd_dedy = [shape_5hd_dedy[i] for i in (0, 1, 3, 2, 4)]
    tensor_dedy = tvm.placeholder(shape_5hd_dedy, name="dedy", dtype=tensor_content.get("dtype")["fmap_dedy"])
    tensor_weight = tvm.placeholder(weight.get("shape"), name="filter", dtype=tensor_content.get("dtype")["weight"])
    if option_input.get("bias") is not None:
        desc_bias = option_input["bias"]
        dtype_bias = desc_bias.dtype if hasattr(desc_bias, 'op') else desc_bias.get("dtype")
        # cin is align to 16
        c_in = align(tensor_content.get("ori_nchw_shape")["fmap_dedx"][1], 16)
        tensor_bias = tvm.placeholder((c_in,), name="tensor_bias", dtype=dtype_bias)
    else:
        tensor_bias = None

    if tensor_dict.get("compress_index") is not None:
        desc_compress_index = tensor_dict.get("compress_index")
        tensor_compress_index = tvm.placeholder(desc_compress_index.get("shape"), name="compress_index",
                                                dtype=desc_compress_index.get("dtype"))
    else:
        tensor_compress_index = None

    para_dict = {
        "strides": attrs_dict.get("strides"),
        "padding": attrs_dict.get("pads"),
        "dilations": attrs_dict.get("dilations"),
        "res_dtype": tensor_content.get("dtype")["fmap_dedx"],
        "tensor_bias": tensor_bias,
        "offset_x": attrs_dict.get("offset_x", 0),
        "kernel_name": attrs_dict.get("kernel_name"),
        "group_dict": attrs_dict.get("groups_dict"),
        "fusion_para": fusion_para,
        "output_padding": attrs_dict.get("output_padding", (0, 0, 0, 0)),
        "alg": attrs_dict.get("alg"),
        "compress_index": tensor_compress_index,
        "bias_ori_shape": tensor_content.get("ori_nchw_shape")["fmap_dedx"][1]
    }

    tensor_dedx = tbe.conv2d_backprop_input(filters=tensor_weight,
                                            out_backprop=tensor_dedy,
                                            filter_sizes=tensor_content.get("ori_nchw_shape")["weight"],
                                            input_sizes=tensor_content.get("ori_nchw_shape")["fmap_dedx"],
                                            para_dict=para_dict)
    # schedule
    if op_type in ("deconvolution", "conv2d_transpose"):
        tensor_list = [tensor_dedy, tensor_weight]
    else:
        tensor_list = [tensor_weight, tensor_dedy]
    if tensor_compress_index is not None:
        tensor_list.append(tensor_compress_index)
    if tensor_bias is not None:
        tensor_list.append(tensor_bias)
    tensor_list.append(tensor_dedx)
    return tensor_dedx, tensor_list


def conv2d_transpose_static_compute(tensor_dict, option_input, attrs_dict, op_type="conv2d_transpose"):
    """
    the fusion compute of conv2d_transpose_d, conv2d_backprop_input_d, deconv, depth_conv2d_backprop_input_d
    """
    # get fusion_para
    if op_type == "deconvolution":
        fusion_para = _get_deconvolution_fusion_para(tensor_dict.get("fmap_dedy"))
    else:
        fusion_para = None
    # get the shape, dtype and format
    tensor_content = get_tensor_content(tensor_dict, attrs_dict, op_type)
    cal_attrs(tensor_content, attrs_dict)
    # check of the input, output and attrs
    if op_type != "depthwise_conv2d_backprop_input":
        check_conv2dbp_input_params(tensor_content, option_input, attrs_dict, op_type, fusion_para=fusion_para)
    else:
        # in depth wise, change the fmap dedx shape
        c_in = tensor_content.get("ori_nchw_shape")["fmap_dedx"][1]
        cin_0 = tbe_platform.CUBE_MKN.get(tensor_content.get("dtype")["weight"]).get("mac")[1]
        tensor_content.get("ori_nchw_shape")["fmap_dedx"][1] = align(c_in, cin_0)

    para_dict = {
        "strides": attrs_dict.get("strides"),
        "padding": attrs_dict.get("pads"),
        "dilations": attrs_dict.get("dilations"),
        "res_dtype": tensor_content.get("dtype")["fmap_dedx"],
        "tensor_bias": option_input.get("bias"),
        "offset_x": attrs_dict.get("offset_x", 0),
        "kernel_name": attrs_dict.get("kernel_name"),
        "group_dict": attrs_dict.get("groups_dict"),
        "fusion_para": fusion_para,
        "output_padding": attrs_dict.get("output_padding", (0, 0, 0, 0)),
        "alg": attrs_dict.get("alg"),
        "compress_index": tensor_dict.get("compress_index"),
        "bias_ori_shape": tensor_content.get("ori_nchw_shape")["fmap_dedx"][1]
    }

    res = tbe.conv2d_backprop_input(
        tensor_dict.get("weight"),  tensor_dict.get("fmap_dedy"),
        tensor_content.get("ori_nchw_shape")["weight"],
        tensor_content.get("ori_nchw_shape")["fmap_dedx"],
        para_dict=para_dict)

    return res


def _check_shape_limited(tensor_content, attrs_dict, op_type):
    """
    check the shape of input and output
    """
    weight_nchw_shape = tensor_content.get("ori_nchw_shape")["weight"]
    fmap_dedy_nchw_shape = tensor_content.get("ori_nchw_shape")["fmap_dedy"]
    fmap_dedx_nchw_shape = tensor_content.get("ori_nchw_shape")["fmap_dedx"]
    output_padding = attrs_dict.get("output_padding", (0, 0, 0, 0))
    strides = attrs_dict.get("strides")
    dilations = attrs_dict.get("dilations")
    pads = shape_to_list(attrs_dict.get("pads"))
    # the filter shape must great than 0
    check_attr_range("the h of filter", weight_nchw_shape[H_DIM], _FILTER_HW_MIN)
    check_attr_range("the w of filter", weight_nchw_shape[W_DIM], _FILTER_HW_MIN)

    weight_h_dilation = (weight_nchw_shape[H_DIM] - 1) * dilations[H_DIM] + output_padding[H_DIM] + 1
    weight_w_dilation = (weight_nchw_shape[W_DIM] - 1) * dilations[W_DIM] + output_padding[W_DIM] + 1
    # pads means pad_up, pad_down, pad_left, pad_right
    fmap_dedx_h = fmap_dedx_nchw_shape[H_DIM] + pads[0] + pads[1]
    fmap_dedx_w = fmap_dedx_nchw_shape[W_DIM] + pads[2] + pads[3]
    if -1 not in fmap_dedx_nchw_shape:
        if weight_h_dilation > fmap_dedx_h:
            err_man_cube.raise_err_specific_user(op_type,
                "dedx_h+pad_up+pad_down=[{}] should be no less than kernel_h[{}]".\
                format(fmap_dedx_h, weight_h_dilation))
        if weight_w_dilation > fmap_dedx_w:
            err_man_cube.raise_err_specific_user(op_type,
                "dedx_w+pad_left+pad_right=[{}] should be no less than kernel_w[{}]".\
                format(fmap_dedx_w, weight_w_dilation))
        if ((fmap_dedx_h - weight_h_dilation) // strides[0] + 1) != fmap_dedy_nchw_shape[H_DIM]:
            err_man_cube.raise_err_specific_user(op_type, "fmap_dedx_h not match fmap_dedy_h")
        if ((fmap_dedx_w - weight_w_dilation) // strides[1] + 1) != fmap_dedy_nchw_shape[W_DIM]:
            err_man_cube.raise_err_specific_user(op_type, "fmap_dedx_w not match fmap_dedy_w")

    # the fmap dedx dedy HW must great than 0
    check_attr_range("the h of fmap_dedx", fmap_dedx_nchw_shape[H_DIM], 1)
    check_attr_range("the w of fmap_dedx", fmap_dedx_nchw_shape[W_DIM], 1)
    check_attr_range("the h of fmap_dedy", fmap_dedy_nchw_shape[H_DIM], 1)
    check_attr_range("the w of fmap_dedy", fmap_dedy_nchw_shape[W_DIM], 1)
    if fmap_dedy_nchw_shape[0] != -1 and fmap_dedx_nchw_shape[0] != -1:
        check_equal_rule(fmap_dedy_nchw_shape[0], fmap_dedx_nchw_shape[0], "batch of fmap_dedy", "batch of fmap_dedx")


def _check_attr_limited(attrs_dict):
    """
    check the attrs of op
    """
    # the stride hw must great than 0
    strides = attrs_dict.get("strides")
    dilations = attrs_dict.get("dilations")
    pads = attrs_dict.get("pads")
    groups = attrs_dict.get("groups")
    output_padding = attrs_dict.get("output_padding", (0, 0, 0, 0))
    check_attr_range("the h of stride", strides[0], 1)
    check_attr_range("the w of stride", strides[1], 1)
    # the dilation hw must great than 0, dilation nc must be 1
    check_attr_range("the h of dilations", dilations[H_DIM], 1)
    check_attr_range("the w of dilations", dilations[W_DIM], 1)
    if dilations[0] != 1 or dilations[1] != 1:
        error_manager_cube.raise_err_dilation_invalid(dilations[0], dilations[1])
    # pads lenth must be 4
    if not isinstance(pads, (tuple, list)) or len(pads) != CONV_BACKPROP_SHAPE_DIM:
        args_dict = {
            "errCode": "E60107",
            "param_name": "pads"
        }
        raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
    # group must great than 0
    if groups == 0:
        args_dict = {
            "errCode": "E60108",
            "reason": "groups can not be 0",
        }
        raise RuntimeError(args_dict, error_manager.get_error_message(args_dict))
    # output_padding_n and c must be 0
    if output_padding[0] != 0 or output_padding[1] != 0:
        error_manager_cube.raise_err_message_cube('The N and C axes of output_padding can only be 0.')


def _check_size_limited(tensor_content):
    """
    check the data size is great than int64
    """
    k0 = tbe_platform.CUBE_MKN[tensor_content.get("dtype")["weight"]]['mac'][2]
    weight_shape = tensor_content.get("ori_nchw_shape")["weight"]
    fmap_dedy_shape = tensor_content.get("ori_nchw_shape")["fmap_dedy"]
    fmap_dedx_shape = tensor_content.get("ori_nchw_shape")["fmap_dedx"]
    # reduce axis align to reduce co, n axis reduce to 16
    if -1 not in fmap_dedy_shape:
        weight_size = align(weight_shape[0], k0) * align(weight_shape[1], 16) * weight_shape[2] * weight_shape[3]
        fmap_dedy_size = fmap_dedy_shape[0] * align(fmap_dedy_shape[1], k0) * fmap_dedy_shape[2] * fmap_dedy_shape[3]
        fmap_dedx_size = fmap_dedx_shape[0] * align(fmap_dedx_shape[1], 16) * fmap_dedx_shape[2] * fmap_dedx_shape[3]
        if weight_size * BIT_RATIO_DICT.get(tensor_content.get("dtype")["weight"], 2) > DATA_SIZE_MAX:
            args_dict = {
                "errCode": "E60020",
                "attr_name": "filter_size"
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
        if fmap_dedy_size * BIT_RATIO_DICT.get(tensor_content.get("dtype")["fmap_dedy"], 2) > DATA_SIZE_MAX:
            args_dict = {
                "errCode": "E60020",
                "attr_name": "fmap_dedy_size"
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))
        if fmap_dedx_size * BIT_RATIO_DICT.get(tensor_content.get("dtype")["fmap_dedx"], 2) > DATA_SIZE_MAX:
            args_dict = {
                "errCode": "E60020",
                "attr_name": "fmap_dedx_size"
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))


def _check_l1_size(tensor_content, attrs_dict, fusion_para):
    """
    check the l1 size
    """
    fmap_dedy_shape_w = tensor_content.get("ori_nchw_shape")["fmap_dedy"][W_DIM]
    # only check the l1 size when w not dynamic mode
    if fmap_dedy_shape_w != DYNAMIC_FLAG:
        l1_fusion_flag = fusion_para.get("l1_fusion_type") != -1
        weight_dtype = tensor_content.get("dtype")["weight"]
        weight_ori_shape = tensor_content.get("ori_nchw_shape")["weight"]
        abl1_size = _cal_min_l1_size(weight_ori_shape, weight_dtype, attrs_dict, l1_fusion_flag)
        l1_size = get_soc_spec("L1_SIZE")
        if abl1_size > l1_size:
            args_dict = {
                "errCode": "E60022",
            }
            raise RuntimeError(args_dict, err_man.get_error_message(args_dict))


def _check_sparse_4_2(tensor_content, attrs_dict, op_type):
    if op_type != "conv2d_transpose_d_compress":
        return
    fmap_dedy_dtype = tensor_content.get("dtype").get("fmap_dedy")
    weight_dtype = tensor_content.get("dtype").get("weight")
    compress_index_dtype = tensor_content.get("dtype").get("compress_index")
    groups = attrs_dict.get("groups")
    alg = attrs_dict.get("alg")
    if alg != WEIGHT_SPARSE_4_2:
        error_manager_cube.raise_err_message_cube("only algorithm [{}] can be supported, current is {}".format(
                WEIGHT_SPARSE_4_2, alg))
    if not support_fixpipe():
        error_manager_cube.raise_err_message_cube("current platform not support {} alg".format(WEIGHT_SPARSE_4_2))
    if groups != 1:
            error_manager_cube.raise_err_message_cube("only support groups = 1")
    if fmap_dedy_dtype != "int8" or weight_dtype != "int8" or compress_index_dtype != "int8":
        error_manager_cube.raise_err_message_cube("input tensor dtype should be int8.")


def check_conv2d_transpose(tensor_dict, option_input, attrs_dict, op_type="conv2d_transpose"):
    """
    check the op of conv2d_transpose_d, conv2d_backprop_input_d, deconv, depth_conv2d_backprop_input_d
    """
    # check dtype of input
    if option_input.get("bias") is not None:
        bias_dtype = option_input.get("bias").get("dtype").lower()
        para_check.check_dtype_rule(bias_dtype, ("float16", "float32", "int32", "bfloat16"), "bias")
    # get the ori shape format in nchw mode
    tensor_content = get_tensor_content(tensor_dict, attrs_dict, op_type, "check_supported")
    cal_attrs(tensor_content, attrs_dict, "check_supported")
    if tensor_content.get("dtype")["fmap_dedy"] not in ("int8", "int4") and attrs_dict.get("offset_x", 0) != 0:
        error_manager_cube.raise_err_specific("Conv2dTransposeD", "when x_dtype is not int, offset_x must be 0")
    # check the attr
    _check_attr_limited(attrs_dict)
    # check the shape
    _check_shape_limited(tensor_content, attrs_dict, op_type)
    # check sparse_4_2
    _check_sparse_4_2(tensor_content, attrs_dict, op_type)


def _cal_min_l1_size(weight_ori_shape, input_dtype, attrs_dict, l1_fusion=False):
    """
    cal the l1 size of the op
    """
    weight_h, weight_w = weight_ori_shape[2:4]
    strides = attrs_dict.get("strides")
    dilations = attrs_dict.get("dilations")
    pads = attrs_dict.get("pads")
    output_padding = attrs_dict.get("output_padding", (0, 0, 0, 0))
    k0, n0 = tbe_platform.CUBE_MKN[input_dtype]['mac'][1:3]
    if input_dtype == "float32":
        bl1_size = 2 * k0 * n0 * BIT_RATIO_DICT.get(input_dtype)
    else:
        bl1_size = weight_w * k0 * n0 * BIT_RATIO_DICT.get(input_dtype)
        if not l1_fusion:
            bl1_size = k0 * n0 * BIT_RATIO_DICT.get(input_dtype)
    w_value = (n0 - 1) + (weight_w - 1) * dilations[W_DIM] + output_padding[W_DIM] + 1
    al1_size  = w_value * k0 * BIT_RATIO_DICT.get(input_dtype)
    if l1_fusion:
        if list(pads)[::2] == [0, 0] and [weight_h, weight_w] == [1, 1]:
            al1_size = 0
        else:
            if list(strides) == [1, 1]:
                al1_size = 0
    return al1_size + bl1_size


def get_op_support_info_conv2d_transpose(
        tensor_dict, option_input, attrs_dict, op_type="conv2d_transpose", mode="static"):
    """
    get op split info in conv2d_transpose_d, conv2d_backprop_input_d, deconv, depth_conv2d_backprop_input_d
    """
    fmap_dedy = tensor_dict.get("fmap_dedy")
    fmap_dedy_format = fmap_dedy.get("format")
    fmap_dedy_ori_shape = fmap_dedy.get("ori_shape")
    weight = tensor_dict.get("weight")
    cal_attrs(None, attrs_dict, "op_support")

    weight_ori_nchw_shape = get_nchw_shape(
        weight.get("ori_format"), weight.get("ori_shape"), WEIGHT_FORMAT_LIST, "weight")
    if list(fmap_dedy_ori_shape) != [-2]:
        fmap_dedy_ori_nchw_shape = get_nchw_shape(
            fmap_dedy.get("ori_format"), fmap_dedy_ori_shape, FMAP_FORMAT_LIST, "fmap_dedy")

    overlap_h = -1 if (weight_ori_nchw_shape[H_DIM] == 1 and attrs_dict.get("strides")[0] == 1) else 0
    overlap_w = -1 if (weight_ori_nchw_shape[W_DIM] == 1 and attrs_dict.get("strides")[1] == 1) else 0
    fmap_dedy_index, weight_index = (1, 0) if op_type == "conv2d_backprop_input" else (0, 1)
    bias_index = 2
    if mode != "static" and op_type in ("conv2d_backprop_input", "conv2d_transpose"):
        fmap_dedy_index += 1
        weight_index += 1
        bias_index += 1
    # input/output Serial， axis Serial, (headoverlap, tailoverlap, 0 means with overlap, -1 means without it)
    if fmap_dedy_format == "NC1HWC0":
        # cut N
        axis_split_matrix = [[
            util_select_op_base.SplitInput([fmap_dedy_index, [N_DIM], [-1], [-1]]),
            util_select_op_base.SplitOutput([0, [N_DIM]])
        ]]
        # cut H
        if mode == "static" or overlap_h == -1 or \
            (list(fmap_dedy_ori_shape) != [-2] and fmap_dedy_ori_nchw_shape[H_DIM] > 0):
            axis_split_matrix += [[
                util_select_op_base.SplitInput([fmap_dedy_index, [H_DIM], [overlap_h], [overlap_h]]),
                util_select_op_base.SplitOutput([0, [H_DIM]])
            ]]
        # cut W
        if mode == "static" or overlap_w == -1 or \
            (list(fmap_dedy_ori_shape) != [-2] and fmap_dedy_ori_nchw_shape[W_DIM] > 0):
            axis_split_matrix += [[
                util_select_op_base.SplitInput([fmap_dedy_index, [W_DIM], [overlap_w], [overlap_w]]),
                util_select_op_base.SplitOutput([0, [W_DIM]])
            ]]
        # cut Cin
        c_axis = 1 if fmap_dedy.get("dtype") in ("int4", "int8") else 0
        overlap_c = -1 if fmap_dedy.get("dtype") in ("int4", "int8") else 0
        if option_input.get("bias") is not None:
            axis_split_matrix += [[
                util_select_op_base.SplitInput([weight_index, [c_axis], [overlap_c], [overlap_c]],
                                               [bias_index, [0], [-1], [-1]]),
                util_select_op_base.SplitOutput([0, [1]])
            ]]
        else:
            axis_split_matrix += [[
                util_select_op_base.SplitInput([weight_index, [c_axis], [overlap_c], [overlap_c]]),
                util_select_op_base.SplitOutput([0, [1]])
            ]]
        axis_reduce_list = None
    else:
        axis_split_matrix = None
        axis_reduce_list = None
    if mode == "static":
        min_l1space = _cal_min_l1_size(weight_ori_nchw_shape, weight.get("dtype"), attrs_dict, True)
    else:
        min_l1space = 0
    op_cal_info_in_json = util_select_op_base.get_op_cal_info(
        axis_split_matrix, axis_reduce_list, L1FUSION_INPUT_CTR, min_l1space)

    return op_cal_info_in_json


def support_fixpipe():
    """
    check fixpipe support
    """
    return tbe_platform.intrinsic_check_support(INTRINSIC_FIXPIPE_UNIT_LIST)


def reshape_bias_shape(shape_bias):
    """
    tansform dx bias op shape
    :param shape_bias:
    """
    bias_length = 1
    for i in shape_bias:
        bias_length *= i
    return [(bias_length + 15) // 16 * 16]


def trans_shape_by_index(index, op_info, graph_info):
    if len(op_info.get_input_list()) <= index:
        return
    input_info = op_info.get_input_list()[index]
    peer_output_info = graph_info.get_data_output_by_id(input_info.get_edge_id())

    if peer_output_info is None:
        return
    input_shape = peer_output_info.get_shape()
    if len(input_shape) > 0:
        new_shape = reshape_bias_shape(input_shape)
        peer_output_info.set_shape(new_shape)