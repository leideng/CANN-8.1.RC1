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
dynamic depthwise_conv2d_backprop_filter
"""
from __future__ import absolute_import

import warnings

import impl.dynamic as dyn_impl
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_register
from impl.util.platform_adapter import tvm
from impl.util.util_common import ShapeConverter
from impl.util.util_cube_dynamic import CubeParaProcess
from impl.util.util_cube_dynamic import calc_max_fmap_w
from impl.util.util_cube_dynamic import check_dynamic_range_lower
from impl.util.util_cube_dynamic import check_graph_mode
from impl.util.util_cube_dynamic import check_graph_range
from impl.util.util_cube_dynamic import check_tensor_shape
from impl.util.util_cube_dynamic import correct_conv2d_backprop_range_start
from impl.util.util_cube_dynamic import correct_range
from impl.util.util_cube_dynamic import gen_conv_shape_range
from impl.util.util_cube_dynamic import get_idx_shape_from_format
from impl.util.util_cube_dynamic import is_empty_tensor_scene
from impl.util.util_cube_dynamic import pos_from_format
from tbe.common.utils.conv_util import ConvFeatureMap
from tbe.common.utils.conv_util import ConvKernel
from tbe.common.utils.conv_util import CubeChecker
from tbe.common.utils.conv_util import CubeConstantConfig
from tbe.common.utils.conv_util import trip_strides

BLOCK_SIZE = tbe_platform.BLOCK_REDUCE
DYNAMIC_FLAG = -1
UNKNOWN_RANK_SHAPE = [-2]
DIM_N_NCHW = 0
DIM_C_NCHW = 1
DIM_H_NCHW = 2
DIM_W_NCHW = 3
# shape's dim of input and output must be 4 or 5
FEATURE_MAP_DIM = [4, 5]

# shape's dim of filter must be 4
FILTER_DIM = 4

# shape's dim of strides/pads must be 4
STRIDES_DIM = 4
PADS_DIM = 4

# shape's dim of dilation must be 4
DILATION_DIM = 4

#the bytes length of serveral dtype
BIT_RATIO_DICT = {"int32": 4, "float32": 4, "float16": 2,
                  "uint8": 1, "int8": 1, "uint4": 0.5, "int4": 0.5}

# the max size is 2**63-1
DATA_SIZE_MAX = 9223372036854775807

# fmapH, fmapW must be in [1, 4096]
FMAP_HW_MAX = 4096
FMAP_HW_MIN = 1

# Dedy H, W must be in [2, 4096]
DEDY_HW_MAX = 4096
DEDY_H_MIN = 1
DEDY_W_MIN = 2

# filterH, filterW must be in [1, 255]
FILTER_HW_MAX = 255
FILTER_HW_MIN = 1

# stride must be in [1, 63]
STRIDE_HW_MAX = 63
STRIDE_HW_MIN = 1

# pad must be in [0, 255]
PAD_MAX = 255
PAD_MIN = 0

# C0_SIZE
C0_SIZE = 16
ORI_SHAPE_LEN = 4
SHAPE_LEN = 5

OP_TYPE = "depthwise_conv2d_backprop_filter"
DYNAMIC_RANK_FLAG = [-2]
LOWER_LIMIT = [{"result": "UNSUPPORTED", "reason": {"param_index": [0, 2], "type": ["lower_limit", "lower_limit"]}}]
UNSUPPORTED = [{"result": "UNSUPPORTED"}]
UPPER_LIMIT = [{"result": "UNSUPPORTED", "reason": {"param_index": [0, 2], "type": ["upper_limit", "upper_limit"]}}]
UNSUPPORTED_DICT = {"upper_limit": UPPER_LIMIT, "lower_limit": LOWER_LIMIT, "unsupported": UNSUPPORTED}


def _check_data_format(data_format, expect_format_list):
    """
    check data format
    """
    if data_format not in expect_format_list:
        error_manager_cube.raise_err_input_params_not_expected("depthwise_conv2d_backprop_filter",
                                                     "data_foramt", str(expect_format_list),
                                                     data_format)


def get_and_check_param_dim(input_fm, out_backprop, filter_grad, param_list):
    """
    check input and output format and return NCHW index
    """
    [strides, pads, dilations, data_format] = param_list
    input_ori_format = input_fm.get('ori_format')
    dedy_ori_format = out_backprop.get('ori_format')
    dedw_ori_format = filter_grad.get('ori_format')
    _check_data_format(input_ori_format, ['NCHW', 'NHWC'])
    _check_data_format(dedy_ori_format, ['NCHW', 'NHWC'])
    if not dedw_ori_format:
        dedw_ori_format = "HWCN"
    else:
        _check_data_format(dedw_ori_format, ['HWCK', 'HWCN', 'NCHW', 'NHWC'])
    _check_data_format(data_format, ['NCHW', 'NHWC'])
    if input_ori_format != data_format or dedy_ori_format != data_format:
        error_manager_cube.raise_err_specific_user("depthwise_conv2d_backprop_filter",
            "input_ori_format/dedy_ori_format must be equal with data_format.")
    if len(strides) != STRIDES_DIM:
        error_manager_cube.raise_err_three_paras("E62304", "depthwise_conv2d_backprop_filter",
                                                 "strides", str(STRIDES_DIM), str(len(strides)))
    if len(pads) != PADS_DIM:
        error_manager_cube.raise_err_three_paras("E62304", "depthwise_conv2d_backprop_filter",
                                                 "pads", str(PADS_DIM), str(len(pads)))
    if len(dilations) != DILATION_DIM:
        error_manager_cube.raise_err_three_paras("E62304", "depthwise_conv2d_backprop_filter",
                                                 "dilations", str(DILATION_DIM), str(len(dilations)))
    # index of origin dimension
    dim_n, dim_c, dim_h, dim_w = pos_from_format(input_ori_format)
    return [dim_n, dim_c, dim_h, dim_w]


def _get_pos_from_format(format_in):
    return {
        "pos_n": format_in.find("N"),
        "pos_c": format_in.find("C"),
        "pos_h": format_in.find("H"),
        "pos_w": format_in.find("W")
    }


def _get_attrs(strides, pads, dilations, data_format):
    pos = _get_pos_from_format(data_format)
    pos_n = pos.get("pos_n")
    pos_c = pos.get("pos_c")
    pos_h = pos.get("pos_h")
    pos_w = pos.get("pos_w")
    dilations = [dilations[pos_n], dilations[pos_c],
                 dilations[pos_h], dilations[pos_w]]

    if len(strides) == 4:
        strides = [strides[pos_h], strides[pos_w]]

    return strides, pads, dilations


def tensor_range_infor(is_fmap, tensor_list, param_list, dynamic_flag):
    """
    get range informations.

    Parameters
    ----------
    tensor: dict with keys(ori_shape, ori_format, shape, format, dtype, range).
    same to conv2d_backprop_filter

    Returns

    ----------
    return tensor.
    """
    tensor, x, out_backprop, y = tensor_list
    [strides, pads, dilations, data_format] = param_list
    status = True
    res = None
    if is_fmap:
        tensor = correct_conv2d_backprop_range_start(tensor, y, dilations, pads, data_format)
        strides, pads, dilations = _get_attrs(strides, pads, dilations, data_format)
        param_list = [strides, pads, dilations]
        status, res = calc_max_fmap_w(x, out_backprop, y, param_list, dynamic_flag)
        if not status:
            return status, res
        if res[0] < FMAP_HW_MAX:
            if tensor.get("ori_format") == "NCHW":
                tensor["ori_range"] = (tensor["ori_range"][0], tensor["ori_range"][1], tensor["ori_range"][2], res)
            else:
                tensor["ori_range"] = (tensor["ori_range"][0], tensor["ori_range"][1], res, tensor["ori_range"][-1])

    tensor["ori_shape"] = [-1, tensor["ori_shape"][1], -1, -1] \
                if tensor.get("ori_format") == "NCHW" else [-1, -1, -1, tensor["ori_shape"][-1]]

    return status, res


def _get_groups(fmap, kernel):
    fm = ConvFeatureMap(fmap)
    k = ConvKernel(kernel)
    if fm.fmap_c is not None and fm.fmap_c != -1:
        return fm.fmap_c
    return k.kernel_c


def _convert_kernel(kernel):
    if kernel.get("ori_format").lower() == "hwck":
        kernel["ori_format"] = "HWCN"
    k = ConvKernel(kernel)
    k_format = k.ori_format.lower()
    kernel["ori_shape"] = list(kernel["ori_shape"])
    kernel["ori_shape"][k_format.index("n")] = k.kernel_cout * k.kernel_c
    kernel["ori_shape"][k_format.index("c")] = 1


@tbe_register.register_param_generalization("DepthwiseConv2DBackpropFilter")
def depthwise_conv2d_backprop_filter_generalization(input_fm,
                                                    filter_size,
                                                    out_backprop,
                                                    filter_grad,
                                                    strides,
                                                    dilations=(1, 1, 1, 1),
                                                    pads=(0, 0, 0, 0),
                                                    data_format='NHWC',
                                                    kernel_name=OP_TYPE,
                                                    generalize_config=None):
    """
    depthwise_conv2d_backprop_filter generalization

    Notice
    ------
    run after infershape and before operator compile
    only modify input and output tensors with range

    for use:
        1. te fusion distinguish .o (remove the generalization dim)
        2. pass them to the operator to follow the dynanmic shape process

    Parameters
    ----------
    same to depthwise_conv2d_backprop_filter
    kernel_name : str
        cce kernel name

    Returns
    -------
    list of params list:
        single item under "keep_rank" mode and multiple under "all_shape"
    """
    support_mode = ["keep_rank"]
    is_generalize_config = (generalize_config is not None and generalize_config.get("mode") in support_mode)
    dynamic_flag = check_graph_mode(input_fm)
    have_range = {"out_backprop": out_backprop, "input_fm": input_fm}
    support_format = ["NCHW", "NHWC"]
    if not is_generalize_config:
        return
    for name, tensor in have_range.items():
        # unknow_rank x ori_shape is [-2], others' shape length is 4
        valid = (isinstance(tensor.get("ori_shape"), (list, tuple)) and len(tensor.get("ori_shape")) == ORI_SHAPE_LEN
                 and list(tensor.get("ori_shape")) != DYNAMIC_RANK_FLAG and tensor.get("ori_format") in support_format)
        if not valid:
            warnings.warn(f'In {kernel_name}, the ori_shape of {name} must be 4d, and the ori_format' +
                          'only support NCHW and NHWC, please check your inputs')
            return UNSUPPORTED
        try:
            tensor = gen_conv_shape_range(tensor, kernel_name, dynamic_flag)
        except RuntimeError as err:
            warnings.warn(err)
            return UNSUPPORTED
        finally:
            pass
        message = check_graph_range(tensor, kernel_name, dynamic_flag)
        if message:
            return UNSUPPORTED_DICT.get(message)
        tensor_list = [tensor, input_fm, out_backprop, filter_grad]
        param_list = [strides, pads, dilations, data_format]
        status, res = tensor_range_infor(name == "input_fm", tensor_list, param_list, dynamic_flag)
        if not status:
            return UNSUPPORTED_DICT.get(res)
    param_list = [strides, pads, dilations, data_format]
    try:
        get_and_check_param_dim(input_fm, out_backprop, filter_grad, param_list)
    except RuntimeError as exc:
        warnings.warn(exc)
        return UNSUPPORTED
    finally:
        pass

    result = [[input_fm, filter_size, out_backprop, filter_grad, {"strides": strides}, {"pads": pads},
               {"dilations": dilations}, {"data_format": data_format}]]
    return result


@tbe_register.register_operator('DepthwiseConv2DBackpropFilter')
@para_check.check_input_type(dict, dict, dict, dict, (tuple, list),
                             (tuple, list), (tuple, list), str, str)
def depthwise_conv2d_backprop_filter(input_fm,
                                     filter_size,
                                     out_backprop,
                                     filter_grad,
                                     strides,
                                     dilations=(1, 1, 1, 1),
                                     pads=(0, 0, 0, 0),
                                     data_format='NHWC',
                                     kernel_name="depthwise_conv2d_backprop_filter"):
    """
    algorithm: depthwise_conv2d_backprop_filter

    calculating  depthwise convolution backward filter

    Parameters
    ----------
    input_fm : a dict.
        4-D origin shape of input tensor [N, C, H, W] or [N, H, W, C],
        support float16.

    filter_size : a dict.
        4-D origin shape of input tensor [N, C, H, W] or [N, H, W, C],
        support float16.

    out_backprop: a dict.
        4-D origin shape of input tensor [N, C, H, W] or [N, H, W, C],
        support float16.

    filter_grad : a dict.
        4-D origin shape of filter tensor [H, W, C, K],
        K is depthwise_multiplier, support float32.

    strides : a list/tuple of four ints
        strides size, [1, 1, stride_height, stride_width] or
        [1, stride_height, stride_width, 1].

    dilations : a list/tuple of four ints
        dilations size, [1, 1, dilation_height, dilation_width] or
        [1, dilation_height, dilation_width, 1].

    pads : a list/tuple of four ints
        padding added to each dimension of the input.

    data_format : str
        shape of origine shape of featuremap [N, C, H, W] or [N, H, W, C].

    kernel_name : str
        cce kernel name

    Returns
    -------
    None
    """

    groups = _get_groups(input_fm, filter_grad)
    _convert_kernel(filter_grad)
    dyn_impl.conv2d_backprop_filter(input_fm, filter_size, out_backprop, filter_grad, strides, pads, dilations, groups,
                                    data_format, kernel_name)
