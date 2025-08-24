#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
depthwise_conv2d
"""
import json
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator_compute
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import error_manager_util
from impl.conv2d import conv2d
from impl.conv2d import op_select_format as conv2d_op_select_format
from impl.conv2d import conv2d_compute
from impl.util import util_conv2d


def _depthwise_conv2d_fusion_para(inputs, outputs):
    """
    get L1 fusion para for depthwise_conv2d
    """
    fmap_l1_addr_flag = inputs.op.attrs["L1_addr_flag"] if "L1_addr_flag" in inputs.op.attrs else -1
    fmap_l1_valid_size = inputs.op.attrs["L1_valid_size"] if "L1_valid_size" in inputs.op.attrs else -1

    fusion_para = {
        "fmap_l1_addr_flag": fmap_l1_addr_flag,
        "fmap_l1_valid_size": fmap_l1_valid_size
    }

    return fusion_para


def op_select_format(inputs, weights, bias, offset_w, outputs, strides,
                     dilations, pads, data_format='NHWC',
                     offset_x=0, kernel_name="depthwiseConv2d"):
    if 'ori_shape' not in inputs or 'ori_format' not in inputs:
        error_manager_cube.raise_err_specific_user("depthwise_conv2d: [op_select_format]",
                                                   "ori_shape and ori_format are needed in fmap attrs")
    all_fmt = ["NCHW", "NHWC", "HWCN"]
    fmap_ori_format = inputs.get('ori_format')
    if fmap_ori_format not in all_fmt:
        error_manager_cube.raise_err_specific_user("depthwise_conv2d, fmap only support ori_format: ",
                                                   "[NCHW, NHWC, HWCN]")
    pos_c = fmap_ori_format.find('C')
    groups = inputs.get('ori_shape')[pos_c]
    return conv2d_op_select_format(inputs, weights, bias, offset_w, outputs, strides,
                                   pads, dilations, groups, data_format,
                                   offset_x, "depthwiseConv2d")


# pylint: disable=locally-disabled, too-many-locals, too-many-arguments,
# pylint: disable=unused-argument
# pylint: disable=redefined-builtin
@register_operator_compute("depthwise_conv2d", op_mode="static", support_fusion=True)
def depthwise_compute(fmap,
                      filter,
                      bias,
                      offset_w,
                      out,
                      strides,
                      dilations,
                      pads,
                      data_format='NHWC',
                      offset_x=0,
                      kernel_name="depthwise_conv2d"):
    """
    algorithm: depthwise conv2d compute
    calculating  depthwise compute
    Parameters
    ----------
    fmap : a tensor of featureMap
    filter : a tensor of filter
    bias : a tensor of bias
    offset_w : a tensor of filter offset
    out : a dict of output
        {"shape", "dtype"}
        shape of input tensor [N, C1, H, W, C0],
        support float16.
    strides : a list/tuple of four ints
        strides size, [1, 1, stride_height, stride_width] or
        [1, stride_height, stride_width, 1]
    dilations : a list/tuple of four ints
        dilation size, [1, 1, dilation_height, dilation_width] or
        [1, dilation_height, dilation_width, 1]
    pads : padding added to each dimension of the input
    data_format : a str of featuremap original shape
        shape of origine shape of featuremap [N, C, H, W] or [N, H, W, C]
    offset_x : offset of the input
    Returns
    -------
    None
    """
    if 'ori_shape' not in fmap.op.attrs or 'ori_format' not in fmap.op.attrs:
        error_manager_cube.raise_err_specific_user("depthwise_conv2d",
                                                   "ori_shape and ori_format are needed in fmap.op.attrs")
    fmap_c_dict = {"HWCN": fmap.op.attrs['ori_shape'][2].value,
                   "NCHW": fmap.op.attrs['ori_shape'][1].value,
                   "NHWC": fmap.op.attrs['ori_shape'][3].value}
    groups = fmap_c_dict.get(fmap.op.attrs['ori_format'])
    out['op_type'] = "DepthwiseConv2D"
    out = conv2d_compute(fmap, filter, bias, offset_w, out, strides, pads,
                         dilations, groups=groups, data_format=data_format,
                         offset_x=offset_x, kernel_name=kernel_name)

    return out


def _check_shape(fmap_shape, filter_shape, fmap_data_format):
    """check input shape"""
    _, in_c1, _, _, _ = fmap_shape
    filter_c1, _, _, filter_k, _, _ = filter_shape

    # check feature map API feature map  shape is 5hd
    # The shape of feature map and filter must be 5HD
    if len(fmap_shape) != FEATURE_MAP_DIM:
        dict_args = {
            'errCode': 'E60008',
            'op_name': 'depthwise_conv2d',
            'param_name': 'featuremap',
            'expected_format_list': '[{}]'.format('NC1HWC0'),
            'format': fmap_data_format
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    # check feature map shape of c, equal filter of c
    if in_c1 != filter_c1:
        dict_args = {
            'errCode': 'E60002',
            'op_name': 'depthwise_conv2d',
            'attr_name': 'channel',
            'param1_name': 'fmap',
            'param2_name': 'filter',
            'param1_value': str(in_c1),
            'param2_value': str(filter_c1)
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


def _check_data_format(data_format, expect_format_list):
    """
    check data format
    """
    if data_format not in expect_format_list:
        dict_args = {
            'errCode': 'E50002',
            'op_name': 'depthwise_conv2d',
            'param': 'featuremap',
            'expected_format_list': str(expect_format_list),
            'format': data_format
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


def _check_stride(strides, dim_n, dim_c, dim_h, dim_w):
    """
    check stride type and dim
    """
    if not isinstance(strides, (list, tuple)) and len(strides) == 4:
        dict_args = {'errCode': 'E60107', 'op_name': 'depthwise_conv2d', 'param_name': 'strides'}
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    if strides[dim_n] != 1 or strides[dim_c] != 1:
        error_manager_cube.raise_err_specific_user("depthwise_conv2d", "stride only support 1 in N axis and C axis.")


def _check_dilations(dilations, dim_n, dim_c, dim_h, dim_w):
    """
    check dilations dimension
    """
    if dilations[dim_n] != 1 or dilations[dim_c] != 1:
        dict_args = {
            'errCode': 'E60023',
            'op_name': 'depthwise_conv2d',
            'dilation_n': str(dilations[dim_n]),
            'dilation_c': str(dilations[dim_c])
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


# pylint: disable=locally-disabled, too-many-locals, too-many-arguments,
# pylint: disable=unused-argument
# pylint: disable=redefined-builtin, invalid-name
@para_check.check_op_params(
    para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
    para_check.OPTION_INPUT, para_check.OPTION_INPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_ATTR_LIST_INT, para_check.OPTION_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT, para_check.OPTION_ATTR_STR,
    para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
def depthwise_conv2d(
        x,
        filter,
        bias,
        offset_w,
        y,
        strides,
        dilations=(1, 1, 1, 1),
        pads=(0, 0, 0, 0),
        data_format='NHWC',
        offset_x=0,
        kernel_name="depthwise_conv2d",
):
    """
    algorithm: depthwise conv2d

    calculating  depthwise convolution

    Parameters
    ----------
    x : a dict of featureMap
        {"shape", "dtype", "format"}
        shape of input tensor [N, C1, H, W, C0],
        support float16.

    filter : a dict of filter
        {"shape", "dtype"}
        shape of filter tensor [C1, H, W, K, Co, C0],
        K is depthwise_multiplier, support int.

    bias : a dict of bias
        {"shape", "dtype"}
        shape of bias tensor [C1*C0,]
        support int8.

    offset_w : a dict of filter offset
        {"shape", "dtype"}
        shape of offset tensor [C1, H, W, K, Co, C0]
        support float16.

    y : a dict of output
        {"shape", "dtype"}
        shape of input tensor [N, C1, H, W, C0],
        support float16.

    strides : a list/tuple of four ints
        strides size, [1, 1, stride_height, stride_width] or
        [1, stride_height, stride_width, 1]

    dilations : a list/tuple of four ints
        dilation size, [1, 1, dilation_height, dilation_width] or
        [1, dilation_height, dilation_width, 1]

    pads : padding added to each dimension of the input

    data_format : a str of featuremap original shape
        shape of origine shape of featuremap [N, C, H, W] or [N, H, W, C]

    offset_x : offset of the input

    kernel_name : str
       cce kernel name

    Returns
    -------
    None

    """
    w_shape = filter.get("ori_shape")
    x_shape = x.get("ori_shape")
    if x["ori_format"] == "NCHW":
        x_c = x_shape[1]
    elif x["ori_format"] == "NHWC":
        x_c = x_shape[3]
    if filter["ori_format"] == "HWCN":
        filter_n = w_shape[3]*w_shape[2]
        filter_h = w_shape[0]
        filter_w = w_shape[1]
    elif filter["ori_format"] == "NCHW":
        filter_n = w_shape[0] * w_shape[1]
        filter_h = w_shape[2]
        filter_w = w_shape[3]
    elif filter["ori_format"] == "NHWC":
        filter_n = w_shape[0] * w_shape[3]
        filter_h = w_shape[1]
        filter_w = w_shape[2]
    else:
        dict_args = {
            'errCode': 'E50002',
            'op_name': 'depthwise_conv2d',
            'param': 'filter',
            'expected_format_list': str(["NCHW", "NHWC", "HWCN"]),
            'format': filter["ori_format"]
        }
        raise RuntimeError(
            dict_args, error_manager_util.get_error_message(dict_args))
    filter_c = 1

    filter["ori_shape"] = [filter_n, filter_c, filter_h, filter_w]
    filter["ori_format"] = "NCHW"
    y['op_type'] = "DepthwiseConv2D"

    conv2d(x, filter, bias, offset_w, y, strides, pads, dilations,
          groups=x_c, data_format=data_format, offset_x=offset_x,
          kernel_name=kernel_name)


def get_op_support_info(x,
                        weights,
                        bias,
                        offset_w,
                        outputs,
                        strides,
                        pads,
                        dilations,
                        groups=1,
                        data_format='NCHW',
                        offset_x=0,
                        kernel_name="depthwiseconv2d"):
    """
    algorithm: get_op_support_info

    Notice
    ------
    get the depthwiseconv2d split

    Parameters
    ----------
    x : a dict of featureMap
        {"shape", "dtype", "format"}
        shape of input tensor [N, C1, H, W, C0],
        support float16.

    filter : a dict of filter
        {"shape", "dtype"}
        shape of filter tensor [C1, H, W, K, Co, C0],
        K is depthwise_multiplier, support int.

    bias : a dict of bias
        {"shape", "dtype"}
        shape of bias tensor [C1*C0,]
        support int8.

    offset_w : a dict of filter offset
        {"shape", "dtype"}
        shape of offset tensor [C1, H, W, K, Co, C0]
        support float16.

    y : a dict of output
        {"shape", "dtype"}
        shape of input tensor [N, C1, H, W, C0],
        support float16.

    strides : a list/tuple of four ints
        strides size, [1, 1, stride_height, stride_width] or
        [1, stride_height, stride_width, 1]

    dilations : a list/tuple of four ints
        dilation size, [1, 1, dilation_height, dilation_width] or
        [1, dilation_height, dilation_width, 1]
    kernel_name: str
        kernel name, default value is "depthwiseconv2d"

    Returns
    -------
    None
    """
    bias_idx = 2
    format_x = x.get("format")
    slice_info = util_conv2d.get_op_support_info_static_common(bias, bias_idx, format_x)

    # >>> start: process for dynamic shape
    shape_x = x.get("ori_shape")
    # shape is [-2], all axes do not support split
    if list(shape_x) == [-2]:
        slice_info.get("_op_slice_info").get("splitMaps").clear()
    else:
        # H/W shape is -1, remove corresponding split info
        format_fm = x.get("ori_format")
        overlap_axis = {"H": [2], "W": [3]}
        temp_info = slice_info.get('_op_slice_info').get("splitMaps")
        for name, index in overlap_axis.items():
            if shape_x[format_fm.find(name)] == -1:
                last_maps = filter(lambda splits: splits["inputList"][0]["axis"] != index, temp_info)
                temp_info = list(last_maps)
        slice_info["_op_slice_info"]["splitMaps"] = temp_info
    # <<< end: process for dynamic shape

    return json.dumps(slice_info)
