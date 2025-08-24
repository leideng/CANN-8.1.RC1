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
from tbe import tvm
from tbe.common import register as tbe_register
from tbe.common.utils import para_check
from tbe.common.utils import log
from tbe.common.utils.errormgr import error_manager_cube as err_man
from impl.util.util_cube_dynamic import BIT_RATIO_DICT
from impl.util.platform_adapter import tbe_platform
from impl.util.util_conv2d_dynamic import modify_input_range
from impl.util.util_conv2d_dynamic import check_l1_size
from impl.util.util_conv2d_dynamic import create_fuzz_range
from impl.util.util_conv2d_dynamic import correct_input_range
from impl.util.util_conv2d_dynamic import get_format_attr
from impl.util.util_conv2d_dynamic import check_graph_mode
from impl.util.util_conv2d_dynamic import check_conv2d_range
from .conv2d import conv2d
from .conv2d import conv2d_fusion_compute

NONETYPE = type(None)
ORI_SHAPE_LEN = 4
DYNAMIC_VALUE = -1


def gen_depthwise_conv2d_range(inputs, weights, strides, pads, dilations):
    """
    fuzz input range
    """
    op_type = "depthwise_conv2d"
    x_shape = inputs.get("ori_shape")
    x_format = inputs.get("ori_format")
    x_range = inputs.get("ori_range")
    w_shape = weights.get("ori_shape")
    w_format = weights.get("ori_format")
    data_type = weights.get("dtype")

    idx_n = 0
    if x_format == "NCHW":
        idx_c = 1
        idx_h = 2
        idx_w = 3
        dilh = dilations[2]
        dilw = dilations[3]
    elif x_format == "NHWC":
        idx_h = 1
        idx_w = 2
        idx_c = 3
        dilh = dilations[1]
        dilw = dilations[2]
    else:
        err_man.raise_err_specific_user(op_type, "input fmap format only support NCHW or NHWC")

    # x_range instance when empty
    if not x_range:
        x_range = []
        for idx, _ in enumerate(x_shape):
            if x_shape[idx] == DYNAMIC_VALUE:
                x_range.append([1, -1])
            else:
                x_range.append([x_shape[idx], x_shape[idx]])

    kh, kw = get_format_attr(w_shape, w_format)
    kh_dilate = dilh*(kh - 1) + 1
    kw_dilate = dilw*(kw - 1) + 1
    grade_n = [0, 1, 3, 7, 15, 31, ((1 << 31) - 1)]
    grade_h = [0, 3, 15, 63, 127, 191, 255, 511, 767, 1023, 4096]
    grade_w = [0, 3, 15, 63, 127, 191, 255, 511, 767, 1023, 4096]
    grade_map = {idx_n : grade_n, idx_h : grade_h, idx_w : grade_w}
    input_range = [[], [], [], []]

    for idx, grade_item in grade_map.items():
        # allow input_shape -1 with range
        if x_shape[idx] == DYNAMIC_VALUE:
            input_range[idx] = x_range[idx]
        else:
            input_range[idx] = create_fuzz_range(op_type, x_shape[idx], grade_item)

    input_range[idx_c] = [x_shape[idx_c], x_shape[idx_c]]
    log.debug("depthwise_conv2d fuzz input range is :%s", input_range)
    # output_h or output_w > 0
    correct_input_range(op_type, input_range, x_shape, idx_h, idx_w, kh_dilate, kw_dilate, pads)
    log.debug("depthwise_conv2d fuzz input range is corrected for output_w > 0, :%s", input_range)
    # check fmap exceed l1buffer
    if x_shape[idx_w] != DYNAMIC_VALUE:
        check_l1_size(op_type, inputs, kh_dilate, kw_dilate, strides, pads)
    attr_params = [strides, kh_dilate, kw_dilate, pads]
    new_in_range = modify_input_range(inputs, input_range, data_type, idx_h, idx_w, attr_params)
    log.debug("depthwise_conv2d fuzz input range is modified for no exceed l1buffer, :%s", new_in_range)

    return new_in_range


@tbe_register.register_param_generalization("DepthwiseConv2D")
def depthwise_conv2d_generalization(x, filter, bias, offset_w, y, strides, dilations=(1, 1, 1, 1),
                                    pads=(0, 0, 0, 0), data_format='NHWC', offset_x=0, kernel_name="depthwise_conv2d",
                                    generalize_config=None):
    """
    depthwise_conv2d generalization

    Notice
    ------
    run after infershape and before operator compile
    only modify input and output tensors with range

    for use:
        1. te fusion distinguish .o (remove the generalization dim)
        2. pass them to the operator to follow the dynanmic shape process

    Parameters
    ----------
    same to depthwise_conv2d

    Returns
    -------
    list of params list:
        single item under "keep_rank" mode and multiple under "all_shape"
    """
    def get_tensor_ori_shape():
        if tensor.get("ori_format") == "NCHW":
            return [-1, tensor["ori_shape"][1], -1, -1]
        return [-1, -1, -1, tensor["ori_shape"][3]]
    if generalize_config is None:
        generalize_config = {"mode": "keep_rank"}
    support_mode = ["keep_rank"]
    if generalize_config.get("mode") not in support_mode:
        err_man.raise_err_specific_user("depthwise_conv2d", "invalid generalize mode {}, only support {}".format(
            str(generalize_config.get("mode")), str(support_mode)))
    result = []
    if generalize_config.get("mode") == "keep_rank": # fuzz build situation
        # unknow_rank inputs ori_shape is [-2], others' shape length is 4
        unknow_rank = len(x["ori_shape"]) == 1 and x["ori_shape"][0] == -2
        if unknow_rank:
            err_man.raise_err_specific_user("depthwise_conv2d", "not support unknow_rank under mode {}".format(
                generalize_config.get("mode")))
        log.debug("depthwise_conv2d generalization inputs: %s", x)
        if not check_graph_mode(x):
            x_range = gen_depthwise_conv2d_range(x, filter, strides, pads, dilations)
            x["ori_range"] = x_range
            have_range = {"x": x, "y": y}
            for name, tensor in have_range.items():
                # only change shape NHW dim to -1, range is already set at infershape
                valid = isinstance(tensor.get("ori_shape"), (list, tuple)) and len(tensor["ori_shape"]) == ORI_SHAPE_LEN
                if not valid:
                    err_man.raise_err_specific_user("depthwise_conv2d", "invalid {} ori_shape {}, only support {}d"
                        .format(name, str(tensor.get("ori_shape")), str(ORI_SHAPE_LEN)))
                tensor["ori_shape"] = get_tensor_ori_shape()
        else:
            check_result = check_conv2d_range(x, filter, strides, pads, dilations)
            if check_result:
                log.debug("depthwise_conv2d generalization invalid range, check_result: %s", check_result)
                return check_result
        result.append([x, filter, bias, offset_w, y, strides, dilations, pads, data_format, offset_x, kernel_name])

    log.debug("depthwise_conv2d generalization result: %s", result)
    return result


@tbe_register.register_op_compute("DepthwiseConv2D", op_mode="dynamic", support_fusion=True)
@para_check.check_input_type(tvm.Tensor, tvm.Tensor, (tvm.Tensor, NONETYPE),
                             (tvm.Tensor, NONETYPE), dict, (tuple, list), (tuple, list),
                             (tuple, list), str, int, str)
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
    filter_format = filter.op.attrs['ori_format']
    if filter_format == "HWCN":
        groups = filter.op.attrs['ori_shape'][3].value
    elif filter_format == "NCHW" or filter_format == "NHWC":
        groups = filter.op.attrs['ori_shape'][0].value
    else:
        err_man.raise_err_input_format_invalid("depthwise_conv2d", "filter", \
            ["HWCN", "NCHW", "NHWC"], filter["ori_format"])

    out = conv2d_fusion_compute(fmap, filter, bias, offset_w, out, strides, pads, dilations,
                                groups=groups, data_format=data_format, offset_x=offset_x, kernel_name=kernel_name)
    return out


@tbe_register.register_operator("DepthwiseConv2D")
@para_check.check_input_type(dict, dict, (dict, NONETYPE), (dict, NONETYPE), dict,
                             (tuple, list), (tuple, list), (tuple, list),
                             str, int, str)
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
    x_shape = x.get("ori_shape")
    if x["ori_format"] == "NCHW":
        x_c = x_shape[1]
    elif x["ori_format"] == "NHWC":
        x_c = x_shape[3]
    else:
        err_man.raise_err_input_format_invalid("depthwise_conv2d", "x", \
            ["NCHW", "NHWC"], x["ori_format"])

    w_shape = filter.get("ori_shape")
    if filter["ori_format"] == "HWCN":
        filter_n = w_shape[3] * w_shape[2]
        filter_c = w_shape[2]
        filter_h = w_shape[0]
        filter_w = w_shape[1]
    elif filter["ori_format"] == "NCHW":
        filter_n = w_shape[0] * w_shape[1]
        filter_c = w_shape[1]
        filter_h = w_shape[2]
        filter_w = w_shape[3]
    elif filter["ori_format"] == "NHWC":
        filter_n = w_shape[0] * w_shape[3]
        filter_c = w_shape[3]
        filter_h = w_shape[1]
        filter_w = w_shape[2]
    else:
        err_man.raise_err_input_format_invalid("depthwise_conv2d", "filter", \
            ["HWCN", "NCHW", "NHWC"], filter["ori_format"])

    filter["ori_shape"] = [filter_n, 1, filter_h, filter_w]
    filter["ori_format"] = "NCHW"

    conv2d(x, filter, bias, offset_w, y, strides, pads, dilations,
           groups=x_c, data_format=data_format, offset_x=offset_x, kernel_name=kernel_name)

