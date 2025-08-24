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
deconvolution
"""
from impl.conv2d_transpose_d import conv2d_transpose_d
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.util_deconv_comm import check_conv2d_transpose
from impl.util.util_deconv_comm import conv2d_transpose_static_compute
from impl.util.util_deconv_comm import get_op_support_info_conv2d_transpose
from impl.util.util_deconv_comm import trans_shape_by_index
from tbe.common.utils.errormgr import error_manager_cube
from tbe.common.platform import platform_info
from tbe.common.register import register_pass_for_fusion


@register_pass_for_fusion(match_condition={"op_type":"Deconvolution"})
def trans_dx_bias_shape(graph_info):
    """
    tansform dx bias op shape if necessary
    original func :
    Parameters
    ----------
    graph_info : GraphInfo
        graph info

    Returns
    -------
    """
    if not platform_info.get_soc_spec("L0A_LAYOUT_IS_zN"):
        return
    if graph_info.is_dynamic_shape():
        return
    for op_info in graph_info.get_op_list():
        if op_info is None:
            continue
        if op_info.get_op_type() in ["Deconvolution"]:
            trans_shape_by_index(2, op_info, graph_info)


def check_supported(x,
                    weight,
                    bias,
                    offset_w,
                    y,
                    strides,
                    pads,
                    dilations=(1, 1, 1, 1),
                    groups=1,
                    data_format="NCHW",
                    offset_x=0,
                    kernel_name="deconvolution"):
    """
    batch_x == batch_y

    The h and w must meet:
       hi - (hk - 1)*dk + 1 + padh // strideh = ho
       wi - (wk - 1)*wk + 1 + padw // stridew = wo
    """
    shape_x = x.get("ori_shape")
    dynamic_flag = any([i < 0 for i in shape_x])
    if dynamic_flag:
        return True, ""
    try:
        tensor_dict = {"weight": weight, "fmap_dedy": x, "fmap_dedx": y}
        option_input = {"bias": bias, "offset_w": offset_w}
        attrs_dict = {"strides": strides, "pads": pads, "dilations": dilations,
                      "groups": groups, "data_format": data_format,
                      "offset_x": offset_x, "kernel_name": kernel_name}
        check_conv2d_transpose(tensor_dict, option_input, attrs_dict, op_type="deconvolution")
        return True, ""
    except Exception as e:
        reason = e.args[1]
        return False, reason


def get_op_support_info(x,  # pylint: disable=invalid-name,R0913,R0914,W0613
                        weight,
                        bias,
                        offset_w,
                        y,
                        strides,
                        pads,
                        dilations=(1, 1, 1, 1),
                        groups=1,
                        data_format="NCHW",
                        offset_x=0,
                        kernel_name="deconvolution"):
    """
    get the deconvolution split
    """
    tensor_dict = {"weight": weight, "fmap_dedy": x, "fmap_dedx": y}
    option_input = {"bias": bias, "offset_w": offset_w}
    attrs_dict = {"strides": strides, "pads": pads, "dilations": dilations,
                  "groups": groups, "data_format": data_format,
                  "offset_x": offset_x, "kernel_name": kernel_name}
    return get_op_support_info_conv2d_transpose(tensor_dict, option_input, attrs_dict, op_type="deconvolution")


@para_check.check_op_params(
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.OPTION_INPUT,
    para_check.OPTION_INPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.OPTION_ATTR_LIST_INT,
    para_check.OPTION_ATTR_INT,
    para_check.OPTION_ATTR_STR,
    para_check.OPTION_ATTR_INT,
    para_check.KERNEL_NAME,
)
def deconvolution(  # pylint: disable=invalid-name,R0913,R0914,W0613
    x,
    weight,
    bias,
    offset_w,
    y,
    strides,
    pads,
    dilations=(1, 1, 1, 1),
    groups=1,
    data_format="NCHW",
    offset_x=0,
    kernel_name="deconvolution",
):
    """
    algorithm: deconvolution

    Parameters
    ----------
    x: dict with keys(shape and dtype)
                  The shape of gradients.

    weight: dict with keys(shape and dtype)
            input weight tensor

    bias: dict with keys(shape and dtype)
        The shape of bias.

    offset_w: the offset for weight

    y: dict with keys(shape and dtype)
       deconvolution output tensor, dtype must be assigned

    strides: tuple/list of 2 integers
             filter move stride

    pads: tuple/list of 4 integers
             [pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 4 integers
               filter expand size of dilated deconvolution
    groups: int
            param for group deconvolution

    data_format: str
            An optional string from: "NCHW". Defaults to "NCHW".
            Specify the data format of the input and output data.

    offset_x: offset of gradients in quant mode

    kernel_name: str
                 kernel name, default value is "deconvolution"

    Returns
    -------
    None
    """
    input_size = list(y.get("ori_shape", (0, 0, 0, 0)))
    if len(strides) == 2:
        # convert strides from 2-dim to 4-dim
        stride_h, stride_w = strides
        strides = [1] * 4
        strides[data_format.find('H')] = stride_h
        strides[data_format.find('W')] = stride_w
    conv2d_transpose_d(x, weight, bias, offset_w, y, input_size, strides, pads, dilations=dilations,
                       groups=groups, data_format=data_format, offset_x=offset_x, kernel_name=kernel_name)


@tbe_platform.fusion_manager.register("deconvolution")
def deconvolution_compute(  # pylint: disable=invalid-name,R0913,R0914,W0613
    x,
    weight,
    bias,
    offset_w,
    y,
    strides,
    pads,
    dilations=(1, 1, 1, 1),
    groups=1,
    data_format="NCHW",
    offset_x=0,
    kernel_name="deconvolution",
):
    """
    used for fusion
    Parameters
    ----------
    x: dict with keys(shape and dtype)
                  The shape of gradients.

    weight: dict with keys(shape and dtype)
            input weight tensor

    offset_w: the offset for weight

    bias: dict with keys(shape and dtype)
        The shape of bias.

    y: dict with keys(shape and dtype)
       deconvolution output tensor, dtype must be assigned

    strides: tuple/list of 2 integers
             filter move stride

    pads: tuple/list of 4 integers
             [pad_top, pad_bottom, pad_left, pad_right]

    dilations: tuple/list of 4 integers
               filter expand size of dilated deconvolution
    groups: int
            param for group deconvolution

    data_format: str
            An optional string from: "NCHW". Defaults to "NCHW".
            Specify the data format of the input and output data.

    offset_x: offset of gradients in quant mode

    kernel_name: str
                 kernel name, default value is "deconvolution"

    Returns
    -------
    None
    """
    tensor_dict = {"weight": weight, "fmap_dedy": x, "fmap_dedx": y}
    option_input = {"bias": bias, "offset_w": offset_w}
    attrs_dict = {"strides": strides, "pads": pads, "dilations": dilations,
                  "groups": groups, "data_format": data_format,
                  "offset_x": offset_x, "kernel_name": kernel_name}

    return conv2d_transpose_static_compute(tensor_dict, option_input, attrs_dict, "deconvolution")
