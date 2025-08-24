#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2023-2024 Huawei Technologies Co., Ltd
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
conv2d_transpose_d_compress
"""
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.util_deconv_comm import check_conv2d_transpose
from impl.util.util_deconv_comm import conv2d_transpose_static_compute
from impl.util.util_deconv_comm import conv2d_transpose_static_impl
from impl.util.util_deconv_comm import trans_shape_by_index
from tbe.common.utils.const import WEIGHT_SPARSE_4_2
from tbe.common.platform import platform_info
from tbe.common.register import register_pass_for_fusion


@register_pass_for_fusion(match_condition={"op_type":"Conv2DTransposeDCompress"})
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
        if op_info.get_op_type() in ["Conv2DTransposeDCompress"]:
            trans_shape_by_index(3, op_info, graph_info)


def check_supported(
    x,
    filter_compress,
    compress_index,
    bias,
    offset_w,
    y,
    input_size,
    strides,
    pads,
    dilations=(1, 1, 1, 1),
    groups=1,
    data_format="NHWC",
    output_padding=(0, 0, 0, 0),
    offset_x=0,
    alg=WEIGHT_SPARSE_4_2,
    kernel_name="conv2d_transpose_d_compress",
):
    """
    batch_x == batch_y

    the h and w must meet:
       (hi - 1) * stride_h - (pad_u + pad_d) + (hk - 1) * dilation_h + output_padding_h + 1 = ho
       (wi - 1) * stride_w - (pad_l + pad_r) + (wk - 1) * dilation_w + output_padding_w + 1 = ho
    """
    try:
        tensor_dict = {"fmap_dedy": x, "weight": filter_compress, "compress_index": compress_index, "fmap_dedx": y}
        option_input = {"bias": bias, "offset_w": offset_w}
        attrs_dict = {"input_size": input_size, "strides": strides, "pads": pads, "dilations": dilations,
                      "groups": groups, "data_format": data_format, "output_padding": output_padding,
                      "offset_x": offset_x, "kernel_name": kernel_name, "alg": alg}
        check_conv2d_transpose(tensor_dict, option_input, attrs_dict, op_type="conv2d_transpose_d_compress")
        return True, ""
    except Exception as e:
        reason = e.args[1]
        return False, reason


@para_check.check_op_params(
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.REQUIRED_INPUT,
    para_check.OPTION_INPUT,
    para_check.OPTION_INPUT,
    para_check.REQUIRED_OUTPUT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.REQUIRED_ATTR_LIST_INT,
    para_check.OPTION_ATTR_LIST_INT,
    para_check.OPTION_ATTR_INT,
    para_check.OPTION_ATTR_STR,
    para_check.OPTION_ATTR_LIST_INT,
    para_check.OPTION_ATTR_INT,
    para_check.OPTION_ATTR_STR,
    para_check.KERNEL_NAME,
)
def conv2d_transpose_d_compress(
    x,
    filter_compress,
    compress_index,
    bias,
    offset_w,
    y,
    input_size,
    strides,
    pads,
    dilations=(1, 1, 1, 1),
    groups=1,
    data_format="NHWC",
    output_padding=(0, 0, 0, 0),
    offset_x=0,
    alg=WEIGHT_SPARSE_4_2,
    kernel_name="conv2d_transpose_d_compress",
):
    """
    algorithm: conv2d_transpose_d_compress

    Parameters
    ----------
    x: dict with keys(shape and dtype)
        The shape of gradients.

    filter_compress: dict with keys(shape and dtype)
        input compressed filter tensor.

    compress_index: dict with keys(shape and dtype)
        input ND compress index

    bias: dict with keys(shape and dtype)
        The shape of bias.

    offset_w: dict with keys(shape and dtype) or None
        Input offset_w tensor.

    y: dict with keys(shape and dtype)
       conv2d_transpose_d output tensor, dtype must be assigned.

    input_size: tuple/list of 4 integers
        The shape of feature map. 4-D with shape [batch, height, width, channels]
        or [batch, channels, height, width].

    strides: tuple/list of 4 integers
        filter move stride.

    pads: tuple/list of 4 integers
        [pad_top, pad_bottom, pad_left, pad_right].

    dilations: tuple/list of 4 integers
        filter expand size of dilated conv2d_transpose_d. Default to (1, 1, 1, 1).

    groups: int
        param for group conv2d_transpose_d. Default to 1.

    data_format: str
        input data format. Specify the data format of the input and output data.
        Default to "NHWC".

    output_padding: tuple/list of 4 integers
        The size will be added in the output shape. Default to (0, 0, 0, 0).

    offset_x: int
        offset of gradients in quant mode. Default to 1.

    alg: str
        compress algorithm. Default to weight_sparse_4_2.

    kernel_name: str
        kernel name. Default to "conv2d_transpose_d_compress".

    Returns
    -------
    None
    """
    tensor_dict = {"fmap_dedy": x, "weight": filter_compress, "compress_index": compress_index, "fmap_dedx": y}
    option_input = {"bias": bias, "offset_w": offset_w}
    attrs_dict = {"input_size": input_size, "strides": strides, "pads": pads, "dilations": dilations,
                  "groups": groups, "data_format": data_format, "output_padding": output_padding,
                  "offset_x": offset_x, "kernel_name": kernel_name, "alg": alg}
    tensor_dedx, tensor_list = conv2d_transpose_static_impl(tensor_dict, option_input, attrs_dict,
                                                            "conv2d_transpose_d_compress")
    with tvm.target.cce():
        sch = tbe.auto_schedule(tensor_dedx)
    # build config
    config = {"name": kernel_name, "tensor_list": tensor_list}

    tbe.build(sch, config)


@tbe_platform.fusion_manager.register("conv2d_transpose_d_compress")
def conv2d_transpose_d_compress_compute(
    x,
    filter_compress,
    compress_index,
    bias,
    offset_w,
    y,
    input_size,
    strides,
    pads,
    dilations=(1, 1, 1, 1),
    groups=1,
    data_format="NHWC",
    output_padding=(0, 0, 0, 0),
    offset_x=0,
    alg=WEIGHT_SPARSE_4_2,
    kernel_name="conv2d_transpose_d_compress",
):
    """
    used for fusion
    Parameters
    ----------
    x: Tensor
        input gradients tensor.

    filter_compress: Tensor
        input compressed filter tensor.

    compress_index: Tensor
        input compress index tensor.

    bias: Tensor or None
        input bias tensor.

    offset_w: Tensor or None
        input offset_w tensor.

    y: dict with keys(shape and dtype)
       conv2d_transpose_d_compress output tensor, dtype must be assigned.

    input_size: tuple/list of 4 integers
        the shape of feature map. 4-D with shape [batch, height, width, channels]
        or [batch, channels, height, width].

    strides: tuple/list of 4 integers
        filter move stride.

    pads: tuple/list of 4 integers
        [pad_top, pad_bottom, pad_left, pad_right].

    dilations: tuple/list of 4 integers
        filter expand size of dilated conv2d_transpose_d_compress. Default to (1, 1, 1, 1).

    groups: int
        param for group conv2d_transpose_d_compress. Default to 1.

    data_format: str
        input data format. Specify the data format of the input and output data.
        Default to "NHWC".

    output_padding: tuple/list of 4 integers
        The size will be added in the output shape. Default to (0, 0, 0, 0).

    offset_x: int
        offset of gradients in quant mode. Default to 0.

    alg: str
        compress algorithm. Default to weight_sparse_4_2.

    kernel_name: str
        kernel name. Default to "conv2d_transpose_d_compress".

    Returns
    -------
    Tensor of conv2d_transpose_d_compress
    """
    tensor_dict = {"fmap_dedy": x, "weight": filter_compress, "compress_index": compress_index, "fmap_dedx": y}
    option_input = {"bias": bias, "offset_w": offset_w}
    attrs_dict = {"input_size": input_size, "strides": strides, "pads": pads, "dilations": dilations,
                  "groups": groups, "data_format": data_format, "output_padding": output_padding,
                  "offset_x": offset_x, "kernel_name": kernel_name, "alg": alg}

    return conv2d_transpose_static_compute(tensor_dict, option_input, attrs_dict, "conv2d_transpose_d_compress")