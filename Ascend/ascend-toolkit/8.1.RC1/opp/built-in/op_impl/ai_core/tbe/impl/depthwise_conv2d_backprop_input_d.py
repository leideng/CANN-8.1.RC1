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
Depthwise conv2D backprop input for the computation of
gradients of depthwise convolution with respect to the input.
"""
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.util_deconv_comm import conv2d_transpose_static_compute
from impl.util.util_deconv_comm import conv2d_transpose_static_impl
import te.lang.cce as tbe


@tbe_platform.fusion_manager.register("depthwise_conv2d_backprop_input_d")
def depthwise_conv2d_backprop_input_d_compute(weight, out_backprop, input_grad, input_size,
                                              strides, dilations=(1, 1, 1, 1), pads=(0, 0, 0, 0),
                                              data_format="NHWC",
                                              kernel_name="depthwise_conv2d_backprop_input_d"):
    """
    algorithm: depthwise conv2d backprop input

    computes the gradients of depthwise convolution with respect to the input

    Parameters
    ----------
    weight: tensor
        4-D origin shape and dtype of weight tensor
        support [H, W, C, K], K is channel_multiplier

    out_backprop: tensor
        4-D origin shape and dtype of out_backprop tensor,
        support [N, Co, Ho, Wo] or [N, Ho, Wo, Co],
        gradients w.r.t. the output of the convolution

    input_grad: dict
        4-D origin shape and dtype of input tensor,
        support [N, C, H, W] or [N, H, W, C]

    input_size: a list or tuple of four ints
        shape of input tensor, support [N, C, H, W] or [N, H, W, C]

    strides: a list or tuple of four ints
        the stride of the sliding window for height and width of the input of
        the convolution, support [1, 1, stride_height, stride_width] or
        [1, stride_height, stride_width, 1]

    dilations: an optional list or tuple of four ints
        the dilation factor for each dimension of input
        if set to k > 1, there will be k-1 skipped cells between each
        filter element on that dimension, support [1, 1, dilation_height,
        dilation_width] or [1, dilation_height, dilation_width, 1]

    pads: a list or tuple of four ints
        padding added to each dimension of the input

    data_format : str
        shape of origine shape of featuremap [N, C, H, W] or [N, H, W, C]

    kernel_name: str
        cce kernel name, default value is "depthwise_conv2d_backprop_input"

    Returns
    -------
    dedx tensor
    """
    tensor_dict = {"weight": weight, "fmap_dedy": out_backprop, "fmap_dedx": input_grad}
    option_input = {}
    attrs_dict = {"input_size": input_size, "strides": strides, "pads": pads,
                  "dilations": dilations, "data_format": data_format, "kernel_name": kernel_name}
    return conv2d_transpose_static_compute(tensor_dict, option_input, attrs_dict, "depthwise_conv2d_backprop_input")


# 'pylint: disable=too-many-statements, redefined-builtin
# 'pylint: disable=locally-disabled, too-many-locals, too-many-arguments, invalid-name
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def depthwise_conv2d_backprop_input_d(filter,
                                      out_backprop,
                                      input_grad,
                                      input_size,
                                      strides,
                                      dilations=(1, 1, 1, 1),
                                      pads=(0, 0, 0, 0),
                                      data_format='NHWC',
                                      kernel_name="depthwise_conv2d_backprop_input"):
    """
    algorithm: depthwise conv2d backprop input

    computes the gradients of depthwise convolution with respect to the input

    Parameters
    ----------
    filter: dict
        4-D origin shape and dtype of filter tensor
        support [H, W, C, K], K is channel_multiplier

    out_backprop: dict
        4-D origin shape and dtype of out_backprop tensor,
        support [N, Co, Ho, Wo] or [N, Ho, Wo, Co],
        gradients w.r.t. the output of the convolution

    input_grad: dict
        4-D origin shape and dtype of input tensor,
        support [N, C, H, W] or [N, H, W, C]

    input_size: a list or tuple of four ints
        shape of input tensor, support [N, C, H, W] or [N, H, W, C]

    strides: a list or tuple of four ints
        the stride of the sliding window for height and width of the input of
        the convolution, support [1, 1, stride_height, stride_width] or
        [1, stride_height, stride_width, 1]

    dilations: an optional list or tuple of four ints
        the dilation factor for each dimension of input
        if set to k > 1, there will be k-1 skipped cells between each
        filter element on that dimension, support [1, 1, dilation_height,
        dilation_width] or [1, dilation_height, dilation_width, 1]

    pads: a list or tuple of four ints
        padding added to each dimension of the input

    data_format : str
        shape of origine shape of featuremap [N, C, H, W] or [N, H, W, C]

    kernel_name: str
        cce kernel name, default value is "depthwise_conv2d_backprop_input"

    Returns
    -------
    None
    """
    tensor_dict = {"weight": filter, "fmap_dedy": out_backprop, "fmap_dedx": input_grad}
    option_input = {}
    attrs_dict = {"input_size": input_size, "strides": strides, "pads": pads,
                  "dilations": dilations, "data_format": data_format, "kernel_name": kernel_name}
    tensor_dedx, tensor_list = conv2d_transpose_static_impl(
        tensor_dict, option_input, attrs_dict, "depthwise_conv2d_backprop_input")
    with tvm.target.cce():
        sch = tbe.auto_schedule(tensor_dedx)

    config = {"name": kernel_name, "tensor_list": tensor_list}
    tbe.cce_build_code(sch, config)
