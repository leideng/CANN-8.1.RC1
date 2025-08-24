#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
Compute of depthwise conv2d.
"""
import warnings


def depthwise_conv2d_compute(fmap,
                             weight,
                             depthwise_res_dtype,
                             stride,
                             pad,
                             dilation,
                             para_dict,
                             l1_fusion_para,
                             kernel_name="depthwise_conv2d_compute"):
    """
    algorithm: depthwise_conv2d_compute

    calculating  depthwise convolution compute

    the interface will be eliminated soon!

    Parameters
    ----------
    fmap : feature map placehold
        5-D shape of input tensor [N, C1, H, W, C0]

    weight : filter placehold
        5-D shape of filter tensor [C1, H, W, Co, C0]

    depthwise_res_dtype : dtype of depthwise UB result

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    pad : padding added to each dimension of the input

    dilation : the dilation factor for each dimension of input

    para_dict : bias tensor dict

    Returns
    -------
    depthwise_res : result tensor
       forward depthwise result of out
    """
    warnings.warn("te.lang.cce.te_compute.depthwise_conv2d_compute is expired, "
        "please replace it with the func tbe.dsl.compute.depthwise_conv2d_compute",
        DeprecationWarning)
    from tbe.dsl.compute.depthwise_conv2d_compute import depthwise_conv2d_compute
    return depthwise_conv2d_compute(fmap, weight, depthwise_res_dtype, stride, pad, dilation, para_dict,
                                    l1_fusion_para, kernel_name)


def depthwise_conv2d_backprop_filter_d_compute(fmap,
                                               dout,
                                               kernel_h,
                                               kernel_w,
                                               stride,
                                               pad,
                                               dilations,
                                               w_dtype,
                                               kernel_name="depthwise_conv2d_compute"):
    """
    compute of depthwise conv2d backprop filter
    
    the interface will be eliminated soon!

    Parameters
    ----------
    fmap : tvm tensor
        feature map tensor in tvm.

    dout : tvm tensor
        dout tensor in tvm.

    kernel_h: int
        height of filter.

    kernel_w: int
        width of filter.

    stride: tuple or list or int
        stride of convolution.

    pad: list
        padding added to each dimension of the input.

    w_dtype: str
        the dtype of dfilter.

    Returns
    -------
    depthwise_dfilter_res: tvm tensor
        the tensor of output.
    """
    warnings.warn("te.lang.cce.te_compute.depthwise_conv2d_compute is expired, "
        "please replace it with the func tbe.dsl.compute.depthwise_conv2d_compute",
        DeprecationWarning)
    from tbe.dsl.compute.depthwise_conv2d_compute import depthwise_conv2d_backprop_filter_d_compute
    return depthwise_conv2d_backprop_filter_d_compute(fmap, dout, kernel_h, kernel_w, stride, pad, dilations,
                                                      w_dtype, kernel_name)


def depthwise_conv2d_backprop_input_d_compute(input_shape,
                                              weight,
                                              dout,
                                              weight_sizes,
                                              strides,
                                              pads,
                                              kernel_name="depthwise_conv2d_compute"):
    """
    Computes the gradients of depthwise convolution with respect to the input.

    the interface will be eliminated soon!

    Parameters
    ----------
    input_shape: a list or tuple representing the shape of input,
                6D format [N, C1, 1, H, W, C0]

    weight: a tensor, 5D with shape [C1, Hf*Wf, 1, C0, C0]

    dout: a tensor, 6D format [N, Co1, 1, Ho, Wo, C0]

    weight_sizes: a list or tuple of two ints,
                  the height and width of the weight of the convolution

    strides: a list or tuple of two ints, the stride of the sliding window for
             height and width of the input of the convolution

    pads: padding added to each dimension of the input

    Returns
    -------
    dx_res: compute of the gradients of depthwise convolution
            with respect to the input
    """
    warnings.warn("te.lang.cce.te_compute.depthwise_conv2d_compute is expired, "
        "please replace it with the func tbe.dsl.compute.depthwise_conv2d_compute",
        DeprecationWarning)
    from tbe.dsl.compute.depthwise_conv2d_compute import depthwise_conv2d_backprop_input_d_compute
    return depthwise_conv2d_backprop_input_d_compute(input_shape, weight, dout, weight_sizes, strides, pads,
                                                     kernel_name)
