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
conv2d backprop input DSL interface.
"""
import warnings
from tbe.dsl.compute.conv2d_backprop_input_compute import \
    conv2d_backprop_input_compute as conv2d_backprop_input_compute_new


def conv2d_backprop_input_compute(filters, out_backprop, filter_sizes, input_sizes, para_dict):
    """
    DSL interface of conv2d backprop input

    Parameters
    ----------
    filters : weight tensor of fractal shape

    out_backprop : 5D dE/dY tensor

    filter_sizes : shape of weight, [N, C, H, W]

    input_sizes : shape of dE/dX, [N, C, H, W]

    para_dict:

    strides : list of strides, [strideh, stridew]

    padding : list of padding, [pad_up, pad_down, pad_left, pad_right]

    dilations : list of dilations, [dilation_n, dilation_c, dilation_h, dilation_w]

    res_dtype : dE/dX data type, "float16" by default

    offset_x : offset of x

    offset_w : offset of w

    fusion_para: the l1 fuison para

    kernel_name : cce kernel name

    group_dict : The params of group convolution.

    Returns
    ----------
    dx_ddr: dE/dX tensor
    """
    warnings.warn("te.lang.cce.te_compute.conv2d_backprop_input_compute is expired, "
        "please replace it with the func tbe.dsl.compute.conv2d_backprop_input_compute",
        DeprecationWarning)
    return conv2d_backprop_input_compute_new(filters, out_backprop, filter_sizes, input_sizes, para_dict)
