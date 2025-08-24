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
conv3d backprop input DSL interface.
"""
import warnings
from tbe.dsl.compute.conv3d_backprop_input_compute import conv3d_dx


def conv3d_dx(filter, out_backprop, filter_size, input_size, para_dict):
    """
    DSL interface of conv3d bp dx

    Parameters
    ----------
    filter : weight tensor of fractal shape

    out_backprop : 5D dE/dY tensor

    filter_size : shape of weight, [D, H, W, C, N]

    input_size : shape of dE/dX, [N, D, H, W, C]

    para_dict : dict of parameters
    strides : list of strides, [stridebatch, strided, strideh, stridew, stridechannel]
    pads : list of padding, [pad_front, pad_tail, pad_up, pad_down, pad_left, pad_right]
    dilations : [1, 1, 1, 1, 1] by default
    res_dtype : dE/dX data type, "float16" by default
    kernel_name : conv3d_backprop_input_cce by default
    group_dict : group of parameters

    Returns
    ----------
    dx_ddr: dE/dX tensor
    """
    warnings.warn("te.lang.cce.te_compute.conv3d_backprop_input_compute is expired, "
        "please replace it with the func tbe.dsl.compute.conv3d_backprop_input_compute",
        DeprecationWarning)
    return conv3d_dx(filter, out_backprop, filter_size, input_size, para_dict)
