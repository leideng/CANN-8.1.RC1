#!/usr/bin/env python
# -*- coding: UTF-8 -*-
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
pooling2d compute
"""


# 'pylint: disable=too-many-arguments
def get_caffe_out_size_and_pad(ceil_mode, in_size_h, in_size_w, window_h, window_w,
                               stride_h, stride_w, dilation_h, dilation_w, pad_top,
                               pad_bottom, pad_left, pad_right):
    """
    :param ceil_mode: caffe round_mode params, 0:CEIL(default), 1:FLOOR
    :param in_size_h: input h
    :param in_size_w: input w
    :param window_h: window h
    :param window_w: window w
    :param stride_h: stride h
    :param stride_w: stride w
    :param dilation_h: dilation h
    :param dilation_w: dilation w
    :param pad_top: pad top
    :param pad_bottom: pad bottom
    :param pad_left: pad left
    :param pad_right: pad right
    :return:
    """
    from tbe.dsl.compute.pooling2d import get_caffe_out_size_and_pad
    return get_caffe_out_size_and_pad(ceil_mode, in_size_h, in_size_w, window_h, window_w,
                                      stride_h, stride_w, dilation_h, dilation_w, pad_top,
                                      pad_bottom, pad_left, pad_right)
