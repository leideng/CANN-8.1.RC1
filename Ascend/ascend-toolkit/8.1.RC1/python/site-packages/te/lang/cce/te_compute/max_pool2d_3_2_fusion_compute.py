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
max_pool_v200
"""
import warnings


def max_pool_compute(input_data,  # pylint: disable=too-many-arguments
                     ksize, strides, pad_mode="VALID",
                     padding=(0, 0, 0, 0), ceil_mode=0, data_mode=0):
    """
    Performs max pooling on the input.

    Parameters
    ----------
    input_data: tensor
        tensor of input_data.
    dtype: str
        input and output data type.
    ksize: list or tuple
        A list of `ints` that has length 2
        The size of the window for H, W dimension of the input tensor
    strides: list or tuple
        A list of `ints` that has length 4.
        The stride of the sliding window of the input tensor
    pad_mode: str
        A `string` from: "SAME", "VALID"`.The type of padding algorithm to use
    padding: list or tuple
        A `padding to use
    data_mode: int
        A int can be 0 : CAFFE_DATA_MODE, 1: TENSORFLOW_DATA_MODE
    cei_mode : int
        A int caffe round_mode params, 0:CEIL(default), 1:FLOOR

    Returns:
    -------
    res:
        The result of max pooling
    """
    warnings.warn("te.lang.cce.te_compute.max_pool2d_3_2_fusion_compute.max_pool_compute is deprecated, " \
        "please replace it with tbe.dsl.compute.max_pool2d_3_2_fusion_compute.max_pool_compute",
                  DeprecationWarning)
    from tbe.dsl.compute.max_pool2d_3_2_fusion_compute import max_pool_compute as new_max_pool_compute
    return new_max_pool_compute(input_data, ksize, strides, pad_mode, padding, ceil_mode, data_mode)
