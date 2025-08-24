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
conv2d backprop filter DSL interface.
"""
import warnings
from tbe.dsl.compute.conv2d_backprop_filter_compute \
    import conv2d_backprop_filter_compute as conv2d_backprop_filter_compute_tbe


def conv2d_backprop_filter_compute(input_x, out_backprop, filter_sizes, para_dict):
    """
    the DSL interface of conv2d backprop filter compute

    Parameters:
    ----------
    x : the featuremap data, tvm.placeholder, 5HD shape

    out_backprop : the grads data, tvm.placeholder, 5HD shape

    filter_sizes : 4-D shape, specifies the filter sizes

    para_dict:

    strides : 2-D shape, specifies in height and width dimension

    padding : 4-D shape, specifies in up/down/left/right dimension

    dilations : 4-D shape, specifies in batch/channel/height/width dimension

    groups : The number of filter's group. Default value is 1.

    res_dtype : the output data type

    Returns
    -------
    result tensor of conv2d_backprop_filter compute
    """
    warnings.warn("te.lang.cce.te_compute.conv2d_backprop_filter_compute is expired, "
        "please replace it with the func tbe.dsl.compute.conv2d_backprop_filter_compute",
        DeprecationWarning)

    return conv2d_backprop_filter_compute_tbe(input_x, out_backprop, filter_sizes, para_dict)
