#!/usr/bin/env python
# -*- coding:UTF-8 -*-
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
conv2d DSL interface.
"""
import warnings


def is_support_v200():
    """
    Check if Ascend610/BS9SX1A/Ascend310P/Hi3796CV300CS version.
    ----------

    Returns
    -------
    True:  Ascend610/BS9SX1A/Ascend310P/Hi3796CV300CS version
    False: Other version
    """
    warnings.warn("te.lang.cce.te_compute.conv_compute.is_support_v200 is deprecated, " \
        "please replace it with tbe.dsl.compute.conv_compute.is_support_v200",
                  DeprecationWarning)
    from tbe.dsl.compute.conv_compute import is_support_v200 as new_is_support_v200
    return new_is_support_v200()


def check_conv_shape(shape_in, shape_w, pad_top, pad_bottom,
                     pad_left, pad_right, strideh, stridew, in_dtype, w_dtype,
                     optim_dict=None, dilateh=1, dilatew=1, dynamic_para=None, groups=1):
    """

    Parameters
    ----------
    shape_in: shape of data_in

    shape_w: shape of filter

    padh: the padding shape in H

    padw: the padding shape in weight

    strideh: the stride value in H

    stridew: the stride value in weight

    dilateh: the dilate value in H

    dilatew: the dilate value in weight

    optim_dict: optimize feature dict

    in_dtype: the feature map data type

    w_dtype: the weight data type

    Returns
    -------
    None

    """
    warnings.warn("te.lang.cce.te_compute.conv_compute.check_conv_shape is deprecated, " \
        "please replace it with tbe.dsl.compute.conv_compute.check_conv_shape",
                  DeprecationWarning)
    from tbe.dsl.compute.conv_compute import check_conv_shape as new_check_conv_shape
    return new_check_conv_shape(shape_in, shape_w, pad_top, pad_bottom, pad_left,
                                pad_right, strideh, stridew, in_dtype, w_dtype,
                                optim_dict, dilateh, dilatew, dynamic_para, groups)


def conv_compress(inputs, weight_compress, compress_index, compress_index_shape,
                  para_dict, optim_dict=None, dsl_flag=True):
    """
    This is conv compress compute.
    """
    warnings.warn("te.lang.cce.te_compute.conv_compute.conv_compress is deprecated, " \
        "please replace it func tbe.dsl.compute.conv_compute.conv_compress",
                  DeprecationWarning)
    from tbe.dsl.compute.conv_compute import conv_compress as new_conv_compress
    return new_conv_compress(inputs, weight_compress, compress_index, compress_index_shape,
                             para_dict, optim_dict, dsl_flag)


def conv(data, weight, para_dict, optim_dict=None, dsl_flag=True):
    """
    conv

    Parameters
    ----------
    data: feature map

    weight: filter

    para_dict: dict of params

    dsl_flag: true if not from topi

    Returns
    -------
    tensor: res
    """
    warnings.warn("te.lang.cce.te_compute.conv_compute.conv is deprecated, " \
        "please replace it with func tbe.dsl.compute.conv_compute.conv",
                  DeprecationWarning)
    from tbe.dsl.compute.conv_compute import conv as new_conv
    return new_conv(data, weight, para_dict, optim_dict, dsl_flag)
