#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2023 Huawei Technologies Co., Ltd
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
im2col
"""
# 'pylint: disable=too-many-lines
from tbe.common.utils.errormgr import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.dynamic.extract_image_patches_nchw import ExtractImagePatchesNCHW


def prepare_params(attr_value):
    if len(attr_value) == 1:
        new_attr_value = [1, 1, attr_value[0], attr_value[0]]
    elif len(attr_value) == 2:
        new_attr_value = [1, 1, attr_value[0], attr_value[1]]
    else:
        new_attr_value = attr_value
    return new_attr_value


@register_operator("Im2col", pattern="ExtractImagePatches")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_LIST_INT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_LIST_INT, para_check.KERNEL_NAME)
def im2col(images, y, ksizes, strides, dilations, padding_mode, pads, kernel_name="im2col"):
    """
    calculating data

    Parameters
    ----------
    images : dict
        shape and dtype of input, support float16/float
    y : dict
        shape and dtype of output, should be same shape and type as input
    ksizes: input attr
    strides: input attr
    dilates: input attr
    padding: input attr
    pads: input attr
    kernel_name : str
        kernel name, default value is "image_to_col"

    Returns
    -------
    None
    """
    data_format = images.get("format")
    if data_format != "NCHW":
        error_manager_vector.raise_err_input_format_invalid(kernel_name, "x", ('NCHW'), data_format)

    dtype_input = images.get("dtype").lower()
    if dtype_input not in ("float16", "float", "float32", "bfloat16"):
        error_manager_vector.raise_err_specific_reson(kernel_name,
                                                      "dtype can only be bfloat16, float16, float or float32!")

    is_binary = None in (ksizes, strides, dilations, pads)
    if not is_binary:
        ksizes = prepare_params(ksizes)
        strides = prepare_params(strides)
        dilations = prepare_params(dilations)
        if len(pads) == 1:
            pads = [pads[0], pads[0], pads[0], pads[0]]
        elif len(pads) == 2:
            pads = [pads[0], pads[0], pads[1], pads[1]]
        else:
            pads = pads
    return ExtractImagePatchesNCHW(images, ksizes, strides, dilations, padding_mode, pads).build(kernel_name)
