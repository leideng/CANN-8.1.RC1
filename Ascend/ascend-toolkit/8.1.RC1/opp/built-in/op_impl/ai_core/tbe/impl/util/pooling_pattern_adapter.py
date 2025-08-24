#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
pooling pattern adapter
"""


class PoolingPattern:
    """
    pooling pattern common config from tbe modele
    """
    # the extra_params of pooling pattern dict key
    WINDOW_DIMENSIONS = "window_dimensions"
    WINDOW_STRIDES = "window_strides"
    WINDOW_DILATIONS = "window_dilations"
    PADDING_DIMENSIONS = "padding_dimensions"
    GLOBAL_POOLING = "global_pooling"

    # define the attr info in compile info
    WINDOW_DIMENSIONS_ATTR_IDNEX = "window_dimensions_attr_index"
    WINDOW_STRIDES_ATTR_IDNEX = "window_strides_attr_index"
    WINDOW_DILATIONS_ATTR_IDNEX = "window_dilations_attr_index"
    PADDING_DIMENSIONS_ATTR_IDNEX = "padding_dimensions_attr_index"
    ACTUAL_WINDOW_ORI_INDICES = "actual_window_ori_indices"


class ReduceWindowAttr:
    """
    attrs for reduce window
    """
    GMP = "GMP"
    MAX = "MAX"
    CEIL = "CEIL"
    FLOOR = "FLOOR"
