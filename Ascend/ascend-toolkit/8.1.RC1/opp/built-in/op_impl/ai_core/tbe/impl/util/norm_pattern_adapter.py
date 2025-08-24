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
norm pattern adapter
"""


class NormPattern:
    """
    norm pattern common config from tbe modele
    """
    REDUCE_AXES_TYPE_KEY = "reduce_axes_type"
    # define norm patter name from tbe.dsl.base.shape_classifier.NORM in shape_classifier.py
    PATTERN_NAME = "norm"
    # define reduce mode from tbe.dsl.base.classifier.norm_classifier.UNKNOWN in norm_classifier.py
    REDUCE_UNKNOWN_MODE = "unknown"
    BROADCAST_UNKNOWN_MODE = "unknown"

    # define the reduce type for extra_params
    # come from tbe.dsl.base.classifier.norm_classifier.ReduceAxisType in norm_classifier.py
    REDUCE_ASSIGNED_TYPE = {REDUCE_AXES_TYPE_KEY: "assigned"}
    REDUCE_ANY_TYPE = {REDUCE_AXES_TYPE_KEY: "any"}
    REDUCE_SINGLE_TYPE = {REDUCE_AXES_TYPE_KEY: "single"}
    REDUCE_AFTER_TYPE = {REDUCE_AXES_TYPE_KEY: "after"}
    REDUCE_BEFORE_TYPE = {REDUCE_AXES_TYPE_KEY: "before"}

    # define the reduce key of compile info for attr info
    REDUCE_ATTR_IDX = "reduce_axis_attr_idx"
    REDUCE_ATTR_NAME = "reduce_axis_attr_name"
    REDUCE_ATTR_DTYPE = "reduce_axis_attr_dtype"
