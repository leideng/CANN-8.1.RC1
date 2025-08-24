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
reduce pattern adapter
"""


class ReducePattern:
    """
    reduce pattern common config from tbe modele
    """
    # the extra_params of reduce pattern dict key
    KEEP_DIMS_KEY = "keepdims"
    REDUCE_AXES_TYPE_KEY = "reduce_axes_type"
    # define norm patter name from tbe.dsl.base.shape_classifier.REDUCE in shape_classifier.py
    PATTERN_NAME = "reduce"

    # define the keep dims for extra_params
    KEEP_DIMS_FALSE = {KEEP_DIMS_KEY: False}
    KEEP_DIMS_TRUE = {KEEP_DIMS_KEY: True}
    # define the reduce_axes_type for extra_params
    # come from tbe.dsl.base.classifier.reduce_classifier.ReduceMode  in reduce_classifier.py
    REDUCE_MODE_REDUCE_ALL = {REDUCE_AXES_TYPE_KEY: "all"}
