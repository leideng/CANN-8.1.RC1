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
conv2d tiling case
"""
import warnings


# noinspection PyUnusedLocal
def calc_conv2d(outs, option=None):
    """
    tiling_case func for dynamic shape conv2d

    Parameters
    ----------
    outs: tvm tensor or list of tvm tensor, results for tvm compute

    Returns
    -------
    list of dict, each dict for a tiling case
    """
    warnings.warn("te.lang.dynamic.schedule.conv2d_tilingcase.calc_conv2d is deprecated, " \
        "please replace it with tbe.dsl.unify_schedule.conv2d_tilingcase.calc_conv2d",
                  DeprecationWarning)
    from tbe.dsl.unify_schedule.conv2d_tilingcase import calc_conv2d as new_calc_conv2d
    return new_calc_conv2d(outs, option)
