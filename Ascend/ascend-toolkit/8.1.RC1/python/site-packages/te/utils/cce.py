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
cce
"""
from __future__ import absolute_import as _abs
import warnings


def auto_schedule(outs, option=None):
    """Entry of auto-Schedule.

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of reduce in the format
          of an array of tensors.
    option:
    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    warnings.warn("te.lang.cce.auto_schedule is deprecated, please replace it with the tbe.dsl.auto_schedule",
                  DeprecationWarning)
    import tbe
    return tbe.dsl.auto_schedule(outs, option)


def build(sch, config_map=None):
    """
    :param sch:
    :param config_map:
    :return:
    """
    warnings.warn("te.lang.cce.build is deprecated, please replace it with the tbe.dsl.build",
                  DeprecationWarning)
    import tbe
    return tbe.dsl.build(sch, config_map)
