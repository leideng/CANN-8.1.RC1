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
dilation compute
"""
import warnings
from tbe.dsl.compute.dilation_compute import dilation_compute as dilation_compute_tbe


def dilation_compute(tensor_x, dilations, pads=None, padding_value=0.0):
    """
    dilation_compute
    :param tensor_x: tensor
    :param dilations: list or tuple
    :param pads: list or tuple or None
    :param padding_value: float
    """
    warnings.warn("te.lang.cce.te_compute.dilation_compute is expired, "
        "please replace it with the func tbe.dsl.compute.dilation_compute",
        DeprecationWarning)

    return dilation_compute_tbe(tensor_x, dilations, pads, padding_value)
