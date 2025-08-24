#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
layer_norm_beta_gamma_backprop_v2 tiling case
"""

from tbe.dsl.base.operation import register_tiling_case
from tbe.dsl.base.operation import get_compile_info
from .constants import Pattern
from ..base import operation


@register_tiling_case(pattern=Pattern.LAYER_NORM_BETA_GAMMA_BACKPROP_V2)
def calc_layernorm_beta_gamma_v2(outs, option=None):
    """tiling_case func for dynamic shape layernormbetagammabackpropv2

    Parameters
    ----------
    outs: tvm tensor or list of tvm tensor, results for tvm compute

    Returns
    -------
    list of dict, each dict for a tiling case
    """
    compile_info = get_compile_info()
    dynamic_reduce = compile_info.get("dynamic_reduce")
    dynamic_normal = compile_info.get("dynamic_normal")
    max_last_factor = compile_info.get("max_last_factor")
    core_num = compile_info.get("core_num")
    current_compute = operation.get_context().get_current_compute()
    no_reduce = current_compute.get("no_reduce")
    reduce_dim = current_compute.get("reduce_dim")
    normal_dim = current_compute.get("normal_dim")

    if no_reduce:
        # (-1, )
        return [("no_reduce", 400)]
    elif dynamic_reduce and dynamic_normal:
        # (-1, -1)
        return [("all_dynamic_no_split", 200),
                ("all_dynamic_split_normal", 201),
                ("all_dynamic_split_reduce", 202),
                ("all_dynamic_split_reduce_split_normal", 203),
                ("all_dynamic_split_reduce_i", 204)]
    elif dynamic_reduce:
        # (-1, N)
        if normal_dim <= max_last_factor:
            return [("dynamic_reduce_no_split", 100),
                    ("dynamic_reduce_split_reduce", 101),
                    ("dynamic_reduce_split_reduce_i", 102)]
        else:
            return [("dynamic_reduce_no_split", 100),
                    ("dynamic_reduce_split_reduce", 101)]
    elif dynamic_normal:
        # (N, -1)
        if reduce_dim <= core_num:
            return [("dynamic_normal_no_split", 300),
                    ("dynamic_normal_split_normal", 301)]
        else:
            return [("dynamic_normal_split_reduce", 302),
                    ("dynamic_normal_split_reduce_split_normal", 303),
                    ("dynamic_normal_split_reduce_i", 304)]

