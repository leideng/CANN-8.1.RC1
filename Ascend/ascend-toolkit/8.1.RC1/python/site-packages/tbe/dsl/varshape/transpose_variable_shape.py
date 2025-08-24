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
transpose variable shape
"""
from tbe.common.utils.errormgr import get_error_message
from tbe.common.utils.varshape.variable_shape import register_variable
from tbe.dsl.base import operation


@register_variable("transpose")
def variable_shape(inputs):
    # type: (list) -> list
    if len(inputs) != 1:
        dict_args = {"errCode": "E90001", "detailed_cause": "input numbers error"}
        raise RuntimeError(dict_args, get_error_message(dict_args))
    shape_x = inputs[0].get("shape")
    range_x = inputs[0].get("range")
    shape_out = []
    for i, x in enumerate(shape_x):
        if x == -1:
            _var = operation.var_inner(f"_dim_{i}", range_x[i])
            shape_out.append(_var)
        else:
            shape_out.append(x)
    return shape_out
