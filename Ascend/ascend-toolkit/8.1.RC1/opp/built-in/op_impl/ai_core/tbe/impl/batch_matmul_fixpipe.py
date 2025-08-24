#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2023-2023 Huawei Technologies Co., Ltd
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
batch_matmul_fixpipe
"""
from impl.dynamic.batch_matmul_fixpipe import batch_matmul_fixpipe_impl


def batch_matmul_fixpipe(x1,
                         x2,
                         quant_pre,
                         bias=None,
                         y=None,
                         adj_x1=False,
                         adj_x2=False,
                         kernel_name="batch_matmul_fixpipe"):
    """
    batch_matmul_fixpipe op.

    Parameters:
    input_x1: dict, required
        A dict object, dict with keys(shape, dtype, and range)

    input_x2: dict, required
        A dict object, dict with keys(shape, dtype and range)

    quant_pre: dict, required
        A dict object of const node, const_value is required.

    bias: dict, optional
        A dict object, dict with keys(shape and dtype) or None

    y: dict
        A dict object, dict with keys(shape, dtype, format and range)

    adj_x1: bool
        If true, shape_a == transposed before multiplication

    adj_x2: bool
        If true, shape_b == transposed before multiplication

    kernel_name: str
        cce kernel_name
    """

    return batch_matmul_fixpipe_impl(x1, x2, quant_pre, bias, y, adj_x1, adj_x2, kernel_name)