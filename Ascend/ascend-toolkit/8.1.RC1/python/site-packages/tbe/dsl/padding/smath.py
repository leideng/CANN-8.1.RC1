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
Caculate
"""
from typing import Union

import numpy as np

Number = Union[np.floating, np.integer]


def abs_(a):
    # type: (Number) -> Number
    return np.abs(a)


def exp_(a):
    # type: (Number) -> Number
    return np.exp(a)


def log_(a):
    # type: (Number) -> Number
    return np.log(a)


def rec_(a):
    # type: (Number) -> Number
    return np.reciprocal(a)


def rsqrt_(a):
    # type: (Number) -> Number
    return np.reciprocal(np.sqrt(a))


def sqrt_(a):
    # type: (Number) -> Number
    return np.sqrt(a)


def add_(a, b):
    # type: (Number, Number) -> Number
    return a + b


def sub_(a, b):
    # type: (Number, Number) -> Number
    return a - b


def mul_(a, b):
    # type: (Number, Number) -> Number
    return a * b


def div_(a, b):
    # type: (Number, Number) -> Number
    ret = np.divide(a, b)
    if isinstance(a, np.integer):
        return a//b
    return ret


def max_(a, b):
    # type: (Number, Number) -> Number
    return np.maximum(a, b)


def min_(a, b):
    # type: (Number, Number) -> Number
    return np.minimum(a, b)


def bitwise_and_(a, b):
    # type: (Number, Number) -> Number
    return np.bitwise_and(a, b)


def bitwise_or_(a, b):
    # type: (Number, Number) -> Number
    return np.bitwise_or(a, b)


def bitwise_not_(a):
    # type: (Number) -> Number
    return np.bitwise_not(a)


def relu_(a):
    # type: (Number) -> Number
    return np.maximum(a, 0)


def lrelu_(a, b):
    # type: (Number, Number) -> Number
    return np.where(a < 0, a * b, a).take(0)


def cast_(a, dtype):
    # type: (Number, str) -> Number
    return a.astype(dtype)


def ceil_(a, dtype):
    # type: (Number, str) -> Number
    return np.ceil(a).astype(dtype)


def floor_(a, dtype):
    # type: (Number, str) -> Number
    return np.floor(a).astype(dtype)


def trunc_(a, dtype):
    # type: (Number, str) -> Number
    return np.trunc(a).astype(dtype)


def round_(a, dtype):
    # type: (Number, str) -> Number
    return np.round(a).astype(dtype)


def round_d_(a, dtype):
    # type: (Number, str) -> Number
    return np.round(a).astype(dtype)
