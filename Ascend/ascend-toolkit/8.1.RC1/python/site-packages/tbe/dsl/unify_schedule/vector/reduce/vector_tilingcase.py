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
Abstract skeleton class for Tilingcase
"""

# Standard Packages
from abc import ABC
from abc import abstractmethod


class TilingCaseBase(ABC):
    """
    base class of tiling case
    """
    @abstractmethod
    def __hash__(self):
        """"""

    @abstractmethod
    def __eq__(self, other):
        """"""

    @abstractmethod
    def __ne__(self, other):
        """"""
