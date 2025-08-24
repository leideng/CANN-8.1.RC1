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
multiple simulator
"""
import tbe.dsl.padding.simulator as m_simulator
import tbe.dsl.padding.util as util


class MlaSimulator(m_simulator.Simulator):
    @classmethod
    def get_type(cls):
        # type: () -> str
        return "elewise_multiple_mla"

    def adjust_calc(self):
        # type: () -> None
        util.raise_error("Unsupported.")


class MaddSimulator(m_simulator.Simulator):
    @classmethod
    def get_type(cls):
        # type: () -> str
        return "elewise_multiple_madd"

    def adjust_calc(self):
        util.raise_error("Unsupported.")


class AxpySimulator(m_simulator.Simulator):
    @classmethod
    def get_type(cls):
        # type: () -> str
        return "elewise_binary_scalar_axpy"

    def adjust_calc(self):
        # type: () -> None
        util.raise_error("Unsupported.")


class MaddReluSimulator(m_simulator.Simulator):
    @classmethod
    def get_type(cls):
        # type: () -> str
        return "elewise_multiple_maddrelu"

    def adjust_calc(self):
        # type: () -> None
        util.raise_error("Unsupported.")
