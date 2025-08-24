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
Compute simulator for DSL
"""
import abc
from typing import List
from typing import Type

import tbe.dsl.padding.graph as m_graph
import tbe.dsl.padding.util as util


def get_simulator(node):
    # type: (m_graph.Node) -> Simulator
    insn = util.get_insn(node)
    simulator = SimulatorManager.build(insn, node)
    if simulator is None:
        util.raise_error(f"Can not find simulator by {insn}")

    return simulator


class Simulator(abc.ABC):
    def __init__(self, node):
        # type: (m_graph.Node) -> None
        self._node = node

    def __init_subclass__(cls):
        # type: () -> None
        SimulatorManager.add_class(cls)

    @abc.abstractmethod
    def adjust_calc(self):
        # type: () -> None
        pass

    @abc.abstractclassmethod
    def get_type(cls):
        # type: () -> str
        pass


class SimulatorManager:
    _simulator_classes = [] # type: List[Type[Simulator]]

    @classmethod
    def add_class(cls, simulator_cls):
        # type: (Type[Simulator]) -> None
        cls._simulator_classes.append(simulator_cls)

    @classmethod
    def build(cls, simulator_type, node):
        # type: (str, m_graph.Node) -> Simulator
        for clz in cls._simulator_classes:
            if clz.get_type() == simulator_type:
                return clz(node)
        return None
