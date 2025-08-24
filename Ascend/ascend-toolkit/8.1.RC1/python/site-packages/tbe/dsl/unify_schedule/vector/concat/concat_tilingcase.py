#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright 2021-2021 Huawei Technologies Co., Ltd
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
concat tiling case
"""
from enum import Enum
from enum import auto
from typing import Optional

from tbe.dsl.base.operation import get_context
from tbe.dsl.base.operation import add_build_arg
from tbe.dsl.base.operation import register_build_pointcut
from tbe.dsl.unify_schedule.computation import Computation
from tbe.dsl.unify_schedule.constants import Pattern
from tbe.dsl.unify_schedule.constants import ConcatPattern
from tbe.dsl.unify_schedule import util

DEFAULT = "default"
EMPTY = "concat_empty"

EMPTY_KEY = 2 ** 31 - 1


class TilingStrategy(Enum):
    """
    TilingStrategy
    """
    NONE_CUT = auto()
    CONST = auto()
    GENERAL = auto()
    READ_ALIGN = auto()
    READ_ALIGN_NO_UB = auto()
    ONE_CONCAT = auto()
    GENERAL_NO_ALIGN = auto()
    LAST_HALF_DIVISIBLE = auto()
    LAST_HALF_DIVISIBLE_NO_ALIGN = auto()
    LAST_ALL_ONE = auto()
    EMPTY = auto()


class ConcatComputation(Computation):
    """
    Concat Tilingcase Computation
    """

    def __init__(self, outs, option):
        self.out = outs[0] if isinstance(outs, (list, tuple)) else outs
        self.option = option

    def get_sub_pattern(self):
        return ConcatPattern.C_0

    @classmethod
    def get_instance(cls, outs, option):
        return cls(outs, option)

    @classmethod
    def get_supported_pattern(cls):
        return [Pattern.CONCAT]

    @classmethod
    def get_supported_soc(cls):
        return [DEFAULT]

    def do_tiling_case(self):  # type: () -> list[Any]
        def is_const(shapes):
            return all(isinstance(s, int) for s in shapes)

        def add_double_split_tiling_case(base_key, tiling_strategy):
            for i in range(dim_len):
                case = ConcatTilingCase()
                case.tiling_key = base_key + i
                case.tiling_strategy = tiling_strategy
                case.block_split_axis = i
                tiling_case.append(case)

        def add_single_split_tiling_case(tiling_key, tiling_strategy):
            case = ConcatTilingCase()
            case.tiling_key = tiling_key
            case.tiling_strategy = tiling_strategy
            case.block_split_axis = 0
            tiling_case.append(case)

        shape = util.shape_to_list(self.out.shape)
        dim_len = len(shape)

        tiling_case = []

        mode = get_context().get_current_compute().get("_mode")
        if mode == EMPTY:
            add_single_split_tiling_case(EMPTY_KEY, TilingStrategy.EMPTY)
            return tiling_case

        if is_const(shape):
            add_single_split_tiling_case(1000000, TilingStrategy.CONST)
            return tiling_case

        add_double_split_tiling_case(3000000, TilingStrategy.ONE_CONCAT)

        add_double_split_tiling_case(4000000, TilingStrategy.READ_ALIGN)
        add_single_split_tiling_case(4100000, TilingStrategy.READ_ALIGN_NO_UB)

        if shape[0] == 1:
            return tiling_case[1:]

        # general, no split: no block tiling, no ub tiling , no db
        add_single_split_tiling_case(0, TilingStrategy.NONE_CUT)

        add_double_split_tiling_case(2000000, TilingStrategy.GENERAL)
        add_single_split_tiling_case(2100000, TilingStrategy.GENERAL_NO_ALIGN)

        add_double_split_tiling_case(5000000, TilingStrategy.LAST_HALF_DIVISIBLE)
        add_single_split_tiling_case(5100000, TilingStrategy.LAST_HALF_DIVISIBLE_NO_ALIGN)

        add_single_split_tiling_case(6000000, TilingStrategy.LAST_ALL_ONE)

        return tiling_case


class ConcatTilingCase:
    """
    Concat Tiling Case
    """
    _enable_db: bool

    def __init__(self):
        self._tiling_key = 0
        self._tiling_strategy: Optional[Enum] = None
        self._block_split_axis = 0
        self._enable_db = False

    @property
    def tiling_key(self):
        """
        :return: tiling_key
        """
        return self._tiling_key

    @property
    def tiling_strategy(self):
        """
        :return: tiling_strategy
        """
        return self._tiling_strategy

    @property
    def block_split_axis(self):
        """
        :return: block_split_axis
        """
        return self._block_split_axis

    @property
    def enable_db(self):
        """
        enable_db
        """
        return self._enable_db

    @tiling_key.setter
    def tiling_key(self, value):
        """
        set tiling_key
        :param value:
        :return:
        """
        self._tiling_key = value

    @tiling_strategy.setter
    def tiling_strategy(self, value):
        """
        set tiling_strategy
        :param value:
        :return:
        """
        self._tiling_strategy = value

    @block_split_axis.setter
    def block_split_axis(self, value):
        """
        set block_split_axis
        :param value:
        :return:
        """
        self._block_split_axis = value

    @enable_db.setter
    def enable_db(self, value):
        """
        set enable_db
        :param value:
        :return:
        """
        self._enable_db = value


@register_build_pointcut(pattern=Pattern.CONCAT)
def build_pointcut(func, *args, **kwargs):
    """
    build pointcut
    :param func:
    :param args:
    :param kwargs:
    :return:
    """
    add_build_arg("enable_branch_eliminator_else_case", False)
    func(*args, **kwargs)
