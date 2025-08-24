#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from common_func.ms_constant.number_constant import NumberConstant
from profiling_bean.struct_info.struct_decoder import StructDecoder


class StaticOpMemBean(StructDecoder):
    """
    Operator memory bean under static graph scenarios
    """
    NODE_INDEX_END_MAX = 4294967294
    CORRECT_NODE_INDEX_END_MAX = 4294967295

    def __init__(self: any, *args) -> None:
        filed = args[0]
        self._op_mem_size = filed[3]
        self._op_name = filed[4]
        self._life_start = filed[5]
        self._life_end = filed[6]
        self._total_alloc_mem = filed[7]
        self._dyn_op_name = filed[8]
        self._graph_id = filed[9]

    @property
    def op_mem_size(self: any) -> int:
        """
        Operator memory size
        """
        return self._op_mem_size / NumberConstant.BYTES_TO_KB

    @property
    def op_name(self: any) -> int:
        """
        Operator name
        """
        return self._op_name

    @property
    def life_start(self: any) -> int:
        """
        Serial number of operator memory used
        """
        return self._life_start

    @property
    def life_end(self: any) -> int:
        """
        Serial number of operator memory used
        """
        if self._life_end == self.NODE_INDEX_END_MAX:
            return self.CORRECT_NODE_INDEX_END_MAX
        return self._life_end

    @property
    def total_alloc_mem(self: any) -> int:
        """
        Static graph total allocate memory
        """
        return self._total_alloc_mem

    @property
    def dyn_op_name(self: any) -> int:
        """
        0: invalid, other: dynamic op name of root
        """
        return self._dyn_op_name

    @property
    def graph_id(self: any) -> int:
        """
        Graph id
        """
        return self._graph_id
