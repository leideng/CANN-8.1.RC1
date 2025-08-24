#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

import logging

from common_func.ms_constant.ge_enum_constant import GeTaskType
from msparser.compact_info.compact_info_bean import CompactInfoBean


class NodeBasicInfoBean(CompactInfoBean):
    """
    Node Basic Info Bean
    """

    def __init__(self: any, *args) -> None:
        super().__init__(*args)
        data = args[0]
        self._node_id = data[6]
        self._task_type = data[7]
        self._op_type = data[8]
        self._block_dim = data[9]
        self._op_flag = data[10]
        self._mix_block_dim = self.mix_block_dim

    @property
    def task_type(self: any) -> str:
        """
        for task type
        """
        task_type_dict = GeTaskType.member_map()
        if self._task_type not in task_type_dict:
            logging.error("Unsupported task_type %d", self._task_type)
            return str(self._task_type)
        return task_type_dict.get(self._task_type).name

    @property
    def op_type(self: any) -> str:
        """
        for op type
        """
        return str(self._op_type)

    @property
    def node_id(self: any) -> str:
        """
        for node id
        """
        return str(self._node_id)

    @property
    def op_flag(self: any) -> int:
        """
        for op flag
        """
        return self._op_flag

    @property
    def block_dim(self: any) -> int:
        """
        for block dims
        get lower 16bit data of 32bit
        """
        return self._block_dim & 65535

    @property
    def mix_block_dim(self: any) -> int:
        """
        for mix block dims
        get the product of block dim and higher 16bit
        """
        return (self._block_dim & 65535) * (self._block_dim >> 16)
