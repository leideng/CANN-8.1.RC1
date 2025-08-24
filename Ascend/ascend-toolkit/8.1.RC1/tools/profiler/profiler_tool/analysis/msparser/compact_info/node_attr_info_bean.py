#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from msparser.compact_info.compact_info_bean import CompactInfoBean


class NodeAttrInfoBean(CompactInfoBean):
    """
    node attr info bean
    """

    def __init__(self: any, *args) -> None:
        super().__init__(*args)
        data = args[0]
        self._node_id = data[6]
        self._hash_id = data[8]

    @property
    def node_id(self: any) -> str:
        """
        for node id
        """
        return self._node_id

    @property
    def hash_id(self: any) -> str:
        """
        the hash id for attribute information of operators
        """
        return self._hash_id


