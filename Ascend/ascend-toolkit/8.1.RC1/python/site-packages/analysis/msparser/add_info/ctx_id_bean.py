#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from msparser.add_info.add_info_bean import AddInfoBean


class CtxIdBean(AddInfoBean):
    """
    ctx id bean
    """
    PRE_LEN = 8

    def __init__(self: any, *args) -> None:
        super().__init__(*args)
        data = args[0]
        self._node_id = data[6]
        self._ctx_id_num = data[7]
        self._ctx_id = []
        for fusion_index in range(self.PRE_LEN, self.PRE_LEN + self._ctx_id_num):
            self._ctx_id.append(str(data[fusion_index]))

    @property
    def node_id(self: any) -> str:
        """
        for node id
        """
        return str(self._node_id)

    @property
    def ctx_id_num(self: any) -> int:
        """
        for ctx id num
        """
        return self._ctx_id_num

    @property
    def ctx_id(self: any) -> str:
        """
        for ctx id
        """
        return ','.join(self._ctx_id)
