#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from msparser.compact_info.compact_info_bean import CompactInfoBean


class MemcpyInfoBean(CompactInfoBean):
    """
    memcpy info bean
    """

    def __init__(self: any, *args) -> None:
        super().__init__(*args)
        data = args[0]
        self._data_size = data[6]
        self._direction = data[7]

    @property
    def data_size(self: any) -> int:
        """
        memcpy data size
        """
        return self._data_size

    @property
    def direction(self: any) -> int:
        """
        memcpy directionL: h2d, d2h, d2d, h2h...
        """
        return self._direction
