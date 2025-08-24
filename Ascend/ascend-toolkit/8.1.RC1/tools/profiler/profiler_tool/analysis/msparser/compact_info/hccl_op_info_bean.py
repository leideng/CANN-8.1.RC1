#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

from msparser.compact_info.compact_info_bean import CompactInfoBean


class HcclOpInfoBean(CompactInfoBean):
    """
    Hccl op Info Bean
    """

    RELAY_FLAG_BIT = 0
    RETRY_FLAG_BIT = 1
    ALG_TYPE_PHASE_CNT = 4
    ALG_TYPE_BIT_CNT = 4
    ALG_TYPE_BIT_MASK = 0b1111

    def __init__(self: any, *args) -> None:
        super().__init__(*args)
        data = args[0]
        self._relay = (data[6] >> self.RELAY_FLAG_BIT) & 0x1
        self._retry = (data[6] >> self.RETRY_FLAG_BIT) & 0x1
        self._data_type = data[7]
        self._alg_type = data[8]
        self._count = data[9]
        self._group_name = data[10]

    @property
    def relay(self: any) -> int:
        """
        for relay flag
        """
        return self._relay

    @property
    def retry(self: any) -> int:
        """
        for retry flag
        """
        return self._retry

    @property
    def data_type(self: any) -> str:
        """
        for data type
        """
        return str(self._data_type)

    @property
    def alg_type(self: any) -> str:
        """
        for hccl op alg type
        """
        return str(self._alg_type)

    @property
    def count(self: any) -> int:
        """
        for hccl op transfer data count
        """
        return self._count

    @property
    def group_name(self: any) -> str:
        """
        hash number for hccl group
        """
        return str(self._group_name)

    @property
    def rank_size(self: any) -> int:
        return self._rank_size
