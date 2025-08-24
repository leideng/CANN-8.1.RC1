#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import logging

from msparser.add_info.add_info_bean import AddInfoBean


class Mc2CommInfoBean(AddInfoBean):
    COMM_STREAM_SIZE = 8

    def __init__(self: any, *args) -> None:
        super().__init__(*args)
        data = args[0]
        self._group_name = data[6]
        self._rank_size = data[7]
        self._rank_id = data[8]
        self._usr_rank_id = data[9]
        self._stream_id = data[10]
        self._stream_size = data[11]
        self._comm_stream_ids = data[12:12 + self.COMM_STREAM_SIZE]

    @property
    def group_name(self: any) -> str:
        return str(self._group_name)

    @property
    def rank_size(self: any) -> int:
        return self._rank_size

    @property
    def rank_id(self: any) -> int:
        return self._rank_id

    @property
    def usr_rank_id(self: any) -> int:
        return self._usr_rank_id

    @property
    def stream_id(self: any) -> int:
        return self._stream_id

    @property
    def comm_stream_ids(self: any) -> str:
        if self._stream_size > self.COMM_STREAM_SIZE:
            logging.error("The stream size %d is greater than max stream size %d.",
                          self._stream_size, self.COMM_STREAM_SIZE)
            return ""
        return ",".join(map(str, self._comm_stream_ids[:self._stream_size]))
