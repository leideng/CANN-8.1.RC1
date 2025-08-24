#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from common_func.constant import Constant

from profiling_bean.struct_info.struct_decoder import StructDecoder


class MsprofTxDecoder(StructDecoder):
    """
    class used to decode binary data
    """

    def __init__(self: any, *args: list) -> None:
        filed = args[0]
        self._magic = filed[0]
        self._info_type = filed[6]
        self._data_tag = filed[10]
        self._process_id = filed[11]
        self._thread_id = filed[12]
        self._category = filed[13]
        self._event_type = filed[14]
        self._payload_type = filed[15]
        self._payload_value = filed[16]
        self._start_time = filed[17]
        self._end_time = filed[18]
        self._mark_id = filed[19]
        self._domain = filed[20]
        self._message_type = filed[21]
        self._message = self.decode_byte(filed[22])

    @property
    def magic(self: any) -> int:
        return self._magic

    @property
    def info_type(self: any) -> int:
        return self._info_type

    @property
    def data_tag(self: any) -> int:
        return self._data_tag

    @property
    def process_id(self: any) -> int:
        return self._process_id

    @property
    def thread_id(self: any) -> int:
        return self._thread_id

    @property
    def category(self: any) -> int:
        return self._category

    @property
    def event_type(self: any) -> int:
        return self._event_type

    @property
    def payload_type(self: any) -> int:
        return self._payload_type

    @property
    def payload_value(self: any) -> int:
        return self._payload_value

    @property
    def start_time(self: any) -> int:
        return self._start_time

    @property
    def end_time(self: any) -> int:
        return self._end_time

    @property
    def mark_id(self) -> int:
        return self._mark_id

    @property
    def domain(self) -> int:
        return self._domain

    @property
    def message_type(self: any) -> int:
        return self._message_type

    @property
    def message(self: any) -> str:
        return self._message

    def decode_byte(self: any, value: str) -> str:
        zero_ind = value.find(b'\x00')
        str_value = bytes.decode(value[0:zero_ind])
        return str_value
