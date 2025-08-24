#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from common_func.ms_constant.level_type_constant import LevelDataType
from profiling_bean.struct_info.struct_decoder import StructDecoder


class ApiDataBean(StructDecoder):
    """
    API bean data for the data parsing by acl parser.
    """

    def __init__(self: any, *args) -> None:
        filed = args[0]
        self._struct_type = filed[2]
        self._level = filed[1]
        self._thread_id = filed[3]
        self._start = filed[5]
        self._end = filed[6]
        self._item_id = filed[7]

    @property
    def struct_type(self: any) -> str:
        """
        api data type
        :return: api type
        """
        return str(self._struct_type)

    @property
    def start(self: any) -> int:
        """
        api data start time
        :return: api start time
        """
        return self._start

    @property
    def end(self: any) -> int:
        """
        api data end time
        :return: api end time
        """
        return self._end

    @property
    def thread_id(self: any) -> int:
        """
        api data thread id
        :return: api thread id
        """
        return self._thread_id

    @property
    def item_id(self: any) -> str:
        """
        api data item id
        :return: api item id
        """
        return str(self._item_id)

    @property
    def level(self: any) -> str:
        """
        api data level type
        :return: api level type
        """
        return LevelDataType(self._level).name.lower()

    @property
    def acl_id(self: any) -> str:
        """
        api data acl id
        :return: api acl id
        """
        return str(self._struct_type & 65535)

    @property
    def acl_type(self: any) -> int:
        """
        api data acl type
        :return: api acl type
        """
        return self._struct_type >> 16
