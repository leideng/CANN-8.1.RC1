#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from common_func.db_name_constant import DBNameConstant
from profiling_bean.struct_info.ts_memcpy import TsMemcpy


class TsMemcpyReader:
    """
    class for ts memcpy reader
    """

    def __init__(self: any) -> None:
        self._data = []
        self._table_name = DBNameConstant.TABLE_TS_MEMCPY

    @property
    def data(self: any) -> list:
        """
        get data
        :return: data
        """
        return self._data

    @property
    def table_name(self: any) -> str:
        """
        get table_name
        :return: table_name
        """
        return self._table_name

    def read_binary_data(self: any, file_data: any) -> None:
        """
        read ts memcpy binary data and store them into list
        :param file_data: binary data
        :param index: index
        :return: None
        """
        ts_memcpy_bean = TsMemcpy.decode(file_data)
        if ts_memcpy_bean:
            self.data.append((ts_memcpy_bean.timestamp, ts_memcpy_bean.stream_id,
                              ts_memcpy_bean.task_id, ts_memcpy_bean.task_state))
