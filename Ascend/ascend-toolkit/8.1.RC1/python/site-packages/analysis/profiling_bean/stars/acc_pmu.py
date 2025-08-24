#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from common_func.info_conf_reader import InfoConfReader
from common_func.utils import Utils
from profiling_bean.struct_info.struct_decoder import StructDecoder


class AccPmuDecoder(StructDecoder):
    """
    class used to decode binary data
    """

    def __init__(self: any, *args: tuple) -> None:
        filed = args[0]
        self._func_type = Utils.get_func_type(filed[0])
        self._time_stamp = filed[3]
        self._acc_id = filed[5]
        self._bandwidth = (filed[9], filed[10])
        self._ost = (filed[11], filed[12])

    @property
    def func_type(self: any) -> str:
        """
        get func type
        :return: func type
        """
        return self._func_type

    @property
    def timestamp(self: any) -> object:
        """
        get timestamp
        :return: class object
        """
        return InfoConfReader().time_from_syscnt(self._time_stamp)

    @property
    def bandwidth(self: any) -> tuple:
        """
        get write_bandwidth and read_bandwidth
        :return: (write_bandwidth， read_bandwidth)
        """
        return self._bandwidth

    @property
    def acc_id(self: any) -> int:
        """
        get acc id
        :return: acc_id
        """
        return self._acc_id

    @property
    def ost(self: any) -> tuple:
        """
        get write_ost and read_ost
        :return: (write_ost and read_ost)
        """
        return self._ost
