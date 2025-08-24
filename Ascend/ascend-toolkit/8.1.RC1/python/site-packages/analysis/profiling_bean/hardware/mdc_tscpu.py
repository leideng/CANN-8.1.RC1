#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from profiling_bean.struct_info.struct_decoder import StructDecoder


class MdcTscpuDecoder(StructDecoder):
    """
    class used to decode acsq task
    """

    def __init__(self: any, *args: any) -> None:
        args = args[0]
        self._header = args[0]
        self._perf_backtrace = args[2:22]
        self._pc = args[23]
        self._timestamp = args[24]
        self._pmu_data = args[25:]

    @property
    def header(self: any) -> str:
        """
        get header
        :return: header
        """
        return self._header

    @property
    def perf_backtrace(self: any) -> int:
        """
        get _perf_backtrace
        :return: perf backtrace
        """
        return self._perf_backtrace

    @property
    def pc(self: any) -> int:
        """
        get pc
        :return: pc
        """
        return self._pc

    @property
    def timestamp(self: any) -> int:
        """
        get timestamp
        :return: timestamp
        """
        return self._timestamp

    @property
    def pmu_data(self: any) -> int:
        """
        get pmu_data
        :return: pmu data
        """
        return self._pmu_data
