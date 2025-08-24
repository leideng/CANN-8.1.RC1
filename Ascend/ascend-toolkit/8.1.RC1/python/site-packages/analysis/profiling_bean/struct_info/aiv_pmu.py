#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.

from common_func.utils import Utils
from profiling_bean.struct_info.struct_decoder import StructDecoder


class AivPmuBean(StructDecoder):
    """
    class used to decode aic pmu data
    """

    def __init__(self: any, *args: any) -> None:
        filed = args[0]
        self._stream_id = Utils.get_stream_id(filed[17])
        self._task_id = filed[4]
        self._total_cycle = filed[7]
        self._pmu_list = filed[9:17]

    @property
    def task_id(self: any) -> int:
        """
        get task_id
        :return: task_id
        """
        return self._task_id

    @property
    def stream_id(self: any) -> int:
        """
        get stream_id
        :return: stream_id
        """
        return self._stream_id

    @property
    def pmu_list(self: any) -> list:
        """
        get pmu_list
        :return: pmu_list
        """
        return self._pmu_list

    @property
    def total_cycle(self: any) -> int:
        """
        get total_cycle
        :return: total_cycle
        """
        return self._total_cycle
