#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.


class CpuInfo:
    """
    cpu info class
    """

    def __init__(self: any, *args: any) -> None:
        super(CpuInfo, self).__init__()
        cpu_id, frequency, logical_cpu_count, cpu_name, cpu_type = args
        self._cpu_id = cpu_id
        self._frequency = frequency
        self._logical_cpu_count = logical_cpu_count
        self._cpu_name = cpu_name
        self._cpu_type = cpu_type

    def cpu_id(self: any) -> int:
        """
        cpu id
        :return: cpu id
        """
        return self._cpu_id

    def frequency(self: any) -> int:
        """
        frequency
        :return: frequency
        """
        return self._frequency

    def logical_cpu_count(self: any) -> int:
        """
        logical cpu count
        :return: logical cpu count
        """
        return self._logical_cpu_count

    def cpu_name(self: any) -> int:
        """
        cpu name
        :return: cpu name
        """
        return self._cpu_name

    def cpu_type(self: any) -> int:
        """
        cpu type
        :return: cpu type
        """
        return self._cpu_type
