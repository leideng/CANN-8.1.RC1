#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
from common_func.ms_constant.number_constant import NumberConstant


class TimerBean:

    def __init__(self, time_dict: dict, host_freq: float):
        self.clock_realtime = time_dict.get("clock_realtime", 0)
        self.clock_monotonic_raw = time_dict.get("clock_monotonic_raw", 0)
        self.cntvct = time_dict.get("cntvct", 0)
        self.cntvct_diff = time_dict.get("cntvct_diff", 0)
        self._host_freq = host_freq

    @property
    def host_wall(self):
        return int(self.clock_realtime)

    @property
    def host_mon(self):
        """
        :return: monotonic time(ns)，it will add diff when diff is valid.
        """
        if self.cntvct_diff and self._host_freq:
            return int(self.clock_monotonic_raw) + round(
                int(self.cntvct_diff) * NumberConstant.NANO_SECOND / self._host_freq)
        else:
            return int(self.clock_monotonic_raw)
