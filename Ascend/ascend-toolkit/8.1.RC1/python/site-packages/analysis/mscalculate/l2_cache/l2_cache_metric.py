#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

from common_func.ms_constant.number_constant import NumberConstant
from mscalculate.interface.imetrics import IMetrics


class HitRateMetric(IMetrics):
    """
    hit rate metrics
    """

    def __init__(self: any, hit_value: int, request_value: int) -> None:
        self.hit_value = hit_value
        self.request_value = request_value

    @staticmethod
    def class_name() -> str:
        """
        class name
        """
        return HitRateMetric.__name__

    def run_rules(self: any) -> float:
        """
        run rules
        """
        return self._calculate_hit_rate()

    def _calculate_hit_rate(self: any) -> float:
        return round(self.get_division(self.hit_value, self.request_value), NumberConstant.DECIMAL_ACCURACY)


class VictimRateMetric(IMetrics):
    """
    victim rate metrics
    """

    def __init__(self: any, victim_value: int, request_value: int) -> None:
        self.victim_value = victim_value
        self.request_value = request_value

    @staticmethod
    def class_name() -> str:
        """
        class name
        """
        return VictimRateMetric.__name__

    def run_rules(self: any) -> float:
        """
        run rules
        """
        return self._calculate_victim_rate()

    def _calculate_victim_rate(self: any) -> float:
        return round(self.get_division(self.victim_value, self.request_value), NumberConstant.DECIMAL_ACCURACY)
