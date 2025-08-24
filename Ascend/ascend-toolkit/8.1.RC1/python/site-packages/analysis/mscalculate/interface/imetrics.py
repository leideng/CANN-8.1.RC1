#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

from abc import ABCMeta
from abc import abstractmethod


class IMetrics(metaclass=ABCMeta):
    """
    interface for data metrics
    """

    @staticmethod
    def get_division(divisor: any, dividend: any) -> float:
        """
        get divisor / dividend with specific decimal place.
        """
        if dividend == 0:
            return 0
        return divisor / dividend

    @staticmethod
    def get_mul(value1: any, value2: any) -> float:
        """
        get float value of value1 * value2
        """
        return 1.0 * value1 * value2

    @abstractmethod
    def run_rules(self: any) -> any:
        """
        run the metric rules
        :return: result for running rules
        """
