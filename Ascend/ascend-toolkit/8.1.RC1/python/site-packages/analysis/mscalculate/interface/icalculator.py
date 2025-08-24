#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

from abc import ABCMeta
from abc import abstractmethod


class ICalculator(metaclass=ABCMeta):
    """
    interface for data to calculate
    """

    @abstractmethod
    def calculate(self: any) -> None:
        """
        run the data calculators
        : return: NA
        """

    @abstractmethod
    def save(self: any) -> None:
        """
        generator the calculator list to
        :return: parser list
        """
