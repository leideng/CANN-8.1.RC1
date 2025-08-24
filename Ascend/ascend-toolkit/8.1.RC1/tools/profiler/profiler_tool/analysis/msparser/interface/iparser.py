#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

from abc import ABCMeta
from abc import abstractmethod


class IParser(metaclass=ABCMeta):
    """
    interface for data parser
    """

    @abstractmethod
    def parse(self: any) -> None:
        """
        parse the data under the file path
        :return: NA
        """

    @abstractmethod
    def save(self: any) -> None:
        """
        save the result of data parsing
        :return: NA
        """
