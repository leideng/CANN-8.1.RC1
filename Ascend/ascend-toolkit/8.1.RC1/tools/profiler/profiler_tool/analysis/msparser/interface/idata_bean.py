#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

from abc import ABCMeta
from abc import abstractmethod


class IDataBean(metaclass=ABCMeta):
    """
    interface for data struct
    """

    @abstractmethod
    def decode(self: any, bin_data: any) -> None:
        """
        decode the bin data
        :param bin_data: bin data
        :return: instance
        """

    @abstractmethod
    def construct_bean(self: any, *args: any) -> None:
        """
        refresh the data instance
        :param args: bin data
        :return: True or False
        """
