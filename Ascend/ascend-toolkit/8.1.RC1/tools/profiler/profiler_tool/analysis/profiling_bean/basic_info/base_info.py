#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

from abc import ABCMeta
from abc import abstractmethod


class BaseInfo(metaclass=ABCMeta):
    """
    info base class
    """

    def __init__(self: any) -> None:
        pass

    @abstractmethod
    def run(self: any, project_path: str) -> None:
        """
        run data
        :param project_path: project path
        :return: bean
        """

    @abstractmethod
    def merge_data(self: any) -> any:
        """
        merge data
        :return: bean
        """
