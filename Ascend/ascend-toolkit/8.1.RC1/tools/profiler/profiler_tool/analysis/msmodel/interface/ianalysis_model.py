#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.

from abc import ABCMeta
from abc import abstractmethod


class IAnalysisModel(metaclass=ABCMeta):
    """
    interface for data analysis
    """

    @abstractmethod
    def get_timeline_data(self: any) -> None:
        """
        output the timeline data
        """

    @abstractmethod
    def get_summary_data(self: any) -> None:
        """
        output the summary data
        """
