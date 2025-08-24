#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.

from abc import ABCMeta
from abc import abstractmethod


class StepTraceTagHandler(metaclass=ABCMeta):
    """
    get model_id, index_id, FP, BP, reduce from step trace
    """

    @abstractmethod
    def receive_record(self: any, record: dict) -> None:
        """
        receive record of step trace
        :param record: contain model_id, tag_id, timestamp
        :return: void
        """

    @abstractmethod
    def get_data(self: any) -> dict:
        """
        return data of this handler
        :return: dict
        """
